# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NIXL Transfer Manager for weight and artifact transfers.

This module provides the NixlTransferManager class that handles all NIXL-related
operations including agent creation, memory registration, and RDMA transfers.

Each vLLM worker creates its own NixlTransferManager instance to manage
a single NIXL agent. The primary path is GPU tensor transfer; artifact transfer
also uses the same agent for host DRAM chunk staging.
"""

from __future__ import annotations

import atexit
import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from . import envs
from . import ucx_utils
from .metrics import metrics as transfer_metrics
from ._nixl import load_nixl_api
from .accelerators import (
    AcceleratorBackend,
    CudaAcceleratorBackend,
)
from .types import ManifestMismatchError, TensorDescriptor

if TYPE_CHECKING:
    from .vmm.arena import VmmArena

logger = logging.getLogger("modelexpress.nixl_transfer")

NIXL_AVAILABLE = False
NixlAgent = None
nixl_agent_config = None
_nixl_api = load_nixl_api()
if _nixl_api is not None:
    NixlAgent = _nixl_api.nixl_agent
    nixl_agent_config = _nixl_api.nixl_agent_config
    NIXL_AVAILABLE = True


SUPPORTED_NIXL_BACKENDS = ("UCX", "LIBFABRIC")
DEFAULT_NIXL_BACKEND = "UCX"
NIXL_DRAM_MEM_TYPE = "DRAM"


def is_nixl_available() -> bool:
    """Check if NIXL is available."""
    return NIXL_AVAILABLE


def _resolve_nixl_backend() -> str:
    """Resolve the NIXL backend from MX_NIXL_BACKEND.

    Defaults to UCX. Set MX_NIXL_BACKEND=LIBFABRIC on AWS EFA.
    """
    raw = envs.MX_NIXL_BACKEND
    if raw not in SUPPORTED_NIXL_BACKENDS:
        raise ValueError(
            f"MX_NIXL_BACKEND={raw!r} is not supported. "
            f"Expected one of {SUPPORTED_NIXL_BACKENDS}."
        )
    return raw


def _arena_single_mr_forced() -> bool:
    """Whether to keep the single-MR arena path on a multi-allocation arena.

    MX_ARENA_SINGLE_MR=1 forces it. Only safe on transports that can span
    several cuMemCreate handles in one registration (dmabuf/IB); cuda_ipc
    cannot. Read at call time so tests can toggle the env var.
    """
    return envs.MX_ARENA_SINGLE_MR


def _pool_reg_enabled() -> bool:
    """Whether allocation-level pool registration is enabled.

    MX_POOL_REG=1 enables it; default is per-tensor registration. Read at
    call time so tests can toggle the env var without re-importing.
    """
    return envs.MX_POOL_REG


@dataclass
class PostedRead:
    """A batched RDMA READ that has been posted but not yet waited on.

    Held by the caller between ``post_read_batch`` and ``await_read_batches`` so
    several peers can have transfers in flight at once. The handle is owned by
    ``await_read_batches``, which releases it.
    """

    handle: Any
    remote_agent_name: str
    total_bytes: int
    num_ranges: int
    posted_at: float = field(default_factory=time.perf_counter)


class NixlTransferManager:
    """
    Manages a single NIXL agent and RDMA transfers.

    Each vLLM worker creates its own instance of this class to handle:
    - Creating and managing a NIXL agent for the worker's GPU
    - Registering tensors with NIXL for RDMA access
    - Executing transfers to receive weights from remote sources
    - Registering host DRAM buffers for artifact chunk transfer

    Args:
        agent_name: Name for the NIXL agent
        device_id: GPU device ID for this worker
    """

    def __init__(
        self,
        agent_name: str,
        device_id: int,
        listen_port: int | None = None,
        accelerator_backend: AcceleratorBackend | None = None,
    ):
        self._agent_name = agent_name
        self._device_id = device_id
        self._listen_port = listen_port
        self._accelerator_backend = accelerator_backend or CudaAcceleratorBackend()

        self._backend = _resolve_nixl_backend()
        self._backends = [self._backend]

        self._agent: Any = None
        self._metadata: bytes = b""
        self._tensor_descriptors: list[TensorDescriptor] = []
        self._tensors: dict[str, torch.Tensor] = {}
        # Registration descriptors must be deregistered before destroying the
        # UCX-backed NIXL agent. Dropping an agent with live GPU registrations
        # can abort inside ucp_worker_destroy during framework teardown.
        self._registered_memory: list[Any] = []
        # Remote agents this manager has loaded, so shutdown can disconnect them.
        # Maps agent name -> (ip, port) for agents reached over the P2P socket, or
        # None for agents loaded from a metadata blob.
        #
        # Tracking exists because NIXL only lets the side that *loaded* a peer
        # disconnect from it: invalidateRemoteMD looks the peer up in
        # remoteBackends_ and is a no-op otherwise. In P2P the target loads the
        # source (it sends NIXLCOMM:SEND and gets back the source's metadata) but
        # the source never loads the target, so the target is the only side that
        # can close the pair. Leaving it to process death leaves the source with
        # a half-open QP it has no way to invalidate.
        self._remote_agents: dict[str, tuple[str, int] | None] = {}
        # Last data-plane failure, used by is_healthy(). None means no failure
        # has been observed on a transfer this manager issued.
        self._data_plane_error: str | None = None
        self._atexit_registered = False

    @property
    def agent_name(self) -> str:
        """Get NIXL agent name."""
        return self._agent_name

    @property
    def backends(self) -> list[str]:
        """NIXL backends this agent was created with (see MX_NIXL_BACKEND).

        Callers that issue their own NIXL calls against :attr:`agent` must pass
        this rather than a literal, or the transfer is prepared on a backend the
        agent does not have (e.g. UCX on AWS EFA).
        """
        return list(self._backends)

    @property
    def nixl_metadata(self) -> bytes:
        """Get NIXL metadata for this agent."""
        return self._metadata

    @property
    def listen_port(self) -> int | None:
        """Get the port serving this agent's NIXL metadata."""
        return self._listen_port

    @property
    def tensor_descriptors(self) -> list[TensorDescriptor]:
        """Get tensor descriptors for registered tensors."""
        return self._tensor_descriptors

    def initialize(self) -> None:
        """Initialize the NIXL agent.

        Temporarily overrides UCX_TLS to allow NIXL's UCX context to
        auto-detect RoCE/IB transports, even if the global UCX_TLS is
        restricted to TCP (e.g., for MPI). Restores the original value
        after agent creation.

        Optional per-rank NIC pinning (MX_RDMA_NIC_PIN) is delegated to
        ucx_utils.apply_nic_pin_for_device. Default (env var unset) is a
        no-op. See ucx_utils for the topology probe and env var modes.
        """
        if not NIXL_AVAILABLE:
            raise RuntimeError("NIXL is not available")

        if self._agent is not None:
            return

        self._accelerator_backend.set_device(self._device_id)

        # Let UCX auto-detect transports (RoCE, TCP, etc).
        # OMPI_MCA_pml=ob1 keeps MPI on TCP independently.
        # Only override UCX_TLS if explicitly set to "tcp" (legacy compat).
        saved_ucx_tls = envs.UCX_TLS
        nixl_ucx_tls = envs.NIXL_UCX_TLS
        if nixl_ucx_tls:
            os.environ["UCX_TLS"] = nixl_ucx_tls
            logger.info(f"NIXL UCX_TLS override: {nixl_ucx_tls} (was: {saved_ucx_tls})")
        elif saved_ucx_tls == "tcp":
            os.environ.pop("UCX_TLS", None)
            logger.info("NIXL: removed UCX_TLS=tcp for auto-detection")

        # Optional per-rank NIC pinning, set permanently for the worker's
        # lifetime so any subsequently-created UCP contexts also pin.
        # No-op unless MX_RDMA_NIC_PIN is set. See ucx_utils for full env
        # semantics and the topology probe.
        ucx_utils.apply_nic_pin_for_device(self._device_id)

        try:
            if self._listen_port is not None and nixl_agent_config:
                config = nixl_agent_config(
                    backends=self._backends,
                    enable_listen_thread=True,
                    listen_port=self._listen_port,
                )
                logger.info(f"NIXL listen thread enabled on port {self._listen_port}")
            elif nixl_agent_config:
                config = nixl_agent_config(backends=self._backends)
            else:
                config = None
            self._agent = NixlAgent(self._agent_name, config)
            self._register_atexit()
            logger.info(
                f"NIXL agent '{self._agent_name}' created on device "
                f"{self._device_id} (backend={self._backend})"
            )
        finally:
            if saved_ucx_tls is not None:
                os.environ["UCX_TLS"] = saved_ucx_tls
            elif envs.is_set("UCX_TLS"):
                os.environ.pop("UCX_TLS")

    def _register_atexit(self) -> None:
        """Register manager-owned process teardown once."""
        if self._atexit_registered:
            return
        atexit.register(self.shutdown)
        self._atexit_registered = True

    def _build_tensor_descriptors(
        self, tensors: dict[str, torch.Tensor]
    ) -> list[TensorDescriptor]:
        """Build NIXL TensorDescriptors from a name -> tensor mapping.

        Validates each tensor is contiguous (non-contiguous tensors would
        require copies that misalign RDMA writes) and records the tensor
        objects + descriptor list on self for the receiver path to resolve
        descriptors back by name.

        CRITICAL: self._tensors must hold the SAME tensor objects as
        param.data in vLLM. Do NOT call .contiguous() here - that would
        create copies and RDMA writes would land in the wrong memory.

        We take a shallow copy of the caller's dict (``dict(tensors)``)
        so ``shutdown()``'s cleanup cannot mutate the caller's
        container. The tensor VALUES are the same objects as
        ``param.data``; only the dict container is owned by the manager.
        """
        self._tensors = dict(tensors)
        tensor_descriptors = []
        for name, tensor in tensors.items():
            if not tensor.is_contiguous():
                raise RuntimeError(
                    f"Tensor '{name}' is not contiguous. "
                    "Non-contiguous tensors cannot be used for RDMA transfers."
                )
            tensor_descriptors.append(
                TensorDescriptor(
                    name=name,
                    addr=tensor.data_ptr(),
                    size=tensor.numel() * tensor.element_size(),
                    device_id=self._device_id,
                    dtype=str(tensor.dtype),
                )
            )
        self._tensor_descriptors = tensor_descriptors
        return tensor_descriptors

    def register_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        force_per_tensor: bool = False,
    ) -> bytes:
        """
        Register tensors with NIXL for RDMA access.

        With MX_POOL_REG=1, discovers the unique cudaMalloc allocations
        backing the tensors via cuMemGetAddressRange and registers each
        allocation as a single NIXL block. This dramatically reduces the
        number of memory registrations (kernel ibv_reg_mr calls, rkeys,
        and bytes in the agent metadata blob) without changing transfer
        semantics: receive_from_source still matches by tensor name and
        builds per-tensor RDMA descriptors that target addresses inside
        the registered allocations.

        With MX_POOL_REG unset (default), falls back to per-tensor
        registration.

        CRITICAL: self._tensors must hold the SAME tensor objects as
        param.data in vLLM. Do NOT call .contiguous() here - that would
        create copies and RDMA writes would land in the wrong memory.

        We take a shallow copy of the caller's dict (``dict(tensors)``)
        so shutdown's cleanup cannot mutate the caller's container.
        The tensor VALUES are the same objects as ``param.data``;
        only the dict container is owned by the manager.

        Args:
            tensors: Dictionary of tensor name -> tensor

        Returns:
            NIXL metadata bytes for this agent
        """
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")

        tensor_descriptors = self._build_tensor_descriptors(tensors)

        # Phase 1: Discover CUDA allocation boundaries (if pool reg enabled)
        alloc_discovery_start = time.perf_counter()
        if _pool_reg_enabled() and not force_per_tensor:
            if self._accelerator_backend.supports_pool_reg():
                allocations = self._find_cuda_allocations(tensor_descriptors)
            else:
                allocations = None
                logger.warning(
                    "MX_POOL_REG=1 set but %s does not support pool "
                    "registration; using per-tensor registration",
                    self._accelerator_backend.name,
                )
        else:
            allocations = None
            logger.info(
                "Pool registration disabled (MX_POOL_REG != '1'), using per-tensor registration"
            )
        alloc_discovery_time = time.perf_counter() - alloc_discovery_start

        # Phase 2: Register memory with NIXL (ibv_reg_mr kernel calls)
        nixl_reg_start = time.perf_counter()
        if allocations:
            alloc_tuples = [
                (base, size, self._device_id, "") for base, size in allocations
            ]
            self._registered_memory.append(
                self._agent.register_memory(
                    alloc_tuples,
                    mem_type=self._accelerator_backend.nixl_mem_type,
                    backends=self._backends,
                )
            )
            reg_count = len(allocations)
        else:
            tensor_list = list(tensors.values())
            self._registered_memory.append(
                self._agent.register_memory(tensor_list, backends=self._backends)
            )
            reg_count = len(tensor_list)
        nixl_reg_time = time.perf_counter() - nixl_reg_start

        # Phase 3: Get agent metadata blob
        metadata_start = time.perf_counter()
        self._metadata = self._agent.get_agent_metadata()
        metadata_time = time.perf_counter() - metadata_start

        total_time = alloc_discovery_time + nixl_reg_time + metadata_time
        reduction = (
            (1 - reg_count / len(tensor_descriptors)) * 100 if tensor_descriptors else 0
        )
        total_bytes = sum(d.size for d in tensor_descriptors)

        logger.info(
            f"[TIMING] register_tensors: {total_time:.3f}s total "
            f"(alloc_discovery={alloc_discovery_time:.3f}s, "
            f"nixl_register={nixl_reg_time:.3f}s [{reg_count} regions], "
            f"get_metadata={metadata_time:.3f}s [{len(self._metadata)} bytes])"
        )
        logger.info(
            f"Registered {reg_count} regions from {len(tensor_descriptors)} tensors "
            f"({reduction:.1f}% reduction), {total_bytes / 1e9:.2f} GB total"
        )

        return self._metadata

    def register_arena(
        self, arena: VmmArena, tensors: dict[str, torch.Tensor]
    ) -> bytes:
        """Register a VmmArena's full bump range as a single NIXL region.

        The arena owns a contiguous VA range; at end-of-load the bump
        pointer's [base, base+used) covers every allocation we've ever
        made (including holes from intervening frees). NIXL's
        `register_memory` with `mem_type="VRAM"` over this range
        consumes a dmabuf via `ibv_reg_dmabuf_mr` and produces ONE
        lkey/rkey covering all live tensors.

        The multi-handle case is validated on the dmabuf/IB path only.
        On Blackwell + ConnectX over InfiniBand, against a CUDA VMM range
        with multiple cuMemCreate handles and mid-range holes (chunks
        unmapped + released after the export): registration succeeds, the
        dmabuf attach pins the currently-mapped physical pages, and the
        HCA translation table survives subsequent CUDA-side unmaps.

        It does NOT hold on UCX cuda_ipc, where a fabric handle names one
        cuMemCreate allocation and a single MR would publish an rkey
        covering only the first chunk. That is why this method falls back
        to per-tensor registration when the arena spans several
        allocations, unless MX_ARENA_SINGLE_MR overrides it. Upstream fix:
        openucx/ucx#11283.

        Per-tensor descriptors are still built (tensor name -> addr,
        size, dtype) because the receiver matches by name and computes
        an offset into the single registered region.

        Requires `UCX_CUDA_COPY_REG_WHOLE_ALLOC=off` on the deployment
        until the upstream UCX cuda_copy_md fix ships, otherwise UCX
        internally truncates the requested length via
        cuMemGetAddressRange (which on multi-handle VMM returns
        per-handle bounds, not the full reserve).
        """
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")

        tensor_descriptors = self._build_tensor_descriptors(tensors)

        base, used = arena.registered_range()
        if used == 0:
            logger.warning(
                "register_arena called with empty arena (used=0); falling back "
                "to per-tensor registration"
            )
            return self.register_tensors(tensors)

        if not self._accelerator_backend.supports_vmm():
            logger.warning(
                "%s does not support VMM arena registration; falling back "
                "to per-tensor registration",
                self._accelerator_backend.name,
            )
            return self.register_tensors(tensors)

        # A CUDA fabric/IPC handle names exactly one cuMemCreate allocation:
        # UCX cuda_ipc resolves a region with cuMemRetainAllocationHandle and
        # cuMemGetAddressRange, which report the FIRST allocation under the
        # range rather than the whole reserve. Registering a multi-allocation
        # arena as one MR therefore publishes an rkey covering only its first
        # chunk, and the peer's cuMemcpyDtoDAsync_v2 reads past what it mapped.
        # Measured on GB200 MNNVL: Kimi-K3 arena, 1019 chunks, segfault in
        # uct_cuda_ipc_ep_get_zcopy. Per-tensor registration is correct because
        # the arena does one cuMemCreate per allocation, so every tensor lies
        # wholly inside one handle.
        #
        # dmabuf/IB registration does span several handles, so deployments that
        # validated the single-MR path there can keep it with
        # MX_ARENA_SINGLE_MR=1.
        live_allocs = arena.live_allocation_count
        if live_allocs > 1 and not _arena_single_mr_forced():
            logger.warning(
                "register_arena: arena spans %d physical allocations; a single "
                "MR would publish an rkey covering only the first, which "
                "cuda_ipc cannot address. Falling back to per-tensor "
                "registration for %d tensors over [0x%x, 0x%x). Set "
                "MX_ARENA_SINGLE_MR=1 to force single-MR (dmabuf/IB only).",
                live_allocs,
                len(tensor_descriptors),
                base,
                base + used,
            )
            return self.register_tensors(tensors, force_per_tensor=True)

        # NIXL resolves descriptors by containment, so one tensor outside
        # [base, base+used) fails prep_xfer_dlist for the whole transfer.
        uncovered = [
            d
            for d in tensor_descriptors
            if d.addr < base or (d.addr + d.size) > (base + used)
        ]
        if uncovered:
            logger.warning(
                "register_arena: %d of %d tensors lie outside the arena range "
                "[0x%x, 0x%x); falling back to per-tensor registration. "
                "First uncovered: %s at 0x%x (%d bytes)",
                len(uncovered),
                len(tensor_descriptors),
                base,
                base + used,
                uncovered[0].name,
                uncovered[0].addr,
                uncovered[0].size,
            )
            # Bypass pool reg: it resolves the same per-handle bounds we just
            # found insufficient.
            return self.register_tensors(tensors, force_per_tensor=True)

        nixl_reg_start = time.perf_counter()
        self._registered_memory.append(
            self._agent.register_memory(
                [(base, used, self._device_id, "")],
                mem_type=self._accelerator_backend.nixl_mem_type,
                backends=self._backends,
            )
        )
        nixl_reg_time = time.perf_counter() - nixl_reg_start

        metadata_start = time.perf_counter()
        self._metadata = self._agent.get_agent_metadata()
        metadata_time = time.perf_counter() - metadata_start

        total_bytes = sum(d.size for d in tensor_descriptors)
        reduction = (1 - 1 / len(tensor_descriptors)) * 100 if tensor_descriptors else 0
        logger.info(
            f"[TIMING] register_arena: {nixl_reg_time + metadata_time:.3f}s total "
            f"(nixl_register={nixl_reg_time:.3f}s [1 region, {used / 1e9:.2f} GB], "
            f"get_metadata={metadata_time:.3f}s [{len(self._metadata)} bytes])"
        )
        logger.info(
            f"Registered arena as 1 region from {len(tensor_descriptors)} tensors "
            f"({reduction:.1f}% reduction), {total_bytes / 1e9:.2f} GB live in "
            f"{used / 1e9:.2f} GB arena bump range"
        )

        return self._metadata

    @staticmethod
    def _find_cuda_allocations(
        descriptors: list[TensorDescriptor],
    ) -> list[tuple[int, int]]:
        """
        Find unique CUDA allocations backing the tensor descriptors.

        Uses cuMemGetAddressRange (cuda-python binding for the v2 driver
        ABI) to query each tensor's containing cudaMalloc block. Adjacent
        allocations in virtual address space are NOT merged: UCX's rcache
        produces broken rkeys when a single registered region spans
        multiple cudaMalloc blocks, even when they happen to be adjacent.
        Each unique allocation is registered independently.

        Args:
            descriptors: List of tensor descriptors

        Returns:
            Sorted list of (alloc_base, alloc_size) tuples for unique
            CUDA allocations.
        """
        if not descriptors:
            return []

        from cuda.bindings import driver as cuda_driver

        seen: dict[int, int] = {}  # alloc_base -> alloc_size

        for desc in descriptors:
            err, alloc_base, alloc_size = cuda_driver.cuMemGetAddressRange(desc.addr)
            if err != cuda_driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"cuMemGetAddressRange failed ({err.name}) for tensor "
                    f"'{desc.name}' at 0x{desc.addr:x}. Is the tensor on a CUDA device?"
                )
            base_int = int(alloc_base)
            if base_int not in seen:
                seen[base_int] = int(alloc_size)

        return sorted(seen.items())

    def _wait_for_xfers(
        self,
        handles: list,
        timeout_seconds: float | None,
        label: str,
    ) -> None:
        """Poll several NIXL handles until all complete or one fails.

        Sleeps only when a full sweep completed nothing, so the polling slop is
        paid once for the whole set rather than once per handle.

        Records data-plane failures exactly as :meth:`_wait_for_xfer` does, for the
        same reason: a wedged QP yields neither a completion nor an ERR status, so
        the timeout is the only evidence anything went wrong, and recording it is
        what lets ``is_healthy()`` stop advertising this agent.
        """
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")
        pending = list(handles)
        waited_on_something = bool(pending)
        wait_start = time.perf_counter()
        while pending:
            if (
                timeout_seconds is not None
                and time.perf_counter() - wait_start >= timeout_seconds
            ):
                self._data_plane_error = (
                    f"{label} timed out after {timeout_seconds:.1f}s with "
                    f"{len(pending)} transfer(s) outstanding and no error status "
                    f"from NIXL"
                )
                transfer_metrics.record_nixl_error("timeout")
                raise TimeoutError(
                    f"{label} timed out with {len(pending)} transfer(s) outstanding"
                )
            still_pending = []
            for handle in pending:
                status = self._agent.check_xfer_state(handle)
                if status in ("DONE", "SUCCESS"):
                    continue
                if status in ("ERR", "ERROR", "FAIL"):
                    self._data_plane_error = f"{label} failed with status {status}"
                    transfer_metrics.record_nixl_error("status_error")
                    raise RuntimeError(f"{label} failed with status {status}")
                still_pending.append(handle)
            if len(still_pending) == len(pending):
                time.sleep(0.001)
            pending = still_pending
        # Only once the whole set has completed, and only if there was a set. Nothing
        # is proven by waiting on no handles, and clearing per handle would let a
        # batch that failed on its last one report healthy. Health must not latch
        # either: a completed batch is proof the data plane works, so a worker
        # demoted for one transient timeout can return to READY.
        if waited_on_something:
            self._data_plane_error = None

    def _wait_for_xfer(
        self,
        handle: Any,
        timeout_seconds: float | None,
        label: str,
    ) -> None:
        """Poll a NIXL transfer handle until completion or failure."""
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")
        wait_start = time.perf_counter()
        while True:
            if (
                timeout_seconds is not None
                and time.perf_counter() - wait_start >= timeout_seconds
            ):
                # A timeout here is a data-plane failure even though NIXL never
                # reported one. When a QP is wedged the READ neither completes nor
                # transitions to ERR, so the handle stays incomplete and the
                # timeout is the only evidence that anything went wrong. Recording
                # it is what lets is_healthy() stop advertising this agent.
                self._data_plane_error = (
                    f"{label} timed out after {timeout_seconds:.1f}s with no "
                    f"completion and no error status from NIXL"
                )
                transfer_metrics.record_nixl_error("timeout")
                raise TimeoutError(f"{label} timed out")
            status = self._agent.check_xfer_state(handle)
            if status in ("DONE", "SUCCESS"):
                # A completed transfer is direct proof the data plane works, so it
                # clears any earlier failure. Without this the flag would latch for
                # the life of the process and a worker demoted for one transient
                # timeout could never return to READY, however healthy the fabric
                # became.
                self._data_plane_error = None
                return
            if status in ("ERR", "ERROR", "FAIL"):
                self._data_plane_error = f"{label} failed with status {status}"
                transfer_metrics.record_nixl_error("status_error")
                raise RuntimeError(f"{label} failed with status {status}")
            time.sleep(0.001)

    def fetch_remote_and_wait(
        self,
        remote_agent_name: str,
        ip: str,
        port: int,
        timeout_seconds: float = 120.0,
    ) -> None:
        """Fetch remote NIXL agent metadata via the P2P listen thread.

        Initiates an async fetch and polls until the remote agent's metadata
        is loaded locally. Used in P2P mode instead of add_remote_agent().
        """
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")

        logger.info(f"Fetching remote metadata from {remote_agent_name} at {ip}:{port}")
        self._agent.fetch_remote_metadata(remote_agent_name, ip, port)

        start = time.perf_counter()
        while True:
            if time.perf_counter() - start >= timeout_seconds:
                raise TimeoutError(
                    f"Timed out waiting for remote metadata from "
                    f"{remote_agent_name} at {ip}:{port}"
                )
            if self._agent.check_remote_metadata(remote_agent_name):
                logger.info(
                    f"Remote metadata loaded for {remote_agent_name} "
                    f"({time.perf_counter() - start:.2f}s)"
                )
                self._remote_agents[remote_agent_name] = (ip, port)
                return
            time.sleep(0.01)

    def add_remote_agent(self, source_metadata: bytes) -> str:
        """Load a remote NIXL agent from a metadata blob."""
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")
        remote_agent_name = self._agent.add_remote_agent(source_metadata)
        logger.info(
            "Loaded remote NIXL agent %s from metadata blob (%d bytes)",
            remote_agent_name,
            len(source_metadata),
        )
        self._remote_agents.setdefault(remote_agent_name, None)
        return remote_agent_name

    def remove_remote_agent(self, remote_agent_name: str) -> bool:
        """Disconnect from a remote agent and drop its cached metadata.

        The counterpart to :meth:`add_remote_agent` and
        :meth:`fetch_remote_and_wait`. NIXL's ``invalidateRemoteMD`` both frees the
        cached metadata and disconnects the backend, so this is what returns the
        QP pair to a clean state instead of leaving it half-open.

        Returns True if NIXL accepted the removal. Never raises: this runs on
        teardown paths where the interesting failure has usually already happened,
        and masking it behind a cleanup error would be worse than logging it.
        """
        if self._agent is None:
            return False
        try:
            self._agent.remove_remote_agent(remote_agent_name)
        except Exception as exc:
            # NOT_FOUND is expected if the peer was already invalidated, e.g. it
            # sent us NIXLCOMM:INVL on its way out.
            logger.warning(
                "Failed to remove remote NIXL agent %s: %s", remote_agent_name, exc
            )
            self._remote_agents.pop(remote_agent_name, None)
            return False
        self._remote_agents.pop(remote_agent_name, None)
        logger.info("Disconnected remote NIXL agent %s", remote_agent_name)
        return True

    def disconnect_remote_agents(self) -> int:
        """Disconnect every remote agent this manager loaded.

        Returns the number successfully disconnected. Iterates a copy because
        :meth:`remove_remote_agent` mutates the tracking map.
        """
        if self._agent is None or not self._remote_agents:
            return 0
        removed = 0
        for name in list(self._remote_agents):
            if self.remove_remote_agent(name):
                removed += 1
        return removed

    def receive_from_source(
        self,
        source_metadata: bytes,
        source_tensors: list[TensorDescriptor],
        timeout_seconds: float | None = None,
        remote_agent_name: str | None = None,
        require_exact_match: bool = False,
        destination_tensors: dict[str, torch.Tensor] | None = None,
    ) -> tuple[int, int, float]:
        """
        Receive weights from a remote source via NIXL RDMA.

        Matches source tensors to local tensors by name and issues per-tensor
        RDMA READs. Both sides may have registered either pools (MX_POOL_REG=1)
        or individual tensors; the addresses inside source_tensors and the
        local tensor data_ptrs are what NIXL prep_xfer_dlist resolves against
        the registered memory metadata.

        Args:
            source_metadata: NIXL metadata from the source agent (unused if
                remote_agent_name is set)
            source_tensors: Tensor descriptors from the source
            timeout_seconds: Maximum time to wait for transfer (None for no
                timeout)
            remote_agent_name: If set, use this pre-loaded agent (P2P mode)
                instead of calling add_remote_agent with source_metadata
                (centralized mode)
            require_exact_match: When True, require the source manifest and the
                locally registered tensors to name the exact same set and reject
                a zero-match transfer. Used for cross-family (heterogeneous)
                transfers where a name diff can mean vendor-specific hidden or
                derived tensors, which would otherwise leave part or all of the
                target at dummy values while RDMA reports success. Same-family
                transfers leave this False and tolerate subset transfers.
            destination_tensors: Optional registered destination catalog used for
                name matching. Defaults to the most recently registered catalog.

        Returns:
            Tuple of (total_bytes, total_tensors, duration)

        Raises:
            ManifestMismatchError: On a size/dtype mismatch for a shared tensor,
                or, when ``require_exact_match`` is set, on any tensor-name
                mismatch or a zero-match transfer.
        """
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")

        start_time = time.perf_counter()
        self._accelerator_backend.set_device(self._device_id)
        local_tensors = (
            self._tensors if destination_tensors is None else destination_tensors
        )

        if remote_agent_name is None:
            add_start = time.perf_counter()
            remote_agent_name = self.add_remote_agent(source_metadata)
            add_time = time.perf_counter() - add_start
            logger.info(
                f"[TIMING] add_remote_agent: {add_time:.3f}s "
                f"(agent={remote_agent_name}, blob={len(source_metadata)} bytes)"
            )
        else:
            logger.info(f"Using pre-loaded remote agent {remote_agent_name}")

        # Match source tensors to local tensors by name and build raw
        # (addr, size, device_id) descriptor lists for both sides.
        match_start = time.perf_counter()
        remote_descs: list[tuple[int, int, int]] = []
        local_descs: list[tuple[int, int, int]] = []
        total_bytes = 0

        for src_tensor in source_tensors:
            local_tensor = local_tensors.get(src_tensor.name)
            if local_tensor is None:
                continue
            local_size = local_tensor.numel() * local_tensor.element_size()
            if local_size != src_tensor.size:
                transfer_metrics.record_nixl_receive("rejected")
                raise ManifestMismatchError(
                    f"Tensor '{src_tensor.name}' size mismatch: "
                    f"source={src_tensor.size} bytes, local={local_size} bytes"
                )
            local_dtype = str(local_tensor.dtype)
            if local_dtype != src_tensor.dtype:
                transfer_metrics.record_nixl_receive("rejected")
                raise ManifestMismatchError(
                    f"Tensor '{src_tensor.name}' dtype mismatch: "
                    f"source={src_tensor.dtype!r}, local={local_dtype!r}"
                )
            remote_descs.append(
                (src_tensor.addr, src_tensor.size, src_tensor.device_id)
            )
            local_descs.append(
                (
                    local_tensor.data_ptr(),
                    local_size,
                    self._device_id,
                )
            )
            total_bytes += src_tensor.size

        matched_tensors = len(remote_descs)
        match_time = time.perf_counter() - match_start

        # Downgraded to `partial` by the name-diff check below, which does not
        # return early.
        receive_result = "complete"

        # Name-set diff between the source manifest and the locally registered
        # tensors.
        src_names = {s.name for s in source_tensors}
        local_only = sorted(set(local_tensors) - src_names)
        source_only = sorted(src_names - set(local_tensors))
        if local_only or source_only:
            if require_exact_match:
                # Cross-family transfer: a name diff can mean vendor-specific
                # hidden or derived tensors, so completing the transfer would
                # leave the local-only tensors at dummy values while reporting
                # RDMA success. Fail closed instead.
                transfer_metrics.record_nixl_receive("rejected")
                raise ManifestMismatchError(
                    "Tensor name mismatch on heterogeneous transfer: "
                    f"{len(local_only)} local-only "
                    f"(first: {local_only[:5]}), "
                    f"{len(source_only)} source-only "
                    f"(first: {source_only[:5]})"
                )
            # Completing here leaves the local-only tensors at their dummy
            # values while the transfer still reports success, so the warning is
            # the only evidence today. Downgrade the outcome rather than
            # recording now: this path falls through to the same return as a
            # clean transfer, and recording here would count the receive twice.
            receive_result = "partial"
            logger.warning(
                "Tensor name mismatch between source manifest and local "
                "registration: %d local-only, %d source-only",
                len(local_only),
                len(source_only),
            )

        if not remote_descs:
            if require_exact_match:
                transfer_metrics.record_nixl_receive("rejected")
                raise ManifestMismatchError(
                    "No matching tensors found for heterogeneous transfer"
                )
            logger.warning("No matching tensors found for transfer")
            transfer_metrics.record_nixl_receive("empty")
            return 0, 0, 0.0

        logger.info(
            f"[TIMING] match_tensors: {match_time:.3f}s "
            f"({matched_tensors} tensors, {total_bytes / 1e9:.2f} GB)"
        )

        # Prepare transfer descriptors on both sides.
        prep_start = time.perf_counter()
        src_prepped = self._agent.prep_xfer_dlist(
            agent_name=remote_agent_name,
            xfer_list=remote_descs,
            mem_type=self._accelerator_backend.nixl_mem_type,
            backends=self._backends,
        )
        dst_prepped = self._agent.prep_xfer_dlist(
            agent_name="",
            xfer_list=local_descs,
            mem_type=self._accelerator_backend.nixl_mem_type,
            backends=self._backends,
        )
        prep_time = time.perf_counter() - prep_start
        logger.info(f"[TIMING] prep_xfer_dlist: {prep_time:.3f}s")

        indices = list(range(len(remote_descs)))

        # Execute transfer
        handle = self._agent.make_prepped_xfer(
            operation="READ",
            local_xfer_side=dst_prepped,
            local_indices=indices,
            remote_xfer_side=src_prepped,
            remote_indices=indices,
            backends=self._backends,
        )
        self._agent.transfer(handle)

        try:
            self._wait_for_xfer(handle, timeout_seconds, "Transfer")
        finally:
            self._agent.release_xfer_handle(handle)

        # CRITICAL: Synchronize the device to ensure RDMA writes are visible.
        # GPUDirect RDMA writes bypass torch streams, so we must sync.
        self._accelerator_backend.synchronize(self._device_id)

        duration = time.perf_counter() - start_time
        bandwidth_gbps = (total_bytes * 8) / (duration * 1e9) if duration > 0 else 0.0

        logger.info(
            f"Transfer complete: {matched_tensors} tensors, "
            f"{total_bytes / 1e9:.2f} GB in {duration:.2f}s "
            f"({bandwidth_gbps:.1f} Gbps)"
        )

        transfer_metrics.record_nixl_receive(receive_result)
        return total_bytes, matched_tensors, duration

    def execute_read_batch(
        self,
        remote_agent_name: str,
        ranges: list[tuple[int, int, int, int]],
        mem_type: str | None = None,
        timeout_seconds: float | None = None,
    ) -> tuple[int, int, float]:
        """Issue one batched one-sided RDMA READ over arbitrary byte ranges.

        ``ranges`` is a list of ``(remote_addr, local_addr, nbytes,
        remote_device_id)``. Remote addresses must fall within memory the peer
        (``remote_agent_name``, pre-loaded via ``add_remote_agent``) registered;
        local addresses within memory this agent registered. Unlike
        ``receive_from_source`` (whole-tensor, name-matched), this reads the
        exact sub-tensor runs a reshard pull needs - one dest param filled from
        many non-contiguous source segments across a single READ.

        Equivalent to ``post_read_batch`` followed immediately by
        ``await_read_batches``, i.e. one peer at a time. Prefer posting several
        batches and awaiting them together when reading from multiple peers.

        Returns ``(total_bytes, num_reads, duration)``.
        """
        posted = self.post_read_batch(remote_agent_name, ranges, mem_type=mem_type)
        if posted is None:
            return 0, 0, 0.0
        return self.await_read_batches([posted], timeout_seconds=timeout_seconds)

    def post_read_batch(
        self,
        remote_agent_name: str,
        ranges: list[tuple[int, int, int, int]],
        mem_type: str | None = None,
    ) -> PostedRead | None:
        """Prepare and post one batched RDMA READ **without** waiting for it.

        Same ``ranges`` contract as :meth:`execute_read_batch`. Returns ``None``
        when there are no bytes to move. Every returned :class:`PostedRead` must
        be handed to :meth:`await_read_batches`, which owns releasing the handle;
        dropping one leaks it.
        """
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")
        ranges = [r for r in ranges if r[2] > 0]
        if not ranges:
            return None

        mem = mem_type or self._accelerator_backend.nixl_mem_type
        remote_descs = [
            (remote_addr, nbytes, dev) for (remote_addr, _local, nbytes, dev) in ranges
        ]
        local_descs = [
            (local_addr, nbytes, self._device_id)
            for (_remote, local_addr, nbytes, _dev) in ranges
        ]

        # Diagnostic (DEBUG only): what we ask NIXL to READ from the remote.
        # NIXL_ERR_NOT_FOUND at prep means these (addr,size,dev) aren't in a
        # registered region of remote_agent_name as this agent knows it. Gated so
        # steady-state refits don't pay the check_remote_metadata call + formatting.
        if logger.isEnabledFor(logging.DEBUG):
            try:
                _known = self._agent.check_remote_metadata(remote_agent_name)
            except Exception as exc:  # noqa: BLE001 - diagnostics must never break the transfer
                _known = f"n/a ({exc!r})"
            logger.debug(
                "post_read_batch: agent=%s mem=%s reads=%d remote_metadata_loaded=%s remote_sample=%s local_dev=%d",
                remote_agent_name,
                mem,
                len(remote_descs),
                _known,
                [(hex(a), n, d) for (a, n, d) in remote_descs[:3]],
                self._device_id,
            )

        posted_at = time.perf_counter()
        handle = None
        try:
            src_prepped = self._agent.prep_xfer_dlist(
                agent_name=remote_agent_name,
                xfer_list=remote_descs,
                mem_type=mem,
                backends=self._backends,
            )
            dst_prepped = self._agent.prep_xfer_dlist(
                agent_name="",
                xfer_list=local_descs,
                mem_type=mem,
                backends=self._backends,
            )
            indices = list(range(len(ranges)))
            handle = self._agent.make_prepped_xfer(
                operation="READ",
                local_xfer_side=dst_prepped,
                local_indices=indices,
                remote_xfer_side=src_prepped,
                remote_indices=indices,
                backends=self._backends,
            )
            self._agent.transfer(handle)
        except Exception:
            # Nothing is in flight for this batch, so drop its handle here rather
            # than handing a dead batch to await_read_batches.
            if handle is not None:
                self._release_xfer_handle(handle)
            raise

        return PostedRead(
            handle=handle,
            remote_agent_name=remote_agent_name,
            total_bytes=sum(nbytes for (_r, _l, nbytes, _d) in ranges),
            num_ranges=len(ranges),
            posted_at=posted_at,
        )

    def _release_xfer_handle(self, handle: Any) -> None:
        """Release one handle, never raising. Used on cleanup paths where a
        release failure must not mask the error that got us here.

        Logged at WARNING rather than DEBUG: a refused release leaks the
        descriptor list backing the transfer, and because every caller is a
        cleanup path the leak has no other symptom. At DEBUG it is invisible in
        production, where nobody runs the client at that level.
        """
        try:
            self._agent.release_xfer_handle(handle)
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask the cause
            logger.warning("release_xfer_handle failed, handle leaked: %r", exc)

    def await_read_batches(
        self,
        posted: list,
        timeout_seconds: float | None = None,
    ) -> tuple[int, int, float]:
        """Wait for posted READ batches to complete, then release every handle.

        Accepts ``None`` entries so callers can pass ``post_read_batch`` results
        straight through. Releases all handles even when one transfer fails, and
        synchronizes the device once for the whole set rather than per batch.

        Returns ``(total_bytes, num_reads, duration)`` aggregated over the set.
        """
        batches = [p for p in posted if p is not None]
        if not batches:
            return 0, 0, 0.0

        try:
            self._wait_for_xfers(
                [p.handle for p in batches],
                timeout_seconds,
                "NIXL reshard READ batch",
            )
        finally:
            for batch in batches:
                self._release_xfer_handle(batch.handle)

        self._accelerator_backend.synchronize(self._device_id)
        return (
            sum(p.total_bytes for p in batches),
            sum(p.num_ranges for p in batches),
            time.perf_counter() - min(p.posted_at for p in batches),
        )

    def register_dram_buffer(self, buffer: torch.Tensor) -> Any:
        """Register one CPU buffer as NIXL DRAM and refresh agent metadata."""
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")
        if buffer.device.type != "cpu":
            raise RuntimeError("NIXL DRAM buffer must be a CPU tensor")
        if buffer.dtype != torch.uint8:
            raise RuntimeError("NIXL DRAM buffer must use torch.uint8")
        if not buffer.is_contiguous():
            raise RuntimeError("NIXL DRAM buffer must be contiguous")

        registered = self._agent.register_memory([buffer], backends=self._backends)
        self._metadata = self._agent.get_agent_metadata()
        return registered

    def refresh_agent_metadata(self) -> bytes:
        """Refresh and return agent metadata without registering new memory."""
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")
        self._metadata = self._agent.get_agent_metadata()
        return self._metadata

    def deregister_memory(self, registered: Any) -> None:
        """Deregister a memory descriptor returned by register_memory."""
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")
        if registered is not None:
            self._agent.deregister_memory(registered)
            self._metadata = self._agent.get_agent_metadata()

    def receive_dram_into_buffer(
        self,
        remote_agent_name: str,
        remote_addr: int,
        local_buffer: torch.Tensor,
        size: int,
        remote_device_id: int = 0,
        remote_mem_type: str = NIXL_DRAM_MEM_TYPE,
        timeout_seconds: float | None = None,
    ) -> float:
        """Read a remote DRAM range into a registered local CPU uint8 buffer."""
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")
        if size < 0:
            raise ValueError("NIXL DRAM transfer size must be non-negative")
        if local_buffer.device.type != "cpu":
            raise RuntimeError("NIXL DRAM destination must be a CPU tensor")
        if local_buffer.dtype != torch.uint8:
            raise RuntimeError("NIXL DRAM destination must use torch.uint8")
        if not local_buffer.is_contiguous():
            raise RuntimeError("NIXL DRAM destination must be contiguous")
        if size > local_buffer.numel():
            raise ValueError(
                f"NIXL DRAM transfer size {size} exceeds destination buffer "
                f"size {local_buffer.numel()}"
            )
        if size == 0:
            return 0.0

        start_time = time.perf_counter()
        handle = None
        try:
            src_prepped = self._agent.prep_xfer_dlist(
                agent_name=remote_agent_name,
                xfer_list=[(remote_addr, size, remote_device_id)],
                mem_type=remote_mem_type,
                backends=self._backends,
            )
            dst_prepped = self._agent.prep_xfer_dlist(
                agent_name="",
                xfer_list=[(local_buffer.data_ptr(), size, 0)],
                mem_type=NIXL_DRAM_MEM_TYPE,
                backends=self._backends,
            )
            handle = self._agent.make_prepped_xfer(
                operation="READ",
                local_xfer_side=dst_prepped,
                local_indices=[0],
                remote_xfer_side=src_prepped,
                remote_indices=[0],
                backends=self._backends,
            )
            self._agent.transfer(handle)

            self._wait_for_xfer(
                handle,
                timeout_seconds,
                "NIXL DRAM transfer",
            )
            duration = time.perf_counter() - start_time
            logger.info(
                "NIXL DRAM READ complete: %.2f MiB in %.3fs",
                size / (1024 * 1024),
                duration,
            )
            return duration
        finally:
            if handle is not None:
                self._agent.release_xfer_handle(handle)

    def is_healthy(self) -> bool:
        """Whether the agent is initialized and has no observed transfer failure."""
        if self._agent is None or len(self._metadata) == 0:
            return False
        return self._data_plane_error is None

    @property
    def data_plane_error(self) -> str | None:
        """Last data-plane failure observed on a transfer, or None."""
        return self._data_plane_error

    def shutdown(self) -> None:
        """Disconnect remote agents before releasing local NIXL resources."""
        if self._atexit_registered:
            atexit.unregister(self.shutdown)
            self._atexit_registered = False
        disconnected = self.disconnect_remote_agents()
        if self._agent is not None:
            for registered in reversed(self._registered_memory):
                try:
                    self._agent.deregister_memory(registered)
                except Exception:
                    logger.warning(
                        "Failed to deregister NIXL memory during shutdown",
                        exc_info=True,
                    )
        self._registered_memory = []
        self._agent = None
        self._metadata = b""
        self._tensor_descriptors = []
        self._tensors = {}
        self._remote_agents = {}
        logger.info(
            "NixlTransferManager shutdown complete (%d remote agent(s) disconnected)",
            disconnected,
        )
