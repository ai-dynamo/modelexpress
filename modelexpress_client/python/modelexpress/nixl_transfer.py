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

import logging
import os
import time
from typing import TYPE_CHECKING, Any

import torch

from . import envs
from . import ucx_utils
from .accelerators import (
    AcceleratorBackend,
    CudaAcceleratorBackend,
)
from .refit_timing import add_refit_bytes, refit_span
from .types import ManifestMismatchError, TensorDescriptor

if TYPE_CHECKING:
    from .vmm.arena import VmmArena

logger = logging.getLogger("modelexpress.nixl_transfer")

NIXL_AVAILABLE = False
NixlAgent = None
nixl_agent_config = None
try:
    from nixl._api import nixl_agent as NixlAgent
    from nixl._api import nixl_agent_config
    NIXL_AVAILABLE = True
except ImportError:
    pass


SUPPORTED_NIXL_BACKENDS = ("UCX", "LIBFABRIC")
DEFAULT_NIXL_BACKEND = "UCX"
NIXL_DRAM_MEM_TYPE = "DRAM"


from dataclasses import dataclass


@dataclass
class SlicedTransferRequest:
    """One per-tensor sub-slice pull request for :meth:`receive_sliced_from_source`.

    Each request says "read ``slice_bytes`` bytes starting at offset
    ``source_offset_bytes`` from the remote tensor named ``name``, and
    write them into the local memory backing ``dest_view`` (which must be
    a contiguous tensor of exactly ``slice_bytes`` bytes — typically a
    ``narrow`` of a pre-registered destination buffer)."

    The combined transfer is built from a list of these — N requests
    produces one NIXL transfer with N (remote, local) descriptor pairs.
    """

    name: str
    source_offset_bytes: int
    slice_bytes: int
    dest_view: torch.Tensor


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


def _pool_reg_enabled() -> bool:
    """Whether allocation-level pool registration is enabled.

    MX_POOL_REG=1 enables it; default is per-tensor registration. Read at
    call time so tests can toggle the env var without re-importing.
    """
    return envs.MX_POOL_REG


# NIXL memtype strings for the two locations we support. Kept as module
# constants so tests and adapters can reference them without hardcoding
# strings.
#
# NIXL's Python API accepts both the native names ("VRAM", "DRAM") and
# lowercase aliases ("cuda", "cpu"). We use the aliases because the
# existing GPU path already hard-codes "cuda" and the CPU alias matches
# ``tensor.device.type == "cpu"`` — keeps the mapping obvious in
# ``_resolve_local_mem_type``. NIXL's ``prep_xfer_dlist`` requires the
# exact key (case-sensitive), so lowercase "dram" (which we used
# initially) fails with KeyError. Verified from
# ``nixl_cu12._api.nixl_agent.__init__`` on the deployed image
# (2026-07-02): both "cpu" and "DRAM" map to ``DRAM_SEG``.
_MEM_TYPE_CUDA = "cuda"
_MEM_TYPE_DRAM = "cpu"


def _resolve_local_mem_type(tensors: dict[str, torch.Tensor]) -> str:
    """Pick the NIXL memtype for a group of locally-registered tensors.

    All tensors in one NixlTransferManager must share a memtype because
    ``prep_xfer_dlist`` takes a single ``mem_type`` per side. We enforce
    uniformity here to fail fast at register time rather than mid-transfer.

    Returns:
        ``"cuda"`` if every tensor is on a CUDA device.
        ``"dram"`` if every tensor is on CPU (pinned or otherwise —
        the caller controls whether it's pinned; NIXL just needs the
        DRAM memtype for host-side registration).

    Raises:
        ValueError: on empty input or mixed-device sets.
    """
    if not tensors:
        raise ValueError(
            "_resolve_local_mem_type: empty tensor set; nothing to register"
        )
    devices = {t.device.type for t in tensors.values()}
    if devices == {"cuda"}:
        return _MEM_TYPE_CUDA
    if devices == {"cpu"}:
        return _MEM_TYPE_DRAM
    raise ValueError(
        f"_resolve_local_mem_type: mixed or unsupported device set "
        f"{sorted(devices)!r}. Register CUDA tensors and CPU tensors "
        "with separate NixlTransferManager instances."
    )


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
        # Memory type of the LOCAL tensors registered via register_tensors.
        # Set by register_tensors from the first tensor's device; used by
        # the transfer paths' prep_xfer_dlist local-side call. Defaults
        # to "cuda" for back-compat with pre-DRAM behavior.
        #
        # DRAM support (pinned-CPU staging, Istvan Phase 0.5) lets the
        # receiver cache buffers on pinned host memory instead of GPU
        # HBM, freeing ~model-shard-sized HBM at the cost of an async
        # H2D copy per refit cycle (which the pipeline already needs
        # to hand tensors to load_weights).
        self._local_mem_type: str = "cuda"

    @property
    def agent_name(self) -> str:
        """Get NIXL agent name."""
        return self._agent_name

    @property
    def nixl_metadata(self) -> bytes:
        """Get NIXL metadata for this agent."""
        return self._metadata

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
                logger.info(
                    f"NIXL listen thread enabled on port {self._listen_port}"
                )
            elif nixl_agent_config:
                config = nixl_agent_config(backends=self._backends)
            else:
                config = None
            self._agent = NixlAgent(self._agent_name, config)
            logger.info(
                f"NIXL agent '{self._agent_name}' created on device "
                f"{self._device_id} (backend={self._backend})"
            )
        finally:
            if saved_ucx_tls is not None:
                os.environ["UCX_TLS"] = saved_ucx_tls
            elif envs.is_set("UCX_TLS"):
                os.environ.pop("UCX_TLS")

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
            tensor_descriptors.append(TensorDescriptor(
                name=name,
                addr=tensor.data_ptr(),
                size=tensor.numel() * tensor.element_size(),
                device_id=self._device_id,
                dtype=str(tensor.dtype),
            ))
        self._tensor_descriptors = tensor_descriptors
        return tensor_descriptors

    def register_tensors(self, tensors: dict[str, torch.Tensor]) -> bytes:
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

        # Detect and validate memtype uniformity across the tensor set.
        # NIXL's prep_xfer_dlist local side needs a single memtype per
        # transfer, so we don't support mixed device+host registrations
        # in one manager. Callers wanting both should use two managers
        # (rare — the current pinned-CPU staging path allocates all
        # buffers on host).
        self._local_mem_type = _resolve_local_mem_type(tensors)

        tensor_descriptors = self._build_tensor_descriptors(tensors)

        # Phase 1: Discover CUDA allocation boundaries (if pool reg enabled)
        # Pool registration only applies to CUDA tensors — the discovery
        # path calls cuMemGetAddressRange, which is a device-memory API.
        # For DRAM registrations we always fall through to per-tensor.
        alloc_discovery_start = time.perf_counter()
        # Pool reg only for on-device (cuda) local buffers on a backend that
        # supports it. The DRAM/host (Phase 0.5) path uses per-tensor reg.
        if _pool_reg_enabled() and self._local_mem_type == "cuda":
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
            if _pool_reg_enabled() and self._local_mem_type != "cuda":
                logger.info(
                    "Pool registration skipped: local memtype is %r; "
                    "using per-tensor registration",
                    self._local_mem_type,
                )
            else:
                logger.info("Pool registration disabled (MX_POOL_REG != '1'), using per-tensor registration")
        alloc_discovery_time = time.perf_counter() - alloc_discovery_start

        # Phase 2: Register memory with NIXL (ibv_reg_mr kernel calls)
        nixl_reg_start = time.perf_counter()
        if allocations:
            alloc_tuples = [
                (base, size, self._device_id, "")
                for base, size in allocations
            ]
            self._agent.register_memory(
                alloc_tuples,
                mem_type=self._accelerator_backend.nixl_mem_type,
                backends=self._backends,
            )
            reg_count = len(allocations)
        else:
            tensor_list = list(tensors.values())
            # For DRAM (pinned-CPU) tensors we must pass mem_type
            # explicitly — NIXL's torch-tensor auto-detect keys off
            # tensor.is_cuda but the explicit path leaves nothing to
            # infer.
            self._agent.register_memory(
                tensor_list,
                mem_type=self._local_mem_type,
                backends=self._backends,
            )
            reg_count = len(tensor_list)
        nixl_reg_time = time.perf_counter() - nixl_reg_start

        # Phase 3: Get agent metadata blob
        metadata_start = time.perf_counter()
        self._metadata = self._agent.get_agent_metadata()
        metadata_time = time.perf_counter() - metadata_start

        total_time = alloc_discovery_time + nixl_reg_time + metadata_time
        reduction = (1 - reg_count / len(tensor_descriptors)) * 100 if tensor_descriptors else 0
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

    def rebind_tensors(self, tensors: dict[str, torch.Tensor]) -> None:
        """Point the active local tensor set at ``tensors`` without re-registering.

        Buffer-caching call paths register a set of destination buffers
        once (e.g. ``_mx_megatron_buffers`` in the Dynamo / NeMo-RL
        extensions) and reuse them across refit cycles. Meanwhile other
        code paths — notably ``receive_weights_scratch`` — call
        :meth:`register_tensors` mid-flight and replace ``self._tensors``
        with a different (temporary) set. Without a rebind step,
        subsequent transfers to the cached buffers would use the stale
        tensor set for name→address resolution and, worse for DRAM,
        would use the stale ``self._local_mem_type``.

        This method rebuilds the local descriptor list and re-derives
        ``self._local_mem_type`` from the given tensors. It does NOT call
        ``register_memory`` — the underlying buffers must already be
        NIXL-registered (via a prior ``register_tensors``); NIXL keeps
        those registrations live independently of ``self._tensors``.

        Raises:
            RuntimeError: if the agent is not initialized.
            ValueError: from :func:`_resolve_local_mem_type` on empty
                input or mixed-device sets.
        """
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")
        self._local_mem_type = _resolve_local_mem_type(tensors)
        self._build_tensor_descriptors(tensors)

    def register_arena(self, arena: VmmArena, tensors: dict[str, torch.Tensor]) -> bytes:
        """Register a VmmArena's full bump range as a single NIXL region.

        The arena owns a contiguous VA range; at end-of-load the bump
        pointer's [base, base+used) covers every allocation we've ever
        made (including holes from intervening frees). NIXL's
        `register_memory` with `mem_type="VRAM"` over this range
        consumes a dmabuf via `ibv_reg_dmabuf_mr` and produces ONE
        lkey/rkey covering all live tensors.

        Empirically validated on Blackwell + ConnectX over InfiniBand
        against a CUDA VMM range with multiple cuMemCreate handles and
        mid-range holes (chunks unmapped + released after the export):
        registration succeeds, the dmabuf attach pins the currently-
        mapped physical pages, and the HCA translation table survives
        subsequent CUDA-side unmaps.

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

        nixl_reg_start = time.perf_counter()
        self._agent.register_memory(
            [(base, used, self._device_id, "")],
            mem_type=self._accelerator_backend.nixl_mem_type,
            backends=self._backends,
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

        logger.info(
            f"Fetching remote metadata from {remote_agent_name} at {ip}:{port}"
        )
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
        return remote_agent_name

    def receive_from_source(
        self,
        source_metadata: bytes,
        source_tensors: list[TensorDescriptor],
        timeout_seconds: float | None = None,
        remote_agent_name: str | None = None,
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

        Returns:
            Tuple of (total_bytes, total_tensors, duration)
        """
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")

        start_time = time.perf_counter()
        self._accelerator_backend.set_device(self._device_id)

        if remote_agent_name is None:
            add_start = time.perf_counter()
            with refit_span(
                "setup_registration",
                metadata={"operation": "add_remote_agent"},
            ):
                remote_agent_name = self._agent.add_remote_agent(source_metadata)
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

        with refit_span(
            "transfer_planning",
            metadata={"operation": "match_tensors"},
        ):
            for src_tensor in source_tensors:
                local_tensor = self._tensors.get(src_tensor.name)
                if local_tensor is None:
                    continue
                local_size = local_tensor.numel() * local_tensor.element_size()
                if local_size != src_tensor.size:
                    raise ManifestMismatchError(
                        f"Tensor '{src_tensor.name}' size mismatch: "
                        f"source={src_tensor.size} bytes, local={local_size} bytes"
                    )
                local_dtype = str(local_tensor.dtype)
                if local_dtype != src_tensor.dtype:
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

        if not remote_descs:
            logger.warning("No matching tensors found for transfer")
            return 0, 0, 0.0

        logger.info(
            f"[TIMING] match_tensors: {match_time:.3f}s "
            f"({matched_tensors} tensors, {total_bytes / 1e9:.2f} GB)"
        )

        # Prepare transfer descriptors on both sides.
        # Remote side is always "cuda" — the source is a trainer/other
        # receiver publishing GPU tensors. Local side depends on where
        # this receiver allocated its dest buffers (see
        # self._local_mem_type, set by register_tensors).
        prep_start = time.perf_counter()
        with refit_span(
            "transfer_planning",
            metadata={"operation": "prepare_descriptors"},
        ):
            src_prepped = self._agent.prep_xfer_dlist(
                agent_name=remote_agent_name,
                xfer_list=remote_descs,
                mem_type=self._accelerator_backend.nixl_mem_type,
                backends=self._backends,
            )
            dst_prepped = self._agent.prep_xfer_dlist(
                agent_name="",
                xfer_list=local_descs,
                # Local side uses our dynamic memtype (cuda for GPU buffers, cpu
                # for the Phase-0.5 host path); falls back to the backend default.
                mem_type=self._local_mem_type
                or self._accelerator_backend.nixl_mem_type,
                backends=self._backends,
            )
        prep_time = time.perf_counter() - prep_start
        logger.info(f"[TIMING] prep_xfer_dlist: {prep_time:.3f}s")

        indices = list(range(len(remote_descs)))

        # Execute transfer
        with refit_span("wire_transfer"):
            handle = self._agent.make_prepped_xfer(
                operation="READ",
                local_xfer_side=dst_prepped,
                local_indices=indices,
                remote_xfer_side=src_prepped,
                remote_indices=indices,
                backends=self._backends,
            )
            self._agent.transfer(handle)

            # Wait for completion
            start_wait = time.perf_counter()
            while True:
                if (
                    timeout_seconds is not None
                    and time.perf_counter() - start_wait >= timeout_seconds
                ):
                    self._agent.release_xfer_handle(handle)
                    raise TimeoutError("Transfer timed out")

                status = self._agent.check_xfer_state(handle)
                if status in ("DONE", "SUCCESS"):
                    self._agent.release_xfer_handle(handle)
                    break
                if status in ("ERR", "ERROR", "FAIL"):
                    self._agent.release_xfer_handle(handle)
                    raise RuntimeError(f"Transfer failed with status {status}")
                time.sleep(0.001)
        add_refit_bytes(total_bytes)

        # CRITICAL: Synchronize the device to ensure RDMA writes are visible.
        # GPUDirect RDMA writes bypass torch streams, so we must sync — but
        # only for on-device buffers. For the DRAM (pinned-CPU / Phase 0.5)
        # path RDMA hits host memory directly, so no device sync is required
        # (and skipping lets CPU-only unit tests exercise this path).
        if self._local_mem_type == "cuda":
            with refit_span("receive_sync"):
                self._accelerator_backend.synchronize(self._device_id)

        duration = time.perf_counter() - start_time
        bandwidth_gbps = (total_bytes * 8) / (duration * 1e9) if duration > 0 else 0.0

        logger.info(
            f"Transfer complete: {matched_tensors} tensors, "
            f"{total_bytes / 1e9:.2f} GB in {duration:.2f}s "
            f"({bandwidth_gbps:.1f} Gbps)"
        )

        return total_bytes, matched_tensors, duration

    def receive_sliced_from_source(
        self,
        source_metadata: bytes,
        source_tensors: list[TensorDescriptor],
        slice_requests: list["SlicedTransferRequest"],
        timeout_seconds: float | None = None,
        remote_agent_name: str | None = None,
    ) -> tuple[int, int, float]:
        """Pull sub-slices of remote tensors directly into local dest views.

        Unlike :meth:`receive_from_source` (which transfers full tensors into
        pre-registered named buffers), this primitive transfers per-tensor
        sub-ranges directly into caller-provided dest views — one combined
        NIXL transfer with N (source slice, dest view) descriptor pairs.

        This is the v1 "sliced pull" needed for bandwidth-optimal mixed-TP
        in the target-wider direction: each receiver pulls only the bytes
        it actually needs from each source rank, instead of pulling the
        full source manifest and slicing on the host (the v0 fallback in
        :func:`MxRefitReceiver.receive_weights_scratch`).

        Args:
            source_metadata: NIXL agent metadata from the remote source.
            source_tensors: source manifest (one ``TensorDescriptor`` per
                published tensor, ``addr`` + ``size`` referring to the
                source's GPU memory). The request's ``name`` field is
                looked up against this list.
            slice_requests: each request specifies a single (source tensor,
                source byte offset, byte count, local dest view) tuple.
                The local dest view must be **contiguous** in memory — if
                the caller wants a non-contiguous slice (e.g. a row-parallel
                axis-1 ``narrow``), it must use the v0 scratch+copy path
                instead, since RDMA writes need a flat byte range.
            timeout_seconds: as for :meth:`receive_from_source`.
            remote_agent_name: as for :meth:`receive_from_source`. When
                None, ``add_remote_agent(source_metadata)`` is called once
                up front.

        Returns:
            ``(total_bytes_transferred, num_slices, elapsed_seconds)``.

        Raises:
            RuntimeError: if any dest view is non-contiguous, if a request
                names a tensor not in the source manifest, or if the
                source offset/size goes past the source tensor's end.
        """
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")
        if not slice_requests:
            return 0, 0, 0.0

        start_time = time.perf_counter()
        torch.cuda.set_device(self._device_id)

        if remote_agent_name is None:
            remote_agent_name = self._agent.add_remote_agent(source_metadata)

        # Index source tensors by name for fast lookup.
        by_name: dict[str, TensorDescriptor] = {t.name: t for t in source_tensors}

        remote_descs: list[tuple[int, int, int]] = []
        local_descs: list[tuple[int, int, int]] = []
        total_bytes = 0

        for req in slice_requests:
            src = by_name.get(req.name)
            if src is None:
                raise RuntimeError(
                    f"receive_sliced_from_source: tensor {req.name!r} not in "
                    f"source manifest (have {len(by_name)} tensors)"
                )
            if req.source_offset_bytes < 0:
                raise RuntimeError(
                    f"receive_sliced_from_source: negative source_offset_bytes "
                    f"on {req.name!r}"
                )
            if req.source_offset_bytes + req.slice_bytes > src.size:
                raise RuntimeError(
                    f"receive_sliced_from_source: slice on {req.name!r} runs "
                    f"past end of source (offset={req.source_offset_bytes} + "
                    f"size={req.slice_bytes} > src.size={src.size})"
                )
            dest = req.dest_view
            if not dest.is_contiguous():
                raise RuntimeError(
                    f"receive_sliced_from_source: dest view for {req.name!r} "
                    f"is non-contiguous (shape={tuple(dest.shape)}, "
                    f"stride={dest.stride()}). Use receive_weights_scratch + "
                    f"host-side copy for non-contiguous slices (e.g. "
                    f"row-parallel axis-1 narrows)."
                )
            dest_bytes = dest.numel() * dest.element_size()
            if dest_bytes != req.slice_bytes:
                raise RuntimeError(
                    f"receive_sliced_from_source: dest view size {dest_bytes} "
                    f"!= slice_bytes {req.slice_bytes} on {req.name!r}"
                )

            remote_descs.append(
                (src.addr + req.source_offset_bytes, req.slice_bytes, src.device_id)
            )
            local_descs.append(
                (dest.data_ptr(), req.slice_bytes, self._device_id)
            )
            total_bytes += req.slice_bytes

        # One combined transfer. Same memtype logic as receive_from_source:
        # remote is always "cuda" (trainer GPU), local uses whatever the
        # receiver registered under.
        src_prepped = self._agent.prep_xfer_dlist(
            agent_name=remote_agent_name,
            xfer_list=remote_descs,
            mem_type="cuda",
            backends=self._backends,
        )
        dst_prepped = self._agent.prep_xfer_dlist(
            agent_name="",
            xfer_list=local_descs,
            mem_type=self._local_mem_type,
            backends=self._backends,
        )
        indices = list(range(len(remote_descs)))
        handle = self._agent.make_prepped_xfer(
            operation="READ",
            local_xfer_side=dst_prepped,
            local_indices=indices,
            remote_xfer_side=src_prepped,
            remote_indices=indices,
            backends=self._backends,
        )
        self._agent.transfer(handle)

        start_wait = time.perf_counter()
        while True:
            if timeout_seconds is not None and time.perf_counter() - start_wait >= timeout_seconds:
                self._agent.release_xfer_handle(handle)
                raise TimeoutError("Sliced transfer timed out")
            status = self._agent.check_xfer_state(handle)
            if status in ("DONE", "SUCCESS"):
                self._agent.release_xfer_handle(handle)
                break
            if status in ("ERR", "ERROR", "FAIL"):
                self._agent.release_xfer_handle(handle)
                raise RuntimeError(f"Sliced transfer failed with status {status}")
            time.sleep(0.001)

        if self._local_mem_type == "cuda":
            torch.cuda.synchronize(self._device_id)
        duration = time.perf_counter() - start_time
        bw_gbps = (total_bytes * 8) / (duration * 1e9) if duration > 0 else 0.0
        logger.info(
            f"Sliced transfer complete: {len(slice_requests)} slices, "
            f"{total_bytes / 1e9:.2f} GB in {duration:.2f}s ({bw_gbps:.1f} Gbps)"
        )
        return total_bytes, len(slice_requests), duration
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

            wait_start = time.perf_counter()
            while True:
                if (
                    timeout_seconds is not None
                    and time.perf_counter() - wait_start >= timeout_seconds
                ):
                    raise TimeoutError("NIXL DRAM transfer timed out")
                status = self._agent.check_xfer_state(handle)
                if status in ("DONE", "SUCCESS"):
                    duration = time.perf_counter() - start_time
                    logger.info(
                        "NIXL DRAM READ complete: %.2f MiB in %.3fs",
                        size / (1024 * 1024),
                        duration,
                    )
                    return duration
                if status in ("ERR", "ERROR", "FAIL"):
                    raise RuntimeError(f"NIXL DRAM transfer failed with status {status}")
                time.sleep(0.001)
        finally:
            if handle is not None:
                self._agent.release_xfer_handle(handle)

    def is_healthy(self) -> bool:
        """Check if the NIXL agent is initialized and has registered metadata."""
        return self._agent is not None and len(self._metadata) > 0

    def shutdown(self) -> None:
        """Clean up NIXL resources.

        Rebinds ``_tensor_descriptors`` and ``_tensors`` to fresh empty
        containers instead of mutating in place. Belt-and-suspenders:
        even if a future caller bypasses ``register_tensors`` and
        aliases ``_tensors`` directly, shutdown will not mutate the
        shared container out from under them.
        """
        self._agent = None
        self._metadata = b""
        self._tensor_descriptors = []
        self._tensors = {}
        logger.info("NixlTransferManager shutdown complete")
