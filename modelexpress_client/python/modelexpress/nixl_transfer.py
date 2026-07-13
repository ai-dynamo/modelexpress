# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NIXL Transfer Manager for GPU-to-GPU weight transfers.

This module provides the NixlTransferManager class that handles all NIXL-related
operations including agent creation, tensor registration, and RDMA transfers.

Each vLLM worker creates its own NixlTransferManager instance to manage
a single NIXL agent for that worker's GPU.
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING, Any

import torch

from . import ucx_utils
from .tensor_utils import (
    debug_tensor_enabled,
    tensor_debug_summary,
    tensor_descriptor_layout,
)
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


def is_nixl_available() -> bool:
    """Check if NIXL is available."""
    return NIXL_AVAILABLE


def _resolve_nixl_backend() -> str:
    """Resolve the NIXL backend from MX_NIXL_BACKEND.

    Defaults to UCX. Set MX_NIXL_BACKEND=LIBFABRIC on AWS EFA.
    """
    raw = os.environ.get("MX_NIXL_BACKEND", DEFAULT_NIXL_BACKEND).strip().upper()
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
    return os.environ.get("MX_POOL_REG", "0") == "1"


def _source_tensor_summary(desc: TensorDescriptor) -> str:
    return (
        f"name={desc.name!r} dtype={desc.dtype} nbytes={desc.size} "
        f"shape={desc.shape or []} stride={desc.stride or []} "
        f"storage_offset={desc.storage_offset} "
        f"storage_nbytes={desc.storage_nbytes} "
        f"original_shape={desc.original_shape or []} "
        f"original_dtype={desc.original_dtype!r} "
        f"original_nbytes={desc.original_nbytes} "
        f"layout_kind={desc.layout_kind!r} addr=0x{desc.addr:x} "
        f"device={desc.device_id}"
    )


class NixlTransferManager:
    """
    Manages a single NIXL agent and RDMA transfers for GPU tensors.

    Each vLLM worker creates its own instance of this class to handle:
    - Creating and managing a NIXL agent for the worker's GPU
    - Registering tensors with NIXL for RDMA access
    - Executing transfers to receive weights from remote sources

    Args:
        agent_name: Name for the NIXL agent
        device_id: GPU device ID for this worker
    """

    def __init__(self, agent_name: str, device_id: int, listen_port: int | None = None):
        self._agent_name = agent_name
        self._device_id = device_id
        self._listen_port = listen_port

        self._backend = _resolve_nixl_backend()
        self._backends = [self._backend]

        self._agent: Any = None
        self._metadata: bytes = b""
        self._tensor_descriptors: list[TensorDescriptor] = []
        self._tensors: dict[str, torch.Tensor] = {}

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

        torch.cuda.set_device(self._device_id)

        # Let UCX auto-detect transports (RoCE, TCP, etc).
        # OMPI_MCA_pml=ob1 keeps MPI on TCP independently.
        # Only override UCX_TLS if explicitly set to "tcp" (legacy compat).
        saved_ucx_tls = os.environ.get("UCX_TLS")
        nixl_ucx_tls = os.environ.get("NIXL_UCX_TLS")
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
            elif "UCX_TLS" in os.environ:
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
                **tensor_descriptor_layout(
                    tensor,
                    "storage_view" if name.endswith(".__storage") else "contiguous",
                ),
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

        tensor_descriptors = self._build_tensor_descriptors(tensors)

        # Phase 1: Discover CUDA allocation boundaries (if pool reg enabled)
        alloc_discovery_start = time.perf_counter()
        if _pool_reg_enabled():
            allocations = self._find_cuda_allocations(tensor_descriptors)
        else:
            allocations = None
            logger.info("Pool registration disabled (MX_POOL_REG != '1'), using per-tensor registration")
        alloc_discovery_time = time.perf_counter() - alloc_discovery_start

        # Phase 2: Register memory with NIXL (ibv_reg_mr kernel calls)
        nixl_reg_start = time.perf_counter()
        if allocations:
            alloc_tuples = [
                (base, size, self._device_id, "")
                for base, size in allocations
            ]
            self._agent.register_memory(alloc_tuples, mem_type="cuda", backends=self._backends)
            reg_count = len(allocations)
        else:
            tensor_list = list(tensors.values())
            self._agent.register_memory(tensor_list, backends=self._backends)
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

    def register_arena(self, arena: VmmArena, tensors: dict[str, torch.Tensor]) -> bytes:
        """Register a VmmArena's full bump range as a single NIXL region.

        The arena owns a contiguous VA range; at end-of-load the bump
        pointer's [base, base+used) covers every allocation we've ever
        made (including holes from intervening frees). NIXL's
        `register_memory` with `mem_type="cuda"` over this range
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

        nixl_reg_start = time.perf_counter()
        self._agent.register_memory(
            [(base, used, self._device_id, "")],
            mem_type="cuda",
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
        torch.cuda.set_device(self._device_id)

        if remote_agent_name is None:
            add_start = time.perf_counter()
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
        source_names = {t.name for t in source_tensors}
        local_names = set(self._tensors)
        common_names = source_names & local_names
        if len(source_names) != len(local_names):
            logger.warning(
                "Tensor manifest count differs before RDMA: source=%d local=%d "
                "common=%d source_only=%d local_only=%d",
                len(source_names),
                len(local_names),
                len(common_names),
                len(source_names - local_names),
                len(local_names - source_names),
            )

        for src_tensor in source_tensors:
            local_tensor = self._tensors.get(src_tensor.name)
            if local_tensor is None:
                continue
            local_size = local_tensor.numel() * local_tensor.element_size()
            if debug_tensor_enabled(src_tensor.name):
                logger.warning(
                    "Target manifest tensor: %s | source_addr=0x%x "
                    "source_size=%d source_dtype=%s source_device=%d",
                    tensor_debug_summary(src_tensor.name, local_tensor),
                    src_tensor.addr,
                    src_tensor.size,
                    src_tensor.dtype,
                    src_tensor.device_id,
                )
            if local_size != src_tensor.size:
                raise ManifestMismatchError(
                    f"Tensor '{src_tensor.name}' size mismatch: "
                    f"source={src_tensor.size} bytes, local={local_size} bytes; "
                    f"source_layout=({_source_tensor_summary(src_tensor)}); "
                    f"local_layout=({tensor_debug_summary(src_tensor.name, local_tensor)})"
                )
            local_dtype = str(local_tensor.dtype)
            if local_dtype != src_tensor.dtype:
                raise ManifestMismatchError(
                    f"Tensor '{src_tensor.name}' dtype mismatch: "
                    f"source={src_tensor.dtype!r}, local={local_dtype!r}"
                )
            local_layout = tensor_descriptor_layout(
                local_tensor,
                "storage_view" if src_tensor.name.endswith(".__storage") else "contiguous",
            )
            if src_tensor.shape and list(src_tensor.shape) != local_layout["shape"]:
                raise ManifestMismatchError(
                    f"Tensor '{src_tensor.name}' shape mismatch: "
                    f"source={list(src_tensor.shape)}, local={local_layout['shape']}"
                )
            if src_tensor.stride and list(src_tensor.stride) != local_layout["stride"]:
                raise ManifestMismatchError(
                    f"Tensor '{src_tensor.name}' stride mismatch: "
                    f"source={list(src_tensor.stride)}, local={local_layout['stride']}"
                )
            if (
                src_tensor.storage_nbytes
                and src_tensor.storage_nbytes != local_layout["storage_nbytes"]
            ):
                raise ManifestMismatchError(
                    f"Tensor '{src_tensor.name}' storage size mismatch: "
                    f"source={src_tensor.storage_nbytes} bytes, "
                    f"local={local_layout['storage_nbytes']} bytes"
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
        prep_start = time.perf_counter()
        src_prepped = self._agent.prep_xfer_dlist(
            agent_name=remote_agent_name,
            xfer_list=remote_descs,
            mem_type="cuda",
            backends=self._backends,
        )
        dst_prepped = self._agent.prep_xfer_dlist(
            agent_name="",
            xfer_list=local_descs,
            mem_type="cuda",
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

        # Wait for completion
        start_wait = time.perf_counter()
        while True:
            if timeout_seconds is not None and time.perf_counter() - start_wait >= timeout_seconds:
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

        # CRITICAL: Synchronize CUDA to ensure RDMA writes are visible.
        # GPUDirect RDMA writes bypass CUDA streams, so we must sync.
        torch.cuda.synchronize(self._device_id)

        duration = time.perf_counter() - start_time
        bandwidth_gbps = (total_bytes * 8) / (duration * 1e9) if duration > 0 else 0.0

        logger.info(
            f"Transfer complete: {matched_tensors} tensors, "
            f"{total_bytes / 1e9:.2f} GB in {duration:.2f}s "
            f"({bandwidth_gbps:.1f} Gbps)"
        )

        return total_bytes, matched_tensors, duration

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
