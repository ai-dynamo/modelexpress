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
from typing import Any

import torch

from .types import TensorDescriptor

logger = logging.getLogger("modelexpress.nixl_transfer")

# MX_POOL_REG=0 disables allocation-level registration,
# falling back to per-tensor registration.
POOL_REG_ENABLED = os.environ.get("MX_POOL_REG", "1") != "0"

# MX_COALESCE_TRANSFERS=1 enables transfer coalescing,
# merging adjacent per-tensor RDMA READs into larger regions.
# Disabled by default: benchmarks show per-tensor transfers are
# faster due to better UCX pipelining. Independent of MX_POOL_REG.
COALESCE_ENABLED = os.environ.get("MX_COALESCE_TRANSFERS", "0") == "1"

NIXL_AVAILABLE = False
NixlAgent = None
nixl_agent_config = None
try:
    from nixl._api import nixl_agent as NixlAgent
    from nixl._api import nixl_agent_config
    NIXL_AVAILABLE = True
except ImportError:
    pass


def is_nixl_available() -> bool:
    """Check if NIXL is available."""
    return NIXL_AVAILABLE


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

        self._agent: Any = None
        self._metadata: bytes = b""
        self._tensor_descriptors: list[TensorDescriptor] = []
        self._tensors: dict[str, torch.Tensor] = {}
        self._alloc_ends: list[int] = []

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

    @property
    def alloc_ends(self) -> list[int]:
        """Get CUDA allocation end addresses (for coalescing boundaries)."""
        return self._alloc_ends

    def initialize(self) -> None:
        """Initialize the NIXL agent."""
        if not NIXL_AVAILABLE:
            raise RuntimeError("NIXL is not available")

        if self._agent is not None:
            return

        torch.cuda.set_device(self._device_id)

        if self._listen_port is not None and nixl_agent_config:
            config = nixl_agent_config(
                backends=["UCX"],
                enable_listen_thread=True,
                listen_port=self._listen_port,
            )
            logger.info(
                f"NIXL listen thread enabled on port {self._listen_port}"
            )
        elif nixl_agent_config:
            config = nixl_agent_config(backends=["UCX"])
        else:
            config = None
        self._agent = NixlAgent(self._agent_name, config)
        logger.info(f"NIXL agent '{self._agent_name}' created on device {self._device_id}")

    def register_tensors(
        self,
        tensors: dict[str, torch.Tensor],
        vmm_range: tuple[int, int] | None = None,
    ) -> bytes:
        """
        Register tensors with NIXL for RDMA access.

        Three registration modes (best to worst):
        1. VMM: caller provides a single (base, size) VA range covering all
           tensors (from vmm_compact). One ibv_reg_mr call.
        2. Pool: discover CUDA allocation boundaries via cuMemGetAddressRange.
           One ibv_reg_mr per cudaMalloc block (~300 calls for DSV3).
        3. Per-tensor: register each tensor individually (~2600 calls).

        Per-tensor descriptors are always preserved for application-level
        name-based matching during transfers regardless of registration mode.

        CRITICAL: self._tensors must hold the SAME tensor objects as
        param.data in vLLM. Do NOT call .contiguous() - that would create
        copies and RDMA writes would go to the wrong memory.

        Args:
            tensors: Dictionary of tensor name -> tensor
            vmm_range: Optional (base_addr, total_size) of the VMM arena.
                When provided, registers this single range instead of
                discovering allocations or registering per-tensor.

        Returns:
            NIXL metadata bytes for this agent
        """
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")

        self._tensors = tensors
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

        # Phase 1: Determine registration regions
        alloc_discovery_start = time.perf_counter()
        if vmm_range is not None:
            # VMM mode: all tensors are in a single contiguous VA range.
            # Register the individual tensors (not a single big tensor)
            # because NIXL's prep_xfer_dlist matches against registered
            # tensor addresses, not memory regions. The win is that all
            # tensors share one underlying cudaMalloc/VMM allocation, so
            # ibv_reg_mr is called once for the range and UCX caches the
            # rkey. Individual tensor registrations just add descriptors.
            logger.info(
                f"VMM registration: {len(tensors)} tensors in single VA range "
                f"at 0x{vmm_range[0]:x}, {vmm_range[1] / 1e9:.2f} GB"
            )
        elif POOL_REG_ENABLED:
            allocations = self._find_cuda_allocations(tensor_descriptors)
        else:
            allocations = None
            logger.info("Pool registration disabled (MX_POOL_REG=0), using per-tensor registration")
        alloc_discovery_time = time.perf_counter() - alloc_discovery_start

        # Phase 2: Register memory with NIXL (ibv_reg_mr kernel calls)
        nixl_reg_start = time.perf_counter()
        if vmm_range is not None:
            # VMM two-phase registration:
            # 1. Register the full VMM range as one tensor -> one ibv_reg_mr
            #    call, populates UCX rcache for the entire VA range.
            # 2. Register individual tensors -> NIXL creates per-tensor
            #    descriptors for prep_xfer_dlist matching, but UCX returns
            #    cached rkeys (no additional ibv_reg_mr kernel calls).
            # This mirrors how __storage views work: one underlying
            # registered region, multiple logical tensors inside it.
            import torch
            vmm_base, vmm_size = vmm_range
            device = torch.device("cuda", self._device_id)
            vmm_storage = torch._C._construct_storage_from_data_pointer(
                vmm_base, device, vmm_size
            )
            vmm_tensor = torch.empty(0, dtype=torch.uint8, device=device)
            vmm_tensor.set_(vmm_storage, 0, (vmm_size,))
            self._vmm_tensor = vmm_tensor  # prevent GC

            range_start = time.perf_counter()
            self._agent.register_memory([vmm_tensor], backends=["UCX"])
            range_time = time.perf_counter() - range_start

            tensor_list = list(tensors.values())
            per_tensor_start = time.perf_counter()
            self._agent.register_memory(tensor_list, backends=["UCX"])
            per_tensor_time = time.perf_counter() - per_tensor_start

            logger.info(
                f"VMM two-phase registration: range={range_time:.3f}s (1 ibv_reg_mr), "
                f"per-tensor={per_tensor_time:.3f}s ({len(tensor_list)} descriptors, "
                f"rcache hits expected)"
            )
            self._alloc_ends = [vmm_base + vmm_size]
            reg_count = len(tensor_list) + 1
        elif allocations:
            alloc_tuples = [
                (base, size, self._device_id, "")
                for base, size in allocations
            ]
            self._agent.register_memory(alloc_tuples, mem_type="cuda", backends=["UCX"])
            self._alloc_ends = sorted(base + size for base, size in allocations)
            reg_count = len(allocations)
        else:
            # Per-tensor registration (baseline mode or all tensors have addr 0)
            tensor_list = list(tensors.values())
            self._agent.register_memory(tensor_list, backends=["UCX"])
            self._alloc_ends = []
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

    @staticmethod
    def _find_cuda_allocations(
        descriptors: list[TensorDescriptor],
    ) -> list[tuple[int, int]]:
        """
        Find unique CUDA allocations backing the tensor descriptors.

        Uses cuMemGetAddressRange to discover the actual cudaMalloc block
        boundaries for each tensor. This is critical for correct NIXL pool
        registration: UCX's rcache produces broken rkeys when a registered
        region spans multiple cudaMalloc allocations, even if they happen
        to be adjacent in virtual address space.

        Args:
            descriptors: List of tensor descriptors

        Returns:
            List of (alloc_base, alloc_size) tuples for unique CUDA allocations
        """
        if not descriptors:
            return []

        import ctypes

        cuda = ctypes.CDLL("libcuda.so")
        seen: dict[int, int] = {}  # alloc_base -> alloc_size

        for desc in descriptors:
            base = ctypes.c_uint64()
            alloc_size = ctypes.c_size_t()
            ret = cuda.cuMemGetAddressRange_v2(
                ctypes.byref(base), ctypes.byref(alloc_size), ctypes.c_uint64(desc.addr)
            )
            if ret != 0:
                raise RuntimeError(
                    f"cuMemGetAddressRange_v2 failed (error {ret}) for tensor "
                    f"at 0x{desc.addr:x}. Is the tensor on a CUDA device?"
                )
            alloc_base = base.value
            if alloc_base not in seen:
                seen[alloc_base] = alloc_size.value

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
        poll_interval = 0.01  # start at 10ms, cap at 100ms
        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= timeout_seconds:
                raise TimeoutError(
                    f"Timed out waiting for remote metadata from "
                    f"{remote_agent_name} at {ip}:{port} after {timeout_seconds:.1f}s"
                )
            if self._agent.check_remote_metadata(remote_agent_name):
                logger.info(
                    f"Remote metadata loaded for {remote_agent_name} "
                    f"({elapsed:.2f}s)"
                )
                return
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 2, 0.1)

    def receive_from_source(
        self,
        source_metadata: bytes,
        source_tensors: list[TensorDescriptor],
        timeout_seconds: float | None = None,
        coalesce_transfers: bool = True,
        remote_agent_name: str | None = None,
        source_alloc_ends: list[int] | None = None,
    ) -> tuple[int, int, float]:
        """
        Receive weights from a remote source via NIXL RDMA.

        Matches source tensors to local tensors by name, optionally coalesces
        contiguous regions to reduce RDMA descriptor overhead, then executes
        the transfer. Both sides must have registered memory pools that cover
        the tensor address ranges (handled by register_tensors).

        Args:
            source_metadata: NIXL metadata from the source agent (unused if remote_agent_name set)
            source_tensors: Tensor descriptors from the source
            timeout_seconds: Maximum time to wait for transfer (None for no timeout)
            coalesce_transfers: If True, coalesce contiguous memory regions (optimization)
            remote_agent_name: If set, use this pre-loaded agent (P2P mode) instead of
                calling add_remote_agent with source_metadata (centralized mode)
            source_alloc_ends: CUDA allocation end addresses from the source, used
                to constrain coalescing so merged regions don't span multiple
                cudaMalloc blocks.

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

        # Phase A: Match source tensors to local tensors by name
        match_start = time.perf_counter()
        remote_descs: list[tuple[int, int, int]] = []
        local_tensor_list: list[torch.Tensor] = []
        total_bytes = 0

        for src_tensor in source_tensors:
            if src_tensor.name not in self._tensors:
                continue
            local_tensor = self._tensors[src_tensor.name]
            remote_descs.append((src_tensor.addr, src_tensor.size, src_tensor.device_id))
            local_tensor_list.append(local_tensor)
            total_bytes += src_tensor.size

        matched_tensors = len(remote_descs)
        match_time = time.perf_counter() - match_start
        if not remote_descs:
            logger.warning("No matching tensors found for transfer")
            return 0, 0, 0.0

        # Phase B: Coalesce contiguous regions (or build raw descriptors)
        coalesce_start = time.perf_counter()
        if coalesce_transfers and COALESCE_ENABLED:
            remote_descs, local_descs, coalesced_count = self._coalesce_transfers(
                remote_descs, local_tensor_list, source_alloc_ends
            )
            if coalesced_count < matched_tensors:
                reduction_pct = (1 - coalesced_count / matched_tensors) * 100
                logger.info(
                    f"Coalesced {matched_tensors} tensors -> {coalesced_count} transfer regions "
                    f"({reduction_pct:.1f}% reduction)"
                )
            use_raw_descriptors = True
        else:
            local_descs = [
                (t.data_ptr(), t.numel() * t.element_size(), self._device_id)
                for t in local_tensor_list
            ]
            coalesced_count = matched_tensors
            use_raw_descriptors = True
        coalesce_time = time.perf_counter() - coalesce_start

        # Log actual wire bytes vs tensor bytes to detect bloat from coalescing gaps
        wire_bytes = sum(d[1] for d in remote_descs)
        if wire_bytes != total_bytes:
            logger.info(
                f"[TIMING] wire bytes mismatch: tensor_bytes={total_bytes / 1e9:.3f} GB, "
                f"wire_bytes={wire_bytes / 1e9:.3f} GB, "
                f"overhead={((wire_bytes - total_bytes) / total_bytes) * 100:.1f}%"
            )
        else:
            logger.info(
                f"[TIMING] wire bytes match tensor bytes: {total_bytes / 1e9:.3f} GB "
                f"({coalesced_count} descs)"
            )

        # Phase C: Prepare transfer descriptors
        prep_start = time.perf_counter()
        src_prepped = self._agent.prep_xfer_dlist(
            agent_name=remote_agent_name,
            xfer_list=remote_descs,
            mem_type="cuda",
            backends=["UCX"],
        )

        dst_prepped = self._agent.prep_xfer_dlist(
            agent_name="",
            xfer_list=local_descs,
            mem_type="cuda",
            backends=["UCX"],
        )

        indices = list(range(len(remote_descs)))

        handle = self._agent.make_prepped_xfer(
            operation="READ",
            local_xfer_side=dst_prepped,
            local_indices=indices,
            remote_xfer_side=src_prepped,
            remote_indices=indices,
            backends=["UCX"],
        )
        prep_time = time.perf_counter() - prep_start

        # Phase D: Execute transfer
        xfer_start = time.perf_counter()
        self._agent.transfer(handle)
        dispatch_time = time.perf_counter() - xfer_start

        # Phase E: Wait for completion
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
        wait_time = time.perf_counter() - start_wait

        # Phase F: CUDA sync (GPUDirect RDMA writes bypass CUDA streams)
        sync_start = time.perf_counter()
        torch.cuda.synchronize(self._device_id)
        sync_time = time.perf_counter() - sync_start

        logger.info(
            f"[TIMING] transfer phases: match={match_time:.3f}s, "
            f"coalesce={coalesce_time:.3f}s, prep={prep_time:.3f}s, "
            f"dispatch={dispatch_time:.3f}s, wait={wait_time:.3f}s, "
            f"cuda_sync={sync_time:.3f}s "
            f"({coalesced_count} descs, {total_bytes / 1e9:.2f} GB)"
        )

        duration = time.perf_counter() - start_time
        bandwidth_gbps = (total_bytes * 8) / (duration * 1e9) if duration > 0 else 0.0

        if coalesced_count < matched_tensors:
            logger.info(
                f"Transfer complete: {matched_tensors} tensors ({coalesced_count} regions), "
                f"{total_bytes / 1e9:.2f} GB in {duration:.2f}s "
                f"({bandwidth_gbps:.1f} Gbps)"
            )
        else:
            logger.info(
                f"Transfer complete: {matched_tensors} tensors, "
                f"{total_bytes / 1e9:.2f} GB in {duration:.2f}s "
                f"({bandwidth_gbps:.1f} Gbps)"
            )

        return total_bytes, matched_tensors, duration

    def _coalesce_transfers(
        self,
        remote_descs: list[tuple[int, int, int]],
        local_tensors: list[torch.Tensor],
        source_alloc_ends: list[int] | None = None,
    ) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]], int]:
        """
        Coalesce contiguous memory regions into larger transfer blocks.

        Model weights are often allocated contiguously in memory. By detecting
        adjacent regions and merging them, we reduce RDMA descriptor overhead
        from 1327 descriptors to potentially dozens.

        NIXL's prep_xfer_dlist accepts both tensor objects AND raw (addr, size, device_id)
        tuples. We use raw tuples for both sides to enable true coalescing.

        Args:
            remote_descs: List of (addr, size, device_id) tuples
            local_tensors: List of local tensors
            source_alloc_ends: Sorted list of CUDA allocation end addresses from source.
                Coalescing will not merge regions that would span multiple source allocations.

        Returns:
            Tuple of (coalesced_remote_descs, coalesced_local_descs, count)
            Note: local_descs are now tuples, not tensors!
        """
        if len(remote_descs) <= 1:
            # Convert single tensor to descriptor
            if local_tensors:
                t = local_tensors[0]
                local_descs = [(t.data_ptr(), t.numel() * t.element_size(), self._device_id)]
            else:
                local_descs = []
            return remote_descs, local_descs, len(remote_descs)

        # Build indexed list with local tensor info
        # (remote_desc, local_addr, local_size)
        indexed = []
        for remote, local in zip(remote_descs, local_tensors, strict=True):
            local_addr = local.data_ptr()
            local_size = local.numel() * local.element_size()
            indexed.append((remote, local_addr, local_size))

        # Sort by remote address to find contiguous regions
        indexed.sort(key=lambda x: x[0][0])

        # Coalesce contiguous regions
        coalesced_remote = []
        coalesced_local = []

        i = 0
        while i < len(indexed):
            # Start a new region
            start_remote_addr = indexed[i][0][0]
            start_local_addr = indexed[i][1]
            current_remote_end = start_remote_addr + indexed[i][0][1]
            current_local_end = start_local_addr + indexed[i][2]
            device_id = indexed[i][0][2]

            # Try to extend by checking next tensors
            j = i + 1
            while j < len(indexed):
                next_remote_addr = indexed[j][0][0]
                next_remote_size = indexed[j][0][1]
                next_local_addr = indexed[j][1]
                next_local_size = indexed[j][2]
                next_device = indexed[j][0][2]

                # Check if both remote AND local are contiguous
                remote_contiguous = (next_remote_addr == current_remote_end)
                local_contiguous = (next_local_addr == current_local_end)
                same_device = (next_device == device_id)

                # Don't merge across the source's CUDA allocation boundaries.
                # UCX's rcache produces broken rkeys when a registered region
                # spans multiple cudaMalloc blocks. Each coalesced transfer
                # region must be coverable by a single source registration.
                crosses_alloc = False
                if remote_contiguous and source_alloc_ends:
                    from bisect import bisect_right
                    idx = bisect_right(source_alloc_ends, start_remote_addr)
                    if idx < len(source_alloc_ends):
                        alloc_end = source_alloc_ends[idx]
                        merged_end = next_remote_addr + next_remote_size
                        if merged_end > alloc_end:
                            crosses_alloc = True

                if remote_contiguous and local_contiguous and same_device and not crosses_alloc:
                    # Extend region
                    current_remote_end = next_remote_addr + next_remote_size
                    current_local_end = next_local_addr + next_local_size
                    j += 1
                else:
                    break

            # Calculate total region sizes
            total_remote_size = current_remote_end - start_remote_addr
            total_local_size = current_local_end - start_local_addr

            # Add coalesced region descriptors
            coalesced_remote.append((start_remote_addr, total_remote_size, device_id))
            coalesced_local.append((start_local_addr, total_local_size, self._device_id))

            i = j

        # Log coalescing results
        original_count = len(remote_descs)
        coalesced_count = len(coalesced_remote)
        if coalesced_count < original_count:
            reduction_pct = 100 * (1 - coalesced_count / original_count)
            logger.info(
                f"Coalesced {original_count} tensors into {coalesced_count} regions "
                f"({reduction_pct:.1f}% reduction in descriptors)"
            )

        return coalesced_remote, coalesced_local, coalesced_count

    def is_healthy(self) -> bool:
        """Check if the NIXL agent is initialized and has registered metadata."""
        return self._agent is not None and len(self._metadata) > 0

    def shutdown(self) -> None:
        """Clean up NIXL resources."""
        self._agent = None
        self._metadata = b""
        self._tensor_descriptors.clear()
        self._tensors.clear()
        logger.info("NixlTransferManager shutdown complete")
