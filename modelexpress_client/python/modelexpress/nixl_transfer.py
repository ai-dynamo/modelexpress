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
import time
from typing import Any

import torch

from .types import TensorDescriptor

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
        self._tensor_descriptors: list[TensorDescriptor] = []
        self._tensors: dict[str, torch.Tensor] = {}

    @property
    def agent_name(self) -> str:
        """Get the NIXL agent name."""
        return self._agent_name

    @property
    def tensor_descriptors(self) -> list[TensorDescriptor]:
        """Get tensor descriptors for registered tensors."""
        return self._tensor_descriptors

    @property
    def alloc_ends(self) -> list[int]:
        """Get CUDA allocation end addresses (for coalescing boundaries)."""
        return getattr(self, '_alloc_ends', [])

    def fetch_remote_and_wait(
        self,
        agent_name: str,
        ip: str,
        port: int,
        timeout: float = 120.0,
    ) -> str:
        """Fetch remote NIXL metadata via native P2P exchange and wait for it.

        Uses NIXL's built-in listen thread protocol: fetch_remote_metadata()
        initiates the fetch, then check_remote_metadata() polls until the
        remote agent's metadata is loaded.

        Args:
            agent_name: The remote NIXL agent's name.
            ip: The remote worker's IP address.
            port: The remote worker's NIXL listen port.
            timeout: Maximum seconds to wait for the metadata to arrive.

        Returns:
            The remote agent name (as reported by the loaded metadata).
        """
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")

        logger.info(
            f"Fetching remote metadata from {ip}:{port} (agent={agent_name})"
        )
        self._agent.fetch_remote_metadata(agent_name, ip, port)

        start = time.perf_counter()
        poll_interval = 0.01  # start at 10ms, cap at 100ms
        while True:
            elapsed = time.perf_counter() - start
            if elapsed >= timeout:
                raise TimeoutError(
                    f"Timed out waiting for remote metadata from "
                    f"{agent_name} at {ip}:{port} after {timeout:.1f}s"
                )
            if self._agent.check_remote_metadata(agent_name):
                logger.info(
                    f"Remote metadata loaded for agent '{agent_name}' "
                    f"in {elapsed:.2f}s"
                )
                return agent_name
            time.sleep(poll_interval)
            poll_interval = min(poll_interval * 2, 0.1)

    def initialize(self) -> None:
        """Initialize the NIXL agent.

        If listen_port was specified, enables the NIXL listen thread for
        native peer-to-peer metadata exchange.
        """
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
                f"NIXL agent '{self._agent_name}' with listen thread on port "
                f"{self._listen_port}, device {self._device_id}"
            )
        else:
            config = nixl_agent_config(backends=["UCX"]) if nixl_agent_config else None
            logger.info(
                f"NIXL agent '{self._agent_name}' created on device {self._device_id}"
            )

        self._agent = NixlAgent(self._agent_name, config)

    def register_tensors(self, tensors: dict[str, torch.Tensor]) -> bytes:
        """
        Register tensors with NIXL for RDMA access.

        Detects contiguous memory pools and registers each pool as a single
        block, minimizing NIXL registration overhead (kernel calls, rkeys,
        metadata blob size). Per-tensor descriptors are always preserved for
        application-level name-based matching during transfers.

        CRITICAL: self._tensors must hold the SAME tensor objects as
        param.data in vLLM. Do NOT call .contiguous() - that would create
        copies and RDMA writes would go to the wrong memory.

        Args:
            tensors: Dictionary of tensor name -> tensor

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

        # Register memory at CUDA allocation boundaries to reduce NIXL
        # registration overhead. Uses cuMemGetAddressRange to discover actual
        # cudaMalloc blocks and registers each as a whole unit via raw tuples.
        # This matches UCX's rcache granularity (one rkey per cudaMalloc block).
        allocations = self._find_cuda_allocations(tensor_descriptors)
        if allocations:
            alloc_tuples = [
                (base, size, self._device_id, "")
                for base, size in allocations
            ]
            self._agent.register_memory(alloc_tuples, mem_type="cuda", backends=["UCX"])
            self._alloc_ends = sorted(base + size for base, size in allocations)
            reg_count = len(allocations)
        else:
            # Fallback: per-tensor registration (e.g. all tensors have addr 0)
            tensor_list = list(tensors.values())
            self._agent.register_memory(tensor_list, backends=["UCX"])
            self._alloc_ends = []
            reg_count = len(tensor_list)

        reduction = (1 - reg_count / len(tensor_descriptors)) * 100 if tensor_descriptors else 0
        logger.info(
            f"Registered {reg_count} regions from {len(tensor_descriptors)} tensors "
            f"({reduction:.1f}% reduction in NIXL registrations)"
        )

        return self._agent.get_agent_metadata()

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
            if desc.addr == 0 or desc.size == 0:
                continue
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

    def receive_from_source(
        self,
        source_metadata: bytes | None,
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
            source_metadata: NIXL metadata from the source agent. Pass None if
                the remote agent was loaded via fetch_remote_and_wait().
            source_tensors: Tensor descriptors from the source
            timeout_seconds: Maximum time to wait for transfer (None for no timeout)
            coalesce_transfers: If True, coalesce contiguous memory regions
            remote_agent_name: If provided, skip add_remote_agent and use this name
                directly (used with NIXL native P2P exchange).

        Returns:
            Tuple of (total_bytes, total_tensors, duration)
        """
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")

        start_time = time.perf_counter()
        torch.cuda.set_device(self._device_id)

        # Load remote agent - either from blob or already fetched via native P2P
        if remote_agent_name is not None:
            logger.info(f"Using pre-fetched remote agent '{remote_agent_name}'")
        elif source_metadata is not None:
            remote_agent_name = self._agent.add_remote_agent(source_metadata)
            logger.info(f"Added remote agent {remote_agent_name}")
        else:
            raise ValueError(
                "Either source_metadata or remote_agent_name must be provided"
            )

        # Match source tensors to local tensors by name
        remote_descs: list[tuple[int, int, int]] = []
        local_tensor_list: list[torch.Tensor] = []
        total_bytes = 0

        for src_tensor in source_tensors:
            if src_tensor.name not in self._tensors:
                continue
            if src_tensor.addr == 0 or src_tensor.size == 0:
                continue
            local_tensor = self._tensors[src_tensor.name]
            if local_tensor.data_ptr() == 0:
                continue
            remote_descs.append((src_tensor.addr, src_tensor.size, src_tensor.device_id))
            local_tensor_list.append(local_tensor)
            total_bytes += src_tensor.size

        matched_tensors = len(remote_descs)
        if not remote_descs:
            logger.warning("No matching tensors found for transfer")
            return 0, 0, 0.0

        # Coalesce contiguous regions to reduce RDMA descriptor overhead.
        # This is a pure transfer-time optimization, independent of how
        # memory pools were registered.
        if coalesce_transfers:
            logger.info(
                f"Coalescing with {len(source_alloc_ends or [])} source alloc boundaries"
            )
            remote_descs, local_descs, coalesced_count = self._coalesce_transfers(
                remote_descs, local_tensor_list, source_alloc_ends
            )
            if coalesced_count < matched_tensors:
                reduction_pct = (1 - coalesced_count / matched_tensors) * 100
                logger.info(
                    f"Coalesced {matched_tensors} tensors -> {coalesced_count} transfer regions "
                    f"({reduction_pct:.1f}% reduction)"
                )
        else:
            local_descs = [
                (t.data_ptr(), t.numel() * t.element_size(), self._device_id)
                for t in local_tensor_list
            ]
            coalesced_count = matched_tensors

        # --- DIAGNOSTIC: test remote descriptors against exchanged metadata ---
        self._diagnose_remote_descs(remote_descs, remote_agent_name, coalesced_count)

        # Build transfer descriptor lists and execute via initialize_xfer.
        remote_xfer = self._agent.get_xfer_descs(remote_descs, mem_type="cuda")
        local_xfer = self._agent.get_xfer_descs(local_descs, mem_type="cuda")
        handle = self._agent.initialize_xfer(
            "READ", local_xfer, remote_xfer, remote_agent_name, backends=["UCX"],
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

        # GPUDirect RDMA writes bypass CUDA streams
        torch.cuda.synchronize(self._device_id)

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

    def _diagnose_remote_descs(
        self,
        remote_descs: list[tuple[int, int, int]],
        remote_agent_name: str,
        desc_count: int,
    ) -> None:
        """Diagnostic: test remote descriptors against exchanged metadata.

        Tests each remote descriptor individually via prep_xfer_dlist to check
        covering index lookups. Also runs a synthetic control test to verify
        NIXL P2P exchange works in this same process.
        """
        logger.info(
            f"[DIAG] Testing {desc_count} remote descs against agent '{remote_agent_name}'"
        )

        # --- Test 1: individual remote descriptor lookups ---
        pass_count = 0
        fail_count = 0
        first_failures: list[str] = []
        for i, (addr, size, dev) in enumerate(remote_descs):
            try:
                xfer_desc = self._agent.get_xfer_descs([(addr, size, dev)], mem_type="cuda")
                prepped = self._agent.prep_xfer_dlist(remote_agent_name, xfer_desc)
                self._agent.release_dlist_handle(prepped)
                pass_count += 1
            except Exception as e:
                fail_count += 1
                if len(first_failures) < 5:
                    first_failures.append(
                        f"  desc[{i}]: addr=0x{addr:x}, size={size}, dev={dev} -> {e}"
                    )
        logger.info(
            f"[DIAG] Remote desc lookup: {pass_count} PASS, {fail_count} FAIL "
            f"(out of {len(remote_descs)})"
        )
        for line in first_failures:
            logger.info(f"[DIAG] {line}")

        # --- Test 2: local self-test (our own registrations) ---
        local_pass = 0
        local_fail = 0
        for name, tensor in list(self._tensors.items())[:5]:
            try:
                desc = [(tensor.data_ptr(), tensor.numel() * tensor.element_size(), self._device_id)]
                xfer_desc = self._agent.get_xfer_descs(desc, mem_type="cuda")
                prepped = self._agent.prep_xfer_dlist("", xfer_desc)
                self._agent.release_dlist_handle(prepped)
                local_pass += 1
            except Exception:
                local_fail += 1
        logger.info(f"[DIAG] Local self-test (first 5): {local_pass} PASS, {local_fail} FAIL")

        # --- Test 3: synthetic control test ---
        # Allocate fresh tensors in this process, register as raw-alloc,
        # create a second agent, exchange metadata, and verify covering index.
        # This proves whether NIXL works in this process environment.
        try:
            import ctypes
            from nixl._api import nixl_agent as NixlAgentCls
            from nixl._api import nixl_agent_config as nixl_config

            NUM_CTRL = 16
            ctrl_tensors = [
                torch.full((1024,), 1.0, dtype=torch.float32, device=f"cuda:{self._device_id}")
                for _ in range(NUM_CTRL)
            ]

            # Find allocations
            cuda = ctypes.CDLL("libcuda.so")
            allocs: dict[int, int] = {}
            for t in ctrl_tensors:
                base = ctypes.c_uint64()
                sz = ctypes.c_size_t()
                ret = cuda.cuMemGetAddressRange_v2(
                    ctypes.byref(base), ctypes.byref(sz), ctypes.c_uint64(t.data_ptr())
                )
                if ret == 0:
                    allocs[base.value] = sz.value

            # Create control agent with raw-alloc registration
            ctrl_config = nixl_config(backends=["UCX"])
            ctrl_agent = NixlAgentCls("diag-control", ctrl_config)
            alloc_tuples = [(b, s, self._device_id, "") for b, s in sorted(allocs.items())]
            ctrl_agent.register_memory(alloc_tuples, mem_type="cuda", backends=["UCX"])

            # Exchange metadata between main agent and control agent
            ctrl_meta = ctrl_agent.get_agent_metadata()
            self._agent.add_remote_agent(ctrl_meta)

            # Test: can main agent resolve control agent's tensor addresses?
            ctrl_pass = 0
            ctrl_fail = 0
            for t in ctrl_tensors:
                try:
                    desc = [(t.data_ptr(), t.numel() * t.element_size(), self._device_id)]
                    xfer_desc = self._agent.get_xfer_descs(desc, mem_type="cuda")
                    prepped = self._agent.prep_xfer_dlist("diag-control", xfer_desc)
                    self._agent.release_dlist_handle(prepped)
                    ctrl_pass += 1
                except Exception:
                    ctrl_fail += 1

            logger.info(
                f"[DIAG] Synthetic control test: {ctrl_pass} PASS, {ctrl_fail} FAIL "
                f"({len(allocs)} allocs from {NUM_CTRL} tensors)"
            )

            # Clean up control agent
            self._agent.remove_remote_agent("diag-control")
            del ctrl_agent
        except Exception as e:
            logger.warning(f"[DIAG] Synthetic control test failed to run: {e}")

        # --- Summary ---
        if fail_count > 0:
            logger.warning(
                f"[DIAG] SUMMARY: {fail_count}/{len(remote_descs)} remote descs FAIL covering "
                f"index after P2P exchange. Control test shows NIXL works in this process. "
                f"The bug is in how model tensors are registered vs how they appear in the manifest."
            )
        else:
            logger.info("[DIAG] SUMMARY: All remote descs pass covering index lookup.")

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

        NIXL's get_xfer_descs accepts both tensor objects AND raw (addr, size, device_id)
        tuples. We use raw tuples for both sides to enable true coalescing.

        Args:
            remote_descs: List of (addr, size, device_id) tuples
            local_tensors: List of local tensors

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

    def shutdown(self) -> None:
        """Clean up NIXL resources."""
        self._agent = None
        self._tensor_descriptors.clear()
        self._tensors.clear()
        logger.info("NixlTransferManager shutdown complete")
