# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NIXL Transfer Manager for GPU-to-GPU weight transfers.

This module provides the NixlTransferManager class that handles all NIXL-related
operations including agent creation, tensor registration, and RDMA transfers.

Each vLLM worker creates its own NixlTransferManager instance to manage
a single NIXL agent for that worker's GPU.

OPTIMIZATION: Pipelined Transfers
---------------------------------
Based on https://le.qun.ch/en/blog/2025/09/07/rl-weight-transfer/ and
https://research.perplexity.ai/articles/weight-transfer-for-rl-post-training-in-under-2-seconds

The pipelined approach splits transfers into batches that can execute concurrently:
- Multiple in-flight RDMA transfers (configurable batch size)
- Non-blocking status checks using polling
- CUDA events to track GPU-side completion
- Overlapping transfer preparation with execution

Environment Variables:
- MX_PIPELINE_ENABLED: Enable pipelined transfers (default: 1)
- MX_PIPELINE_BATCH_SIZE: Number of concurrent transfers (default: 8)
- MX_PIPELINE_POLL_INTERVAL_MS: Polling interval in ms (default: 1)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from .types import TensorDescriptor

logger = logging.getLogger("modelexpress.nixl_transfer")


def _timing_print(msg: str) -> None:
    """
    Print timing info with maximum visibility for k8s logs.
    Uses stderr and explicit flush to ensure capture from worker processes.
    """
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[MX-TIMING] [{timestamp}] {msg}"
    # Write to stderr - more reliably captured by container runtimes
    sys.stderr.write(log_msg + "\n")
    sys.stderr.flush()
    # Also write to stdout
    sys.stdout.write(log_msg + "\n")
    sys.stdout.flush()


class TransferState(Enum):
    """State of a transfer task in the pipeline."""
    PENDING = "pending"           # Not started
    PREPARING = "preparing"       # Building descriptors
    SUBMITTED = "submitted"       # RDMA submitted, waiting
    COMPLETED = "completed"       # Transfer done
    FAILED = "failed"             # Transfer failed


@dataclass
class TransferTask:
    """
    Represents a single transfer task in the pipeline.
    
    Each task handles a batch of tensors/regions to transfer.
    Tasks flow through states: PENDING -> PREPARING -> SUBMITTED -> COMPLETED
    """
    task_id: int
    remote_descs: list[tuple[int, int, int]]  # (addr, size, device_id)
    local_descs: list  # Either tensors or (addr, size, device_id) tuples
    use_raw_descriptors: bool
    
    # State tracking
    state: TransferState = TransferState.PENDING
    handle: Any = None
    start_time: float = 0.0
    end_time: float = 0.0
    
    # Size tracking
    total_bytes: int = 0
    tensor_count: int = 0
    
    # Error info
    error: str | None = None
    
    def __post_init__(self):
        self.total_bytes = sum(d[1] for d in self.remote_descs)
        self.tensor_count = len(self.remote_descs)


class PipelineStats:
    """Statistics for pipelined transfers."""
    
    def __init__(self):
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.total_bytes = 0
        self.total_time = 0.0
        self.max_concurrent = 0
        self.poll_count = 0
        
    def log_summary(self, device_id: int):
        """Log pipeline performance summary."""
        if self.total_time > 0:
            bandwidth_gbps = (self.total_bytes * 8) / (self.total_time * 1e9)
        else:
            bandwidth_gbps = 0
            
        summary_msg = (
            f"[Worker {device_id}] [Pipeline Stats] "
            f"Tasks: {self.completed_tasks}/{self.total_tasks} completed, "
            f"{self.failed_tasks} failed | "
            f"Data: {self.total_bytes / 1e9:.2f} GB | "
            f"Time: {self.total_time:.3f}s | "
            f"Bandwidth: {bandwidth_gbps:.1f} Gbps | "
            f"Max concurrent: {self.max_concurrent} | "
            f"Polls: {self.poll_count}"
        )
        logger.info(summary_msg)
        
        # EXPLICIT TIMING OUTPUT - guaranteed to be captured
        _timing_print(f"Worker {device_id}: PIPELINE - {self.total_bytes/1e9:.2f} GB in {self.total_time:.3f}s = {bandwidth_gbps:.1f} Gbps ({self.completed_tasks} tasks, max_concurrent={self.max_concurrent})")

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

    def __init__(self, agent_name: str, device_id: int):
        self._agent_name = agent_name
        self._device_id = device_id

        self._agent: Any = None
        self._metadata: bytes = b""
        self._tensor_descriptors: list[TensorDescriptor] = []
        self._tensors: dict[str, torch.Tensor] = {}
        self._registered_regions: list[tuple[int, int]] | None = None
        self._current_remote_agent: str = ""  # For pipelined transfers

    @property
    def nixl_metadata(self) -> bytes:
        """Get NIXL metadata for this agent."""
        return self._metadata

    @property
    def tensor_descriptors(self) -> list[TensorDescriptor]:
        """Get tensor descriptors for registered tensors."""
        return self._tensor_descriptors

    def initialize(self) -> None:
        """Initialize the NIXL agent with optimal NIC binding for GPUDirect RDMA."""
        if not NIXL_AVAILABLE:
            raise RuntimeError("NIXL is not available")

        if self._agent is not None:
            return

        torch.cuda.set_device(self._device_id)
        
        # OPTIMIZATION: Bind each GPU worker to its optimal NIC based on PCIe topology
        # This mapping is based on nvidia-smi topo -m showing PIX (direct PCIe) connections:
        # GPU0-3 are on NUMA 0, GPU4-7 are on NUMA 1
        # GPU0->NIC4(mlx5_4), GPU1->NIC5(mlx5_5), GPU2->NIC6(mlx5_6), GPU3->NIC7(mlx5_7)
        # GPU4->NIC0(mlx5_0), GPU5->NIC1(mlx5_1), GPU6->NIC2(mlx5_2), GPU7->NIC3(mlx5_3)
        gpu_to_nic = {
            0: "mlx5_4", 1: "mlx5_5", 2: "mlx5_6", 3: "mlx5_7",
            4: "mlx5_0", 5: "mlx5_1", 6: "mlx5_2", 7: "mlx5_3",
        }
        
        # Set UCX environment variables for optimal RDMA performance
        nic_device = gpu_to_nic.get(self._device_id, f"mlx5_{self._device_id}")
        
        # Check if binding is enabled (can be disabled for debugging)
        if os.environ.get("MX_NIC_BINDING", "1") == "1":
            os.environ["UCX_NET_DEVICES"] = f"{nic_device}:1"
            _timing_print(f"Worker {self._device_id}: NIC binding GPU{self._device_id} -> {nic_device}")
        
        # Enable GPUDirect RDMA for zero-copy GPU-to-GPU transfers
        if os.environ.get("UCX_IB_GPU_DIRECT_RDMA") is None:
            os.environ["UCX_IB_GPU_DIRECT_RDMA"] = "yes"
        
        # Optimize for large transfers
        if os.environ.get("UCX_RC_MAX_RD_ATOMIC") is None:
            os.environ["UCX_RC_MAX_RD_ATOMIC"] = "16"

        config = nixl_agent_config(backends=["UCX"]) if nixl_agent_config else None
        self._agent = NixlAgent(self._agent_name, config)
        logger.info(f"NIXL agent '{self._agent_name}' created on device {self._device_id} with NIC {nic_device}")

    def register_tensors(self, tensors: dict[str, torch.Tensor]) -> bytes:
        """
        Register tensors with NIXL for RDMA access.

        CRITICAL: We must ensure self._tensors contains the SAME tensor objects
        that are registered with NIXL, so receive_from_source uses correct memory.

        If MX_CONTIGUOUS_REG=1, detects and registers contiguous memory regions
        as single blocks, reducing descriptor overhead significantly.

        Args:
            tensors: Dictionary of tensor name -> tensor

        Returns:
            NIXL metadata bytes for this agent
        """
        import os
        
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")

        # CRITICAL: Do NOT call .contiguous() here!
        # The tensors must be the exact same objects as param.data in vLLM,
        # otherwise RDMA writes to copies and vLLM uses originals = garbage.
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
        
        # Check if contiguous region registration is enabled
        use_contiguous = os.environ.get("MX_CONTIGUOUS_REG", "0") == "1"
        
        if use_contiguous:
            # Register contiguous memory regions as single blocks
            regions = self._find_contiguous_regions(tensor_descriptors)
            logger.info(
                f"[Contiguous Registration] Found {len(regions)} contiguous regions "
                f"from {len(tensor_descriptors)} tensors "
                f"({(1 - len(regions)/len(tensor_descriptors))*100:.1f}% reduction)"
            )
            
            # Register regions using raw address tuples
            # Format: (addr, size, device_id, mem_type) - 4-tuple required by NIXL API
            region_tuples = [(r[0], r[1], self._device_id, "cuda") for r in regions]
            self._agent.register_memory(region_tuples, mem_type="cuda", backends=["UCX"])
            self._registered_regions = regions
            logger.info(f"Registered {len(regions)} contiguous regions with NIXL")
            # Debug: Log first few registered region addresses
            if len(regions) > 0:
                logger.info(f"[Contiguous Registration] DEBUG - First 3 regions: {[(hex(r[0]), r[1]) for r in regions[:3]]}")
        else:
            # Traditional: register individual tensors
            tensor_list = list(tensors.values())
            self._agent.register_memory(tensor_list, backends=["UCX"])
            self._registered_regions = None
            logger.info(f"Registered {len(tensor_list)} individual tensors with NIXL")
        
        self._metadata = self._agent.get_agent_metadata()
        return self._metadata

    def get_registered_descriptors(self) -> list[TensorDescriptor]:
        """
        Get the descriptors that were actually registered with NIXL.
        
        When MX_CONTIGUOUS_REG=1, returns contiguous region descriptors.
        Otherwise, returns individual tensor descriptors.
        
        This is important for publishing to the server - we must publish
        what was actually registered, not the original tensors.
        """
        if self._registered_regions is not None:
            # Return region descriptors with synthetic names
            return [
                TensorDescriptor(
                    name=f"__region_{i}__",
                    addr=addr,
                    size=size,
                    device_id=self._device_id,
                    dtype="contiguous_region",
                )
                for i, (addr, size) in enumerate(self._registered_regions)
            ]
        else:
            # Return original tensor descriptors
            return self._tensor_descriptors

    def _find_contiguous_regions(
        self, descriptors: list[TensorDescriptor]
    ) -> list[tuple[int, int]]:
        """
        Find contiguous memory regions from tensor descriptors.
        
        Sorts tensors by address and merges adjacent ones into larger regions.
        This reduces the number of NIXL registrations significantly.
        
        CRITICAL: After finding regions, we sort by size (descending) to ensure
        deterministic ordering across source and target. Without this, regions
        would be ordered by local memory addresses which differ between nodes.
        
        Args:
            descriptors: List of tensor descriptors
            
        Returns:
            List of (start_addr, total_size) tuples for contiguous regions
        """
        if not descriptors:
            return []
        
        # Sort by address
        sorted_descs = sorted(descriptors, key=lambda d: d.addr)
        
        regions = []
        current_start = sorted_descs[0].addr
        current_end = current_start + sorted_descs[0].size
        
        for desc in sorted_descs[1:]:
            if desc.addr == current_end:
                # Contiguous - extend region
                current_end = desc.addr + desc.size
            else:
                # Gap - save current region and start new one
                regions.append((current_start, current_end - current_start))
                current_start = desc.addr
                current_end = desc.addr + desc.size
        
        # Don't forget the last region
        regions.append((current_start, current_end - current_start))
        
        # CRITICAL: Sort regions by size (descending) for deterministic ordering
        # across different nodes. Without this, regions are ordered by local
        # memory addresses which differ between source and target.
        # Sort by (-size, original_index) to be stable and deterministic.
        indexed_regions = [(size, i, addr) for i, (addr, size) in enumerate(regions)]
        indexed_regions.sort(key=lambda x: (-x[0], x[1]))  # Descending size, then original index
        
        # Rebuild regions in deterministic order
        regions = [(addr, size) for size, _, addr in indexed_regions]
        
        logger.info(f"[Contiguous Registration] Sorted {len(regions)} regions by size (deterministic ordering)")
        
        return regions

    def receive_from_source(
        self,
        source_metadata: bytes,
        source_tensors: list[TensorDescriptor],
        timeout_seconds: float | None = None,
        coalesce_transfers: bool = True,
        use_pipeline: bool | None = None,
    ) -> tuple[int, int, float]:
        """
        Receive weights from a remote source via NIXL RDMA.

        Args:
            source_metadata: NIXL metadata from the source agent
            source_tensors: Tensor descriptors from the source
            timeout_seconds: Maximum time to wait for transfer (None for no timeout)
            coalesce_transfers: If True, coalesce contiguous memory regions (optimization)
            use_pipeline: If True, use pipelined transfers. Default from MX_PIPELINE_ENABLED env.

        Returns:
            Tuple of (total_bytes, total_tensors, duration)
        """
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")
        
        # Check if pipelining is enabled (default: enabled)
        if use_pipeline is None:
            use_pipeline = os.environ.get("MX_PIPELINE_ENABLED", "1") == "1"
        
        if use_pipeline:
            logger.info(f"[Worker {self._device_id}] Using pipelined transfers (MX_PIPELINE_ENABLED=1)")
            return self.receive_from_source_pipelined(
                source_metadata=source_metadata,
                source_tensors=source_tensors,
                timeout_seconds=timeout_seconds,
            )
        
        logger.info(f"[Worker {self._device_id}] Using sequential transfers (MX_PIPELINE_ENABLED=0)")

        start_time = time.perf_counter()
        torch.cuda.set_device(self._device_id)

        # Add remote agent
        remote_agent_name = self._agent.add_remote_agent(source_metadata)
        logger.info(f"Added remote agent {remote_agent_name}")

        # Check if source is sending region descriptors (MX_CONTIGUOUS_REG=1 on source)
        is_region_transfer = (
            len(source_tensors) > 0 and 
            source_tensors[0].name.startswith("__region_")
        )
        
        if is_region_transfer:
            # REGION-BASED TRANSFER: Source registered contiguous regions
            # We must also have registered regions and match by index
            if self._registered_regions is None:
                logger.error("Source sent region descriptors but we didn't register regions!")
                logger.error("Set MX_CONTIGUOUS_REG=1 on target to enable region transfer")
                raise RuntimeError("Region transfer mismatch: target must also use MX_CONTIGUOUS_REG=1")
            
            logger.info(f"Region-based transfer: {len(source_tensors)} source regions -> {len(self._registered_regions)} local regions")
            
            # Validate region counts match
            if len(source_tensors) != len(self._registered_regions):
                logger.warning(
                    f"Region count mismatch: source has {len(source_tensors)}, "
                    f"local has {len(self._registered_regions)}. Proceeding with min."
                )
            
            # Build transfer lists by region index
            remote_descs = []
            local_descs = []  # Will be (addr, size, device_id) tuples
            total_bytes = 0
            matched_count = min(len(source_tensors), len(self._registered_regions))
            
            for i in range(matched_count):
                src_region = source_tensors[i]
                local_addr, local_size = self._registered_regions[i]
                
                # Verify sizes match (regions should be same size)
                if src_region.size != local_size:
                    logger.warning(f"Region {i} size mismatch: source={src_region.size}, local={local_size}")
                
                remote_descs.append((src_region.addr, src_region.size, src_region.device_id))
                local_descs.append((local_addr, local_size, self._device_id))
                total_bytes += src_region.size
            
            matched_tensors = matched_count
            use_raw_descriptors = True
            coalesced_count = matched_count
            
            logger.info(f"[Region Transfer] Matched {matched_count} regions, {total_bytes / 1e9:.2f} GB")
            
            # Debug: Log first few region addresses for comparison
            if matched_count > 0:
                logger.info(f"[Region Transfer] DEBUG - First 3 source regions: {[(hex(r[0]), r[1]) for r in remote_descs[:3]]}")
                logger.info(f"[Region Transfer] DEBUG - First 3 local regions: {[(hex(r[0]), r[1]) for r in local_descs[:3]]}")
            
        else:
            # TENSOR-BASED TRANSFER: Match by tensor name (baseline)
            remote_descs = []
            local_tensor_list = []
            total_bytes = 0
            matched_tensors = 0

            for src_tensor in source_tensors:
                if src_tensor.name not in self._tensors:
                    continue
                local_tensor = self._tensors[src_tensor.name]
                remote_descs.append((src_tensor.addr, src_tensor.size, src_tensor.device_id))
                local_tensor_list.append(local_tensor)
                total_bytes += src_tensor.size
                matched_tensors += 1

            if not remote_descs:
                logger.warning("No matching tensors found for transfer")
                return 0, 0, 0.0
            
            # For tensor-based, we might still coalesce if enabled
            local_descs = local_tensor_list
            use_raw_descriptors = False
            coalesced_count = matched_tensors

        # OPTIMIZATION: Coalesce contiguous memory regions to reduce descriptor overhead
        # Skip if we're doing region-based transfer (already optimized at registration time)
        if is_region_transfer:
            # Region transfer already has optimal descriptors, skip coalescing
            logger.info(f"[Region Transfer] Skipping coalesce - already optimized with {coalesced_count} regions")
        elif coalesce_transfers:
            logger.info(f"[Coalesce] Starting coalescing of {len(remote_descs)} descriptors...")
            remote_descs, local_descs, coalesced_count = self._coalesce_transfers(
                remote_descs, local_tensor_list
            )
            reduction_pct = (1 - coalesced_count / matched_tensors) * 100 if matched_tensors > 0 else 0
            logger.info(
                f"[Coalesce] Reduced {matched_tensors} descriptors -> {coalesced_count} regions "
                f"({reduction_pct:.1f}% reduction)"
            )
            # local_descs are now (addr, size, device_id) tuples, not tensors
            use_raw_descriptors = True
        else:
            logger.info(f"[Coalesce] DISABLED - transferring {matched_tensors} individual tensors")
            # Fall back to tensor list
            local_descs = local_tensor_list
            use_raw_descriptors = False
            coalesced_count = matched_tensors

        # Prepare transfer
        src_prepped = self._agent.prep_xfer_dlist(
            agent_name=remote_agent_name,
            xfer_list=remote_descs,
            mem_type="cuda",
            backends=["UCX"],
        )
        
        if use_raw_descriptors:
            # Use raw address descriptors for coalesced regions
            dst_prepped = self._agent.prep_xfer_dlist(
                agent_name="",
                xfer_list=local_descs,
                mem_type="cuda",
                backends=["UCX"],
            )
        else:
            # Use tensor objects
            dst_prepped = self._agent.prep_xfer_dlist(
                agent_name="",
                xfer_list=local_descs,
                mem_type="cuda",
                backends=["UCX"],
            )

        indices = list(range(len(remote_descs)))

        # Execute transfer
        handle = self._agent.make_prepped_xfer(
            operation="READ",
            local_xfer_side=dst_prepped,
            local_indices=indices,
            remote_xfer_side=src_prepped,
            remote_indices=indices,
            backends=["UCX"],
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

        # CRITICAL: Synchronize CUDA to ensure RDMA writes are visible
        # GPUDirect RDMA writes bypass CUDA streams, so we must sync
        torch.cuda.synchronize(self._device_id)

        duration = time.perf_counter() - start_time
        bandwidth_gbps = (total_bytes * 8) / (duration * 1e9) if duration > 0 else 0.0

        if coalesce_transfers and coalesced_count < matched_tensors:
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
        for remote, local in zip(remote_descs, local_tensors):
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
                # Strict check: no gaps allowed for RDMA correctness
                remote_contiguous = (next_remote_addr == current_remote_end)
                local_contiguous = (next_local_addr == current_local_end)
                same_device = (next_device == device_id)
                
                if remote_contiguous and local_contiguous and same_device:
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

    def receive_from_source_pipelined(
        self,
        source_metadata: bytes,
        source_tensors: list[TensorDescriptor],
        timeout_seconds: float | None = None,
        batch_size: int | None = None,
        poll_interval_ms: float | None = None,
    ) -> tuple[int, int, float]:
        """
        Receive weights from a remote source using pipelined RDMA transfers.
        
        This implementation splits transfers into batches that can execute concurrently,
        with non-blocking status checks. This is based on the optimization strategies from:
        - https://le.qun.ch/en/blog/2025/09/07/rl-weight-transfer/
        - https://research.perplexity.ai/articles/weight-transfer-for-rl-post-training-in-under-2-seconds
        
        Pipeline stages:
        1. PENDING: Task created, waiting to start
        2. PREPARING: Building transfer descriptors
        3. SUBMITTED: RDMA transfer in flight
        4. COMPLETED: Transfer finished
        
        Args:
            source_metadata: NIXL metadata from the source agent
            source_tensors: Tensor descriptors from the source
            timeout_seconds: Maximum time to wait for all transfers
            batch_size: Number of concurrent transfers (default from MX_PIPELINE_BATCH_SIZE)
            poll_interval_ms: Polling interval in ms (default from MX_PIPELINE_POLL_INTERVAL_MS)
        
        Returns:
            Tuple of (total_bytes, total_tensors, duration)
        """
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")
        
        # Configuration from environment or parameters
        batch_size = batch_size or int(os.environ.get("MX_PIPELINE_BATCH_SIZE", "8"))
        poll_interval_ms = poll_interval_ms or float(os.environ.get("MX_PIPELINE_POLL_INTERVAL_MS", "1"))
        poll_interval = poll_interval_ms / 1000.0
        
        # EXPLICIT TIMING OUTPUT at start
        _timing_print(f"Worker {self._device_id}: PIPELINED TRANSFER START - batch_size={batch_size}, poll_interval={poll_interval_ms}ms")
        
        start_time = time.perf_counter()
        torch.cuda.set_device(self._device_id)
        
        # Add remote agent
        remote_agent_name = self._agent.add_remote_agent(source_metadata)
        logger.info(f"[Pipeline] Added remote agent {remote_agent_name}")
        
        # Build matched tensor pairs
        is_region_transfer = (
            len(source_tensors) > 0 and 
            source_tensors[0].name.startswith("__region_")
        )
        
        if is_region_transfer:
            # Region-based transfer
            if self._registered_regions is None:
                raise RuntimeError("Region transfer mismatch: target must also use MX_CONTIGUOUS_REG=1")
            
            matched_pairs = []
            matched_count = min(len(source_tensors), len(self._registered_regions))
            
            for i in range(matched_count):
                src_region = source_tensors[i]
                local_addr, local_size = self._registered_regions[i]
                matched_pairs.append((
                    (src_region.addr, src_region.size, src_region.device_id),
                    (local_addr, local_size, self._device_id),
                    True  # use_raw_descriptors
                ))
            
            logger.info(f"[Pipeline] Region-based: {matched_count} regions")
            
        else:
            # Tensor-based transfer - match by name
            matched_pairs = []
            
            for src_tensor in source_tensors:
                if src_tensor.name not in self._tensors:
                    continue
                local_tensor = self._tensors[src_tensor.name]
                matched_pairs.append((
                    (src_tensor.addr, src_tensor.size, src_tensor.device_id),
                    local_tensor,
                    False  # use tensor objects
                ))
            
            if not matched_pairs:
                logger.warning("[Pipeline] No matching tensors found")
                return 0, 0, 0.0
            
            logger.info(f"[Pipeline] Tensor-based: {len(matched_pairs)} matched")
        
        # Create transfer tasks - split into batches
        tasks = self._create_transfer_tasks(matched_pairs, batch_size)
        stats = PipelineStats()
        stats.total_tasks = len(tasks)
        
        logger.info(
            f"[Pipeline] Created {len(tasks)} tasks from {len(matched_pairs)} transfers "
            f"(batch_size={batch_size}, poll_interval={poll_interval_ms}ms)"
        )
        
        # Pipeline queues
        pending: deque[TransferTask] = deque(tasks)
        in_flight: list[TransferTask] = []
        completed: list[TransferTask] = []
        failed: list[TransferTask] = []
        
        # Prep descriptors (can be reused)
        all_remote_descs = [p[0] for p in matched_pairs]
        all_local_descs = [p[1] for p in matched_pairs]
        use_raw = matched_pairs[0][2] if matched_pairs else True
        
        # Store remote agent name for task submission
        self._current_remote_agent = remote_agent_name
        
        logger.info(f"[Pipeline] Ready to transfer {len(all_remote_descs)} items")
        
        # Main pipeline loop
        timeout_start = time.perf_counter()
        
        while pending or in_flight:
            # Check timeout
            if timeout_seconds is not None:
                if time.perf_counter() - timeout_start >= timeout_seconds:
                    # Clean up in-flight transfers
                    for task in in_flight:
                        if task.handle is not None:
                            try:
                                self._agent.release_xfer_handle(task.handle)
                            except Exception:
                                pass
                    raise TimeoutError(f"Pipeline timed out after {timeout_seconds}s")
            
            # Submit new tasks up to batch_size
            while pending and len(in_flight) < batch_size:
                task = pending.popleft()
                self._submit_task(task)
                in_flight.append(task)
                stats.max_concurrent = max(stats.max_concurrent, len(in_flight))
            
            # Poll in-flight tasks for completion
            stats.poll_count += 1
            still_in_flight = []
            
            for task in in_flight:
                if task.state == TransferState.SUBMITTED:
                    try:
                        status = self._agent.check_xfer_state(task.handle)
                        
                        if status in ("DONE", "SUCCESS"):
                            task.state = TransferState.COMPLETED
                            task.end_time = time.perf_counter()
                            self._agent.release_xfer_handle(task.handle)
                            completed.append(task)
                            stats.completed_tasks += 1
                            stats.total_bytes += task.total_bytes
                            
                        elif status in ("ERR", "ERROR", "FAIL"):
                            task.state = TransferState.FAILED
                            task.error = f"Transfer failed with status {status}"
                            task.end_time = time.perf_counter()
                            self._agent.release_xfer_handle(task.handle)
                            failed.append(task)
                            stats.failed_tasks += 1
                            logger.warning(f"[Pipeline] Task {task.task_id} failed: {status}")
                            
                        else:
                            # Still in progress
                            still_in_flight.append(task)
                            
                    except Exception as e:
                        task.state = TransferState.FAILED
                        task.error = str(e)
                        task.end_time = time.perf_counter()
                        failed.append(task)
                        stats.failed_tasks += 1
                        logger.warning(f"[Pipeline] Task {task.task_id} exception: {e}")
                else:
                    still_in_flight.append(task)
            
            in_flight = still_in_flight
            
            # Brief sleep to avoid busy-spinning
            if in_flight:
                time.sleep(poll_interval)
        
        # CRITICAL: Synchronize CUDA to ensure RDMA writes are visible
        torch.cuda.synchronize(self._device_id)
        
        duration = time.perf_counter() - start_time
        stats.total_time = duration
        
        # Log summary
        stats.log_summary(self._device_id)
        
        total_bytes = sum(t.total_bytes for t in completed)
        total_tensors = sum(t.tensor_count for t in completed)
        
        if failed:
            logger.error(f"[Pipeline] {len(failed)} tasks failed!")
            for task in failed[:3]:  # Log first 3 failures
                logger.error(f"[Pipeline]   Task {task.task_id}: {task.error}")
        
        # EXPLICIT TIMING OUTPUT at function return
        bandwidth_gbps = (total_bytes * 8) / (duration * 1e9) if duration > 0 else 0
        _timing_print(f"Worker {self._device_id}: PIPELINED TRANSFER DONE - {total_bytes/1e9:.2f} GB, {total_tensors} tensors, {duration:.3f}s, {bandwidth_gbps:.1f} Gbps")
        
        return total_bytes, total_tensors, duration
    
    def _create_transfer_tasks(
        self,
        matched_pairs: list[tuple],
        batch_size: int,
    ) -> list[TransferTask]:
        """
        Create transfer tasks by batching matched pairs.
        
        Each task handles a subset of the transfers, allowing concurrent execution.
        """
        tasks = []
        
        # Calculate how many items per batch
        # For optimal pipelining, we want multiple batches
        total_items = len(matched_pairs)
        items_per_batch = max(1, total_items // batch_size)
        
        # If we have fewer items than batch_size, just use single-item batches
        if total_items <= batch_size:
            items_per_batch = 1
        
        task_id = 0
        for i in range(0, total_items, items_per_batch):
            batch = matched_pairs[i:i + items_per_batch]
            
            remote_descs = [p[0] for p in batch]
            local_descs = [p[1] for p in batch]
            use_raw = batch[0][2] if batch else True
            
            # For tensor objects, convert to raw descriptors
            if not use_raw:
                local_descs = [
                    (t.data_ptr(), t.numel() * t.element_size(), self._device_id)
                    for t in local_descs
                ]
                use_raw = True
            
            task = TransferTask(
                task_id=task_id,
                remote_descs=remote_descs,
                local_descs=local_descs,
                use_raw_descriptors=use_raw,
            )
            tasks.append(task)
            task_id += 1
        
        return tasks
    
    def _submit_task(self, task: TransferTask) -> None:
        """
        Submit a transfer task for execution.
        
        This prepares and submits the RDMA transfer for this task's batch.
        Uses the remote agent name stored during receive_from_source_pipelined.
        """
        task.state = TransferState.PREPARING
        task.start_time = time.perf_counter()
        
        try:
            batch_size = len(task.remote_descs)
            
            # Prepare descriptors for this specific batch
            batch_src_prepped = self._agent.prep_xfer_dlist(
                agent_name=self._current_remote_agent,
                xfer_list=task.remote_descs,
                mem_type="cuda",
                backends=["UCX"],
            )
            
            batch_dst_prepped = self._agent.prep_xfer_dlist(
                agent_name="",
                xfer_list=task.local_descs,
                mem_type="cuda",
                backends=["UCX"],
            )
            
            indices = list(range(batch_size))
            
            # Submit transfer
            task.handle = self._agent.make_prepped_xfer(
                operation="READ",
                local_xfer_side=batch_dst_prepped,
                local_indices=indices,
                remote_xfer_side=batch_src_prepped,
                remote_indices=indices,
                backends=["UCX"],
            )
            self._agent.transfer(task.handle)
            task.state = TransferState.SUBMITTED
            
        except Exception as e:
            task.state = TransferState.FAILED
            task.error = str(e)
            task.end_time = time.perf_counter()
            logger.error(f"[Pipeline] Failed to submit task {task.task_id}: {e}")

    def shutdown(self) -> None:
        """Clean up NIXL resources."""
        self._agent = None
        self._metadata = b""
        self._tensor_descriptors.clear()
        self._tensors.clear()
        logger.info("NixlTransferManager shutdown complete")
