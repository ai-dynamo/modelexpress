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

    def __init__(self, agent_name: str, device_id: int):
        self._agent_name = agent_name
        self._device_id = device_id

        self._agent: Any = None
        self._metadata: bytes = b""
        self._tensor_descriptors: list[TensorDescriptor] = []
        self._tensors: dict[str, torch.Tensor] = {}
        self._registered_regions: list[tuple[int, int]] | None = None

    @property
    def nixl_metadata(self) -> bytes:
        """Get NIXL metadata for this agent."""
        return self._metadata

    @property
    def tensor_descriptors(self) -> list[TensorDescriptor]:
        """Get tensor descriptors for registered tensors."""
        return self._tensor_descriptors

    def initialize(self) -> None:
        """Initialize the NIXL agent."""
        if not NIXL_AVAILABLE:
            raise RuntimeError("NIXL is not available")

        if self._agent is not None:
            return

        torch.cuda.set_device(self._device_id)

        config = nixl_agent_config(backends=["UCX"]) if nixl_agent_config else None
        self._agent = NixlAgent(self._agent_name, config)
        logger.info(f"NIXL agent '{self._agent_name}' created on device {self._device_id}")

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

        return regions

    def receive_from_source(
        self,
        source_metadata: bytes,
        source_tensors: list[TensorDescriptor],
        timeout_seconds: float | None = None,
        coalesce_transfers: bool = True,
    ) -> tuple[int, int, float]:
        """
        Receive weights from a remote source via NIXL RDMA.

        Args:
            source_metadata: NIXL metadata from the source agent
            source_tensors: Tensor descriptors from the source
            timeout_seconds: Maximum time to wait for transfer (None for no timeout)
            coalesce_transfers: If True, coalesce contiguous memory regions (optimization)

        Returns:
            Tuple of (total_bytes, total_tensors, duration)
        """
        if self._agent is None:
            raise RuntimeError("NIXL agent not initialized")

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

    def shutdown(self) -> None:
        """Clean up NIXL resources."""
        self._agent = None
        self._metadata = b""
        self._tensor_descriptors.clear()
        self._tensors.clear()
        logger.info("NixlTransferManager shutdown complete")
