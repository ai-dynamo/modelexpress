# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NIXL GDS Transfer Manager for direct file-to-GPU weight loading.

This module provides the GdsTransferManager class that wraps NIXL's GDS
(GPUDirect Storage) backend for zero-copy transfers from storage to GPU memory.

Unlike the UCX-based NixlTransferManager (for remote P2P transfers), this
manager operates locally: it reads files directly into GPU memory, bypassing
CPU bounce buffers entirely.

GDS constraints handled here:
- File offsets and sizes must be 4KB-aligned
- Each I/O request must fit within cuFile's per_buffer_cache_size

Framework-agnostic — depends only on nixl and torch.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import torch

logger = logging.getLogger("modelexpress.gds_transfer")

NIXL_AVAILABLE = False
NixlAgent = None
NixlAgentConfig = None
try:
    from nixl._api import nixl_agent as NixlAgent
    from nixl._api import nixl_agent_config as NixlAgentConfig
    NIXL_AVAILABLE = True
except ImportError:
    pass


def is_gds_available() -> bool:
    """Check if NIXL (required for GDS) is available."""
    return NIXL_AVAILABLE


# GDS requires 4KB-aligned file offsets and transfer sizes.
GDS_ALIGNMENT = 4096

# GPU page size.  In native GDS mode, cuFile adds the GPU buffer's page
# offset to the requested I/O size.  The total must fit within the
# per_buffer_cache_size, so we subtract one GPU page from the max chunk.
GPU_PAGE_SIZE = 65536

# Default max chunk size per GDS I/O (must be <= cuFile per_buffer_cache_size).
_DEFAULT_MAX_CHUNK = 16 * 1024 * 1024  # 16 MB


def _read_max_chunk_from_cufile() -> int:
    """Read max safe chunk size from cufile.json.

    Supports both the new slab-based config (``gpu_bounce_buffer_slab_config``)
    and the legacy ``per_buffer_cache_size_kb`` parameter.

    For slab config, the max usable slab for nvidia-fs is 16MB (slabs >16MB
    are for P2P mode only).
    """
    cufile_path = os.environ.get("CUFILE_ENV_PATH_JSON", "/etc/cufile.json")
    max_nvfs_slab_kb = 16384  # nvidia-fs only supports slabs <= 16MB
    try:
        with open(cufile_path) as f:
            text = f.read()
        # Strip C-style // comments but not inside quoted strings.
        text = re.sub(
            r'("(?:[^"\\]|\\.)*")|//[^\n]*',
            lambda m: m.group(1) if m.group(1) else "",
            text,
        )
        config = json.loads(text)
        props = config.get("properties", {})

        # New slab-based config takes priority
        slab_cfg = props.get("gpu_bounce_buffer_slab_config")
        if slab_cfg:
            slab_sizes = slab_cfg.get("slab_size_kb", [])
            slab_counts = slab_cfg.get("slab_count", [])
            io_bs = props.get("io_batchsize", 128)
            # Batch API needs io_batchsize shadow buffers.  Find the largest
            # slab whose count >= io_batchsize (that's what batch uses).
            batch_slab_kb = None
            for sz, cnt in zip(slab_sizes, slab_counts):
                if cnt >= io_bs and sz <= max_nvfs_slab_kb:
                    batch_slab_kb = sz
            if batch_slab_kb is None:
                batch_slab_kb = min(slab_sizes) if slab_sizes else 1024
            effective_kb = batch_slab_kb - (GPU_PAGE_SIZE // 1024)
            logger.info(
                "cufile.json slab config: slabs=%s counts=%s io_batchsize=%d "
                "-> batch_slab=%dKB, effective max_chunk=%dKB",
                slab_sizes, slab_counts, io_bs, batch_slab_kb, effective_kb,
            )
            return effective_kb * 1024

        # Legacy config
        per_buffer_kb = props.get("per_buffer_cache_size_kb", 1024)
        max_device_kb = props.get("max_device_cache_size_kb", 131072)
        io_batchsize = props.get("io_batchsize", 128)
        shadow_kb = max_device_kb // io_batchsize if io_batchsize > 0 else per_buffer_kb
        effective_kb = min(per_buffer_kb, shadow_kb) - (GPU_PAGE_SIZE // 1024)
        logger.info(
            "cufile.json legacy config: per_buffer=%dKB, max_device=%dKB, "
            "io_batchsize=%d -> effective max_chunk=%dKB (after GPU page reserve)",
            per_buffer_kb, max_device_kb, io_batchsize, effective_kb,
        )
        return effective_kb * 1024
    except Exception as e:
        logger.warning("Failed to parse %s: %s, using default %d KB",
                       cufile_path, e, _DEFAULT_MAX_CHUNK // 1024)
        return _DEFAULT_MAX_CHUNK


class GdsTransferManager:
    """
    Manages NIXL GDS backend for direct file-to-GPU transfers.

    Transfers are per-tensor with automatic 4KB alignment and chunking
    (following the proven pattern from mini-gds).

    The target GPU is taken from ``torch.cuda.current_device()`` at
    ``initialize()`` time.
    """

    def __init__(self, agent_name: str):
        self._agent_name = agent_name
        self._device_id: int | None = None
        self._agent: Any = None
        self._max_chunk_size = _read_max_chunk_from_cufile()

    @property
    def agent_name(self) -> str:
        return self._agent_name

    def initialize(self) -> None:
        """Initialize the NIXL agent with GDS backend."""
        if not NIXL_AVAILABLE:
            raise RuntimeError(
                "NIXL is not available. Install with: pip install nixl[cu12]"
            )
        if self._agent is not None:
            return

        self._device_id = torch.cuda.current_device()

        # Create agent with multithreaded GDS backend
        thread_count = int(os.environ.get("MX_GDS_THREADS", "8"))
        config = NixlAgentConfig(backends=["GDS_MT"], num_threads=thread_count)
        self._agent = NixlAgent(self._agent_name, config)

        logger.info(
            "GDS_MT agent '%s' created on device %d (threads=%d, max_chunk=%dKB)",
            self._agent_name, self._device_id, thread_count,
            self._max_chunk_size // 1024,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def batch_load_file(
        self,
        fd: int,
        file_size: int,
        tensor_list: list[tuple[int, int]],
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Load multiple tensors from one file in a single batch transfer.

        All chunks for all tensors are submitted at once, letting GDS_MT
        threads work in parallel.  One FILE registration per file.

        Args:
            fd: Open file descriptor.
            file_size: Total file size (to cap reads at EOF).
            tensor_list: ``[(file_offset, tensor_size), ...]``
            device: Target CUDA device.

        Returns:
            List of ``uint8`` GPU tensors (same order as *tensor_list*).
        """
        if self._agent is None:
            raise RuntimeError("GDS agent not initialized")

        max_chunk = self._max_chunk_size

        # Phase 1: Plan all chunks for all tensors
        # Each entry: (aligned_offset, aligned_size, prefix, useful, tensor_idx, bytes_loaded)
        chunk_plans: list[tuple[int, int, int, int, int, int]] = []
        result_buffers = []

        for ti, (file_offset, tensor_size) in enumerate(tensor_list):
            result_buffers.append(
                torch.empty(tensor_size, dtype=torch.uint8, device=device)
            )
            bytes_loaded = 0
            while bytes_loaded < tensor_size:
                cur_offset = file_offset + bytes_loaded
                remaining = tensor_size - bytes_loaded

                aligned_offset = (cur_offset // GDS_ALIGNMENT) * GDS_ALIGNMENT
                prefix_bytes = cur_offset - aligned_offset
                useful_bytes = min(remaining, max_chunk - prefix_bytes)

                aligned_size = (
                    (prefix_bytes + useful_bytes + GDS_ALIGNMENT - 1)
                    // GDS_ALIGNMENT
                ) * GDS_ALIGNMENT
                aligned_size = min(aligned_size, file_size - aligned_offset)

                chunk_plans.append((
                    aligned_offset, aligned_size,
                    prefix_bytes, useful_bytes,
                    ti, bytes_loaded,
                ))
                bytes_loaded += useful_bytes

        num_chunks = len(chunk_plans)

        # Phase 2: Allocate one staging buffer, build FILE + VRAM region lists
        total_staging = sum(p[1] for p in chunk_plans)
        staging = torch.empty(total_staging, dtype=torch.uint8, device=device)
        staging_base = staging.data_ptr()

        file_regions = []
        vram_regions = []
        staging_offset = 0

        for aligned_offset, aligned_size, _, _, _, _ in chunk_plans:
            file_regions.append((aligned_offset, aligned_size, fd, ""))
            vram_regions.append((
                staging_base + staging_offset,
                aligned_size,
                self._device_id,
                "",
            ))
            staging_offset += aligned_size

        # Phase 3: Batch register (one call each)
        file_descs = self._agent.register_memory(file_regions, "FILE")
        vram_descs = self._agent.register_memory(vram_regions, "VRAM")

        # Phase 4: Submit ALL transfers at once → GDS_MT threads parallelize
        handle = self._agent.initialize_xfer(
            "READ", vram_descs.trim(), file_descs.trim(), self._agent.name
        )

        state = self._agent.transfer(handle)
        if state == "ERR":
            self._agent.deregister_memory(file_descs)
            self._agent.deregister_memory(vram_descs)
            raise RuntimeError("GDS batch transfer failed")

        # Poll for completion
        timeout = float(os.environ.get("MX_GDS_TIMEOUT", "120"))
        t0 = time.perf_counter()
        spins = 0
        while True:
            state = self._agent.check_xfer_state(handle)
            if state == "DONE":
                break
            if state == "ERR":
                self._agent.release_xfer_handle(handle)
                self._agent.deregister_memory(file_descs)
                self._agent.deregister_memory(vram_descs)
                raise RuntimeError("GDS batch transfer error")
            if time.perf_counter() - t0 > timeout:
                self._agent.release_xfer_handle(handle)
                self._agent.deregister_memory(file_descs)
                self._agent.deregister_memory(vram_descs)
                raise TimeoutError("GDS batch transfer timeout")
            spins += 1
            if spins > 100:
                time.sleep(0.0001)
                spins = 0

        self._agent.release_xfer_handle(handle)
        self._agent.deregister_memory(file_descs)
        self._agent.deregister_memory(vram_descs)

        logger.info(
            "Batch transfer: %d tensors, %d chunks, %.1f MB staging",
            len(tensor_list), num_chunks, total_staging / 1e6,
        )

        # Phase 5: Copy from staging to result buffers
        staging_offset = 0
        for aligned_offset, aligned_size, prefix, useful, ti, bytes_loaded in chunk_plans:
            result_buffers[ti][bytes_loaded : bytes_loaded + useful] = (
                staging[staging_offset + prefix : staging_offset + prefix + useful]
            )
            staging_offset += aligned_size

        return result_buffers

    def shutdown(self) -> None:
        """Clean up NIXL GDS resources."""
        self._agent = None
        logger.info("GdsTransferManager shutdown complete")
