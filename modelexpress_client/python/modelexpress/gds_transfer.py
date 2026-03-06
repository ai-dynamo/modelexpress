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
        self._chunk_buffer: torch.Tensor | None = None
        self._chunk_gpu_descs: Any = None
        override = os.environ.get("MX_GDS_MAX_CHUNK_KB")
        if override:
            self._max_chunk_size = int(override) * 1024
        else:
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

    def load_tensor(
        self,
        fd: int,
        file_offset: int,
        tensor_size: int,
        device: torch.device,
        file_size: int = 0,
    ) -> torch.Tensor:
        """Load raw bytes from file directly to a GPU tensor via GDS.

        Args:
            fd: Open file descriptor.
            file_offset: Absolute byte offset of the tensor data in the file.
            tensor_size: Number of bytes to read.
            device: Target CUDA device.
            file_size: Total file size (to avoid reading past EOF).

        Returns:
            A ``uint8`` GPU tensor of length *tensor_size*.
        """
        if self._agent is None:
            raise RuntimeError("GDS agent not initialized")

        if file_size == 0:
            file_size = os.fstat(fd).st_size

        result_buffer = torch.empty(tensor_size, dtype=torch.uint8, device=device)
        bytes_loaded = 0

        while bytes_loaded < tensor_size:
            cur_offset = file_offset + bytes_loaded
            remaining = tensor_size - bytes_loaded

            aligned_offset = (cur_offset // GDS_ALIGNMENT) * GDS_ALIGNMENT
            prefix_bytes = cur_offset - aligned_offset
            useful_bytes = min(remaining, self._max_chunk_size - prefix_bytes)

            aligned_size = (
                (prefix_bytes + useful_bytes + GDS_ALIGNMENT - 1)
                // GDS_ALIGNMENT
            ) * GDS_ALIGNMENT
            aligned_size = min(aligned_size, file_size - aligned_offset)

            # Reuse chunk buffer and its VRAM registration
            if self._chunk_buffer is None or self._chunk_buffer.numel() < aligned_size:
                if self._chunk_gpu_descs is not None:
                    self._agent.deregister_memory(self._chunk_gpu_descs)
                self._chunk_buffer = torch.empty(
                    max(aligned_size, self._max_chunk_size),
                    dtype=torch.uint8, device=device,
                )
                self._chunk_gpu_descs = self._agent.register_memory(
                    self._chunk_buffer, "VRAM"
                )
            chunk_buf = self._chunk_buffer[:aligned_size]

            self._transfer_chunk(fd, aligned_offset, chunk_buf, aligned_size)

            result_buffer[bytes_loaded : bytes_loaded + useful_bytes] = (
                chunk_buf[prefix_bytes : prefix_bytes + useful_bytes]
            )
            bytes_loaded += useful_bytes

        return result_buffer

    def _transfer_chunk(
        self, fd: int, file_offset: int, gpu_buffer: torch.Tensor, size: int
    ) -> None:
        """Transfer one aligned chunk: file -> GPU via NIXL GDS."""
        file_descs = self._agent.register_memory(
            [(file_offset, size, fd, "")], "FILE"
        )
        file_xfer = file_descs.trim()
        gpu_xfer = self._agent.get_xfer_descs(gpu_buffer, "VRAM")

        handle = self._agent.initialize_xfer(
            "READ", gpu_xfer, file_xfer, self._agent.name
        )

        state = self._agent.transfer(handle)
        if state == "ERR":
            self._agent.deregister_memory(file_descs)
            raise RuntimeError(f"GDS transfer failed at offset {file_offset}")

        timeout = float(os.environ.get("MX_GDS_TIMEOUT", "60"))
        t0 = time.perf_counter()
        spins = 0
        while True:
            state = self._agent.check_xfer_state(handle)
            if state == "DONE":
                break
            if state == "ERR":
                self._agent.release_xfer_handle(handle)
                self._agent.deregister_memory(file_descs)
                raise RuntimeError(f"GDS transfer error at offset {file_offset}")
            if time.perf_counter() - t0 > timeout:
                self._agent.release_xfer_handle(handle)
                self._agent.deregister_memory(file_descs)
                raise TimeoutError(f"GDS transfer timeout at offset {file_offset}")
            spins += 1
            if spins > 100:
                time.sleep(0.0001)
                spins = 0

        self._agent.release_xfer_handle(handle)
        self._agent.deregister_memory(file_descs)

    def shutdown(self) -> None:
        """Clean up NIXL GDS resources."""
        if self._chunk_gpu_descs is not None and self._agent is not None:
            self._agent.deregister_memory(self._chunk_gpu_descs)
            self._chunk_gpu_descs = None
        self._chunk_buffer = None
        self._agent = None
        logger.info("GdsTransferManager shutdown complete")
