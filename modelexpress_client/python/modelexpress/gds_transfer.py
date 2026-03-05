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
try:
    from nixl._api import nixl_agent as NixlAgent
    NIXL_AVAILABLE = True
except ImportError:
    pass


def is_gds_available() -> bool:
    """Check if NIXL (required for GDS) is available."""
    return NIXL_AVAILABLE


# GDS requires 4KB-aligned file offsets and transfer sizes.
GDS_ALIGNMENT = 4096

# Default max chunk size per GDS I/O (must be <= cuFile per_buffer_cache_size).
_DEFAULT_MAX_CHUNK = 16 * 1024 * 1024  # 16 MB


def _read_max_chunk_from_cufile() -> int:
    """Read max safe chunk size from cufile.json.

    The effective per-request limit is the smaller of:
      - ``per_buffer_cache_size_kb``
      - ``max_device_cache_size_kb / io_batchsize``
    """
    cufile_path = os.environ.get("CUFILE_ENV_PATH_JSON", "/etc/cufile.json")
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
        per_buffer_kb = props.get("per_buffer_cache_size_kb", 1024)
        max_device_kb = props.get("max_device_cache_size_kb", 131072)
        io_batchsize = props.get("io_batchsize", 128)
        shadow_kb = max_device_kb // io_batchsize if io_batchsize > 0 else per_buffer_kb
        effective_kb = min(per_buffer_kb, shadow_kb)
        logger.info(
            "cufile.json: per_buffer=%dKB, max_device=%dKB, io_batchsize=%d "
            "-> effective max_chunk=%dKB",
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

        # Create agent without pre-configured backends, then add GDS
        self._agent = NixlAgent(self._agent_name)
        self._agent.create_backend("GDS")

        logger.info(
            "GDS agent '%s' created on device %d (max_chunk=%dKB)",
            self._agent_name, self._device_id,
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
    ) -> torch.Tensor:
        """Load raw bytes from file directly to a GPU tensor via GDS.

        Handles 4KB alignment and chunking automatically.

        Args:
            fd: Open file descriptor (``os.open``).
            file_offset: Absolute byte offset of the tensor data in the file.
            tensor_size: Number of bytes to read.
            device: Target CUDA device.

        Returns:
            A ``uint8`` GPU tensor of length *tensor_size*.
        """
        if self._agent is None:
            raise RuntimeError("GDS agent not initialized")

        result_buffer = torch.empty(tensor_size, dtype=torch.uint8, device=device)
        bytes_loaded = 0

        while bytes_loaded < tensor_size:
            cur_offset = file_offset + bytes_loaded
            remaining = tensor_size - bytes_loaded

            # Align offset down to 4KB boundary
            aligned_offset = (cur_offset // GDS_ALIGNMENT) * GDS_ALIGNMENT
            prefix_bytes = cur_offset - aligned_offset

            # Cap useful bytes so aligned_size stays <= max_chunk_size
            useful_bytes = min(remaining, self._max_chunk_size - prefix_bytes)

            # Round up to 4KB
            aligned_size = (
                (prefix_bytes + useful_bytes + GDS_ALIGNMENT - 1)
                // GDS_ALIGNMENT
            ) * GDS_ALIGNMENT

            # Reuse chunk buffer to avoid repeated GPU allocation
            if self._chunk_buffer is None or self._chunk_buffer.numel() < aligned_size:
                self._chunk_buffer = torch.empty(
                    max(aligned_size, self._max_chunk_size),
                    dtype=torch.uint8, device=device,
                )
            chunk_buf = self._chunk_buffer[:aligned_size]

            # Execute one GDS transfer
            self._transfer_chunk(fd, aligned_offset, chunk_buf, aligned_size)

            # Copy the useful slice into result
            result_buffer[bytes_loaded : bytes_loaded + useful_bytes] = (
                chunk_buf[prefix_bytes : prefix_bytes + useful_bytes]
            )
            bytes_loaded += useful_bytes

        return result_buffer

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _transfer_chunk(
        self, fd: int, file_offset: int, gpu_buffer: torch.Tensor, size: int
    ) -> None:
        """Transfer one aligned chunk: file → GPU via NIXL GDS."""

        # Register FILE region: (offset, size, fd, tag)
        file_descs = self._agent.register_memory(
            [(file_offset, size, fd, f"c_{file_offset}")], "FILE"
        )
        file_xfer = file_descs.trim()

        # Register GPU buffer
        gpu_descs = self._agent.register_memory(gpu_buffer, "VRAM")
        gpu_xfer = self._agent.get_xfer_descs(gpu_buffer, "VRAM")

        # Execute transfer (READ = file → GPU)
        handle = self._agent.initialize_xfer(
            "READ", gpu_xfer, file_xfer, self._agent.name
        )

        state = self._agent.transfer(handle)
        if state == "ERR":
            self._agent.deregister_memory(file_descs)
            self._agent.deregister_memory(gpu_descs)
            raise RuntimeError(f"GDS transfer failed at offset {file_offset}")

        # Poll for completion
        timeout = float(os.environ.get("MX_GDS_TIMEOUT", "60"))
        t0 = time.perf_counter()
        while True:
            state = self._agent.check_xfer_state(handle)
            if state == "DONE":
                break
            if state == "ERR":
                self._agent.release_xfer_handle(handle)
                self._agent.deregister_memory(file_descs)
                self._agent.deregister_memory(gpu_descs)
                raise RuntimeError(f"GDS transfer error at offset {file_offset}")
            if time.perf_counter() - t0 > timeout:
                self._agent.release_xfer_handle(handle)
                self._agent.deregister_memory(file_descs)
                self._agent.deregister_memory(gpu_descs)
                raise TimeoutError(f"GDS transfer timeout at offset {file_offset}")
            time.sleep(0.001)

        self._agent.release_xfer_handle(handle)
        self._agent.deregister_memory(file_descs)
        self._agent.deregister_memory(gpu_descs)

    def shutdown(self) -> None:
        """Clean up NIXL GDS resources."""
        self._chunk_buffer = None
        self._agent = None
        logger.info("GdsTransferManager shutdown complete")
