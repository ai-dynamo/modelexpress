# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NIXL GDS Transfer Manager for direct file-to-GPU weight loading.

Uses NIXL's GDS_MT (multithreaded GPUDirect Storage) backend for
zero-copy transfers from NVMe storage to GPU memory.

Framework-agnostic — depends only on nixl and torch.
"""

from __future__ import annotations

import logging
import os
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


# Max chunk size per GDS I/O request.
_DEFAULT_MAX_CHUNK = 128 * 1024 * 1024  # 128 MB


class GdsTransferManager:
    """
    Manages NIXL GDS_MT backend for direct file-to-GPU transfers.

    Supports batch loading: all tensors from a file are submitted in
    a single NIXL transfer so GDS_MT threads work in parallel.
    """

    def __init__(self, agent_name: str):
        self._agent_name = agent_name
        self._device_id: int | None = None
        self._agent: Any = None
        override = os.environ.get("MX_GDS_MAX_CHUNK_KB")
        self._max_chunk_size = int(override) * 1024 if override else _DEFAULT_MAX_CHUNK

    @property
    def agent_name(self) -> str:
        return self._agent_name

    def initialize(self) -> None:
        """Initialize the NIXL agent with GDS_MT backend."""
        if not NIXL_AVAILABLE:
            raise RuntimeError(
                "NIXL is not available. Install with: pip install nixl[cu12]"
            )
        if self._agent is not None:
            return

        self._device_id = torch.cuda.current_device()

        thread_count = int(os.environ.get("MX_GDS_THREADS", "8"))
        config = NixlAgentConfig(backends=["GDS_MT"], num_threads=thread_count)
        self._agent = NixlAgent(self._agent_name, config)

        logger.info(
            "GDS_MT agent '%s' created on device %d (threads=%d, max_chunk=%dMB)",
            self._agent_name, self._device_id, thread_count,
            self._max_chunk_size // (1024 * 1024),
        )

    def batch_load_file(
        self,
        fd: int,
        file_size: int,
        tensor_list: list[tuple[int, int]],
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Load multiple tensors from one file in a single batch transfer.

        GDS_MT supports non-aligned offsets, so tensors are read directly
        into result buffers without intermediate staging or alignment logic.
        Large tensors are split into chunks of max_chunk_size.

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

        # Phase 1: Allocate result buffers, plan chunks
        result_buffers = []
        file_regions = []
        vram_regions = []

        for file_offset, tensor_size in tensor_list:
            buf = torch.empty(tensor_size, dtype=torch.uint8, device=device)
            result_buffers.append(buf)
            gpu_base = buf.data_ptr()

            loaded = 0
            while loaded < tensor_size:
                chunk = min(tensor_size - loaded, max_chunk)
                # Cap at EOF
                chunk = min(chunk, file_size - (file_offset + loaded))

                file_regions.append((file_offset + loaded, chunk, fd, ""))
                vram_regions.append((gpu_base + loaded, chunk, self._device_id, ""))
                loaded += chunk

        # Phase 2: Batch register (one call each, one cuFileHandleRegister per fd)
        file_descs = self._agent.register_memory(file_regions, "FILE")
        vram_descs = self._agent.register_memory(vram_regions, "VRAM")

        # Phase 3: Submit all at once -> GDS_MT threads parallelize
        handle = self._agent.initialize_xfer(
            "READ", vram_descs.trim(), file_descs.trim(), self._agent.name
        )

        state = self._agent.transfer(handle)
        if state == "ERR":
            self._agent.deregister_memory(file_descs)
            self._agent.deregister_memory(vram_descs)
            raise RuntimeError("GDS batch transfer failed")

        # Phase 4: Wait for completion
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

        return result_buffers

    def shutdown(self) -> None:
        """Clean up NIXL GDS resources."""
        self._agent = None
        logger.info("GdsTransferManager shutdown complete")
