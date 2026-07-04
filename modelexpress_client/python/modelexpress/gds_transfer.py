# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NIXL GDS Transfer Manager for direct file-to-GPU weight loading.

Uses NIXL's GDS_MT (multithreaded GPUDirect Storage) backend for
zero-copy transfers from NVMe storage to GPU memory.

Environment variables:
    MX_GDS_MAX_CHUNK_KB: Maximum chunk size in KB (default: 131072 = 128 MB)
    MX_GDS_THREADS: Number of GDS transfer threads (default: 8)
    MX_GDS_TIMEOUT: Transfer timeout in seconds (default: 120)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from . import envs
from .accelerators import AcceleratorBackend

logger = logging.getLogger("modelexpress.gds_transfer")

_MX_GDS_TIMEOUT = envs.MX_GDS_TIMEOUT

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
    """
    Check if GPUDirect Storage is available at the system level.

    NIXL silently falls back to POSIX I/O when GDS is not available,
    so we cannot rely on agent creation to detect GDS support. Instead
    we check for the nvidia_fs kernel module and libcufile shared library
    which are the actual prerequisites for GDS transfers.
    """
    if not NIXL_AVAILABLE:
        return False

    if not _nvidia_fs_loaded():
        logger.debug("GDS not available: nvidia_fs kernel module not loaded")
        return False

    if not _cufile_loadable():
        logger.debug("GDS not available: libcufile.so not found")
        return False

    logger.debug("GDS available: nvidia_fs loaded and libcufile present")
    return True


def _nvidia_fs_loaded() -> bool:
    """Check if the nvidia_fs kernel module is loaded via /proc/modules."""
    try:
        with open("/proc/modules") as f:
            for line in f:
                if line.startswith("nvidia_fs "):
                    return True
        return False
    except OSError:
        return False


def _cufile_loadable() -> bool:
    """Check if libcufile.so can be loaded by the dynamic linker."""
    try:
        import ctypes
        ctypes.CDLL("libcufile.so")
        return True
    except OSError:
        return False


_DEFAULT_MAX_CHUNK = 128 * 1024 * 1024  # 128 MB


@dataclass(frozen=True)
class GdsReadRequest:
    fd: int
    file_offset: int
    dst_addr: int
    byte_count: int
    device: int
    label: str | None = None


@dataclass
class GdsReadTransfer:
    handle: object | None
    file_descs: object | None
    vram_descs: object | None
    label: str
    request_count: int
    total_bytes: int
    request_contexts: tuple[str, ...] = ()
    state: str = "INIT"
    released: bool = False


class GdsTransferManager:
    """
    Manages NIXL GDS_MT backend for direct file-to-GPU transfers.

    Supports batch loading: all tensors from a file are submitted in
    a single NIXL transfer so GDS_MT threads work in parallel.

    Usage as context manager:
        with GdsTransferManager(agent_name="mx-gds-0", accelerator_backend=backend) as gds:
            gds.batch_load_file(fd, file_size, tensor_list, device)
    """

    def __init__(self, agent_name: str, accelerator_backend: AcceleratorBackend):
        self._agent_name = agent_name
        self._device_id: int | None = None
        self._agent: Any = None
        self._accelerator_backend = accelerator_backend
        override = envs.MX_GDS_MAX_CHUNK_KB
        self._max_chunk_size = int(override) * 1024 if override else _DEFAULT_MAX_CHUNK

    def __enter__(self) -> GdsTransferManager:
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
        return None

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

        self._device_id = self._accelerator_backend.current_device()

        thread_count = envs.MX_GDS_THREADS
        config = NixlAgentConfig(backends=["GDS_MT"], num_threads=thread_count)
        self._agent = NixlAgent(self._agent_name, config)

        logger.info(
            "GDS_MT agent '%s' created on device %d (threads=%d, max_chunk=%dMB)",
            self._agent_name, self._device_id, thread_count,
            self._max_chunk_size // (1024 * 1024),
        )

    def prepare_read(
        self,
        requests: Sequence[GdsReadRequest],
        *,
        label: str | None = None,
    ) -> GdsReadTransfer:
        """Prepare and register one GDS read transfer without starting it."""
        if self._agent is None:
            raise RuntimeError("GDS agent not initialized")
        if not requests:
            raise ValueError("requests must not be empty")

        file_regions = []
        vram_regions = []
        labels = []
        request_contexts = []
        total_bytes = 0
        for request in requests:
            request_label = request.label or (
                f"fd={request.fd}:{request.file_offset}+{request.byte_count}"
            )
            request_context = (
                f"label={request_label}, fd={request.fd}, "
                f"file_offset={request.file_offset}, "
                f"byte_count={request.byte_count}, dst_addr={request.dst_addr}, "
                f"device={request.device}"
            )
            labels.append(request_label)
            request_contexts.append(request_context)

            file_regions.append(
                (request.file_offset, request.byte_count, request.fd, "")
            )
            vram_regions.append(
                (request.dst_addr, request.byte_count, request.device, "")
            )
            total_bytes += request.byte_count

        transfer_label = label or (
            labels[0] if len(labels) == 1 else f"gds-read:{len(labels)}-ranges"
        )
        contexts = tuple(request_contexts)
        transfer_context = self._read_transfer_context(transfer_label, contexts)

        file_descs = None
        vram_descs = None
        handle = None
        try:
            file_descs = self._agent.register_memory(file_regions, "FILE")
            vram_descs = self._agent.register_memory(vram_regions, "VRAM")
            handle = self._agent.initialize_xfer(
                "READ",
                vram_descs.trim(),
                file_descs.trim(),
                self._agent.name,
            )
            return GdsReadTransfer(
                handle=handle,
                file_descs=file_descs,
                vram_descs=vram_descs,
                label=transfer_label,
                request_count=len(requests),
                total_bytes=total_bytes,
                request_contexts=contexts,
            )
        except BaseException as exc:
            transfer = GdsReadTransfer(
                handle=handle,
                file_descs=file_descs,
                vram_descs=vram_descs,
                label=transfer_label,
                request_count=len(requests),
                total_bytes=total_bytes,
                request_contexts=contexts,
            )
            self.release(transfer)
            if not isinstance(exc, Exception):
                raise
            raise RuntimeError(
                f"Failed to prepare GDS transfer: {transfer_context}: {exc}"
            ) from exc

    def start(self, transfer: GdsReadTransfer) -> None:
        """Post a prepared GDS read transfer."""
        if transfer.released:
            raise RuntimeError("GDS transfer was already released")
        if transfer.handle is None:
            raise RuntimeError("GDS transfer has no handle")
        if transfer.state != "INIT":
            raise RuntimeError("GDS transfer must be in state INIT to start")

        try:
            state = self._agent.transfer(transfer.handle)
        except BaseException as exc:
            transfer.state = "ERR"
            if not isinstance(exc, Exception):
                raise
            context = self._read_transfer_context(
                transfer.label, transfer.request_contexts
            )
            raise RuntimeError(
                f"Failed to start GDS transfer: {context}: {exc}"
            ) from exc

        transfer.state = str(state)
        if state == "ERR":
            raise RuntimeError(f"GDS transfer failed to start with {state!r} state")
        if state not in {"PROC", "DONE"}:
            raise RuntimeError(
                f"GDS transfer returned unexpected {state!r} state"
            )

    def wait(self, transfer: GdsReadTransfer) -> None:
        """Wait for a started GDS read transfer to complete."""
        if transfer.released:
            raise RuntimeError("GDS transfer was already released")
        if transfer.handle is None:
            raise RuntimeError("GDS transfer has no handle")

        start = time.perf_counter()
        spins = 0

        if transfer.state == "INIT":
            raise RuntimeError("GDS transfer was not started")
        if transfer.state not in {"PROC", "DONE"}:
            raise RuntimeError(
                f"GDS transfer cannot wait from state {transfer.state!r}"
            )

        while transfer.state != "DONE":
            if time.perf_counter() - start > _MX_GDS_TIMEOUT:
                transfer.state = "ERR"
                raise TimeoutError(
                    f"GDS transfer timeout after {_MX_GDS_TIMEOUT}s"
                )
            try:
                state = self._agent.check_xfer_state(transfer.handle)
            except BaseException as exc:
                transfer.state = "ERR"
                if not isinstance(exc, Exception):
                    raise
                context = self._read_transfer_context(
                    transfer.label, transfer.request_contexts
                )
                raise RuntimeError(
                    f"Failed to check GDS transfer state: {context}: {exc}"
                ) from exc

            transfer.state = str(state)
            if state == "ERR":
                raise RuntimeError(f"GDS transfer failed to complete with {state!r} state")
            if state not in {"PROC", "DONE"}:
                raise RuntimeError(
                    f"GDS transfer returned unexpected {state!r} state"
                )

            spins += 1
            if spins > 100:
                time.sleep(0.0001)
                spins = 0

        transfer.state = "DONE"
        logger.debug(
            "GDS read transfer complete: label=%s ranges=%d bytes=%d",
            transfer.label,
            transfer.request_count,
            transfer.total_bytes,
        )

    def release(self, transfer: GdsReadTransfer) -> None:
        """Release a prepared GDS read transfer."""
        if transfer.released:
            return

        if transfer.handle is not None:
            self._agent.release_xfer_handle(transfer.handle)
            transfer.handle = None

        if transfer.file_descs is not None:
            self._agent.deregister_memory(transfer.file_descs)
            transfer.file_descs = None

        if transfer.vram_descs is not None:
            self._agent.deregister_memory(transfer.vram_descs)
            transfer.vram_descs = None

        transfer.released = True

    @staticmethod
    def _read_transfer_context(
        label: str,
        request_contexts: Sequence[str],
    ) -> str:
        if not request_contexts:
            return f"label={label}"
        return f"label={label}; requests=[{'; '.join(request_contexts)}]"

    def batch_load_file(
        self,
        fd: int,
        file_size: int,
        tensor_list: list[tuple[int, int]],
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Load multiple tensors from one file in a single batch transfer.

        All tensors are submitted at once so GDS_MT threads work in parallel.
        Large tensors are split into chunks of max_chunk_size.

        Args:
            fd: Open file descriptor (from os.open).
            file_size: Total file size (to cap reads at EOF).
            tensor_list: [(file_offset, tensor_size), ...]
            device: Target CUDA device.

        Returns:
            List of uint8 GPU tensors (same order as tensor_list).
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
                chunk = min(chunk, file_size - (file_offset + loaded))
                if chunk <= 0:
                    raise RuntimeError(
                        f"GDS read beyond EOF: file_offset={file_offset}, "
                        f"loaded={loaded}, file_size={file_size}"
                    )

                file_regions.append((file_offset + loaded, chunk, fd, ""))
                vram_regions.append((gpu_base + loaded, chunk, self._device_id, ""))
                loaded += chunk

        # Phase 2: Batch register
        file_descs = self._agent.register_memory(file_regions, "FILE")
        vram_descs = self._agent.register_memory(vram_regions, "VRAM")

        # Phase 3: Submit all at once
        handle = self._agent.initialize_xfer(
            "READ", vram_descs.trim(), file_descs.trim(), self._agent.name
        )

        state = self._agent.transfer(handle)
        if state == "ERR":
            self._agent.release_xfer_handle(handle)
            self._free_nixl_memory(file_descs, vram_descs)
            raise RuntimeError("GDS batch transfer failed")

        # Phase 4: Wait for completion
        timeout = envs.MX_GDS_TIMEOUT
        t0 = time.perf_counter()
        spins = 0
        while True:
            state = self._agent.check_xfer_state(handle)
            if state == "DONE":
                break
            if state == "ERR":
                self._agent.release_xfer_handle(handle)
                self._free_nixl_memory(file_descs, vram_descs)
                raise RuntimeError("GDS batch transfer error")
            if time.perf_counter() - t0 > timeout:
                self._agent.release_xfer_handle(handle)
                self._free_nixl_memory(file_descs, vram_descs)
                raise TimeoutError("GDS batch transfer timeout")
            spins += 1
            if spins > 100:
                time.sleep(0.0001)
                spins = 0

        self._agent.release_xfer_handle(handle)
        self._free_nixl_memory(file_descs, vram_descs)

        return result_buffers

    def _free_nixl_memory(self, file_descs: Any, vram_descs: Any) -> None:
        """Deregister FILE and VRAM descriptors from the NIXL agent."""
        self._agent.deregister_memory(file_descs)
        self._agent.deregister_memory(vram_descs)

    def shutdown(self) -> None:
        """Clean up NIXL GDS resources."""
        self._agent = None
        logger.info("GdsTransferManager shutdown complete")
