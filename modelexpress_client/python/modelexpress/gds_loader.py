# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Framework-agnostic GDS model loader.

Loads model weights from safetensors files directly to GPU memory via NIXL's
GDS (GPUDirect Storage) backend, bypassing CPU bounce buffers entirely.

The target GPU is determined from the active accelerator backend, matching the
behavior of vLLM/sglang default loaders on CUDA.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import time
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import torch

from .accelerators import AcceleratorBackend
from .gds_constants import _GDS_ALIGNMENT
from .gds_transfer import (
    GdsReadRequest,
    GdsReadTransfer,
    GdsTransferManager,
    is_gds_available,
)

logger = logging.getLogger("modelexpress.gds_loader")

# Complete dtype mapping from the safetensors spec:
# https://huggingface.co/docs/safetensors/metadata_parsing#accepted-dtypes
SAFETENSORS_DTYPE_MAP: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


@dataclass(frozen=True)
class MxFileReadSource:
    allocation_id: str
    file_path: str
    file_offset: int
    byte_count: int


@dataclass(frozen=True)
class MxDeviceReadTarget:
    allocation_id: str
    va: int
    device: int
    byte_count: int


class MxGdsLoader:
    """
    Load model weights from safetensors files directly to GPU via GDS.

    Framework-agnostic. Can be used from vLLM, sglang, or standalone.

    Usage::

        loader = MxGdsLoader(accelerator_backend)
        tensors = loader.load("/path/to/model")

        # Or stream per-file:
        for name, tensor in loader.load_iter("/path/to/model"):
            process(name, tensor)
    """

    def __init__(self, accelerator_backend: AcceleratorBackend):
        self._gds_manager: GdsTransferManager | None = None
        self._device_id: int | None = None
        self._accelerator_backend = accelerator_backend

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, model_path: str) -> dict[str, torch.Tensor]:
        """Load all tensors from model_path to GPU."""
        result: dict[str, torch.Tensor] = {}
        for name, tensor in self.load_iter(model_path):
            result[name] = tensor
        return result

    def load_iter(
        self,
        model_path: str,
        *,
        use_tqdm: bool = True,
        revision: str | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """
        Yield (tensor_name, gpu_tensor) pairs loaded via GDS.

        Each safetensors file is batch-loaded through a single GDS
        transfer, then its tensors are yielded one by one.
        """
        load_start = time.perf_counter()
        model_path = self._resolve_model_path(model_path, revision=revision)

        if not is_gds_available():
            raise RuntimeError(
                "GDS is not available. Check nvidia_fs module and libcufile."
            )

        self._device_id = self._accelerator_backend.current_device()
        self._ensure_gds_manager()

        file_tensor_map = self._resolve_safetensors_files(model_path)

        file_jobs = []
        for file_path, tensor_names in file_tensor_map.items():
            header_info = self._parse_safetensors_header(file_path)
            file_tensors = {
                name: header_info[name]
                for name in tensor_names
                if name in header_info
            }
            if file_tensors:
                file_jobs.append((file_path, file_tensors))

        if not file_jobs:
            return

        # Prefetch pipeline: load file[i+1] while yielding file[i]
        total_files = len(file_jobs)
        pbar = None
        if use_tqdm:
            from tqdm import tqdm
            pbar = tqdm(
                total=total_files,
                desc="Loading safetensors via GDS",
                unit="file",
            )

        pool = ThreadPoolExecutor(max_workers=1)
        try:
            pending = pool.submit(self._load_file_tensors, *file_jobs[0])

            for i in range(total_files):
                loaded = pending.result()
                if pbar is not None:
                    pbar.update(1)

                if i + 1 < total_files:
                    pending = pool.submit(
                        self._load_file_tensors, *file_jobs[i + 1]
                    )

                for name, tensor in loaded.items():
                    yield name, tensor

            logger.info("GDS load complete in %.2fs", time.perf_counter() - load_start)
        finally:
            if pbar is not None:
                pbar.close()
            pool.shutdown(wait=True)

    def restore_gms_snapshot(
        self,
        *,
        grouped_sources: Mapping[
            str, Sequence[tuple[MxFileReadSource, MxDeviceReadTarget]]
        ],
        device: int,
        max_workers: int = 16,
        chunk_size_bytes: int | None = None,
        max_inflight_batches: int | None = None,
    ) -> dict[str, object]:
        """Restore GMS snapshot file ranges into existing GPU VAs via GDS."""
        if max_inflight_batches is None:
            max_inflight_batches = max_workers
        max_inflight_batches = max(1, int(max_inflight_batches))

        if not grouped_sources:
            return {
                "total_bytes": 0,
                "elapsed_s": 0.0,
                "selected_strategy": "gds",
                "source_count": 0,
                "file_count": 0,
                "max_inflight_batches": max_inflight_batches,
            }
        if not is_gds_available():
            raise RuntimeError("GDS is not available")

        device_id = int(device)
        if device_id < 0:
            raise RuntimeError(
                f"GMS transfer device must be non-negative: {device_id}"
            )
        source_count = 0
        total_bytes = 0
        for pairs in grouped_sources.values():
            for source, target in pairs:
                source_count += 1
                total_bytes += int(source.byte_count)
                target_device = int(target.device)
                if target_device != device_id:
                    raise RuntimeError(
                        "MxGdsLoader.restore_gms_snapshot expects one CUDA device per "
                        f"call: expected={device} got={target.device}"
                    )

        torch.cuda.set_device(device_id)
        self._device_id = device_id
        self._ensure_gds_manager()

        start = time.perf_counter()
        pending: list[tuple[GdsReadTransfer, int]] = []
        fd: int | None = None
        transfer: GdsReadTransfer | None = None
        try:
            for file_path, pairs in sorted(grouped_sources.items()):
                fd = self._open_gds_file(file_path)
                file_size = os.fstat(fd).st_size
                requests = self._build_read_requests(
                    file_path=file_path,
                    fd=fd,
                    file_size=file_size,
                    pairs=pairs,
                    chunk_size_bytes=chunk_size_bytes,
                )
                transfer = self._gds_manager.prepare_read(
                    requests,
                    label=f"gms:{Path(file_path).name}",
                )
                self._gds_manager.start(transfer)
                pending.append((transfer, fd))
                transfer = None
                fd = None

                if len(pending) >= max_inflight_batches:
                    self._wait_and_release(*pending.pop(0))

            while pending:
                self._wait_and_release(*pending.pop(0))
        except Exception as exc:
            if fd is not None:
                try:
                    if transfer is None:
                        os.close(fd)
                    else:
                        self._release_transfer(transfer, fd)
                except Exception as cleanup_exc:
                    logger.warning(
                        "Failed to clean up current GDS restore file fd=%d: %s",
                        fd,
                        cleanup_exc,
                    )
            for pending_transfer, pending_fd in pending:
                try:
                    self._release_transfer(pending_transfer, pending_fd)
                except Exception as cleanup_exc:
                    logger.warning(
                        "Failed to clean up pending GDS restore transfer fd=%d: %s",
                        pending_fd,
                        cleanup_exc,
                    )
            raise RuntimeError(f"GMS snapshot GDS restore failed: {exc}") from exc

        elapsed = time.perf_counter() - start
        throughput = total_bytes / elapsed / (1024**3) if elapsed > 0 else 0.0
        logger.info(
            "GMS snapshot GDS restore complete: %.2f GiB in %.3fs "
            "(%.2f GiB/s, files=%d, sources=%d, max_inflight=%d)",
            total_bytes / (1024**3),
            elapsed,
            throughput,
            len(grouped_sources),
            source_count,
            max_inflight_batches,
        )
        return {
            "total_bytes": total_bytes,
            "elapsed_s": elapsed,
            "selected_strategy": "gds",
            "source_count": source_count,
            "file_count": len(grouped_sources),
            "max_inflight_batches": max_inflight_batches,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_read_requests(
        self,
        *,
        file_path: str,
        fd: int,
        file_size: int,
        pairs: Sequence[tuple[MxFileReadSource, MxDeviceReadTarget]],
        chunk_size_bytes: int | None,
    ) -> list[GdsReadRequest]:
        """Build GDS read requests for one shard file's source/target pairs.

        Validates the GDS-specific invariants (positive byte_count, aligned
        non-negative file_offset, aligned positive VA, reads within EOF) and
        splits ranges larger than chunk_size_bytes into aligned chunks.
        """
        if chunk_size_bytes is not None:
            if chunk_size_bytes <= 0:
                raise ValueError("chunk_size_bytes must be positive when set")
            if chunk_size_bytes % _GDS_ALIGNMENT != 0:
                raise ValueError(
                    f"chunk_size_bytes must be a multiple of {_GDS_ALIGNMENT} when set"
                )

        requests: list[GdsReadRequest] = []
        for source, target in pairs:
            source_size = int(source.byte_count)
            if source_size <= 0:
                raise RuntimeError(
                    "GMS transfer byte_count must be positive for "
                    f"{source.allocation_id}: {source.byte_count}"
                )

            file_offset = int(source.file_offset)
            if file_offset < 0:
                raise RuntimeError(
                    "GMS transfer file_offset must be non-negative for "
                    f"{source.allocation_id}: {source.file_offset}"
                )
            if file_offset % _GDS_ALIGNMENT != 0:
                raise RuntimeError(
                    f"GMS transfer file_offset must be {_GDS_ALIGNMENT}-byte aligned for "
                    f"{source.allocation_id}: {source.file_offset}"
                )

            target_va = int(target.va)
            if target_va <= 0:
                raise RuntimeError(
                    "GMS transfer target VA must be positive for "
                    f"{source.allocation_id}: {target.va}"
                )
            if target_va % _GDS_ALIGNMENT != 0:
                raise RuntimeError(
                    f"GMS transfer target VA must be {_GDS_ALIGNMENT}-byte aligned for "
                    f"{source.allocation_id}: {target.va}"
                )

            end = file_offset + source_size
            if end > file_size:
                raise RuntimeError(
                    "GDS read beyond EOF for allocation "
                    f"{source.allocation_id}: path={file_path} end={end} "
                    f"file_size={file_size}"
                )

            chunk_limit = source_size
            if chunk_size_bytes is not None:
                chunk_limit = int(chunk_size_bytes)

            done = 0
            while done < source_size:
                chunk = min(chunk_limit, source_size - done)
                requests.append(
                    GdsReadRequest(
                        fd=fd,
                        file_offset=file_offset + done,
                        dst_addr=target_va + done,
                        byte_count=chunk,
                        device=int(target.device),
                        label=(
                            f"allocation_id={source.allocation_id} "
                            f"file_path={file_path}"
                        ),
                    )
                )
                done += chunk

        return requests

    def _wait_and_release(self, transfer: GdsReadTransfer, fd: int) -> None:
        """Wait for a started transfer, then release it and close its file."""
        if self._gds_manager is None:
            raise RuntimeError("GDS manager is not initialized")
        try:
            self._gds_manager.wait(transfer)
        except Exception:
            try:
                self._release_transfer(transfer, fd)
            except Exception as cleanup_exc:
                logger.warning(
                    "Failed to release GDS transfer after wait failure fd=%d: %s",
                    fd,
                    cleanup_exc,
                )
            raise
        self._release_transfer(transfer, fd)

    def _release_transfer(
        self,
        transfer: GdsReadTransfer,
        fd: int,
    ) -> None:
        if self._gds_manager is None:
            raise RuntimeError("GDS manager is not initialized")
        try:
            self._gds_manager.release(transfer)
        finally:
            os.close(fd)

    @staticmethod
    def _resolve_model_path(
        model_path: str, revision: str | None = None
    ) -> str:
        """Resolve model_path to a local directory."""
        p = Path(model_path)
        if p.is_dir():
            return str(p.resolve())

        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(model_path, revision=revision)
        logger.info("Resolved HF model '%s' -> %s", model_path, local_dir)
        return local_dir

    @staticmethod
    def _open_gds_file(file_path: str) -> int:
        direct_flag = getattr(os, "O_DIRECT", 0)
        if direct_flag == 0:
            raise RuntimeError("O_DIRECT is not available on this platform")
        try:
            return os.open(file_path, os.O_RDONLY | direct_flag)
        except OSError as exc:
            raise RuntimeError(
                f"failed to open {file_path!r} for GDS read: {exc}"
            ) from exc

    def _ensure_gds_manager(self) -> None:
        """Lazily create and initialize the GDS transfer manager."""
        if self._gds_manager is not None:
            return

        agent_name = f"mx-gds-{self._device_id}-{uuid.uuid4().hex[:8]}"
        self._gds_manager = GdsTransferManager(
            agent_name=agent_name,
            accelerator_backend=self._accelerator_backend,
        )
        self._gds_manager.initialize()
        logger.info("GDS manager initialized for device %d", self._device_id)

    def _resolve_safetensors_files(
        self, model_path: str
    ) -> dict[str, list[str]]:
        """
        Discover safetensors files and map each to its tensor names.

        Supports sharded (index.json) and single-file layouts.
        """
        model_dir = Path(model_path)

        # Try sharded index first
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                index = json.load(f)

            weight_map: dict[str, str] = index.get("weight_map", {})
            if not weight_map:
                raise RuntimeError(f"Empty weight_map in {index_path}")

            file_tensors: dict[str, list[str]] = defaultdict(list)
            for tensor_name, filename in weight_map.items():
                abs_path = str(model_dir / filename)
                file_tensors[abs_path].append(tensor_name)

            logger.info(
                "Found sharded model: %d files, %d tensors",
                len(file_tensors), len(weight_map),
            )
            return dict(file_tensors)

        # Try single file
        single_path = model_dir / "model.safetensors"
        if single_path.exists():
            header_info = self._parse_safetensors_header(str(single_path))
            tensor_names = list(header_info.keys())
            logger.info("Found single safetensors file: %d tensors", len(tensor_names))
            return {str(single_path): tensor_names}

        # Fallback: glob
        st_files = sorted(model_dir.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(f"No .safetensors files found in {model_path}")

        file_tensors_map: dict[str, list[str]] = {}
        for st_file in st_files:
            header_info = self._parse_safetensors_header(str(st_file))
            file_tensors_map[str(st_file)] = list(header_info.keys())

        total = sum(len(v) for v in file_tensors_map.values())
        logger.info(
            "Found %d safetensors files via glob: %d tensors",
            len(file_tensors_map), total,
        )
        return file_tensors_map

    def _parse_safetensors_header(self, file_path: str) -> dict[str, dict]:
        """
        Parse a safetensors file header without loading tensor data.

        Returns:
            {tensor_name: {"file_offset": int, "size": int, "dtype": str, "shape": list}}
        """
        with open(file_path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                raise RuntimeError(f"Invalid safetensors file: {file_path}")

            header_size = struct.unpack("<Q", raw)[0]

            if header_size > 100 * 1024 * 1024:
                raise RuntimeError(
                    f"Safetensors header too large ({header_size} bytes): {file_path}"
                )

            header_bytes = f.read(header_size)

        header = json.loads(header_bytes)
        data_start = 8 + header_size

        result: dict[str, dict] = {}
        for name, info in header.items():
            if name == "__metadata__":
                continue

            offsets = info["data_offsets"]
            result[name] = {
                "file_offset": data_start + offsets[0],
                "size": offsets[1] - offsets[0],
                "dtype": info["dtype"],
                "shape": info["shape"],
            }

        return result

    def _load_file_tensors(
        self,
        file_path: str,
        tensor_infos: dict[str, dict],
    ) -> dict[str, torch.Tensor]:
        """
        Load all tensors from one safetensors file via GDS.

        All tensors are submitted in a single batch so GDS_MT threads
        work in parallel. Reads go directly into result tensors.
        """
        device = self._accelerator_backend.torch_device(self._device_id)

        sorted_names = sorted(
            tensor_infos.keys(),
            key=lambda n: tensor_infos[n]["file_offset"],
        )

        tensor_list = []
        tensor_meta = []
        for name in sorted_names:
            info = tensor_infos[name]
            st_dtype = info["dtype"]
            torch_dtype = SAFETENSORS_DTYPE_MAP.get(st_dtype)
            if torch_dtype is None:
                raise RuntimeError(
                    f"Unsupported safetensors dtype '{st_dtype}' "
                    f"for tensor '{name}'"
                )
            tensor_list.append((info["file_offset"], info["size"]))
            tensor_meta.append((name, torch_dtype, info["shape"]))

        fd = os.open(file_path, os.O_RDONLY)
        file_size = os.fstat(fd).st_size

        try:
            raw_tensors = self._gds_manager.batch_load_file(
                fd, file_size, tensor_list, device,
            )
        finally:
            os.close(fd)

        result: dict[str, torch.Tensor] = {}
        for raw, (name, torch_dtype, shape) in zip(raw_tensors, tensor_meta, strict=True):
            result[name] = raw.view(torch_dtype).reshape(shape)

        logger.info("Loaded %s", Path(file_path).name)
        return result

    def shutdown(self) -> None:
        """Release GDS resources."""
        if self._gds_manager is not None:
            self._gds_manager.shutdown()
            self._gds_manager = None
