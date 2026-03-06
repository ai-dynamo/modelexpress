# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Framework-agnostic GDS Model Loader.

Loads model weights from safetensors files directly to GPU memory via NIXL's
GDS (GPUDirect Storage) backend, bypassing CPU bounce buffers entirely.

This is the public API — usable from vLLM, sglang, or any other framework:

    from modelexpress.gds_loader import MxGdsLoader

    loader = MxGdsLoader()
    tensors = loader.load("/path/to/model")

The target GPU is determined automatically from ``torch.cuda.current_device()``,
matching the behavior of vLLM/sglang's default loaders — the calling framework
is responsible for setting up the CUDA device context.
"""

from __future__ import annotations

import json
import logging
import os
import struct
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import torch

from .gds_transfer import GdsTransferManager, is_gds_available

logger = logging.getLogger("modelexpress.gds_loader")

# Safetensors dtype string → torch dtype mapping.
# Reference: https://huggingface.co/docs/safetensors/metadata_parsing
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


class MxGdsLoader:
    """
    Load model weights from safetensors files directly to GPU via GDS.

    Framework-agnostic.  Can be used from vLLM, sglang, or any other framework.

    The target GPU is inferred from ``torch.cuda.current_device()`` at the
    time ``load()`` / ``load_iter()`` is called, so the calling framework
    controls device placement through its normal CUDA context — no explicit
    ``device_id`` parameter is needed (same pattern as vLLM / sglang default
    loaders).

    Usage::

        loader = MxGdsLoader()

        # Load all tensors at once
        tensors = loader.load("/path/to/model")

        # Or stream tensors per-file (memory-efficient for large models)
        for name, tensor in loader.load_iter("/path/to/model"):
            process(name, tensor)
    """

    def __init__(self):
        self._gds_manager: GdsTransferManager | None = None
        self._device_id: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, model_path: str) -> dict[str, torch.Tensor]:
        """
        Load all tensors from *model_path* to GPU.

        Args:
            model_path: Directory containing ``.safetensors`` files
                (HuggingFace model layout).

        Returns:
            Dictionary mapping tensor names to GPU tensors.
        """
        result: dict[str, torch.Tensor] = {}
        for name, tensor in self.load_iter(model_path):
            result[name] = tensor
        return result

    def load_iter(
        self, model_path: str
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """
        Yield ``(tensor_name, gpu_tensor)`` pairs loaded via GDS.

        Each ``.safetensors`` file is batch-loaded through a single GDS
        transfer, then its tensors are yielded one by one.  This is the
        preferred API for frameworks that consume weight iterators (e.g.
        vLLM's ``model.load_weights()``).

        The target GPU is taken from ``torch.cuda.current_device()`` on the
        first call.  The calling framework sets the device context.

        Args:
            model_path: Directory containing ``.safetensors`` files.

        Yields:
            ``(name, tensor)`` on the current CUDA device.
        """
        load_start = time.perf_counter()
        model_path = self._resolve_model_path(model_path)

        if not is_gds_available():
            raise RuntimeError(
                "NIXL is not available. GDS loading requires nixl with GDS "
                "support. Install with: pip install nixl[cu12]"
            )

        # Resolve device from current CUDA context (set by the framework)
        self._device_id = torch.cuda.current_device()

        # Initialize GDS manager (once, reused across files)
        self._ensure_gds_manager()

        # Discover safetensors files and their tensor mappings
        file_tensor_map = self._resolve_safetensors_files(model_path)

        for file_path, tensor_names in file_tensor_map.items():
            # Parse header to get offsets/sizes/dtypes/shapes
            header_info = self._parse_safetensors_header(file_path)

            # Filter to only the tensors in this file
            file_tensors = {
                name: header_info[name]
                for name in tensor_names
                if name in header_info
            }

            if not file_tensors:
                continue

            # Batch-load all tensors from this file via GDS
            loaded = self._load_file_tensors(file_path, file_tensors)
            for name, tensor in loaded.items():
                yield name, tensor

        logger.info("GDS load complete in %.2fs", time.perf_counter() - load_start)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        """Resolve *model_path* to a local directory.

        If it already exists on disk, return it directly.
        Otherwise treat it as a HuggingFace model ID and locate (or
        download) the snapshot via ``huggingface_hub``.
        """
        p = Path(model_path)
        if p.is_dir():
            return str(p.resolve())

        # HuggingFace model ID → local cache
        from huggingface_hub import snapshot_download
        local_dir = snapshot_download(model_path)
        logger.info("Resolved HF model '%s' → %s", model_path, local_dir)
        return local_dir

    def _ensure_gds_manager(self) -> None:
        """Lazily create and initialize the GDS transfer manager."""
        if self._gds_manager is not None:
            return

        agent_name = f"mx-gds-{self._device_id}-{uuid.uuid4().hex[:8]}"
        self._gds_manager = GdsTransferManager(agent_name=agent_name)
        self._gds_manager.initialize()
        logger.info(
            "GDS manager initialized for device %d", self._device_id
        )

    def _resolve_safetensors_files(
        self, model_path: str
    ) -> dict[str, list[str]]:
        """
        Discover safetensors files and map each to its tensor names.

        Supports two HuggingFace layouts:
        1. Sharded: ``model.safetensors.index.json`` present
        2. Single: one ``model.safetensors`` file

        Returns:
            ``{absolute_file_path: [tensor_name, ...]}``
        """
        model_dir = Path(model_path)

        # Try sharded index first
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                index = json.load(f)

            weight_map: dict[str, str] = index.get("weight_map", {})
            if not weight_map:
                raise RuntimeError(
                    f"Empty weight_map in {index_path}"
                )

            # Invert: filename → [tensor_names]
            file_tensors: dict[str, list[str]] = defaultdict(list)
            for tensor_name, filename in weight_map.items():
                abs_path = str(model_dir / filename)
                file_tensors[abs_path].append(tensor_name)

            logger.info(
                "Found sharded model: %d files, %d tensors",
                len(file_tensors),
                len(weight_map),
            )
            return dict(file_tensors)

        # Try single file
        single_path = model_dir / "model.safetensors"
        if single_path.exists():
            header_info = self._parse_safetensors_header(str(single_path))
            tensor_names = list(header_info.keys())
            logger.info(
                "Found single safetensors file: %d tensors", len(tensor_names)
            )
            return {str(single_path): tensor_names}

        # Fallback: glob for any .safetensors files
        st_files = sorted(model_dir.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(
                f"No .safetensors files found in {model_path}"
            )

        file_tensors = {}
        for st_file in st_files:
            header_info = self._parse_safetensors_header(str(st_file))
            file_tensors[str(st_file)] = list(header_info.keys())

        total = sum(len(v) for v in file_tensors.values())
        logger.info(
            "Found %d safetensors files via glob: %d tensors",
            len(file_tensors),
            total,
        )
        return file_tensors

    def _parse_safetensors_header(
        self, file_path: str
    ) -> dict[str, dict]:
        """
        Parse a safetensors file header without loading tensor data.

        Safetensors format:
        - 8 bytes: ``header_size`` (uint64 LE)
        - ``header_size`` bytes: JSON header
        - Remaining: contiguous tensor data

        The JSON header maps tensor names to::

            {
                "dtype": "BF16",
                "shape": [4096, 4096],
                "data_offsets": [start, end]  # relative to data section
            }

        Returns:
            ``{tensor_name: {"file_offset": int, "size": int,
            "dtype": str, "shape": list}}``
        """
        with open(file_path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8:
                raise RuntimeError(f"Invalid safetensors file: {file_path}")

            header_size = struct.unpack("<Q", raw)[0]

            # Sanity check — header shouldn't exceed 100 MB
            if header_size > 100 * 1024 * 1024:
                raise RuntimeError(
                    f"Safetensors header too large ({header_size} bytes): "
                    f"{file_path}"
                )

            header_bytes = f.read(header_size)

        header = json.loads(header_bytes)
        data_start = 8 + header_size  # absolute offset of tensor data

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
        work in parallel.  No intermediate staging buffer — reads go
        directly into the result tensors.

        Args:
            file_path: Path to the ``.safetensors`` file.
            tensor_infos: ``{name: {"file_offset", "size", "dtype", "shape"}}``
                from ``_parse_safetensors_header()``.

        Returns:
            ``{tensor_name: gpu_tensor}``
        """
        device = torch.device("cuda", self._device_id)

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
        for raw, (name, torch_dtype, shape) in zip(raw_tensors, tensor_meta):
            result[name] = raw.view(torch_dtype).reshape(shape)

        logger.info("Loaded %s", Path(file_path).name)
        return result

    def shutdown(self) -> None:
        """Release GDS resources."""
        if self._gds_manager is not None:
            self._gds_manager.shutdown()
            self._gds_manager = None
