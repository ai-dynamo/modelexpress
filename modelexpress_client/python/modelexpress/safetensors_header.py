# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safetensors header parsing and shard selection shared by the direct-to-GPU loaders.

Parses only the JSON header, never tensor data, so callers can plan byte-range
reads (GDS file reads, NIXL OBJ ranged GETs) before touching the payload.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

import torch

INDEX_FILE = "model.safetensors.index.json"
SINGLE_FILE = "model.safetensors"
# Leading little-endian u64 holding the JSON header length.
HEADER_LENGTH_BYTES = 8
MAX_HEADER_BYTES = 100 * 1024 * 1024


def _optional_dtypes() -> dict[str, torch.dtype]:
    """Dtypes that only exist in newer torch releases."""
    names = {
        "U16": "uint16",
        "U32": "uint32",
        "U64": "uint64",
        "F8_E8M0": "float8_e8m0fnu",
        "F8_E4M3FNUZ": "float8_e4m3fnuz",
        "F8_E5M2FNUZ": "float8_e5m2fnuz",
    }
    return {
        st_name: getattr(torch, torch_name)
        for st_name, torch_name in names.items()
        if hasattr(torch, torch_name)
    }


# Dtype mapping from the safetensors spec:
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
    **_optional_dtypes(),
}


@dataclass(frozen=True)
class TensorEntry:
    """One tensor in a safetensors file; offsets are relative to the data section."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True)
class SafetensorsHeader:
    """Parsed header with absolute data-section bounds within the file."""

    header_len: int
    data_start: int
    data_end: int
    tensors: tuple[TensorEntry, ...]


def header_length(prefix: bytes, source: str, *, max_header_bytes: int = MAX_HEADER_BYTES) -> int:
    """Return the JSON header length encoded in the first 8 bytes of a file."""
    if len(prefix) < HEADER_LENGTH_BYTES:
        raise ValueError(f"Invalid safetensors file (shorter than 8 bytes): {source}")
    (length,) = struct.unpack("<Q", prefix[:HEADER_LENGTH_BYTES])
    if length > max_header_bytes:
        raise ValueError(f"Safetensors header too large ({length} bytes): {source}")
    return length


def torch_dtype(dtype: str, tensor_name: str) -> torch.dtype:
    """Map a safetensors dtype string to a torch dtype."""
    mapped = SAFETENSORS_DTYPE_MAP.get(dtype)
    if mapped is None:
        raise ValueError(f"Unsupported safetensors dtype '{dtype}' for tensor '{tensor_name}'")
    return mapped


def parse_header(header_bytes: bytes, source: str) -> SafetensorsHeader:
    """Parse and validate the JSON header (the bytes after the 8-byte length).

    Validates that offsets are ordered, tensors do not overlap, and each
    tensor's byte size matches its shape and dtype.
    """
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid safetensors header JSON in {source}: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors header is not a JSON object: {source}")

    header_len = len(header_bytes)
    data_start = HEADER_LENGTH_BYTES + header_len
    entries: list[TensorEntry] = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        try:
            dtype = info["dtype"]
            shape = tuple(int(dim) for dim in info["shape"])
            begin, end = (int(offset) for offset in info["data_offsets"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Malformed entry for tensor '{name}' in {source}") from exc
        if begin < 0 or end < begin:
            raise ValueError(f"Invalid data_offsets [{begin}, {end}] for tensor '{name}' in {source}")
        itemsize = torch_dtype(dtype, name).itemsize
        expected = math.prod(shape) * itemsize
        if end - begin != expected:
            raise ValueError(
                f"Tensor '{name}' in {source} spans {end - begin} bytes but "
                f"shape {list(shape)} x {dtype} needs {expected}"
            )
        entries.append(TensorEntry(name=name, dtype=dtype, shape=shape, begin=begin, end=end))

    entries.sort(key=lambda entry: (entry.begin, entry.end))
    for prev, cur in pairwise(entries):
        if cur.begin < prev.end:
            raise ValueError(f"Tensors '{prev.name}' and '{cur.name}' overlap in {source}")

    data_len = max((entry.end for entry in entries), default=0)
    return SafetensorsHeader(
        header_len=header_len,
        data_start=data_start,
        data_end=data_start + data_len,
        tensors=tuple(entries),
    )


def index_weight_map(index: dict, source: str) -> dict[str, str]:
    """Return the ``weight_map`` of a parsed index file, which must not be empty."""
    weight_map = index.get("weight_map") or {}
    if not weight_map:
        raise ValueError(f"Empty weight_map in {source}")
    return weight_map


def select_shards(filenames: Iterable[str], weight_map: dict[str, str] | None) -> list[str]:
    """Pick a checkpoint's shard files, sorted.

    Uses the index ``weight_map`` when there is one, then a single
    ``model.safetensors``, then every ``*.safetensors`` file.
    """
    if weight_map is not None:
        return sorted(set(weight_map.values()))
    names = set(filenames)
    if SINGLE_FILE in names:
        return [SINGLE_FILE]
    return sorted(name for name in names if name.endswith(".safetensors"))
