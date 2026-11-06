# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the ModelExpress RL workflow."""

from __future__ import annotations

import functools
import json
import struct
import threading
import zlib
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

import numpy as np


_WorkItem = TypeVar("_WorkItem")
_WorkResult = TypeVar("_WorkResult")


_ADLER32_BASE = 65521


class Adler32Checksum:
    def __init__(self) -> None:
        self._value = 1

    def update(self, data: Any) -> None:
        self._value = zlib.adler32(data, self._value)

    def extend(self, following: Adler32Checksum, length: int) -> None:
        """Append the ``length`` bytes that ``following`` checksummed from a fresh start.

        This is zlib's ``adler32_combine``, so parts can be checksummed in parallel.
        """
        remainder = length % _ADLER32_BASE
        low = self._value & 0xFFFF
        high = (remainder * low) % _ADLER32_BASE
        low += (following._value & 0xFFFF) + _ADLER32_BASE - 1
        high += (
            ((self._value >> 16) & 0xFFFF)
            + ((following._value >> 16) & 0xFFFF)
            + _ADLER32_BASE
            - remainder
        )
        self._value = (low % _ADLER32_BASE) | ((high % _ADLER32_BASE) << 16)

    def hexdigest(self) -> str:
        return f"{self._value:08x}"


def checksum_factory(checksum_format: str) -> Adler32Checksum:
    if checksum_format == "adler32":
        return Adler32Checksum()
    raise ValueError(f"unsupported checksum format {checksum_format!r}")


def threadpool_map(
    items: Iterable[_WorkItem],
    process: Callable[[_WorkItem], _WorkResult],
    *,
    max_workers: int,
    thread_name_prefix: str,
) -> Iterator[_WorkResult]:
    """Map work in a bounded thread pool while preserving input order."""
    inflight = deque()
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix=thread_name_prefix,
    ) as pool:
        for item in items:
            inflight.append(pool.submit(process, item))
            if len(inflight) >= 2 * max_workers:
                yield inflight.popleft().result()
        while inflight:
            yield inflight.popleft().result()


def compute_delta(
    current: np.ndarray, base: np.ndarray
) -> tuple[np.ndarray | None, int]:
    """Compute a full-size bytewise XOR delta."""
    if len(current) != len(base):
        raise RuntimeError("tensor changed byte size")
    delta = np.bitwise_xor(current, base)
    changed_bytes = int(np.count_nonzero(delta))
    return (delta if changed_bytes else None), changed_bytes


def compress_delta(delta: np.ndarray) -> np.ndarray:
    """Compress one XOR delta as an independent level-1 zstd frame."""
    import zstandard

    return np.frombuffer(
        zstandard.ZstdCompressor(level=1).compress(delta),
        dtype=np.uint8,
    )


def _tied_names(root: Path) -> set[str]:
    config = root / "config.json"
    if not config.is_file():
        return set()
    try:
        tied = json.loads(config.read_text()).get("tie_word_embeddings", False)
    except (OSError, ValueError):
        return set()
    return {"lm_head.weight"} if tied else set()


@functools.lru_cache(maxsize=16)
def _weight_files(index: bytes) -> tuple[str, ...]:
    return tuple(sorted(set(json.loads(index)["weight_map"].values())))


def _checkpoint_paths(root: Path) -> list[Path]:
    if root.is_file():
        paths = [root]
    else:
        index = root / "model.safetensors.index.json"
        if index.is_file():
            paths = [root / name for name in _weight_files(index.read_bytes())]
        else:
            paths = sorted(root.glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no safetensors files found under {root}")
    return paths


def _read_header_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        header_size_data = handle.read(8)
        if len(header_size_data) != 8:
            raise ValueError(f"invalid safetensors header in {path}")
        (header_size,) = struct.unpack("<Q", header_size_data)
        return header_size_data + handle.read(header_size)


# One parsed header per file name. Deltas rewrite tensor bytes but never headers,
# so every version of a checkpoint reuses the same parse.
_HeaderEntries = tuple[tuple[str, int, int, dict], ...]
_parsed_headers: dict[str, tuple[bytes, int, _HeaderEntries]] = {}
_merged_index: dict[str, Any] = {}
_index_lock = threading.Lock()


def _parse_header(path: Path) -> tuple[int, _HeaderEntries]:
    """Return a file's data-region offset and ``(name, begin, size, metadata)``."""
    data = _read_header_bytes(path)
    with _index_lock:
        cached = _parsed_headers.get(path.name)
    if cached is not None and cached[0] == data:
        return cached[1], cached[2]
    header, data_start = read_safetensors_header(data, str(path))
    entries = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        begin, end = info["data_offsets"]
        entries.append(
            (
                name,
                begin,
                end - begin,
                {
                    "name": name,
                    "shape": info["shape"],
                    "dtype": info["dtype"],
                    "byte_size": end - begin,
                },
            )
        )
    parsed = tuple(entries)
    with _index_lock:
        _parsed_headers[path.name] = (data, data_start, parsed)
    return data_start, parsed


def read_safetensors_header(data: bytes, source: str) -> tuple[dict, int]:
    """Parse a safetensors header and return it with the data-region offset."""
    if len(data) < 8:
        raise ValueError(f"invalid safetensors header in {source}")
    (header_size,) = struct.unpack("<Q", data[:8])
    data_start = 8 + header_size
    if data_start > len(data):
        raise ValueError(f"invalid safetensors header in {source}")
    try:
        header = json.loads(data[8:data_start])
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError(f"invalid safetensors header in {source}") from error
    if not isinstance(header, dict):
        raise ValueError(f"invalid safetensors header in {source}")
    return header, data_start


def _index_tensors(
    paths: list[Path], tied: set[str]
) -> tuple[dict[str, tuple[Path, int, int]], dict[str, dict]]:
    """Return byte locations and metadata for every checkpoint tensor.

    Unchanged headers return the previous metadata object, and the previous
    locations too when the paths are also unchanged, so callers must not mutate
    either.
    """
    parsed = [_parse_header(path) for path in paths]
    entries = tuple(file_entries for _start, file_entries in parsed)
    tied_names = frozenset(tied)
    with _index_lock:
        cached = _merged_index.get("value")
    same_headers = (
        cached is not None
        and cached[1] == tied_names
        and len(cached[0]) == len(entries)
        and all(old is new for old, new in zip(cached[0], entries))
    )
    if same_headers and cached[2] == paths:
        return cached[3], cached[4]

    locations: dict[str, tuple[Path, int, int]] = {}
    metadata: dict[str, dict] = cached[4] if same_headers else {}
    for path, (data_start, file_entries) in zip(paths, parsed):
        for name, begin, size, info in file_entries:
            if name in tied:
                continue
            if name in locations:
                raise ValueError(f"duplicate safetensors entry {name!r}")
            locations[name] = (path, data_start + begin, size)
            if not same_headers:
                metadata[name] = info
    with _index_lock:
        _merged_index["value"] = (entries, tied_names, list(paths), locations, metadata)
    return locations, metadata


def make_tensor_reader(
    checkpoint: str | Path,
) -> tuple[Callable[[str], np.ndarray], dict[str, dict]]:
    """Index safetensors once and return direct byte reads by tensor name."""
    _, locations, metadata = index_checkpoint_tensors(checkpoint)

    def read(name: str) -> np.ndarray:
        path, offset, size = locations[name]
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(size)
        if len(data) != size:
            raise ValueError(f"short safetensors read for {name!r}")
        return np.frombuffer(data, dtype=np.uint8)

    return read, metadata


def index_checkpoint_tensors(
    checkpoint: str | Path,
) -> tuple[list[Path], dict[str, tuple[Path, int, int]], dict[str, dict]]:
    """Return safetensors paths, byte locations, and metadata for a checkpoint."""
    root = Path(checkpoint)
    paths = _checkpoint_paths(root)
    tied = _tied_names(root if root.is_dir() else root.parent)
    locations, metadata = _index_tensors(paths, tied)
    return paths, locations, metadata


__all__ = [
    "Adler32Checksum",
    "checksum_factory",
    "compress_delta",
    "compute_delta",
    "index_checkpoint_tensors",
    "make_tensor_reader",
    "read_safetensors_header",
    "threadpool_map",
]
