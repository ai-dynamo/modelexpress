# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for the ModelExpress RL workflow."""

from __future__ import annotations

import json
import struct
import zlib
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

import numpy as np


_WorkItem = TypeVar("_WorkItem")
_WorkResult = TypeVar("_WorkResult")


class Adler32Checksum:
    def __init__(self) -> None:
        self._value = 1

    def update(self, data: Any) -> None:
        self._value = zlib.adler32(data, self._value)

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


def _checkpoint_paths(root: Path) -> list[Path]:
    if root.is_file():
        paths = [root]
    else:
        index = root / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text())["weight_map"]
            paths = [root / name for name in sorted(set(weight_map.values()))]
        else:
            paths = sorted(root.glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"no safetensors files found under {root}")
    return paths


def _read_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as handle:
        header_size_data = handle.read(8)
        if len(header_size_data) != 8:
            raise ValueError(f"invalid safetensors header in {path}")
        (header_size,) = struct.unpack("<Q", header_size_data)
        header, _ = read_safetensors_header(
            header_size_data + handle.read(header_size),
            str(path),
        )
        return header_size, header


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
    """Return byte locations and metadata for every checkpoint tensor."""

    locations: dict[str, tuple[Path, int, int]] = {}
    metadata: dict[str, dict] = {}
    for path in paths:
        header_size, header = _read_header(path)
        for source_name, info in header.items():
            if source_name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            name = source_name
            if name in tied:
                continue
            if name in locations:
                raise ValueError(f"duplicate safetensors entry {name!r}")
            locations[name] = (path, 8 + header_size + begin, end - begin)
            metadata[name] = {
                "name": name,
                "shape": info["shape"],
                "dtype": info["dtype"],
                "byte_size": end - begin,
            }
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
