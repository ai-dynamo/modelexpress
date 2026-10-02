# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded local replay of draft tensors captured during a cold source load."""

from __future__ import annotations

import atexit
import shutil
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


class SpoolLimitExceeded(RuntimeError):
    """The configured local draft spool cannot hold another tensor."""


class DraftSpool:
    """Retain raw draft tensors until SGLang invokes its second load pass."""

    def __init__(self, root: str | Path, *, max_bytes: int, memory_bytes: int):
        if max_bytes <= 0 or memory_bytes < 0:
            raise ValueError("draft spool limits must be positive")
        self.path = Path(tempfile.mkdtemp(prefix="mx-draft-", dir=root))
        self._max_bytes = max_bytes
        self._memory_bytes = min(memory_bytes, max_bytes)
        self._bytes = 0
        self._in_memory_bytes = 0
        self._entries: list[tuple[str, torch.Tensor | Path]] = []
        self._complete = False
        self._closed = False

    def capture(self, name: str, tensor: torch.Tensor) -> None:
        if self._complete or self._closed:
            raise RuntimeError("draft spool is no longer writable")
        size = tensor.numel() * tensor.element_size()
        if self._bytes + size > self._max_bytes:
            raise SpoolLimitExceeded("draft spool byte limit exceeded")
        copied = tensor.detach().to(device="cpu", copy=True).contiguous()
        if self._in_memory_bytes + size <= self._memory_bytes:
            self._entries.append((name, copied))
            self._in_memory_bytes += size
        else:
            file_path = self.path / f"{len(self._entries):08d}.safetensors"
            save_file({"weight": copied}, str(file_path))
            self._entries.append((name, file_path))
        self._bytes += size

    def finish(self) -> None:
        if self._closed or not self._entries:
            raise RuntimeError("draft spool has no draft tensors")
        self._complete = True

    def replay(self) -> Iterator[tuple[str, torch.Tensor]]:
        if not self._complete or self._closed:
            raise RuntimeError("draft spool is not complete")
        for name, stored in self._entries:
            if isinstance(stored, Path):
                yield name, load_file(str(stored), device="cpu")["weight"]
            else:
                yield name, stored

    def validate(self) -> None:
        """Check disk entries before SGLang mutates the draft model."""
        if not self._complete or self._closed:
            raise RuntimeError("draft spool is not complete")
        for _, stored in self._entries:
            if isinstance(stored, Path):
                load_file(str(stored), device="cpu")["weight"]

    @property
    def size_bytes(self) -> int:
        return self._bytes

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._entries.clear()
        shutil.rmtree(self.path, ignore_errors=True)


_ready_spools: dict[tuple[bytes, int, int, str], DraftSpool] = {}
_spool_lock = threading.Lock()


def store_ready_spool(key: tuple[bytes, int, int, str], spool: DraftSpool) -> None:
    with _spool_lock:
        old = _ready_spools.get(key)
        _ready_spools[key] = spool
    if old is not None:
        old.close()


def take_ready_spool(key: tuple[bytes, int, int, str]) -> DraftSpool | None:
    with _spool_lock:
        return _ready_spools.pop(key, None)


def _close_all_spools() -> None:
    with _spool_lock:
        spools = list(_ready_spools.values())
        _ready_spools.clear()
    for spool in spools:
        spool.close()


atexit.register(_close_all_spools)
