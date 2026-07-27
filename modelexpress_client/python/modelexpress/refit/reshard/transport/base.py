# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Transport abstraction for the reshard pull.

A ``Transport`` executes a batch of one-sided READs: for each descriptor, copy
``nbytes`` from a remote source (``session`` + absolute ``src_addr``) into the
local destination (absolute ``dst_addr``). This is exactly the shape a NIXL
READ descriptor list takes, so the planning/pull code is transport-neutral - the
real NIXL agent (``nixl.py``) is one implementation, the in-memory reference
below is another.

``InMemoryReferenceTransport`` performs the moves with ``ctypes.memmove`` over
real (CPU) addresses. It is a faithful stand-in for validation: a transfer plan
that reconstructs the right bytes here reconstructs the right bytes on the wire,
because both consume the identical ``(src_addr, dst_addr, nbytes)`` triples.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class ReadDescriptor:
    """One remote->local READ: copy ``nbytes`` from ``session``'s ``src_addr``
    to the local ``dst_addr``. Addresses are absolute (shard base + offset for
    the source; param ``data_ptr()`` + dest byte offset for the destination)."""

    session: str
    src_addr: int
    dst_addr: int
    nbytes: int


@runtime_checkable
class Transport(Protocol):
    """Executes a batch of one-sided READs. Implementations may group by
    ``session`` (one remote endpoint / NIXL agent per session) internally."""

    def read(self, descriptors: list) -> None: ...


class InMemoryReferenceTransport:
    """Reference ``Transport``: raw local byte moves via ``ctypes.memmove`` over
    real CPU addresses. ``session`` is ignored (all addresses are absolute and
    local). Used by tests and for local correctness verification; not for the
    wire. Callers must keep the backing tensors alive for the addresses to stay
    valid across the ``read`` call."""

    def __init__(self) -> None:
        self.bytes_moved = 0
        self.reads_issued = 0

    def read(self, descriptors: list) -> None:
        for d in descriptors:
            if d.nbytes:
                ctypes.memmove(d.dst_addr, d.src_addr, d.nbytes)
                self.bytes_moved += d.nbytes
                self.reads_issued += 1
