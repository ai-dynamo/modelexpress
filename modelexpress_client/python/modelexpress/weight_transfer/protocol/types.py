# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core data types for trainer-inference weight synchronization.

All types are pure Python dataclasses with no torch dependency so they can
be instantiated on any process (trainer, inference worker, MX server).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SyncMode(str, Enum):
    PULL = "pull"
    PUSH = "push"


@dataclass
class TrainerShard:
    """One trainer rank's ownership of a 2-D tile within a parameter tensor.

    Supports both row-only sharding (FSDP / DP) and 2-D tile sharding
    (TP × FSDP hybrid, MoE expert parallelism).
    """

    agent_index: int
    row_start: int
    row_end: int
    device_addr: int
    row_bytes: int
    device_id: int
    col_start: int = 0
    col_end: int = -1   # -1 = full width; resolved against TrainerTensor.shape

    @property
    def num_rows(self) -> int:
        return self.row_end - self.row_start

    @property
    def size_bytes(self) -> int:
        return self.num_rows * self.row_bytes


@dataclass
class TrainerTensor:
    """All shard descriptors for one parameter tensor across trainer ranks."""

    name: str
    dtype: str          # e.g. "torch.bfloat16"
    shape: list[int]
    shards: list[TrainerShard] = field(default_factory=list)

    @property
    def num_rows(self) -> int:
        return self.shape[0] if self.shape else 0

    def _resolved_col_end(self, shard: TrainerShard) -> int:
        """Return the effective col_end, resolving -1 to the full column count."""
        if shard.col_end == -1:
            return self.shape[1] if len(self.shape) > 1 else 1
        return shard.col_end

    def shard_for_row(self, row: int) -> TrainerShard | None:
        """Return the shard that owns *row* (for row-only sharding)."""
        for s in self.shards:
            if s.row_start <= row < s.row_end:
                return s
        return None

    def shard_for_elem(self, row: int, col: int) -> TrainerShard | None:
        """Return the shard that owns element (row, col); handles row-only and 2-D tiles."""
        for s in self.shards:
            col_end = self._resolved_col_end(s)
            if s.row_start <= row < s.row_end and s.col_start <= col < col_end:
                return s
        return None


@dataclass
class TrainerTable:
    """Complete trainer memory layout for one broadcast step."""

    agents: list[bytes]
    tensors: list[TrainerTensor]
    step: int = 0

    def tensor_by_name(self, name: str) -> TrainerTensor | None:
        for t in self.tensors:
            if t.name == name:
                return t
        return None


@dataclass
class InferenceShard:
    """One inference worker's live parameter memory region (PUSH mode)."""

    agent_index: int
    param_name: str
    device_addr: int
    size_bytes: int
    device_id: int


@dataclass
class InferenceTable:
    """Inference worker GPU memory layout for PUSH-mode weight sync."""

    agents: list[bytes]
    shards: list[InferenceShard]
    worker_rank: int = 0

    def shards_for_param(self, name: str) -> list[InferenceShard]:
        return [s for s in self.shards if s.param_name == name]


@dataclass
class ResolvedRegion:
    """A parameter slice resolved from an op chain to element runs."""

    tensor_name: str
    src_elem_runs: list[int]      # flat: [offset, count, ...]
    dst_addr: int
    dst_elem_runs: list[int]      # flat: [offset, count, ...]
    element_size: int
    dst_device_id: int = 0


@dataclass
class RdmaDescriptor:
    """One NIXL RDMA READ or WRITE descriptor."""

    agent_index: int
    src_addr: int
    dst_addr: int
    nbytes: int


@dataclass
class M2nDescriptor:
    """One descriptor in a globally-coordinated M2N transfer.

    Extends RdmaDescriptor with a destination agent index for the trainer to
    identify which inference worker each descriptor targets.
    """

    src_agent_index: int
    dst_agent_index: int
    src_addr: int
    dst_addr: int
    nbytes: int

    def to_rdma_descriptor(self) -> RdmaDescriptor:
        """Convert to RdmaDescriptor for use with NixlExecutor (PULL path)."""
        return RdmaDescriptor(
            agent_index=self.src_agent_index,
            src_addr=self.src_addr,
            dst_addr=self.dst_addr,
            nbytes=self.nbytes,
        )
