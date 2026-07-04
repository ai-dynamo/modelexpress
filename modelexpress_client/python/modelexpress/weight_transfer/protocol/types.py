# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core data types for trainer-inference weight synchronization.

All types are pure Python dataclasses with no torch dependency so they can
be instantiated on any process (trainer, inference worker, MX server).

Two sync directions are supported:

  PULL -- inference workers read from trainer GPU memory (trainer is passive).
          Described by TrainerTable (trainer publishes) + pull plan.

  PUSH -- trainer writes directly into inference worker GPU memory (workers
          are passive).  Described by InferenceTable (workers publish) + push
          plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SyncMode(str, Enum):
    PULL = "pull"
    PUSH = "push"


# ---------------------------------------------------------------------------
# Trainer side (PULL source / PUSH sender)
# ---------------------------------------------------------------------------


@dataclass
class TrainerShard:
    """One trainer rank's ownership of a 2-D tile within a parameter tensor.

    Supports both row-only sharding (FSDP / DP) and 2-D tile sharding
    (TP × FSDP hybrid, MoE expert parallelism).

    Attributes:
        agent_index: Index into TrainerTable.agents (NIXL metadata blob).
        row_start: First dim-0 row this rank owns (inclusive).
        row_end: First dim-0 row this rank does NOT own (exclusive).
        device_addr: GPU virtual address of element [row_start, col_start]
            in trainer memory.  For row-only shards this equals the address
            of the first owned row.
        row_bytes: Bytes per **shard row** = (col_end - col_start) * elem_size.
            For row-only shards this equals prod(shape[1:]) * elem_size.
        device_id: CUDA device index on the trainer node.
        col_start: First dim-1 column this rank owns (inclusive).
            Defaults to 0 (full-width shard — backwards compatible).
        col_end: First dim-1 column this rank does NOT own (exclusive).
            -1 means "full width" and is resolved by TrainerTensor at
            query time.  Defaults to -1 (backwards compatible).

    Sharding modes
    --------------
    Row-only (FSDP / DP):
        col_start=0, col_end=-1 (or full width).  Each shard spans all
        columns for a contiguous range of rows.

    Column-only (row-parallel TP):
        row_start=0, row_end=full_rows.  Each shard spans all rows for a
        contiguous range of columns.  Arises in Megatron-LM row-parallel
        layers (o_proj, down_proj).

    2-D tile (FSDP × TP hybrid):
        Each rank owns a rectangular tile [row_start:row_end, col_start:col_end].
        device_addr points to element [row_start, col_start] in shard-local
        storage, and row_bytes is the width of the tile in bytes.
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
    """All shard descriptors for one parameter tensor across trainer ranks.

    For row-only sharding shards together cover [0, shape[0]).
    For 2-D sharding shards tile the full [0, shape[0]) × [0, shape[1]) space.
    """

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
        """Return the shard that owns element (row, col) in the full tensor.

        Handles both row-only shards (col_start=0, col_end=-1) and 2-D tiles.
        """
        for s in self.shards:
            col_end = self._resolved_col_end(s)
            if s.row_start <= row < s.row_end and s.col_start <= col < col_end:
                return s
        return None


@dataclass
class TrainerTable:
    """Complete trainer memory layout for one broadcast step.

    Attributes:
        agents: NIXL metadata blobs, one per trainer rank.
        tensors: Parameter tensors with their shard maps.
        step: Training step index (used to detect stale plans).
    """

    agents: list[bytes]
    tensors: list[TrainerTensor]
    step: int = 0

    def tensor_by_name(self, name: str) -> TrainerTensor | None:
        for t in self.tensors:
            if t.name == name:
                return t
        return None


# ---------------------------------------------------------------------------
# Inference side (PUSH target)
# ---------------------------------------------------------------------------


@dataclass
class InferenceShard:
    """One inference worker's ownership of a live parameter memory region.

    Used in PUSH mode: the trainer writes directly into this address range.

    Attributes:
        agent_index: Index into InferenceTable.agents.
        param_name: Parameter name in the model (HuggingFace convention).
        device_addr: GPU virtual address of the parameter's storage start.
        size_bytes: Total bytes of the parameter tensor.
        device_id: CUDA device index on the inference node.
    """

    agent_index: int
    param_name: str
    device_addr: int
    size_bytes: int
    device_id: int


@dataclass
class InferenceTable:
    """Inference worker GPU memory layout for PUSH-mode weight sync.

    Published by inference workers so the trainer can write directly into
    their live parameter memory without a separate gather/scatter step.
    """

    agents: list[bytes]
    shards: list[InferenceShard]
    worker_rank: int = 0

    def shards_for_param(self, name: str) -> list[InferenceShard]:
        return [s for s in self.shards if s.param_name == name]


# ---------------------------------------------------------------------------
# Plan types (produced by planner, consumed by transport)
# ---------------------------------------------------------------------------


@dataclass
class ResolvedRegion:
    """A parameter slice resolved from an op chain to element runs.

    Produced by the client-side resolver (torch-dependent) and sent to
    the server (or local router) for final address computation.

    Attributes:
        tensor_name: Trainer tensor name (pre-HF-rename).
        src_elem_runs: Flat list [offset, count, offset, count, ...] of
            contiguous element runs in trainer tensor storage.
        dst_addr: Destination GPU address (vLLM parameter data_ptr).
        dst_elem_runs: Flat list of element runs in destination storage.
        element_size: Bytes per element (e.g. 2 for bfloat16).
        dst_device_id: CUDA device index of the destination.
    """

    tensor_name: str
    src_elem_runs: list[int]      # flat: [offset, count, ...]
    dst_addr: int
    dst_elem_runs: list[int]      # flat: [offset, count, ...]
    element_size: int
    dst_device_id: int = 0


@dataclass
class RdmaDescriptor:
    """One NIXL RDMA READ or WRITE descriptor.

    Attributes:
        agent_index: Index into TrainerTable.agents or InferenceTable.agents.
        src_addr: Source GPU virtual address.
        dst_addr: Destination GPU virtual address.
        nbytes: Transfer size in bytes.
    """

    agent_index: int
    src_addr: int
    dst_addr: int
    nbytes: int


@dataclass
class M2nDescriptor:
    """One descriptor in a globally-coordinated M2N transfer.

    Extends RdmaDescriptor with a destination agent index so the trainer side
    can identify which inference worker each descriptor targets.

    Attributes:
        src_agent_index: Index into TrainerTable.agents (trainer rank).
        dst_agent_index: Index identifying the target inference worker.
        src_addr: Source GPU virtual address on the trainer.
        dst_addr: Destination GPU virtual address on the inference worker.
        nbytes: Transfer size in bytes.
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
