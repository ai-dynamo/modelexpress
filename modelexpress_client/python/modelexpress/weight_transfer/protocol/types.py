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
    """One trainer rank's ownership of a row range within a parameter tensor.

    Attributes:
        agent_index: Index into TrainerTable.agents (NIXL metadata blob).
        row_start: First dim-0 row this rank owns (inclusive).
        row_end: First dim-0 row this rank does NOT own (exclusive).
        device_addr: GPU virtual address of row_start in trainer memory.
        row_bytes: Bytes per row = prod(shape[1:]) * elem_size.
        device_id: CUDA device index on the trainer node.
    """

    agent_index: int
    row_start: int
    row_end: int
    device_addr: int
    row_bytes: int
    device_id: int

    @property
    def num_rows(self) -> int:
        return self.row_end - self.row_start

    @property
    def size_bytes(self) -> int:
        return self.num_rows * self.row_bytes


@dataclass
class TrainerTensor:
    """All shard descriptors for one parameter tensor across trainer ranks.

    Shards are non-overlapping and together cover [0, shape[0]).
    """

    name: str
    dtype: str          # e.g. "torch.bfloat16"
    shape: list[int]
    shards: list[TrainerShard] = field(default_factory=list)

    @property
    def num_rows(self) -> int:
        return self.shape[0] if self.shape else 0

    def shard_for_row(self, row: int) -> TrainerShard | None:
        for s in self.shards:
            if s.row_start <= row < s.row_end:
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
