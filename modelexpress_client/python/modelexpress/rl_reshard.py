# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-agnostic RL reshard planning primitives.

This module intentionally does not move tensor bytes. It owns the pure
selection/mapping step between source tensor shards and receiver buffers so
framework adapters can provide tensor handles without embedding layout policy.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TensorShardSpec:
    """One source tensor shard advertised by a trainer or inference replica."""

    name: str
    worker_rank: int
    shape: tuple[int, ...]
    dtype: str
    tensor_parallel_rank: int = 0
    pipeline_parallel_rank: int = 0
    expert_ids: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        object.__setattr__(self, "dtype", str(self.dtype))
        object.__setattr__(self, "expert_ids", frozenset(int(expert) for expert in self.expert_ids))


@dataclass(frozen=True)
class TensorReceiveSpec:
    """One receiver tensor buffer that needs a compatible source shard."""

    name: str
    receiver_rank: int
    shape: tuple[int, ...]
    dtype: str
    tensor_parallel_rank: int = 0
    pipeline_parallel_rank: int = 0
    expert_ids: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        object.__setattr__(self, "dtype", str(self.dtype))
        object.__setattr__(self, "expert_ids", frozenset(int(expert) for expert in self.expert_ids))


@dataclass(frozen=True)
class TransferPlanEntry:
    """A single exact-shard transfer from a source worker to a receiver."""

    tensor_name: str
    source_worker_rank: int
    receiver_rank: int
    source: TensorShardSpec
    target: TensorReceiveSpec


@dataclass(frozen=True)
class MissingTensor:
    """A receiver tensor that could not be matched to a compatible source."""

    target: TensorReceiveSpec
    reason: str


@dataclass(frozen=True)
class TransferPlan:
    """Result of planning exact shard transfers for one receiver/update."""

    entries: tuple[TransferPlanEntry, ...]
    missing: tuple[MissingTensor, ...]

    @property
    def complete(self) -> bool:
        return not self.missing

    def raise_if_incomplete(self) -> None:
        if self.complete:
            return
        details = ", ".join(
            f"{item.target.name}@rank{item.target.receiver_rank}: {item.reason}"
            for item in self.missing
        )
        raise ValueError(f"incomplete RL reshard plan: {details}")


def plan_exact_transfers(
    sources: Iterable[TensorShardSpec],
    targets: Iterable[TensorReceiveSpec],
    *,
    same_rank_only: bool = True,
) -> TransferPlan:
    """Plan exact tensor-shard transfers from source specs to targets.

    Exact means source and target have the same tensor name, shape, dtype,
    tensor-parallel rank, pipeline-parallel rank, and compatible expert
    ownership. This is the no-allgather baseline that rank-local RL transfer
    needs before cross-rank reshard planning is added.
    """
    source_specs = tuple(sources)
    entries = []
    missing = []

    for target in targets:
        candidates = [
            source
            for source in source_specs
            if _is_compatible_source(source, target)
            and (not same_rank_only or source.worker_rank == target.receiver_rank)
        ]
        if not candidates:
            missing.append(MissingTensor(target, _missing_reason(source_specs, target, same_rank_only)))
            continue

        source = min(
            candidates,
            key=lambda candidate: (
                candidate.worker_rank != target.receiver_rank,
                candidate.worker_rank,
                candidate.name,
            ),
        )
        entries.append(
            TransferPlanEntry(
                tensor_name=target.name,
                source_worker_rank=source.worker_rank,
                receiver_rank=target.receiver_rank,
                source=source,
                target=target,
            )
        )

    return TransferPlan(tuple(entries), tuple(missing))


def _is_compatible_source(source: TensorShardSpec, target: TensorReceiveSpec) -> bool:
    return (
        source.name == target.name
        and source.shape == target.shape
        and source.dtype == target.dtype
        and source.tensor_parallel_rank == target.tensor_parallel_rank
        and source.pipeline_parallel_rank == target.pipeline_parallel_rank
        and _expert_ownership_matches(source.expert_ids, target.expert_ids)
    )


def _expert_ownership_matches(
    source_experts: frozenset[int],
    target_experts: frozenset[int],
) -> bool:
    if not source_experts and not target_experts:
        return True
    if source_experts and target_experts:
        return target_experts.issubset(source_experts)
    return False


def _missing_reason(
    sources: tuple[TensorShardSpec, ...],
    target: TensorReceiveSpec,
    same_rank_only: bool,
) -> str:
    if same_rank_only and any(_is_compatible_source(source, target) for source in sources):
        return "compatible source exists on a different rank"
    if any(source.name == target.name for source in sources):
        return "tensor exists but shape, dtype, layout, or expert ownership differs"
    return "tensor not found"
