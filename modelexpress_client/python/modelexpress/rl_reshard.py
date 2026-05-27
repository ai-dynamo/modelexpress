# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-agnostic RL reshard planning primitives.

This module intentionally does not move tensor bytes. It owns the pure
selection/mapping step between source tensor shards and receiver buffers so
framework adapters can provide tensor handles without embedding layout policy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class TensorShardSpec:
    """One source tensor shard advertised by a trainer or inference replica."""

    name: str
    worker_rank: int
    shape: tuple[int, ...]
    dtype: str
    global_shape: tuple[int, ...] = ()
    shard_offsets: tuple[int, ...] = ()
    tensor_parallel_rank: int = 0
    pipeline_parallel_rank: int = 0
    expert_ids: frozenset[int] = field(default_factory=frozenset)
    expert_order: tuple[int, ...] = ()
    expert_axis: int | None = None

    def __post_init__(self) -> None:
        shape = tuple(int(dim) for dim in self.shape)
        global_shape = (
            tuple(int(dim) for dim in self.global_shape)
            if self.global_shape
            else shape
        )
        shard_offsets = (
            tuple(int(offset) for offset in self.shard_offsets)
            if self.shard_offsets
            else tuple(0 for _dim in shape)
        )
        _validate_shard_geometry(self.name, shape, global_shape, shard_offsets)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "global_shape", global_shape)
        object.__setattr__(self, "shard_offsets", shard_offsets)
        object.__setattr__(self, "dtype", str(self.dtype))
        expert_ids, expert_order, expert_axis = _normalize_expert_layout(
            self.name,
            shape,
            self.expert_ids,
            self.expert_order,
            self.expert_axis,
        )
        object.__setattr__(self, "expert_ids", expert_ids)
        object.__setattr__(self, "expert_order", expert_order)
        object.__setattr__(self, "expert_axis", expert_axis)


@dataclass(frozen=True)
class TensorReceiveSpec:
    """One receiver tensor buffer that needs a compatible source shard."""

    name: str
    receiver_rank: int
    shape: tuple[int, ...]
    dtype: str
    global_shape: tuple[int, ...] = ()
    shard_offsets: tuple[int, ...] = ()
    tensor_parallel_rank: int = 0
    pipeline_parallel_rank: int = 0
    expert_ids: frozenset[int] = field(default_factory=frozenset)
    expert_order: tuple[int, ...] = ()
    expert_axis: int | None = None

    def __post_init__(self) -> None:
        shape = tuple(int(dim) for dim in self.shape)
        global_shape = (
            tuple(int(dim) for dim in self.global_shape)
            if self.global_shape
            else shape
        )
        shard_offsets = (
            tuple(int(offset) for offset in self.shard_offsets)
            if self.shard_offsets
            else tuple(0 for _dim in shape)
        )
        _validate_shard_geometry(self.name, shape, global_shape, shard_offsets)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "global_shape", global_shape)
        object.__setattr__(self, "shard_offsets", shard_offsets)
        object.__setattr__(self, "dtype", str(self.dtype))
        expert_ids, expert_order, expert_axis = _normalize_expert_layout(
            self.name,
            shape,
            self.expert_ids,
            self.expert_order,
            self.expert_axis,
        )
        object.__setattr__(self, "expert_ids", expert_ids)
        object.__setattr__(self, "expert_order", expert_order)
        object.__setattr__(self, "expert_axis", expert_axis)


@dataclass(frozen=True)
class TensorSlice:
    """A local tensor slice described by offsets and shape."""

    offsets: tuple[int, ...]
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        offsets = tuple(int(offset) for offset in self.offsets)
        shape = tuple(int(dim) for dim in self.shape)
        if len(offsets) != len(shape):
            raise ValueError("tensor slice offsets and shape must have the same rank")
        if any(offset < 0 for offset in offsets):
            raise ValueError("tensor slice offsets must be non-negative")
        if any(dim <= 0 for dim in shape):
            raise ValueError("tensor slice shape dimensions must be positive")
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "shape", shape)


@dataclass(frozen=True)
class TransferPlanEntry:
    """A single planned tensor segment transfer."""

    tensor_name: str
    source_worker_rank: int
    receiver_rank: int
    source: TensorShardSpec
    target: TensorReceiveSpec
    source_slice: TensorSlice | None = None
    target_slice: TensorSlice | None = None

    def __post_init__(self) -> None:
        if self.source_slice is None:
            object.__setattr__(
                self,
                "source_slice",
                TensorSlice(tuple(0 for _dim in self.source.shape), self.source.shape),
            )
        if self.target_slice is None:
            object.__setattr__(
                self,
                "target_slice",
                TensorSlice(tuple(0 for _dim in self.target.shape), self.target.shape),
            )


@dataclass(frozen=True)
class MissingTensor:
    """A receiver tensor that could not be matched to a compatible source."""

    target: TensorReceiveSpec
    reason: str


@dataclass(frozen=True)
class TransferPlan:
    """Result of planning tensor transfers for one receiver/update."""

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
    ownership. This is the no-allgather baseline for rank-local RL transfer.
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
            missing.append(
                MissingTensor(
                    target,
                    _missing_reason(source_specs, target, same_rank_only),
                )
            )
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


def plan_dense_reshard_transfers(
    sources: Iterable[TensorShardSpec],
    targets: Iterable[TensorReceiveSpec],
    *,
    same_rank_only: bool = False,
) -> TransferPlan:
    """Plan dense rectangular shard intersections from sources to targets.

    This is a pure planning primitive for tensor-parallel layout changes. It
    can produce multiple entries for one target when the target shard spans
    more than one source shard. Execution still belongs to the transfer layer.
    """
    source_specs = tuple(sources)
    entries = []
    missing = []

    for target in targets:
        candidates = [
            source
            for source in source_specs
            if _is_dense_reshard_candidate(source, target)
            and (not same_rank_only or source.worker_rank == target.receiver_rank)
        ]
        target_entries = []
        for source in sorted(
            candidates,
            key=lambda candidate: (
                candidate.worker_rank != target.receiver_rank,
                candidate.worker_rank,
                candidate.shard_offsets,
            ),
        ):
            slice_pairs = _intersect_local_slices(source, target)
            if not slice_pairs:
                continue
            for source_slice, target_slice in slice_pairs:
                target_entries.append(
                    TransferPlanEntry(
                        tensor_name=target.name,
                        source_worker_rank=source.worker_rank,
                        receiver_rank=target.receiver_rank,
                        source=source,
                        target=target,
                        source_slice=source_slice,
                        target_slice=target_slice,
                    )
                )

        if not _target_coverage_complete(target_entries, target.shape):
            missing.append(
                MissingTensor(
                    target,
                    _dense_missing_reason(source_specs, target, same_rank_only),
                )
            )
            continue
        entries.extend(target_entries)

    return TransferPlan(tuple(entries), tuple(missing))


def source_specs_from_shape_registry(
    shape_registry: Mapping[str, Any],
    *,
    worker_rank: int,
) -> tuple[TensorShardSpec, ...]:
    """Build source specs from RL shape-registry metadata."""
    return tuple(
        TensorShardSpec(
            name=name,
            worker_rank=worker_rank,
            shape=_shape_from_registry_entry(name, entry),
            dtype=_dtype_from_registry_entry(name, entry),
            global_shape=_tuple_from_registry_entry(entry, "global_shape"),
            shard_offsets=_tuple_from_registry_entry(entry, "shard_offsets"),
            tensor_parallel_rank=_int_from_registry_entry(entry, "tensor_parallel_rank"),
            pipeline_parallel_rank=_int_from_registry_entry(entry, "pipeline_parallel_rank"),
            expert_ids=_expert_ids_from_registry_entry(entry),
            expert_order=_expert_order_from_registry_entry(entry),
            expert_axis=_optional_int_from_registry_entry(entry, "expert_axis"),
        )
        for name, entry in shape_registry.items()
    )


def receive_specs_from_shape_registry(
    shape_registry: Mapping[str, Any],
    *,
    receiver_rank: int,
) -> tuple[TensorReceiveSpec, ...]:
    """Build receiver specs from RL shape-registry metadata."""
    return tuple(
        TensorReceiveSpec(
            name=name,
            receiver_rank=receiver_rank,
            shape=_shape_from_registry_entry(name, entry),
            dtype=_dtype_from_registry_entry(name, entry),
            global_shape=_tuple_from_registry_entry(entry, "global_shape"),
            shard_offsets=_tuple_from_registry_entry(entry, "shard_offsets"),
            tensor_parallel_rank=_int_from_registry_entry(entry, "tensor_parallel_rank"),
            pipeline_parallel_rank=_int_from_registry_entry(entry, "pipeline_parallel_rank"),
            expert_ids=_expert_ids_from_registry_entry(entry),
            expert_order=_expert_order_from_registry_entry(entry),
            expert_axis=_optional_int_from_registry_entry(entry, "expert_axis"),
        )
        for name, entry in shape_registry.items()
    )


def receive_specs_from_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    receiver_rank: int,
) -> tuple[TensorReceiveSpec, ...]:
    """Build default dense receiver specs from caller-owned tensors."""
    return tuple(
        TensorReceiveSpec(
            name=name,
            receiver_rank=receiver_rank,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
        )
        for name, tensor in tensors.items()
    )


def _is_compatible_source(source: TensorShardSpec, target: TensorReceiveSpec) -> bool:
    return (
        source.name == target.name
        and source.shape == target.shape
        and source.global_shape == target.global_shape
        and source.shard_offsets == target.shard_offsets
        and source.dtype == target.dtype
        and source.tensor_parallel_rank == target.tensor_parallel_rank
        and source.pipeline_parallel_rank == target.pipeline_parallel_rank
        and _expert_ownership_matches(source.expert_ids, target.expert_ids)
    )


def _is_dense_reshard_candidate(source: TensorShardSpec, target: TensorReceiveSpec) -> bool:
    return (
        source.name == target.name
        and source.dtype == target.dtype
        and source.global_shape == target.global_shape
        and source.pipeline_parallel_rank == target.pipeline_parallel_rank
        and _expert_ownership_can_contribute(source, target)
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


def _expert_ownership_can_contribute(
    source: TensorShardSpec,
    target: TensorReceiveSpec,
) -> bool:
    if _uses_expert_slicing(source, target):
        return bool(source.expert_ids.intersection(target.expert_ids))
    return _expert_ownership_matches(source.expert_ids, target.expert_ids)


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


def _dense_missing_reason(
    sources: tuple[TensorShardSpec, ...],
    target: TensorReceiveSpec,
    same_rank_only: bool,
) -> str:
    candidates = [
        source
        for source in sources
        if _is_dense_reshard_candidate(source, target)
    ]
    if same_rank_only and any(_intersect_local_slices(source, target) for source in candidates):
        return "compatible source coverage exists on a different rank"
    if candidates:
        return "source coverage is incomplete or overlapping"
    if any(source.name == target.name for source in sources):
        return "tensor exists but dtype, global shape, pipeline rank, or expert ownership differs"
    return "tensor not found"


def _intersect_local_slices(
    source: TensorShardSpec,
    target: TensorReceiveSpec,
) -> tuple[tuple[TensorSlice, TensorSlice], ...]:
    if _uses_expert_slicing(source, target):
        return _intersect_expert_slices(source, target)

    starts = []
    shape = []
    for source_offset, source_dim, target_offset, target_dim in zip(
        source.shard_offsets,
        source.shape,
        target.shard_offsets,
        target.shape,
        strict=True,
    ):
        start = max(source_offset, target_offset)
        end = min(source_offset + source_dim, target_offset + target_dim)
        if start >= end:
            return ()
        starts.append(start)
        shape.append(end - start)

    source_offsets = tuple(
        start - offset
        for start, offset in zip(starts, source.shard_offsets, strict=True)
    )
    target_offsets = tuple(
        start - offset
        for start, offset in zip(starts, target.shard_offsets, strict=True)
    )
    return ((
        TensorSlice(source_offsets, tuple(shape)),
        TensorSlice(target_offsets, tuple(shape)),
    ),)


def _uses_expert_slicing(
    source: TensorShardSpec,
    target: TensorReceiveSpec,
) -> bool:
    return (
        source.expert_axis is not None
        and target.expert_axis is not None
        and source.expert_axis == target.expert_axis
        and bool(source.expert_order)
        and bool(target.expert_order)
    )


def _intersect_expert_slices(
    source: TensorShardSpec,
    target: TensorReceiveSpec,
) -> tuple[tuple[TensorSlice, TensorSlice], ...]:
    expert_axis = source.expert_axis
    if expert_axis is None or target.expert_axis is None:
        return ()
    source_index_by_expert = {
        expert: index
        for index, expert in enumerate(source.expert_order)
    }
    target_index_by_expert = {
        expert: index
        for index, expert in enumerate(target.expert_order)
    }
    non_expert_slices = _intersect_non_expert_dims(source, target, expert_axis)
    if non_expert_slices is None:
        return ()
    source_base_offsets, target_base_offsets, base_shape = non_expert_slices
    entries = []
    for expert in target.expert_order:
        source_expert_index = source_index_by_expert.get(expert)
        target_expert_index = target_index_by_expert.get(expert)
        if source_expert_index is None or target_expert_index is None:
            continue
        source_offsets = list(source_base_offsets)
        target_offsets = list(target_base_offsets)
        shape = list(base_shape)
        source_offsets[expert_axis] = source_expert_index
        target_offsets[expert_axis] = target_expert_index
        shape[expert_axis] = 1
        entries.append(
            (
                TensorSlice(tuple(source_offsets), tuple(shape)),
                TensorSlice(tuple(target_offsets), tuple(shape)),
            )
        )
    return tuple(entries)


def _intersect_non_expert_dims(
    source: TensorShardSpec,
    target: TensorReceiveSpec,
    expert_axis: int,
) -> tuple[list[int], list[int], list[int]] | None:
    source_offsets = [0 for _dim in source.shape]
    target_offsets = [0 for _dim in target.shape]
    shape = list(target.shape)
    for dim, (source_offset, source_dim, target_offset, target_dim) in enumerate(
        zip(
            source.shard_offsets,
            source.shape,
            target.shard_offsets,
            target.shape,
            strict=True,
        )
    ):
        if dim == expert_axis:
            continue
        start = max(source_offset, target_offset)
        end = min(source_offset + source_dim, target_offset + target_dim)
        if start >= end:
            return None
        source_offsets[dim] = start - source_offset
        target_offsets[dim] = start - target_offset
        shape[dim] = end - start
    return source_offsets, target_offsets, shape


def _target_coverage_complete(
    entries: list[TransferPlanEntry],
    target_shape: tuple[int, ...],
) -> bool:
    return _covered_numel(entries) == _numel(target_shape) and not _target_slices_overlap(entries)


def _target_slices_overlap(entries: list[TransferPlanEntry]) -> bool:
    slices = [
        entry.target_slice
        for entry in entries
        if entry.target_slice is not None
    ]
    for index, left in enumerate(slices):
        for right in slices[index + 1:]:
            if _slices_overlap(left, right):
                return True
    return False


def _slices_overlap(left: TensorSlice, right: TensorSlice) -> bool:
    for left_offset, left_dim, right_offset, right_dim in zip(
        left.offsets,
        left.shape,
        right.offsets,
        right.shape,
        strict=True,
    ):
        if left_offset >= right_offset + right_dim:
            return False
        if right_offset >= left_offset + left_dim:
            return False
    return True


def _covered_numel(entries: list[TransferPlanEntry]) -> int:
    return sum(
        _numel(entry.target_slice.shape)
        for entry in entries
        if entry.target_slice is not None
    )


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _validate_shard_geometry(
    name: str,
    shape: tuple[int, ...],
    global_shape: tuple[int, ...],
    shard_offsets: tuple[int, ...],
) -> None:
    if len(shape) != len(global_shape) or len(shape) != len(shard_offsets):
        raise ValueError(
            f"tensor {name!r} shape, global_shape, and shard_offsets must have the same rank"
        )
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"tensor {name!r} shape dimensions must be positive")
    if any(dim <= 0 for dim in global_shape):
        raise ValueError(f"tensor {name!r} global_shape dimensions must be positive")
    if any(offset < 0 for offset in shard_offsets):
        raise ValueError(f"tensor {name!r} shard_offsets must be non-negative")
    for offset, dim, global_dim in zip(shard_offsets, shape, global_shape, strict=True):
        if offset + dim > global_dim:
            raise ValueError(f"tensor {name!r} shard extends beyond global_shape")


def _shape_from_registry_entry(name: str, entry: Any) -> tuple[int, ...]:
    if not isinstance(entry, Mapping):
        raise ValueError(f"shape registry entry for {name!r} must be an object")
    shape = entry.get("shape")
    if not isinstance(shape, list):
        raise ValueError(f"shape registry entry for {name!r} must include shape")
    return tuple(int(dim) for dim in shape)


def _dtype_from_registry_entry(name: str, entry: Any) -> str:
    if not isinstance(entry, Mapping):
        raise ValueError(f"shape registry entry for {name!r} must be an object")
    dtype = entry.get("dtype")
    if dtype is None:
        raise ValueError(f"shape registry entry for {name!r} must include dtype")
    return str(dtype)


def _int_from_registry_entry(entry: Any, key: str) -> int:
    if not isinstance(entry, Mapping):
        return 0
    return int(entry.get(key, 0))


def _optional_int_from_registry_entry(entry: Any, key: str) -> int | None:
    if not isinstance(entry, Mapping):
        return None
    value = entry.get(key)
    if value is None:
        return None
    return int(value)


def _tuple_from_registry_entry(entry: Any, key: str) -> tuple[int, ...]:
    if not isinstance(entry, Mapping):
        return ()
    value = entry.get(key, [])
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"shape registry {key} must be a list")
    return tuple(int(item) for item in value)


def _expert_ids_from_registry_entry(entry: Any) -> frozenset[int]:
    return frozenset(_expert_ids_sequence_from_registry_entry(entry))


def _expert_order_from_registry_entry(entry: Any) -> tuple[int, ...]:
    return _expert_ids_sequence_from_registry_entry(entry)


def _expert_ids_sequence_from_registry_entry(entry: Any) -> tuple[int, ...]:
    if not isinstance(entry, Mapping):
        return ()
    expert_ids = entry.get("expert_ids", [])
    if expert_ids is None:
        return ()
    if not isinstance(expert_ids, list):
        raise ValueError("shape registry expert_ids must be a list")
    return tuple(int(expert) for expert in expert_ids)


def _normalize_expert_layout(
    name: str,
    shape: tuple[int, ...],
    expert_ids: Iterable[int],
    expert_order: Iterable[int],
    expert_axis: int | None,
) -> tuple[frozenset[int], tuple[int, ...], int | None]:
    normalized_ids, normalized_order = _normalize_expert_ids(
        expert_ids,
        expert_order,
    )
    return (
        normalized_ids,
        normalized_order,
        _normalize_expert_axis(name, shape, normalized_order, expert_axis),
    )


def _normalize_expert_ids(
    expert_ids: Iterable[int],
    expert_order: Iterable[int],
) -> tuple[frozenset[int], tuple[int, ...]]:
    raw_expert_ids = tuple(int(expert) for expert in expert_ids)
    normalized_ids = frozenset(raw_expert_ids)
    normalized_order = tuple(int(expert) for expert in expert_order)
    if not normalized_order and raw_expert_ids:
        normalized_order = (
            tuple(sorted(normalized_ids))
            if isinstance(expert_ids, (set, frozenset))
            else raw_expert_ids
        )
    if normalized_order and frozenset(normalized_order) != normalized_ids:
        raise ValueError("expert_order must contain the same expert IDs as expert_ids")
    if len(normalized_order) != len(set(normalized_order)):
        raise ValueError("expert_order must not contain duplicate expert IDs")
    return normalized_ids, normalized_order


def _normalize_expert_axis(
    name: str,
    shape: tuple[int, ...],
    expert_order: tuple[int, ...],
    expert_axis: int | None,
) -> int | None:
    if not expert_order:
        if expert_axis is not None:
            raise ValueError(f"tensor {name!r} expert_axis requires expert_ids")
        return None

    if expert_axis is None:
        return None
    axis = int(expert_axis)
    if axis < 0:
        axis += len(shape)
    if axis < 0 or axis >= len(shape):
        raise ValueError(f"tensor {name!r} expert_axis is out of range")
    if shape[axis] != len(expert_order):
        raise ValueError(
            f"tensor {name!r} expert_axis dimension must match expert_ids length"
        )
    return axis
