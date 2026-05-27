# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for materializing RL slice transfer plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product

import torch

from modelexpress.rl_reshard import TensorSlice, TransferPlan, TransferPlanEntry
from modelexpress.types import TensorDescriptor


@dataclass(frozen=True)
class SliceTransferManifest:
    """Concrete tensors and descriptors needed to execute a slice plan."""

    target_tensors: dict[str, torch.Tensor]
    source_descriptors: list[TensorDescriptor]
    output_tensors: list[tuple[str, torch.Tensor]]
    post_transfer_copies: tuple["StagedTensorCopy", ...] = ()

    def finalize(self) -> None:
        """Apply any staged target copies after transfer completion."""
        _apply_post_transfer_copies(self.post_transfer_copies)


@dataclass(frozen=True)
class SourceSliceTransferManifest:
    """Descriptors to pull from one source worker."""

    source_worker_rank: int
    source_descriptors: list[TensorDescriptor]


@dataclass(frozen=True)
class GroupedSliceTransferManifest:
    """Concrete tensors and per-source descriptors for multi-source fan-in."""

    target_tensors: dict[str, torch.Tensor]
    source_transfers: list[SourceSliceTransferManifest]
    output_tensors: list[tuple[str, torch.Tensor]]
    post_transfer_copies: tuple["StagedTensorCopy", ...] = ()

    def finalize(self) -> None:
        """Apply any staged target copies after transfer completion."""
        _apply_post_transfer_copies(self.post_transfer_copies)


@dataclass(frozen=True)
class StagedTensorCopy:
    """Copy from a contiguous receive buffer into a caller-owned tensor view."""

    source: torch.Tensor
    target: torch.Tensor


def build_slice_transfer_manifest(
    plan: TransferPlan,
    *,
    source_descriptors: list[TensorDescriptor],
    target_tensors: dict[str, torch.Tensor],
) -> SliceTransferManifest:
    """Build NIXL-compatible views/descriptors for a complete slice plan."""
    if not plan.complete:
        raise ValueError("cannot materialize an incomplete RL slice transfer plan")

    source_by_name = {descriptor.name: descriptor for descriptor in source_descriptors}
    transfer_targets = {}
    transfer_sources = []
    post_transfer_copies: list[StagedTensorCopy] = []
    output_names = []
    entry_counts = _entry_counts(plan)
    multi_source_names = _multi_source_tensor_names(plan)
    if multi_source_names:
        raise RuntimeError(
            "ModelExpress dense slice transfer currently materializes one source "
            f"descriptor group per tensor; multi-source tensors are not supported: "
            f"{multi_source_names}"
        )

    entry_offsets: dict[str, int] = {}
    for entry in plan.entries:
        source_descriptor = source_by_name.get(entry.source.name)
        if source_descriptor is None:
            raise RuntimeError(
                f"ModelExpress source descriptors missing planned entry {entry.source.name!r}"
            )
        entry_index = entry_offsets.get(entry.tensor_name, 0)
        entry_offsets[entry.tensor_name] = entry_index + 1
        transfer_name = _transfer_name(
            entry.tensor_name,
            entry_index,
            entry_counts[entry.tensor_name],
        )
        materialized_entries = _materialize_entry(
            entry,
            transfer_name=transfer_name,
            source_descriptor=source_descriptor,
            target_tensors=target_tensors,
        )
        for materialized in materialized_entries:
            transfer_targets[materialized.source_descriptor.name] = materialized.target_tensor
            transfer_sources.append(materialized.source_descriptor)
            if materialized.post_transfer_copy is not None:
                post_transfer_copies.append(materialized.post_transfer_copy)
        if entry.target.name not in output_names:
            output_names.append(entry.target.name)

    return SliceTransferManifest(
        target_tensors=transfer_targets,
        source_descriptors=transfer_sources,
        output_tensors=[
            (name, target_tensors[name])
            for name in output_names
        ],
        post_transfer_copies=tuple(post_transfer_copies),
    )


def build_grouped_slice_transfer_manifest(
    plan: TransferPlan,
    *,
    source_descriptors_by_rank: Mapping[int, list[TensorDescriptor]],
    target_tensors: dict[str, torch.Tensor],
) -> GroupedSliceTransferManifest:
    """Build NIXL-compatible views/descriptors for a multi-source slice plan."""
    if not plan.complete:
        raise ValueError("cannot materialize an incomplete RL slice transfer plan")

    descriptors_by_rank_and_name = {
        worker_rank: {descriptor.name: descriptor for descriptor in descriptors}
        for worker_rank, descriptors in source_descriptors_by_rank.items()
    }
    entry_counts = _entry_counts(plan)
    transfer_targets = {}
    transfer_sources_by_rank: dict[int, list[TensorDescriptor]] = {}
    post_transfer_copies: list[StagedTensorCopy] = []
    output_names = []

    entry_offsets: dict[str, int] = {}
    for entry in plan.entries:
        source_descriptors = descriptors_by_rank_and_name.get(entry.source_worker_rank)
        if source_descriptors is None:
            raise RuntimeError(
                "ModelExpress source descriptors missing planned worker rank "
                f"{entry.source_worker_rank}"
            )
        source_descriptor = source_descriptors.get(entry.source.name)
        if source_descriptor is None:
            raise RuntimeError(
                f"ModelExpress source descriptors missing planned entry {entry.source.name!r}"
            )
        entry_index = entry_offsets.get(entry.tensor_name, 0)
        entry_offsets[entry.tensor_name] = entry_index + 1
        transfer_name = _transfer_name(
            entry.tensor_name,
            entry_index,
            entry_counts[entry.tensor_name],
        )
        materialized_entries = _materialize_entry(
            entry,
            transfer_name=transfer_name,
            source_descriptor=source_descriptor,
            target_tensors=target_tensors,
        )
        for materialized in materialized_entries:
            transfer_targets[materialized.source_descriptor.name] = materialized.target_tensor
            transfer_sources_by_rank.setdefault(entry.source_worker_rank, []).append(
                materialized.source_descriptor
            )
            if materialized.post_transfer_copy is not None:
                post_transfer_copies.append(materialized.post_transfer_copy)
        if entry.target.name not in output_names:
            output_names.append(entry.target.name)

    return GroupedSliceTransferManifest(
        target_tensors=transfer_targets,
        source_transfers=[
            SourceSliceTransferManifest(worker_rank, descriptors)
            for worker_rank, descriptors in transfer_sources_by_rank.items()
        ],
        output_tensors=[
            (name, target_tensors[name])
            for name in output_names
        ],
        post_transfer_copies=tuple(post_transfer_copies),
    )


def _require_slice(tensor_slice: TensorSlice | None) -> TensorSlice:
    if tensor_slice is None:
        raise ValueError("slice transfer plan entry is missing slice metadata")
    return tensor_slice


def _entry_counts(plan: TransferPlan) -> dict[str, int]:
    entry_counts = {}
    for entry in plan.entries:
        entry_counts[entry.tensor_name] = entry_counts.get(entry.tensor_name, 0) + 1
    return entry_counts


def _multi_source_tensor_names(plan: TransferPlan) -> list[str]:
    source_ranks_by_name: dict[str, set[int]] = {}
    for entry in plan.entries:
        source_ranks_by_name.setdefault(entry.tensor_name, set()).add(
            entry.source_worker_rank
        )
    return sorted(
        tensor_name
        for tensor_name, source_ranks in source_ranks_by_name.items()
        if len(source_ranks) > 1
    )


def _transfer_name(tensor_name: str, index: int, entry_count: int) -> str:
    if entry_count == 1:
        return tensor_name
    return f"{tensor_name}.__mx_slice_{index}"


def _fragment_transfer_name(transfer_name: str, index: int, fragment_count: int) -> str:
    if fragment_count == 1:
        return transfer_name
    return f"{transfer_name}.__mx_fragment_{index}"


@dataclass(frozen=True)
class _MaterializedEntry:
    target_tensor: torch.Tensor
    source_descriptor: TensorDescriptor
    post_transfer_copy: StagedTensorCopy | None


def _materialize_entry(
    entry: TransferPlanEntry,
    *,
    transfer_name: str,
    source_descriptor: TensorDescriptor,
    target_tensors: dict[str, torch.Tensor],
) -> tuple[_MaterializedEntry, ...]:
    target_tensor = target_tensors.get(entry.target.name)
    if target_tensor is None:
        raise RuntimeError(
            f"ModelExpress target tensors missing planned entry {entry.target.name!r}"
        )
    source_slice = _require_slice(entry.source_slice)
    target_slice = _require_slice(entry.target_slice)
    fragments = _source_fragments(
        source_shape=entry.source.shape,
        source_slice=source_slice,
        target_slice=target_slice,
        tensor_name=entry.source.name,
    )
    source_element_size = _source_element_size(source_descriptor, entry.source.shape)
    materialized_entries = []
    for fragment_index, fragment in enumerate(fragments):
        source_offset, source_elements = _contiguous_slice_span(
            entry.source.shape,
            fragment.source_slice,
            tensor_name=entry.source.name,
            side="source",
        )
        fragment_name = _fragment_transfer_name(
            transfer_name,
            fragment_index,
            len(fragments),
        )
        target_tensor_view, post_transfer_copy = _materialize_target_view(
            entry=entry,
            target_tensor=target_tensor,
            target_slice=fragment.target_slice,
        )
        materialized_entries.append(
            _MaterializedEntry(
                target_tensor=target_tensor_view,
                source_descriptor=TensorDescriptor(
                    name=fragment_name,
                    addr=source_descriptor.addr + source_offset * source_element_size,
                    size=source_elements * source_element_size,
                    device_id=source_descriptor.device_id,
                    dtype=source_descriptor.dtype,
                ),
                post_transfer_copy=post_transfer_copy,
            )
        )
    return tuple(materialized_entries)


@dataclass(frozen=True)
class _TransferFragment:
    source_slice: TensorSlice
    target_slice: TensorSlice


def _source_fragments(
    *,
    source_shape: tuple[int, ...],
    source_slice: TensorSlice,
    target_slice: TensorSlice,
    tensor_name: str,
) -> tuple[_TransferFragment, ...]:
    if source_slice.shape != target_slice.shape:
        raise RuntimeError(
            f"ModelExpress source and target slice shape mismatch for {tensor_name!r}"
        )
    if _slice_is_contiguous(source_shape, source_slice, tensor_name=tensor_name):
        return (_TransferFragment(source_slice, target_slice),)

    suffix_start = len(source_slice.shape) - 1
    while suffix_start > 0 and _slice_covers_full_dim(
        source_shape,
        source_slice,
        suffix_start,
    ):
        suffix_start -= 1

    prefix_shape = source_slice.shape[:suffix_start]
    fragments = []
    for prefix_offsets in product(*(range(dim) for dim in prefix_shape)):
        source_offsets = list(source_slice.offsets)
        target_offsets = list(target_slice.offsets)
        fragment_shape = list(source_slice.shape)
        for dim, relative_offset in enumerate(prefix_offsets):
            source_offsets[dim] += relative_offset
            target_offsets[dim] += relative_offset
            fragment_shape[dim] = 1
        fragments.append(
            _TransferFragment(
                source_slice=TensorSlice(tuple(source_offsets), tuple(fragment_shape)),
                target_slice=TensorSlice(tuple(target_offsets), tuple(fragment_shape)),
            )
        )
    return tuple(fragments)


def _materialize_target_view(
    *,
    entry: TransferPlanEntry,
    target_tensor: torch.Tensor,
    target_slice: TensorSlice,
) -> tuple[torch.Tensor, StagedTensorCopy | None]:
    _validate_slice_bounds(
        entry.target.shape,
        target_slice,
        tensor_name=entry.target.name,
        side="target",
    )
    target_view = _slice_view(target_tensor, target_slice)
    if tuple(target_view.shape) != target_slice.shape:
        raise RuntimeError(
            f"ModelExpress target slice for {entry.target.name!r} exceeds target tensor bounds"
        )
    if target_view.is_contiguous():
        return target_view, None
    transfer_target = torch.empty(
        tuple(target_view.shape),
        dtype=target_view.dtype,
        device=target_view.device,
    )
    return transfer_target, StagedTensorCopy(
        source=transfer_target,
        target=target_view,
    )


def _slice_view(tensor: torch.Tensor, tensor_slice: TensorSlice) -> torch.Tensor:
    slices = tuple(
        slice(offset, offset + dim)
        for offset, dim in zip(tensor_slice.offsets, tensor_slice.shape, strict=True)
    )
    return tensor[slices]


def _contiguous_slice_span(
    tensor_shape: tuple[int, ...],
    tensor_slice: TensorSlice,
    *,
    tensor_name: str,
    side: str,
) -> tuple[int, int]:
    start, elements, contiguous = _slice_linear_span(
        tensor_shape,
        tensor_slice,
        tensor_name=tensor_name,
        side=side,
    )
    if not contiguous:
        raise RuntimeError(
            f"ModelExpress {side} slice for {tensor_name!r} is not contiguous"
        )
    return start, elements


def _slice_is_contiguous(
    tensor_shape: tuple[int, ...],
    tensor_slice: TensorSlice,
    *,
    tensor_name: str,
) -> bool:
    _start, _elements, contiguous = _slice_linear_span(
        tensor_shape,
        tensor_slice,
        tensor_name=tensor_name,
        side="source",
    )
    return contiguous


def _slice_linear_span(
    tensor_shape: tuple[int, ...],
    tensor_slice: TensorSlice,
    *,
    tensor_name: str,
    side: str,
) -> tuple[int, int, bool]:
    _validate_slice_bounds(
        tensor_shape,
        tensor_slice,
        tensor_name=tensor_name,
        side=side,
    )
    strides = _contiguous_strides(tensor_shape)
    start = sum(
        offset * stride
        for offset, stride in zip(tensor_slice.offsets, strides, strict=True)
    )
    last = sum(
        (offset + dim - 1) * stride
        for offset, dim, stride in zip(
            tensor_slice.offsets,
            tensor_slice.shape,
            strides,
            strict=True,
        )
    )
    elements = _numel(tensor_slice.shape)
    return start, elements, last - start + 1 == elements


def _slice_covers_full_dim(
    tensor_shape: tuple[int, ...],
    tensor_slice: TensorSlice,
    dim: int,
) -> bool:
    return tensor_slice.offsets[dim] == 0 and tensor_slice.shape[dim] == tensor_shape[dim]


def _validate_slice_bounds(
    tensor_shape: tuple[int, ...],
    tensor_slice: TensorSlice,
    *,
    tensor_name: str,
    side: str,
) -> None:
    if len(tensor_shape) != len(tensor_slice.shape):
        raise RuntimeError(
            f"ModelExpress {side} slice rank mismatch for {tensor_name!r}"
        )
    for offset, dim, tensor_dim in zip(
        tensor_slice.offsets,
        tensor_slice.shape,
        tensor_shape,
        strict=True,
    ):
        if offset + dim > tensor_dim:
            raise RuntimeError(
                f"ModelExpress {side} slice for {tensor_name!r} exceeds tensor bounds"
            )


def _apply_post_transfer_copies(copies: tuple[StagedTensorCopy, ...]) -> None:
    for copy in copies:
        copy.target.copy_(copy.source)


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


def _source_element_size(
    descriptor: TensorDescriptor,
    source_shape: tuple[int, ...],
) -> int:
    elements = _numel(source_shape)
    if elements <= 0 or descriptor.size % elements != 0:
        raise RuntimeError(
            f"ModelExpress source descriptor {descriptor.name!r} size does not match plan shape"
        )
    return descriptor.size // elements


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total
