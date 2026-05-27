# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for materializing RL slice transfer plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from modelexpress.rl_reshard import TensorSlice, TransferPlan, TransferPlanEntry
from modelexpress.types import TensorDescriptor


@dataclass(frozen=True)
class SliceTransferManifest:
    """Concrete tensors and descriptors needed to execute a slice plan."""

    target_tensors: dict[str, torch.Tensor]
    source_descriptors: list[TensorDescriptor]
    output_tensors: list[tuple[str, torch.Tensor]]


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
    output_names = []
    entry_counts = _entry_counts(plan)
    duplicated_names = sorted(
        tensor_name
        for tensor_name, count in entry_counts.items()
        if count > 1
    )
    if duplicated_names:
        raise RuntimeError(
            "ModelExpress dense slice transfer currently materializes one source "
            f"descriptor per tensor; multi-source tensors are not supported: {duplicated_names}"
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
        target_view, transfer_source = _materialize_entry(
            entry,
            transfer_name=transfer_name,
            source_descriptor=source_descriptor,
            target_tensors=target_tensors,
        )
        transfer_targets[transfer_name] = target_view
        transfer_sources.append(transfer_source)
        if entry.target.name not in output_names:
            output_names.append(entry.target.name)

    return SliceTransferManifest(
        target_tensors=transfer_targets,
        source_descriptors=transfer_sources,
        output_tensors=[
            (name, target_tensors[name])
            for name in output_names
        ],
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
        target_view, transfer_source = _materialize_entry(
            entry,
            transfer_name=transfer_name,
            source_descriptor=source_descriptor,
            target_tensors=target_tensors,
        )
        transfer_targets[transfer_name] = target_view
        transfer_sources_by_rank.setdefault(entry.source_worker_rank, []).append(transfer_source)
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


def _transfer_name(tensor_name: str, index: int, entry_count: int) -> str:
    if entry_count == 1:
        return tensor_name
    return f"{tensor_name}.__mx_slice_{index}"


def _materialize_entry(
    entry: TransferPlanEntry,
    *,
    transfer_name: str,
    source_descriptor: TensorDescriptor,
    target_tensors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, TensorDescriptor]:
    target_tensor = target_tensors.get(entry.target.name)
    if target_tensor is None:
        raise RuntimeError(
            f"ModelExpress target tensors missing planned entry {entry.target.name!r}"
        )
    source_slice = _require_slice(entry.source_slice)
    target_slice = _require_slice(entry.target_slice)
    source_offset, source_elements = _contiguous_slice_span(
        entry.source.shape,
        source_slice,
        tensor_name=entry.source.name,
        side="source",
    )
    _contiguous_slice_span(
        entry.target.shape,
        target_slice,
        tensor_name=entry.target.name,
        side="target",
    )
    target_view = _slice_view(target_tensor, target_slice)
    if not target_view.is_contiguous():
        raise RuntimeError(
            f"ModelExpress target slice for {entry.target.name!r} is not contiguous"
        )

    source_element_size = _source_element_size(source_descriptor, entry.source.shape)
    return target_view, TensorDescriptor(
        name=transfer_name,
        addr=source_descriptor.addr + source_offset * source_element_size,
        size=source_elements * source_element_size,
        device_id=source_descriptor.device_id,
        dtype=source_descriptor.dtype,
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
    if len(tensor_shape) != len(tensor_slice.shape):
        raise RuntimeError(
            f"ModelExpress {side} slice rank mismatch for {tensor_name!r}"
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
    if last - start + 1 != elements:
        raise RuntimeError(
            f"ModelExpress {side} slice for {tensor_name!r} is not contiguous"
        )
    return start, elements


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
