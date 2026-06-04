# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intersection planner for source-owned tensor slices and receiver requests."""

from __future__ import annotations

from itertools import product
from typing import Sequence

from .resharding_ranges import (
    TensorRange,
    _validate_range_inside_shape,
    intersect_ranges,
    range_extents,
    range_volume,
    row_major_strides,
)
from .resharding_types import (
    CoverageError,
    IncompatibleManifestError,
    QuantizationMetadataError,
    QuantizationScope,
    RetryPolicy,
    SegmentPlan,
    SliceOwnership,
    SliceRequest,
    _normalize_dtype,
    dtype_itemsize,
)

_STRICT_LAYOUT_KEYS = {
    "axis_order",
    "moe_expert_axis",
    "packing",
    "quant_block_shape",
    "storage_layout",
}


def classify_tensor_family(
    tensor_name: str,
    *,
    layout_tags: dict[str, str | int | bool] | None = None,
    quantization_scope: QuantizationScope | str = QuantizationScope.ABSENT,
) -> str:
    """Classify tensor handling requirements for planner artifacts."""

    layout_tags = layout_tags or {}
    scope = QuantizationScope(quantization_scope)
    lowered = tensor_name.lower()

    if scope == QuantizationScope.GLOBAL_REQUIRED:
        return "quantization-global-required-fallback"
    if scope == QuantizationScope.GENERATED_ON_TARGET:
        return "generated-on-target"
    if "moe_expert_axis" in layout_tags or ".experts." in lowered:
        return "moe-expert-axis-shard"
    if any(key in layout_tags for key in _STRICT_LAYOUT_KEYS):
        return "layout-sensitive-slice"
    return "plain-slice"


def classify_quantization_scope(
    ownership: SliceOwnership,
    request: SliceRequest,
) -> str:
    """Classify whether a source/request pair can be assembled zero-copy."""

    scopes = {ownership.quantization_scope, request.quantization_scope}
    if QuantizationScope.GLOBAL_REQUIRED in scopes:
        return "fallback-required"
    if QuantizationScope.GENERATED_ON_TARGET in scopes:
        return "generated-on-target"
    if QuantizationScope.LOCAL in scopes:
        return "local"
    return "absent"


def plan_segments(
    ownerships: Sequence[SliceOwnership],
    requests: Sequence[SliceRequest],
) -> list[SegmentPlan]:
    """Build contiguous transfer segments for receiver-side slice requests.

    The planner enforces exact coverage. Every requested element must be
    covered by exactly one source ownership range; duplicate ownership is
    rejected here so recovery replicas can be modeled explicitly later.
    """

    plans: list[SegmentPlan] = []
    owners_by_tensor: dict[str, list[SliceOwnership]] = {}
    for ownership in ownerships:
        owners_by_tensor.setdefault(ownership.tensor_name, []).append(ownership)

    for request in requests:
        owners = owners_by_tensor.get(request.tensor_name, [])
        if not owners:
            raise CoverageError(
                f"no source ownerships for tensor {request.tensor_name!r}",
                missing_ranges=[request.requested_range],
            )
        _validate_request_compatibility(request, owners)

        missing: list[TensorRange] = []
        duplicate: list[TensorRange] = []
        request_plans: list[SegmentPlan] = []

        for cell in _partition_request_range(request.requested_range, owners):
            covering = [
                owner for owner in owners if _contains_range(owner.source_range, cell)
            ]
            if not covering:
                missing.append(cell)
                continue
            if len(covering) > 1:
                duplicate.append(cell)
                continue

            owner = covering[0]
            for segment_range in _contiguous_segment_ranges(cell, owner, request):
                request_plans.append(_build_segment_plan(owner, request, segment_range))

        if missing or duplicate:
            pieces: list[str] = []
            if missing:
                pieces.append(f"{len(missing)} missing range(s)")
            if duplicate:
                pieces.append(f"{len(duplicate)} duplicate range(s)")
            raise CoverageError(
                f"tensor {request.tensor_name!r} has " + " and ".join(pieces),
                missing_ranges=missing,
                duplicate_ranges=duplicate,
            )

        plans.extend(request_plans)

    return plans


def _validate_request_compatibility(
    request: SliceRequest,
    owners: Sequence[SliceOwnership],
) -> None:
    reference_shape = owners[0].global_shape
    if len(request.requested_range) != len(reference_shape):
        raise IncompatibleManifestError(
            f"tensor {request.tensor_name!r} request rank does not match source rank"
        )
    _validate_range_inside_shape(
        request.requested_range,
        reference_shape,
        "requested_range",
    )

    for owner in owners:
        if owner.global_shape != reference_shape:
            raise IncompatibleManifestError(
                f"tensor {request.tensor_name!r} has inconsistent global_shape "
                f"{owner.global_shape} vs {reference_shape}"
            )
        if request.model_name and owner.model_name != request.model_name:
            raise IncompatibleManifestError(
                f"source model {owner.model_name!r} does not match request "
                f"{request.model_name!r}"
            )
        if request.model_version and owner.model_version != request.model_version:
            raise IncompatibleManifestError(
                f"source version {owner.model_version!r} does not match request "
                f"{request.model_version!r}"
            )
        if _normalize_dtype(owner.dtype) != _normalize_dtype(request.dtype):
            raise IncompatibleManifestError(
                f"tensor {request.tensor_name!r} dtype mismatch: "
                f"source={owner.dtype!r}, request={request.dtype!r}"
            )
        owner_itemsize = dtype_itemsize(owner.dtype, owner.element_size_bytes)
        request_itemsize = dtype_itemsize(request.dtype, request.element_size_bytes)
        if owner_itemsize != request_itemsize:
            raise IncompatibleManifestError(
                f"tensor {request.tensor_name!r} element size mismatch: "
                f"source={owner_itemsize}, request={request_itemsize}"
            )
        _validate_layout_tags(owner, request)
        if classify_quantization_scope(owner, request) == "fallback-required":
            raise QuantizationMetadataError(
                f"tensor {request.tensor_name!r} requires global quantization "
                "metadata; use a fallback path instead of zero-copy resharding"
            )


def _validate_layout_tags(ownership: SliceOwnership, request: SliceRequest) -> None:
    for key in _STRICT_LAYOUT_KEYS:
        source_value = ownership.layout_tags.get(key)
        target_value = request.layout_tags.get(key)
        if (
            source_value is not None
            and target_value is not None
            and source_value != target_value
        ):
            raise IncompatibleManifestError(
                f"layout tag {key!r} mismatch for tensor {request.tensor_name!r}: "
                f"source={source_value!r}, request={target_value!r}"
            )


def _partition_request_range(
    requested_range: TensorRange,
    owners: Sequence[SliceOwnership],
) -> list[TensorRange]:
    boundaries: list[set[int]] = [
        {start, end} for start, end in requested_range
    ]
    for owner in owners:
        overlap = intersect_ranges(requested_range, owner.source_range)
        if overlap is None:
            continue
        for axis, (start, end) in enumerate(overlap):
            boundaries[axis].add(start)
            boundaries[axis].add(end)

    axis_intervals = []
    for axis_boundaries in boundaries:
        ordered = sorted(axis_boundaries)
        axis_intervals.append(
            [(ordered[i], ordered[i + 1]) for i in range(len(ordered) - 1)]
        )
    return [tuple(cell) for cell in product(*axis_intervals)]


def _contains_range(container: TensorRange, candidate: TensorRange) -> bool:
    return all(
        container_start <= candidate_start and candidate_end <= container_end
        for (container_start, container_end), (candidate_start, candidate_end)
        in zip(container, candidate)
    )


def _contiguous_segment_ranges(
    cell: TensorRange,
    owner: SliceOwnership,
    request: SliceRequest,
) -> list[TensorRange]:
    ndim = len(cell)
    source_shape = range_extents(owner.source_range)
    target_shape = request.target_shape
    source_strides = owner.strides or row_major_strides(source_shape)
    target_strides = request.destination_strides or row_major_strides(target_shape)

    if source_strides[-1] != 1 or target_strides[-1] != 1:
        return list(_element_ranges(cell))

    first_contiguous_axis = ndim - 1
    for axis in range(ndim - 2, -1, -1):
        suffix_full = all(
            _axis_full_in_parent(cell, owner.source_range, suffix_axis)
            and _axis_full_in_parent(cell, request.requested_range, suffix_axis)
            for suffix_axis in range(axis + 1, ndim)
        )
        if not suffix_full:
            break
        if not (
            _stride_merges_axis(source_shape, source_strides, axis)
            and _stride_merges_axis(target_shape, target_strides, axis)
        ):
            break
        first_contiguous_axis = axis

    if first_contiguous_axis == 0:
        return [cell]

    prefix_axes = [
        range(cell[axis][0], cell[axis][1])
        for axis in range(first_contiguous_axis)
    ]
    segments: list[TensorRange] = []
    for prefix_coords in product(*prefix_axes):
        axes: list[tuple[int, int]] = []
        for axis in range(ndim):
            if axis < first_contiguous_axis:
                coord = prefix_coords[axis]
                axes.append((coord, coord + 1))
            else:
                axes.append(cell[axis])
        segments.append(tuple(axes))
    return segments


def _element_ranges(cell: TensorRange) -> list[TensorRange]:
    axes = [range(start, end) for start, end in cell]
    return [
        tuple((coord, coord + 1) for coord in coords)
        for coords in product(*axes)
    ]


def _axis_full_in_parent(cell: TensorRange, parent: TensorRange, axis: int) -> bool:
    return cell[axis] == parent[axis]


def _stride_merges_axis(
    shape: tuple[int, ...],
    strides: tuple[int, ...],
    axis: int,
) -> bool:
    return strides[axis] == strides[axis + 1] * shape[axis + 1]


def _build_segment_plan(
    owner: SliceOwnership,
    request: SliceRequest,
    segment_range: TensorRange,
) -> SegmentPlan:
    element_size = dtype_itemsize(request.dtype, request.element_size_bytes)
    source_offset = owner.storage_offset_bytes + (
        _element_offset(segment_range, owner.source_range, owner.strides) * element_size
    )
    target_offset = request.target_offset_bytes + (
        _element_offset(
            segment_range,
            request.requested_range,
            request.destination_strides,
        ) * element_size
    )
    byte_count = range_volume(segment_range) * element_size
    return SegmentPlan(
        source_id=owner.stable_source_id,
        worker_id=owner.worker_id,
        tensor_name=owner.tensor_name,
        source_range=segment_range,
        target_range=segment_range,
        source_byte_offset=source_offset,
        target_byte_offset=target_offset,
        bytes=byte_count,
        lease_version=owner.source_lease,
        retry_policy=RetryPolicy.REPLAN_FROM_ALTERNATE,
        nixl_descriptor_id=owner.nixl_descriptor_id,
        target_id=request.stable_target_id,
        target_runtime=request.runtime_framework,
        worker_rank=owner.worker_rank,
    )


def _element_offset(
    global_range: TensorRange,
    parent_range: TensorRange,
    strides: tuple[int, ...] | None,
) -> int:
    parent_shape = range_extents(parent_range)
    effective_strides = strides or row_major_strides(parent_shape)
    coords = [
        global_axis_start - parent_axis_start
        for (global_axis_start, _), (parent_axis_start, _) in zip(
            global_range,
            parent_range,
        )
    ]
    return sum(coord * stride for coord, stride in zip(coords, effective_strides))
