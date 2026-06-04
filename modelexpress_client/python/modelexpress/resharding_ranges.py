# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Range math helpers for cross-parallelism tensor resharding."""

from __future__ import annotations

import json
from typing import Sequence

TensorRange = tuple[tuple[int, int], ...]

def normalize_range(value: Sequence[Sequence[int]]) -> TensorRange:
    """Normalize and validate a half-open tensor range."""

    normalized = tuple((int(start), int(end)) for start, end in value)
    if not normalized:
        raise ValueError("tensor ranges must have at least one axis")
    for start, end in normalized:
        if start < 0:
            raise ValueError(f"range start must be non-negative: {(start, end)}")
        if end <= start:
            raise ValueError(f"range end must be greater than start: {(start, end)}")
    return normalized


def range_to_list(value: TensorRange) -> list[list[int]]:
    """Convert an internal range tuple to JSON-friendly lists."""

    return [[start, end] for start, end in value]


def range_to_json_key(value: TensorRange) -> str:
    """Return a compact stable range key for metrics."""

    return json.dumps(range_to_list(value), separators=(",", ":"))


def intersect_ranges(left: TensorRange, right: TensorRange) -> TensorRange | None:
    """Return the rectangular intersection of two half-open ranges."""

    left = normalize_range(left)
    right = normalize_range(right)
    if len(left) != len(right):
        raise ValueError("ranges must have the same rank")

    axes: list[tuple[int, int]] = []
    for (left_start, left_end), (right_start, right_end) in zip(left, right):
        start = max(left_start, right_start)
        end = min(left_end, right_end)
        if end <= start:
            return None
        axes.append((start, end))
    return tuple(axes)


def range_extents(value: TensorRange) -> tuple[int, ...]:
    """Return the size of each range axis."""

    value = normalize_range(value)
    return tuple(end - start for start, end in value)


def range_volume(value: TensorRange) -> int:
    """Return the element count in a rectangular range."""

    volume = 1
    for extent in range_extents(value):
        volume *= extent
    return volume

def row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return element strides for a contiguous row-major tensor."""

    shape = _normalize_shape(shape)
    strides: list[int] = [1] * len(shape)
    running = 1
    for axis in range(len(shape) - 1, -1, -1):
        strides[axis] = running
        running *= shape[axis]
    return tuple(strides)


def _validate_range_inside_shape(
    value: TensorRange,
    shape: tuple[int, ...],
    label: str,
) -> None:
    if len(value) != len(shape):
        raise ValueError(f"{label} rank must match shape rank")
    for axis, ((start, end), dim) in enumerate(zip(value, shape)):
        if end > dim:
            raise ValueError(
                f"{label} axis {axis} range {(start, end)} exceeds shape dim {dim}"
            )


def _normalize_shape(shape: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(int(dim) for dim in shape)
    if not normalized:
        raise ValueError("shape must have at least one dimension")
    if any(dim <= 0 for dim in normalized):
        raise ValueError("shape dimensions must be positive")
    return normalized
