# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receiver-side helpers for runtime-owned refit tensors."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch

from .resharding import (
    SegmentPlan,
    SliceRequest,
    TensorRange,
    normalize_range,
    range_extents,
)


@dataclass(frozen=True)
class InstalledSegment:
    """A copied segment for smoke tests and receiver-side metrics."""

    tensor_name: str
    target_key: str
    target_range: TensorRange
    bytes: int
    source_id: str


def build_receiver_requests_from_runtime_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    model_name: str,
    model_version: str,
    runtime_framework: str,
    requested_ranges: Mapping[str, TensorRange] | None = None,
    layout_tags_by_tensor: Mapping[str, Mapping[str, str | int | bool]] | None = None,
    target_id_prefix: str = "",
) -> list[SliceRequest]:
    """Create receiver-side requests from framework-owned target tensors.

    ``requested_ranges`` maps each local runtime tensor to the global tensor
    range it represents. If omitted, the tensor is treated as a full-tensor
    request.
    """

    requests: list[SliceRequest] = []
    for tensor_name, tensor in tensors.items():
        requested_range = _requested_range_for_tensor(
            tensor_name,
            tensor,
            requested_ranges=requested_ranges,
        )
        layout_tags = _target_layout_tags(
            tensor,
            runtime_framework=runtime_framework,
        )
        if layout_tags_by_tensor and tensor_name in layout_tags_by_tensor:
            layout_tags.update(layout_tags_by_tensor[tensor_name])

        target_id = f"{target_id_prefix}:{tensor_name}" if target_id_prefix else ""
        requests.append(
            SliceRequest(
                tensor_name=tensor_name,
                requested_range=requested_range,
                target_shape=tuple(int(dim) for dim in tensor.shape),
                dtype=_torch_dtype_name(tensor.dtype),
                target_id=target_id,
                model_name=model_name,
                model_version=model_version,
                destination_strides=tuple(int(stride) for stride in tensor.stride()),
                runtime_framework=runtime_framework,
                layout_tags=layout_tags,
                element_size_bytes=int(tensor.element_size()),
            )
        )
    return requests


def install_segment_payloads_into_runtime_tensors(
    segment_payloads: Iterable[tuple[SegmentPlan, torch.Tensor]],
    target_tensors: Mapping[str, torch.Tensor],
    *,
    target_ranges: Mapping[str, TensorRange] | None = None,
    allow_dtype_cast: bool = False,
) -> list[InstalledSegment]:
    """Copy planned segment payloads into runtime-owned target tensors.

    This is the receiver install equivalent of NIXL landing bytes into a target
    buffer. Tests pass payload tensors directly; the production data plane should
    provide those bytes through one-sided reads.
    """

    installed: list[InstalledSegment] = []
    for plan, payload in segment_payloads:
        target_key, target = _resolve_target_tensor(plan, target_tensors)
        base_range = _target_base_range(
            plan,
            target,
            target_key=target_key,
            target_ranges=target_ranges,
        )
        local_slices = _local_slices(plan.target_range, base_range)
        expected_shape = range_extents(plan.target_range)
        prepared_payload = _prepare_payload(
            payload,
            target=target,
            expected_shape=expected_shape,
            allow_dtype_cast=allow_dtype_cast,
        )

        with torch.no_grad():
            target[local_slices].copy_(prepared_payload)

        installed.append(
            InstalledSegment(
                tensor_name=plan.tensor_name,
                target_key=target_key,
                target_range=plan.target_range,
                bytes=plan.bytes,
                source_id=plan.source_id,
            )
        )
    return installed


def _requested_range_for_tensor(
    tensor_name: str,
    tensor: torch.Tensor,
    *,
    requested_ranges: Mapping[str, TensorRange] | None,
) -> TensorRange:
    if requested_ranges and tensor_name in requested_ranges:
        requested_range = normalize_range(requested_ranges[tensor_name])
    else:
        requested_range = tuple((0, int(dim)) for dim in tensor.shape)
    if range_extents(requested_range) != tuple(int(dim) for dim in tensor.shape):
        raise ValueError(
            f"requested range for {tensor_name!r} does not match target tensor shape"
        )
    return requested_range


def _target_layout_tags(
    tensor: torch.Tensor,
    *,
    runtime_framework: str,
) -> dict[str, str | int | bool]:
    return {
        "runtime_framework": runtime_framework,
        "storage_layout": "row-major" if tensor.is_contiguous() else "strided",
        "target_contiguous": bool(tensor.is_contiguous()),
    }


def _torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _resolve_target_tensor(
    plan: SegmentPlan,
    target_tensors: Mapping[str, torch.Tensor],
) -> tuple[str, torch.Tensor]:
    if plan.target_id and plan.target_id in target_tensors:
        return plan.target_id, target_tensors[plan.target_id]
    if plan.tensor_name in target_tensors:
        return plan.tensor_name, target_tensors[plan.tensor_name]
    raise KeyError(
        f"no target tensor for plan target_id={plan.target_id!r} "
        f"tensor_name={plan.tensor_name!r}"
    )


def _target_base_range(
    plan: SegmentPlan,
    target: torch.Tensor,
    *,
    target_key: str,
    target_ranges: Mapping[str, TensorRange] | None,
) -> TensorRange:
    if target_ranges:
        if target_key in target_ranges:
            return normalize_range(target_ranges[target_key])
        if plan.target_id and plan.target_id in target_ranges:
            return normalize_range(target_ranges[plan.target_id])
        if plan.tensor_name in target_ranges:
            return normalize_range(target_ranges[plan.tensor_name])
    return tuple((0, int(dim)) for dim in target.shape)


def _local_slices(target_range: TensorRange, base_range: TensorRange) -> tuple[slice, ...]:
    target_range = normalize_range(target_range)
    base_range = normalize_range(base_range)
    if len(target_range) != len(base_range):
        raise ValueError("target range rank must match target tensor base range")

    slices: list[slice] = []
    for axis, ((start, end), (base_start, base_end)) in enumerate(
        zip(target_range, base_range)
    ):
        if start < base_start or end > base_end:
            raise ValueError(
                f"target_range axis {axis} {(start, end)} is outside target "
                f"base range {(base_start, base_end)}"
            )
        slices.append(slice(start - base_start, end - base_start))
    return tuple(slices)


def _prepare_payload(
    payload: torch.Tensor,
    *,
    target: torch.Tensor,
    expected_shape: tuple[int, ...],
    allow_dtype_cast: bool,
) -> torch.Tensor:
    if tuple(int(dim) for dim in payload.shape) != expected_shape:
        payload = payload.reshape(expected_shape)
    if payload.dtype != target.dtype:
        if not allow_dtype_cast:
            raise TypeError(
                f"payload dtype {payload.dtype} does not match target dtype "
                f"{target.dtype}"
            )
        payload = payload.to(dtype=target.dtype)
    if payload.device != target.device:
        payload = payload.to(device=target.device)
    return payload
