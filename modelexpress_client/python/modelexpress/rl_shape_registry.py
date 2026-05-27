# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shape-registry helpers for framework-agnostic RL weight transfer."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch


def shape_registry_from_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    tensor_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build shape/dtype metadata needed to allocate receive buffers."""
    metadata_by_name = tensor_metadata or {}
    unknown_names = sorted(set(metadata_by_name) - set(tensors))
    if unknown_names:
        raise ValueError(f"shape metadata references unknown tensors: {unknown_names}")

    registry = {}
    for name, tensor in tensors.items():
        extra = metadata_by_name.get(name, {})
        if not isinstance(extra, Mapping):
            raise ValueError(f"shape metadata for {name!r} must be an object")
        entry = dict(extra)
        _validate_shape_metadata_matches_tensor(name, entry, tensor)
        _normalize_expert_metadata(name, entry, tensor)
        entry["shape"] = list(tensor.shape)
        entry["dtype"] = str(tensor.dtype)
        registry[name] = entry
    return registry


def infer_expert_axis_from_shape(
    shape: Sequence[int],
    expert_ids: Iterable[int],
    *,
    tensor_name: str = "",
) -> int:
    """Infer the local expert dimension when tensor shape is unambiguous."""
    expert_count = len(tuple(expert_ids))
    if expert_count <= 0:
        raise ValueError("expert_ids must not be empty when inferring expert_axis")

    normalized_shape = tuple(int(dim) for dim in shape)
    matches = [axis for axis, dim in enumerate(normalized_shape) if dim == expert_count]
    label = f" for {tensor_name!r}" if tensor_name else ""
    if not matches:
        raise ValueError(
            f"cannot infer expert_axis{label}: no tensor dimension matches "
            f"{expert_count} expert IDs"
        )
    if len(matches) > 1:
        raise ValueError(
            f"cannot infer expert_axis{label}: dimensions {matches} all match "
            f"{expert_count} expert IDs; specify expert_axis explicitly"
        )
    return matches[0]


def torch_dtype_from_string(dtype: str) -> torch.dtype:
    """Parse dtype strings emitted by ``str(torch.dtype)``."""
    name = dtype.removeprefix("torch.")
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported tensor dtype {dtype!r}")
    return value


def allocate_tensors_from_shape_registry(
    shape_registry: dict[str, Any],
    *,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Allocate empty tensors from shape registry metadata."""
    tensors = {}
    for name, entry in shape_registry.items():
        if not isinstance(entry, dict):
            raise ValueError(f"shape registry entry for {name!r} must be an object")
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        if not isinstance(shape, list) or dtype is None:
            raise ValueError(
                f"shape registry entry for {name!r} must include shape and dtype"
            )
        tensors[name] = torch.empty(
            tuple(int(dim) for dim in shape),
            dtype=torch_dtype_from_string(str(dtype)),
            device=device,
        )
    return tensors


def _validate_shape_metadata_matches_tensor(
    name: str,
    entry: Mapping[str, Any],
    tensor: torch.Tensor,
) -> None:
    shape = entry.get("shape")
    if shape is not None and tuple(int(dim) for dim in shape) != tuple(tensor.shape):
        raise ValueError(f"shape metadata for {name!r} does not match tensor shape")
    dtype = entry.get("dtype")
    if dtype is not None and str(dtype) != str(tensor.dtype):
        raise ValueError(f"shape metadata for {name!r} does not match tensor dtype")


def _normalize_expert_metadata(
    name: str,
    entry: dict[str, Any],
    tensor: torch.Tensor,
) -> None:
    if "expert_ids" not in entry:
        if entry.get("expert_axis") is not None:
            raise ValueError(
                f"shape metadata for {name!r} expert_axis requires expert_ids"
            )
        return

    expert_ids = entry.get("expert_ids")
    if not _is_ordered_expert_sequence(expert_ids):
        raise ValueError(
            f"shape metadata for {name!r} expert_ids must be a list or tuple"
        )

    normalized_ids = [int(expert_id) for expert_id in expert_ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError(
            f"shape metadata for {name!r} expert_ids must not contain duplicates"
        )
    entry["expert_ids"] = normalized_ids

    if not normalized_ids:
        if entry.get("expert_axis") is not None:
            raise ValueError(
                f"shape metadata for {name!r} expert_axis requires expert_ids"
            )
        return

    expert_axis = entry.get("expert_axis")
    if expert_axis is None:
        entry["expert_axis"] = infer_expert_axis_from_shape(
            tensor.shape,
            normalized_ids,
            tensor_name=name,
        )
        return

    axis = int(expert_axis)
    if axis < 0:
        axis += tensor.ndim
    if axis < 0 or axis >= tensor.ndim:
        raise ValueError(f"shape metadata for {name!r} expert_axis is out of range")
    if int(tensor.shape[axis]) != len(normalized_ids):
        raise ValueError(
            f"shape metadata for {name!r} expert_axis dimension must match "
            "expert_ids length"
        )
    entry["expert_axis"] = axis


def _is_ordered_expert_sequence(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    return isinstance(value, (list, tuple))
