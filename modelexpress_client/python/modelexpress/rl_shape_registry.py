# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shape-registry helpers for framework-agnostic RL weight transfer."""

from __future__ import annotations

from collections.abc import Mapping
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
        entry["shape"] = list(tensor.shape)
        entry["dtype"] = str(tensor.dtype)
        registry[name] = entry
    return registry


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
