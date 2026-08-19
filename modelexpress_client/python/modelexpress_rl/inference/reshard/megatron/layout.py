# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lower native Megatron tensor geometry into the reshard transfer planner.

Megatron publishes rank-local tensors in its native fused representation.
Translation into HF/vLLM names therefore happens after transfer, but transfer
planning must still use the inference rank's destination TP slice. This module
creates the same ``RecordedCopy`` contract produced by loader capture, targeting
native staging buffers that the Megatron translator consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

from modelexpress.refit.reshard.slice_plan import _row_major_strides
from modelexpress.refit.reshard.types import CaptureResult, RecordedCopy

REPLICATED = "replicated"
COLUMN_ROLES = frozenset(
    {
        "column",
        "qkv_column",
        "gated_mlp_column",
        "vocab_parallel",
        "expert_column",
    }
)
ROW_ROLES = frozenset({"row", "expert_row"})
SUPPORTED_ROLES = COLUMN_ROLES | ROW_ROLES | {REPLICATED}


def _is_integral(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


@dataclass(frozen=True)
class MegatronTargetSpec:
    """One native Megatron tensor requested by an inference TP rank."""

    source_name: str
    role: str
    global_shape: tuple[int, ...]
    dtype: Any
    shard_axis: int | None = None
    staging_name: str | None = None
    descriptor_extras: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in SUPPORTED_ROLES:
            raise ValueError(f"unsupported Megatron role {self.role!r}")
        if not self.global_shape or any(
            not _is_integral(extent) or extent <= 0 for extent in self.global_shape
        ):
            raise ValueError(
                f"{self.source_name}: global_shape must contain positive integer extents"
            )


@dataclass(frozen=True)
class MegatronTargetLayout:
    tp_size: int
    tp_rank: int

    def __post_init__(self) -> None:
        if not _is_integral(self.tp_size):
            raise ValueError("tp_size must be an integer")
        if not _is_integral(self.tp_rank):
            raise ValueError("tp_rank must be an integer")
        if self.tp_size < 1:
            raise ValueError("tp_size must be at least 1")
        if not 0 <= self.tp_rank < self.tp_size:
            raise ValueError(f"tp_rank {self.tp_rank} is outside [0, {self.tp_size})")


def _role_axis(spec: MegatronTargetSpec) -> int | None:
    if spec.role == REPLICATED:
        if spec.shard_axis is not None:
            raise ValueError(
                f"{spec.source_name}: replicated tensor cannot set shard_axis"
            )
        return None
    if spec.shard_axis is not None:
        axis = int(spec.shard_axis)
    elif spec.role == "expert_column":
        axis = 1 if spec.descriptor_extras.get("expert_layout") == "leading_axis" else 0
    elif spec.role == "expert_row":
        axis = 2 if spec.descriptor_extras.get("expert_layout") == "leading_axis" else 1
    elif spec.role in COLUMN_ROLES:
        axis = 0
    else:
        axis = 1
    if not 0 <= axis < len(spec.global_shape):
        raise ValueError(
            f"{spec.source_name}: shard axis {axis} is invalid for "
            f"shape {spec.global_shape}"
        )
    return axis


def _partition(extent: int, layout: MegatronTargetLayout) -> tuple[int, int]:
    """Return the exact equal Megatron TP partition for this target rank."""

    if extent % layout.tp_size:
        raise ValueError(
            f"extent {extent} is not divisible by target TP {layout.tp_size}; "
            "explicit padded geometry is required"
        )
    width = extent // layout.tp_size
    lo = layout.tp_rank * width
    return lo, lo + width


def lower_megatron_target(
    specs: list[MegatronTargetSpec],
    layout: MegatronTargetLayout,
) -> tuple[CaptureResult, dict[str, tuple[tuple[int, ...], Any]]]:
    """Create reshard capture records and native staging layouts for one TP rank.

    Each copy describes a destination-owned narrow of the global native
    Megatron tensor. ``plan_transfer`` then intersects that narrow with the
    trainer's published TP/PP/EP shards and emits only the required reads.
    """

    copies: list[RecordedCopy] = []
    param_layout: dict[str, tuple[tuple[int, ...], Any]] = {}
    staging_sources: dict[str, str] = {}
    for spec in specs:
        staging_name = spec.staging_name or spec.source_name
        previous_source = staging_sources.get(staging_name)
        if previous_source is not None:
            raise ValueError(
                f"duplicate staging_name {staging_name!r} for "
                f"{previous_source!r} and {spec.source_name!r}"
            )
        staging_sources[staging_name] = spec.source_name

        axis = _role_axis(spec)
        local_shape = list(spec.global_shape)
        op_chain: tuple = ()
        if axis is not None and layout.tp_size > 1:
            lo, hi = _partition(int(spec.global_shape[axis]), layout)
            local_shape[axis] = hi - lo
            op_chain = (("narrow", (axis, lo, hi - lo), ()),)

        local_shape_tuple = tuple(local_shape)
        copies.append(
            RecordedCopy(
                src_name=spec.source_name,
                op_chain=op_chain,
                param_name=staging_name,
                dest_offset=0,
                dest_shape=local_shape_tuple,
                dest_stride=tuple(_row_major_strides(local_shape_tuple)),
                dest_dtype=spec.dtype,
            )
        )
        param_layout[staging_name] = (local_shape_tuple, spec.dtype)

    return CaptureResult(copies=copies), param_layout


__all__ = [
    "MegatronTargetLayout",
    "MegatronTargetSpec",
    "lower_megatron_target",
]
