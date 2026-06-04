# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner for cross-parallelism tensor resharding.

This module is intentionally independent from the live RDMA loader. It models
the metadata and range-intersection step needed before ModelExpress can issue
multi-source NIXL reads into one target tensor buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import time
from itertools import product
from pathlib import Path
from typing import Any, Sequence

TensorRange = tuple[tuple[int, int], ...]


class QuantizationScope(str, Enum):
    """How quantization metadata is scoped for a tensor slice."""

    ABSENT = "absent"
    LOCAL = "local"
    GLOBAL_REQUIRED = "global-required"
    GENERATED_ON_TARGET = "generated-on-target"


class RetryPolicy(str, Enum):
    """Planner hint for segment-level retry behavior."""

    REPLAN_FROM_ALTERNATE = "replan-from-alternate-holder"
    FALLBACK_TO_STORAGE = "fallback-to-storage"
    FAIL_FAST = "fail-fast"


class TransferStrategy(str, Enum):
    """High-level simulator recommendation."""

    DIRECT_BIPARTITE = "direct-bipartite-p2p"
    PRIMARY_REPLICA_FANOUT = "primary-replica-fanout"


class CompetitiveStrategy(str, Enum):
    """Strategies compared in the broader refit simulator."""

    MX_DIRECT_BIPARTITE = "mx-direct-bipartite-p2p"
    MX_PRIMARY_REPLICA_FANOUT = "mx-primary-replica-fanout"
    NCCL_RESHARD = "nccl-reshard-fixed-membership"
    CHECKPOINT_ENGINE_FULL_GATHER = "checkpoint-engine-full-gather-apply"


class ReshardingPlanError(ValueError):
    """Base class for resharding planner failures."""


class CoverageError(ReshardingPlanError):
    """Requested ranges are missing coverage or covered more than once."""

    def __init__(
        self,
        message: str,
        *,
        missing_ranges: Sequence[TensorRange] | None = None,
        duplicate_ranges: Sequence[TensorRange] | None = None,
    ):
        super().__init__(message)
        self.missing_ranges = list(missing_ranges or [])
        self.duplicate_ranges = list(duplicate_ranges or [])


class IncompatibleManifestError(ReshardingPlanError):
    """Source ownership metadata cannot satisfy the receiver request."""


class QuantizationMetadataError(ReshardingPlanError):
    """Tensor requires global quantization metadata before zero-copy assembly."""


_DTYPE_ITEMSIZE_BYTES: dict[str, int] = {
    "bool": 1,
    "bfloat16": 2,
    "bf16": 2,
    "float16": 2,
    "half": 2,
    "fp16": 2,
    "float32": 4,
    "fp32": 4,
    "float": 4,
    "float64": 8,
    "double": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "uint16": 2,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "fp8": 1,
}

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


@dataclass(frozen=True)
class SliceOwnership:
    """A source-published tensor range and its transfer metadata."""

    model_name: str
    model_version: str
    tensor_name: str
    global_shape: tuple[int, ...]
    dtype: str
    source_range: TensorRange
    worker_id: str
    source_id: str = ""
    worker_rank: int | None = None
    source_lease: str = ""
    nixl_descriptor_id: str = ""
    storage_offset_bytes: int = 0
    strides: tuple[int, ...] | None = None
    contiguous: bool = True
    layout_tags: dict[str, str | int | bool] = field(default_factory=dict)
    quantization_scope: QuantizationScope | str = QuantizationScope.ABSENT
    element_size_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "global_shape", _normalize_shape(self.global_shape))
        object.__setattr__(self, "source_range", normalize_range(self.source_range))
        object.__setattr__(
            self,
            "quantization_scope",
            QuantizationScope(self.quantization_scope),
        )
        if self.strides is not None:
            object.__setattr__(self, "strides", tuple(int(s) for s in self.strides))
        _validate_range_inside_shape(self.source_range, self.global_shape, "source_range")
        if self.strides is not None and len(self.strides) != len(self.global_shape):
            raise ValueError("source strides must have one entry per tensor axis")
        if not self.contiguous and self.strides is None:
            raise ValueError("non-contiguous ownership must publish explicit strides")

    @property
    def stable_source_id(self) -> str:
        """Return the source identifier used in plans."""

        return self.source_id or self.worker_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "tensor_name": self.tensor_name,
            "global_shape": list(self.global_shape),
            "dtype": self.dtype,
            "source_range": range_to_list(self.source_range),
            "worker_id": self.worker_id,
            "source_id": self.source_id,
            "worker_rank": self.worker_rank,
            "source_lease": self.source_lease,
            "nixl_descriptor_id": self.nixl_descriptor_id,
            "storage_offset_bytes": self.storage_offset_bytes,
            "strides": list(self.strides) if self.strides is not None else None,
            "contiguous": self.contiguous,
            "layout_tags": dict(self.layout_tags),
            "quantization_scope": self.quantization_scope.value,
            "element_size_bytes": self.element_size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SliceOwnership":
        return cls(
            model_name=data["model_name"],
            model_version=data["model_version"],
            tensor_name=data["tensor_name"],
            global_shape=tuple(data["global_shape"]),
            dtype=data["dtype"],
            source_range=normalize_range(data["source_range"]),
            worker_id=data["worker_id"],
            source_id=data.get("source_id", ""),
            worker_rank=data.get("worker_rank"),
            source_lease=data.get("source_lease", ""),
            nixl_descriptor_id=data.get("nixl_descriptor_id", ""),
            storage_offset_bytes=int(data.get("storage_offset_bytes", 0)),
            strides=tuple(data["strides"]) if data.get("strides") is not None else None,
            contiguous=bool(data.get("contiguous", True)),
            layout_tags=dict(data.get("layout_tags", {})),
            quantization_scope=data.get("quantization_scope", QuantizationScope.ABSENT),
            element_size_bytes=data.get("element_size_bytes"),
        )


@dataclass(frozen=True)
class SliceRequest:
    """A receiver-side tensor range request."""

    tensor_name: str
    requested_range: TensorRange
    target_shape: tuple[int, ...]
    dtype: str
    target_id: str = ""
    model_name: str = ""
    model_version: str = ""
    target_offset_bytes: int = 0
    destination_strides: tuple[int, ...] | None = None
    runtime_framework: str = ""
    layout_tags: dict[str, str | int | bool] = field(default_factory=dict)
    quantization_scope: QuantizationScope | str = QuantizationScope.ABSENT
    element_size_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_range", normalize_range(self.requested_range))
        object.__setattr__(self, "target_shape", _normalize_shape(self.target_shape))
        object.__setattr__(
            self,
            "quantization_scope",
            QuantizationScope(self.quantization_scope),
        )
        if self.destination_strides is not None:
            object.__setattr__(
                self,
                "destination_strides",
                tuple(int(s) for s in self.destination_strides),
            )
        if len(self.requested_range) != len(self.target_shape):
            raise ValueError("target_shape rank must match requested_range rank")
        if range_extents(self.requested_range) != self.target_shape:
            raise ValueError("target_shape must equal requested_range extents")
        if (
            self.destination_strides is not None
            and len(self.destination_strides) != len(self.target_shape)
        ):
            raise ValueError("destination_strides must have one entry per target axis")

    @property
    def stable_target_id(self) -> str:
        """Return a stable target id for metrics."""

        if self.target_id:
            return self.target_id
        return (
            f"{self.runtime_framework}:{self.tensor_name}:"
            f"{range_to_json_key(self.requested_range)}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tensor_name": self.tensor_name,
            "requested_range": range_to_list(self.requested_range),
            "target_shape": list(self.target_shape),
            "dtype": self.dtype,
            "target_id": self.target_id,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "target_offset_bytes": self.target_offset_bytes,
            "destination_strides": (
                list(self.destination_strides)
                if self.destination_strides is not None
                else None
            ),
            "runtime_framework": self.runtime_framework,
            "layout_tags": dict(self.layout_tags),
            "quantization_scope": self.quantization_scope.value,
            "element_size_bytes": self.element_size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SliceRequest":
        return cls(
            tensor_name=data["tensor_name"],
            requested_range=normalize_range(data["requested_range"]),
            target_shape=tuple(data["target_shape"]),
            dtype=data["dtype"],
            target_id=data.get("target_id", ""),
            model_name=data.get("model_name", ""),
            model_version=data.get("model_version", ""),
            target_offset_bytes=int(data.get("target_offset_bytes", 0)),
            destination_strides=(
                tuple(data["destination_strides"])
                if data.get("destination_strides") is not None
                else None
            ),
            runtime_framework=data.get("runtime_framework", ""),
            layout_tags=dict(data.get("layout_tags", {})),
            quantization_scope=data.get("quantization_scope", QuantizationScope.ABSENT),
            element_size_bytes=data.get("element_size_bytes"),
        )


@dataclass(frozen=True)
class SegmentPlan:
    """A contiguous transfer segment from one source range to one target range."""

    source_id: str
    worker_id: str
    tensor_name: str
    source_range: TensorRange
    target_range: TensorRange
    source_byte_offset: int
    target_byte_offset: int
    bytes: int
    lease_version: str = ""
    retry_policy: RetryPolicy | str = RetryPolicy.REPLAN_FROM_ALTERNATE
    nixl_descriptor_id: str = ""
    target_id: str = ""
    target_runtime: str = ""
    worker_rank: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_range", normalize_range(self.source_range))
        object.__setattr__(self, "target_range", normalize_range(self.target_range))
        object.__setattr__(self, "retry_policy", RetryPolicy(self.retry_policy))
        if range_extents(self.source_range) != range_extents(self.target_range):
            raise ValueError("source_range and target_range must have matching extents")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "worker_id": self.worker_id,
            "tensor_name": self.tensor_name,
            "source_range": range_to_list(self.source_range),
            "target_range": range_to_list(self.target_range),
            "source_byte_offset": self.source_byte_offset,
            "target_byte_offset": self.target_byte_offset,
            "bytes": self.bytes,
            "lease_version": self.lease_version,
            "retry_policy": self.retry_policy.value,
            "nixl_descriptor_id": self.nixl_descriptor_id,
            "target_id": self.target_id,
            "target_runtime": self.target_runtime,
            "worker_rank": self.worker_rank,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SegmentPlan":
        return cls(
            source_id=data["source_id"],
            worker_id=data["worker_id"],
            tensor_name=data["tensor_name"],
            source_range=normalize_range(data["source_range"]),
            target_range=normalize_range(data["target_range"]),
            source_byte_offset=int(data["source_byte_offset"]),
            target_byte_offset=int(data["target_byte_offset"]),
            bytes=int(data["bytes"]),
            lease_version=data.get("lease_version", ""),
            retry_policy=data.get("retry_policy", RetryPolicy.REPLAN_FROM_ALTERNATE),
            nixl_descriptor_id=data.get("nixl_descriptor_id", ""),
            target_id=data.get("target_id", ""),
            target_runtime=data.get("target_runtime", ""),
            worker_rank=data.get("worker_rank"),
        )


@dataclass(frozen=True)
class BandwidthAssumptions:
    """Bandwidth values used by the simulator."""

    trainer_to_inference_gbps: float
    inference_to_inference_gbps: float

    def __post_init__(self) -> None:
        if self.trainer_to_inference_gbps <= 0:
            raise ValueError("trainer_to_inference_gbps must be positive")
        if self.inference_to_inference_gbps <= 0:
            raise ValueError("inference_to_inference_gbps must be positive")


@dataclass(frozen=True)
class CompetitiveAssumptions:
    """Bandwidth and latency values for competitive refit comparisons."""

    trainer_to_inference_gbps: float
    inference_to_inference_gbps: float
    nccl_reshard_gbps: float
    checkpoint_storage_gbps: float
    per_segment_latency_us: float = 0.0
    planner_duration_ms: float = 0.0
    publish_duration_ms: float = 0.0
    activation_install_duration_ms: float = 0.0
    nccl_fixed_overhead_ms: float = 0.0
    checkpoint_fixed_overhead_ms: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "trainer_to_inference_gbps",
            "inference_to_inference_gbps",
            "nccl_reshard_gbps",
            "checkpoint_storage_gbps",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in (
            "per_segment_latency_us",
            "planner_duration_ms",
            "publish_duration_ms",
            "activation_install_duration_ms",
            "nccl_fixed_overhead_ms",
            "checkpoint_fixed_overhead_ms",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")


@dataclass(frozen=True)
class SimulationResult:
    """Stable simulator output suitable for committed JSON artifacts."""

    preferred_strategy: TransferStrategy
    trainer_to_inference_bytes: int
    inference_side_fanout_bytes: int
    redundant_cross_boundary_factor: float
    segment_count: int
    source_count_per_target_tensor: dict[str, int]
    source_balance_bytes: dict[str, int]
    target_balance_bytes: dict[str, int]
    uncovered_ranges: list[TensorRange]
    predicted_bottleneck: str
    planner_duration_ms: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preferred_strategy",
            TransferStrategy(self.preferred_strategy),
        )
        object.__setattr__(
            self,
            "uncovered_ranges",
            [normalize_range(r) for r in self.uncovered_ranges],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferred_strategy": self.preferred_strategy.value,
            "trainer_to_inference_bytes": self.trainer_to_inference_bytes,
            "inference_side_fanout_bytes": self.inference_side_fanout_bytes,
            "redundant_cross_boundary_factor": self.redundant_cross_boundary_factor,
            "segment_count": self.segment_count,
            "source_count_per_target_tensor": dict(self.source_count_per_target_tensor),
            "source_balance_bytes": dict(self.source_balance_bytes),
            "target_balance_bytes": dict(self.target_balance_bytes),
            "uncovered_ranges": [range_to_list(r) for r in self.uncovered_ranges],
            "predicted_bottleneck": self.predicted_bottleneck,
            "planner_duration_ms": self.planner_duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationResult":
        return cls(
            preferred_strategy=data["preferred_strategy"],
            trainer_to_inference_bytes=int(data["trainer_to_inference_bytes"]),
            inference_side_fanout_bytes=int(data["inference_side_fanout_bytes"]),
            redundant_cross_boundary_factor=float(
                data["redundant_cross_boundary_factor"]
            ),
            segment_count=int(data["segment_count"]),
            source_count_per_target_tensor=dict(data["source_count_per_target_tensor"]),
            source_balance_bytes={
                str(k): int(v) for k, v in data["source_balance_bytes"].items()
            },
            target_balance_bytes={
                str(k): int(v) for k, v in data["target_balance_bytes"].items()
            },
            uncovered_ranges=[normalize_range(r) for r in data["uncovered_ranges"]],
            predicted_bottleneck=data["predicted_bottleneck"],
            planner_duration_ms=data.get("planner_duration_ms"),
        )


@dataclass(frozen=True)
class StrategyCost:
    """Estimated cost for one refit strategy."""

    strategy: CompetitiveStrategy | str
    trainer_to_inference_bytes: int
    inference_side_fanout_bytes: int
    trainer_collective_bytes: int
    checkpoint_storage_bytes: int
    segment_count: int
    redundant_cross_boundary_factor: float
    estimated_duration_ms: float
    predicted_bottleneck: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", CompetitiveStrategy(self.strategy))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))

    @property
    def total_network_bytes(self) -> int:
        return (
            self.trainer_to_inference_bytes
            + self.inference_side_fanout_bytes
            + self.trainer_collective_bytes
            + self.checkpoint_storage_bytes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.value,
            "trainer_to_inference_bytes": self.trainer_to_inference_bytes,
            "inference_side_fanout_bytes": self.inference_side_fanout_bytes,
            "trainer_collective_bytes": self.trainer_collective_bytes,
            "checkpoint_storage_bytes": self.checkpoint_storage_bytes,
            "segment_count": self.segment_count,
            "redundant_cross_boundary_factor": self.redundant_cross_boundary_factor,
            "estimated_duration_ms": self.estimated_duration_ms,
            "predicted_bottleneck": self.predicted_bottleneck,
            "total_network_bytes": self.total_network_bytes,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyCost":
        return cls(
            strategy=data["strategy"],
            trainer_to_inference_bytes=int(data["trainer_to_inference_bytes"]),
            inference_side_fanout_bytes=int(data["inference_side_fanout_bytes"]),
            trainer_collective_bytes=int(data["trainer_collective_bytes"]),
            checkpoint_storage_bytes=int(data["checkpoint_storage_bytes"]),
            segment_count=int(data["segment_count"]),
            redundant_cross_boundary_factor=float(
                data["redundant_cross_boundary_factor"]
            ),
            estimated_duration_ms=float(data["estimated_duration_ms"]),
            predicted_bottleneck=data["predicted_bottleneck"],
            notes=tuple(data.get("notes", ())),
        )


@dataclass(frozen=True)
class CompetitiveSimulationResult:
    """Comparison across MX, NCCL-style, and checkpoint-engine refit paths."""

    preferred_strategy: CompetitiveStrategy | str
    costs: tuple[StrategyCost, ...]
    unique_requested_bytes: int
    unique_full_tensor_bytes: int
    target_request_count: int
    trainer_source_count: int
    segment_count: int
    source_count_per_target_tensor: dict[str, int]
    source_balance_bytes: dict[str, int]
    target_balance_bytes: dict[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preferred_strategy",
            CompetitiveStrategy(self.preferred_strategy),
        )
        object.__setattr__(self, "costs", tuple(self.costs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "preferred_strategy": self.preferred_strategy.value,
            "costs": [cost.to_dict() for cost in self.costs],
            "unique_requested_bytes": self.unique_requested_bytes,
            "unique_full_tensor_bytes": self.unique_full_tensor_bytes,
            "target_request_count": self.target_request_count,
            "trainer_source_count": self.trainer_source_count,
            "segment_count": self.segment_count,
            "source_count_per_target_tensor": dict(self.source_count_per_target_tensor),
            "source_balance_bytes": dict(self.source_balance_bytes),
            "target_balance_bytes": dict(self.target_balance_bytes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompetitiveSimulationResult":
        return cls(
            preferred_strategy=data["preferred_strategy"],
            costs=tuple(StrategyCost.from_dict(item) for item in data["costs"]),
            unique_requested_bytes=int(data["unique_requested_bytes"]),
            unique_full_tensor_bytes=int(data["unique_full_tensor_bytes"]),
            target_request_count=int(data["target_request_count"]),
            trainer_source_count=int(data["trainer_source_count"]),
            segment_count=int(data["segment_count"]),
            source_count_per_target_tensor=dict(data["source_count_per_target_tensor"]),
            source_balance_bytes={
                str(k): int(v) for k, v in data["source_balance_bytes"].items()
            },
            target_balance_bytes={
                str(k): int(v) for k, v in data["target_balance_bytes"].items()
            },
        )


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


def dtype_itemsize(dtype: str, override: int | None = None) -> int:
    """Return item size in bytes for common tensor dtypes."""

    if override is not None:
        if override <= 0:
            raise ValueError("element_size_bytes must be positive")
        return int(override)
    key = dtype.removeprefix("torch.").lower()
    if key not in _DTYPE_ITEMSIZE_BYTES:
        raise IncompatibleManifestError(
            f"unknown dtype {dtype!r}; set element_size_bytes explicitly"
        )
    return _DTYPE_ITEMSIZE_BYTES[key]


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


def simulate_resharding(
    ownerships: Sequence[SliceOwnership],
    requests: Sequence[SliceRequest],
    bandwidth: BandwidthAssumptions,
    *,
    include_timing: bool = False,
) -> SimulationResult:
    """Plan requests and estimate direct P2P vs replica fan-out cost."""

    start = time.perf_counter()
    plans = plan_segments(ownerships, requests)
    planner_duration_ms = (time.perf_counter() - start) * 1000 if include_timing else None

    direct_cross_bytes = sum(plan.bytes for plan in plans)
    unique_request_keys = {
        (req.tensor_name, req.requested_range, req.dtype, req.element_size_bytes)
        for req in requests
    }
    unique_cross_bytes = 0
    for key in unique_request_keys:
        tensor_name, requested_range, dtype, element_size_bytes = key
        unique_cross_bytes += range_volume(requested_range) * dtype_itemsize(
            dtype, element_size_bytes
        )
    inference_fanout_bytes = max(0, direct_cross_bytes - unique_cross_bytes)

    trainer_gbps = bandwidth.trainer_to_inference_gbps
    inference_gbps = bandwidth.inference_to_inference_gbps
    direct_seconds = _bytes_to_seconds(direct_cross_bytes, trainer_gbps)
    fanout_seconds = _bytes_to_seconds(unique_cross_bytes, trainer_gbps)
    fanout_seconds += _bytes_to_seconds(inference_fanout_bytes, inference_gbps)

    if fanout_seconds < direct_seconds:
        strategy = TransferStrategy.PRIMARY_REPLICA_FANOUT
        trainer_to_inference_bytes = unique_cross_bytes
    else:
        strategy = TransferStrategy.DIRECT_BIPARTITE
        trainer_to_inference_bytes = direct_cross_bytes

    source_balance: dict[str, int] = {}
    target_balance: dict[str, int] = {}
    target_sources: dict[str, set[str]] = {}
    for plan in plans:
        source_balance[plan.source_id] = source_balance.get(plan.source_id, 0) + plan.bytes
        target_key = plan.target_id or plan.tensor_name
        target_balance[target_key] = target_balance.get(target_key, 0) + plan.bytes
        target_sources.setdefault(target_key, set()).add(plan.source_id)

    bottleneck = _predict_bottleneck(
        trainer_to_inference_bytes=trainer_to_inference_bytes,
        inference_fanout_bytes=inference_fanout_bytes,
        bandwidth=bandwidth,
    )

    factor = (
        direct_cross_bytes / unique_cross_bytes
        if unique_cross_bytes
        else 1.0
    )
    return SimulationResult(
        preferred_strategy=strategy,
        trainer_to_inference_bytes=trainer_to_inference_bytes,
        inference_side_fanout_bytes=inference_fanout_bytes,
        redundant_cross_boundary_factor=round(factor, 6),
        segment_count=len(plans),
        source_count_per_target_tensor={
            key: len(value) for key, value in sorted(target_sources.items())
        },
        source_balance_bytes=dict(sorted(source_balance.items())),
        target_balance_bytes=dict(sorted(target_balance.items())),
        uncovered_ranges=[],
        predicted_bottleneck=bottleneck,
        planner_duration_ms=planner_duration_ms,
    )


def simulate_competitive_refit(
    ownerships: Sequence[SliceOwnership],
    requests: Sequence[SliceRequest],
    assumptions: CompetitiveAssumptions,
) -> CompetitiveSimulationResult:
    """Compare MX/NIXL slice planning against broader refit baselines.

    The NCCL-style and checkpoint-engine baselines intentionally model full
    tensor materialization. That is the conservative comparison point for MX's
    slice-overlap path: if requested ranges are narrow or replicated, MX should
    avoid moving full tensors across the trainer/inference boundary.
    """

    plans = plan_segments(ownerships, requests)
    direct_cross_bytes = sum(plan.bytes for plan in plans)
    unique_requested_bytes = _unique_requested_bytes(requests)
    unique_full_tensor_bytes = _unique_full_tensor_bytes(ownerships, requests)
    inference_fanout_bytes = max(0, direct_cross_bytes - unique_requested_bytes)
    source_balance, target_balance, target_sources = _balance_metrics(plans)
    trainer_source_count = len({owner.stable_source_id for owner in ownerships})
    target_request_count = len(requests)

    mx_direct_duration = _duration_ms(
        trainer_to_inference_bytes=direct_cross_bytes,
        inference_side_fanout_bytes=0,
        trainer_collective_bytes=0,
        checkpoint_storage_bytes=0,
        segment_count=len(plans),
        assumptions=assumptions,
        fixed_overhead_ms=(
            assumptions.publish_duration_ms
            + assumptions.planner_duration_ms
            + assumptions.activation_install_duration_ms
        ),
    )
    mx_fanout_duration = _duration_ms(
        trainer_to_inference_bytes=unique_requested_bytes,
        inference_side_fanout_bytes=inference_fanout_bytes,
        trainer_collective_bytes=0,
        checkpoint_storage_bytes=0,
        segment_count=len(plans),
        assumptions=assumptions,
        fixed_overhead_ms=(
            assumptions.publish_duration_ms
            + assumptions.planner_duration_ms
            + assumptions.activation_install_duration_ms
        ),
    )

    nccl_bytes = unique_full_tensor_bytes * max(1, target_request_count)
    nccl_collective_bytes = unique_full_tensor_bytes * max(
        0,
        trainer_source_count - 1,
    )
    checkpoint_storage_bytes = unique_full_tensor_bytes * (1 + target_request_count)

    costs = (
        StrategyCost(
            strategy=CompetitiveStrategy.MX_DIRECT_BIPARTITE,
            trainer_to_inference_bytes=direct_cross_bytes,
            inference_side_fanout_bytes=0,
            trainer_collective_bytes=0,
            checkpoint_storage_bytes=0,
            segment_count=len(plans),
            redundant_cross_boundary_factor=_ratio(
                direct_cross_bytes,
                unique_requested_bytes,
            ),
            estimated_duration_ms=mx_direct_duration,
            predicted_bottleneck=_predict_competitive_bottleneck(
                trainer_to_inference_bytes=direct_cross_bytes,
                inference_side_fanout_bytes=0,
                trainer_collective_bytes=0,
                checkpoint_storage_bytes=0,
                assumptions=assumptions,
            ),
            notes=(
                "one trainer-to-inference transfer per target request segment",
                "elastic MX leases/versioning remain available",
            ),
        ),
        StrategyCost(
            strategy=CompetitiveStrategy.MX_PRIMARY_REPLICA_FANOUT,
            trainer_to_inference_bytes=unique_requested_bytes,
            inference_side_fanout_bytes=inference_fanout_bytes,
            trainer_collective_bytes=0,
            checkpoint_storage_bytes=0,
            segment_count=len(plans),
            redundant_cross_boundary_factor=_ratio(
                unique_requested_bytes + inference_fanout_bytes,
                unique_requested_bytes,
            ),
            estimated_duration_ms=mx_fanout_duration,
            predicted_bottleneck=_predict_competitive_bottleneck(
                trainer_to_inference_bytes=unique_requested_bytes,
                inference_side_fanout_bytes=inference_fanout_bytes,
                trainer_collective_bytes=0,
                checkpoint_storage_bytes=0,
                assumptions=assumptions,
            ),
            notes=(
                "trainer sends each unique requested slice once",
                "rollout replicas receive inference-side fanout",
            ),
        ),
        StrategyCost(
            strategy=CompetitiveStrategy.NCCL_RESHARD,
            trainer_to_inference_bytes=nccl_bytes,
            inference_side_fanout_bytes=0,
            trainer_collective_bytes=nccl_collective_bytes,
            checkpoint_storage_bytes=0,
            segment_count=target_request_count,
            redundant_cross_boundary_factor=_ratio(nccl_bytes, unique_requested_bytes),
            estimated_duration_ms=(
                _bytes_to_seconds(
                    nccl_bytes + nccl_collective_bytes,
                    assumptions.nccl_reshard_gbps,
                )
                * 1000
                + assumptions.nccl_fixed_overhead_ms
            ),
            predicted_bottleneck="fixed-membership-collective",
            notes=(
                "models fixed-membership homogeneous collective reshaping",
                "full tensor materialization baseline",
            ),
        ),
        StrategyCost(
            strategy=CompetitiveStrategy.CHECKPOINT_ENGINE_FULL_GATHER,
            trainer_to_inference_bytes=0,
            inference_side_fanout_bytes=0,
            trainer_collective_bytes=unique_full_tensor_bytes,
            checkpoint_storage_bytes=checkpoint_storage_bytes,
            segment_count=target_request_count,
            redundant_cross_boundary_factor=_ratio(
                checkpoint_storage_bytes,
                unique_requested_bytes,
            ),
            estimated_duration_ms=(
                _bytes_to_seconds(
                    checkpoint_storage_bytes,
                    assumptions.checkpoint_storage_gbps,
                )
                * 1000
                + _bytes_to_seconds(
                    unique_full_tensor_bytes,
                    assumptions.nccl_reshard_gbps,
                )
                * 1000
                + assumptions.checkpoint_fixed_overhead_ms
            ),
            predicted_bottleneck="checkpoint-storage",
            notes=(
                "models trainer full gather plus inference-side apply",
                "no trainer-side inference-layout coupling required",
            ),
        ),
    )
    preferred = min(costs, key=lambda cost: cost.estimated_duration_ms).strategy
    return CompetitiveSimulationResult(
        preferred_strategy=preferred,
        costs=costs,
        unique_requested_bytes=unique_requested_bytes,
        unique_full_tensor_bytes=unique_full_tensor_bytes,
        target_request_count=target_request_count,
        trainer_source_count=trainer_source_count,
        segment_count=len(plans),
        source_count_per_target_tensor={
            key: len(value) for key, value in sorted(target_sources.items())
        },
        source_balance_bytes=dict(sorted(source_balance.items())),
        target_balance_bytes=dict(sorted(target_balance.items())),
    )


def segment_plans_to_json(plans: Sequence[SegmentPlan]) -> str:
    """Serialize segment plans as stable JSON."""

    return json.dumps(
        [plan.to_dict() for plan in plans],
        indent=2,
        sort_keys=True,
    )


def segment_plans_from_json(payload: str) -> list[SegmentPlan]:
    """Deserialize segment plans from JSON."""

    return [SegmentPlan.from_dict(item) for item in json.loads(payload)]


def write_json_artifact(
    data: (
        SegmentPlan
        | SimulationResult
        | CompetitiveSimulationResult
        | Sequence[SegmentPlan]
    ),
    path: str | Path,
) -> None:
    """Write a stable JSON artifact for planner or simulator output."""

    if isinstance(data, (SimulationResult, CompetitiveSimulationResult)):
        payload: Any = data.to_dict()
    elif isinstance(data, SegmentPlan):
        payload = data.to_dict()
    else:
        payload = [item.to_dict() for item in data]

    Path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _unique_requested_bytes(requests: Sequence[SliceRequest]) -> int:
    unique_request_keys = {
        (
            req.model_name,
            req.model_version,
            req.tensor_name,
            req.requested_range,
            req.dtype,
            req.element_size_bytes,
        )
        for req in requests
    }
    total = 0
    for _, _, _, requested_range, dtype, element_size_bytes in unique_request_keys:
        total += range_volume(requested_range) * dtype_itemsize(
            dtype,
            element_size_bytes,
        )
    return total


def _unique_full_tensor_bytes(
    ownerships: Sequence[SliceOwnership],
    requests: Sequence[SliceRequest],
) -> int:
    tensor_keys: dict[tuple[str, str, str, str, int | None], tuple[int, ...]] = {}
    for owner in ownerships:
        if not any(
            request.tensor_name == owner.tensor_name
            and (not request.model_name or request.model_name == owner.model_name)
            and (
                not request.model_version
                or request.model_version == owner.model_version
            )
            for request in requests
        ):
            continue
        key = (
            owner.model_name,
            owner.model_version,
            owner.tensor_name,
            owner.dtype,
            owner.element_size_bytes,
        )
        existing_shape = tensor_keys.get(key)
        if existing_shape is not None and existing_shape != owner.global_shape:
            raise IncompatibleManifestError(
                f"tensor {owner.tensor_name!r} has inconsistent global_shape "
                f"{existing_shape} vs {owner.global_shape}"
            )
        tensor_keys[key] = owner.global_shape

    total = 0
    for (_, _, _, dtype, element_size_bytes), global_shape in tensor_keys.items():
        full_range = tuple((0, dim) for dim in global_shape)
        total += range_volume(full_range) * dtype_itemsize(dtype, element_size_bytes)
    return total


def _balance_metrics(
    plans: Sequence[SegmentPlan],
) -> tuple[dict[str, int], dict[str, int], dict[str, set[str]]]:
    source_balance: dict[str, int] = {}
    target_balance: dict[str, int] = {}
    target_sources: dict[str, set[str]] = {}
    for plan in plans:
        source_balance[plan.source_id] = source_balance.get(plan.source_id, 0) + plan.bytes
        target_key = plan.target_id or plan.tensor_name
        target_balance[target_key] = target_balance.get(target_key, 0) + plan.bytes
        target_sources.setdefault(target_key, set()).add(plan.source_id)
    return source_balance, target_balance, target_sources


def _duration_ms(
    *,
    trainer_to_inference_bytes: int,
    inference_side_fanout_bytes: int,
    trainer_collective_bytes: int,
    checkpoint_storage_bytes: int,
    segment_count: int,
    assumptions: CompetitiveAssumptions,
    fixed_overhead_ms: float,
) -> float:
    transfer_seconds = 0.0
    transfer_seconds += _bytes_to_seconds(
        trainer_to_inference_bytes,
        assumptions.trainer_to_inference_gbps,
    )
    transfer_seconds += _bytes_to_seconds(
        inference_side_fanout_bytes,
        assumptions.inference_to_inference_gbps,
    )
    transfer_seconds += _bytes_to_seconds(
        trainer_collective_bytes,
        assumptions.nccl_reshard_gbps,
    )
    transfer_seconds += _bytes_to_seconds(
        checkpoint_storage_bytes,
        assumptions.checkpoint_storage_gbps,
    )
    latency_ms = (segment_count * assumptions.per_segment_latency_us) / 1000.0
    return round((transfer_seconds * 1000) + latency_ms + fixed_overhead_ms, 6)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return round(numerator / denominator, 6)


def _predict_competitive_bottleneck(
    *,
    trainer_to_inference_bytes: int,
    inference_side_fanout_bytes: int,
    trainer_collective_bytes: int,
    checkpoint_storage_bytes: int,
    assumptions: CompetitiveAssumptions,
) -> str:
    components = {
        "trainer-to-inference": _bytes_to_seconds(
            trainer_to_inference_bytes,
            assumptions.trainer_to_inference_gbps,
        ),
        "inference-side-fanout": _bytes_to_seconds(
            inference_side_fanout_bytes,
            assumptions.inference_to_inference_gbps,
        ),
        "trainer-collective": _bytes_to_seconds(
            trainer_collective_bytes,
            assumptions.nccl_reshard_gbps,
        ),
        "checkpoint-storage": _bytes_to_seconds(
            checkpoint_storage_bytes,
            assumptions.checkpoint_storage_gbps,
        ),
    }
    bottleneck, seconds = max(components.items(), key=lambda item: item[1])
    if seconds == 0:
        return "fixed-overhead"
    return bottleneck


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
        if source_value is not None and target_value is not None and source_value != target_value:
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


def _normalize_dtype(dtype: str) -> str:
    return dtype.removeprefix("torch.").lower()


def _bytes_to_seconds(byte_count: int, gbps: float) -> float:
    return (byte_count * 8) / (gbps * 1e9)


def _predict_bottleneck(
    *,
    trainer_to_inference_bytes: int,
    inference_fanout_bytes: int,
    bandwidth: BandwidthAssumptions,
) -> str:
    trainer_seconds = _bytes_to_seconds(
        trainer_to_inference_bytes,
        bandwidth.trainer_to_inference_gbps,
    )
    fanout_seconds = _bytes_to_seconds(
        inference_fanout_bytes,
        bandwidth.inference_to_inference_gbps,
    )
    if fanout_seconds > trainer_seconds:
        return "inference-side-fanout"
    if trainer_seconds > fanout_seconds:
        return "trainer-to-inference"
    return "balanced"
