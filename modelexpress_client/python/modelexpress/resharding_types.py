# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed metadata models for cross-parallelism tensor resharding."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from .resharding_ranges import (
    TensorRange,
    _normalize_shape,
    _validate_range_inside_shape,
    normalize_range,
    range_extents,
    range_to_json_key,
    range_to_list,
)


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


def _normalize_dtype(dtype: str) -> str:
    return dtype.removeprefix("torch.").lower()


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
