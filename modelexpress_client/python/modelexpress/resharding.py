# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility facade for cross-parallelism tensor resharding.

The implementation is split across focused modules, but existing callers import
from `modelexpress.resharding`. Keep this file as the stable public surface.
"""

from __future__ import annotations

from .resharding_artifacts import (
    segment_plans_from_json,
    segment_plans_to_json,
    write_json_artifact,
)
from .resharding_planner import (
    classify_quantization_scope,
    classify_tensor_family,
    plan_segments,
)
from .resharding_ranges import (
    TensorRange,
    intersect_ranges,
    normalize_range,
    range_extents,
    range_to_json_key,
    range_to_list,
    range_volume,
    row_major_strides,
)
from .resharding_simulator import (
    simulate_competitive_refit,
    simulate_resharding,
)
from .resharding_types import (
    BandwidthAssumptions,
    CompetitiveAssumptions,
    CompetitiveSimulationResult,
    CompetitiveStrategy,
    CoverageError,
    IncompatibleManifestError,
    QuantizationMetadataError,
    QuantizationScope,
    RetryPolicy,
    SegmentPlan,
    SimulationResult,
    SliceOwnership,
    SliceRequest,
    StrategyCost,
    TransferStrategy,
    dtype_itemsize,
)

__all__ = [
    "BandwidthAssumptions",
    "CompetitiveAssumptions",
    "CompetitiveSimulationResult",
    "CompetitiveStrategy",
    "CoverageError",
    "IncompatibleManifestError",
    "QuantizationMetadataError",
    "QuantizationScope",
    "RetryPolicy",
    "SegmentPlan",
    "SimulationResult",
    "SliceOwnership",
    "SliceRequest",
    "StrategyCost",
    "TensorRange",
    "TransferStrategy",
    "classify_quantization_scope",
    "classify_tensor_family",
    "dtype_itemsize",
    "intersect_ranges",
    "normalize_range",
    "plan_segments",
    "range_extents",
    "range_to_json_key",
    "range_to_list",
    "range_volume",
    "row_major_strides",
    "segment_plans_from_json",
    "segment_plans_to_json",
    "simulate_competitive_refit",
    "simulate_resharding",
    "write_json_artifact",
]
