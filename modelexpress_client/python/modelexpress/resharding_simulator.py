# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cost simulators for MX refit strategy selection."""

from __future__ import annotations

import time
from typing import Sequence

from .resharding_planner import plan_segments
from .resharding_ranges import TensorRange, range_volume
from .resharding_types import (
    BandwidthAssumptions,
    CompetitiveAssumptions,
    CompetitiveSimulationResult,
    CompetitiveStrategy,
    IncompatibleManifestError,
    SegmentPlan,
    SimulationResult,
    SliceOwnership,
    SliceRequest,
    StrategyCost,
    TransferStrategy,
    dtype_itemsize,
)


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
    unique_cross_bytes = _unique_requested_bytes(requests)
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

    source_balance, target_balance, target_sources = _balance_metrics(plans)

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
