# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cross-parallelism resharding planner."""

import json

import pytest

from modelexpress.resharding import (
    BandwidthAssumptions,
    CoverageError,
    IncompatibleManifestError,
    QuantizationMetadataError,
    QuantizationScope,
    SimulationResult,
    SliceOwnership,
    SliceRequest,
    TransferStrategy,
    classify_quantization_scope,
    classify_tensor_family,
    intersect_ranges,
    plan_segments,
    range_volume,
    segment_plans_from_json,
    segment_plans_to_json,
    simulate_resharding,
    write_json_artifact,
)
from modelexpress.refit_poc import metadata_smokes, run_planner_only


def _owner(
    *,
    source_range,
    worker_id="worker-0",
    source_id="source-0",
    dtype="bfloat16",
    global_shape=(8, 4),
    storage_offset_bytes=0,
    layout_tags=None,
    quantization_scope=QuantizationScope.ABSENT,
):
    return SliceOwnership(
        model_name="qwen",
        model_version="v1",
        tensor_name="layers.0.mlp.weight",
        global_shape=global_shape,
        dtype=dtype,
        source_range=source_range,
        source_id=source_id,
        worker_id=worker_id,
        worker_rank=0,
        source_lease=f"lease-{worker_id}",
        nixl_descriptor_id=f"desc-{worker_id}",
        storage_offset_bytes=storage_offset_bytes,
        layout_tags=layout_tags or {"storage_layout": "row-major", "tp": 2},
        quantization_scope=quantization_scope,
    )


def _request(
    *,
    requested_range,
    target_id="target-0",
    dtype="bfloat16",
    target_offset_bytes=0,
    runtime_framework="vllm",
    layout_tags=None,
    quantization_scope=QuantizationScope.ABSENT,
):
    return SliceRequest(
        tensor_name="layers.0.mlp.weight",
        requested_range=requested_range,
        target_shape=tuple(end - start for start, end in requested_range),
        dtype=dtype,
        target_id=target_id,
        model_name="qwen",
        model_version="v1",
        target_offset_bytes=target_offset_bytes,
        runtime_framework=runtime_framework,
        layout_tags=layout_tags or {"storage_layout": "row-major", "tp": 4},
        quantization_scope=quantization_scope,
    )


def _covered_points(plans):
    points = set()
    for plan in plans:
        for row in range(plan.target_range[0][0], plan.target_range[0][1]):
            for col in range(plan.target_range[1][0], plan.target_range[1][1]):
                assert (row, col) not in points
                points.add((row, col))
    return points


def test_intersect_ranges_and_volume():
    intersection = intersect_ranges(((0, 4), (2, 8)), ((2, 6), (0, 4)))
    assert intersection == ((2, 4), (2, 4))
    assert range_volume(intersection) == 4
    assert intersect_ranges(((0, 1),), ((1, 2),)) is None


def test_multi_source_row_slice_spanning_two_trainers():
    owners = [
        _owner(source_range=((0, 3), (0, 4)), worker_id="w0", source_id="s0"),
        _owner(
            source_range=((3, 8), (0, 4)),
            worker_id="w1",
            source_id="s1",
            storage_offset_bytes=100,
        ),
    ]
    request = _request(requested_range=((2, 6), (0, 4)), target_offset_bytes=16)

    plans = plan_segments(owners, [request])

    assert len(plans) == 2
    assert [plan.source_id for plan in plans] == ["s0", "s1"]
    assert plans[0].target_range == ((2, 3), (0, 4))
    assert plans[0].source_byte_offset == 16
    assert plans[0].target_byte_offset == 16
    assert plans[0].bytes == 8
    assert plans[1].target_range == ((3, 6), (0, 4))
    assert plans[1].source_byte_offset == 100
    assert plans[1].target_byte_offset == 24
    assert plans[1].bytes == 24


def test_non_aligned_column_shards_cover_target_slice():
    owners = [
        _owner(
            source_range=((0, 4), (0, 3)),
            worker_id="w0",
            source_id="s0",
            global_shape=(4, 8),
        ),
        _owner(
            source_range=((0, 4), (3, 8)),
            worker_id="w1",
            source_id="s1",
            global_shape=(4, 8),
        ),
    ]
    request = _request(requested_range=((1, 3), (2, 6)))

    plans = plan_segments(owners, [request])

    assert len(plans) == 4
    assert sum(plan.bytes for plan in plans) == 16
    assert {plan.source_id for plan in plans} == {"s0", "s1"}
    assert _covered_points(plans) == {
        (row, col)
        for row in range(1, 3)
        for col in range(2, 6)
    }


def test_missing_coverage_is_rejected():
    owners = [_owner(source_range=((0, 2), (0, 4)))]
    request = _request(requested_range=((0, 4), (0, 4)))

    with pytest.raises(CoverageError) as exc_info:
        plan_segments(owners, [request])

    assert exc_info.value.missing_ranges == [((2, 4), (0, 4))]
    assert exc_info.value.duplicate_ranges == []


def test_duplicate_coverage_is_rejected():
    owners = [
        _owner(source_range=((0, 3), (0, 4)), worker_id="w0", source_id="s0"),
        _owner(source_range=((2, 4), (0, 4)), worker_id="w1", source_id="s1"),
    ]
    request = _request(requested_range=((0, 4), (0, 4)))

    with pytest.raises(CoverageError) as exc_info:
        plan_segments(owners, [request])

    assert exc_info.value.duplicate_ranges == [((2, 3), (0, 4))]


def test_dtype_mismatch_is_rejected():
    owners = [_owner(source_range=((0, 4), (0, 4)), dtype="bfloat16")]
    request = _request(requested_range=((0, 4), (0, 4)), dtype="float16")

    with pytest.raises(IncompatibleManifestError, match="dtype mismatch"):
        plan_segments(owners, [request])


def test_layout_mismatch_is_rejected_for_storage_sensitive_tags():
    owners = [
        _owner(
            source_range=((0, 4), (0, 4)),
            layout_tags={"storage_layout": "row-major", "tp": 2},
        )
    ]
    request = _request(
        requested_range=((0, 4), (0, 4)),
        layout_tags={"storage_layout": "blocked", "tp": 4},
    )

    with pytest.raises(IncompatibleManifestError, match="storage_layout"):
        plan_segments(owners, [request])


def test_quantization_scope_flags_global_metadata_fallback():
    owner = _owner(
        source_range=((0, 4), (0, 4)),
        quantization_scope=QuantizationScope.LOCAL,
    )
    request = _request(requested_range=((0, 4), (0, 4)))
    assert classify_quantization_scope(owner, request) == "local"

    global_owner = _owner(
        source_range=((0, 4), (0, 4)),
        quantization_scope=QuantizationScope.GLOBAL_REQUIRED,
    )
    with pytest.raises(QuantizationMetadataError):
        plan_segments([global_owner], [request])


def test_tensor_family_classification_marks_moe_and_quantization_fallback():
    assert classify_tensor_family(
        "model.layers.0.mlp.experts.w1.weight",
        layout_tags={"moe_expert_axis": 0, "storage_layout": "row-major"},
    ) == "moe-expert-axis-shard"
    assert classify_tensor_family(
        "model.layers.0.mlp.experts.w1.weight_scale_inv",
        quantization_scope=QuantizationScope.GLOBAL_REQUIRED,
    ) == "quantization-global-required-fallback"
    assert classify_tensor_family(
        "model.rotary_emb.inv_freq",
        quantization_scope=QuantizationScope.GENERATED_ON_TARGET,
    ) == "generated-on-target"


def test_refit_poc_metadata_smokes_cover_moe_and_cross_framework():
    result = metadata_smokes()
    assert result["qwen_moe_expert_axis"]["passed"] is True
    assert result["qwen_moe_expert_axis"]["source_ids"] == [
        "moe-source-0",
        "moe-source-1",
    ]
    assert result["cross_framework_compatible_requests"]["frameworks"] == [
        "vllm",
        "sglang",
    ]
    assert result["tensor_family_classification"][
        "model.layers.0.mlp.experts.w1.weight"
    ] == "moe-expert-axis-shard"


def test_refit_poc_planner_only_writes_artifact(tmp_path):
    artifact = tmp_path / "planner-only.json"
    result = run_planner_only(artifact)
    assert artifact.exists()
    assert result["proof"]["target_slice_spans_multiple_trainers"] is True
    assert result["proof"]["failed_then_succeeded"] is True
    assert result["planner"]["primary_segment_plans"][0]["source_id"] == "trainer-rank0"
    assert result["planner"]["primary_segment_plans"][1]["source_id"] == "trainer-rank1"


def test_segment_plan_serialization_round_trip():
    owners = [
        _owner(source_range=((0, 3), (0, 4)), worker_id="w0", source_id="s0"),
        _owner(source_range=((3, 8), (0, 4)), worker_id="w1", source_id="s1"),
    ]
    request = _request(requested_range=((2, 6), (0, 4)))
    plans = plan_segments(owners, [request])

    payload = segment_plans_to_json(plans)
    assert segment_plans_from_json(payload) == plans


def test_simulator_prefers_replica_fanout_when_inference_fabric_is_fast(tmp_path):
    owners = [_owner(source_range=((0, 4), (0, 4)), global_shape=(4, 4))]
    requests = [
        _request(requested_range=((0, 4), (0, 4)), target_id="vllm-0"),
        _request(requested_range=((0, 4), (0, 4)), target_id="vllm-1"),
    ]

    result = simulate_resharding(
        owners,
        requests,
        BandwidthAssumptions(
            trainer_to_inference_gbps=10,
            inference_to_inference_gbps=100,
        ),
    )

    assert result.preferred_strategy == TransferStrategy.PRIMARY_REPLICA_FANOUT
    assert result.trainer_to_inference_bytes == 32
    assert result.inference_side_fanout_bytes == 32
    assert result.redundant_cross_boundary_factor == 2.0
    assert result.source_count_per_target_tensor == {"vllm-0": 1, "vllm-1": 1}

    artifact = tmp_path / "resharding-simulation.json"
    write_json_artifact(result, artifact)
    parsed = json.loads(artifact.read_text())
    assert SimulationResult.from_dict(parsed) == result
    assert parsed["preferred_strategy"] == "primary-replica-fanout"
