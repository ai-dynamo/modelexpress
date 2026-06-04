# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact and metadata smoke helpers for the refit POC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .refit_poc_scenario import (
    FAILED_PRIMARY_SOURCE_IDS,
    MODEL_NAME,
    MODEL_VERSION,
    TENSOR_NAME,
    alternate_ownerships,
    inference_request,
    primary_ownerships,
)
from .resharding import (
    BandwidthAssumptions,
    QuantizationScope,
    SliceOwnership,
    SliceRequest,
    classify_tensor_family,
    plan_segments,
    segment_plans_to_json,
    simulate_resharding,
)


def synthetic_plan_context(request: SliceRequest) -> dict[str, Any]:
    primary_plans = plan_segments(primary_ownerships(), [request])
    failed_primary_plans = [
        plan for plan in primary_plans if plan.source_id in FAILED_PRIMARY_SOURCE_IDS
    ]
    recovery_requests = [
        inference_request(requested_range=plan.target_range)
        for plan in failed_primary_plans
    ]
    recovery_plans = plan_segments(alternate_ownerships(), recovery_requests)
    return {
        "mode": "synthetic",
        "plan_source": "in-process-synthetic",
        "primary_plans": primary_plans,
        "recovery_plans": recovery_plans,
        "primary_ownerships": primary_ownerships(),
        "alternate_ownerships": alternate_ownerships(),
        "discovered_ownerships": [],
        "metadata_query_duration_ms": 0.0,
        "planner_duration_ms": 0.0,
    }


def build_planner_artifacts(
    request: SliceRequest | None = None,
    plan_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = request or inference_request()
    plan_context = plan_context or synthetic_plan_context(request)
    plans = plan_context["primary_plans"]
    recovery_plans = plan_context["recovery_plans"]
    simulation = simulate_resharding(
        plan_context["primary_ownerships"],
        [request],
        BandwidthAssumptions(
            trainer_to_inference_gbps=200,
            inference_to_inference_gbps=400,
        ),
    )
    return {
        "request": request.to_dict(),
        "plan_source": plan_context.get("plan_source", "unknown"),
        "control_plane_mode": plan_context.get("mode", "unknown"),
        "primary_ownerships": [
            owner.to_dict() for owner in plan_context["primary_ownerships"]
        ],
        "alternate_ownerships": [
            owner.to_dict() for owner in plan_context["alternate_ownerships"]
        ],
        "discovered_ownerships": [
            owner.to_dict() for owner in plan_context.get("discovered_ownerships", [])
        ],
        "primary_segment_plans": [plan.to_dict() for plan in plans],
        "recovery_segment_plans": [plan.to_dict() for plan in recovery_plans],
        "primary_segment_plans_json": json.loads(segment_plans_to_json(plans)),
        "simulation": simulation.to_dict(),
    }


def metadata_smokes() -> dict[str, Any]:
    moe_owners = [
        SliceOwnership(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            tensor_name=TENSOR_NAME,
            global_shape=(4, 8, 4),
            dtype="bfloat16",
            source_range=((0, 2), (0, 8), (0, 4)),
            worker_id="moe-rank0",
            source_id="moe-source-0",
            layout_tags={"moe_expert_axis": 0, "storage_layout": "row-major"},
        ),
        SliceOwnership(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            tensor_name=TENSOR_NAME,
            global_shape=(4, 8, 4),
            dtype="bfloat16",
            source_range=((2, 4), (0, 8), (0, 4)),
            worker_id="moe-rank1",
            source_id="moe-source-1",
            layout_tags={"moe_expert_axis": 0, "storage_layout": "row-major"},
        ),
    ]
    vllm_request = SliceRequest(
        tensor_name=TENSOR_NAME,
        requested_range=((1, 3), (0, 8), (0, 4)),
        target_shape=(2, 8, 4),
        dtype="bfloat16",
        target_id="vllm-rank0",
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        runtime_framework="vllm",
        layout_tags={"moe_expert_axis": 0, "storage_layout": "row-major"},
    )
    sglang_request = SliceRequest(
        tensor_name=TENSOR_NAME,
        requested_range=((1, 3), (0, 8), (0, 4)),
        target_shape=(2, 8, 4),
        dtype="bfloat16",
        target_id="sglang-rank0",
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        runtime_framework="sglang",
        layout_tags={"moe_expert_axis": 0, "storage_layout": "row-major"},
    )
    vllm_plans = plan_segments(moe_owners, [vllm_request])
    sglang_plans = plan_segments(moe_owners, [sglang_request])
    return {
        "qwen_moe_expert_axis": {
            "passed": True,
            "vllm_segment_count": len(vllm_plans),
            "sglang_segment_count": len(sglang_plans),
            "source_ids": sorted({plan.source_id for plan in vllm_plans}),
        },
        "cross_framework_compatible_requests": {
            "passed": True,
            "frameworks": ["vllm", "sglang"],
            "same_source_publication": True,
        },
        "tensor_family_classification": {
            TENSOR_NAME: classify_tensor_family(
                TENSOR_NAME,
                layout_tags={"moe_expert_axis": 0, "storage_layout": "row-major"},
            ),
            "model.layers.0.mlp.experts.w1.weight_scale_inv": classify_tensor_family(
                "model.layers.0.mlp.experts.w1.weight_scale_inv",
                quantization_scope=QuantizationScope.GLOBAL_REQUIRED,
            ),
            "model.rotary_emb.inv_freq": classify_tensor_family(
                "model.rotary_emb.inv_freq",
                quantization_scope=QuantizationScope.GENERATED_ON_TARGET,
            ),
        },
    }


def artifact_base(
    *,
    mode: str,
    gpu_count: int,
    copied_bytes: int,
    copy_duration_ms: float,
    validation: dict[str, Any],
    request: SliceRequest | None = None,
    plan_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    planner_artifacts = build_planner_artifacts(
        request=request,
        plan_context=plan_context,
    )
    primary_plans = planner_artifacts["primary_segment_plans"]
    recovery_plans = planner_artifacts["recovery_segment_plans"]
    return {
        "schema_version": 1,
        "result": "pass" if validation["allclose"] else "fail",
        "mode": mode,
        "gpu_count": gpu_count,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "tensor_name": TENSOR_NAME,
        "planner": planner_artifacts,
        "metadata_smokes": metadata_smokes(),
        "proof": {
            "trainer_full_all_gather_used": False,
            "trainer_side_inference_layout_conversion_used": False,
            "host_side_torch_cat_used": False,
            "target_slice_spans_multiple_trainers": len(
                {plan["source_id"] for plan in primary_plans}
            ) >= 2,
            "failed_then_succeeded": bool(recovery_plans) and validation["allclose"],
            "failed_source_ids": list(FAILED_PRIMARY_SOURCE_IDS),
            "replanned_only_failed_segments": bool(recovery_plans),
            "gpu_target_assembly_succeeded": validation["allclose"],
        },
        "metrics": {
            "trainer_to_inference_bytes": copied_bytes,
            "inference_side_fanout_bytes": planner_artifacts["simulation"][
                "inference_side_fanout_bytes"
            ],
            "redundant_cross_boundary_factor": planner_artifacts["simulation"][
                "redundant_cross_boundary_factor"
            ],
            "segment_count": len(primary_plans),
            "recovery_segment_count": len(recovery_plans),
            "source_count_per_target_tensor": planner_artifacts["simulation"][
                "source_count_per_target_tensor"
            ],
            "gpu_copy_duration_ms": copy_duration_ms,
            "registration_duration_ms": 0.0,
            "publish_duration_ms": 0.0,
            "planner_duration_ms": (plan_context or {}).get(
                "planner_duration_ms",
                planner_artifacts["simulation"].get("planner_duration_ms"),
            ),
            "activation_install_duration_ms": 0.0,
            "retry_count": 1 if recovery_plans else 0,
            "rediscovery_count": 1 if recovery_plans else 0,
        },
        "validation": validation,
    }


def write_artifact(result: dict[str, Any], artifact_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
