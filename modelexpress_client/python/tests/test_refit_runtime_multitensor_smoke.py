# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
import torch

from modelexpress.refit_runtime_multitensor_smoke import (
    build_runtime_multitensor_plan,
    run_runtime_multitensor_nixl_staging_smoke,
    run_runtime_multitensor_refit_smoke,
)


def _runtime_tensors() -> dict[str, torch.Tensor]:
    return {
        "model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(
            8, 4
        ),
        "lm_head.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
    }


@pytest.mark.parametrize("runtime_framework", ["vllm", "sglang"])
def test_runtime_multitensor_plan_splits_each_tensor_across_source_ranks(
    runtime_framework: str,
) -> None:
    tensors = _runtime_tensors()

    requests, owners, plans = build_runtime_multitensor_plan(
        tensors,
        runtime_framework=runtime_framework,
        model_name="tiny-runtime-multitensor",
        model_version="step-42",
    )

    assert [request.tensor_name for request in requests] == [
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
    assert {request.runtime_framework for request in requests} == {runtime_framework}
    assert len(owners) == 4
    assert len(plans) == 4
    assert {
        (owner.tensor_name, owner.source_id, owner.source_range) for owner in owners
    } == {
        ("model.embed_tokens.weight", "trainer-rank0", ((0, 4), (0, 4))),
        ("model.embed_tokens.weight", "trainer-rank1", ((4, 8), (0, 4))),
        ("lm_head.weight", "trainer-rank0", ((0, 3), (0, 4))),
        ("lm_head.weight", "trainer-rank1", ((3, 6), (0, 4))),
    }
    assert {plan.tensor_name: plan.target_id for plan in plans} == {
        "model.embed_tokens.weight": f"{runtime_framework}:model.embed_tokens.weight",
        "lm_head.weight": f"{runtime_framework}:lm_head.weight",
    }
    assert sum(plan.bytes for plan in plans) == 224


@pytest.mark.parametrize("runtime_framework", ["vllm", "sglang"])
def test_runtime_multitensor_refit_installs_and_rolls_back_bundle(
    tmp_path,
    runtime_framework: str,
) -> None:
    tensors = _runtime_tensors()
    originals = {name: tensor.detach().clone() for name, tensor in tensors.items()}
    artifact_path = tmp_path / f"{runtime_framework}-runtime-multitensor.json"

    result = run_runtime_multitensor_refit_smoke(
        tensors,
        runtime_framework=runtime_framework,
        model_name="tiny-runtime-multitensor",
        model_version="step-43",
        previous_model_version="step-42",
        artifact_path=artifact_path,
    )

    assert result["result"] == "pass"
    assert result["runtime_framework"] == runtime_framework
    assert result["target_tensor_count"] == 2
    assert result["proof"]["target_tensor_count_gt1"] is True
    assert result["proof"]["multi_tensor_refit_transaction_used"] is True
    assert result["proof"]["receiver_requests_from_runtime_owned_tensors"] is True
    assert result["proof"]["receiver_installed_into_runtime_owned_tensors"] is True
    assert result["proof"]["trainer_optimizer_step_publisher_used"] is True
    assert result["proof"]["actual_nixl_reads_used"] is False
    assert result["proof"]["gpu_nixl_reads_used"] is False
    assert result["proof"]["live_runtime_engine_used"] is False
    assert result["proof"]["trainer_full_all_gather_used"] is False
    assert result["proof"]["trainer_side_inference_layout_conversion_used"] is False
    assert result["validation"]["allclose"] is True
    assert result["validation"]["checksum_matches"] is True
    assert result["validation"]["restored_original"] is True
    assert result["metrics"]["segment_count"] == 4
    assert result["metrics"]["target_tensor_count"] == 2
    assert result["metrics"]["trainer_to_inference_bytes"] == 224
    assert result["metrics"]["target_tensor_bytes"] == 224
    assert result["metrics"]["redundant_cross_boundary_factor"] == 1.0
    assert result["metrics"]["source_count_per_target_tensor"] == {
        "model.embed_tokens.weight": 2,
        "lm_head.weight": 2,
    }
    assert result["transaction"]["rolled_back"] is True

    for name, original in originals.items():
        torch.testing.assert_close(tensors[name], original)

    payload = json.loads(artifact_path.read_text())
    assert payload["result"] == "pass"
    assert payload["target_tensor_names"] == [
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
    assert len(payload["requests"]) == 2
    assert len(payload["source_ownerships"]) == 4
    assert len(payload["segment_plans"]) == 4


@pytest.mark.parametrize("runtime_framework", ["vllm", "sglang"])
def test_runtime_multitensor_nixl_staging_installs_and_rolls_back_bundle(
    tmp_path,
    runtime_framework: str,
) -> None:
    tensors = _runtime_tensors()
    originals = {name: tensor.detach().clone() for name, tensor in tensors.items()}
    artifact_path = tmp_path / f"{runtime_framework}-runtime-multitensor-staging.json"

    result = run_runtime_multitensor_nixl_staging_smoke(
        tensors,
        runtime_framework=runtime_framework,
        model_name="tiny-runtime-multitensor",
        model_version="step-44",
        previous_model_version="step-43",
        artifact_path=artifact_path,
    )

    assert result["result"] == "pass"
    assert result["runtime_framework"] == runtime_framework
    assert result["target_tensor_count"] == 2
    assert result["proof"]["multi_tensor_refit_transaction_used"] is True
    assert result["proof"]["multi_tensor_nixl_staging_contract_used"] is True
    assert result["proof"]["nixl_read_descriptor_groups_planned"] is True
    assert result["proof"]["nixl_reads_land_into_staging_tensors"] is True
    assert result["proof"]["runtime_update_from_nixl_staging_tensors"] is True
    assert result["proof"]["actual_nixl_reads_used"] is False
    assert result["proof"]["gpu_nixl_reads_used"] is False
    assert result["proof"]["nixl_reads_land_directly_in_runtime_tensor"] is False
    assert result["validation"]["nixl_staging_allclose"] is True
    assert result["validation"]["nixl_staging_checksum_matches"] is True
    assert result["validation"]["allclose"] is True
    assert result["validation"]["checksum_matches"] is True
    assert result["validation"]["restored_original"] is True
    assert result["metrics"]["segment_count"] == 4
    assert result["metrics"]["staging_tensor_count"] == 2
    assert result["metrics"]["trainer_to_inference_bytes"] == 224
    assert result["metrics"]["target_tensor_bytes"] == 224
    assert result["metrics"]["source_count_per_target_tensor"] == {
        "model.embed_tokens.weight": 2,
        "lm_head.weight": 2,
    }
    assert [group["source_id"] for group in result["nixl"]["planned_read_groups"]] == [
        "trainer-rank0",
        "trainer-rank1",
    ]
    assert result["nixl"]["planned_read_groups"][0]["tensor_names"] == [
        "lm_head.weight",
        "model.embed_tokens.weight",
    ]
    assert result["nixl"]["actual_read_groups"] == []

    for name, original in originals.items():
        torch.testing.assert_close(tensors[name], original)

    payload = json.loads(artifact_path.read_text())
    assert payload["proof"]["multi_tensor_nixl_staging_contract_used"] is True
    assert payload["nixl"]["actual_nixl_reads_used"] is False
    assert set(payload["nixl_staging_targets"]) == {
        "model.embed_tokens.weight",
        "lm_head.weight",
    }


def test_runtime_multitensor_requires_more_than_one_tensor() -> None:
    with pytest.raises(ValueError, match="at least two tensors"):
        run_runtime_multitensor_refit_smoke(
            {"lm_head.weight": torch.empty((6, 4), dtype=torch.float32)},
            runtime_framework="vllm",
        )


def test_runtime_multitensor_rejects_unknown_runtime_framework() -> None:
    with pytest.raises(ValueError, match="runtime_framework"):
        run_runtime_multitensor_refit_smoke(
            _runtime_tensors(),
            runtime_framework="trtllm",
        )
