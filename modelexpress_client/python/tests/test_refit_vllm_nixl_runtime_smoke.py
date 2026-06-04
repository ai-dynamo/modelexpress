# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from modelexpress.refit_vllm_nixl_runtime_smoke import (
    _clear_torchrun_env_for_runtime_engine,
    _range_slices,
    _replacement_tensor,
    _restore_torchrun_env,
    build_vllm_runtime_nixl_plan,
    inspect_vllm_runtime_nixl_scenario,
    materialize_vllm_nixl_source_tensor,
    run_vllm_receiver_refit_from_nixl_staging_tensor,
)


class TinyVllmOwnedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = nn.Linear(4, 6, bias=False)


class FakeVllmLLM:
    def __init__(self, worker_module: nn.Module) -> None:
        self.worker_module = worker_module
        self.apply_model_calls = 0

    def apply_model(self, func):
        self.apply_model_calls += 1
        return [func(self.worker_module)]


def test_vllm_nixl_runtime_clears_torchrun_env_for_engine(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    saved = _clear_torchrun_env_for_runtime_engine()

    assert saved == {
        "RANK": "2",
        "WORLD_SIZE": "3",
        "MASTER_ADDR": "127.0.0.1",
    }
    assert "RANK" not in os.environ
    assert "WORLD_SIZE" not in os.environ
    assert "MASTER_ADDR" not in os.environ
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"

    _restore_torchrun_env(saved)

    assert os.environ["RANK"] == "2"
    assert os.environ["WORLD_SIZE"] == "3"
    assert os.environ["MASTER_ADDR"] == "127.0.0.1"


def test_vllm_nixl_runtime_plan_splits_source_rank_owned_rows():
    target = torch.empty((7, 4), dtype=torch.float32)

    request, owners, plans = build_vllm_runtime_nixl_plan(
        target,
        tensor_name="lm_head.weight",
        model_name="tiny-vllm-nixl",
        model_version="step-21",
    )

    assert request.runtime_framework == "vllm"
    assert request.layout_tags["runtime_lifecycle"] == (
        "apply-model-update-from-nixl-staging"
    )
    assert [owner.source_id for owner in owners] == ["trainer-rank0", "trainer-rank1"]
    assert [owner.worker_rank for owner in owners] == [0, 1]
    assert owners[0].source_range == ((0, 3), (0, 4))
    assert owners[1].source_range == ((3, 7), (0, 4))
    assert [plan.source_id for plan in plans] == ["trainer-rank0", "trainer-rank1"]
    assert [plan.target_byte_offset for plan in plans] == [0, 48]
    assert [plan.bytes for plan in plans] == [48, 64]


def test_vllm_nixl_runtime_source_tensors_reconstruct_replacement():
    target = torch.empty((6, 4), dtype=torch.float32)
    _, owners, _ = build_vllm_runtime_nixl_plan(
        target,
        tensor_name="lm_head.weight",
        model_name="tiny-vllm-nixl",
        model_version="step-22",
    )
    expected = _replacement_tensor(target.shape, dtype=target.dtype, device="cpu")
    assembled = torch.empty_like(expected)

    for owner in owners:
        source = materialize_vllm_nixl_source_tensor(
            owner,
            dtype=target.dtype,
            device=torch.device("cpu"),
        )
        assembled[_range_slices(owner.source_range)].copy_(source)

    torch.testing.assert_close(assembled, expected)


def test_vllm_nixl_runtime_inspects_worker_tensor_with_apply_model():
    module = TinyVllmOwnedModule()
    llm = FakeVllmLLM(module)

    scenario = inspect_vllm_runtime_nixl_scenario(
        llm,
        model_name="tiny-vllm-nixl",
        model_version="step-inspect",
        preferred_tensor_name="lm_head.weight",
    )

    assert llm.apply_model_calls == 1
    assert scenario["target_tensor_name"] == "lm_head.weight"
    assert scenario["target_shape"] == [6, 4]
    assert scenario["request"]["runtime_framework"] == "vllm"
    assert [plan["source_id"] for plan in scenario["segment_plans"]] == [
        "trainer-rank0",
        "trainer-rank1",
    ]


def test_vllm_nixl_runtime_refit_installs_and_restores_worker_weight(tmp_path):
    module = TinyVllmOwnedModule()
    with torch.no_grad():
        module.lm_head.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(6, 4))
    original = module.lm_head.weight.detach().clone()
    llm = FakeVllmLLM(module)
    scenario = inspect_vllm_runtime_nixl_scenario(
        llm,
        model_name="tiny-vllm-nixl",
        model_version="step-23",
        preferred_tensor_name="lm_head.weight",
    )
    assembled = _replacement_tensor(original.shape, dtype=original.dtype, device="cpu")
    artifact_path = tmp_path / "vllm-nixl-runtime-smoke.json"

    result = run_vllm_receiver_refit_from_nixl_staging_tensor(
        llm,
        assembled=assembled,
        scenario=scenario,
        model_name="tiny-vllm-nixl",
        model_version="step-23",
        vllm_version="unit-test",
        artifact_path=artifact_path,
        model_path="unit-tiny-qwen2",
        nixl_reads=[
            {"source_id": "trainer-rank0", "bytes": 48},
            {"source_id": "trainer-rank1", "bytes": 48},
        ],
        nixl_metrics={"raw_nixl_read_duration_ms": 1.5},
        distributed={"world_size": 3, "target_rank": 2},
    )

    assert llm.apply_model_calls == 2
    assert result["result"] == "pass"
    assert result["proof"]["actual_nixl_reads_used"] is True
    assert result["proof"]["source_rank_owned_trainer_tensors_used"] is True
    assert result["proof"]["trainer_like_source_processes_used"] is True
    assert result["proof"]["real_training_loop_used"] is False
    assert result["proof"]["synthetic_trainer_payloads_used"] is False
    assert result["proof"]["synthetic_source_values_used"] is True
    assert result["proof"]["nixl_reads_land_directly_in_runtime_tensor"] is False
    assert result["proof"]["runtime_update_from_nixl_staging_tensor"] is True
    assert result["proof"]["runtime_update_payload_copied_through_apply_model"] is True
    assert result["validation"]["nixl_staging_allclose"] is True
    assert result["validation"]["checksum_matches"] is True
    assert result["validation"]["restored_original"] is True
    assert result["metrics"]["segment_count"] == 2
    assert result["metrics"]["raw_nixl_read_duration_ms"] == 1.5
    torch.testing.assert_close(module.lm_head.weight, original)

    payload = json.loads(artifact_path.read_text())
    assert payload["proof"]["actual_nixl_reads_used"] is True
    assert payload["distributed"] == {"world_size": 3, "target_rank": 2}
