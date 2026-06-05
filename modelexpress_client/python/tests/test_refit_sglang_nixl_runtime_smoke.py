# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
import json
import os

import torch

from modelexpress.refit_sglang_nixl_runtime_smoke import (
    _clear_torchrun_env_for_runtime_engine,
    _range_slices,
    _replacement_tensor,
    _restore_torchrun_env,
    build_sglang_runtime_nixl_plan,
    materialize_sglang_nixl_source_tensor,
    run_sglang_receiver_refit_from_nixl_staging_tensor,
)


class FakeSglangEngine:
    def __init__(self, weight: torch.Tensor) -> None:
        self.weights = {"lm_head.weight": weight.detach().clone()}
        self.get_weight_calls = 0
        self.update_weight_calls = 0

    def get_weights_by_name(self, name: str, truncate_size: int = 100):
        self.get_weight_calls += 1
        weight = self.weights.get(name)
        if weight is None:
            return None
        return weight.tolist()[:truncate_size]

    def update_weights_from_tensor(
        self,
        named_tensors,
        load_format=None,
        flush_cache=True,
    ):
        assert load_format == "direct"
        assert flush_cache is True
        self.update_weight_calls += 1
        for name, tensor in named_tensors:
            self.weights[name] = tensor.detach().clone()
        return True, "Success"


class Bfloat16StorageFakeSglangEngine(FakeSglangEngine):
    def update_weights_from_tensor(
        self,
        named_tensors,
        load_format=None,
        flush_cache=True,
    ):
        assert load_format == "direct"
        assert flush_cache is True
        self.update_weight_calls += 1
        for name, tensor in named_tensors:
            self.weights[name] = tensor.detach().to(torch.bfloat16).to(torch.float32)
        return True, "Success"


def test_sglang_nixl_runtime_clears_torchrun_env_for_engine(monkeypatch):
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


def test_sglang_nixl_runtime_plan_splits_source_rank_owned_rows():
    original = torch.empty((7, 4), dtype=torch.float32)

    request, owners, plans = build_sglang_runtime_nixl_plan(
        original,
        tensor_name="lm_head.weight",
        model_name="tiny-sglang-nixl",
        model_version="step-21",
    )

    assert request.runtime_framework == "sglang"
    assert request.layout_tags["runtime_lifecycle"] == (
        "engine-update-weights-from-nixl-staging"
    )
    assert [owner.source_id for owner in owners] == ["trainer-rank0", "trainer-rank1"]
    assert [owner.worker_rank for owner in owners] == [0, 1]
    assert owners[0].source_range == ((0, 3), (0, 4))
    assert owners[1].source_range == ((3, 7), (0, 4))
    assert owners[0].layout_tags["optimizer_step_publisher"] is True
    assert owners[1].layout_tags["synthetic_training_objective"] is True
    assert [plan.source_id for plan in plans] == ["trainer-rank0", "trainer-rank1"]
    assert [plan.target_byte_offset for plan in plans] == [0, 48]
    assert [plan.bytes for plan in plans] == [48, 64]


def test_sglang_nixl_runtime_source_tensors_reconstruct_replacement():
    original = torch.empty((6, 4), dtype=torch.float32)
    _, owners, _ = build_sglang_runtime_nixl_plan(
        original,
        tensor_name="lm_head.weight",
        model_name="tiny-sglang-nixl",
        model_version="step-22",
    )
    expected = _replacement_tensor(original.shape, dtype=original.dtype, device="cpu")
    assembled = torch.empty_like(expected)

    for owner in owners:
        source = materialize_sglang_nixl_source_tensor(
            owner,
            dtype=original.dtype,
            device=torch.device("cpu"),
        )
        assembled[_range_slices(owner.source_range)].copy_(source)

    torch.testing.assert_close(assembled, expected)


def test_sglang_nixl_runtime_refit_installs_and_restores_engine_weight(tmp_path):
    original = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    engine = FakeSglangEngine(original)
    request, owners, plans = build_sglang_runtime_nixl_plan(
        original,
        tensor_name="lm_head.weight",
        model_name="tiny-sglang-nixl",
        model_version="step-23",
    )
    assembled = _replacement_tensor(original.shape, dtype=original.dtype, device="cpu")
    artifact_path = tmp_path / "sglang-nixl-runtime-smoke.json"

    result = run_sglang_receiver_refit_from_nixl_staging_tensor(
        engine,
        tensor_name="lm_head.weight",
        original=original,
        assembled=assembled,
        request=request,
        source_ownerships=owners,
        segment_plans=plans,
        model_name="tiny-sglang-nixl",
        model_version="step-23",
        sglang_version="unit-test",
        artifact_path=artifact_path,
        model_path="unit-tiny-llama",
        nixl_reads=[
            {"source_id": "trainer-rank0", "bytes": 48},
            {"source_id": "trainer-rank1", "bytes": 48},
        ],
        nixl_metrics={"raw_nixl_read_duration_ms": 1.25},
        distributed={"world_size": 3, "target_rank": 2},
        actual_nixl_reads_used=True,
    )

    assert result["result"] == "pass"
    assert result["proof"]["actual_nixl_reads_used"] is True
    assert result["proof"]["source_rank_owned_trainer_tensors_used"] is True
    assert result["proof"]["trainer_like_source_processes_used"] is True
    assert result["proof"]["trainer_optimizer_step_publisher_used"] is True
    assert result["proof"]["trainer_owned_parameter_tensor_used"] is True
    assert result["proof"]["real_training_loop_used"] is False
    assert result["proof"]["real_rl_training_loop_used"] is False
    assert result["proof"]["synthetic_training_objective_used"] is True
    assert result["proof"]["synthetic_trainer_payloads_used"] is False
    assert result["proof"]["synthetic_source_values_used"] is False
    assert result["proof"]["static_replacement_formula_source_values_used"] is False
    assert result["trainer_source_update"]["optimizer_step_publisher_used"] is True
    assert result["proof"]["nixl_reads_land_directly_in_runtime_tensor"] is False
    assert result["proof"]["runtime_update_from_nixl_staging_tensor"] is True
    assert result["validation"]["nixl_staging_allclose"] is True
    assert result["validation"]["checksum_matches"] is True
    assert result["validation"]["restored_original"] is True
    assert result["metrics"]["segment_count"] == 2
    assert result["metrics"]["raw_nixl_read_duration_ms"] == 1.25
    assert engine.update_weight_calls == 2
    torch.testing.assert_close(engine.weights["lm_head.weight"], original)

    payload = json.loads(artifact_path.read_text())
    assert payload["proof"]["actual_nixl_reads_used"] is True
    assert payload["distributed"] == {"world_size": 3, "target_rank": 2}


def test_sglang_nixl_runtime_refit_uses_source_metadata_step_count(tmp_path):
    original = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    engine = FakeSglangEngine(original)
    request, owners, plans = build_sglang_runtime_nixl_plan(
        original,
        tensor_name="lm_head.weight",
        model_name="tiny-sglang-nixl",
        model_version="step-25",
    )
    annotated_owners = [
        replace(
            owner,
            layout_tags={
                **owner.layout_tags,
                "trainer_loop_step_index": 2,
                "learning_rate": "0.125",
            },
        )
        for owner in owners
    ]
    assembled = _replacement_tensor(
        original.shape,
        dtype=original.dtype,
        device="cpu",
        step_count=2,
    )

    result = run_sglang_receiver_refit_from_nixl_staging_tensor(
        engine,
        tensor_name="lm_head.weight",
        original=original,
        assembled=assembled,
        request=request,
        source_ownerships=annotated_owners,
        segment_plans=plans,
        model_name="tiny-sglang-nixl",
        model_version="step-25",
        sglang_version="unit-test",
        artifact_path=tmp_path / "sglang-nixl-runtime-step2-smoke.json",
        model_path="unit-tiny-llama",
    )

    assert result["result"] == "pass"
    assert result["proof"]["receiver_expected_update_from_source_metadata"] is True
    assert result["validation"]["expected_optimizer_step_count"] == 2
    assert result["validation"]["expected_learning_rate"] == 0.125
    assert result["trainer_source_update"]["optimizer_step_count"] == 2
    torch.testing.assert_close(engine.weights["lm_head.weight"], original)


def test_sglang_nixl_runtime_refit_accepts_runtime_bfloat16_roundtrip(tmp_path):
    original = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    engine = Bfloat16StorageFakeSglangEngine(original)
    request, owners, plans = build_sglang_runtime_nixl_plan(
        original,
        tensor_name="lm_head.weight",
        model_name="tiny-sglang-nixl",
        model_version="step-24",
    )
    assembled = _replacement_tensor(original.shape, dtype=original.dtype, device="cpu")

    result = run_sglang_receiver_refit_from_nixl_staging_tensor(
        engine,
        tensor_name="lm_head.weight",
        original=original,
        assembled=assembled,
        request=request,
        source_ownerships=owners,
        segment_plans=plans,
        model_name="tiny-sglang-nixl",
        model_version="step-24",
        sglang_version="unit-test",
        artifact_path=tmp_path / "sglang-bf16-runtime-smoke.json",
        model_path="unit-tiny-llama",
        runtime_storage_dtype=torch.bfloat16,
    )

    assert result["result"] == "pass"
    assert result["validation"]["runtime_storage_dtype"] == "bfloat16"
    assert result["validation"]["checksum_matches"] is True
    assert result["validation"]["full_precision_max_abs_error"] > 0
    assert result["proof"]["runtime_update_from_nixl_staging_tensor"] is True
    assert engine.update_weight_calls == 2
