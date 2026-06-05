# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import torch
import torch.nn as nn

from modelexpress.refit_sglang_receiver_smoke import (
    run_sglang_receiver_refit_on_engine,
    run_sglang_receiver_multitensor_refit_on_engine,
    run_sglang_receiver_refit_on_module,
)


class TinySglangOwnedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)


class FakeSglangEngine:
    def __init__(self, weight: torch.Tensor | dict[str, torch.Tensor]) -> None:
        if isinstance(weight, dict):
            self.weights = {
                name: tensor.detach().clone() for name, tensor in weight.items()
            }
        else:
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


def test_sglang_receiver_smoke_installs_and_restores_runtime_owned_tensor(tmp_path):
    module = TinySglangOwnedModule()
    with torch.no_grad():
        module.model.embed_tokens.weight.copy_(
            torch.arange(32, dtype=torch.float32).reshape(8, 4)
        )
    original = module.model.embed_tokens.weight.detach().clone()
    artifact_path = tmp_path / "sglang-receiver-smoke.json"

    result = run_sglang_receiver_refit_on_module(
        module,
        model_name="tiny-sglang",
        model_version="step-17",
        module_path="unit.sglang.model",
        sglang_version="unit-test",
        sglang_imported=False,
        artifact_path=artifact_path,
    )

    assert result["result"] == "pass"
    assert result["runtime_framework"] == "sglang"
    assert result["framework_version"] == "unit-test"
    assert result["sglang_version"] == "unit-test"
    assert result["target_tensor_name"] == "model.embed_tokens.weight"
    assert result["target_key"] == "sglang:model.embed_tokens.weight"
    assert result["request"]["runtime_framework"] == "sglang"
    assert result["request"]["layout_tags"]["runtime_module_path"] == (
        "unit.sglang.model"
    )
    assert result["request"]["layout_tags"]["sglang_tensor_name"] == (
        "model.embed_tokens.weight"
    )
    assert result["proof"]["sglang_owned_target_tensor"] is True
    assert result["proof"]["runtime_owned_target_tensor"] is True
    assert result["proof"]["receiver_request_from_runtime_owned_tensor"] is True
    assert result["proof"]["target_slice_spans_multiple_trainers"] is True
    assert result["proof"]["receiver_installed_into_sglang_owned_tensor"] is True
    assert result["proof"]["receiver_installed_into_runtime_owned_tensor"] is True
    assert result["proof"]["runtime_imported"] is False
    assert result["proof"]["real_runtime_engine_used"] is False
    assert result["proof"]["actual_nixl_reads_used"] is False
    assert result["proof"]["synthetic_trainer_payloads_used"] is True
    assert [plan["source_id"] for plan in result["segment_plans"]] == [
        "trainer-rank0",
        "trainer-rank1",
    ]
    assert [owner["source_lease"] for owner in result["source_ownerships"]] == [
        "trainer-rank0-live-sglang-receiver-smoke",
        "trainer-rank1-live-sglang-receiver-smoke",
    ]
    torch.testing.assert_close(module.model.embed_tokens.weight, original)

    payload = json.loads(artifact_path.read_text())
    assert payload["validation"]["checksum_matches"] is True
    assert payload["metrics"]["trainer_to_inference_bytes"] == 128
    assert payload["metrics"]["segment_count"] == 2


def test_sglang_receiver_smoke_uses_engine_weight_update_path(tmp_path):
    original = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    engine = FakeSglangEngine(original)
    artifact_path = tmp_path / "sglang-engine-receiver-smoke.json"

    result = run_sglang_receiver_refit_on_engine(
        engine,
        model_name="tiny-sglang-engine",
        model_version="step-engine",
        sglang_version="unit-test",
        preferred_tensor_name="lm_head.weight",
        artifact_path=artifact_path,
        model_path="unit-tiny-llama",
    )

    assert result["result"] == "pass"
    assert result["runtime_framework"] == "sglang"
    assert result["target_tensor_name"] == "lm_head.weight"
    assert result["module_path"] == "sglang.Engine"
    assert result["proof"]["sglang_engine_started"] is True
    assert result["proof"]["sglang_engine_get_weights_by_name_used"] is True
    assert result["proof"]["sglang_engine_update_weights_from_tensor_used"] is True
    assert result["proof"]["sglang_engine_owned_target_tensor"] is True
    assert result["proof"]["receiver_request_from_sglang_engine_weight"] is True
    assert result["proof"]["receiver_segment_assembly_used"] is True
    assert result["proof"]["receiver_installed_into_sglang_engine_owned_tensor"] is True
    assert result["proof"]["real_runtime_engine_used"] is True
    assert result["proof"]["actual_nixl_reads_used"] is False
    assert result["proof"]["synthetic_trainer_payloads_used"] is True
    assert result["validation"]["assembled_allclose"] is True
    assert result["validation"]["checksum_matches"] is True
    assert result["validation"]["restored_original"] is True
    assert engine.get_weight_calls >= 3
    assert engine.update_weight_calls == 2
    torch.testing.assert_close(engine.weights["lm_head.weight"], original)

    payload = json.loads(artifact_path.read_text())
    assert payload["proof"]["sglang_engine_update_weights_from_tensor_used"] is True
    assert payload["metrics"]["segment_count"] == 2


def test_sglang_receiver_smoke_uses_engine_multitensor_update_path(tmp_path):
    originals = {
        "model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(
            8, 4
        ),
        "lm_head.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
    }
    engine = FakeSglangEngine(originals)
    artifact_path = tmp_path / "sglang-engine-multitensor-receiver-smoke.json"

    result = run_sglang_receiver_multitensor_refit_on_engine(
        engine,
        model_name="tiny-sglang-engine",
        model_version="step-engine-multitensor",
        sglang_version="unit-test",
        preferred_tensor_names=(
            "model.embed_tokens.weight",
            "lm_head.weight",
        ),
        artifact_path=artifact_path,
        model_path="unit-tiny-llama",
    )

    assert result["result"] == "pass"
    assert result["runtime_framework"] == "sglang"
    assert result["target_tensor_count"] == 2
    assert result["target_tensor_names"] == [
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
    assert result["proof"]["sglang_engine_started"] is True
    assert result["proof"]["sglang_engine_get_weights_by_name_used"] is True
    assert result["proof"]["sglang_engine_update_weights_from_tensor_used"] is True
    assert result["proof"]["sglang_engine_owned_target_tensors"] is True
    assert result["proof"]["receiver_requests_from_sglang_engine_weights"] is True
    assert result["proof"]["receiver_segment_assembly_used"] is True
    assert (
        result["proof"]["receiver_installed_into_sglang_engine_owned_tensors"] is True
    )
    assert result["proof"]["real_runtime_engine_used"] is True
    assert result["proof"]["live_runtime_engine_used"] is True
    assert result["proof"]["actual_nixl_reads_used"] is False
    assert result["validation"]["assembled_allclose"] is True
    assert result["validation"]["checksum_matches"] is True
    assert result["validation"]["restored_original"] is True
    assert result["metrics"]["segment_count"] == 4
    assert result["metrics"]["trainer_to_inference_bytes"] == 224
    assert result["metrics"]["source_count_per_target_tensor"] == {
        "model.embed_tokens.weight": 2,
        "lm_head.weight": 2,
    }
    assert engine.get_weight_calls >= 6
    assert engine.update_weight_calls == 2
    for name, original in originals.items():
        torch.testing.assert_close(engine.weights[name], original)

    payload = json.loads(artifact_path.read_text())
    assert payload["proof"]["sglang_engine_update_weights_from_tensor_used"] is True
    assert payload["target_tensor_count"] == 2
