# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from modelexpress.refit_vllm_receiver_smoke import (
    _find_vllm_module,
    _select_refit_tensor,
    _select_refit_tensors,
    _source_ownerships_for_tensor,
    run_receiver_refit_on_module,
    run_receiver_multitensor_refit_on_vllm_llm,
    run_receiver_refit_on_vllm_llm,
)


class TinyVllmOwnedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = nn.Linear(4, 6, bias=False)


class TinyMultitensorVllmOwnedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 6, bias=False)


def test_vllm_receiver_smoke_installs_and_restores_runtime_owned_tensor(tmp_path):
    module = TinyVllmOwnedModule()
    with torch.no_grad():
        module.lm_head.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(6, 4))
    original = module.lm_head.weight.detach().clone()
    artifact_path = tmp_path / "vllm-receiver-smoke.json"

    result = run_receiver_refit_on_module(
        module,
        model_name="tiny-vllm",
        model_version="step-11",
        module_path="unit.llm.model",
        vllm_version="unit-test",
        artifact_path=artifact_path,
    )

    assert result["result"] == "pass"
    assert result["runtime_framework"] == "vllm"
    assert result["framework_version"] == "unit-test"
    assert result["target_tensor_name"] == "lm_head.weight"
    assert result["target_key"] == "vllm:lm_head.weight"
    assert result["request"]["runtime_framework"] == "vllm"
    assert result["request"]["layout_tags"]["runtime_module_path"] == "unit.llm.model"
    assert result["request"]["layout_tags"]["vllm_tensor_name"] == "lm_head.weight"
    assert result["proof"]["vllm_owned_target_tensor"] is True
    assert result["proof"]["runtime_owned_target_tensor"] is True
    assert result["proof"]["receiver_request_from_runtime_owned_tensor"] is True
    assert result["proof"]["target_slice_spans_multiple_trainers"] is True
    assert result["proof"]["receiver_installed_into_vllm_owned_tensor"] is True
    assert result["proof"]["receiver_installed_into_runtime_owned_tensor"] is True
    assert result["proof"]["restored_original_tensor"] is True
    assert result["proof"]["runtime_imported"] is False
    assert result["proof"]["real_runtime_engine_used"] is False
    assert result["proof"]["actual_nixl_reads_used"] is False
    assert result["proof"]["synthetic_trainer_payloads_used"] is True
    assert [plan["source_id"] for plan in result["segment_plans"]] == [
        "trainer-rank0",
        "trainer-rank1",
    ]
    torch.testing.assert_close(module.lm_head.weight, original)

    payload = json.loads(artifact_path.read_text())
    assert payload["validation"]["checksum_matches"] is True
    assert payload["metrics"]["trainer_to_inference_bytes"] == 96
    assert payload["metrics"]["segment_count"] == 2


def test_vllm_receiver_smoke_uses_apply_model_for_worker_owned_tensor(tmp_path):
    module = TinyVllmOwnedModule()
    with torch.no_grad():
        module.lm_head.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(6, 4))
    original = module.lm_head.weight.detach().clone()

    class FakeVllmLLM:
        def __init__(self, worker_module):
            self.worker_module = worker_module
            self.apply_model_calls = 0

        def apply_model(self, func):
            self.apply_model_calls += 1
            return [func(self.worker_module)]

    llm = FakeVllmLLM(module)
    artifact_path = tmp_path / "vllm-apply-model-receiver-smoke.json"

    result = run_receiver_refit_on_vllm_llm(
        llm,
        model_name="tiny-vllm",
        model_version="step-apply-model",
        vllm_version="unit-test",
        artifact_path=artifact_path,
    )

    assert llm.apply_model_calls == 1
    assert result["result"] == "pass"
    assert result["runtime_module_discovery_path"] == "vllm.apply_model"
    assert result["worker_result_count"] == 1
    assert result["proof"]["vllm_apply_model_used"] is True
    assert result["proof"]["vllm_worker_owned_target_tensor"] is True
    assert result["proof"]["real_runtime_engine_used"] is True
    assert result["proof"]["vllm_apply_model_insecure_serialization_enabled"] is False
    assert result["validation"]["checksum_matches"] is True
    assert result["validation"]["restored_original"] is True
    torch.testing.assert_close(module.lm_head.weight, original)

    payload = json.loads(artifact_path.read_text())
    assert payload["proof"]["vllm_apply_model_used"] is True


def test_vllm_receiver_multitensor_smoke_uses_apply_model(tmp_path):
    module = TinyMultitensorVllmOwnedModule()
    with torch.no_grad():
        module.model.embed_tokens.weight.copy_(
            torch.arange(32, dtype=torch.float32).reshape(8, 4)
        )
        module.lm_head.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(6, 4))
    originals = {
        name: tensor.detach().clone() for name, tensor in module.named_parameters()
    }

    class FakeVllmLLM:
        def __init__(self, worker_module):
            self.worker_module = worker_module
            self.apply_model_calls = 0

        def apply_model(self, func):
            self.apply_model_calls += 1
            return [func(self.worker_module)]

    llm = FakeVllmLLM(module)
    artifact_path = tmp_path / "vllm-apply-model-multitensor-smoke.json"

    result = run_receiver_multitensor_refit_on_vllm_llm(
        llm,
        model_name="tiny-vllm",
        model_version="step-apply-model-multitensor",
        vllm_version="unit-test",
        preferred_tensor_names=(
            "model.embed_tokens.weight",
            "lm_head.weight",
        ),
        artifact_path=artifact_path,
    )

    assert llm.apply_model_calls == 1
    assert result["result"] == "pass"
    assert result["runtime_module_discovery_path"] == "vllm.apply_model"
    assert result["worker_result_count"] == 1
    assert result["target_tensor_count"] == 2
    assert result["target_tensor_names"] == [
        "model.embed_tokens.weight",
        "lm_head.weight",
    ]
    assert result["proof"]["vllm_apply_model_used"] is True
    assert result["proof"]["vllm_worker_owned_target_tensors"] is True
    assert result["proof"]["real_runtime_engine_used"] is True
    assert result["proof"]["live_runtime_engine_used"] is True
    assert result["proof"]["actual_nixl_reads_used"] is False
    assert result["validation"]["allclose"] is True
    assert result["validation"]["checksum_matches"] is True
    assert result["validation"]["restored_original"] is True
    assert result["metrics"]["segment_count"] == 4
    assert result["metrics"]["trainer_to_inference_bytes"] == 224
    assert result["metrics"]["source_count_per_target_tensor"] == {
        "model.embed_tokens.weight": 2,
        "lm_head.weight": 2,
    }
    for name, original in originals.items():
        torch.testing.assert_close(dict(module.named_parameters())[name], original)

    payload = json.loads(artifact_path.read_text())
    assert payload["proof"]["vllm_apply_model_used"] is True
    assert payload["target_tensor_count"] == 2


def test_vllm_receiver_smoke_source_ownerships_split_rows_exactly():
    tensor = torch.empty((7, 4), dtype=torch.bfloat16)

    owners = _source_ownerships_for_tensor(
        tensor_name="lm_head.weight",
        tensor=tensor,
        model_name="tiny-vllm",
        model_version="step-11",
    )

    assert [owner.source_id for owner in owners] == ["trainer-rank0", "trainer-rank1"]
    assert owners[0].source_range == ((0, 3), (0, 4))
    assert owners[1].source_range == ((3, 7), (0, 4))
    assert owners[0].dtype == "bfloat16"
    assert owners[0].element_size_bytes == 2


def test_vllm_receiver_smoke_selects_preferred_tensor_suffix():
    tensors = [
        ("model.layers.0.down_proj.weight", torch.empty((4, 4))),
        ("lm_head.weight", torch.empty((6, 4))),
    ]

    name, selected = _select_refit_tensor(tensors)

    assert name == "lm_head.weight"
    assert tuple(selected.shape) == (6, 4)


def test_vllm_receiver_smoke_selects_multiple_preferred_tensors():
    tensors = [
        ("model.layers.0.down_proj.weight", torch.empty((4, 4))),
        ("model.embed_tokens.weight", torch.empty((8, 4))),
        ("lm_head.weight", torch.empty((6, 4))),
    ]

    selected = _select_refit_tensors(
        tensors,
        preferred_names=("model.embed_tokens.weight", "lm_head.weight"),
    )

    assert list(selected) == ["model.embed_tokens.weight", "lm_head.weight"]


def test_vllm_receiver_smoke_rejects_missing_preferred_tensor():
    with pytest.raises(RuntimeError, match="preferred vLLM tensor"):
        _select_refit_tensor(
            [("lm_head.weight", torch.empty((6, 4)))],
            preferred_name="model.embed_tokens.weight",
        )


def test_vllm_receiver_smoke_finds_nested_module_without_tensor_truthiness():
    class Root:
        pass

    class Engine:
        pass

    root = Root()
    root.llm_engine = Engine()
    root.llm_engine.model = TinyVllmOwnedModule()

    path, module = _find_vllm_module(root)

    assert path == "llm.llm_engine.model"
    assert isinstance(module, TinyVllmOwnedModule)
