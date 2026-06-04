# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import torch
import torch.nn as nn

from modelexpress.refit_sglang_receiver_smoke import (
    run_sglang_receiver_refit_on_module,
)


class TinySglangOwnedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(8, 4)


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
