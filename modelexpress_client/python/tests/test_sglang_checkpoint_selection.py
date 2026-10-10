# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""File selection for the two passes of a shared SGLang checkpoint."""

import fnmatch
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from modelexpress.engines.sglang.draft_weights import (
    Qwen3_5ForCausalLMMTPWeightRule,
    Qwen3_5ForCausalLMWeightRule,
    draft_weight_adapter_for,
)


def select_role_shards(*args):
    from modelexpress.engines.sglang.checkpoint_selection import select_role_shards

    return select_role_shards(*args)


def test_qwen35_selects_main_and_draft_shards_without_discarding_mixed_shards(
    tmp_path,
):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.weight": "main.safetensors",
                    "model.layers.1.weight": "mixed.safetensors",
                    "mtp.fc.weight": "mixed.safetensors",
                    "mtp.layers.0.weight": "draft.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    files = [
        str(tmp_path / "main.safetensors"),
        str(tmp_path / "mixed.safetensors"),
        str(tmp_path / "draft.safetensors"),
    ]
    main = draft_weight_adapter_for("Qwen3_5ForCausalLM", role="main")
    draft = draft_weight_adapter_for("Qwen3_5ForCausalLMMTP", role="draft")

    assert select_role_shards(str(tmp_path), files, main, "main") == files[:2]
    assert select_role_shards(str(tmp_path), files, draft, "draft") == files[1:]


def test_qwen4_text_only_main_skips_visual_only_shard_but_keeps_mixed_shard(
    tmp_path,
):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.weight": "main.safetensors",
                    "visual.blocks.0.weight": "visual.safetensors",
                    "model.layers.1.weight": "mixed.safetensors",
                    "visual.blocks.1.weight": "mixed.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    files = [
        str(tmp_path / name)
        for name in ("main.safetensors", "visual.safetensors", "mixed.safetensors")
    ]
    main = draft_weight_adapter_for(
        "Qwen4ExpForConditionalGeneration",
        role="main",
        model=SimpleNamespace(language_model_only=True),
    )

    assert select_role_shards(str(tmp_path), files, main, "main") == [
        files[0],
        files[2],
    ]
    multimodal_main = draft_weight_adapter_for(
        "Qwen4ExpForConditionalGeneration",
        role="main",
        model=SimpleNamespace(language_model_only=False),
    )
    assert select_role_shards(str(tmp_path), files, multimodal_main, "main") == files


def test_glm_nextn_selects_extra_layer_and_preserves_mixed_shard(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {
                "model.layers.77.mlp.weight": "main.safetensors",
                "model.layers.78.shared_head.norm.weight": "mixed.safetensors",
                "model.layers.77.self_attn.weight": "mixed.safetensors",
                "model.layers.78.self_attn.weight": "draft.safetensors",
            }}
        ),
        encoding="utf-8",
    )
    files = [
        str(tmp_path / name)
        for name in ("main.safetensors", "mixed.safetensors", "draft.safetensors")
    ]
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=78, num_nextn_predict_layers=1)
    )
    main = draft_weight_adapter_for(
        "GlmMoeDsaForCausalLM", role="main", model=model
    )
    draft = draft_weight_adapter_for(
        "GlmMoeDsaForCausalLMNextN", role="draft", model=model
    )

    assert select_role_shards(str(tmp_path), files, main, "main") == files[:2]
    assert select_role_shards(str(tmp_path), files, draft, "draft") == files[1:]


def test_deepseek_v4_dspark_selects_only_its_three_draft_shards(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {
                "layers.0.ffn.experts.0.w1.weight": "main.safetensors",
                "mtp.0.ffn.experts.0.w1.weight": "draft-0.safetensors",
                "mtp.1.ffn.experts.0.w1.weight": "draft-1.safetensors",
                "mtp.2.ffn.experts.0.w1.weight": "draft-2.safetensors",
            }}
        ),
        encoding="utf-8",
    )
    files = [str(tmp_path / name) for name in (
        "main.safetensors", "draft-0.safetensors", "draft-1.safetensors", "draft-2.safetensors"
    )]
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=43,
            num_nextn_predict_layers=1,
            dspark_target_layer_ids=[40, 41, 42],
        )
    )
    main = draft_weight_adapter_for("DeepseekV4ForCausalLM", role="main", model=model)
    draft = draft_weight_adapter_for(
        "DeepseekV4ForCausalLMDSpark", role="draft", model=model
    )

    assert select_role_shards(str(tmp_path), files, main, "main") == files[:1]
    assert select_role_shards(str(tmp_path), files, draft, "draft") == files[1:]


def test_qwen35_missing_or_incomplete_index_falls_back_to_all_shards(tmp_path):
    files = [str(tmp_path / "main.safetensors"), str(tmp_path / "draft.safetensors")]
    selector = Qwen3_5ForCausalLMMTPWeightRule()

    assert select_role_shards(str(tmp_path), files, selector, "draft") is None

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.weight": "main.safetensors"}}),
        encoding="utf-8",
    )
    assert select_role_shards(
        str(tmp_path), files, Qwen3_5ForCausalLMWeightRule(), "main"
    ) is None

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.fc.weight": "missing.safetensors"}}),
        encoding="utf-8",
    )
    assert select_role_shards(str(tmp_path), files, selector, "draft") is None


def test_qwen35_draft_without_matching_tensor_falls_back(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layers.0.weight": "main.safetensors"}}),
        encoding="utf-8",
    )
    files = [str(tmp_path / "main.safetensors")]

    assert select_role_shards(
        str(tmp_path), files, Qwen3_5ForCausalLMMTPWeightRule(), "draft"
    ) is None


def test_model_weight_rule_error_keeps_full_checkpoint_fallback(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"main.weight": "main.safetensors"}}),
        encoding="utf-8",
    )

    class BrokenRule:
        def covers(self, name, role):
            raise ValueError("unsupported tensor name")

    assert select_role_shards(
        str(tmp_path), [str(tmp_path / "main.safetensors")], BrokenRule(), "main"
    ) is None


def test_object_store_reads_only_index_before_selecting_shards(monkeypatch):
    calls = []

    def pull_files(uri, directory, allow_pattern):
        calls.append((uri, allow_pattern))
        if not any(
            fnmatch.fnmatch("model/model.safetensors.index.json", pattern)
            for pattern in allow_pattern
        ):
            return
        Path(directory, "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {
                "model.layers.0.weight": "main.safetensors",
                "mtp.fc.weight": "draft.safetensors",
            }}), encoding="utf-8",
        )

    monkeypatch.setitem(sys.modules, "runai_model_streamer", SimpleNamespace(
        pull_files=pull_files,
    ))
    files = [
        "s3://bucket/model/main.safetensors",
        "s3://bucket/model/draft.safetensors",
    ]

    assert select_role_shards(
        "s3://bucket/model", files, Qwen3_5ForCausalLMMTPWeightRule(), "draft"
    ) == files[1:]
    assert calls == [
        ("s3://bucket/model", ["model.safetensors.index.json", "*/model.safetensors.index.json"])
    ]


def test_object_store_index_read_error_keeps_sglang_full_load(monkeypatch):
    def pull_files(*args, **kwargs):
        raise RuntimeError("object store index unavailable")

    monkeypatch.setitem(
        sys.modules,
        "runai_model_streamer",
        SimpleNamespace(pull_files=pull_files),
    )
    files = ["s3://bucket/model/main.safetensors"]
    assert select_role_shards(
        "s3://bucket/model", files, Qwen3_5ForCausalLMWeightRule(), "main"
    ) is None
