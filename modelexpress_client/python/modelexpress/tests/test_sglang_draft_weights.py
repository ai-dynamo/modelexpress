# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-specific draft selection and tensor namespace tests."""

from types import SimpleNamespace

import pytest
import torch
from modelexpress import p2p_pb2
from modelexpress.engines.sglang.draft_weights import (
    Qwen3_5ForCausalLMMTPWeightRule,
    Qwen3_5MoeForCausalLMWeightRule,
    draft_tensor_namespace,
    draft_weight_adapter_for,
)


def test_qwen35_selector_uses_its_sglang_model_rule_not_a_generic_prefix():
    selector = draft_weight_adapter_for("Qwen3_5ForCausalLMMTP", role="draft")

    assert selector.covers("mtp.fc.weight", "draft")
    assert selector.covers("model.mtp.layers.0.w", "draft")
    assert not selector.covers("model.layers.0.w", "draft")


@pytest.mark.parametrize(
    ("class_name", "loads_visual"),
    [
        ("Qwen3_5ForCausalLM", False),
        ("Qwen3_5MoeForCausalLM", False),
        ("Qwen3_5ForConditionalGeneration", True),
        ("Qwen3_5MoeForConditionalGeneration", True),
    ],
)
def test_qwen35_main_rules_match_each_sglang_model_class(class_name, loads_visual):
    selector = draft_weight_adapter_for(class_name, role="main")

    assert selector is not None
    assert selector.covers("model.layers.0.mlp.weight", "main")
    assert not selector.covers("mtp.fc.weight", "main")
    assert selector.covers("visual.blocks.0.weight", "main") is loads_visual
    assert not selector.covers("model.rotary_emb.inv_freq", "main")


def test_qwen35_draft_rule_skips_rotary_frequency_like_sglang():
    selector = draft_weight_adapter_for("Qwen3_5ForCausalLMMTP", role="draft")

    assert selector is not None
    assert selector.covers("mtp.fc.weight", "draft")
    assert selector.covers("model.embed_tokens.weight", "draft")
    assert selector.covers("model.language_model.embed_tokens.weight", "draft")
    assert not selector.covers("mtp.rotary_emb.inv_freq", "draft")


def test_selection_contract_covers_qwen_roles_and_shared_draft_storage():
    selector = Qwen3_5ForCausalLMMTPWeightRule()
    main = draft_weight_adapter_for("Qwen3_5ForCausalLM", role="main")
    embed = torch.nn.Parameter(torch.ones(2))
    own = torch.nn.Parameter(torch.ones(2))

    class Draft(torch.nn.Module):
        def get_embed_and_head(self):
            return embed, embed

    assert selector.compatible_main(main)
    assert selector.covers("mtp.fc.weight", "draft")
    assert not selector.covers("mtp.fc.weight", "main")
    assert selector.publishable(Draft(), {"embed": embed, "layer": own}) == {
        "layer": own
    }


def test_nextn_main_does_not_cover_layers_beyond_its_base_layer():
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=61, num_nextn_predict_layers=1)
    )
    selector = draft_weight_adapter_for(
        "DeepseekV3ForCausalLM", role="main", model=model
    )

    assert selector is not None
    assert selector.covers("model.layers.0.mlp.weight", "main")
    assert not selector.covers("model.layers.61.mlp.weight", "main")
    assert not selector.covers("model.layers.62.mlp.weight", "main")


def test_nextn_draft_matches_sglang_prefix_and_unconditional_skips():
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=61, num_nextn_predict_layers=1)
    )
    selector = draft_weight_adapter_for(
        "DeepseekV3ForCausalLMNextN", role="draft", model=model
    )

    assert selector is not None
    assert selector.covers("model.layers.61.mlp.weight", "draft")
    assert selector.covers("model.layers.610.mlp.weight", "draft")
    assert not selector.covers("model.layers.61.rotary_emb.inv_freq", "draft")
    assert not selector.covers("model.layers.61.shared_head.head.weight", "draft")


def test_deepseek_v4_dspark_draft_pairs_with_main_and_reads_three_stages():
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

    assert draft is not None
    assert draft.compatible_main(main)
    for stage in range(3):
        weight = f"mtp.{stage}.ffn.experts.0.w1.weight"
        assert draft.covers(weight, "draft")
        assert not main.covers(weight, "main")
    assert not draft.covers("layers.0.ffn.experts.0.w1.weight", "draft")


def test_deepseek_v4_dspark_draft_publishes_own_tensors_without_eagle_shared_hook():
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=43, num_nextn_predict_layers=1),
        embed_tokens=None,
        lm_head=None,
    )
    draft = draft_weight_adapter_for(
        "DeepseekV4ForCausalLMDSpark", role="draft", model=model
    )
    own = torch.nn.Parameter(torch.ones(2))

    assert draft is not None
    assert draft.publishable(model, {"stages.0.weight": own}) == {
        "stages.0.weight": own
    }


def test_qwen35_shared_weights_are_identified_by_tensor_not_name():
    selector = Qwen3_5ForCausalLMMTPWeightRule()
    embed = torch.nn.Parameter(torch.ones(2))
    head = torch.nn.Parameter(torch.ones(2))
    own = torch.nn.Parameter(torch.ones(2))

    class Draft(torch.nn.Module):
        def get_embed_and_head(self):
            return embed, head

    tensors = {"renamed.embedding": embed, "renamed.head": head, "layer": own}
    assert selector.publishable(Draft(), tensors) == {"layer": own}


def test_qwen35_shared_storage_view_is_not_registered_for_peer_transfer():
    selector = Qwen3_5ForCausalLMMTPWeightRule()
    embed = torch.nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))
    head = torch.nn.Parameter(torch.ones(2))

    class Draft(torch.nn.Module):
        def get_embed_and_head(self):
            return embed, head

    raw_storage_view = embed.detach().view(torch.uint8)
    own = torch.ones(2)
    assert selector.publishable(
        Draft(), {"renamed.embedding.__storage": raw_storage_view, "layer": own}
    ) == {"layer": own}


def test_qwen35_unrelated_empty_tensor_is_not_treated_as_shared():
    selector = Qwen3_5ForCausalLMMTPWeightRule()
    embed = torch.nn.Parameter(torch.empty(0))
    head = torch.nn.Parameter(torch.empty(0))

    class Draft(torch.nn.Module):
        def get_embed_and_head(self):
            return embed, head

    assert list(selector.publishable(Draft(), {"other": torch.empty(0)})) == ["other"]


def test_unknown_model_has_no_draft_weight_adapter():
    assert isinstance(
        draft_weight_adapter_for("Qwen3_5MoeForCausalLM", role="main"),
        Qwen3_5MoeForCausalLMWeightRule,
    )
    assert draft_weight_adapter_for("DeepseekV3ForCausalLM", role="main") is None


@pytest.mark.parametrize(
    ("main_class", "draft_class"),
    [
        ("Dots3NoteForCausalLM", "Dots3NoteForCausalLMNextN"),
        ("DeepseekV3ForCausalLM", "DeepseekV3ForCausalLMNextN"),
        ("DeepseekV32ForCausalLM", "DeepseekV3ForCausalLMNextN"),
        ("GlmMoeDsaForCausalLM", "GlmMoeDsaForCausalLMNextN"),
        ("DeepseekV4ForCausalLM", "DeepseekV4ForCausalLMNextN"),
        ("Glm4MoeForCausalLM", "Glm4MoeForCausalLMNextN"),
        ("Glm4MoeLiteForCausalLM", "Glm4MoeLiteForCausalLMNextN"),
        ("Glm5NextForConditionalGeneration", "Glm5NextForConditionalGenerationNextN"),
        ("GlmOcrForConditionalGeneration", "GlmOcrForConditionalGenerationNextN"),
        ("LongcatFlashForCausalLM", "LongcatFlashForCausalLMNextN"),
        ("MiMoForCausalLM", "MiMoMTP"),
        ("MiMoV2ForCausalLM", "MiMoV2MTP"),
        ("MiMoV2FlashForCausalLM", "MiMoV2MTP"),
        ("Step3p5ForCausalLM", "Step3p5MTP"),
        ("Step3p7ForConditionalGeneration", "Step3p5MTP"),
        ("InklingForConditionalGeneration", "InklingForConditionalGenerationMTP"),
        ("GigaChat35ForCausalLM", "GigaChat35ForCausalLMNextN"),
        ("BailingMoeForCausalLM", "BailingMoeForCausalLMNextN"),
        ("BailingMoeV2ForCausalLM", "BailingMoeForCausalLMNextN"),
        ("BailingMoeV2_5ForCausalLM", "BailingMoeForCausalLMNextN"),
        ("BailingMoeV3ForCausalLM", "BailingMoeForCausalLMNextN"),
        ("Ernie4_5_MoeForCausalLM", "Ernie4_5_MoeForCausalLMMTP"),
        ("Qwen3NextForCausalLM", "Qwen3NextForCausalLMMTP"),
        ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLMMTP"),
        ("Qwen3MoeForCausalLM", "Qwen3MoeForCausalLMMTP"),
        ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLMMTP"),
        ("Qwen3_5MoeForConditionalGeneration", "Qwen3_5ForCausalLMMTP"),
        ("Qwen3_5ForCausalLM", "Qwen3_5ForCausalLMMTP"),
        ("Qwen3_5MoeForCausalLM", "Qwen3_5ForCausalLMMTP"),
        ("InternS2PreviewForConditionalGeneration", "Qwen3_5ForCausalLMMTP"),
        ("InternS2MobiusForConditionalGeneration", "Qwen3_5ForCausalLMMTP"),
        ("ExaoneMoEForCausalLM", "ExaoneMoEForCausalLMMTP"),
        ("ExaoneMoeForCausalLM", "ExaoneMoEForCausalLMMTP"),
        ("NemotronHForCausalLM", "NemotronHForCausalLMMTP"),
        ("NemotronHPuzzleForCausalLM", "NemotronHForCausalLMMTP"),
        ("NemotronH_Omni_Reasoning_V3", "NemotronHForCausalLMMTP"),
        ("HYV3ForCausalLM", "HYV3ForCausalLMNextN"),
        ("HYV4ForCausalLM", "HYV4ForCausalLMNextN"),
    ],
)
def test_latest_sglang_two_pass_mtp_model_pairs_have_specific_rules(
    main_class, draft_class
):
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=61, num_nextn_predict_layers=1),
        mtp_layer_id=0,
        draft_model_idx=0,
    )
    main = draft_weight_adapter_for(main_class, role="main", model=model)
    draft = draft_weight_adapter_for(draft_class, role="draft", model=model)

    assert main is not None, main_class
    assert draft is not None, draft_class
    assert draft.compatible_main(main), (main_class, draft_class)


def test_longcat_main_and_draft_follow_distinct_mtp_prefixes():
    main = draft_weight_adapter_for("LongcatFlashForCausalLM", role="main")
    draft = draft_weight_adapter_for("LongcatFlashForCausalLMNextN", role="draft")

    assert main.covers("model.layers.0.mlp.weight", "main")
    assert not main.covers("model.mtp.layers.0.mlp.weight", "main")
    assert draft.covers("model.mtp.layers.0.mlp.weight", "draft")
    assert draft.covers("model.mtp.norm.weight", "draft")
    assert not draft.covers("model.layers.0.mlp.weight", "draft")
    assert not draft.covers("model.mtp.embed_tokens.weight", "draft")


def test_step3p5_draft_keeps_its_own_embedding_and_selected_extra_layer():
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=61, num_nextn_predict_layers=1),
        draft_model_idx=0,
    )
    main = draft_weight_adapter_for("Step3p5ForCausalLM", role="main", model=model)
    draft = draft_weight_adapter_for("Step3p5MTP", role="draft", model=model)

    assert not main.covers("model.layers.61.mlp.weight", "main")
    assert draft.covers("model.layers.61.mlp.weight", "draft")
    assert draft.covers("model.embed_tokens.weight", "draft")
    assert not draft.covers("model.layers.0.mlp.weight", "draft")
    assert not draft.covers("model.layers.62.mlp.weight", "draft")


def test_ernie_draft_selects_only_its_configured_mtp_layer():
    model = SimpleNamespace(config=SimpleNamespace(), mtp_layer_id=1)
    main = draft_weight_adapter_for("Ernie4_5_MoeForCausalLM", role="main", model=model)
    draft = draft_weight_adapter_for(
        "Ernie4_5_MoeForCausalLMMTP", role="draft", model=model
    )

    assert not main.covers("model.mtp_block.1.mlp.weight", "main")
    assert draft.covers("model.mtp_block.1.mlp.weight", "draft")
    assert draft.covers("model.mtp_emb_norm.1.weight", "draft")
    assert not draft.covers("model.mtp_block.0.mlp.weight", "draft")
    assert not draft.covers("model.layers.0.mlp.weight", "draft")


def test_nemotron_standalone_draft_head_is_not_discarded_from_p2p():
    draft = draft_weight_adapter_for("NemotronHForCausalLMMTP", role="draft")
    embed = torch.nn.Parameter(torch.ones(2))
    head = torch.nn.Parameter(torch.ones(2))

    class Draft(torch.nn.Module):
        _owns_lm_head = True

        def get_embed_and_head(self):
            return embed, head

    assert draft.covers("mtp.layers.0.weight", "draft")
    assert draft.covers("lm_head.weight", "draft")
    assert draft.publishable(Draft(), {"embed": embed, "head": head}) == {"head": head}


@pytest.mark.parametrize(
    ("main_class", "draft_class", "layer"),
    [
        ("DeepseekV3ForCausalLM", "DeepseekV3ForCausalLMNextN", 61),
        ("DeepseekV32ForCausalLM", "DeepseekV3ForCausalLMNextN", 61),
        ("Glm4MoeForCausalLM", "Glm4MoeForCausalLMNextN", 92),
        ("Glm4MoeLiteForCausalLM", "Glm4MoeLiteForCausalLMNextN", 46),
        ("GlmMoeDsaForCausalLM", "GlmMoeDsaForCausalLMNextN", 78),
    ],
)
def test_nextn_selects_only_its_configured_extra_layer(main_class, draft_class, layer):
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=layer, num_nextn_predict_layers=1)
    )
    main = draft_weight_adapter_for(main_class, role="main", model=model)
    draft = draft_weight_adapter_for(draft_class, role="draft", model=model)

    assert main is not None and draft is not None
    assert draft.compatible_main(main)
    own = f"model.layers.{layer}.self_attn.q_proj.weight"
    assert main.covers(own, "main") is False
    assert draft.covers(own, "draft") is True
    assert main.covers("model.layers.0.mlp.weight", "main") is True
    assert draft.covers("model.layers.0.mlp.weight", "draft") is False
    # SGLang checks startswith("model.layers.<N>") without a trailing dot.
    assert (
        draft.covers(f"model.layers.{layer}0.self_attn.q_proj.weight", "draft") is True
    )
    assert (
        draft.covers(f"model.layers.{layer}.shared_head.head.weight", "draft") is False
    )


def test_nextn_rejects_missing_or_unsupported_checkpoint_layout():
    missing = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=61))
    single_layer = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=1, num_nextn_predict_layers=1)
    )
    multiple = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=61, num_nextn_predict_layers=2)
    )
    assert (
        draft_weight_adapter_for("DeepseekV3ForCausalLM", role="main", model=missing)
        is None
    )
    assert (
        draft_weight_adapter_for(
            "DeepseekV3ForCausalLM", role="main", model=single_layer
        )
        is None
    )
    assert (
        draft_weight_adapter_for("DeepseekV3ForCausalLM", role="main", model=multiple)
        is None
    )
    assert (
        draft_weight_adapter_for("DeepseekV4ForCausalLM", role="main", model=multiple)
        is None
    )


def test_nextn_does_not_pair_deepseek_and_glm_manifests():
    model = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=61, num_nextn_predict_layers=1)
    )
    deepseek = draft_weight_adapter_for(
        "DeepseekV3ForCausalLM", role="main", model=model
    )
    glm = draft_weight_adapter_for(
        "GlmMoeDsaForCausalLMNextN", role="draft", model=model
    )
    assert not glm.compatible_main(deepseek)


def test_nextn_rejects_a_different_extra_layer_for_the_same_model_pair():
    main = draft_weight_adapter_for(
        "DeepseekV3ForCausalLM",
        role="main",
        model=SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=61, num_nextn_predict_layers=1)
        ),
    )
    draft = draft_weight_adapter_for(
        "DeepseekV3ForCausalLMNextN",
        role="draft",
        model=SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=62, num_nextn_predict_layers=1)
        ),
    )

    assert not draft.compatible_main(main)


def test_nextn_draft_excludes_target_shared_embed_and_head_storage():
    config = SimpleNamespace(num_hidden_layers=78, num_nextn_predict_layers=1)
    embed = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    head = torch.nn.Parameter(torch.ones(2))

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config

        def get_embed_and_head(self):
            return embed, head

    model = Draft()
    selector = draft_weight_adapter_for(
        "GlmMoeDsaForCausalLMNextN", role="draft", model=model
    )
    own = torch.ones(2)
    assert selector.publishable(
        model,
        {
            "embed.__storage": embed.detach().view(torch.uint8),
            "head": head,
            "decoder": own,
        },
    ) == {"decoder": own}


def test_draft_namespace_changes_with_revision_or_sglang_version():
    identity = p2p_pb2.SourceIdentity(model_name="org/qwen", revision="rev-a")
    main = draft_weight_adapter_for("Qwen3_5ForCausalLM", role="main")
    selector = Qwen3_5ForCausalLMMTPWeightRule()
    one = draft_tensor_namespace(
        identity, main, selector, "Qwen3_5ForCausalLMMTP", "0.5.16", "s3://bucket/qwen"
    )
    assert one.startswith("mx_draft::") and one.endswith("::")
    identity.revision = "rev-b"
    assert (
        draft_tensor_namespace(
            identity,
            main,
            selector,
            "Qwen3_5ForCausalLMMTP",
            "0.5.16",
            "s3://bucket/qwen",
        )
        != one
    )
    identity.revision = "rev-a"
    assert (
        draft_tensor_namespace(
            identity,
            main,
            selector,
            "Qwen3_5ForCausalLMMTP",
            "0.5.17",
            "s3://bucket/qwen",
        )
        != one
    )


def test_draft_namespace_requires_explicit_revision_and_engine_version():
    identity = p2p_pb2.SourceIdentity(model_name="org/qwen")
    main = draft_weight_adapter_for("Qwen3_5ForCausalLM", role="main")
    selector = Qwen3_5ForCausalLMMTPWeightRule()
    assert (
        draft_tensor_namespace(
            identity,
            main,
            selector,
            "Qwen3_5ForCausalLMMTP",
            "0.5.16",
            "s3://bucket/qwen",
        )
        is None
    )
    identity.revision = "rev-a"
    assert (
        draft_tensor_namespace(
            identity, main, selector, "Qwen3_5ForCausalLMMTP", "", "s3://bucket/qwen"
        )
        is None
    )


def test_draft_namespace_distinguishes_exact_main_model_classes():
    identity = p2p_pb2.SourceIdentity(model_name="org/qwen", revision="rev-a")
    text_main = draft_weight_adapter_for("Qwen3_5ForCausalLM", role="main")
    vision_main = draft_weight_adapter_for(
        "Qwen3_5ForConditionalGeneration", role="main"
    )
    draft = Qwen3_5ForCausalLMMTPWeightRule()

    assert draft_tensor_namespace(
        identity,
        text_main,
        draft,
        "Qwen3_5ForCausalLMMTP",
        "0.5.16",
        "s3://bucket/qwen",
    ) != draft_tensor_namespace(
        identity,
        vision_main,
        draft,
        "Qwen3_5ForCausalLMMTP",
        "0.5.16",
        "s3://bucket/qwen",
    )
