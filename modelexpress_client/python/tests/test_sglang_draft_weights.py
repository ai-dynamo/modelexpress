# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-specific draft selection and tensor namespace tests."""

from types import SimpleNamespace

import pytest
import torch
from modelexpress import p2p_pb2
from modelexpress.engines.sglang.draft_weights import (
    Qwen35MtpWeights,
    draft_tensor_namespace,
    draft_weight_adapter_for,
)


def test_qwen35_selector_uses_its_sglang_model_rule_not_a_generic_prefix():
    selector = Qwen35MtpWeights()

    assert selector.supports_main("Qwen3_5MoeForCausalLM")
    assert selector.supports_draft("Qwen3_5ForCausalLMMTP")
    assert selector.includes("mtp.fc.weight")
    assert selector.includes("model.mtp.layers.0.w")
    assert not selector.includes("model.layers.0.w")
    assert not selector.supports_main("DeepseekV3ForCausalLM")


def test_qwen35_shared_weights_are_identified_by_tensor_not_name():
    selector = Qwen35MtpWeights()
    embed = torch.nn.Parameter(torch.ones(2))
    head = torch.nn.Parameter(torch.ones(2))
    own = torch.nn.Parameter(torch.ones(2))

    class Draft(torch.nn.Module):
        def get_embed_and_head(self):
            return embed, head

    tensors = {"renamed.embedding": embed, "renamed.head": head, "layer": own}
    assert selector.transferable_tensors(Draft(), tensors) == {"layer": own}


def test_qwen35_shared_storage_view_is_not_registered_for_peer_transfer():
    selector = Qwen35MtpWeights()
    embed = torch.nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))
    head = torch.nn.Parameter(torch.ones(2))

    class Draft(torch.nn.Module):
        def get_embed_and_head(self):
            return embed, head

    raw_storage_view = embed.detach().view(torch.uint8)
    own = torch.ones(2)
    assert selector.transferable_tensors(
        Draft(), {"renamed.embedding.__storage": raw_storage_view, "layer": own}
    ) == {"layer": own}


def test_qwen35_unrelated_empty_tensor_is_not_treated_as_shared():
    selector = Qwen35MtpWeights()
    embed = torch.nn.Parameter(torch.empty(0))
    head = torch.nn.Parameter(torch.empty(0))

    class Draft(torch.nn.Module):
        def get_embed_and_head(self):
            return embed, head

    assert list(selector.transferable_tensors(Draft(), {"other": torch.empty(0)})) == [
        "other"
    ]


def test_unknown_model_has_no_draft_weight_adapter():
    assert isinstance(
        draft_weight_adapter_for("Qwen3_5MoeForCausalLM", role="main"),
        Qwen35MtpWeights,
    )
    assert draft_weight_adapter_for("DeepseekV3ForCausalLM", role="main") is None


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
def test_nextn_selects_only_its_configured_extra_layer(
    main_class, draft_class, layer
):
    model = SimpleNamespace(
        config=SimpleNamespace(
            num_hidden_layers=layer, num_nextn_predict_layers=1
        )
    )
    main = draft_weight_adapter_for(main_class, role="main", model=model)
    draft = draft_weight_adapter_for(draft_class, role="draft", model=model)

    assert main is not None and draft is not None
    assert main.compatibility_tag == draft.compatibility_tag
    own = f"model.layers.{layer}.self_attn.q_proj.weight"
    assert main.uses_checkpoint_tensor(own, "main") is False
    assert draft.uses_checkpoint_tensor(own, "draft") is True
    assert main.uses_checkpoint_tensor("model.layers.0.mlp.weight", "main") is True
    assert draft.uses_checkpoint_tensor("model.layers.0.mlp.weight", "draft") is False
    assert draft.uses_checkpoint_tensor(
        f"model.layers.{layer}0.self_attn.q_proj.weight", "draft"
    ) is False
    assert draft.uses_checkpoint_tensor(
        f"model.layers.{layer}.shared_head.head.weight", "draft"
    ) is False


def test_nextn_rejects_missing_or_unsupported_checkpoint_layout():
    missing = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=61))
    single_layer = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=1, num_nextn_predict_layers=1)
    )
    multiple = SimpleNamespace(
        config=SimpleNamespace(num_hidden_layers=61, num_nextn_predict_layers=2)
    )
    assert draft_weight_adapter_for(
        "DeepseekV3ForCausalLM", role="main", model=missing
    ) is None
    assert draft_weight_adapter_for(
        "DeepseekV3ForCausalLM", role="main", model=single_layer
    ) is None
    assert draft_weight_adapter_for(
        "DeepseekV3ForCausalLM", role="main", model=multiple
    ) is None
    assert draft_weight_adapter_for(
        "DeepseekV4ForCausalLM", role="main", model=multiple
    ) is None


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
    assert deepseek.compatibility_tag != glm.compatibility_tag


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
    assert selector.transferable_tensors(
        model,
        {
            "embed.__storage": embed.detach().view(torch.uint8),
            "head": head,
            "decoder": own,
        },
    ) == {"decoder": own}


def test_draft_namespace_changes_with_revision_or_sglang_version():
    identity = p2p_pb2.SourceIdentity(model_name="org/qwen", revision="rev-a")
    selector = Qwen35MtpWeights()
    one = draft_tensor_namespace(
        identity, selector, "Qwen3_5ForCausalLMMTP", "0.5.16", "s3://bucket/qwen"
    )
    assert one.startswith("mx_draft::") and one.endswith("::")
    identity.revision = "rev-b"
    assert (
        draft_tensor_namespace(
            identity, selector, "Qwen3_5ForCausalLMMTP", "0.5.16", "s3://bucket/qwen"
        )
        != one
    )
    identity.revision = "rev-a"
    assert (
        draft_tensor_namespace(
            identity, selector, "Qwen3_5ForCausalLMMTP", "0.5.17", "s3://bucket/qwen"
        )
        != one
    )


def test_draft_namespace_requires_explicit_revision_and_engine_version():
    identity = p2p_pb2.SourceIdentity(model_name="org/qwen")
    selector = Qwen35MtpWeights()
    assert (
        draft_tensor_namespace(
            identity, selector, "Qwen3_5ForCausalLMMTP", "0.5.16", "s3://bucket/qwen"
        )
        is None
    )
    identity.revision = "rev-a"
    assert (
        draft_tensor_namespace(
            identity, selector, "Qwen3_5ForCausalLMMTP", "", "s3://bucket/qwen"
        )
        is None
    )
