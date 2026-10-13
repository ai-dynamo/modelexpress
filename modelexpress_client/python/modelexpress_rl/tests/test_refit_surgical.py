# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from modelexpress_rl.inference.engines.vllm.surgical import (
    changed_since,
    module_groups,
)
from modelexpress_rl.inference.receiver import DeltaChange

PREFIX = "language_model.model.layers"
CHECKPOINT = [
    f"{PREFIX}.0.input_layernorm.weight",
    f"{PREFIX}.0.mlp.gate_proj.weight",
    f"{PREFIX}.0.mlp.up_proj.weight",
    f"{PREFIX}.0.mlp.down_proj.weight",
    f"{PREFIX}.1.self_attn.q_a_proj.weight",
    f"{PREFIX}.1.self_attn.kv_a_proj_with_mqa.weight",
    f"{PREFIX}.1.self_attn.q_a_layernorm.weight",
    f"{PREFIX}.1.mlp.experts.0.gate_proj.weight_packed",
    f"{PREFIX}.1.mlp.experts.0.gate_proj.weight_scale",
    f"{PREFIX}.1.mlp.experts.1.down_proj.weight_packed",
    f"{PREFIX}.1.mlp.shared_experts.gate_proj.weight",
    f"{PREFIX}.1.mlp.shared_experts.up_proj.weight",
    f"{PREFIX}.1.mlp.gate.weight",
    f"{PREFIX}.1.mlp.gate.e_score_correction_bias",
    "mm_projector.proj.2.weight",
    "mm_projector.proj.2.bias",
]


def test_module_groups_adds_packed_siblings_of_changed_tensors():
    groups = module_groups(
        {f"{PREFIX}.0.mlp.up_proj.weight", f"{PREFIX}.0.input_layernorm.weight"},
        CHECKPOINT,
        {"gate_up_proj": ["gate_proj", "up_proj"]},
    )

    assert groups == {
        f"{PREFIX}.0.input_layernorm.weight",
        f"{PREFIX}.0.mlp.gate_proj.weight",
        f"{PREFIX}.0.mlp.up_proj.weight",
    }


def test_module_groups_loads_every_expert_of_a_changed_fused_moe():
    groups = module_groups(
        {f"{PREFIX}.1.mlp.experts.0.gate_proj.weight_scale"}, CHECKPOINT, {}
    )

    assert groups == {
        f"{PREFIX}.1.mlp.experts.0.gate_proj.weight_packed",
        f"{PREFIX}.1.mlp.experts.0.gate_proj.weight_scale",
        f"{PREFIX}.1.mlp.experts.1.down_proj.weight_packed",
    }


def test_module_groups_use_default_fused_attention_and_keep_module_params():
    groups = module_groups(
        {
            f"{PREFIX}.1.self_attn.kv_a_proj_with_mqa.weight",
            f"{PREFIX}.1.mlp.gate.weight",
            "mm_projector.proj.2.weight",
        },
        CHECKPOINT,
        {},
    )

    assert groups == {
        f"{PREFIX}.1.self_attn.q_a_proj.weight",
        f"{PREFIX}.1.self_attn.kv_a_proj_with_mqa.weight",
        f"{PREFIX}.1.mlp.gate.weight",
        f"{PREFIX}.1.mlp.gate.e_score_correction_bias",
        "mm_projector.proj.2.weight",
        "mm_projector.proj.2.bias",
    }


def test_module_groups_reject_changed_names_absent_from_checkpoint():
    with pytest.raises(ValueError, match="not in the checkpoint"):
        module_groups({"missing.weight"}, CHECKPOINT, {})


LINEAGE = (
    DeltaChange("v2", "v3", frozenset({"a", "b"})),
    DeltaChange("v3", "v4", frozenset({"c"})),
)


def test_changed_since_unions_deltas_after_the_live_version():
    assert changed_since(LINEAGE, live_version="v2", target_version="v4") == {
        "a",
        "b",
        "c",
    }
    assert changed_since(LINEAGE, live_version="v3", target_version="v4") == {"c"}


@pytest.mark.parametrize(
    ("lineage", "live", "target"),
    [
        (LINEAGE, None, "v4"),
        (LINEAGE, "v1", "v4"),
        (LINEAGE, "v4", "v4"),
        (LINEAGE, "v2", "v5"),
        ((), "v2", "v4"),
    ],
)
def test_changed_since_requires_a_lineage_from_live_to_target(lineage, live, target):
    assert changed_since(lineage, live_version=live, target_version=target) is None
