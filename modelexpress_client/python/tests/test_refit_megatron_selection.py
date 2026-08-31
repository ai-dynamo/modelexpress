# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from modelexpress_rl.train.engines.megatron.selection import (
    _qkv_descriptor_extras,
    build_megatron_tensor_specs,
    model_express_role_for_mapping,
)


class AutoMapping:
    is_expert = False
    permute_dims = None

    def __init__(self, detected: str, *, is_expert: bool = False):
        self.detected = detected
        self.is_expert = is_expert

    def _detect_parallelism_type(self, _module):
        return self.detected


class RowParallelMapping:
    is_expert = False


class ColumnParallelMapping:
    is_expert = False


class DirectMapping:
    is_expert = False


class QKVMapping:
    is_expert = False


class GatedMLPMapping:
    is_expert = False


class ExpertGatedMLPMapping(GatedMLPMapping):
    is_expert = True

    def __init__(self):
        self.hf_param = {
            "gate": "model.layers.0.mlp.experts.4.gate_proj.weight",
            "up": "model.layers.0.mlp.experts.4.up_proj.weight",
        }


@pytest.mark.parametrize(
    ("detected", "tensor_ndim", "expected"),
    [
        ("column", 2, "column"),
        ("row", 2, "row"),
        ("row", 1, "replicated"),
        ("replicated", 1, "replicated"),
    ],
)
def test_auto_mapping_uses_owning_module_classification(
    detected, tensor_ndim, expected
):
    assert (
        model_express_role_for_mapping(
            mapping=AutoMapping(detected),
            megatron_module=SimpleNamespace(),
            tensor_ndim=tensor_ndim,
        )
        == expected
    )


def test_row_parallel_bias_is_replicated():
    assert (
        model_express_role_for_mapping(
            mapping=RowParallelMapping(),
            megatron_module=SimpleNamespace(),
            tensor_ndim=1,
        )
        == "replicated"
    )


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        (ColumnParallelMapping(), "column"),
        (DirectMapping(), "replicated"),
        (QKVMapping(), "qkv_column"),
        (GatedMLPMapping(), "gated_mlp_column"),
    ],
)
def test_explicit_mapping_roles(mapping, expected):
    assert (
        model_express_role_for_mapping(
            mapping=mapping,
            megatron_module=SimpleNamespace(),
            tensor_ndim=2,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("mapping", "expected"),
    [
        (ExpertGatedMLPMapping(), "expert_column"),
        (AutoMapping("row", is_expert=True), "expert_row"),
    ],
)
def test_expert_mapping_roles(mapping, expected):
    assert (
        model_express_role_for_mapping(
            mapping=mapping,
            megatron_module=SimpleNamespace(),
            tensor_ndim=2,
        )
        == expected
    )


def test_expert_specs_use_expert_tensor_parallel_geometry():
    gated = ExpertGatedMLPMapping()
    task = SimpleNamespace(
        mapping=gated,
        megatron_module=SimpleNamespace(),
        param_weight=torch.zeros(8, 3),
        global_param_name="decoder.layers.0.mlp.experts.linear_fc1.weight4",
    )

    specs = build_megatron_tensor_specs(
        conversion_tasks=[task],
        transformer_config=SimpleNamespace(),
        tensor_parallel_size=4,
        tensor_parallel_rank=3,
        expert_tensor_parallel_size=2,
        expert_tensor_parallel_rank=1,
    )

    assert len(specs) == 1
    assert specs[0].role == "expert_column"
    assert specs[0].global_shape == (16, 3)
    assert specs[0].local_shard_range == (8, 16)
    assert specs[0].hf_names == (
        "model.layers.0.mlp.experts.4.gate_proj.weight",
        "model.layers.0.mlp.experts.4.up_proj.weight",
    )


def test_expert_row_specs_use_expert_tensor_parallel_geometry():
    mapping = AutoMapping("row", is_expert=True)
    mapping.hf_param = "model.layers.0.mlp.experts.4.down_proj.weight"
    task = SimpleNamespace(
        mapping=mapping,
        megatron_module=SimpleNamespace(),
        param_weight=torch.zeros(3, 4),
        global_param_name="decoder.layers.0.mlp.experts.linear_fc2.weight4",
    )

    specs = build_megatron_tensor_specs(
        conversion_tasks=[task],
        transformer_config=SimpleNamespace(),
        tensor_parallel_size=4,
        tensor_parallel_rank=3,
        expert_tensor_parallel_size=2,
        expert_tensor_parallel_rank=1,
    )

    assert len(specs) == 1
    assert specs[0].role == "expert_row"
    assert specs[0].global_shape == (3, 8)
    assert specs[0].local_shard_range == (4, 8)


def test_qkv_bias_uses_the_qkv_column_role():
    assert (
        model_express_role_for_mapping(
            mapping=QKVMapping(),
            megatron_module=SimpleNamespace(),
            tensor_ndim=1,
        )
        == "qkv_column"
    )


def test_unknown_mapping_fails_instead_of_assuming_replicated():
    class UnknownMapping:
        is_expert = False

    with pytest.raises(NotImplementedError, match="UnknownMapping"):
        model_express_role_for_mapping(
            mapping=UnknownMapping(),
            megatron_module=SimpleNamespace(),
            tensor_ndim=2,
        )


def test_transformed_auto_mapping_subclass_is_not_treated_as_plain_auto_mapping():
    class RMSNorm2ZeroCenteredRMSNormMapping(AutoMapping):
        pass

    with pytest.raises(NotImplementedError, match="RMSNorm2ZeroCenteredRMSNormMapping"):
        model_express_role_for_mapping(
            mapping=RMSNorm2ZeroCenteredRMSNormMapping("replicated"),
            megatron_module=SimpleNamespace(),
            tensor_ndim=1,
        )


def test_qkv_metadata_supports_fewer_kv_heads_than_tp_ranks():
    extras = _qkv_descriptor_extras(
        transformer_config=SimpleNamespace(
            num_attention_heads=64,
            num_query_groups=2,
            kv_channels=128,
        ),
        tensor_parallel_size=8,
        global_rows=8704,
    )

    assert extras == {
        "qkv_interleave": "by_head",
        "head_dim": "128",
        "num_heads": "64",
        "num_kv_heads": "2",
    }


def test_qkv_metadata_keeps_local_counts_when_both_are_divisible():
    extras = _qkv_descriptor_extras(
        transformer_config=SimpleNamespace(
            num_attention_heads=32,
            num_query_groups=8,
            kv_channels=64,
        ),
        tensor_parallel_size=8,
        global_rows=3072,
    )

    assert extras["num_heads_local"] == "4"
    assert extras["num_kv_heads_local"] == "1"


def test_qkv_metadata_rejects_rows_that_disagree_with_head_geometry():
    with pytest.raises(ValueError, match="fused QKV rows"):
        _qkv_descriptor_extras(
            transformer_config=SimpleNamespace(
                num_attention_heads=64,
                num_query_groups=2,
                kv_channels=128,
            ),
            tensor_parallel_size=8,
            global_rows=1,
        )
