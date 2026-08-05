# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch

from modelexpress.refit.source import megatron_bridge
from modelexpress.refit.source.canonical import CanonicalTensorSpec
from modelexpress.refit.source.megatron_bridge import (
    MegatronBridgeHfBucketConfig,
    for_each_megatron_hf_bucket,
)


@dataclass(frozen=True)
class QkvMapping:
    hf_param: dict[str, str]


@dataclass(frozen=True)
class GroupedExpertMapping:
    hf_param: str
    group_key: str
    is_grouped_export: bool = True


@dataclass(frozen=True)
class ConversionTask:
    param_name: str
    mapping: object
    vp_stage: int | None = 0
    param_weight: torch.Tensor | None = None


class GroupedBridge:
    def __init__(self, outputs):
        self.outputs = outputs
        self.tasks = [
            ConversionTask(
                "decoder.layers.0.self_attention.linear_qkv.weight",
                QkvMapping(
                    {
                        "q": "model.layers.0.self_attn.q_proj.weight",
                        "k": "model.layers.0.self_attn.k_proj.weight",
                        "v": "model.layers.0.self_attn.v_proj.weight",
                    }
                ),
            ),
            ConversionTask(
                "decoder.layers.0.mlp.experts.weight0",
                GroupedExpertMapping(
                    "model.layers.0.mlp.experts.weight", "layer-0-experts"
                ),
            ),
            ConversionTask(
                "decoder.layers.0.mlp.experts.weight1",
                GroupedExpertMapping(
                    "model.layers.0.mlp.experts.weight", "layer-0-experts"
                ),
            ),
        ]
        self.export_calls = []
        self.drained = False

    def get_conversion_tasks(self, model):
        assert model is MODEL
        return self.tasks

    def export_hf_weights(self, model, **kwargs):
        assert model is MODEL
        self.export_calls.append(kwargs)
        try:
            yield from self.outputs
        finally:
            self.drained = True


MODEL = object()


def _schema():
    names = (
        "model.layers.0.mlp.experts.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.norm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    )
    return tuple(CanonicalTensorSpec(name, (2, 2), torch.float32) for name in names)


def test_grouped_bridge_outputs_are_reordered_to_complete_canonical_hf_order(tmp_path):
    values = {
        name: torch.arange(4, dtype=torch.float32).reshape(2, 2) + ordinal
        for ordinal, name in enumerate(spec.name for spec in _schema())
    }
    # The QKV conversion emits a non-contiguous tensor view and names that are
    # separated by another HF tensor in the authoritative canonical order.
    values["model.layers.0.self_attn.q_proj.weight"] = torch.arange(
        8, dtype=torch.float32
    ).reshape(2, 4)[:, ::2]
    bridge = GroupedBridge(
        [
            (
                "model.layers.0.self_attn.q_proj.weight",
                values["model.layers.0.self_attn.q_proj.weight"],
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                values["model.layers.0.self_attn.k_proj.weight"],
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                values["model.layers.0.self_attn.v_proj.weight"],
            ),
            (
                "model.layers.0.mlp.experts.weight",
                values["model.layers.0.mlp.experts.weight"],
            ),
            (
                "model.layers.0.self_attn.norm.weight",
                values["model.layers.0.self_attn.norm.weight"],
            ),
        ]
    )
    buckets = []

    for_each_megatron_hf_bucket(
        MODEL,
        MegatronBridgeHfBucketConfig(
            bridge=bridge,
            canonical_schema=_schema(),
            bucket_bytes=32,
            spool_directory=tmp_path,
        ),
        buckets.append,
    )

    flattened = [(name, tensor) for bucket in buckets for name, tensor in bucket]
    assert [name for name, _tensor in flattened] == [spec.name for spec in _schema()]
    assert all(
        tensor.device.type == "cpu" and tensor.is_contiguous()
        for _, tensor in flattened
    )
    assert all(sum(tensor.nbytes for _, tensor in bucket) <= 32 for bucket in buckets)
    assert bridge.drained
    assert len(bridge.export_calls) == 1
    assert bridge.export_calls[0]["conversion_tasks"] == bridge.tasks


def test_real_megatron_bridge_qkv_and_grouped_contracts_when_installed(
    monkeypatch, tmp_path
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    param_mapping = pytest.importorskip(
        "megatron.bridge.models.conversion.param_mapping",
        reason="Megatron-Bridge is installed only in the Miles target image",
    )
    model_bridge = pytest.importorskip(
        "megatron.bridge.models.conversion.model_bridge",
        reason="Megatron-Bridge is installed only in the Miles target image",
    )

    q_name = "model.layers.0.self_attn.q_proj.weight"
    k_name = "model.layers.0.self_attn.k_proj.weight"
    v_name = "model.layers.0.self_attn.v_proj.weight"
    expert_name = "model.layers.0.mlp.experts.down_proj.weight"
    norm_name = "model.layers.0.self_attn.norm.weight"
    qkv_native_name = "decoder.layers.0.self_attention.linear_qkv.weight"

    def noncontiguous_copy(tensor):
        storage = torch.empty(
            (*tensor.shape[:-1], tensor.shape[-1] * 2), dtype=tensor.dtype
        )
        view = storage[..., ::2]
        view.copy_(tensor)
        assert not view.is_contiguous()
        return view

    provider = SimpleNamespace(
        attention_output_gate=False,
        hidden_size=4,
        kv_channels=1,
        num_attention_heads=4,
        num_query_groups=2,
    )
    q = noncontiguous_copy(torch.arange(16, dtype=torch.float32).reshape(4, 4))
    k = noncontiguous_copy(torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100)
    v = noncontiguous_copy(torch.arange(8, dtype=torch.float32).reshape(2, 4) + 200)
    packed = param_mapping.merge_qkv_weights(provider, q, k, v)
    packed = noncontiguous_copy(packed)

    qkv_mapping = param_mapping.QKVMapping(
        qkv_native_name,
        q=q_name,
        k=k_name,
        v=v_name,
    )
    qkv_mapping._get_config = lambda _module: provider
    qkv_mapping.broadcast_obj_from_pp_rank = lambda value, _key: value
    qkv_mapping._tp_mapping.megatron_to_hf = lambda weights, _module: {
        qkv_native_name: weights
    }
    converted = qkv_mapping.megatron_to_hf(packed, object())

    torch.testing.assert_close(converted[q_name], q)
    torch.testing.assert_close(converted[k_name], k)
    torch.testing.assert_close(converted[v_name], v)

    expert_native_names = (
        "decoder.layers.0.mlp.experts.weight0",
        "decoder.layers.0.mlp.experts.weight1",
    )
    expert_mappings = tuple(
        param_mapping.FusedExpertMapping(native_name, expert_name)
        for native_name in expert_native_names
    )
    assert all(mapping.is_grouped_export for mapping in expert_mappings)
    assert {mapping.group_key for mapping in expert_mappings} == {expert_name}
    for native_name, mapping in zip(expert_native_names, expert_mappings, strict=True):
        delegate = param_mapping.DirectMapping(native_name, expert_name)
        delegate.broadcast_from_pp_rank = lambda tensor, **_kwargs: tensor
        mapping._mapping = delegate

    WeightConversionTask = model_bridge.WeightConversionTask
    HFWeightTuple = model_bridge.HFWeightTuple
    assert HFWeightTuple._fields == (
        "param_name",
        "weight",
        "megatron_param_name",
    )
    tasks = [
        WeightConversionTask(
            qkv_native_name,
            qkv_native_name,
            qkv_mapping,
            pp_rank=0,
            vp_stage=0,
            param_weight=packed,
        ),
        *(
            WeightConversionTask(
                native_name,
                native_name,
                mapping,
                pp_rank=0,
                vp_stage=0,
                param_weight=torch.zeros((2, 2), dtype=torch.float32),
            )
            for native_name, mapping in zip(
                expert_native_names, expert_mappings, strict=True
            )
        ),
        WeightConversionTask(
            "decoder.layers.0.self_attention.norm.weight",
            "decoder.layers.0.self_attention.norm.weight",
            param_mapping.DirectMapping(
                "decoder.layers.0.self_attention.norm.weight", norm_name
            ),
            pp_rank=0,
            vp_stage=0,
            param_weight=torch.zeros((2, 2), dtype=torch.float32),
        ),
    ]
    expert_parts = (
        noncontiguous_copy(torch.arange(4, dtype=torch.float32).reshape(2, 2)),
        noncontiguous_copy(torch.arange(4, dtype=torch.float32).reshape(2, 2) + 10),
    )
    converted_experts = tuple(
        mapping.megatron_to_hf(part, object())
        for mapping, part in zip(expert_mappings, expert_parts, strict=True)
    )
    assert all(
        converted[expert_name] is part
        for converted, part in zip(converted_experts, expert_parts, strict=True)
    )
    monkeypatch.setattr(
        model_bridge.parallel_state,
        "get_expert_model_parallel_world_size",
        lambda: 1,
    )
    grouped_buffers = {}
    model_config = SimpleNamespace(num_moe_experts=len(expert_parts))
    accumulate = model_bridge.MegatronModelBridge._accumulate_grouped_export
    assert (
        accumulate(
            object(),
            tasks[1],
            converted_experts[0],
            model_config,
            grouped_buffers,
            {},
        )
        is None
    )
    grouped = accumulate(
        object(),
        tasks[2],
        converted_experts[1],
        model_config,
        grouped_buffers,
        {},
    )
    assert grouped is not None
    expert = grouped[expert_name]
    torch.testing.assert_close(expert, torch.stack(expert_parts))
    assert grouped_buffers == {}
    norm = torch.arange(4, dtype=torch.float32).reshape(2, 2) + 300
    q_output = noncontiguous_copy(converted[q_name])
    outputs = (
        HFWeightTuple(q_name, q_output, qkv_native_name),
        HFWeightTuple(k_name, converted[k_name], qkv_native_name),
        HFWeightTuple(v_name, converted[v_name], qkv_native_name),
        HFWeightTuple(expert_name, expert, expert_native_names[-1]),
        HFWeightTuple(norm_name, norm, tasks[-1].param_name),
    )

    class NativeContractBridge:
        def get_conversion_tasks(self, model):
            assert model is MODEL
            return tasks

        def export_hf_weights(
            self,
            model,
            *,
            cpu,
            show_progress,
            conversion_tasks,
            merge_adapter_weights,
        ):
            assert model is MODEL
            assert cpu is True
            assert show_progress is False
            assert merge_adapter_weights is False
            assert len(conversion_tasks) == len(tasks)
            assert all(
                actual is expected
                for actual, expected in zip(conversion_tasks, tasks, strict=True)
            )
            yield from outputs

    schema = (
        CanonicalTensorSpec(expert_name, (2, 2, 2), torch.float32),
        CanonicalTensorSpec(k_name, (2, 4), torch.float32),
        CanonicalTensorSpec(norm_name, (2, 2), torch.float32),
        CanonicalTensorSpec(q_name, (4, 4), torch.float32),
        CanonicalTensorSpec(v_name, (2, 4), torch.float32),
    )
    buckets = []
    for_each_megatron_hf_bucket(
        MODEL,
        MegatronBridgeHfBucketConfig(
            bridge=NativeContractBridge(),
            canonical_schema=schema,
            bucket_bytes=64,
            spool_directory=tmp_path,
            model_context=lambda: nullcontext(),
        ),
        buckets.append,
    )

    flattened = [(name, tensor) for bucket in buckets for name, tensor in bucket]
    assert [name for name, _tensor in flattened] == [spec.name for spec in schema]
    assert all(tensor.is_contiguous() for _name, tensor in flattened)
    captured = dict(flattened)
    torch.testing.assert_close(captured[q_name], q)
    torch.testing.assert_close(captured[k_name], k)
    torch.testing.assert_close(captured[v_name], v)
    torch.testing.assert_close(captured[expert_name], expert)


def test_bridge_capture_fails_closed_after_draining_incomplete_coverage(tmp_path):
    bridge = GroupedBridge(
        [("model.layers.0.self_attn.k_proj.weight", torch.zeros((2, 2)))]
    )

    with pytest.raises(ValueError, match="complete canonical HF coverage"):
        for_each_megatron_hf_bucket(
            MODEL,
            MegatronBridgeHfBucketConfig(
                bridge=bridge,
                canonical_schema=_schema(),
                bucket_bytes=32,
                spool_directory=tmp_path,
            ),
            lambda _bucket: None,
        )

    assert bridge.drained


def test_bridge_uses_authoritative_weights_context_and_native_vocab_metadata(tmp_path):
    state = {"entered": False}
    original = torch.zeros(3, dtype=torch.float32)
    current = torch.tensor([4.0, 5.0, 99.0])
    task = ConversionTask("weight", object(), param_weight=original)

    class NativeTupleBridge:
        def get_conversion_tasks(self, _model):
            assert state["entered"]
            return [task]

        def export_hf_weights(self, _model, **kwargs):
            assert state["entered"]
            [converted] = kwargs["conversion_tasks"]
            yield "weight", converted.param_weight, "embedding.word_embeddings.weight"

    @contextmanager
    def model_context():
        state["entered"] = True
        try:
            yield
        finally:
            state["entered"] = False

    buckets = []
    for_each_megatron_hf_bucket(
        MODEL,
        MegatronBridgeHfBucketConfig(
            bridge=NativeTupleBridge(),
            canonical_schema=(CanonicalTensorSpec("weight", (2,), torch.float32),),
            bucket_bytes=16,
            spool_directory=tmp_path,
            model_context=model_context,
            weights_getter=lambda: {"vp_stages.0.weight": current},
            vocab_size=2,
        ),
        buckets.append,
    )

    assert state["entered"] is False
    [(name, tensor)] = [item for bucket in buckets for item in bucket]
    assert name == "weight"
    assert torch.equal(tensor, current[:2])


def test_single_expert_grouped_mapping_is_valid_when_coverage_is_complete(tmp_path):
    mapping = GroupedExpertMapping("expert.weight", "expert.weight")
    bridge = GroupedBridge(
        [("expert.weight", torch.tensor([[1.0, 2.0]], dtype=torch.float32))]
    )
    bridge.tasks = [ConversionTask("experts.weight0", mapping)]
    buckets = []

    for_each_megatron_hf_bucket(
        MODEL,
        MegatronBridgeHfBucketConfig(
            bridge=bridge,
            canonical_schema=(
                CanonicalTensorSpec("expert.weight", (1, 2), torch.float32),
            ),
            bucket_bytes=16,
            spool_directory=tmp_path,
        ),
        buckets.append,
    )

    assert [name for bucket in buckets for name, _tensor in bucket] == ["expert.weight"]


@pytest.mark.parametrize(
    ("failure_stage", "expected_gathers"),
    (("setup", 2), ("tasks", 3)),
)
def test_distributed_preflight_synchronizes_local_failure_before_export(
    monkeypatch,
    tmp_path,
    failure_stage,
    expected_gathers,
):
    bridge = GroupedBridge(
        [("expert.weight", torch.tensor([[1.0, 2.0]], dtype=torch.float32))]
    )
    bridge.tasks = [
        ConversionTask(
            "experts.weight0",
            GroupedExpertMapping("expert.weight", "expert.weight"),
        )
    ]
    model_context = None
    failure = f"local {failure_stage} failed"

    if failure_stage == "setup":

        @contextmanager
        def failing_context():
            raise RuntimeError(failure)
            yield

        model_context = failing_context
    else:

        def fail_conversion_tasks(_model):
            raise RuntimeError(failure)

        bridge.get_conversion_tasks = fail_conversion_tasks

    gathers = []

    def all_gather_object(outputs, contribution, group=None):
        assert group == "metadata"
        gathers.append(contribution)
        outputs[:] = [contribution, contribution]

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group=None: 2,
    )
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: None)

    with pytest.raises(ValueError, match=failure):
        for_each_megatron_hf_bucket(
            MODEL,
            MegatronBridgeHfBucketConfig(
                bridge=bridge,
                canonical_schema=(
                    CanonicalTensorSpec("expert.weight", (1, 2), torch.float32),
                ),
                bucket_bytes=16,
                spool_directory=tmp_path,
                rank=lambda: 0,
                metadata_group="metadata",
                model_context=model_context,
            ),
            lambda _bucket: None,
        )

    assert len(gathers) == expected_gathers
    assert isinstance(gathers[-1], str)
    assert failure in gathers[-1]
    assert bridge.export_calls == []


def test_distributed_context_teardown_failure_is_collectively_reported(
    monkeypatch,
    tmp_path,
):
    bridge = GroupedBridge(
        [("expert.weight", torch.tensor([[1.0, 2.0]], dtype=torch.float32))]
    )
    bridge.tasks = [
        ConversionTask(
            "experts.weight0",
            GroupedExpertMapping("expert.weight", "expert.weight"),
        )
    ]

    @contextmanager
    def failing_teardown():
        yield
        raise RuntimeError("local teardown failed")

    gathers = []

    def all_gather_object(outputs, contribution, group=None):
        assert group == "metadata"
        gathers.append(contribution)
        if len(gathers) == 4:
            outputs[:] = [contribution, None]
            return
        outputs[:] = [contribution, contribution]

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group=None: 2,
    )
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: None)

    with pytest.raises(ValueError, match="rank 0: local teardown failed"):
        for_each_megatron_hf_bucket(
            MODEL,
            MegatronBridgeHfBucketConfig(
                bridge=bridge,
                canonical_schema=(
                    CanonicalTensorSpec("expert.weight", (1, 2), torch.float32),
                ),
                bucket_bytes=16,
                spool_directory=tmp_path,
                rank=lambda: 0,
                metadata_group="metadata",
                model_context=failing_teardown,
            ),
            lambda _bucket: None,
        )

    assert len(gathers) == 4
    assert gathers[-1] == "local teardown failed"


def test_distributed_preflight_rejects_task_plan_disagreement_before_export(
    monkeypatch,
    tmp_path,
):
    bridge = GroupedBridge(
        [("expert.weight", torch.tensor([[1.0, 2.0]], dtype=torch.float32))]
    )
    bridge.tasks = [
        ConversionTask(
            "experts.weight0",
            GroupedExpertMapping("expert.weight", "expert.weight"),
        )
    ]
    gathers = []

    def all_gather_object(outputs, contribution, group=None):
        assert group == "metadata"
        gathers.append(contribution)
        if len(gathers) == 1:
            outputs[:] = [contribution, contribution]
            return
        if len(gathers) == 2:
            assert isinstance(contribution, tuple) and len(contribution) == 2
            local_error, _task_plan = contribution
            assert local_error is None
            outputs[:] = [contribution, (None, ("different-task-plan",))]
            return
        assert len(gathers) == 3
        outputs[:] = [contribution, contribution]

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group=None: 2,
    )
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: None)

    with pytest.raises(ValueError, match="task plan differs"):
        for_each_megatron_hf_bucket(
            MODEL,
            MegatronBridgeHfBucketConfig(
                bridge=bridge,
                canonical_schema=(
                    CanonicalTensorSpec("expert.weight", (1, 2), torch.float32),
                ),
                bucket_bytes=16,
                spool_directory=tmp_path,
                rank=lambda: 0,
                metadata_group="metadata",
            ),
            lambda _bucket: None,
        )

    assert len(gathers) == 3
    assert bridge.export_calls == []


def test_distributed_preflight_ignores_rank_local_bridge_task_ownership(
    monkeypatch,
    tmp_path,
):
    from types import SimpleNamespace

    mapping = GroupedExpertMapping("expert.weight", "expert.weight")
    bridge = GroupedBridge(
        [("expert.weight", torch.tensor([[1.0, 2.0]], dtype=torch.float32))]
    )
    bridge.tasks = [
        SimpleNamespace(
            param_name="local.experts.weight0",
            global_param_name="experts.weight0",
            mapping=mapping,
            pp_rank=0,
            vp_stage=0,
            megatron_module=object(),
            param_weight=torch.ones((1, 2), dtype=torch.float32),
        )
    ]
    remote_placeholder = SimpleNamespace(
        param_name="experts.weight0",
        global_param_name="experts.weight0",
        mapping=mapping,
        pp_rank=1,
        vp_stage=None,
        megatron_module=None,
        param_weight=None,
    )
    remote_task_plan = megatron_bridge._task_plan([remote_placeholder])
    gathers = []

    def all_gather_object(outputs, contribution, group=None):
        assert group == "metadata"
        gathers.append(contribution)
        if len(gathers) == 2:
            outputs[:] = [contribution, (None, remote_task_plan)]
            return
        outputs[:] = [contribution, contribution]

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group=None: 2,
    )
    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(torch.distributed, "barrier", lambda group=None: None)

    for_each_megatron_hf_bucket(
        MODEL,
        MegatronBridgeHfBucketConfig(
            bridge=bridge,
            canonical_schema=(
                CanonicalTensorSpec("expert.weight", (1, 2), torch.float32),
            ),
            bucket_bytes=16,
            spool_directory=tmp_path,
            rank=lambda: 0,
            metadata_group="metadata",
        ),
        lambda _bucket: None,
    )

    assert len(gathers) == 4
    assert len(bridge.export_calls) == 1
