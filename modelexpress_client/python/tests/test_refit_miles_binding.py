# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
from modelexpress_rl.integrations import miles


def _alias(name, role, shape, axis=None, shard_range=None):
    return SimpleNamespace(
        hf_name=name,
        role=role,
        global_shape=shape,
        shard_axis=axis,
        local_shard_range=shard_range,
    )


def _spec(
    name,
    tensor,
    *,
    hf_names,
    role,
    global_shape,
    axis=None,
    shard_range=None,
    aliases,
    metadata=None,
):
    return SimpleNamespace(
        native_name=name,
        tensor=tensor,
        hf_names=hf_names,
        role=role,
        global_shape=global_shape,
        placement_kind="replicated" if axis is None else "contiguous_tp",
        shard_axis=axis,
        local_shard_range=shard_range,
        tensor_model_parallel=axis is not None,
        partition_dim=-1 if axis is None else axis,
        partition_stride=1,
        parallel_mode=None,
        source_rank=3,
        aliases=aliases,
        conversion_metadata=metadata or {},
    )


def _tensors_and_specs():
    qkv = torch.arange(48, dtype=torch.bfloat16).reshape(12, 4)
    gate_up = torch.arange(32, dtype=torch.bfloat16).reshape(8, 4)
    row = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    replicated = torch.arange(4, dtype=torch.bfloat16)
    padded_vocab = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    specs = [
        _spec(
            "qkv",
            qkv,
            hf_names=("q", "k", "v"),
            role="qkv",
            global_shape=(24, 4),
            axis=0,
            shard_range=(12, 24),
            aliases=(
                _alias("q", "q", (16, 4), 0, (8, 16)),
                _alias("k", "k", (4, 4), 0, (2, 4)),
                _alias("v", "v", (4, 4), 0, (2, 4)),
            ),
            metadata={
                "head_dim": 2,
                "local_query_groups": 1,
                "query_heads_per_group": 4,
            },
        ),
        _spec(
            "gate_up",
            gate_up,
            hf_names=("gate", "up"),
            role="gate_up",
            global_shape=(16, 4),
            axis=0,
            shard_range=(8, 16),
            aliases=(
                _alias("gate", "gate", (8, 4), 0, (4, 8)),
                _alias("up", "up", (8, 4), 0, (4, 8)),
            ),
        ),
        _spec(
            "row",
            row,
            hf_names=("down",),
            role="down_proj",
            global_shape=(4, 8),
            axis=1,
            shard_range=(4, 8),
            aliases=(_alias("down", "down_proj", (4, 8), 1, (4, 8)),),
        ),
        _spec(
            "replicated",
            replicated,
            hf_names=("norm",),
            role="final_norm",
            global_shape=(4,),
            aliases=(_alias("norm", "final_norm", (4,)),),
        ),
        _spec(
            "padded_vocab",
            padded_vocab,
            hf_names=("embed",),
            role="embedding",
            global_shape=(12, 4),
            axis=0,
            shard_range=(6, 12),
            aliases=(_alias("embed", "embedding", (10, 4), 0, (6, 10)),),
            metadata={
                "layout": "padded_vocab",
                "padded_vocab_size": 12,
                "vocab_size": 10,
            },
        ),
    ]
    return (qkv, gate_up, row, replicated, padded_vocab), specs


def _registration():
    return SimpleNamespace(
        model_name="Qwen/Qwen3",
        worker_id="trainer-3",
        cohort_id="rollout-session-a",
        source_geometry={"global_rank": 3, "tp_rank": 1, "tp_size": 2},
        logical_groups=("model",),
        rollout_workers=(),
    )


def _request(specs, step=1):
    return SimpleNamespace(
        version=str(step),
        training_step=step,
        logical_group="model",
        cohort_id="rollout-session-a",
        worker_id="trainer-3",
        source_geometry={"global_rank": 3, "tp_rank": 1, "tp_size": 2},
        tensors=specs,
        atomic_units=(),
    )


def test_megatron_alias_geometry_matches_miles():
    tensors, specs = _tensors_and_specs()
    result = miles._build_specs(_request(specs))
    published = miles.build_hf_aliases(result, agent_name="test")
    assert {item.name for item in published} == {
        "q",
        "k",
        "v",
        "gate",
        "up",
        "down",
        "norm",
        "embed",
    }
    by_name = {item.name: item for item in published}
    assert by_name["embed"].shards[0].shape == (4, 4)
    assert by_name["embed"].shards[0].addr == specs[-1].tensor.data_ptr()
    for tensor, spec in zip(result, specs, strict=True):
        assert tensor.tensor.data_ptr() == spec.tensor.data_ptr()


def test_publisher_reuses_registration_across_versions_and_cohorts(monkeypatch):
    tensors, specs = _tensors_and_specs()
    client = Mock(source_slot_id="slot")
    client.bind_tensors.return_value = "slot"
    initialize = Mock(return_value=client)
    monkeypatch.setattr(miles.ModelExpressTrainerClient, "initialize", initialize)
    monkeypatch.setattr(miles, "_routable_worker_host", lambda: "10.0.0.1")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    synchronize = Mock()
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    publisher = miles.MilesModelExpressPublisher()
    registration = _registration()
    publisher.configure(registration)
    assert publisher.prepare(specs) == "slot"
    publisher.publish_and_execute(_request(specs))
    registration.cohort_id = "replacement"
    publisher.configure(registration)
    request = _request(specs, step=2)
    request.cohort_id = "replacement"
    publisher.publish_and_execute(request)
    assert initialize.call_count == 1
    assert client.bind_tensors.call_count == 1
    assert client.publish_version.call_count == 2
    assert synchronize.call_count == 2
    specs[0].tensor = specs[0].tensor.clone()
    with pytest.raises(RuntimeError, match="storage changed"):
        publisher.prepare(specs)
    publisher.close()


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "0.0.0.0", "::1"])
def test_rejects_unroutable_trainer_host(monkeypatch, host):
    monkeypatch.setenv("MX_WORKER_HOST", host)
    with pytest.raises(ValueError):
        miles._routable_worker_host()


@pytest.mark.parametrize("enabled", [False, True])
def test_megatron_digest_policy_does_not_invalidate_unverified_plans(
    monkeypatch, enabled
):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1" if enabled else "0")
    _, specs = _tensors_and_specs()
    inputs = miles._build_specs(_request(specs))
    first = miles.build_hf_aliases(inputs, agent_name="stable")
    for item in inputs:
        item.tensor.add_(1)
    second = miles.build_hf_aliases(inputs, agent_name="stable")
    assert [t.shards[0].addr for t in first] == [t.shards[0].addr for t in second]
    if enabled:
        assert all(
            a.shards[0].digest != b.shards[0].digest
            for a, b in zip(first, second, strict=True)
        )
    else:
        assert first == second
        assert all(shard.digest is None for item in first for shard in item.shards)
