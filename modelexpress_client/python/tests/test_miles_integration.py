# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from modelexpress import envs
from modelexpress.integrations.miles import MilesModelExpressPublisher


class _Manager:
    def __init__(self, **kwargs):
        self.agent_name = kwargs["agent_name"]
        self.kwargs = kwargs
        self.initialize_calls = 0
        self.register_calls = []
        self.shutdown_calls = 0

    def initialize(self):
        self.initialize_calls += 1

    def register_tensors(self, tensors):
        self.register_calls.append(dict(tensors))

    def shutdown(self):
        self.shutdown_calls += 1


class _Client:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _Rendezvous:
    def __init__(self, client, **kwargs):
        self.client = client
        self.kwargs = kwargs
        self.active_heartbeats = 0
        self.max_active_heartbeats = 0
        self.publish_calls = 0
        self.close_calls = 0

    def publish(self, _blob):
        # Current-main rendezvous replaces the old status heartbeat before
        # starting the next one.
        self.active_heartbeats = 1
        self.max_active_heartbeats = max(
            self.max_active_heartbeats, self.active_heartbeats
        )
        self.publish_calls += 1
        return "source-id"

    def close(self):
        self.active_heartbeats = 0
        self.close_calls += 1


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


def _publisher(monkeypatch):
    monkeypatch.setenv("MX_WORKER_HOST", "10.2.3.4")
    monkeypatch.setenv("MX_METADATA_PORT", "19000")
    manager = _Manager(agent_name="unused", device_id=0, listen_port=1)
    client = _Client()
    rendezvous_holder = {}
    published_calls = []

    def manager_factory(**kwargs):
        manager.__init__(**kwargs)
        return manager

    def rendezvous_factory(created_client, **kwargs):
        rendezvous = _Rendezvous(created_client, **kwargs)
        rendezvous_holder["value"] = rendezvous
        return rendezvous

    def publish_fn(
        *,
        manager,
        rendezvous,
        published,
        metadata_endpoint,
        publisher_step,
    ):
        published_calls.append(
            (manager, rendezvous, published, metadata_endpoint, publisher_step)
        )
        return rendezvous.publish(b"one-payload-publication")

    publisher = MilesModelExpressPublisher(
        manager_factory=manager_factory,
        client_factory=lambda **_kwargs: client,
        rendezvous_factory=rendezvous_factory,
        publish_fn=publish_fn,
        device_id_factory=lambda: 2,
    )
    publisher.configure(_registration())
    assert envs.MX_WORKER_HOST == "10.2.3.4"
    return publisher, manager, client, rendezvous_holder["value"], published_calls


def test_real_alias_addresses_stable_registration_and_no_heartbeat_leak(monkeypatch):
    publisher, manager, client, rendezvous, calls = _publisher(monkeypatch)
    tensors, specs = _tensors_and_specs()

    publisher.publish_and_execute(_request(specs, step=1))
    publisher.publish_and_execute(_request(specs, step=2))

    assert manager.initialize_calls == 1
    assert len(manager.register_calls) == 1
    assert manager.kwargs["listen_port"] == 19002
    assert tuple(manager.register_calls[0]) == tuple(spec.native_name for spec in specs)
    assert manager.register_calls[0]["padded_vocab"].shape == (6, 4)
    assert len(calls) == 2
    assert calls[0][3] == "10.2.3.4:19002"
    assert [call[4] for call in calls] == [1, 2]
    by_name = {item.name: item for item in calls[0][2]}
    qkv, gate_up, row, replicated, padded_vocab = tensors
    assert [shard.addr for shard in by_name["q"].shards] == [qkv.data_ptr()]
    assert [shard.addr for shard in by_name["k"].shards] == [qkv[8:].data_ptr()]
    assert [shard.addr for shard in by_name["v"].shards] == [qkv[10:].data_ptr()]
    assert by_name["gate"].shards[0].addr == gate_up.data_ptr()
    assert by_name["up"].shards[0].addr == gate_up[4:].data_ptr()
    assert by_name["down"].shards[0].addr == row.data_ptr()
    assert by_name["norm"].shards[0].addr == replicated.data_ptr()
    assert by_name["embed"].shards[0].addr == padded_vocab.data_ptr()
    assert by_name["embed"].shards[0].shape == (4, 4)
    assert by_name["embed"].shards[0].shard_offset == (6, 0)
    assert rendezvous.publish_calls == 2
    assert rendezvous.max_active_heartbeats == 1

    publisher.close()
    assert rendezvous.active_heartbeats == 0
    assert rendezvous.close_calls == manager.shutdown_calls == client.close_calls == 1


def test_changed_native_address_is_rejected_without_reregistration(monkeypatch):
    publisher, manager, _client, rendezvous, _calls = _publisher(monkeypatch)
    _tensors, specs = _tensors_and_specs()
    publisher.publish_and_execute(_request(specs))
    changed = list(specs)
    changed[2] = SimpleNamespace(**vars(specs[2]))
    changed[2].tensor = specs[2].tensor.clone()

    with pytest.raises(RuntimeError, match="addresses"):
        publisher.publish_and_execute(_request(changed, step=2))

    assert len(manager.register_calls) == 1
    assert rendezvous.publish_calls == 1
    publisher.close()


def test_changed_cohort_rebuilds_publisher_session(monkeypatch):
    monkeypatch.setenv("MX_WORKER_HOST", "10.2.3.4")
    monkeypatch.setenv("MX_METADATA_PORT", "19000")
    managers = []
    clients = []
    rendezvous = []

    def manager_factory(**kwargs):
        manager = _Manager(**kwargs)
        managers.append(manager)
        return manager

    def client_factory(**_kwargs):
        client = _Client()
        clients.append(client)
        return client

    def rendezvous_factory(client, **kwargs):
        item = _Rendezvous(client, **kwargs)
        rendezvous.append(item)
        return item

    def publish_fn(*, rendezvous, publisher_step, **_kwargs):
        assert publisher_step > 0
        return rendezvous.publish(b"payload")

    publisher = MilesModelExpressPublisher(
        manager_factory=manager_factory,
        client_factory=client_factory,
        rendezvous_factory=rendezvous_factory,
        publish_fn=publish_fn,
        device_id_factory=lambda: 2,
    )
    tensors, specs = _tensors_and_specs()
    publisher.configure(_registration())
    publisher.publish_and_execute(_request(specs, step=1))

    next_registration = SimpleNamespace(**vars(_registration()))
    next_registration.cohort_id = "rollout-session-b"
    publisher.configure(next_registration)
    next_request = SimpleNamespace(**vars(_request(specs, step=2)))
    next_request.cohort_id = "rollout-session-b"
    publisher.publish_and_execute(next_request)

    assert len(managers) == len(clients) == len(rendezvous) == 2
    assert managers[0].agent_name != managers[1].agent_name
    assert managers[0].shutdown_calls == 1
    assert clients[0].close_calls == 1
    assert rendezvous[0].close_calls == 1
    assert len(managers[1].register_calls) == 1
    publisher.close()


def test_configure_requires_shared_model_name_and_routable_host(monkeypatch):
    monkeypatch.setenv("MX_WORKER_HOST", "localhost")
    publisher = MilesModelExpressPublisher(device_id_factory=lambda: 0)
    missing_model = SimpleNamespace(**vars(_registration()))
    del missing_model.model_name

    with pytest.raises(ValueError, match="registration.model_name"):
        publisher.configure(missing_model)
    with pytest.raises(ValueError, match="receiver-routable"):
        publisher.configure(_registration())
