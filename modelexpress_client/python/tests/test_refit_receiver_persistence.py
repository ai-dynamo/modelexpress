# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from modelexpress.refit_receiver import _DTYPE_MAP

from modelexpress.refit_receiver import MxRefitReceiver, SourceRef
from modelexpress.refit_timing import RefitTimingRecorder, use_refit_timing


class _Agent:
    def __init__(self):
        self.active = set()
        self.add_calls = []
        self.remove_calls = []

    def add_remote_agent(self, metadata):
        name = bytes(metadata).decode()
        if name in self.active:
            raise RuntimeError(f"duplicate remote agent: {name}")
        self.active.add(name)
        self.add_calls.append(name)
        return name

    def remove_remote_agent(self, name):
        self.active.discard(name)
        self.remove_calls.append(name)


class _Nixl:
    def __init__(self):
        self._agent = _Agent()
        self._tensors = {}
        self.register_calls = 0
        self.rebind_calls = 0
        self.receive_calls = []

    def register_tensors(self, tensors):
        self.register_calls += 1
        self._tensors = dict(tensors)

    def rebind_tensors(self, tensors):
        self.rebind_calls += 1
        self._tensors = dict(tensors)

    def receive_from_source(
        self,
        *,
        source_metadata,
        source_tensors,
        timeout_seconds,
        remote_agent_name,
    ):
        names = [tensor.name for tensor in source_tensors]
        self.receive_calls.append((names, remote_agent_name))
        return sum(tensor.size for tensor in source_tensors), len(names), 0.01


class _Client:
    def __init__(self, workers):
        self.workers = workers

    def get_metadata(self, *, mx_source_id, worker_id):
        worker = self.workers.get(worker_id)
        return SimpleNamespace(found=worker is not None, worker=worker)


def _tensor(name, size=16, dtype="float32"):
    return SimpleNamespace(
        name=name,
        addr=100,
        size=size,
        device_id=0,
        dtype=dtype,
    )


def test_scratch_dtype_map_supports_fp8_checkpoint_descriptors():
    assert _DTYPE_MAP["torch.float8_e4m3fn"] is torch.float8_e4m3fn
    assert _DTYPE_MAP["float8_e4m3fn"] is torch.float8_e4m3fn


def _worker(agent_name, tensors=None):
    return SimpleNamespace(
        agent_name="",
        nixl_metadata=agent_name.encode(),
        tensors=tensors or [_tensor("a"), _tensor("b")],
    )


def _source(worker_id, source_id="source"):
    return SourceRef(
        mx_source_id=source_id,
        worker_id=worker_id,
        model_name="model",
        worker_rank=0,
        training_step=1,
    )


@pytest.fixture
def receiver(monkeypatch):
    real_empty = torch.empty

    def _cpu_empty(*args, **kwargs):
        kwargs.pop("device", None)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", _cpu_empty)
    obj = MxRefitReceiver("receiver", device_id=0)
    obj._initialized = True
    obj._nixl = _Nixl()
    return obj


def test_scratch_registration_persists_while_wire_subset_changes(receiver):
    worker = _worker("trainer-r0")
    receiver._client = _Client({"worker-0": worker})

    first = list(
        receiver.receive_weights_scratch(
            _source("worker-0", "v1"),
            include_names={"a"},
        )
    )
    second = list(
        receiver.receive_weights_scratch(
            _source("worker-0", "v2"),
            include_names={"b"},
        )
    )

    assert [name for name, _ in first] == ["a"]
    assert [name for name, _ in second] == ["b"]
    assert receiver._nixl.register_calls == 1
    assert receiver._nixl.rebind_calls == 1
    assert receiver._nixl._agent.add_calls == ["trainer-r0"]
    assert receiver._nixl.receive_calls == [
        (["a"], "trainer-r0"),
        (["b"], "trainer-r0"),
    ]


def test_persistence_cache_status_is_structured(receiver):
    worker = _worker("trainer-r0")
    receiver._client = _Client({"worker-0": worker})
    timing = RefitTimingRecorder(backend="test", version=2)

    with use_refit_timing(timing):
        list(receiver.receive_weights_scratch(_source("worker-0", "v1")))
        list(receiver.receive_weights_scratch(_source("worker-0", "v2")))

    setup = timing.as_dict()["stages"]["setup_registration"]
    assert setup["status"] == "mixed"
    assert setup["statuses"] == ["cache_miss", "cache_hit"]
    assert setup["count"] == 4
    assert setup["metadata"] == {
        "scratch_buffer_cache": "hit",
        "remote_agent_cache": "hit",
    }


def test_prune_allows_restarted_worker_to_reuse_agent_name(receiver):
    receiver._client = _Client(
        {
            "old-worker": _worker("trainer-r0"),
            "new-worker": _worker("trainer-r0"),
        }
    )

    list(receiver.receive_weights_scratch(_source("old-worker")))
    receiver.prune_scratch_remote_agents({"new-worker"})
    list(receiver.receive_weights_scratch(_source("new-worker")))

    assert receiver._nixl._agent.add_calls == ["trainer-r0", "trainer-r0"]
    assert receiver._nixl._agent.remove_calls == ["trainer-r0"]
    assert set(receiver._scratch_remote_agents) == {"new-worker"}


def test_metadata_change_reloads_agent(receiver):
    receiver._client = _Client({"worker-0": _worker("trainer-r0-v1")})
    list(receiver.receive_weights_scratch(_source("worker-0")))

    receiver._client.workers["worker-0"] = _worker("trainer-r0-v2")
    list(receiver.receive_weights_scratch(_source("worker-0", "v2")))

    assert receiver._nixl._agent.add_calls == ["trainer-r0-v1", "trainer-r0-v2"]
    assert receiver._nixl._agent.remove_calls == ["trainer-r0-v1"]


def test_layout_change_fails_loudly(receiver):
    receiver._client = _Client({"worker-0": _worker("trainer-r0")})
    list(receiver.receive_weights_scratch(_source("worker-0")))

    receiver._client.workers["worker-0"] = _worker(
        "trainer-r0",
        tensors=[_tensor("a", size=32), _tensor("b")],
    )
    with pytest.raises(RuntimeError, match="layout changed"):
        list(receiver.receive_weights_scratch(_source("worker-0", "v2")))
