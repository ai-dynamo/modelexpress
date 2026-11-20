# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from modelexpress import nixl_transfer, p2p_pb2
from modelexpress.adapter import EngineAdapter, StrategyRecoveryError
from modelexpress.load_strategy import execute_load_strategies
from modelexpress.load_strategy import base, rdma_strategy
from modelexpress.load_strategy.context import LoadContext
from modelexpress.load_strategy.default_strategy import DefaultStrategy


class _Agent:
    pass


@pytest.fixture
def draft_receive(monkeypatch):
    events = []
    managers = []
    failures = []
    failed_agents = []
    receive_failures = []
    initialize_failures = []
    registration_failures = []
    shutdown_failures = []

    def assert_released():
        assert all(agent() is None for agent in failed_agents)
        assert all(error.__traceback__ is None for error in failures)

    class Manager:
        def __init__(self, agent_name, device_id, listen_port, accelerator_backend, *, ucx_mode="auto"):
            self.agent_name = agent_name
            self.listen_port = listen_port
            self.tensors = {}
            self.tensor_descriptors = []
            self._agent = _Agent()
            managers.append(self)

        def fail(self, error):
            agent = self._agent
            failed_agents.append(weakref.ref(agent))
            failures.append(error)
            raise error

        def initialize(self):
            events.append("initialize")
            if initialize_failures:
                self.fail(initialize_failures.pop(0))

        def register_additional_tensors(self, tensors):
            events.append("register-main" if self is shared else "register-receiver")
            if registration_failures and self is not shared:
                self.fail(registration_failures.pop(0))
            self.tensors.update(tensors)

        def add_remote_agent(self, metadata):
            return "source-agent"

        def remove_remote_agent(self, agent_name):
            events.append("release-peer")

        def receive_from_source(self, **kwargs):
            events.append("receive")
            assert_released()
            if receive_failures:
                self.fail(receive_failures.pop(0))
            for tensor in kwargs["destination_tensors"].values():
                tensor.data.fill_(7)
            return 16, 1, 0.01

        def shutdown(self, *, strict=False):
            events.append("shutdown-main" if self is shared else "shutdown-receiver")
            if shutdown_failures:
                raise shutdown_failures.pop(0)
            self._agent = None
            self.tensors = {}

        def deregister_tensors(self, tensors):
            events.append("deregister")
            for name in tensors:
                self.tensors.pop(name, None)

    class Adapter(EngineAdapter):
        def discover_tensors(self, result):
            return {"weight": result.model.weight}

        def reinit_for_retry(self, result):
            events.append("reinit")
            assert_released()
            result.model = torch.nn.Linear(4, 1, bias=False)
            result.value = result.model
            return result

        def after_rdma_receive(self, result):
            events.append("postprocess")
            result.model.weight = torch.nn.Parameter(result.model.weight.detach().clone())
            return result

        def load_via_native(self, result):
            events.append("local-load")
            assert_released()
            result.model.weight.data.fill_(3)
            return result

    backend = SimpleNamespace(
        name="cuda",
        is_accel_tensor=lambda tensor: True,
        supports_rdma_p2p=lambda: True,
        synchronize=lambda: None,
    )
    shared = Manager("main", 0, 5555, backend)
    main_weight = torch.ones(4)
    shared.tensors = {"main.weight": main_weight}
    ctx = LoadContext(
        model_config=SimpleNamespace(model="model"),
        load_config=SimpleNamespace(),
        target_device=torch.device("cpu"),
        global_rank=0,
        worker_rank=0,
        local_rank=0,
        device_id=0,
        identity=p2p_pb2.SourceIdentity(model_name="model"),
        mx_client=MagicMock(),
        worker_id="target",
        mx_server_url="server:8001",
        adapter=Adapter(),
        accelerator_backend=backend,
        p2p_role="draft",
        shared_nixl_manager=shared,
    )
    strategy = rdma_strategy.RdmaStrategy()
    source = p2p_pb2.WorkerMetadata(
        accelerator="cuda",
        nixl_metadata=b"source",
        tensor_source=p2p_pb2.TensorSourceMetadata(tensors=[p2p_pb2.TensorDescriptor(
            name="mx_draft::weight", addr=4096, size=16, device_id=0,
            dtype="torch.float32",
        )]),
    )
    candidates = [
        p2p_pb2.SourceInstanceRef(mx_source_id="source", worker_id=f"worker-{index}")
        for index in range(2)
    ]
    monkeypatch.setattr(nixl_transfer, "NixlTransferManager", Manager)
    monkeypatch.setattr(base, "is_nixl_available", lambda: True)
    monkeypatch.setattr(strategy, "is_available", lambda ctx: True)
    monkeypatch.setattr(strategy, "_find_source_instances", lambda ctx: candidates)
    monkeypatch.setattr(strategy, "_fetch_worker_metadata", lambda *args: source)
    monkeypatch.setattr(strategy, "_accelerator_compatible", lambda *args: True)

    def publish(tensors, **kwargs):
        assert ctx.nixl_manager is shared
        assert shared.tensors == {"main.weight": main_weight, **tensors}
        assert all(manager._agent is None for manager in managers[1:])
        events.append("publish")
        return True

    monkeypatch.setattr(base, "extend_published_tensors", publish)
    return SimpleNamespace(
        ctx=ctx, strategy=strategy, events=events, managers=managers,
        candidates=candidates, receive_failures=receive_failures,
        initialize_failures=initialize_failures,
        registration_failures=registration_failures,
        shutdown_failures=shutdown_failures, assert_released=assert_released,
    )


def _load(case):
    model = torch.nn.Linear(4, 1, bias=False)
    loaded = execute_load_strategies(
        model, case.ctx, [case.strategy, DefaultStrategy()]
    )
    assert case.managers[0]._agent is not None
    assert "shutdown-main" not in case.events
    assert "deregister" not in case.events
    assert case.ctx.nixl_manager is case.managers[0]
    assert case.ctx.draft_published is True
    return model, loaded


def test_receive_uses_temporary_agent_then_publishes_final_tensors(draft_receive):
    case = draft_receive
    _, loaded = _load(case)

    assert torch.all(loaded.weight == 7)
    assert len(case.managers) == 2
    receiver = case.managers[1]
    assert receiver.listen_port == 0
    assert receiver.agent_name.startswith("mx-auto-worker0-")
    assert case.ctx.tensors["mx_draft::weight"] is loaded.weight
    assert case.events == [
        "initialize", "register-receiver", "receive", "release-peer",
        "postprocess", "shutdown-receiver", "register-main", "publish",
    ]


@pytest.mark.parametrize("is_reload", [False, True])
def test_receive_timeout_releases_agent_before_next_source(draft_receive, is_reload):
    case = draft_receive
    case.ctx.is_reload = is_reload
    case.receive_failures.append(TimeoutError("read timed out"))

    original, loaded = _load(case)

    assert (original is loaded) is is_reload
    assert ("reinit" in case.events) is not is_reload
    assert len(case.managers) == 3
    assert len({manager.agent_name for manager in case.managers}) == 3
    assert case.events.index("shutdown-receiver") < case.events.index("initialize", 1)
    assert torch.all(loaded.weight == 7)
    case.assert_released()


@pytest.mark.parametrize("is_reload", [False, True])
def test_receive_timeout_releases_agent_before_local_fallback(draft_receive, is_reload):
    case = draft_receive
    case.ctx.is_reload = is_reload
    case.candidates.pop()
    case.receive_failures.append(TimeoutError("read timed out"))

    original, loaded = _load(case)

    assert (original is loaded) is is_reload
    assert ("reinit" in case.events) is not is_reload
    assert case.events.index("shutdown-receiver") < case.events.index("local-load")
    assert len(case.managers) == 2
    assert torch.all(loaded.weight == 3)
    case.assert_released()


@pytest.mark.parametrize("failure_phase", ["initialize", "registration"])
@pytest.mark.parametrize("is_reload", [False, True])
def test_receiver_setup_failure_releases_owned_manager(draft_receive, failure_phase, is_reload):
    case = draft_receive
    case.ctx.is_reload = is_reload
    getattr(case, f"{failure_phase}_failures").append(RuntimeError("setup failed"))

    _load(case)

    assert len(case.managers) == 3
    assert case.events.count("shutdown-receiver") == 2
    assert ("reinit" in case.events) is not is_reload
    assert case.events.index("shutdown-receiver") < case.events.index("initialize", 1)
    case.assert_released()


@pytest.mark.parametrize("failure_phase", ["initialize", "registration", "receive", "finalize"])
def test_receiver_cleanup_failure_stops_loading(draft_receive, failure_phase):
    case = draft_receive
    if failure_phase != "finalize":
        getattr(case, f"{failure_phase}_failures").append(RuntimeError("attempt failed"))
    case.shutdown_failures.append(RuntimeError("shutdown failed"))

    with pytest.raises(StrategyRecoveryError, match="shutdown|shut down|clean up|close|release"):
        _load(case)

    assert "reinit" not in case.events
    assert "local-load" not in case.events
    assert "publish" not in case.events
    assert len(case.managers) == 2
    assert case.managers[0]._agent is not None


def test_no_source_uses_local_load_without_receiver(draft_receive):
    case = draft_receive
    case.candidates.clear()

    _, loaded = _load(case)

    assert len(case.managers) == 1
    assert case.events == ["local-load", "register-main", "publish"]
    assert torch.all(loaded.weight == 3)


def test_shared_registration_failure_does_not_publish_or_close_main(draft_receive, monkeypatch):
    case = draft_receive
    shared = case.managers[0]

    def fail_registration(tensors):
        raise RuntimeError("shared registration failed")

    monkeypatch.setattr(shared, "register_additional_tensors", fail_registration)
    model = torch.nn.Linear(4, 1, bias=False)

    loaded = execute_load_strategies(model, case.ctx, [case.strategy, DefaultStrategy()])

    assert loaded is model
    assert torch.all(loaded.weight == 7)
    assert case.ctx.nixl_manager is None
    assert case.ctx.draft_published is False
    assert list(shared.tensors) == ["main.weight"]
    assert shared._agent is not None
    assert case.managers[1]._agent is None
    assert "publish" not in case.events
    assert "local-load" not in case.events
