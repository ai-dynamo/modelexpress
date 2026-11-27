# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from modelexpress.adapter import StrategyRecoveryError
from modelexpress.engines.vllm import loader as loader_mod
from modelexpress.load_strategy import base
from .test_vllm_loader import _make_load_context, _make_loader, _model_config


class RecordingManager:
    def __init__(self, events):
        self.events = events
        self.tensor_descriptors = []
        self.tensors = {}
        self.closed = False

    def register_tensors(self, tensors, **kwargs):
        assert not self.closed
        self.tensors = dict(tensors)
        self.tensor_descriptors = list(tensors)
        self.events.append("register-main")

    def register_additional_tensors(self, tensors):
        assert not self.closed
        for name, tensor in tensors.items():
            assert name not in self.tensors or self.tensors[name] is tensor
        self.tensors.update(tensors)
        self.tensor_descriptors = list(self.tensors)
        self.events.append("register-draft")

    def shutdown(self):
        self.events.append("shutdown")
        self.closed = True
        self.tensors.clear()
        self.tensor_descriptors.clear()


@pytest.fixture
def draft_reload(monkeypatch):
    from modelexpress.metadata import publish

    monkeypatch.setenv("MX_P2P_COLD_START", "0")
    monkeypatch.setenv("MX_DISABLE_DRAFT_P2P", "0")
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "0")
    monkeypatch.setenv("MX_ARTIFACT_READY_URL", "")
    monkeypatch.setenv("MODEL_EXPRESS_URL", "test-server")
    for name in (
        "_loader_registry", "_nixl_managers", "_tensor_registry",
        "_draft_publication_gates",
    ):
        monkeypatch.setattr(loader_mod, name, {})
    monkeypatch.setattr(publish, "_worker_servers", {})
    monkeypatch.setattr(publish, "_heartbeat_threads", {})
    events = []
    publications = []
    main_weight = torch.tensor([17.0])
    owner = _make_loader()
    manager = RecordingManager(events)
    main_ctx = _make_load_context(
        tensors={"main.weight": main_weight}, nixl_manager=manager
    )
    owner._ctx = main_ctx
    manager.register_tensors(main_ctx.tensors)
    loader_mod._loader_registry[0] = owner
    loader_mod._nixl_managers[0] = manager
    loader_mod._tensor_registry[0] = main_ctx.tensors
    draft = _make_loader()
    model = torch.nn.Linear(1, 1, bias=False)
    config = SimpleNamespace(speculative_config=SimpleNamespace())
    model_config = _model_config(runner_type="draft")
    state = SimpleNamespace(
        events=events, publications=publications, owner=owner, draft=draft,
        model=model, config=config, model_config=model_config,
        main_weight=main_weight, manager=manager, fail=False,
    )

    def context(*args):
        ctx = _make_load_context(model_config=model_config)
        ctx.adapter.discover_tensors = lambda result: {
            **dict(result.model.named_parameters()), **dict(result.model.named_buffers()),
        }
        state.context = ctx
        return ctx

    def load(existing, ctx):
        events.append("write")
        if ctx.is_reload:
            assert existing is model
            assert not publications
            with torch.no_grad():
                existing.weight.fill_(3)
        if state.fail:
            raise RuntimeError("all strategies failed")
        base.register_tensors(existing, ctx)
        base.publish_metadata(ctx)
        return existing

    def unpublish(ctx, **kwargs):
        events.append("unpublish")
        publications.clear()

    def publish(ctx):
        events.append("publish")
        publications.append(dict(ctx.tensors))

    monkeypatch.setattr(loader_mod, "build_vllm_load_context", context)
    monkeypatch.setattr(loader_mod, "initialize_model", lambda **kwargs: model)
    monkeypatch.setattr(draft, "_run_load_strategy_chain", load)
    monkeypatch.setattr(loader_mod, "unpublish_metadata", unpublish)
    monkeypatch.setattr(loader_mod, "publish_metadata", publish)
    monkeypatch.setattr(loader_mod, "schedule_vllm_cache_artifact_publish", lambda ctx: None)
    monkeypatch.setattr(base, "is_nixl_available", lambda: True)
    monkeypatch.setattr(base, "undiscovered_parameters", lambda *args, **kwargs: [])
    monkeypatch.setattr(base, "_init_nixl_manager", lambda *args: RecordingManager(events))
    monkeypatch.setattr(loader_mod, "_init_nixl_manager", lambda *args: RecordingManager(events))
    monkeypatch.setattr(base, "extend_published_tensors", lambda *args, **kwargs: True)
    draft.load_model(config, model_config)
    publications.append(dict(main_ctx.tensors))
    events.clear()
    return state


def test_draft_load_records_own_context_without_owning_main(draft_reload):
    state = draft_reload
    assert state.draft._ctx is None
    assert state.draft._draft_ctx is state.context
    assert loader_mod.get_model_loader(0) is state.owner


def test_draft_reload_rejects_owner_generation_changed_before_claim(draft_reload, monkeypatch):
    state = draft_reload
    replacement = _make_load_context(nixl_manager=RecordingManager(state.events))

    def acquire(blocking=False):
        state.owner._ctx = replacement
        return True

    claim = SimpleNamespace(acquire=acquire, release=MagicMock())
    monkeypatch.setattr(state.owner, "_reload_lock", claim, raising=False)
    with pytest.raises(RuntimeError, match="publication changed"):
        state.draft.reload_model(state.config, state.model_config, state.model)
    assert not state.events
    assert not state.manager.closed
    claim.release.assert_called_once_with()


def test_draft_reload_rejects_central_publication_without_reader_barrier(
    draft_reload, monkeypatch,
):
    from modelexpress.metadata import publish

    state = draft_reload
    monkeypatch.setattr(publish, "_worker_servers", {})
    monkeypatch.setattr(publish, "_heartbeat_threads", {0: MagicMock()})
    with pytest.raises(RuntimeError, match="centrally published"):
        state.draft.reload_model(state.config, state.model_config, state.model)
    assert not state.events
    assert not state.manager.closed


def test_artifact_deregistration_failure_aborts_reload_before_agent_shutdown(
    draft_reload, monkeypatch,
):
    from modelexpress.engines.vllm import artifacts
    from modelexpress.metadata.artifact_transfer import NixlArtifactChunkManager
    from modelexpress.metadata.nixl_artifact_transport import NixlPublicationHandle

    state = draft_reload
    state.manager.register_dram_buffer = MagicMock()
    state.manager.deregister_memory = MagicMock(side_effect=RuntimeError("registration busy"))
    chunks = NixlArtifactChunkManager(state.manager, max_buffers=1)
    slot = chunks._slots[chunks._reserve_slot(16)]
    source = SimpleNamespace(heartbeat=MagicMock(), artifact_chunk_manager=chunks)
    monkeypatch.setattr(artifacts, "_scheduled_publishers", {})
    monkeypatch.setattr(artifacts, "_published_sources", {(0, 2): NixlPublicationHandle(source)})
    try:
        with pytest.raises(RuntimeError, match="registration busy"):
            state.draft.reload_model(state.config, state.model_config, state.model)
        assert "write" not in state.events
        assert not state.manager.closed
        assert slot.registered is not None
    finally:
        state.manager.deregister_memory.side_effect = None
        chunks.close()


def test_strict_unpublish_waits_for_metadata_publish_before_stopping_servicer(monkeypatch):
    from modelexpress.metadata import publish
    from modelexpress.metadata.publisher import PublisherThread

    entered = threading.Event()
    release = threading.Event()
    boundary = threading.Event()
    state = SimpleNamespace(live=True, source_id=None)
    errors = []

    class StopEvent(threading.Event):
        def set(self):
            super().set()
            boundary.set()

    def publish_fn():
        entered.set()
        assert release.wait(5)
        assert state.live
        state.source_id = "source"
        return state.source_id

    def stop_worker(**kwargs):
        state.live = False
        boundary.set()

    heartbeat = PublisherThread(
        MagicMock(), worker_id="worker", worker_rank=0,
        publish_fn=publish_fn, interval_secs=0,
    )
    heartbeat._started = True
    heartbeat._stop_event = StopEvent()
    worker = SimpleNamespace(stop=stop_worker)
    monkeypatch.setattr(publish, "_heartbeat_threads", {0: heartbeat})
    monkeypatch.setattr(publish, "_worker_servers", {0: worker})

    def stop():
        try:
            base.unpublish_metadata_for_worker(worker_rank=0, device_id=0, strict=True)
        except Exception as error:
            errors.append(error)

    heartbeat.start()
    stopper = threading.Thread(target=stop)
    try:
        assert entered.wait(2)
        stopper.start()
        assert boundary.wait(2)
        assert state.live
    finally:
        release.set()
        if stopper.ident is not None:
            stopper.join(timeout=5)
        heartbeat.stop()
    assert not stopper.is_alive()
    assert not errors
    assert state.source_id == "source"
    assert not state.live


def test_draft_reload_stops_current_owner_before_write_and_republishes(draft_reload, monkeypatch):
    from modelexpress.metadata import publish

    state = draft_reload
    monkeypatch.setattr(publish, "_central_manifests", {})
    monkeypatch.setattr(base, "extend_published_tensors", publish.extend_published_tensors)
    assert state.draft.reload_model(state.config, state.model_config, state.model) is state.model
    assert state.events == [
        "unpublish", "shutdown", "register-main", "write", "register-draft", "publish",
    ]
    assert state.main_weight.item() == 17
    assert state.owner.tensors["main.weight"] is state.main_weight
    assert state.publications == [{**state.owner.tensors}]
    assert set(state.owner.tensors) == {"main.weight", "mx_draft::weight"}
    assert state.manager.closed
    assert state.draft._draft_ctx.shared_nixl_manager is state.owner.nixl_manager
    assert not state.draft._draft_ctx.draft_published
    assert loader_mod._nixl_managers[0] is state.owner.nixl_manager


def test_repeated_draft_reload_replaces_changed_addresses_and_names(draft_reload):
    state = draft_reload
    first_weight = state.model.weight
    state.draft.reload_model(state.config, state.model_config, state.model)
    state.model.weight = torch.nn.Parameter(torch.ones(1, 1))
    state.model.register_buffer("extra", torch.ones(2))
    state.events.clear()
    state.draft.reload_model(state.config, state.model_config, state.model)
    assert state.owner.tensors["mx_draft::weight"] is state.model.weight
    assert all(tensor is not first_weight for tensor in state.owner.tensors.values())
    assert state.publications == [dict(state.owner.tensors)]
    assert len(state.owner.nixl_manager.tensor_descriptors) == len(state.owner.tensors)
    assert "mx_draft::extra" in state.owner.tensors
    del state.model.extra
    state.draft.reload_model(state.config, state.model_config, state.model)
    assert "mx_draft::extra" not in state.owner.tensors


def test_draft_reload_identity_is_a_snapshot_of_its_own_load(draft_reload):
    state = draft_reload
    state.context.identity.revision = "draft-checkpoint"
    state.owner._ctx.identity.revision = "main-checkpoint"
    state.draft.reload_model(state.config, state.model_config, state.model)
    assert state.context.loaded_identity.revision == "draft-checkpoint"
    state.context.identity.revision = "new-draft-checkpoint"
    assert state.context.loaded_identity.revision == "draft-checkpoint"


def test_draft_reload_all_failure_invalidates_state_and_keeps_source_stopped(draft_reload):
    state = draft_reload
    state.fail = True
    with pytest.raises(RuntimeError, match="all strategies failed"):
        state.draft.reload_model(state.config, state.model_config, state.model)
    assert state.draft._draft_ctx is None
    assert state.publications == []
    assert state.owner.tensors == {"main.weight": state.main_weight}
    assert state.main_weight.item() == 17


def test_draft_reload_stop_failure_never_writes(draft_reload, monkeypatch):
    state = draft_reload
    previous = state.context
    monkeypatch.setattr(
        loader_mod, "unpublish_metadata",
        MagicMock(side_effect=RuntimeError("stop failed")),
    )
    with pytest.raises(RuntimeError, match="stop failed"):
        state.draft.reload_model(state.config, state.model_config, state.model)
    assert "write" not in state.events
    assert not state.manager.closed
    assert state.draft._draft_ctx is previous


def test_draft_reload_uses_owner_manager_after_main_reload(draft_reload):
    state = draft_reload
    current_manager = RecordingManager(state.events)
    current_manager.register_tensors({"main.weight": state.main_weight})
    state.owner._ctx = _make_load_context(
        tensors={"main.weight": state.main_weight}, nixl_manager=current_manager
    )
    loader_mod._nixl_managers[0] = current_manager
    loader_mod._tensor_registry[0] = state.owner._ctx.tensors
    state.events.clear()
    state.draft.reload_model(state.config, state.model_config, state.model)
    assert current_manager.closed
    assert not state.manager.closed
    assert state.context.shared_nixl_manager is state.owner.nixl_manager


def test_strict_publisher_stop_retains_resources_if_thread_is_running():
    from modelexpress.metadata.publisher import PublisherThread

    cleanup = MagicMock()
    manager = MagicMock()
    publisher = PublisherThread(
        MagicMock(), mx_source_id="source", worker_id="worker", worker_rank=0,
        nixl_manager=manager, cleanup_fn=cleanup,
    )
    publisher._thread = MagicMock()
    publisher._thread.is_alive.return_value = True
    with pytest.raises(RuntimeError, match="publisher.*stop"):
        publisher.stop(strict=True)
    cleanup.assert_not_called()
    assert publisher._nixl_manager is manager
    publisher._thread.is_alive.return_value = False
    publisher.stop(strict=True)
    cleanup.assert_called_once()
    assert publisher._nixl_manager is None
    assert publisher._publish_fn is None


def test_publisher_force_cleanup_runs_once_with_force():
    from modelexpress.metadata.publisher import PublisherThread

    cleanup = MagicMock()
    publisher = PublisherThread(
        MagicMock(), mx_source_id="source", worker_id="worker", worker_rank=0,
        cleanup_fn=cleanup,
    )
    publisher.force_cleanup()
    publisher.force_cleanup()
    cleanup.assert_called_once_with(force=True)


def test_strict_worker_stop_waits_and_retains_state_on_timeout():
    from modelexpress.metadata.worker_server import WorkerGrpcServer

    worker = WorkerGrpcServer(tensor_protos=[], mx_source_id=None, port=0)
    server = MagicMock()
    worker._server = server
    worker._servicer = MagicMock()
    server.stop.return_value.wait.return_value = False
    with pytest.raises(RuntimeError, match="gRPC.*stop"):
        worker.stop(strict=True)
    assert worker._server is server
    assert worker._servicer is not None
    server.stop.return_value.wait.return_value = True
    worker.stop(strict=True)
    server.stop.return_value.wait.assert_called_with(timeout=10.0)
    assert worker._server is None
    assert worker._servicer is None


def test_strict_unpublish_retains_failed_publisher_for_retry(monkeypatch):
    from modelexpress.metadata import publish

    heartbeat = MagicMock()
    heartbeat.stop.side_effect = RuntimeError("publisher failed to stop")
    worker = MagicMock()
    monkeypatch.setattr(publish, "_heartbeat_threads", {0: heartbeat})
    monkeypatch.setattr(publish, "_worker_servers", {0: worker})
    with pytest.raises(RuntimeError, match="publisher failed"):
        base.unpublish_metadata(_make_load_context(), strict=True)
    worker.stop.assert_called_once_with(strict=True)
    assert publish._heartbeat_threads[0] is heartbeat


def test_stopping_successful_artifact_scheduler_releases_its_lease(tmp_path, monkeypatch):
    from modelexpress.metadata import artifact_lifecycle

    lease_path = tmp_path / "lease"
    lease = MagicMock()
    other_lease = MagicMock()
    monkeypatch.setattr(
        artifact_lifecycle, "_publish_leases",
        {lease_path: lease, tmp_path / "other": other_lease},
    )
    artifact_lifecycle.clear_publish_scheduled(
        SimpleNamespace(mx_source_id="published"), lease_path, force=True,
    )
    lease.close.assert_called_once()
    other_lease.close.assert_not_called()


def test_artifact_borrowers_stop_before_agent_but_keep_buffers_alive(draft_reload, monkeypatch):
    from modelexpress.engines.vllm import artifacts
    from modelexpress.metadata.nixl_artifact_transport import NixlPublicationHandle

    state = draft_reload
    scheduler = MagicMock()
    scheduler.stop.side_effect = lambda **kwargs: state.events.append("scheduler-stop")
    source = MagicMock()
    source.heartbeat.stop.side_effect = lambda **kwargs: state.events.append("artifact-stop")
    buffer = torch.ones(1)
    source.artifact_chunk_manager.close.side_effect = lambda **kwargs: (
        state.events.append("release-artifact") or [buffer]
    )
    handle = NixlPublicationHandle(source)
    monkeypatch.setattr(artifacts, "_scheduled_publishers", {(0, 2): scheduler})
    monkeypatch.setattr(artifacts, "_published_sources", {(0, 2): handle})
    state.draft.reload_model(state.config, state.model_config, state.model)
    assert state.events[:7] == [
        "scheduler-stop", "artifact-stop", "unpublish", "release-artifact",
        "shutdown", "release-artifact", "register-main",
    ]
    assert artifacts._scheduled_publishers == {}
    assert artifacts._published_sources == {}


@pytest.mark.parametrize("as_buffer", [False, True])
def test_draft_shared_main_storage_is_rejected_before_stop(draft_reload, as_buffer):
    state = draft_reload
    shared = state.main_weight.view(1, 1)
    if as_buffer:
        state.model.register_buffer("shared", shared)
    else:
        state.model.weight = torch.nn.Parameter(shared)
    with pytest.raises(RuntimeError, match="shares.*main"):
        state.draft.reload_model(state.config, state.model_config, state.model)
    assert state.events == []
    assert state.publications
    assert state.main_weight.item() == 17


def test_draft_reload_without_owner_does_not_create_main_manager(draft_reload):
    state = draft_reload
    loader_mod._loader_registry.clear()
    state.publications.clear()
    state.draft.reload_model(state.config, state.model_config, state.model)
    assert state.events == ["write"]
    assert not state.context.p2p_enabled
    assert state.context.loaded_identity is not None


def test_unpublished_draft_does_not_stop_or_create_main_manager(draft_reload):
    state = draft_reload
    state.owner._ctx.p2p_enabled = False
    state.owner._ctx.nixl_manager = None
    state.owner._ctx.tensors = {}
    state.publications.clear()
    state.draft.reload_model(state.config, state.model_config, state.model)
    assert state.events == ["write"]
    assert state.context.shared_nixl_manager is None


def test_draft_reload_after_sleep_uses_current_owner(draft_reload):
    state = draft_reload
    state.owner.on_sleep(1)
    state.events.clear()
    state.draft.reload_model(state.config, state.model_config, state.model)
    assert state.owner.nixl_manager is not state.manager
    assert state.owner.nixl_manager is state.context.shared_nixl_manager
    assert state.publications == [dict(state.owner.tensors)]


@pytest.mark.parametrize("failure_phase", ["initialize", "register"])
@pytest.mark.parametrize("draft_p2p", [False, True])
def test_draft_reload_manager_failure_allows_local_load(
    draft_reload, monkeypatch, failure_phase, draft_p2p,
):
    state = draft_reload
    monkeypatch.setenv("MX_DISABLE_DRAFT_P2P", "0" if draft_p2p else "1")
    manager = RecordingManager(state.events)
    manager.register_tensors = MagicMock(side_effect=RuntimeError("registration failed"))
    factory = MagicMock(
        side_effect=RuntimeError("initialize failed") if failure_phase == "initialize" else None,
        return_value=manager,
    )
    monkeypatch.setattr(loader_mod, "_init_nixl_manager", factory)

    assert state.draft.reload_model(state.config, state.model_config, state.model) is state.model
    assert state.model.weight.item() == 3
    assert state.main_weight.item() == 17
    assert state.manager.closed
    assert state.owner.nixl_manager is None
    assert manager.closed is (failure_phase == "register")
    assert state.publications == []


@pytest.mark.parametrize("failure_phase", ["load", "registration"])
def test_successful_reload_after_failure_restores_shared_publication(
    draft_reload, monkeypatch, failure_phase,
):
    state = draft_reload
    if failure_phase == "load":
        state.fail = True
        with pytest.raises(RuntimeError, match="all strategies failed"):
            state.draft.reload_model(state.config, state.model_config, state.model)
    else:
        manager = RecordingManager(state.events)
        manager.register_tensors = MagicMock(side_effect=RuntimeError("registration failed"))
        monkeypatch.setattr(loader_mod, "_init_nixl_manager", lambda *args: manager)
        state.draft.reload_model(state.config, state.model_config, state.model)
    assert state.publications == []
    assert state.owner.nixl_manager is None
    state.fail = False
    monkeypatch.setattr(loader_mod, "_init_nixl_manager", lambda *args: RecordingManager(state.events))

    assert state.draft.reload_model(state.config, state.model_config, state.model) is state.model
    assert state.model.weight.item() == 3
    assert state.main_weight.item() == 17
    assert state.owner.nixl_manager is state.context.shared_nixl_manager
    assert state.owner.nixl_manager is not None
    assert state.publications == [dict(state.owner.tensors)]
    assert set(state.owner.tensors) == {"main.weight", "mx_draft::weight"}
    assert (state.context.loaded_identity is None) is (failure_phase == "load")


def test_new_manager_cleanup_failure_aborts_before_loading(draft_reload, monkeypatch):
    state = draft_reload
    manager = RecordingManager(state.events)
    manager.register_tensors = MagicMock(side_effect=RuntimeError("registration failed"))
    manager.shutdown = MagicMock(side_effect=RuntimeError("cleanup failed"))
    monkeypatch.setattr(loader_mod, "_init_nixl_manager", lambda *args: manager)

    with pytest.raises(StrategyRecoveryError, match="cleanup failed"):
        state.draft.reload_model(state.config, state.model_config, state.model)

    assert "write" not in state.events
    assert not state.publications


def test_draft_reload_reentry_is_rejected_before_stop(draft_reload):
    state = draft_reload
    state.owner._reload_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="already"):
            state.draft.reload_model(state.config, state.model_config, state.model)
    finally:
        state.owner._reload_lock.release()
    assert state.events == []


def test_draft_reader_drain_failure_does_not_write_or_close_agent(draft_reload, monkeypatch):
    from modelexpress.metadata import publish

    state = draft_reload
    previous = state.draft._draft_ctx
    failure = TimeoutError("active readers")
    worker = MagicMock()
    worker.drain_tensor_reads.side_effect = failure
    monkeypatch.setattr(publish, "_worker_servers", {0: worker})
    gathered = []

    def gather(adapter, value):
        gathered.append(value)
        return value, (value[0], None)

    monkeypatch.setattr(type(state.context.adapter), "all_gather_state", gather)
    with pytest.raises(TimeoutError, match="active readers") as raised:
        state.draft.reload_model(state.config, state.model_config, state.model)

    assert raised.value is failure
    assert gathered == [("prepare:draft_reload", "active readers")]
    assert not state.events
    assert not state.manager.closed
    assert state.draft._draft_ctx is previous
    assert not state.owner._reload_lock.locked()
    assert not state.draft._reload_lock.locked()


@pytest.mark.parametrize("has_owner", [False, True])
def test_draft_reload_stops_before_loading_when_peer_cannot_prepare(
    draft_reload, monkeypatch, has_owner,
):
    state = draft_reload
    previous = state.draft._draft_ctx
    original_weight = state.model.weight.detach().clone()
    if not has_owner:
        loader_mod._loader_registry.clear()
        state.publications.clear()

    def gather(adapter, value):
        return value, (value[0], "active readers")

    monkeypatch.setattr(type(state.context.adapter), "all_gather_state", gather)
    with pytest.raises(StrategyRecoveryError, match="Rank 1.*draft reload.*active readers"):
        state.draft.reload_model(state.config, state.model_config, state.model)

    assert "write" not in state.events
    assert torch.equal(state.model.weight, original_weight)
    assert not state.publications
    assert state.draft._draft_ctx is previous
    assert not state.owner._reload_lock.locked()
    assert not state.draft._reload_lock.locked()
    if has_owner:
        assert state.manager.closed
        assert state.owner.nixl_manager is None
        assert 0 not in loader_mod._nixl_managers
        assert state.owner._draft_publication_pending


def test_local_only_draft_reload_restores_main_publication(draft_reload, monkeypatch):
    state = draft_reload
    monkeypatch.setenv("MX_DISABLE_DRAFT_P2P", "1")
    state.draft.reload_model(state.config, state.model_config, state.model)
    assert state.publications == [{"main.weight": state.main_weight}]
    assert state.draft._draft_ctx is state.context
    assert state.context.nixl_manager is None


def test_unknown_previous_draft_catalog_refuses_reload_before_mutation(draft_reload):
    state = draft_reload
    state.draft._draft_ctx = None
    with pytest.raises(RuntimeError, match="previous draft"):
        state.draft.reload_model(state.config, state.model_config, state.model)
    assert state.events == []


def test_all_failure_releases_unpublished_new_manager(draft_reload):
    state = draft_reload
    state.fail = True
    with pytest.raises(RuntimeError, match="all strategies failed"):
        state.draft.reload_model(state.config, state.model_config, state.model)
    assert state.owner.nixl_manager is None
    assert 0 not in loader_mod._nixl_managers


def test_disjoint_views_in_one_allocation_are_not_shared_weights():
    storage = torch.ones(8)
    model = torch.nn.Linear(2, 2, bias=False)
    model.weight = torch.nn.Parameter(storage[4:].reshape(2, 2))
    assert not loader_mod._shares_main_storage(model, [storage[:4]])
    assert loader_mod._shares_main_storage(model, [storage[:5]])


def test_main_reload_reentry_is_rejected_while_loading(monkeypatch):
    loader = _make_loader()
    config = SimpleNamespace(speculative_config=None)

    def load(*args):
        with pytest.raises(RuntimeError, match="already"):
            loader.reload_model(config, _model_config(), torch.nn.Linear(1, 1))
        return args[3]

    monkeypatch.setattr(loader, "_load_or_reload_model", load)
    loader.reload_model(config, _model_config(), torch.nn.Linear(1, 1))
    assert not loader._reload_lock.locked()


def test_strict_worker_stop_waits_for_cancelled_python_handler(monkeypatch):
    import threading
    import grpc
    from modelexpress import p2p_pb2, p2p_pb2_grpc
    from modelexpress.metadata.worker_server import WorkerGrpcServer, WorkerServiceServicer

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocked_handler(self, request, context):
        entered.set()
        release.wait(15)
        finished.set()
        return p2p_pb2.GetTensorManifestResponse()

    monkeypatch.setattr(WorkerServiceServicer, "GetTensorManifest", blocked_handler)
    worker = WorkerGrpcServer(tensor_protos=[], mx_source_id=None, port=0)
    channel = None
    try:
        port = worker.start()
        channel = grpc.insecure_channel(f"localhost:{port}")
        stub = p2p_pb2_grpc.WorkerServiceStub(channel)
        request = stub.GetTensorManifest.future(p2p_pb2.GetTensorManifestRequest())
        assert entered.wait(2)
        with pytest.raises(RuntimeError, match="handlers.*stop"):
            worker.stop(grace=0, strict=True)
        assert not finished.is_set()
        assert worker._servicer is not None
        with pytest.raises(grpc.RpcError):
            request.result(timeout=1)
    finally:
        release.set()
        if channel is not None:
            channel.close()
        worker.stop(grace=0, strict=True)


def test_manager_initialization_failure_is_cleaned_before_retry(monkeypatch):
    manager = MagicMock()
    manager.initialize.side_effect = RuntimeError("initialize failed")
    monkeypatch.setattr("modelexpress.nixl_transfer.NixlTransferManager", lambda **kwargs: manager)
    with pytest.raises(RuntimeError, match="initialize failed"):
        base._init_nixl_manager(0, 0, "auto")
    manager.shutdown.assert_called_once()


def test_cache_schedule_failure_does_not_invalidate_successful_weights(draft_reload, monkeypatch):
    state = draft_reload
    monkeypatch.setattr(
        loader_mod, "schedule_vllm_cache_artifact_publish",
        MagicMock(side_effect=OSError("lease failed")),
    )
    assert state.draft.reload_model(state.config, state.model_config, state.model) is state.model
    assert state.draft._draft_ctx is state.context
    assert state.publications == [dict(state.owner.tensors)]
    assert not state.owner.nixl_manager.closed


def test_draft_reload_with_unavailable_nixl_stays_local(draft_reload):
    state = draft_reload
    state.owner._ctx.nixl_manager = None
    state.owner._ctx.tensors.clear()
    state.context.nixl_manager = None
    state.context.shared_nixl_manager = None
    state.context.tensors.clear()
    state.publications.clear()
    state.draft.reload_model(state.config, state.model_config, state.model)
    assert state.events == ["write"]
    assert state.context.nixl_manager is None


def test_draft_reload_after_main_registration_failure_stays_local(draft_reload):
    state = draft_reload
    state.owner._ctx.nixl_manager = None
    state.owner._ctx.tensors = {"main.weight": state.main_weight}
    state.context.nixl_manager = None
    state.context.shared_nixl_manager = None
    state.context.tensors.clear()
    state.publications.clear()
    state.draft.reload_model(state.config, state.model_config, state.model)
    assert state.events == ["write"]
    assert state.context.nixl_manager is None


def test_worker_executor_cancels_queued_work_and_can_retry_drain():
    import threading
    from modelexpress.metadata.worker_server import _WorkerExecutor

    entered = threading.Event()
    release = threading.Event()
    executor = _WorkerExecutor(max_workers=1)

    def blocking():
        entered.set()
        release.wait(5)

    running = executor.submit(blocking)
    queued = executor.submit(lambda: pytest.fail("cancelled work ran"))
    try:
        assert entered.wait(1)
        with pytest.raises(RuntimeError, match="handlers.*stop"):
            executor.drain(timeout=0)
        assert queued.cancelled()
        with pytest.raises(RuntimeError, match="shutdown"):
            executor.submit(lambda: None)
    finally:
        release.set()
        running.result(timeout=2)
        executor.drain(timeout=1)


@pytest.mark.parametrize("revision,pinned", [("main", False), ("1" * 40, True)])
def test_draft_restore_without_prior_identity_requires_pinned_revision(draft_reload, revision, pinned):
    state = draft_reload
    state.draft._draft_ctx = None
    loader_mod._loader_registry.clear()
    state.publications.clear()
    state.model_config.revision = revision
    state.draft.reload_model(state.config, state.model_config, state.model, restore=True)
    assert (state.context.loaded_identity is not None) is pinned
    assert state.context.require_exact_tensor_catalog is pinned


def test_draft_registration_failure_republishes_main_only(draft_reload, monkeypatch):
    state = draft_reload
    manager = RecordingManager(state.events)
    manager.register_additional_tensors = MagicMock(side_effect=RuntimeError("draft registration failed"))
    monkeypatch.setattr(loader_mod, "_init_nixl_manager", lambda *args: manager)
    assert state.draft.reload_model(state.config, state.model_config, state.model) is state.model
    assert state.publications == [{"main.weight": state.main_weight}]
    assert state.draft._draft_ctx is state.context
    assert state.owner.tensors == {"main.weight": state.main_weight}
    assert state.owner.nixl_manager is manager
    assert not state.owner._draft_publication_pending


def test_failed_reload_reports_load_error_when_new_manager_shutdown_fails(
    draft_reload, monkeypatch,
):
    state = draft_reload
    manager = RecordingManager(state.events)
    manager.shutdown = MagicMock(side_effect=RuntimeError("cleanup failed"))
    monkeypatch.setattr(loader_mod, "_init_nixl_manager", lambda *args: manager)
    state.fail = True

    with pytest.raises(RuntimeError, match="all strategies failed"):
        state.draft.reload_model(state.config, state.model_config, state.model)

    assert manager.shutdown.called
    assert state.owner.nixl_manager is manager
    assert 0 not in loader_mod._nixl_managers
    assert state.publications == []


def test_artifact_shutdown_failure_retains_buffers_and_leases(draft_reload, monkeypatch):
    from modelexpress.engines.vllm import artifacts
    from modelexpress.metadata.nixl_artifact_transport import NixlPublicationHandle

    state = draft_reload
    publisher = MagicMock()
    source = MagicMock()
    handle = NixlPublicationHandle(source)
    monkeypatch.setattr(artifacts, "_scheduled_publishers", {(0, 2): publisher})
    monkeypatch.setattr(artifacts, "_published_sources", {(0, 2): handle})
    monkeypatch.setattr(state.manager, "shutdown", MagicMock(side_effect=RuntimeError("agent cleanup failed")))
    with pytest.raises(RuntimeError, match="agent cleanup failed"):
        state.draft.reload_model(state.config, state.model_config, state.model)
    assert "write" not in state.events
    assert artifacts._published_sources[(0, 2)] is handle
    assert artifacts._scheduled_publishers[(0, 2)] is publisher
    publisher.force_cleanup.assert_not_called()
    source.artifact_chunk_manager.close.assert_called_once_with(retain_buffers=True, strict=True)


def test_worker_executor_serializes_shutdown_with_submissions(monkeypatch):
    from concurrent import futures
    from modelexpress.metadata.worker_server import _WorkerExecutor

    executor = _WorkerExecutor(max_workers=1)
    shutdown = futures.ThreadPoolExecutor.shutdown
    states = []

    def checked_shutdown(instance, **kwargs):
        states.append(getattr(instance._pending_lock, "_is_owned", lambda: False)())
        return shutdown(instance, **kwargs)

    monkeypatch.setattr(futures.ThreadPoolExecutor, "shutdown", checked_shutdown)
    executor.drain(timeout=1)
    assert states == [True]
