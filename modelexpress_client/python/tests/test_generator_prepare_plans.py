# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock
from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from modelexpress.adapter import StrategyRecoveryError
from modelexpress_rl import TrainerMeshRef
from modelexpress_rl.control import TrainerMesh, WeightVersion, WeightVersionState
from modelexpress_rl.object_storage import ObjectStorageSource, ObjectStorageType
from modelexpress.refit.timing import RefitTimingRecorder, refit_span, use_refit_timing
from modelexpress_rl.inference.methods.load_time_tensor import LoadTimeTensorNixlUpdateMethod
from modelexpress_rl.inference.plan import (
    MethodCapabilities,
    EngineCapabilities,
    ObjectStorageUpdateSource,
    PreparedEngineTensors,
    ResolvedTrainerSource,
    StreamingSettings,
    UpdateMethod,
    WeightSource,
)
from modelexpress_rl.train import WeightPayloadFormat
from modelexpress_rl.inference.session import WeightUpdateSession
from modelexpress_rl.inference.plan import WeightUpdatePlanner
from modelexpress_rl.inference.source.trainer import _binding_sources
from modelexpress_rl.inference.source.trainer import TrainerPreparationInputs
from modelexpress_rl.inference.source import TrainerSourceResolver
from modelexpress_rl.inference.source.generator import GeneratorSourceResolver
from modelexpress_rl.inference.source.object_storage import ObjectStorageSourceResolver
from modelexpress_rl.inference import nixl_staged_transfer
from modelexpress_rl.train.manifest import bound_tensor_manifest
from tests.test_refit_bounded_descriptors import harness, _check_values


class _PrepareMethod(UpdateMethod):
    def __init__(self, sources, *, error=None):
        self._capabilities = MethodCapabilities(
            payload_formats=frozenset({WeightPayloadFormat.FULL_TENSOR}),
            sources=frozenset(sources),
            artifact_type=PreparedEngineTensors,
        )
        self.error = error
        self.prepare_calls = []
        self._prepared_mesh = None

    @property
    def capabilities(self):
        return self._capabilities

    @property
    def prepared_mesh(self):
        return self._prepared_mesh

    def prepare(self, preparation=None):
        self.prepare_calls.append(preparation)
        if self.error is not None:
            raise self.error

    def stage(self, *, version, source):
        raise AssertionError("preparation test method cannot stage")

    def release(self, prepared):
        pass


class _BareUpdateMethod(UpdateMethod):
    @property
    def capabilities(self):
        return MethodCapabilities(
            payload_formats=frozenset({WeightPayloadFormat.FULL_TENSOR}),
            sources=frozenset({WeightSource.OBJECT_STORAGE}),
            artifact_type=PreparedEngineTensors,
        )

    def stage(self, *, version, source):
        raise AssertionError("preparation test method cannot stage")

    def release(self, prepared):
        pass


def _resolver(kind, spec=TrainerSourceResolver):
    resolver = Mock(spec=spec)
    resolver.kind = kind
    resolver.needs_mesh = kind is WeightSource.TRAINER
    resolver.prepare_inputs.return_value = None
    return resolver


def _session_with_methods(methods, resolvers=()):
    installer = Mock()
    installer.capabilities = EngineCapabilities(
        artifact_types=frozenset({PreparedEngineTensors})
    )
    return WeightUpdateSession(
        planner=WeightUpdatePlanner(
            resolvers=tuple(resolvers),
            methods=tuple(methods),
            installer=installer,
            max_transfer_attempts=3,
        ),
        start_lease=Mock(),
    )


def _mesh(generation=1):
    return TrainerMesh("mesh", "test", generation, {})


def _prepare(session, generation=1, validate_mesh=None):
    return session.prepare(
        _mesh(generation), validate_mesh=validate_mesh or (lambda: None)
    )


def test_update_method_default_mesh_hook_is_noop():
    method = _BareUpdateMethod()
    assert method.prepared_mesh is None
    assert _session_with_methods([method]).prepare(None) == {}


def test_prepare_dispatches_trainer_resolver_inputs():
    mesh = _mesh(3)
    sources = {"binding": object()}
    resolver = _resolver(WeightSource.TRAINER)
    resolver.prepare_inputs.side_effect = (
        lambda *, mesh, validate_mesh: TrainerPreparationInputs(
            mesh, sources, validate_mesh
        )
    )
    method = _PrepareMethod({WeightSource.TRAINER})
    session = _session_with_methods([method], [resolver])

    assert method.prepared_mesh is None
    validate = Mock()
    assert session.prepare(mesh, validate_mesh=validate) == {}
    preparation = method.prepare_calls[0]
    assert isinstance(preparation, TrainerPreparationInputs)
    assert preparation.mesh is mesh
    assert preparation.sources is sources
    resolver.prepare_inputs.assert_called_once_with(mesh=mesh, validate_mesh=validate)
    assert preparation.validate_mesh is validate
    validate_mesh = preparation.validate_mesh
    validate_mesh()
    validate.assert_called_once_with()


def test_prepare_uses_resolver_mesh_requirement():
    mesh_method = _PrepareMethod({WeightSource.TRAINER})
    object_method = _PrepareMethod({WeightSource.OBJECT_STORAGE})
    trainer = _resolver(WeightSource.TRAINER)
    storage = _resolver(WeightSource.OBJECT_STORAGE, ObjectStorageSourceResolver)
    storage.prepare_inputs.return_value = None
    session = _session_with_methods([mesh_method, object_method], [trainer, storage])

    assert trainer.needs_mesh is True
    assert storage.needs_mesh is False
    assert session.prepare(None) == {}
    assert mesh_method.prepare_calls == []
    assert object_method.prepare_calls == [None]
    trainer.prepare_inputs.assert_not_called()
    storage.prepare_inputs.assert_called_once_with()


def test_planner_prepare_targets_one_source_and_propagates_its_error():
    inputs = object()
    storage_resolver = _resolver(
        WeightSource.OBJECT_STORAGE, ObjectStorageSourceResolver
    )
    storage_resolver.prepare_inputs.return_value = inputs
    peer_resolver = _resolver(WeightSource.GENERATOR, GeneratorSourceResolver)
    storage_method = _PrepareMethod({WeightSource.OBJECT_STORAGE})
    peer_method = _PrepareMethod({WeightSource.GENERATOR})
    planner = _session_with_methods(
        [storage_method, peer_method], [storage_resolver, peer_resolver]
    )._planner

    planner.prepare(None, source_kind=WeightSource.OBJECT_STORAGE)
    assert storage_method.prepare_calls == [inputs]
    assert peer_method.prepare_calls == []
    storage_resolver.prepare_inputs.assert_called_once_with()
    peer_resolver.prepare_inputs.assert_not_called()

    error = RuntimeError("storage preparation failed")
    storage_resolver.prepare_inputs.side_effect = error
    with pytest.raises(RuntimeError, match="storage preparation failed"):
        planner.prepare(None, source_kind=WeightSource.OBJECT_STORAGE)
    assert peer_method.prepare_calls == []
    peer_resolver.prepare_inputs.assert_not_called()


def test_prepare_isolates_errors_by_method_capability_sources():
    error = RuntimeError("peer preparation failed")
    failed = _PrepareMethod(
        {WeightSource.GENERATOR, WeightSource.OBJECT_STORAGE}, error=error
    )
    healthy = _PrepareMethod({WeightSource.TRAINER})
    session = _session_with_methods(
        [failed, healthy],
        [
            _resolver(WeightSource.GENERATOR, GeneratorSourceResolver),
            _resolver(WeightSource.TRAINER),
            _resolver(WeightSource.OBJECT_STORAGE, ObjectStorageSourceResolver),
        ],
    )

    assert session.prepare(_mesh(), validate_mesh=Mock()) == {
        WeightSource.GENERATOR: error,
        WeightSource.OBJECT_STORAGE: error,
    }
    assert len(failed.prepare_calls) == 2
    assert len(healthy.prepare_calls) == 1


def test_prepare_visits_sources_in_order_after_recoverable_failure():
    events = []
    object_method = _PrepareMethod({WeightSource.OBJECT_STORAGE})
    trainer_method = _PrepareMethod({WeightSource.TRAINER})
    peer_method = _PrepareMethod({WeightSource.GENERATOR})
    for name, method in (
        ("object", object_method), ("trainer_method", trainer_method),
        ("peer", peer_method),
    ):
        original = method.prepare

        def record(preparation=None, *, name=name, original=original):
            events.append(name)
            return original(preparation)

        method.prepare = record

    trainer = _resolver(WeightSource.TRAINER)
    error = RuntimeError("trainer manifest unavailable")

    def fail_trainer(*, mesh, validate_mesh):
        events.append("trainer_resolver")
        raise error

    trainer.prepare_inputs.side_effect = fail_trainer
    session = _session_with_methods(
        [object_method, trainer_method, peer_method],
        [
            _resolver(WeightSource.OBJECT_STORAGE, ObjectStorageSourceResolver),
            trainer,
            _resolver(WeightSource.GENERATOR, GeneratorSourceResolver),
        ],
    )

    assert session.prepare(_mesh(), validate_mesh=Mock()) == {WeightSource.TRAINER: error}
    assert events == ["object", "trainer_resolver", "peer"]


def test_prepare_skips_cached_mesh_before_resolver_io():
    mesh = _mesh(8)
    resolver = _resolver(WeightSource.TRAINER)
    method = _PrepareMethod({WeightSource.TRAINER})
    method._prepared_mesh = mesh.ref
    session = _session_with_methods([method], [resolver])

    assert session.prepare(mesh, validate_mesh=Mock()) == {}
    resolver.prepare_inputs.assert_not_called()
    assert method.prepare_calls == []


def test_planner_applies_shared_payload_base_and_installer_compatibility():
    method = _PrepareMethod({WeightSource.OBJECT_STORAGE})
    method._capabilities = replace(
        method.capabilities,
        payload_formats=frozenset({WeightPayloadFormat.XOR_DELTA}),
        requires_base_version=True,
    )
    resolver = _resolver(WeightSource.OBJECT_STORAGE, ObjectStorageSourceResolver)
    resolver.supports.return_value = True
    resolver.payload_format.return_value = WeightPayloadFormat.XOR_DELTA
    source = ObjectStorageUpdateSource(
        storage=ObjectStorageSource(ObjectStorageType.S3, "s3://bucket/weights"),
        payload_format=WeightPayloadFormat.XOR_DELTA,
    )
    resolver.candidates.side_effect = lambda version: iter((source,))
    installer = Mock()
    installer.capabilities = EngineCapabilities(
        artifact_types=frozenset({PreparedEngineTensors})
    )
    planner = WeightUpdatePlanner(
        resolvers=(resolver,), methods=(method,), installer=installer,
        max_transfer_attempts=3,
    )
    version = WeightVersion(
        version_id="v",
        model_name="model",
        payload_format=WeightPayloadFormat.XOR_DELTA,
        layout_signature="layout",
        state=WeightVersionState.READY,
        created_at_unix_ms=0,
        trainer_mesh_generation=0,
        object_storage=source.storage,
    )

    assert list(planner.plans(version)) == []
    with pytest.raises(RuntimeError, match="no legal source"):
        planner.validate(version)

    version = replace(version, base_version_id="base")
    assert len(list(planner.plans(version))) == planner._max_transfer_attempts
    planner.validate(version)
    planner.prepare(None, source_kind=WeightSource.OBJECT_STORAGE)
    assert method.prepare_calls == [None]

    installer.capabilities = EngineCapabilities(artifact_types=frozenset())
    planner.prepare(None, source_kind=WeightSource.OBJECT_STORAGE)
    assert method.prepare_calls == [None]
    assert list(planner.plans(version)) == []
    with pytest.raises(RuntimeError, match="no legal source"):
        planner.validate(version)


def test_prepare_keeps_fatal_errors_fatal_for_generic_methods():
    error = StrategyRecoveryError("unsafe preparation")
    method = _PrepareMethod({WeightSource.GENERATOR}, error=error)
    later = _PrepareMethod({WeightSource.OBJECT_STORAGE})
    session = _session_with_methods(
        [method, later],
        [
            _resolver(WeightSource.GENERATOR, GeneratorSourceResolver),
            _resolver(WeightSource.OBJECT_STORAGE, ObjectStorageSourceResolver),
        ],
    )

    with pytest.raises(StrategyRecoveryError, match="unsafe preparation"):
        session.prepare(None)
    assert later.prepare_calls == []


def _sources(harness):
    return _binding_sources("slot", bound_tensor_manifest([
        {"name": name, "dtype": str(tensor.dtype), "elsize": tensor.element_size(),
         "full_shape": tuple(tensor.shape),
         "shards": [{"shard_offset": (0, 0), "shape": tuple(tensor.shape)}]}
        for name, tensor in harness.sources.items()
    ]))


def _session(harness):
    resolver = _resolver(WeightSource.TRAINER)

    def prepare_mesh(*, mesh, validate_mesh):
        with refit_span("source_preparation", accumulate_metadata=True):
            return TrainerPreparationInputs(mesh, _sources(harness), validate_mesh)

    resolver.prepare_inputs.side_effect = prepare_mesh
    capture = Mock(return_value=(harness.capture, harness.layout))
    harness.transfer._capture_layout = capture
    method = LoadTimeTensorNixlUpdateMethod(transfer=harness.transfer)
    session = _session_with_methods([method], [resolver])
    return session, resolver, capture


@pytest.mark.parametrize("harness", [None, StreamingSettings(1024, "cpu")], indirect=True)
def test_prepare_compiles_without_transfer_and_stage_reuses_plan(harness, monkeypatch):
    harness.transfer._device = torch.device("cpu")
    monkeypatch.setattr(nixl_staged_transfer, "classic_cuda_alloc", nullcontext)
    session, resolver, capture = _session(harness)
    before = list(harness.events)
    mesh = TrainerMeshRef("mesh", 1)
    assert _prepare(session) == {}
    assert harness.events == before
    assert not harness.transports
    assert harness.transfer._active is None
    assert session.prepare(None) == {}
    assert _prepare(session) == {}
    assert capture.call_count == resolver.prepare_inputs.call_count == 1
    assert harness.transfer.prepared_mesh == mesh
    monkeypatch.setattr(harness.transfer, "_compile_full_copy", Mock(side_effect=AssertionError("recompiled")))
    monkeypatch.setattr(harness.transfer, "_compile_streaming", Mock(side_effect=AssertionError("recompiled")))
    prepared = harness.prepare()
    assert not harness.captures
    assert prepared.metrics["plan_cache_hits"] == 1
    if harness.streaming is not None:
        _, installed = harness.collect(prepared)
        _check_values(harness, installed)
    else:
        # Exercise physical reads without requiring CUDA stage instrumentation.
        prepared.transport.post_reads(list(prepared.descriptors))
        exact = harness.transfer._recv_buffers["a.weight"]
        assert torch.equal(exact[2:18], harness.sources["exact"].reshape(-1))


def test_failed_recheck_keeps_previous_plan_and_can_retry(harness):
    session, _, capture = _session(harness)
    assert _prepare(session, 1) == {}
    old = harness.transfer._mesh_plan
    errors = _prepare(
        session, 2, validate_mesh=Mock(side_effect=RuntimeError("generation changed"))
    )
    assert str(errors[WeightSource.TRAINER]) == "generation changed"
    assert harness.transfer._mesh_plan is old
    assert _prepare(session, 2) == {}
    assert harness.transfer.prepared_mesh == TrainerMeshRef("mesh", 2)
    assert capture.call_count == 3


@pytest.mark.parametrize("failure_site", ["resolver", "prepare"])
def test_prepare_returns_recoverable_errors_by_source(harness, monkeypatch, failure_site):
    session, resolver, _ = _session(harness)
    error = RuntimeError("mesh unavailable")
    if failure_site == "resolver":
        resolver.prepare_inputs.side_effect = error
    else:
        monkeypatch.setattr(session._planner._methods[0], "prepare", Mock(side_effect=error))

    assert _prepare(session) == {WeightSource.TRAINER: error}


def test_prepare_keeps_fatal_recovery_errors_raised(harness, monkeypatch):
    session, _, _ = _session(harness)
    error = StrategyRecoveryError("unsafe capture")
    monkeypatch.setattr(session._planner._methods[0], "prepare", Mock(side_effect=error))
    with pytest.raises(StrategyRecoveryError, match="unsafe capture"):
        _prepare(session)


def test_prepare_skips_mesh_when_trainer_resolver_is_not_configured(harness, monkeypatch):
    session, resolver, _ = _session(harness)
    method = session._planner._methods[0]
    prepare = Mock()
    compile_full = Mock()
    compile_streaming = Mock()
    monkeypatch.setattr(method, "prepare", prepare)
    monkeypatch.setattr(harness.transfer, "_compile_full_copy", compile_full)
    monkeypatch.setattr(harness.transfer, "_compile_streaming", compile_streaming)
    session._planner._resolvers = ()

    assert _prepare(session) == {}
    resolver.prepare_inputs.assert_not_called()
    prepare.assert_not_called()
    compile_full.assert_not_called()
    compile_streaming.assert_not_called()


def test_prepared_coverage_rejects_changed_physical_schema(harness):
    session, _, _ = _session(harness)
    _prepare(session)
    sources = harness.transfer._mesh_plan.sources
    sources["exact"] = replace(sources["exact"], dtype=torch.bfloat16, elsize=2)
    with pytest.raises(RuntimeError, match="coverage differs"):
        harness.prepare()
    assert not harness.transports


def test_automatic_preparation_records_cold_planning(harness):
    session, _, _ = _session(harness)
    recorder = RefitTimingRecorder(backend="test", version="v")
    with use_refit_timing(recorder):
        _prepare(session)
        harness.prepare()
        source = ResolvedTrainerSource(snapshot=harness.transfer.cached_trainer_source())
        method = session._planner._methods[0]
        prepared = method.stage(version=None, source=source)
    assert recorder.cold is True
    assert recorder.has_measurements("transfer_planning")
    assert recorder.has_measurements("source_preparation")
    method.release(prepared)
    warm = RefitTimingRecorder(backend="test", version="next")
    with use_refit_timing(warm):
        method.stage(version=None, source=source)
    assert warm.cold is False
