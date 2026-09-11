# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch

import modelexpress_rl.inference.engines as engines_module
import modelexpress_rl.inference.runtime as runtime_module
from modelexpress import p2p_pb2
from modelexpress_rl import ObjectStorageType, WeightPayloadFormat, WeightSource
from modelexpress_rl.inference.adapter import GeneratorEngineContext
from modelexpress_rl.inference.plan import (
    EngineCapabilities,
    EngineInstaller,
    MethodCapabilities,
    PreparedEngineTensors,
    UpdateMethod,
)
from modelexpress_rl.inference.receiver import ObjectStorageGeneratorConfig
from modelexpress_rl.inference.runtime import (
    EngineRuntime,
    FullTensorEngineCapability,
    initialize_generator_runtime,
)


class _Installer(EngineInstaller):
    @property
    def capabilities(self):
        return EngineCapabilities(artifact_types=frozenset({PreparedEngineTensors}))

    def install(self, prepared):
        return prepared


class _Method(UpdateMethod):
    def __init__(self, sources):
        self._sources = frozenset(sources)
        self.closed = False

    @property
    def capabilities(self):
        return MethodCapabilities(
            payload_formats=frozenset(
                {WeightPayloadFormat.FULL_TENSOR, WeightPayloadFormat.XOR_DELTA}
            ),
            sources=self._sources,
            artifact_type=PreparedEngineTensors,
        )

    def prepare(self, *, version, source):
        raise AssertionError("composition test does not stage weights")

    def release(self, prepared):
        pass

    def close(self):
        self.closed = True


class _P2P:
    def __init__(self, *, server_url):
        self.server_url = server_url
        self.closed = False

    def close(self):
        self.closed = True


_DEFAULT_RUNTIME_TENSORS = object()
_DEFAULT_NIXL_MANAGER = object()


def _full_tensor_engine(
    *,
    runtime_tensors=_DEFAULT_RUNTIME_TENSORS,
    nixl_manager=_DEFAULT_NIXL_MANAGER,
):
    if runtime_tensors is _DEFAULT_RUNTIME_TENSORS:
        runtime_tensors = {"weight": torch.ones(1)}
    if nixl_manager is _DEFAULT_NIXL_MANAGER:
        nixl_manager = object()
    return EngineRuntime(
        model_name="test/model",
        installer=_Installer(),
        full_tensor=FullTensorEngineCapability(
            device_id=2,
            device="cuda:2",
            worker_rank=3,
            capture_layout=lambda manifest: manifest,
            runtime_tensors=runtime_tensors,
            source_worker_id=(
                "inference-worker-3" if runtime_tensors is not None else None
            ),
            unpublish_runtime_tensors=lambda: None,
            publish_runtime_tensors=lambda _version_id: None,
            build_identity=lambda version_id: p2p_pb2.SourceIdentity(
                model_name="test/model",
                revision=version_id,
            ),
            nixl_manager=nixl_manager,
        ),
    )


@pytest.mark.parametrize(
    ("configured_source_order", "expected_source_order"),
    [
        (
            None,
            (WeightSource.GENERATOR, WeightSource.OBJECT_STORAGE),
        ),
        (
            (WeightSource.OBJECT_STORAGE, WeightSource.GENERATOR),
            (WeightSource.OBJECT_STORAGE, WeightSource.GENERATOR),
        ),
    ],
)
def test_object_storage_runtime_preserves_source_order(
    monkeypatch,
    tmp_path,
    configured_source_order,
    expected_source_order,
):
    context = GeneratorEngineContext()
    monkeypatch.setattr(
        engines_module,
        "_create_engine_runtime",
        lambda received: _full_tensor_engine(),
    )
    p2p = _P2P(server_url="mx:8000")
    monkeypatch.setattr(runtime_module, "MxClient", lambda **_kwargs: p2p)
    monkeypatch.setattr(
        runtime_module, "_NixlStagedTransfer", lambda **_kwargs: object()
    )
    full_tensor = _Method({WeightSource.GENERATOR, WeightSource.TRAINER})
    canonical = _Method({WeightSource.OBJECT_STORAGE})
    monkeypatch.setattr(
        runtime_module,
        "RuntimeTensorNixlUpdateMethod",
        lambda **_kwargs: full_tensor,
    )
    monkeypatch.setattr(
        runtime_module,
        "CanonicalDeltaUpdateMethod",
        lambda **_kwargs: canonical,
    )
    storage = ObjectStorageGeneratorConfig(
        storage_type=ObjectStorageType.S3,
        initial_base_version_id="base-a",
        seed_checkpoint_path=Path(tmp_path / "launch"),
        refit_checkpoint_dir=Path(tmp_path / "cache"),
    )

    runtime = initialize_generator_runtime(
        engine_context=context,
        worker_id="generator-3",
        server_url="mx:8000",
        object_storage=storage,
        source_order=configured_source_order,
        max_transfer_attempts=3,
        rpc_timeout_seconds=30,
        service=lambda: object(),
        start_lease=lambda _version_id: object(),
    )

    assert runtime.methods == (canonical, full_tensor)
    assert runtime.session._planner.source_order == expected_source_order
    assert runtime.initial_version_id == "base-a"
    assert [
        resolver.kind for resolver in runtime.session._planner._resolvers
    ] == list(expected_source_order)
    generator_resolvers = [
        resolver
        for resolver in runtime.session._planner._resolvers
        if resolver.kind is WeightSource.GENERATOR
    ]
    if generator_resolvers:
        assert generator_resolvers[0]._worker_id == "inference-worker-3"
    runtime.close()
    runtime.close()
    assert canonical.closed
    assert full_tensor.closed
    assert p2p.closed


@pytest.mark.parametrize(
    "source_order",
    [None, (WeightSource.GENERATOR, WeightSource.OBJECT_STORAGE)],
)
@pytest.mark.parametrize(
    ("runtime_tensors", "nixl_manager"),
    [
        (None, object()),
        ({"weight": torch.ones(1)}, None),
    ],
)
def test_missing_inference_context_uses_object_storage_without_p2p(
    monkeypatch,
    tmp_path,
    source_order,
    runtime_tensors,
    nixl_manager,
):
    context = GeneratorEngineContext()
    monkeypatch.setattr(
        engines_module,
        "_create_engine_runtime",
        lambda received: _full_tensor_engine(
            runtime_tensors=runtime_tensors,
            nixl_manager=nixl_manager,
        ),
    )
    monkeypatch.setattr(
        runtime_module,
        "MxClient",
        lambda **_kwargs: pytest.fail("P2P client must not be created"),
    )
    canonical = _Method({WeightSource.OBJECT_STORAGE})
    monkeypatch.setattr(
        runtime_module,
        "CanonicalDeltaUpdateMethod",
        lambda **_kwargs: canonical,
    )
    storage = ObjectStorageGeneratorConfig(
        storage_type=ObjectStorageType.S3,
        initial_base_version_id="base-a",
        seed_checkpoint_path=Path(tmp_path / "launch"),
        refit_checkpoint_dir=Path(tmp_path / "cache"),
    )

    runtime = initialize_generator_runtime(
        engine_context=context,
        worker_id="generator-3",
        server_url="mx:8000",
        object_storage=storage,
        source_order=source_order,
        max_transfer_attempts=3,
        rpc_timeout_seconds=30,
        service=lambda: object(),
        start_lease=lambda _version_id: object(),
    )

    assert runtime.session._planner.source_order == (WeightSource.OBJECT_STORAGE,)
    assert runtime.methods == (canonical,)
    runtime.close()


def test_trainer_only_runtime_does_not_open_generator_listener(monkeypatch):
    context = GeneratorEngineContext()
    monkeypatch.setattr(
        engines_module, "_create_engine_runtime", lambda received: _full_tensor_engine()
    )
    transfer_kwargs = {}

    def create_transfer(**kwargs):
        transfer_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(runtime_module, "_NixlStagedTransfer", create_transfer)
    full_tensor = _Method({WeightSource.GENERATOR, WeightSource.TRAINER})
    method_kwargs = {}

    def create_method(**kwargs):
        method_kwargs.update(kwargs)
        return full_tensor

    monkeypatch.setattr(
        runtime_module,
        "LoadTimeTensorNixlUpdateMethod",
        create_method,
    )

    runtime = initialize_generator_runtime(
        engine_context=context,
        worker_id="generator-3",
        server_url="mx:8000",
        object_storage=None,
        source_order=(WeightSource.TRAINER,),
        max_transfer_attempts=3,
        rpc_timeout_seconds=30,
        service=lambda: object(),
        start_lease=lambda _version_id: object(),
    )

    assert transfer_kwargs["listen_port"] is None
    assert set(method_kwargs) == {"transfer", "capture_layout"}
    assert runtime.p2p_client is None
    runtime.close()


def test_object_storage_runtime_survives_p2p_initialization_failure(
    monkeypatch,
    tmp_path,
):
    context = GeneratorEngineContext()
    monkeypatch.setattr(
        engines_module,
        "_create_engine_runtime",
        lambda received: _full_tensor_engine(),
    )
    p2p = _P2P(server_url="mx:8000")
    monkeypatch.setattr(runtime_module, "MxClient", lambda **_kwargs: p2p)
    monkeypatch.setattr(
        runtime_module,
        "_NixlStagedTransfer",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("NIXL unavailable")),
    )
    canonical = _Method({WeightSource.OBJECT_STORAGE})
    monkeypatch.setattr(
        runtime_module,
        "CanonicalDeltaUpdateMethod",
        lambda **_kwargs: canonical,
    )
    storage = ObjectStorageGeneratorConfig(
        storage_type=ObjectStorageType.S3,
        initial_base_version_id="base-a",
        seed_checkpoint_path=Path(tmp_path / "launch"),
        refit_checkpoint_dir=Path(tmp_path / "cache"),
    )

    runtime = initialize_generator_runtime(
        engine_context=context,
        worker_id="generator-3",
        server_url="mx:8000",
        object_storage=storage,
        source_order=None,
        max_transfer_attempts=3,
        rpc_timeout_seconds=30,
        service=lambda: object(),
        start_lease=lambda _version_id: object(),
    )

    assert runtime.methods == (canonical,)
    assert runtime.session._planner.source_order == (WeightSource.OBJECT_STORAGE,)
    assert [
        resolver.kind for resolver in runtime.session._planner._resolvers
    ] == [WeightSource.OBJECT_STORAGE]
    assert p2p.closed
    runtime.close()
    assert canonical.closed


def test_generator_runtime_closes_resources_when_resolver_creation_fails(
    monkeypatch,
):
    context = GeneratorEngineContext()
    monkeypatch.setattr(
        engines_module, "_create_engine_runtime", lambda received: _full_tensor_engine()
    )
    p2p = _P2P(server_url="mx:8000")
    monkeypatch.setattr(runtime_module, "MxClient", lambda **_kwargs: p2p)
    monkeypatch.setattr(
        runtime_module, "_NixlStagedTransfer", lambda **_kwargs: object()
    )
    full_tensor = _Method({WeightSource.GENERATOR, WeightSource.TRAINER})
    monkeypatch.setattr(
        runtime_module,
        "RuntimeTensorNixlUpdateMethod",
        lambda **_kwargs: full_tensor,
    )
    monkeypatch.setattr(
        runtime_module,
        "GeneratorSourceResolver",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("resolver failed")),
    )

    with pytest.raises(RuntimeError, match="resolver failed"):
        initialize_generator_runtime(
            engine_context=context,
            worker_id="generator-3",
            server_url="mx:8000",
            object_storage=None,
            source_order=(WeightSource.GENERATOR,),
            max_transfer_attempts=3,
            rpc_timeout_seconds=30,
            service=lambda: object(),
            start_lease=lambda _version_id: object(),
        )

    assert full_tensor.closed
    assert p2p.closed
