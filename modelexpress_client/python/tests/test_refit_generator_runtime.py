# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import modelexpress_rl.inference.engines as engines_module
import modelexpress_rl.inference.runtime as runtime_module
import pytest
import torch
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
from modelexpress_rl.inference.version_chain import resolve_replay_chain


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
        self.requires_full_root = False

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
    canonical.requires_full_root = True
    resolve_chain = Mock(return_value=())
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
        resolve_replay_chain=resolve_chain,
    )

    for needs_bootstrap in (True, False):
        canonical.requires_full_root = needs_bootstrap
        resolve_chain.reset_mock()
        runtime.session._resolve_replay_chain(SimpleNamespace(version_id="target"))
        resolve_chain.assert_called_once_with("target", True)

    assert runtime.methods == (canonical, full_tensor)
    assert runtime.session._planner.source_order == expected_source_order
    assert runtime.initial_version_id == "base-a"
    assert [resolver.kind for resolver in runtime.session._planner._resolvers] == list(
        expected_source_order
    )
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


@pytest.mark.parametrize("requires_full_root", [False, True])
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
    requires_full_root,
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
    canonical.requires_full_root = requires_full_root
    resolve_chain = Mock(return_value=())
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
        resolve_replay_chain=resolve_chain,
    )

    assert runtime.session._planner.source_order == (WeightSource.OBJECT_STORAGE,)
    assert runtime.methods == (canonical,)
    runtime.session._resolve_replay_chain(SimpleNamespace(version_id="target"))
    resolve_chain.assert_called_once_with("target", requires_full_root)

    canonical.requires_full_root = False
    versions = {
        f"v{index}": SimpleNamespace(
            version_id=f"v{index}",
            base_version_id=f"v{index - 1}" if index else None,
            payload_format=(
                WeightPayloadFormat.XOR_DELTA
                if index else WeightPayloadFormat.FULL_HF_CHECKPOINT
            ),
            object_storage=object(),
            layout_signature="",
        )
        for index in range(65)
    }
    fetched = Mock(side_effect=versions.__getitem__)
    resolve_chain.side_effect = lambda target, from_full_root: resolve_replay_chain(
        target_version_id=target,
        fetch_ready_version=fetched,
        max_chain_length=64,
        stop_before_version_id=None if from_full_root else "v63",
    )
    chain = runtime.session._resolve_replay_chain(versions["v64"])
    assert chain == (versions["v64"],)
    fetched.assert_called_once_with("v64")
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
    assert [resolver.kind for resolver in runtime.session._planner._resolvers] == [
        WeightSource.OBJECT_STORAGE
    ]
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


@pytest.mark.parametrize("blocked_first", [False, True])
@pytest.mark.parametrize(
    "failure", ["drain_failed", "close_failed", "source_failed", "in_progress"]
)
def test_runtime_close_keeps_all_resources_when_streaming_cleanup_is_unproven(
    blocked_first, failure
):
    from types import SimpleNamespace

    from modelexpress_rl.inference.methods import LoadTimeTensorNixlUpdateMethod
    from modelexpress_rl.inference.plan import PreparedStreamingTensors
    from modelexpress_rl.inference.runtime import GeneratorRuntime

    arena = torch.ones(2)
    closed = []
    transfer = SimpleNamespace(arena=arena, close=lambda: closed.append("transfer"))
    method = LoadTimeTensorNixlUpdateMethod(transfer=transfer, capture_layout=None)
    prepared = PreparedStreamingTensors(lambda: iter(()), frozenset({"weight"}), {})
    prepared.ownership.iterator = iter(({"weight": arena},))
    if failure != "in_progress":
        setattr(prepared.ownership, failure, True)
    method._active_streamed = prepared
    other = _Method({WeightSource.GENERATOR})
    p2p = _P2P(server_url="mx:8000")
    runtime = GeneratorRuntime(
        engine=_full_tensor_engine(),
        methods=(method, other) if blocked_first else (other, method),
        session=object(),
        p2p_client=p2p,
        initial_version_id=None,
    )

    for _ in range(2):
        with pytest.raises(RuntimeError, match="reset the process"):
            runtime.close()
        assert not runtime._closed
        assert not p2p.closed
        assert not other.closed
        assert closed == []
        assert method._active_streamed is prepared
        assert prepared.ownership.iterator is not None
        assert transfer.arena is arena


@pytest.mark.parametrize(
    "enabled,supported,expected",
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_streaming_requires_opt_in_and_engine_capability(
    monkeypatch, tmp_path, enabled, supported, expected
):
    from dataclasses import replace
    from unittest.mock import Mock
    from modelexpress_rl.inference.plan import PreparedCheckpointArtifact

    monkeypatch.setenv("MX_REFIT_FULL_STREAMING", str(enabled))
    engine = _full_tensor_engine()
    installer = Mock(
        capabilities=EngineCapabilities(
            artifact_types=frozenset({PreparedCheckpointArtifact}),
            streamed_checkpoints=supported,
        )
    )
    monkeypatch.setattr(
        engines_module,
        "_create_engine_runtime",
        lambda context: replace(engine, installer=installer),
    )
    canonical = _Method({WeightSource.OBJECT_STORAGE})
    constructor = Mock(return_value=canonical)
    monkeypatch.setattr(runtime_module, "CanonicalDeltaUpdateMethod", constructor)
    runtime = initialize_generator_runtime(
        engine_context=GeneratorEngineContext(),
        worker_id="generator-0",
        server_url="mx:8000",
        object_storage=ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType.S3,
            initial_base_version_id="v0",
            seed_checkpoint_path=tmp_path / "launch",
            refit_checkpoint_dir=tmp_path / "cache",
        ),
        source_order=(WeightSource.OBJECT_STORAGE,),
        max_transfer_attempts=1,
        rpc_timeout_seconds=30,
        service=lambda: object(),
        start_lease=lambda version: object(),
    )
    assert constructor.call_args.kwargs["stream_full_checkpoints"] is expected
    runtime.close()
