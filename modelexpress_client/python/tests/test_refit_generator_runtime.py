# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

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
    GeneratorRuntime,
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


def _full_tensor_engine():
    return EngineRuntime(
        model_name="test/model",
        installer=_Installer(),
        full_tensor=FullTensorEngineCapability(
            device_id=2,
            device="cuda:2",
            worker_rank=3,
            accelerator="cuda",
            capture_layout=lambda manifest: manifest,
            parameter_layout=lambda: {},
            build_identity=lambda version_id: p2p_pb2.SourceIdentity(
                model_name="test/model",
                revision=version_id,
            ),
        ),
    )


def test_object_storage_runtime_skips_full_tensor_transport(
    monkeypatch,
    tmp_path,
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
    canonical = _Method({WeightSource.OBJECT_STORAGE})
    monkeypatch.setattr(
        runtime_module,
        "FullTensorNixlUpdateMethod",
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

    runtime = GeneratorRuntime.initialize(
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
    assert runtime.initial_version_id == "base-a"
    assert [
        resolver.kind for resolver in runtime.session._planner._resolvers
    ] == [WeightSource.OBJECT_STORAGE]
    runtime.close()
    runtime.close()
    assert canonical.closed
    assert not full_tensor.closed
    assert not p2p.closed


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
        "FullTensorNixlUpdateMethod",
        lambda **_kwargs: full_tensor,
    )
    monkeypatch.setattr(
        runtime_module,
        "GeneratorSourceResolver",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("resolver failed")),
    )

    with pytest.raises(RuntimeError, match="resolver failed"):
        GeneratorRuntime.initialize(
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
