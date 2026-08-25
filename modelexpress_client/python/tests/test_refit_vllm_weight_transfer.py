# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import dataclass, fields
from types import ModuleType, SimpleNamespace
from typing import Generic, TypeVar

import pytest
import torch


TInitInfo = TypeVar("TInitInfo")
TUpdateInfo = TypeVar("TUpdateInfo")


@dataclass
class _WeightTransferInitInfo:
    pass


@dataclass
class _WeightTransferUpdateInfo:
    pass


class _WeightTransferEngine(Generic[TInitInfo, TUpdateInfo]):
    def __init__(self, config, vllm_config, device, model) -> None:
        self.config = config
        self.vllm_config = vllm_config
        self.parallel_config = vllm_config.parallel_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.model = model

    def parse_init_info(self, values):
        try:
            return self.init_info_cls(**values)
        except TypeError as error:
            raise ValueError(str(error)) from error

    def parse_update_info(self, values):
        try:
            return self.update_info_cls(**values)
        except TypeError as error:
            raise ValueError(str(error)) from error

    def update_weights(self, values) -> None:
        self.receive_weights(self.parse_update_info(values))


@dataclass
class _WeightTransferConfig:
    backend: str


base = ModuleType("vllm.distributed.weight_transfer.base")
base.WeightTransferEngine = _WeightTransferEngine
base.WeightTransferInitInfo = _WeightTransferInitInfo
base.WeightTransferUpdateInfo = _WeightTransferUpdateInfo
config = ModuleType("vllm.config.weight_transfer")
config.WeightTransferConfig = _WeightTransferConfig
sys.modules[base.__name__] = base
sys.modules[config.__name__] = config

from modelexpress_rl.inference.engines.vllm import weight_transfer  # noqa: E402
from modelexpress_rl.inference.engines.vllm.weight_transfer import (  # noqa: E402
    MxRefitInitInfo,
    MxRefitUpdateInfo,
    MxRefitWeightTransferEngine,
)
from modelexpress_rl.train import WeightPayloadFormat  # noqa: E402


INIT_INFO = {
    "model_name": "test/model",
    "initial_base_version_id": "base-a",
    "launch_checkpoint": "/models/launch",
    "preparation_cache_dir": "/cache/modelexpress",
    "server_url": "mx:8001",
    "s3_endpoint_url": "http://minio:9000",
    "s3_region_name": "us-west-2",
    "registration_ttl_seconds": 90,
    "lease_ttl_seconds": 60,
    "max_transfer_attempts": 4,
    "rpc_timeout_seconds": 12.5,
}


@pytest.fixture
def engine():
    return MxRefitWeightTransferEngine(
        _WeightTransferConfig(backend="modelexpress"),
        SimpleNamespace(
            parallel_config=SimpleNamespace(rank=2),
            model_config=SimpleNamespace(model="test/model"),
        ),
        torch.device("cpu"),
        object(),
    )


def test_init_info_maps_to_current_generator_api(monkeypatch, engine):
    captured = []
    client = SimpleNamespace(close=lambda: None)

    class GeneratorClient:
        @classmethod
        def initialize(cls, generator_config):
            captured.append(generator_config)
            return client

    monkeypatch.setattr(weight_transfer, "ModelExpressGeneratorClient", GeneratorClient)

    engine.init_transfer_engine(engine.parse_init_info(INIT_INFO))

    assert engine._client is client
    assert len(captured) == 1
    generator_config = captured[0]
    assert generator_config.engine_context.model is engine.model
    assert generator_config.engine_context.vllm_config is engine.vllm_config
    assert generator_config.model_name == "test/model"
    assert generator_config.payload_format is WeightPayloadFormat.XOR_DELTA
    assert generator_config.server_url == "mx:8001"
    assert generator_config.registration_ttl_seconds == 90
    assert generator_config.lease_ttl_seconds == 60
    assert generator_config.max_transfer_attempts == 4
    assert generator_config.rpc_timeout_seconds == 12.5
    assert generator_config.s3.initial_base_version_id == "base-a"
    assert generator_config.s3.launch_checkpoint == "/models/launch"
    assert generator_config.s3.preparation_cache_dir == "/cache/modelexpress"
    assert generator_config.s3.endpoint_url == "http://minio:9000"
    assert generator_config.s3.region_name == "us-west-2"


def test_init_info_requires_launch_checkpoint(engine):
    values = dict(INIT_INFO)
    del values["launch_checkpoint"]

    with pytest.raises(ValueError, match="launch_checkpoint"):
        engine.parse_init_info(values)


def test_update_stages_applies_and_releases_exact_version(engine):
    events = []

    class Staged:
        def release(self):
            events.append(("release",))

    staged = Staged()

    class Client:
        def stage_weight(self, *, version):
            events.append(("stage", version.version_id))
            return staged

        def apply_weight(self, value):
            events.append(("apply", value))

    engine._client = Client()

    engine.start_weight_update()
    engine.update_weights({"version_id": "opaque-a"})
    engine.finish_weight_update()

    assert events == [
        ("stage", "opaque-a"),
        ("apply", staged),
        ("release",),
    ]


def test_update_releases_staged_weight_when_apply_fails(engine):
    events = []

    class Staged:
        def release(self):
            events.append("release")

    class Client:
        def stage_weight(self, *, version):
            return Staged()

        def apply_weight(self, staged):
            raise RuntimeError("install failed")

    engine._client = Client()

    with pytest.raises(RuntimeError, match="install failed"):
        engine.receive_weights(MxRefitUpdateInfo(version_id="opaque-a"))

    assert events == ["release"]


def test_engine_lifecycle_guards_and_cleanup(engine):
    with pytest.raises(AssertionError):
        engine.receive_weights(MxRefitUpdateInfo(version_id="opaque-a"))

    calls = []
    engine._client = SimpleNamespace(close=lambda: calls.append("close"))
    engine.shutdown()
    engine.shutdown()

    assert calls == ["close"]
    assert engine._client is None
    assert MxRefitWeightTransferEngine.supports_draft_weight_update is False
    assert {field.name for field in fields(MxRefitInitInfo)} == set(INIT_INFO)
    assert {field.name for field in fields(MxRefitUpdateInfo)} == {"version_id"}
    with pytest.raises(NotImplementedError, match="receiver-pulled"):
        MxRefitWeightTransferEngine.trainer_send_weights(iter(()), {})
