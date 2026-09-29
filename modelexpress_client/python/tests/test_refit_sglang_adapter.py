# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from modelexpress_rl import ModelExpressGeneratorClient, ObjectStorageType
from modelexpress_rl.inference.engines.sglang import (
    SglangGeneratorContext,
    _create_sglang_engine_runtime,
    get_modelexpress_generator,
)
from modelexpress_rl.inference.engines.sglang.installer import _SglangInstaller
from modelexpress_rl.inference.plan import PreparedCheckpointArtifact
from modelexpress_rl.inference.receiver import (
    PreparedCheckpoint,
    ReceiverInstallError,
)


def _runner(tmp_path):
    checkpoint = tmp_path / "launch"
    return SimpleNamespace(
        model=object(),
        device="cpu",
        model_config=SimpleNamespace(
            model_path=str(checkpoint),
            revision=None,
            dtype=torch.float32,
        ),
        server_args=SimpleNamespace(
            download_dir=None,
            model_loader_extra_config=None,
            modelexpress_config={
                "model_name": "model",
                "initial_base_version_id": "base-a",
                "refit_checkpoint_dir": str(tmp_path / "refit"),
            },
        ),
        loader=SimpleNamespace(
            _prepare_weights=Mock(return_value=(str(checkpoint), None, None)),
        ),
    )


def _install_sglang_modules(monkeypatch, loader=None, setup_error=None):
    @dataclass
    class Source:
        model_or_path: str
        revision: str | None
        prefix: str = ""
        fall_back_to_pt: bool = True
        allow_patterns_overrides: list[str] | None = None
        model_config: object | None = None

    class DefaultModelLoader:
        pass

    if loader is None:
        loader = DefaultModelLoader()
        loader._get_weights_iterator = Mock(return_value=iter([]))
        loader.load_weights_and_postprocess = Mock()
    else:
        DefaultModelLoader = type(loader)
    DefaultModelLoader.Source = Source

    modules = {
        name: ModuleType(name)
        for name in (
            "sglang",
            "sglang.srt",
            "sglang.srt.configs",
            "sglang.srt.configs.load_config",
            "sglang.srt.model_loader",
            "sglang.srt.model_loader.loader",
            "sglang.srt.model_loader.utils",
        )
    }
    modules["sglang.srt.configs.load_config"].LoadConfig = lambda **values: (
        SimpleNamespace(**values)
    )
    modules["sglang.srt.configs.load_config"].LoadFormat = SimpleNamespace(
        SAFETENSORS="safetensors"
    )
    loader_module = modules["sglang.srt.model_loader.loader"]
    loader_module.DefaultModelLoader = DefaultModelLoader
    loader_module.get_model_loader = (
        Mock(side_effect=setup_error)
        if setup_error is not None
        else lambda *_args: loader
    )
    modules["sglang.srt.model_loader.utils"].set_default_torch_dtype = lambda _dtype: (
        nullcontext()
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return loader


def _prepared(tmp_path):
    checkpoint = PreparedCheckpoint("target-a", tmp_path / "prepared", {})
    return PreparedCheckpointArtifact(checkpoint)


def test_sglang_engine_runtime_exposes_checkpoint_installer(tmp_path):
    runner = _runner(tmp_path)

    runtime = _create_sglang_engine_runtime(SglangGeneratorContext(runner))

    assert runtime.model_name == runner.model_config.model_path
    assert isinstance(runtime.installer, _SglangInstaller)
    assert runtime.full_tensor is None


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("seed_checkpoint", [None, "/models/seed"])
@pytest.mark.parametrize(
    "server_config",
    [{"url": "mx:8001"}, {"server_url": "mx:8001", "url": "unused:8001"}],
)
def test_generator_factory_reuses_client_and_shared_config(
    tmp_path, monkeypatch, as_json, seed_checkpoint, server_config
):
    runner = _runner(tmp_path)
    runner.server_args.modelexpress_config.update(
        **server_config,
        seed_checkpoint_path=seed_checkpoint,
        object_storage_endpoint_url="http://minio:9000",
        object_storage_region_name="us-west-2",
        registration_ttl_seconds=60,
        lease_ttl_seconds=90,
        max_transfer_attempts=4,
        max_replay_chain_length=32,
        rpc_timeout_seconds=15.0,
        full_hf_checkpoint_interval=5,
        refit_checkpoint_max_size_gb=200,
    )
    if as_json:
        runner.server_args.modelexpress_config = json.dumps(
            runner.server_args.modelexpress_config
        )
    generator = object()
    initialize = Mock(return_value=generator)
    monkeypatch.setattr(ModelExpressGeneratorClient, "initialize", initialize)

    assert get_modelexpress_generator(runner) is generator
    assert get_modelexpress_generator(runner) is generator

    initialize.assert_called_once()
    config = initialize.call_args.args[0]
    assert config.engine_context.model_runner is runner
    assert config.model_name == "model"
    assert config.server_url == "mx:8001"
    assert config.registration_ttl_seconds == 60
    assert config.lease_ttl_seconds == 90
    assert config.max_transfer_attempts == 4
    assert config.max_replay_chain_length == 32
    assert config.rpc_timeout_seconds == 15.0
    assert not hasattr(config, "full_hf_checkpoint_interval")
    assert vars(config.object_storage) == {
        "endpoint_url": "http://minio:9000",
        "initial_base_version_id": "base-a",
        "seed_checkpoint_path": seed_checkpoint or str(tmp_path / "launch"),
        "region_name": "us-west-2",
        "refit_checkpoint_dir": str(tmp_path / "refit"),
        "refit_checkpoint_max_size_gb": 200,
        "storage_type": ObjectStorageType.S3,
    }
    assert runner.modelexpress_generator is generator
    if seed_checkpoint is None:
        runner.loader._prepare_weights.assert_called_once_with(
            runner.model_config.model_path, runner.model_config.revision, False
        )
    else:
        runner.loader._prepare_weights.assert_not_called()


def test_generator_factory_preserves_sglang_defaults(tmp_path, monkeypatch):
    initialize = Mock()
    monkeypatch.setattr(ModelExpressGeneratorClient, "initialize", initialize)

    get_modelexpress_generator(_runner(tmp_path))

    config = initialize.call_args.args[0]
    assert (
        config.max_transfer_attempts,
        config.max_replay_chain_length,
        config.rpc_timeout_seconds,
        config.object_storage.refit_checkpoint_max_size_gb,
    ) == (3, 64, 30.0, 500)


def test_generator_factory_can_retry_initialization(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    generator = object()
    initialize = Mock(side_effect=[RuntimeError("initialization failed"), generator])
    monkeypatch.setattr(ModelExpressGeneratorClient, "initialize", initialize)

    with pytest.raises(RuntimeError, match="initialization failed"):
        get_modelexpress_generator(runner)

    assert get_modelexpress_generator(runner) is generator
    assert initialize.call_count == 2


def test_sglang_install_uses_the_prepared_checkpoint(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    installer = _SglangInstaller(runner)
    loader = _install_sglang_modules(monkeypatch)

    installer.install(_prepared(tmp_path))

    source = loader._get_weights_iterator.call_args.args[0]
    assert Path(source.model_or_path) == tmp_path / "prepared"
    assert isinstance(source, type(loader).Source)
    assert source.allow_patterns_overrides is None
    assert source.model_config is runner.model_config
    loader.load_weights_and_postprocess.assert_called_once()


def test_sglang_install_rejects_unsupported_prepared_artifact(tmp_path):
    installer = _SglangInstaller(_runner(tmp_path))

    with pytest.raises(TypeError, match="requires a prepared checkpoint"):
        installer.install(object())


@pytest.mark.parametrize(
    ("setup_error", "load_error", "message"),
    [
        (RuntimeError("setup failed"), None, "setup failed"),
        (None, RuntimeError("load failed"), "load failed"),
    ],
)
def test_sglang_install_wraps_errors(
    tmp_path,
    monkeypatch,
    setup_error,
    load_error,
    message,
):
    runner = _runner(tmp_path)
    installer = _SglangInstaller(runner)

    class Loader:
        def _get_weights_iterator(self, _source):
            return iter([])

        def load_weights_and_postprocess(self, *_args):
            if load_error is not None:
                raise load_error

    _install_sglang_modules(monkeypatch, Loader(), setup_error)

    with pytest.raises(ReceiverInstallError, match=message):
        installer.install(_prepared(tmp_path))
