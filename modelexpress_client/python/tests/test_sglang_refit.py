# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from modelexpress.engines.sglang.refit import receiver as refit
from modelexpress.refit.factory import RolloutBackend, build_delta_receiver


def runner(tmp_path):
    loader = SimpleNamespace(
        _prepare_weights=Mock(return_value=(tmp_path / "hf", None, None))
    )
    return SimpleNamespace(
        model=object(),
        device="cpu",
        loader=loader,
        model_config=SimpleNamespace(
            model_path=str(tmp_path / "hf"),
            revision=None,
            dtype=torch.float32,
        ),
        server_args=SimpleNamespace(
            model_path="original-model",
            load_format="auto",
            modelexpress_model_id="model",
            modelexpress_catalog_endpoint="mx:8001",
            modelexpress_delta_s3_endpoint="http://minio:9000",
            modelexpress_preparation_cache_dir=str(tmp_path / "cache"),
            modelexpress_initial_version="0",
            modelexpress_ready_timeout_seconds=123,
            download_dir=None,
            model_loader_extra_config=None,
        ),
    )


def receiver(tmp_path):
    model_runner = runner(tmp_path)
    return (
        refit.SglangWeightReceiver(
            refit.ReceiverConfig(
                model_id="model",
                catalog_endpoint="mx:8001",
                initial_version="0",
                preparation_cache_dir=tmp_path / "cache",
            ),
            "host:0",
            model_runner,
        ),
        model_runner,
    )


def install_sglang_modules(monkeypatch, loader=None, setup_error=None):
    class DefaultModelLoader:
        pass

    if loader is None:
        loader = DefaultModelLoader()
        loader._get_weights_iterator = Mock(
            return_value=iter([("weight", torch.ones(1))])
        )
        loader.load_weights_and_postprocess = Mock()
    else:
        DefaultModelLoader = type(loader)

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
    load_config = modules["sglang.srt.configs.load_config"]
    load_config.LoadConfig = lambda **values: SimpleNamespace(**values)
    load_config.LoadFormat = SimpleNamespace(SAFETENSORS="safetensors")
    loader_module = modules["sglang.srt.model_loader.loader"]
    loader_module.DefaultModelLoader = DefaultModelLoader
    if setup_error is None:
        loader_module.get_model_loader = lambda *_args: loader
    else:
        loader_module.get_model_loader = Mock(side_effect=setup_error)
    modules["sglang.srt.model_loader.utils"].set_default_torch_dtype = lambda _dtype: (
        nullcontext()
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return loader


def test_factory_builds_sglang_receiver(tmp_path, monkeypatch):
    captured = {}

    class Receiver:
        def __init__(self, config, receiver_id, model_runner):
            captured.update(
                config=config,
                receiver_id=receiver_id,
                model_runner=model_runner,
            )

    monkeypatch.setattr(refit, "SglangWeightReceiver", Receiver)
    model_runner = runner(tmp_path)
    config = refit.ReceiverConfig(
        model_id="model",
        catalog_endpoint="mx:8001",
        initial_version="0",
        preparation_cache_dir=tmp_path / "cache",
        ready_timeout_seconds=123,
        s3_endpoint_url="http://minio:9000",
    )

    result = build_delta_receiver(
        RolloutBackend.SGLANG,
        config=config,
        receiver_id="host:0",
        model_runner=model_runner,
    )

    assert isinstance(result, Receiver)
    assert captured["config"] is config
    assert captured["receiver_id"] == "host:0"
    assert captured["model_runner"] is model_runner


def test_receiver_constructor_prepares_launch_checkpoint(tmp_path):
    value, model_runner = receiver(tmp_path)

    assert value.launch_checkpoint == tmp_path / "hf"
    model_runner.loader._prepare_weights.assert_called_once_with(
        model_runner.model_config.model_path,
        model_runner.model_config.revision,
        False,
    )


def test_receiver_install_uses_prepared_path_without_reconfiguring_runner(
    tmp_path, monkeypatch
):
    value, model_runner = receiver(tmp_path)
    loader = install_sglang_modules(monkeypatch)
    prepared = refit.PreparedRevision("1", "sha256:target", tmp_path / "prepared", {})

    value.install_prepared_checkpoint(prepared)

    source = loader._get_weights_iterator.call_args.args[0]
    assert Path(source.model_or_path) == prepared.path
    loader.load_weights_and_postprocess.assert_called_once()
    assert model_runner.model_config.model_path == str(tmp_path / "hf")
    assert model_runner.server_args.model_path == "original-model"
    assert model_runner.server_args.load_format == "auto"


def test_receiver_install_classifies_setup_failure_before_write(tmp_path, monkeypatch):
    value, _model_runner = receiver(tmp_path)
    install_sglang_modules(monkeypatch, setup_error=RuntimeError("loader setup failed"))

    with pytest.raises(refit.ReceiverInstallError) as error:
        value.install_prepared_checkpoint(
            refit.PreparedRevision("1", "sha256:target", tmp_path, {})
        )

    assert error.value.mutation_started is False


def test_receiver_install_classifies_load_failure_after_possible_write(
    tmp_path, monkeypatch
):
    value, _model_runner = receiver(tmp_path)

    class Loader:
        def _get_weights_iterator(self, _source):
            return iter([("weight", torch.ones(1))])

        def load_weights_and_postprocess(self, *_args):
            raise RuntimeError("load failed")

    install_sglang_modules(monkeypatch, Loader())

    with pytest.raises(refit.ReceiverInstallError) as error:
        value.install_prepared_checkpoint(
            refit.PreparedRevision("1", "sha256:target", tmp_path, {})
        )

    assert error.value.mutation_started is True
