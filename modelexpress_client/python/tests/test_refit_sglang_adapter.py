# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from modelexpress_rl.inference.engines.sglang import (
    SglangGeneratorContext,
    _create_sglang_engine_runtime,
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
        ),
    )


def _install_sglang_modules(monkeypatch, loader=None, setup_error=None):
    class DefaultModelLoader:
        pass

    if loader is None:
        loader = DefaultModelLoader()
        loader._get_weights_iterator = Mock(return_value=iter([]))
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


def test_sglang_install_uses_the_prepared_checkpoint(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    installer = _SglangInstaller(runner)
    loader = _install_sglang_modules(monkeypatch)

    installer.install(_prepared(tmp_path))

    source = loader._get_weights_iterator.call_args.args[0]
    assert Path(source.model_or_path) == tmp_path / "prepared"
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
