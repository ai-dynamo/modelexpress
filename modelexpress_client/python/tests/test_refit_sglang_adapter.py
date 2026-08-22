# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from modelexpress_rl.inference.engines.sglang import SGLangGeneratorAdapter
from modelexpress_rl.inference.receiver import PreparedCheckpoint, ReceiverInstallError


def _runner(tmp_path):
    checkpoint = tmp_path / "launch"
    return SimpleNamespace(
        model=object(),
        device="cpu",
        loader=SimpleNamespace(
            _prepare_weights=Mock(return_value=(checkpoint, None, None))
        ),
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


def test_sglang_install_uses_the_prepared_checkpoint(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    adapter = object.__new__(SGLangGeneratorAdapter)
    adapter.model_runner = runner
    loader = _install_sglang_modules(monkeypatch)
    prepared = PreparedCheckpoint("target-a", tmp_path / "prepared", {})

    adapter.install_prepared_checkpoint(prepared)

    source = loader._get_weights_iterator.call_args.args[0]
    assert Path(source.model_or_path) == prepared.path
    loader.load_weights_and_postprocess.assert_called_once()


@pytest.mark.parametrize(
    ("setup_error", "load_error", "mutation_started"),
    [
        (RuntimeError("setup failed"), None, False),
        (None, RuntimeError("load failed"), True),
    ],
)
def test_sglang_install_classifies_live_mutation(
    tmp_path,
    monkeypatch,
    setup_error,
    load_error,
    mutation_started,
):
    runner = _runner(tmp_path)
    adapter = object.__new__(SGLangGeneratorAdapter)
    adapter.model_runner = runner

    class Loader:
        def _get_weights_iterator(self, _source):
            return iter([])

        def load_weights_and_postprocess(self, *_args):
            if load_error is not None:
                raise load_error

    _install_sglang_modules(monkeypatch, Loader(), setup_error)

    with pytest.raises(ReceiverInstallError) as error:
        adapter.install_prepared_checkpoint(
            PreparedCheckpoint("target-a", tmp_path / "prepared", {})
        )

    assert error.value.mutation_started is mutation_started
