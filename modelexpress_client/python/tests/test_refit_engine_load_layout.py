# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from modelexpress.refit.reshard.types import IncompleteRefit
from modelexpress_rl.inference.engines.vllm import installer as module
from torch import nn
from tests.test_refit_vllm_installer import _install_fake_vllm


@pytest.fixture
def capture_installer(monkeypatch) -> tuple:
    _install_fake_vllm(monkeypatch, lambda model: None)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    layerwise._get_original_loader = lambda parameter: None
    layerwise._place_kernel_tensors = lambda layer, info: None
    weight_utils = ModuleType("vllm.model_executor.model_loader.weight_utils")
    weight_utils.default_weight_loader = lambda parameter, weight: parameter.data.copy_(
        weight
    )
    monkeypatch.setitem(sys.modules, weight_utils.__name__, weight_utils)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, 2))
            self.capture_calls = 0

        def load_weights(self, weights) -> None:
            self.capture_calls += 1
            for name, weight in weights:
                self.weight.weight_loader(self.weight, weight[:2, :2])

    model = Model()
    installer = module._VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    return installer, model


@pytest.mark.parametrize("clients", [1, 2])
def test_capture_is_reused_across_runtime_parameter_replacement(capture_installer, clients) -> None:
    installer, model = capture_installer
    manifest = [("weight", torch.float32, (3, 3))]
    installers = [installer]
    if clients == 2:
        installers.append(module._VllmInstaller(
            model=model, vllm_config=object(), model_config=object(), device=torch.device("cpu")
        ))
    for value in (2.0, 7.0, -3.0):
        model.weight = nn.Parameter(torch.full((2, 2), value))
        for installer in installers:
            capture, layout = installer.capture(manifest)
            assert layout == {"weight": ((2, 2), torch.float32)}
            assert [(copy.src_name, copy.param_name) for copy in capture.copies] == [
                ("weight", "weight")
            ]
            assert torch.equal(model.weight, torch.full((2, 2), value))
    assert model.capture_calls == clients


@pytest.mark.parametrize(
    "failure,error",
    [("loader", RuntimeError), ("restoration", RuntimeError), ("coverage", IncompleteRefit)],
)
def test_failed_initial_capture_can_be_retried(capture_installer, failure, error) -> None:
    installer, model = capture_installer
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    original_loader = model.load_weights
    manifest = [("weight", torch.float32, (3, 3))]

    def fail(*_args) -> None:
        raise RuntimeError("initialization failed")

    if failure == "loader":
        model.load_weights = fail
    elif failure == "restoration":
        layerwise.LAYERWISE_INFO[model] = SimpleNamespace(kernel_tensors=None, reset=fail)
    else:
        model.load_weights = lambda weights: None
    with pytest.raises(error):
        installer.capture(manifest)
    model.load_weights = original_loader
    layerwise.LAYERWISE_INFO.clear()

    capture, layout = installer.capture(manifest)
    assert capture.copies and layout == {"weight": ((2, 2), torch.float32)}
