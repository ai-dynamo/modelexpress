# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch
from modelexpress.refit.reshard.types import IncompleteRefit
from modelexpress_rl.inference.engines.vllm import installer as module
from torch import nn


@pytest.fixture
def capture_installer(monkeypatch) -> tuple:
    names = (
        "vllm",
        "vllm.config",
        "vllm.model_executor",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.reload",
        "vllm.model_executor.model_loader.reload.layerwise",
    )
    modules = {name: ModuleType(name) for name in names}
    modules["vllm.config"].set_current_vllm_config = lambda config: nullcontext()
    layerwise = modules["vllm.model_executor.model_loader.reload.layerwise"]
    layerwise.LAYERWISE_INFO = {}
    layerwise.initialize_layerwise_reload = lambda model: None
    layerwise._get_original_loader = lambda parameter: None
    layerwise._place_kernel_tensors = lambda layer, info: None
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
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


@pytest.mark.parametrize("quantized", [False, True])
def test_layout_survives_runtime_parameter_replacement(
    capture_installer, quantized
) -> None:
    installer, model = capture_installer
    installer._vllm_config = SimpleNamespace(
        quant_config=object() if quantized else None
    )
    manifest = [("weight", torch.float32, (3, 3))]
    capture, layout = installer.capture(manifest)

    def unavailable(weights) -> None:
        raise AssertionError("engine capture is no longer available")

    model.load_weights = unavailable
    model.weight = nn.Parameter(torch.ones(2, 2))
    first, second_layout = installer.capture(manifest)
    assert second_layout == layout == {"weight": ((2, 2), torch.float32)}
    assert [(copy.src_name, copy.param_name) for copy in first.copies] == [
        ("weight", "weight")
    ]
    assert installer.capture(manifest)[0].copies == first.copies
    assert model.capture_calls == 1


def test_failed_initial_capture_allows_later_initialization(
    capture_installer, monkeypatch
) -> None:
    installer, model = capture_installer
    original = model.load_weights

    def fail(weights) -> None:
        raise RuntimeError("capture failed")

    model.load_weights = fail
    with pytest.raises(RuntimeError, match="capture failed"):
        installer.capture([("weight", torch.float32, (3, 3))])
    model.load_weights = original
    capture, layout = installer.capture([("weight", torch.float32, (4, 4))])
    assert capture.copies and layout == {"weight": ((2, 2), torch.float32)}


def test_failed_restoration_does_not_fix_input_schema(capture_installer) -> None:
    installer, model = capture_installer
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]

    def fail() -> None:
        raise RuntimeError("restoration failed")

    layerwise.LAYERWISE_INFO[model] = SimpleNamespace(kernel_tensors=None, reset=fail)
    with pytest.raises(RuntimeError, match="restoration failed"):
        installer.capture([("weight", torch.float32, (3, 3))])
    layerwise.LAYERWISE_INFO.clear()
    capture, layout = installer.capture([("weight", torch.float32, (4, 4))])
    assert capture.copies and layout == {"weight": ((2, 2), torch.float32)}


def test_incomplete_initial_capture_can_be_retried(capture_installer) -> None:
    installer, model = capture_installer
    loader = model.load_weights
    model.load_weights = lambda weights: None
    with pytest.raises(IncompleteRefit, match="cover every parameter"):
        installer.capture([("weight", torch.float32, (3, 3))])
    model.load_weights = loader
    capture, layout = installer.capture([("weight", torch.float32, (4, 4))])
    assert capture.copies and layout == {"weight": ((2, 2), torch.float32)}


@pytest.mark.parametrize("quantized", [False, True])
def test_unchanged_source_schema_reuses_binding(
    capture_installer, monkeypatch, quantized
) -> None:
    installer, model = capture_installer
    installer._vllm_config = SimpleNamespace(
        quant_config=object() if quantized else None
    )
    manifest = [("weight", torch.float32, (3, 3))]
    expected = installer.capture(manifest)

    def unexpected(*args, **kwargs) -> None:
        raise AssertionError("warm mapping must not convert, bind, copy or trace")

    monkeypatch.setattr(module, "capture_weights", unexpected)
    monkeypatch.setattr(installer, "_parameter_aliases", unexpected)
    model.load_weights = unexpected
    for value in (2.0, 7.0, -3.0):
        model.weight.data.fill_(value)
        capture, layout = installer.capture(list(manifest))
        assert [(copy.src_name, copy.param_name) for copy in capture.copies] == [
            ("weight", "weight")
        ]
        assert layout == expected[1]
    assert model.capture_calls == 1


def test_client_install_plan_is_not_revalidated(capture_installer, monkeypatch) -> None:
    installer, model = capture_installer
    first = installer.capture([("weight", torch.float32, (3, 3))])
    plan = installer._engine_load_layout

    def unexpected(*args, **kwargs) -> None:
        raise AssertionError("retained install plan must not inspect the live model")

    monkeypatch.setattr(model, "named_modules", unexpected)
    monkeypatch.setattr(model, "named_parameters", unexpected)
    monkeypatch.setattr(model, "load_weights", unexpected)
    second = installer.capture([("different_source", torch.float16, (9, 9))])
    assert second[0] is first[0]
    assert second[1] is first[1]
    assert installer._engine_load_layout is plan
    assert not hasattr(installer, "_capture_cache")


def test_install_plan_owns_module_execution_metadata(capture_installer) -> None:
    installer, model = capture_installer
    installer.capture([("weight", torch.float32, (3, 3))])
    plan = installer._engine_load_layout
    assert len(plan.groups) == 1
    owner, parameters = plan.groups[0]
    assert owner is model
    assert parameters == (("weight", "weight"),)
    assert plan.parameters == {"weight": ((2, 2), torch.float32)}


def test_new_client_has_its_own_install_plan(capture_installer) -> None:
    first, model = capture_installer
    schema = [("weight", torch.float32, (3, 3))]
    first.capture(schema)
    second = module._VllmInstaller(
        model=model, vllm_config=object(), model_config=object(), device=torch.device("cpu")
    )
    second.capture(schema)
    assert first._engine_load_layout is not second._engine_load_layout
    assert model.capture_calls == 2
