# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
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


def test_unchanged_layout_checks_key_and_reuses_capture(capture_installer, monkeypatch):
    installer, model = capture_installer
    manifest = [("weight", torch.float32, (3, 3))]
    installer.capture(manifest)
    original_key = installer._capture_key
    checked = []

    def capture_key(manifest):
        checked.append(manifest)
        return original_key(manifest)

    monkeypatch.setattr(installer, "_capture_key", capture_key)
    capture, layout = installer.capture(manifest)
    assert checked == [manifest]
    assert capture.copies and layout == {"weight": ((2, 2), torch.float32)}
    assert model.capture_calls == 1


@pytest.mark.parametrize("failure", ["loader", "restoration"])
def test_failed_initial_capture_can_be_retried(
    capture_installer, failure
) -> None:
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
    with pytest.raises(RuntimeError):
        installer.capture(manifest)
    model.load_weights = original_loader
    layerwise.LAYERWISE_INFO.clear()

    capture, layout = installer.capture(manifest)
    assert capture.copies and layout == {"weight": ((2, 2), torch.float32)}


def test_changed_source_schema_recaptures(capture_installer):
    installer, model = capture_installer
    installer.capture([("weight", torch.float32, (3, 3))])
    installer.capture([("weight", torch.float32, (4, 4))])
    assert model.capture_calls == 2


@pytest.mark.parametrize(
    "change", ["schema", "converter", "loader", "routing", "alias"]
)
def test_changed_layout_checks_compatibility(capture_installer, change):
    installer, model = capture_installer
    model.register_buffer("routing", torch.tensor([0, 1]))
    manifest = [("weight", torch.float32, (3, 3))]
    installer.capture(manifest)
    if change == "schema":
        manifest = [("weight", torch.float32, (4, 4))]
    elif change == "converter":
        installer._convert_native_to_hf = lambda weights: weights
    elif change == "loader":
        original = model.load_weights
        model.load_weights = lambda weights: original(weights)
    elif change == "routing":
        model.routing = torch.tensor([1, 0])
    else:
        model.other = model.weight
    installer.capture(manifest)
    assert model.capture_calls == 2


@pytest.mark.parametrize(
    "failure", ["converter", "loader", "restoration", "snapshot"]
)
def test_failed_recapture_retries(capture_installer, monkeypatch, failure):
    installer, model = capture_installer
    manifest = [("weight", torch.float32, (3, 3))]
    installer.capture(manifest)
    original_loader = model.load_weights
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]

    def fail(*_args):
        raise RuntimeError("recapture failed")

    manifest = [("weight", torch.float32, (4, 4))]
    with monkeypatch.context() as patch:
        if failure == "converter":
            patch.setattr(installer, "_convert_native_to_hf", fail)
        elif failure == "loader":
            patch.setattr(model, "load_weights", fail)
        elif failure == "restoration":
            patch.setitem(
                layerwise.LAYERWISE_INFO,
                model,
                SimpleNamespace(kernel_tensors=None, reset=fail),
            )
        else:
            patch.setattr(module._CaptureSnapshot, "create", fail)
        with pytest.raises(RuntimeError):
            installer.capture(manifest)

    assert model.load_weights == original_loader
    assert installer.capture(manifest)[0].copies


@pytest.mark.parametrize("failure", ["key", "clone"])
def test_failed_cache_validation_preserves_cache(
    capture_installer, monkeypatch, failure
):
    installer, model = capture_installer
    manifest = [("weight", torch.float32, (3, 3))]
    installer.capture(manifest)
    cache = installer._capture_cache

    def fail(*_args):
        raise RuntimeError("validation failed")

    with monkeypatch.context() as patch:
        if failure == "key":
            patch.setattr(installer, "_capture_key", fail)
        else:
            patch.setattr(module._CaptureSnapshot, "clone", fail)
        with pytest.raises(RuntimeError, match="validation failed"):
            installer.capture(manifest)
        assert installer._capture_cache is cache

    assert installer.capture(manifest)[0].copies
    assert model.capture_calls == 1


def test_conversion_and_loader_are_captured_together(capture_installer):
    installer, model = capture_installer
    manifest = [("native", torch.float32, (4, 4))]
    installer._convert_native_to_hf = lambda weights: {"weight": weights["native"]}
    first, _ = installer.capture(manifest)
    installer._convert_native_to_hf = lambda weights: {
        "weight": weights["native"][1:, :]
    }
    changed, _ = installer.capture(manifest)
    assert {record.src_name for record in changed.copies} == {"native"}
    assert first.copies[0].op_chain != changed.copies[0].op_chain
    assert model.capture_calls == 2
