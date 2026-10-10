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


def test_replacement_native_source_composes_views(capture_installer) -> None:
    installer, model = capture_installer
    installer._convert_native_to_hf = lambda weights: {
        "weight": next(iter(weights.values())).transpose(0, 1)
    }
    installer.capture([("first", torch.float32, (3, 3))])

    def unavailable(weights) -> None:
        raise AssertionError("engine capture is no longer available")

    model.load_weights = unavailable
    capture, layout = installer.capture([("replacement", torch.float32, (3, 3))])
    record = capture.copies[0]
    assert record.src_name == "replacement"
    # Execute the recipe against real replacement bytes.
    source = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    value = source
    for operation, args, kwargs in record.op_chain:
        value = getattr(value, operation)(*args, **dict(kwargs))
    assert torch.equal(value, source.T[:2, :2])
    assert layout == {"weight": ((2, 2), torch.float32)}


def test_incompatible_inputs_do_not_replace_layout(capture_installer) -> None:
    installer, model = capture_installer
    manifest = [("weight", torch.float32, (3, 3))]
    expected = installer.capture(manifest)

    def unavailable(weights) -> None:
        raise AssertionError("engine capture is no longer available")

    model.load_weights = unavailable
    for invalid in (
        [("weight", torch.float32, (4, 4))],
        [("other", torch.float32, (3, 3))],
        [("weight", torch.float16, (3, 3))],
    ):
        with pytest.raises(IncompleteRefit, match="fixed engine input layout"):
            installer.capture(invalid)
    assert installer.capture(manifest) == expected


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

    monkeypatch.setattr(module, "convert_source_weights", unexpected)
    monkeypatch.setattr(module.copy, "deepcopy", unexpected)
    model.load_weights = unexpected
    for value in (2.0, 7.0, -3.0):
        model.weight.data.fill_(value)
        capture, layout = installer.capture(list(manifest))
        assert [(copy.src_name, copy.param_name) for copy in capture.copies] == [
            ("weight", "weight")
        ]
        assert layout == expected[1]
    assert model.capture_calls == 1


def test_ordered_schema_rebinds_once_and_isolates_canonical_layout(capture_installer) -> None:
    installer, model = capture_installer
    calls = []

    def convert(weights) -> dict:
        calls.append(tuple(weights))
        return {"weight": next(iter(weights.values()))}

    installer._convert_native_to_hf = convert
    first = [("a", torch.float32, (3, 3)), ("b", torch.float32, (3, 3))]
    capture, parameters = installer.capture(first)
    assert capture.copies[0].src_name == "a"
    assert parameters == {"weight": ((2, 2), torch.float32)}
    replacement = installer.capture(list(reversed(first)))
    assert replacement[0].copies[0].src_name == "b"
    assert capture.copies[0].src_name == "a"
    assert installer.capture(list(reversed(first)))[0].copies == replacement[0].copies
    assert model.capture_calls == 1
    assert calls == [("a", "b"), ("b", "a")]
