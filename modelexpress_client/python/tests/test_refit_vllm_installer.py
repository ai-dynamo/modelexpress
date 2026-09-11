# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch
from modelexpress.refit.reshard.types import IncompleteRefit
from modelexpress.refit.timing import RefitTimingRecorder, use_refit_timing
from modelexpress_rl.inference.engines.vllm.installer import (
    _update_mla_absorbed_weights,
    _VllmInstaller,
)
from modelexpress_rl.inference.plan import PreparedStreamingTensors
from torch import nn


def _install_fake_vllm(monkeypatch, initialize):
    @contextmanager
    def current_config(_config):
        yield

    class QuantizeMethodBase:
        pass

    modules = {
        "vllm": ModuleType("vllm"),
        "vllm.config": ModuleType("vllm.config"),
        "vllm.model_executor": ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": ModuleType("vllm.model_executor.layers"),
        "vllm.model_executor.layers.quantization": ModuleType(
            "vllm.model_executor.layers.quantization"
        ),
        "vllm.model_executor.layers.quantization.base_config": ModuleType(
            "vllm.model_executor.layers.quantization.base_config"
        ),
        "vllm.model_executor.model_loader": ModuleType(
            "vllm.model_executor.model_loader"
        ),
        "vllm.model_executor.model_loader.default_loader": ModuleType(
            "vllm.model_executor.model_loader.default_loader"
        ),
        "vllm.model_executor.model_loader.reload": ModuleType(
            "vllm.model_executor.model_loader.reload"
        ),
        "vllm.model_executor.model_loader.reload.layerwise": ModuleType(
            "vllm.model_executor.model_loader.reload.layerwise"
        ),
    }
    modules["vllm.config"].set_current_vllm_config = current_config
    modules[
        "vllm.model_executor.layers.quantization.base_config"
    ].QuantizeMethodBase = QuantizeMethodBase
    layerwise = modules["vllm.model_executor.model_loader.reload.layerwise"]
    layerwise.LAYERWISE_INFO = {}
    layerwise.initialize_layerwise_reload = initialize
    layerwise.finalize_layerwise_reload = lambda _model, _config: None
    layerwise._copy_and_restore_kernel_tensors = lambda _layer, _info: None
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_installer_resolves_load_time_parameters_after_layerwise_reload(monkeypatch):
    model = nn.Module()
    model.register_parameter("packed", nn.Parameter(torch.zeros(1)))

    def initialize(target):
        del target._parameters["packed"]
        target.register_parameter("weight", nn.Parameter(torch.empty(1, device="meta")))

    _install_fake_vllm(monkeypatch, initialize)
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )

    installer._process_and_commit({"weight": torch.tensor([7.0])})

    assert model.weight.item() == 7.0


@pytest.mark.parametrize("fail_second", [False, True])
def test_streaming_preserves_storage_and_propagates_partial_failure(
    monkeypatch, fail_second
):
    _install_fake_vllm(monkeypatch, lambda model: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))
    addresses = {n: p.data_ptr() for n, p in model.named_parameters()}
    before_second = model[1].weight.detach().clone()
    arena = torch.ones(2, 2)

    def batches():
        yield {"0.weight": arena}
        arena.fill_(2)
        assert torch.equal(model[0].weight, torch.ones(2, 2))
        if fail_second:
            raise RuntimeError("injected transport failure")
        yield {"1.weight": arena}
        arena.fill_(3)

    prepared = PreparedStreamingTensors(batches, frozenset(addresses), {})
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    if fail_second:
        with pytest.raises(RuntimeError, match="injected transport failure"):
            installer.install_streaming(prepared)
        assert torch.equal(model[1].weight, before_second)
    else:
        installer.install_streaming(prepared)
        assert torch.equal(model[1].weight, torch.full((2, 2), 2.0))
    assert torch.equal(model[0].weight, torch.ones(2, 2))
    assert {n: p.data_ptr() for n, p in model.named_parameters()} == addresses


def test_installer_rejects_parameters_left_on_meta(monkeypatch):
    model = nn.Module()

    def initialize(target):
        target.register_parameter("weight", nn.Parameter(torch.empty(1, device="meta")))
        target.register_parameter("orphan", nn.Parameter(torch.empty(1, device="meta")))

    _install_fake_vllm(monkeypatch, initialize)
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )

    with pytest.raises(IncompleteRefit, match="left parameters on the meta device"):
        installer._process_and_commit({"weight": torch.tensor([7.0])})


def test_installer_loads_prepared_checkpoint_inside_vllm_config(monkeypatch, tmp_path):
    active_config = [None]
    events = []

    @contextmanager
    def current_config(config):
        active_config[0] = config
        try:
            yield
        finally:
            active_config[0] = None

    def initialize(_model):
        events.append(("initialize", active_config[0]))

    _install_fake_vllm(monkeypatch, initialize)
    sys.modules["vllm.config"].set_current_vllm_config = current_config
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    layerwise.finalize_layerwise_reload = lambda _model, _config: events.append(
        ("finalize", active_config[0])
    )

    class DefaultModelLoader:
        def __init__(self, load_config):
            events.append(("loader", load_config.load_format))

        def load_weights(self, model, model_config):
            events.append(
                (
                    "load",
                    active_config[0],
                    model_config.model,
                    model_config.revision,
                )
            )
            model.weight.data.fill_(7.0)

    sys.modules[
        "vllm.model_executor.model_loader.default_loader"
    ].DefaultModelLoader = DefaultModelLoader
    synchronized = []
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)

    model = nn.Linear(1, 1, bias=False)
    model_config = SimpleNamespace(model="/launch", revision="main")
    vllm_config = SimpleNamespace(
        load_config=SimpleNamespace(load_format="modelexpress"),
        quant_config=None,
    )
    installer = _VllmInstaller(
        model=model,
        vllm_config=vllm_config,
        model_config=model_config,
        device=torch.device("cpu"),
    )
    prepared = tmp_path / "prepared"
    recorder = RefitTimingRecorder(backend="test", version="version-a")

    with use_refit_timing(recorder):
        installer.install_checkpoint(prepared)

    assert events == [
        ("loader", "safetensors"),
        ("initialize", vllm_config),
        ("load", vllm_config, str(prepared), None),
        ("finalize", vllm_config),
    ]
    assert model.weight.item() == 7.0
    assert model_config.model == "/launch"
    assert model_config.revision == "main"
    assert vllm_config.load_config.load_format == "modelexpress"
    assert synchronized == [torch.device("cpu")]
    assert recorder.as_dict()["stages"]["post_install"]["count"] == 1


def test_installer_caches_parameter_layout():
    twin = nn.Module()
    twin.register_parameter(
        "weight",
        nn.Parameter(torch.empty((2, 3), dtype=torch.float16)),
    )
    calls = []
    installer = object.__new__(_VllmInstaller)
    installer._parameter_layout = None

    def build_meta_twin():
        calls.append("build")
        return twin

    installer._build_meta_twin = build_meta_twin

    first = installer.parameter_layout()
    second = installer.parameter_layout()

    assert first == {"weight": ((2, 3), torch.float16)}
    assert second is first
    assert calls == ["build"]


def test_layerwise_capture_and_streaming_preserve_tied_parameters(monkeypatch):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(2, 2, bias=False)
            self.lm_head = nn.Linear(2, 2, bias=False)
            self.lm_head.weight = self.embedding.weight

        def load_weights(self, weights):
            for name, weight in weights:
                if name == "embedding.weight":
                    self.embedding.weight.weight_loader(self.embedding.weight, weight)

    class Info:
        def __init__(self, parameter):
            self.kernel_tensors = ({"weight": parameter}, {})

        def reset(self):
            self.kernel_tensors = None

    def initialize(model):
        for layer in (model.embedding, model.lm_head):
            layerwise.LAYERWISE_INFO[layer] = Info(layer.weight)
            layer.weight = nn.Parameter(torch.empty_like(layer.weight, device="meta"))

    _install_fake_vllm(monkeypatch, initialize)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]

    def place(layer, info):
        layer.weight = info.kernel_tensors[0]["weight"]

    def commit(layer, info):
        info.kernel_tensors[0]["weight"].data.copy_(layer.weight)
        place(layer, info)

    def finalize(model, config):
        for layer in (model.embedding, model.lm_head):
            info = layerwise.LAYERWISE_INFO[layer]
            if info.kernel_tensors is not None:
                place(layer, info)
                info.reset()

    layerwise._get_original_loader = lambda parameter: None
    layerwise._place_kernel_tensors = place
    layerwise._copy_and_restore_kernel_tensors = commit
    layerwise.finalize_layerwise_reload = finalize
    weight_utils = ModuleType("vllm.model_executor.model_loader.weight_utils")
    weight_utils.default_weight_loader = lambda parameter, weight: parameter.data.copy_(
        weight
    )
    monkeypatch.setitem(sys.modules, weight_utils.__name__, weight_utils)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    model = Model()
    original = model.embedding.weight.detach().clone()
    address = model.embedding.weight.data_ptr()
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    capture, layout = installer.capture([("embedding.weight", torch.float32, (2, 2))])
    assert (
        set(layout)
        == {copy.param_name for copy in capture.copies}
        == {"embedding.weight"}
    )
    assert model.embedding.weight is model.lm_head.weight
    assert torch.equal(model.embedding.weight, original)

    def batches():
        yield {"embedding.weight": torch.full((2, 2), 7.0)}

    installer.install_streaming(
        PreparedStreamingTensors(batches, frozenset(layout), {})
    )
    assert model.embedding.weight is model.lm_head.weight
    assert model.embedding.weight.data_ptr() == address
    assert torch.equal(model.lm_head.weight, torch.full((2, 2), 7.0))


def test_installer_rejects_quantized_mla_derived_weight_refresh():
    model = nn.Module()
    mla = nn.Module()
    mla.kv_b_proj = nn.Linear(1, 1, bias=False)
    mla.W_UV = torch.zeros(1)
    model.add_module("mla", mla)

    with pytest.raises(IncompleteRefit, match="quantized kv_b_proj"):
        _update_mla_absorbed_weights(model, quantized=True)
