# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest
import torch
from modelexpress_rl.inference.engines.vllm.installer import _VllmInstaller
from modelexpress_rl.inference.plan import (
    PreparedCheckpointArtifact,
    PreparedEngineTensors,
    PreparedRuntimeTensors,
    PreparedStreamingTensors,
)
from modelexpress_rl.inference.receiver import DeltaChange, PreparedCheckpoint
from safetensors.torch import save_file
from torch import nn

LAYERWISE = "vllm.model_executor.model_loader.reload.layerwise"
TENSORS = {
    "norm.weight": torch.tensor([1.0, 2.0]),
    "mlp.gate_proj.weight": torch.tensor([3.0]),
    "mlp.up_proj.weight": torch.tensor([4.0]),
    "other.weight": torch.tensor([5.0]),
}


class _Model(nn.Module):
    packed_modules_mapping: ClassVar = {"gate_up_proj": ["gate_proj", "up_proj"]}

    def __init__(self):
        super().__init__()
        self.loads = []

    def load_weights(self, weights):
        batch = [name for name, _tensor in weights]
        self.loads.append(sorted(batch))
        return set(batch)


@pytest.fixture
def fake_vllm(monkeypatch):
    state = SimpleNamespace(full=[], incomplete=[], events=[])

    @contextmanager
    def current_config(_config):
        yield

    class QuantizeMethodBase:
        pass

    class DefaultModelLoader:
        def __init__(self, _load_config):
            pass

        def load_weights(self, model, model_config):
            state.full.append(model_config.model)

        def get_all_weights(self, model_config, _model):
            state.full.append(model_config.model)
            yield from TENSORS.items()

    names = [
        "vllm",
        "vllm.config",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.attention",
        "vllm.model_executor.layers.quantization",
        "vllm.model_executor.layers.quantization.base_config",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.default_loader",
        "vllm.model_executor.model_loader.reload",
        LAYERWISE,
    ]
    modules = {name: ModuleType(name) for name in names}
    modules["vllm.config"].set_current_vllm_config = current_config
    modules[
        "vllm.model_executor.layers.quantization.base_config"
    ].QuantizeMethodBase = QuantizeMethodBase
    modules["vllm.model_executor.layers.attention"].is_deferred_attention_layer = (
        lambda _layer: False
    )
    modules[
        "vllm.model_executor.model_loader.default_loader"
    ].DefaultModelLoader = DefaultModelLoader
    layerwise = modules[LAYERWISE]
    layerwise.LAYERWISE_INFO = {}

    def initialize(_model):
        for layer in state.incomplete:
            layerwise.LAYERWISE_INFO[layer] = SimpleNamespace(
                load_numel=1,
                load_numel_total=2,
                kernel_tensors=None,
                can_load=lambda: True,
            )

    def finalize(_model, _config):
        logging.getLogger(LAYERWISE).warning("%s: Failed to load weights", "RMSNorm")
        state.events.append("finalize")

    layerwise.initialize_layerwise_reload = initialize
    layerwise.finalize_layerwise_reload = finalize
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    monkeypatch.setenv("MX_REFIT_DELTA_SURGICAL", "true")
    return state


def _checkpoint(tmp_path):
    path = tmp_path / "checkpoint"
    path.mkdir()
    save_file(TENSORS, path / "model.safetensors")
    return path


def _installer(model):
    config = SimpleNamespace(
        load_config=SimpleNamespace(load_format="modelexpress"),
        quant_config=None,
    )
    return _VllmInstaller(
        model=model,
        vllm_config=config,
        model_config=SimpleNamespace(model="/launch", revision=None),
        device=torch.device("cpu"),
    )


def _prepared(path, version, changes=()):
    return PreparedCheckpointArtifact(
        PreparedCheckpoint(version, path, {}, delta_changes=changes)
    )


V2 = (DeltaChange("v1", "v2", frozenset({"mlp.up_proj.weight"})),)


def test_delta_from_the_live_version_installs_only_changed_modules(
    fake_vllm, tmp_path, caplog
):
    path = _checkpoint(tmp_path)
    model = _Model()
    installer = _installer(model)
    installer.install(_prepared(path, "v1"))
    caplog.clear()

    with caplog.at_level(logging.INFO):
        metrics = installer.install(_prepared(path, "v2", V2))

    assert fake_vllm.full == [str(path)]
    assert model.loads == [["mlp.gate_proj.weight", "mlp.up_proj.weight"]]
    assert metrics["perf/mx_receive_surgical_tensors"] == 2
    assert metrics["perf/mx_receive_surgical_fallback"] == 0
    assert "Failed to load weights" not in caplog.text
    assert "Surgical checkpoint install" in caplog.text


@pytest.mark.parametrize("flag", [None, "false"])
def test_surgical_install_is_opt_in(fake_vllm, tmp_path, monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("MX_REFIT_DELTA_SURGICAL", raising=False)
    else:
        monkeypatch.setenv("MX_REFIT_DELTA_SURGICAL", flag)
    path = _checkpoint(tmp_path)
    model = _Model()
    installer = _installer(model)
    installer.install(_prepared(path, "v1"))

    metrics = installer.install(_prepared(path, "v2", V2))

    assert fake_vllm.full == [str(path), str(path)]
    assert model.loads == []
    assert "perf/mx_receive_surgical_tensors" not in metrics


def test_unknown_live_version_uses_the_full_install(fake_vllm, tmp_path):
    path = _checkpoint(tmp_path)
    model = _Model()

    _installer(model).install(_prepared(path, "v2", V2))

    assert fake_vllm.full == [str(path)]
    assert model.loads == []


def test_incomplete_module_falls_back_to_the_rest_of_the_checkpoint(
    fake_vllm, tmp_path
):
    path = _checkpoint(tmp_path)
    model = _Model()
    installer = _installer(model)
    installer.install(_prepared(path, "v1"))
    fake_vllm.incomplete.append(nn.Module())

    metrics = installer.install(_prepared(path, "v2", V2))

    assert model.loads == [
        ["mlp.gate_proj.weight", "mlp.up_proj.weight"],
        ["norm.weight", "other.weight"],
    ]
    assert metrics["perf/mx_receive_surgical_fallback"] == 1


def test_failed_install_forgets_the_live_version(fake_vllm, tmp_path):
    path = _checkpoint(tmp_path)
    model = _Model()
    installer = _installer(model)
    installer.install(_prepared(path, "v1"))

    def fail(_weights):
        raise RuntimeError("load failed")

    model.load_weights = fail
    with pytest.raises(RuntimeError, match="load failed"):
        installer.install(_prepared(path, "v2", V2))
    del model.load_weights

    installer.install(
        _prepared(
            path,
            "v3",
            (*V2, DeltaChange("v2", "v3", frozenset({"norm.weight"}))),
        )
    )

    assert fake_vllm.full == [str(path), str(path)]


@pytest.mark.parametrize("kind", ["engine", "runtime", "streaming"])
def test_non_checkpoint_install_forgets_live_version(
    fake_vllm, tmp_path, monkeypatch, kind
):
    path = _checkpoint(tmp_path)
    model = _Model()
    installer = _installer(model)
    installer.install(_prepared(path, "v1"))
    if kind == "streaming":
        prepared = PreparedStreamingTensors(lambda: iter(()), frozenset(), {})
        monkeypatch.setattr(installer, "install_streaming", lambda _prepared: None)
    else:
        artifact, method = {
            "engine": (PreparedEngineTensors, "install_tensors"),
            "runtime": (PreparedRuntimeTensors, "install_runtime_tensors"),
        }[kind]
        prepared = artifact(SimpleNamespace(tensors={}, metrics={}))
        monkeypatch.setattr(installer, method, lambda _tensors: None)
    installer.install(prepared)
    installer.install(_prepared(path, "v2", V2))
    assert fake_vllm.full == [str(path), str(path)]
    assert model.loads == []
