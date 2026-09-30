# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch
from modelexpress.engines.vllm.host_quantization import (
    refresh_host_quantization_state,
)
from modelexpress.refit.reshard.types import IncompleteRefit
from modelexpress.refit.timing import RefitTimingRecorder, use_refit_timing
from modelexpress_rl.inference.engines.vllm import direct_copy
from modelexpress_rl.inference.engines.vllm.installer import (
    _VllmInstaller,
)
from modelexpress_rl.inference.plan import (
    PreparedCheckpointArtifact,
    PreparedDirectGroupTensors,
    PreparedEngineTensors,
    PreparedRuntimeTensors,
    PreparedStreamingTensors,
)
from modelexpress_rl.inference.receiver import PreparedCheckpoint
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
        "vllm.version": ModuleType("vllm.version"),
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
    modules["vllm.version"].__version__ = "0.19.0"
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
    prepared_path = tmp_path / "prepared"
    prepared = PreparedCheckpointArtifact(
        PreparedCheckpoint(
            target_version="version-a",
            path=prepared_path,
            metrics={"bytes_received": 7.0},
        )
    )
    recorder = RefitTimingRecorder(backend="test", version="version-a")

    with use_refit_timing(recorder):
        metrics = installer.install(prepared)

    assert events == [
        ("loader", "safetensors"),
        ("initialize", vllm_config),
        ("load", vllm_config, str(prepared_path), None),
        ("finalize", vllm_config),
    ]
    assert model.weight.item() == 7.0
    assert model_config.model == "/launch"
    assert model_config.revision == "main"
    assert vllm_config.load_config.load_format == "modelexpress"
    assert synchronized == [torch.device("cpu")]
    assert metrics["bytes_received"] == 7.0
    assert metrics["perf/mx_receive_install_time"] >= 0
    assert "streaming_apply_s" not in metrics
    assert recorder.as_dict()["stages"]["post_install"]["count"] == 1


@pytest.mark.parametrize("direct", [False, True])
def test_bounded_install_reports_transfer_and_apply_without_install_only_metric(
    monkeypatch, direct
):
    model = nn.Linear(2, 2, bias=False)
    values = torch.full_like(model.weight, 7.0)
    transfer_metrics = {}

    def batches():
        transfer_metrics.update(wire_s=0.25, bytes_received=16.0)
        yield {"weight": values}

    source = PreparedStreamingTensors(
        batches, frozenset({"weight"}), transfer_metrics
    )
    if direct:
        prepared = direct_copy.prepare_direct_copy(
            model,
            version_id="version-a",
            source=source,
            batch_names=(source.parameter_names,),
        )
        assert isinstance(prepared, PreparedDirectGroupTensors)
    else:
        _install_fake_vllm(monkeypatch, lambda _model: None)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
        prepared = source
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )

    metrics = installer.install(prepared)

    assert torch.equal(model.weight, values)
    assert metrics["wire_s"] == 0.25
    assert metrics["bytes_received"] == 16.0
    assert metrics["streaming_apply_s"] >= 0
    assert "perf/mx_receive_install_time" not in metrics


def test_installer_includes_prepared_engine_tensor_metrics(monkeypatch):
    installer = _VllmInstaller(
        model=nn.Module(),
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(installer, "install_tensors", lambda _tensors: None)
    staged = SimpleNamespace(tensors={}, metrics={"bytes_received": 7.0})

    metrics = installer.install(PreparedEngineTensors(staged=staged))

    assert metrics["bytes_received"] == 7.0
    assert metrics["perf/mx_receive_install_time"] >= 0
    assert "streaming_apply_s" not in metrics


def test_installer_restores_runtime_buffer_created_after_reload_metadata(monkeypatch):
    model = nn.Module()
    original_workspace = torch.tensor([1.0])
    model.register_buffer("workspace", original_workspace)
    info = SimpleNamespace(
        kernel_tensors=({}, {"workspace": original_workspace}),
    )

    def initialize(target):
        delattr(target, "workspace")

    _install_fake_vllm(monkeypatch, initialize)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    layerwise.LAYERWISE_INFO[model] = info

    def finalize(target, _config):
        _, buffers = info.kernel_tensors
        for name, buffer in buffers.items():
            if name in target._buffers:
                buffer.data.copy_(getattr(target, name))
        for name in list(target._parameters) + list(target._buffers):
            delattr(target, name)
        for name, buffer in buffers.items():
            target.register_buffer(name, buffer)

    layerwise.finalize_layerwise_reload = finalize
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )

    installer._reload(lambda: setattr(model, "workspace", torch.tensor([7.0])))

    assert model.workspace is original_workspace
    assert model.workspace.item() == 7.0


def test_installer_accepts_runtime_tensors_written_directly_in_place():
    live = {
        "weight": torch.tensor([7.0, 8.0]),
        "runtime_buffer": torch.tensor([9.0]),
    }
    original_pointers = {name: tensor.data_ptr() for name, tensor in live.items()}
    installer = _VllmInstaller(
        model=nn.Module(),
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
        runtime_tensors=live,
    )
    staged = type("Staged", (), {"tensors": live, "metrics": {"bytes_received": 0}})()

    metrics = installer.install(PreparedRuntimeTensors(staged=staged))

    assert torch.equal(live["weight"], torch.tensor([7.0, 8.0]))
    assert torch.equal(live["runtime_buffer"], torch.tensor([9.0]))
    assert {
        name: tensor.data_ptr() for name, tensor in live.items()
    } == original_pointers
    assert metrics["bytes_received"] == 0


def test_installer_rejects_a_runtime_staging_copy():
    live = {"weight": torch.tensor([1.0])}
    installer = _VllmInstaller(
        model=nn.Module(),
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
        runtime_tensors=live,
    )
    staged = type(
        "Staged",
        (),
        {"tensors": {"weight": torch.tensor([7.0])}, "metrics": {}},
    )()

    with pytest.raises(IncompleteRefit, match="directly into live storage"):
        installer.install(PreparedRuntimeTensors(staged=staged))


@pytest.fixture
def warm_runtime_install(monkeypatch, mock_accelerator_backend_cls):
    backend = mock_accelerator_backend_cls(torch_device_type="cpu")
    monkeypatch.setattr(
        "modelexpress_rl.inference.engines.vllm.installer.accelerator_backend_for",
        lambda _device: backend,
    )
    model = nn.Module()
    attn = nn.Module()
    model.attn = attn
    for key, value in (("q", 0.25), ("k", 0.5), ("v", 0.75)):
        attn.register_buffer(f"_{key}_scale", torch.tensor([value / 2, value]))
        setattr(attn, f"_{key}_scale_float", 1.0)
    attn._k_scale_cpu = torch.tensor(1.0)
    attn._v_scale_cpu = torch.tensor(1.0)
    attn.register_buffer("_prob_scale", torch.tensor(0.125))
    attn._prob_scale_float = 1.0
    attn._o_scale_float = 2.0
    attn.impl = SimpleNamespace(bmm1_scale=3.0, bmm2_scale=4.0, o_sf_scale=5.0)
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        quant_config=object(),
        cache_config=SimpleNamespace(cache_dtype="fp8_e4m3"),
    )
    live = dict(model.named_buffers())
    installer = _VllmInstaller(
        model=model,
        vllm_config=config,
        model_config=config.model_config,
        device=torch.device("cpu"),
        runtime_tensors=live,
    )
    staged = SimpleNamespace(tensors=live, metrics={})
    return installer, attn, PreparedRuntimeTensors(staged=staged)


def test_runtime_refit_refreshes_host_scales_and_invalidates_warm_caches(
    warm_runtime_install,
):
    installer, attn, prepared = warm_runtime_install
    tensors = dict(attn.named_buffers()) | {
        "_k_scale_cpu": attn._k_scale_cpu,
        "_v_scale_cpu": attn._v_scale_cpu,
    }
    pointers = {name: tensor.data_ptr() for name, tensor in tensors.items()}
    for factor in (1, 2):
        for key, value in (("q", 0.25), ("k", 0.5), ("v", 0.75)):
            getattr(attn, f"_{key}_scale").copy_(
                torch.tensor([factor * value / 2, factor * value])
            )
        received = {
            name: tensor.clone() for name, tensor in attn.named_buffers()
        }
        installer.install(prepared)

        assert attn._q_scale_float == factor * 0.25
        assert attn._k_scale_float == attn._k_scale_cpu.item() == factor * 0.5
        assert attn._v_scale_float == attn._v_scale_cpu.item() == factor * 0.75
        assert attn._prob_scale_float == 1.0
        assert attn._o_scale_float is None
        assert attn.impl.bmm1_scale is None
        assert attn.impl.bmm2_scale is None
        assert attn.impl.o_sf_scale is None
        for name, tensor in attn.named_buffers():
            assert torch.equal(tensor, received[name])
        for name, tensor in tensors.items():
            assert getattr(attn, name) is tensor
            assert tensor.data_ptr() == pointers[name]

        # Simulate caches repopulated by inference before the next refit.
        attn._o_scale_float = 6.0
        attn.impl.bmm1_scale = 7.0
        attn.impl.bmm2_scale = 8.0
        attn.impl.o_sf_scale = 9.0


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("_k_scale", torch.tensor(float("nan")), "finite, positive"),
        ("_v_scale", torch.tensor(0.0), "finite, positive"),
        ("_k_scale_cpu", torch.ones(2), "Invalid vLLM attention CPU scale"),
        ("_q_scale_float", None, "Invalid vLLM attention host scalar"),
    ],
)
def test_runtime_refit_rejects_invalid_scale_state(
    warm_runtime_install, attribute, value, message,
):
    installer, attn, prepared = warm_runtime_install
    if attribute in attn._buffers:
        getattr(attn, attribute).fill_(value.item())
    else:
        setattr(attn, attribute, value)
    with pytest.raises(RuntimeError, match=message):
        installer.install(prepared)


def test_runtime_refit_validates_destinations_before_refresh(warm_runtime_install):
    installer, attn, prepared = warm_runtime_install
    with pytest.raises(IncompleteRefit, match="tensor set differs"):
        installer.install_runtime_tensors({})
    with pytest.raises(IncompleteRefit, match="directly into live storage"):
        installer.install_runtime_tensors(
            {name: tensor.clone() for name, tensor in prepared.staged.tensors.items()}
        )
    assert attn._q_scale_float == 1.0
    assert attn._o_scale_float == 2.0
    assert attn.impl.bmm1_scale == 3.0


def test_warm_host_scale_refresh_requires_eager_execution(warm_runtime_install):
    installer, attn, _prepared = warm_runtime_install
    installer._vllm_config.model_config.enforce_eager = False
    with pytest.raises(RuntimeError, match="requires enforce_eager"):
        refresh_host_quantization_state(
            installer._model,
            installer._vllm_config,
            SimpleNamespace(),
            allow_warm=True,
        )
    assert attn._q_scale_float == 1.0
    assert attn._o_scale_float == 2.0


@pytest.mark.parametrize("install_path", ["tensors", "checkpoint"])
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("version", [
    "0.19.0", "0.10.1.1", "0.6.3.post1", "0.19.0.post1",
    "0.19.1rc1.dev12+g1a2b3c", "dev",
])
def test_installer_preserves_vllm_mla_refresh(
    monkeypatch, tmp_path, install_path, quantized, version,
):
    model = nn.Module()
    mla = nn.Module()
    mla.kv_b_proj = nn.Linear(1, 1, bias=False)
    mla.W_UV = torch.zeros(1)
    mla.W_UK_T = torch.zeros(1)
    model.add_module("mla", mla)
    originals = {name: getattr(mla, name) for name in ("W_UV", "W_UK_T")}
    pointers = {name: tensor.data_ptr() for name, tensor in originals.items()}

    _install_fake_vllm(monkeypatch, lambda _model: None)
    sys.modules["vllm.version"].__version__ = version
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]

    def finalize(target, _config):
        # Simulate vLLM's quantization-aware MLA post-load processing.
        value = target.mla.kv_b_proj.weight.detach().flatten()
        target.mla.W_UV = value + 10
        target.mla.W_UK_T = value + 20

    layerwise.finalize_layerwise_reload = finalize

    class DefaultModelLoader:
        def __init__(self, _load_config):
            pass

        def load_weights(self, target, _model_config):
            target.mla.kv_b_proj.weight.data.fill_(7)

    sys.modules[
        "vllm.model_executor.model_loader.default_loader"
    ].DefaultModelLoader = DefaultModelLoader
    synchronized = []
    monkeypatch.setattr(torch.cuda, "synchronize", synchronized.append)
    installer = _VllmInstaller(
        model=model,
        vllm_config=SimpleNamespace(
            quant_config=object() if quantized else None,
            load_config=SimpleNamespace(load_format="modelexpress"),
        ),
        model_config=SimpleNamespace(model="/launch", revision="main"),
        device=torch.device("cpu"),
    )

    if install_path == "tensors":
        installer.install_tensors({"mla.kv_b_proj.weight": torch.tensor([[7.0]])})
    else:
        installer.install_checkpoint(tmp_path)

    for name, expected in (("W_UV", 17), ("W_UK_T", 27)):
        actual = getattr(mla, name)
        assert actual is originals[name]
        assert actual.data_ptr() == pointers[name]
        assert actual.item() == expected
    assert synchronized == [torch.device("cpu")]


@pytest.mark.parametrize("stale_name", ["W_UV", "W_UK_T"])
def test_installer_rejects_missing_mla_refresh(monkeypatch, stale_name):
    _install_fake_vllm(monkeypatch, lambda _model: None)
    model = nn.Module()
    model.W_UV = torch.zeros(1)
    model.W_UK_T = torch.zeros(1)
    model.projection = nn.Parameter(torch.zeros(1))
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]

    def finalize(target, _config):
        for name in ("W_UV", "W_UK_T"):
            if name != stale_name:
                setattr(target, name, target.projection.detach().clone())

    layerwise.finalize_layerwise_reload = finalize
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )

    with pytest.raises(IncompleteRefit, match=rf"{stale_name} was not refreshed"):
        installer._reload(lambda: model.projection.data.fill_(7))


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


@pytest.mark.parametrize("when", ["touched", "elsewhere"])
def test_streaming_detects_retained_arena_storage_in_batch_or_at_the_end(
    monkeypatch, when
):
    """A loaded module retaining an arena view fails before the arena is
    refilled; a module outside the batch retaining one fails in the final sweep."""
    _install_fake_vllm(monkeypatch, lambda model: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False))
    names = frozenset(dict(model.named_parameters()))
    arena = torch.ones(2, 2)
    yielded = []

    def batches():
        if when == "touched":
            model[0].stash = arena.view(-1)
        yielded.append("0.weight")
        yield {"0.weight": arena}
        if when == "elsewhere":
            # Module 0 is not part of the second batch, so only the sweep
            # after the last batch can see this.
            model[0].stash = arena.view(-1)
        yielded.append("1.weight")
        yield {"1.weight": arena}

    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    with pytest.raises(IncompleteRefit, match="retained bounded staging storage"):
        installer.install_streaming(PreparedStreamingTensors(batches, names, {}))
    assert yielded == (["0.weight"] if when == "touched" else ["0.weight", "1.weight"])


def test_streaming_makes_exactly_two_live_walks_per_batch(monkeypatch):
    """Resolve-and-check against the live tree, then scan it. Nothing cached.

    Neither question survives being answered early: a hook can replace a module
    between batches, and it can add a Parameter to a module a later batch owns.
    So resolution and the complete-owner check share one walk, the retention
    scan takes another, and extra batches add exactly two each. Three would mean
    the shared walk split back apart; one would mean a live check was hoisted.
    """
    _install_fake_vllm(monkeypatch, lambda model: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    def walks_for(batch_size):
        model = nn.Sequential(*[nn.Linear(2, 2, bias=False) for _ in range(6)])
        names = frozenset(dict(model.named_parameters()))
        walks = []
        real_named_modules = model.named_modules

        def counted(*args, **kwargs):
            walks.append(1)
            return real_named_modules(*args, **kwargs)

        monkeypatch.setattr(model, "named_modules", counted)
        arena = torch.ones(2, 2)

        def batches():
            for start in range(0, 6, batch_size):
                yield {f"{i}.weight": arena for i in range(start, start + batch_size)}

        installer = _VllmInstaller(
            model=model,
            vllm_config=object(),
            model_config=object(),
            device=torch.device("cpu"),
        )
        metrics = {}
        installer.install_streaming(PreparedStreamingTensors(batches, names, metrics))
        assert "retention_scan_s" in metrics
        assert all(torch.equal(p, torch.ones(2, 2)) for p in model.parameters())
        return len(walks)

    # Reload setup and the final sweep are fixed per install. Six
    # single-parameter batches replace one six-parameter batch, so the only
    # admissible difference is five extra resolve-and-scan pairs.
    assert walks_for(1) - walks_for(6) == 2 * 5


@pytest.mark.parametrize("mode", ["consume_then_remove", "new_module"])
def test_streaming_rejects_cross_module_retention_before_arena_reuse(monkeypatch, mode):
    """A stash anywhere must be caught before the arena is refilled.

    Narrowing the per-batch scan to the modules a batch touched, and deferring
    the rest to a post-install sweep, is not equivalent. `consume_then_remove`
    stashes an arena view on an untouched module, lets the refill change it, has
    a later hook commit the changed value and delete the stash -- the sweep then
    finds nothing and the install silently reports success with wrong bytes.
    `new_module` stashes on a module created during installation, which a sweep
    over a cached module list never visits.
    """
    model = nn.Module()
    model.first = nn.Linear(1, 1, bias=False)
    model.second = nn.Linear(1, 1, bias=False)
    model.other = nn.Module()
    names = frozenset(dict(model.named_parameters()))

    class Info:
        def __init__(self, parameter):
            self.kernel_tensors = ({"weight": parameter}, {})

        def reset(self):
            self.kernel_tensors = None

    def initialize(target):
        for layer in (target.first, target.second):
            layerwise.LAYERWISE_INFO[layer] = Info(layer.weight)
            layer.weight = nn.Parameter(torch.empty_like(layer.weight, device="meta"))

    _install_fake_vllm(monkeypatch, initialize)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    quant_base = sys.modules[
        "vllm.model_executor.layers.quantization.base_config"
    ].QuantizeMethodBase

    def commit(layer, info):
        original = info.kernel_tensors[0]["weight"]
        original.data.copy_(layer.weight)
        layer.weight = original

    monkeypatch.setattr(layerwise, "_copy_and_restore_kernel_tensors", commit)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    class FirstHook(quant_base):
        def process_weights_after_loading(self, layer):
            if mode == "new_module":
                model.dynamic = nn.Module()
                model.dynamic.stash = layer.weight.detach()
            else:
                model.other.stash = layer.weight.detach()

    class SecondHook(quant_base):
        def process_weights_after_loading(self, layer):
            if mode == "consume_then_remove":
                layer.weight = nn.Parameter(layer.weight + model.other.stash)
                del model.other.stash

    model.first.quant_method = FirstHook()
    model.second.quant_method = SecondHook()
    arena = torch.ones(1, 1)
    refilled = []

    def batches():
        yield {"first.weight": arena}
        refilled.append(True)
        arena.fill_(2)
        yield {"second.weight": arena}

    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    with pytest.raises(IncompleteRefit, match="retained bounded staging storage"):
        installer.install_streaming(PreparedStreamingTensors(batches, names, {}))
    assert not refilled, "retention must be rejected before the arena is reused"


@pytest.mark.parametrize("cleanup_point", ["generator_finally", "iterator_close"])
def test_final_sweep_rejects_arena_exposed_by_producer_cleanup(
    monkeypatch, cleanup_point
):
    """The closing sweep must cover every arena the install used.

    A producer can expose an arena reference in its own cleanup -- a generator's
    `finally`, or an iterator's `close()` -- which runs after the last per-batch
    scan. Only the sweep can catch that, and only if it tests the union of every
    batch's storage rather than an empty set.
    """
    _install_fake_vllm(monkeypatch, lambda model: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    model = nn.Sequential(nn.Linear(2, 2, bias=False))
    names = frozenset(dict(model.named_parameters()))
    arena = torch.ones(2, 2)
    cleanup_calls = []

    def expose_arena():
        model[0].stash = arena.view(-1)
        cleanup_calls.append(cleanup_point)

    def generator():
        try:
            yield {"0.weight": arena}
        finally:
            expose_arena()

    class CloseableIterator:
        def __init__(self):
            self.yielded = False

        def __iter__(self):
            return self

        def __next__(self):
            if self.yielded:
                raise StopIteration
            self.yielded = True
            return {"0.weight": arena}

        def close(self):
            expose_arena()

    batches = generator if cleanup_point == "generator_finally" else CloseableIterator
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    with pytest.raises(IncompleteRefit, match="retained bounded staging storage"):
        installer.install_streaming(PreparedStreamingTensors(batches, names, {}))
    assert cleanup_calls == [cleanup_point]


def test_added_live_parameter_invalidates_captured_owner_completeness(monkeypatch):
    """Owner completeness is a live question, not a property of the capture.

    A hook can add a Parameter to a module a later batch owns, so a batch that
    covered its owner completely when the layout was captured no longer does by
    the time it arrives. Resolving only the supplied names cannot see that, so
    the check has to ask the live tree what the owner holds now -- and reject
    before the incomplete owner's hook runs.
    """
    model = nn.Module()
    model.first = nn.Linear(2, 2, bias=False)
    model.second = nn.Linear(2, 2, bias=False)
    names = frozenset(dict(model.named_parameters()))

    class Info:
        def __init__(self, parameter):
            self.kernel_tensors = ({"weight": parameter}, {})

        def reset(self):
            self.kernel_tensors = None

    def initialize(target):
        for layer in (target.first, target.second):
            layerwise.LAYERWISE_INFO[layer] = Info(layer.weight)
            layer.weight = nn.Parameter(torch.empty_like(layer.weight, device="meta"))

    _install_fake_vllm(monkeypatch, initialize)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    quant_base = sys.modules[
        "vllm.model_executor.layers.quantization.base_config"
    ].QuantizeMethodBase

    def commit(layer, info):
        original = info.kernel_tensors[0]["weight"]
        original.data.copy_(layer.weight)
        layer.weight = original

    monkeypatch.setattr(layerwise, "_copy_and_restore_kernel_tensors", commit)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    hook_calls = []

    class FirstHook(quant_base):
        def process_weights_after_loading(self, layer):
            # Independent storage, so this is a completeness question rather
            # than an arena-retention one.
            model.second.bias = nn.Parameter(torch.full((2,), 7.0))
            hook_calls.append("added_second_bias")

    class SecondHook(quant_base):
        def process_weights_after_loading(self, layer):
            hook_calls.append("processed_incomplete_second_owner")

    model.first.quant_method = FirstHook()
    model.second.quant_method = SecondHook()

    def batches():
        yield {"first.weight": torch.ones(2, 2)}
        yield {"second.weight": torch.full((2, 2), 2.0)}

    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    with pytest.raises(
        IncompleteRefit, match="streaming batch splits an owning module"
    ):
        installer.install_streaming(PreparedStreamingTensors(batches, names, {}))
    assert hook_calls == ["added_second_bias"]


@pytest.mark.parametrize("replace_installed_alias", [False, True])
def test_streaming_rejects_uninstalled_or_rebound_shared_bias(
    monkeypatch, replace_installed_alias
):
    _install_fake_vllm(monkeypatch, lambda model: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    model = nn.Module()
    model.gate = nn.Linear(2, 2)
    model.experts = nn.Linear(2, 2)
    model.experts.bias = model.gate.bias
    names = frozenset(dict(model.named_parameters()))

    def batches():
        if replace_installed_alias:
            yield {"gate.weight": torch.ones(2, 2), "gate.bias": torch.ones(2)}
            model.gate.bias = nn.Parameter(torch.full((2,), 9.0))
            model.experts.bias = model.gate.bias
        yield {"experts.weight": torch.ones(2, 2)}

    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    with pytest.raises(
        IncompleteRefit, match="missing canonical parameters=.*gate.bias"
    ):
        installer.install_streaming(PreparedStreamingTensors(batches, names, {}))


@pytest.mark.parametrize("owner", ["gate", "alias_owner"])
@pytest.mark.parametrize("mutation", ["replace", "remove"])
def test_alias_restoration_rejects_detached_owners(owner, mutation):
    model = nn.Module()
    model.gate = nn.Linear(2, 2)
    model.alias_owner = nn.Module()
    model.alias_owner.bias = model.gate.bias
    original = model.gate.bias
    installer = _VllmInstaller(
        model=model, vllm_config=object(), model_config=object(), device=torch.device("cpu")
    )
    aliases = installer._parameter_aliases(model)
    detached = getattr(model, owner)
    if mutation == "replace":
        replacement = nn.Module()
        replacement.bias = nn.Parameter(torch.full((2,), -9.0))
        setattr(model, owner, replacement)
    else:
        delattr(model, owner)

    with pytest.raises(IncompleteRefit, match=f"parameter alias owner '{owner}'"):
        installer._restore_parameter_aliases(aliases)

    assert detached.bias is original
    if mutation == "replace":
        assert torch.equal(getattr(model, owner).bias, torch.full((2,), -9.0))


@pytest.mark.parametrize("replace_alias_owner", [False, True])
def test_alias_restoration_tracks_each_path_to_a_shared_module(replace_alias_owner):
    model = nn.Module()
    model.a = nn.Linear(2, 2, bias=False)
    model.b = model.a
    original = model.a.weight
    installer = _VllmInstaller(
        model=model, vllm_config=object(), model_config=object(), device=torch.device("cpu")
    )
    aliases = installer._parameter_aliases(model)
    if replace_alias_owner:
        model.b = nn.Linear(2, 2, bias=False)
        replacement = model.b.weight
        with pytest.raises(IncompleteRefit, match="parameter alias owner 'b'"):
            installer._restore_parameter_aliases(aliases)
        assert model.b.weight is replacement
    else:
        installer._restore_parameter_aliases(aliases)
        assert model.b.weight is original
    assert model.a.weight is original


@pytest.mark.parametrize("one_batch", [False, True])
def test_streaming_managed_shared_bias_is_reattached_before_dependent_hook(
    monkeypatch, one_batch
):
    model = nn.Module()
    model.gate = nn.Linear(2, 2)
    model.experts = nn.Linear(2, 2)
    model.experts.bias = model.gate.bias
    originals = dict(model.named_parameters())
    names = frozenset(originals)

    class Info:
        def __init__(self, layer):
            self.kernel_tensors = (dict(layer.named_parameters(recurse=False)), {})

        def reset(self):
            self.kernel_tensors = None

    def initialize(target):
        for layer in (target.gate, target.experts):
            layerwise.LAYERWISE_INFO[layer] = Info(layer)
            for name, parameter in list(layer.named_parameters(recurse=False)):
                setattr(
                    layer,
                    name,
                    nn.Parameter(torch.empty_like(parameter, device="meta")),
                )

    _install_fake_vllm(monkeypatch, initialize)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    quant_base = sys.modules[
        "vllm.model_executor.layers.quantization.base_config"
    ].QuantizeMethodBase

    def commit(layer, info):
        for name, original in info.kernel_tensors[0].items():
            original.data.copy_(getattr(layer, name))
            setattr(layer, name, original)

    monkeypatch.setattr(layerwise, "_copy_and_restore_kernel_tensors", commit)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    hooks = []

    class ExpertsHook(quant_base):
        def process_weights_after_loading(self, layer):
            assert layer.bias is model.gate.bias
            assert torch.equal(layer.bias, torch.full((2,), 4.0))
            hooks.append("updated_shared_bias")

    model.experts.quant_method = ExpertsHook()

    def batches():
        gate = {
            "gate.weight": torch.full((2, 2), 3.0),
            "gate.bias": torch.full((2,), 4.0),
        }
        experts = {"experts.weight": torch.full((2, 2), 5.0)}
        if one_batch:
            yield {**gate, **experts}
        else:
            yield gate
            yield experts

    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    installer.install_streaming(PreparedStreamingTensors(batches, names, {}))
    assert hooks == ["updated_shared_bias"]
    assert model.experts.bias is model.gate.bias
    assert all(
        parameter is originals[name] for name, parameter in model.named_parameters()
    )


def test_capture_key_unwraps_reload_loaders_but_guards_original_and_receiver(
    monkeypatch,
):
    _install_fake_vllm(monkeypatch, lambda model: None)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]

    def default_loader(parameter, value):
        parameter.copy_(value)

    def original_loader(parameter):
        loader = getattr(parameter, "weight_loader", default_loader)
        while loader.__name__ == "online_process_loader":
            loader = loader.__wrapped__
        return loader

    layerwise._get_original_loader = original_loader
    model = nn.Linear(2, 2, bias=False)
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    manifest = [("weight", torch.float32, (2, 2))]
    assert not hasattr(model.weight, "weight_loader")
    key = installer._capture_key(manifest)
    model.weight.weight_loader = default_loader
    assert installer._capture_key(manifest) == key
    underlying = model.weight.weight_loader

    def wrapped(loader):
        def online_process_loader(*args, **kwargs):
            return loader(*args, **kwargs)

        online_process_loader.__wrapped__ = loader
        return online_process_loader

    for _ in range(2):
        model.weight.weight_loader = wrapped(wrapped(underlying))
        assert installer._capture_key(manifest) == key
    model.weight.weight_loader = wrapped(lambda parameter, value: parameter.add_(value))
    assert installer._capture_key(manifest) != key

    def ordinary_wrapper(*args, **kwargs):
        return underlying(*args, **kwargs)

    ordinary_wrapper.__wrapped__ = underlying
    model.weight.weight_loader = ordinary_wrapper
    assert installer._capture_key(manifest) != key

    class Loader:
        def load(self, parameter, value):
            parameter.copy_(value)

    first, second = Loader(), Loader()
    model.weight.weight_loader = first.load
    key = installer._capture_key(manifest)
    model.weight.weight_loader = wrapped(first.load)
    assert installer._capture_key(manifest) == key
    model.weight.weight_loader = second.load
    assert installer._capture_key(manifest) != key

    resolved_key = installer._capture_key(manifest)
    del layerwise._get_original_loader
    assert installer._capture_key(manifest) == resolved_key


@pytest.mark.parametrize("missing", ["module", "symbol"])
def test_capture_key_reports_missing_layerwise_api_before_mutation(
    monkeypatch, missing
):
    _install_fake_vllm(monkeypatch, lambda model: None)
    if missing == "module":
        monkeypatch.setitem(
            sys.modules, "vllm.model_executor.model_loader.reload.layerwise", None
        )
    model = nn.Linear(2, 2, bias=False)
    parameter = model.weight
    before = parameter.detach().clone()
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )

    with pytest.raises(
        RuntimeError, match="requires vLLM's layerwise reload APIs"
    ) as raised:
        installer._capture_key([("weight", torch.float32, (2, 2))])

    assert isinstance(raised.value.__cause__, ImportError)
    assert model.weight is parameter
    assert torch.equal(model.weight, before)


def test_layerwise_capture_cache_and_streaming_preserve_tied_parameters(monkeypatch):
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(2, 2, bias=False)
            self.lm_head = nn.Linear(2, 2, bias=False)
            self.lm_head.weight = self.embedding.weight
            self.register_buffer("routing", torch.tensor([0, 1]))
            self.capture_calls = 0

        def load_weights(self, weights):
            self.capture_calls += 1
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
    manifest = [("embedding.weight", torch.float32, (2, 2))]
    cached, _ = installer.capture(manifest)
    assert model.capture_calls == 1
    cached.copies.clear()
    assert installer.capture(manifest)[0].copies
    model.routing.add_(1)
    installer.capture(manifest)
    assert model.capture_calls == 2
    with torch.inference_mode():
        model.routing = torch.tensor([3, 4])
        installer.capture(manifest)
        assert model.capture_calls == 3
        model.routing.add_(1)
        installer.capture(manifest)
        assert model.capture_calls == 4

    for value in (7.0, 11.0, -3.0):

        def batches(value=value):
            yield {"embedding.weight": torch.full((2, 2), value)}

        prepared = PreparedStreamingTensors(batches, frozenset(layout), {})
        installer.install_streaming(prepared)
        assert model.embedding.weight is model.lm_head.weight
        assert model.embedding.weight.data_ptr() == address
        assert torch.equal(model.lm_head.weight, torch.full((2, 2), value))
        assert prepared.transfer_metrics["retention_batch_scans"] == 1
        assert prepared.transfer_metrics["retention_final_scans"] == 1
        installer.capture(manifest)
        assert model.capture_calls == 4

class _Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))
        self.child = nn.Linear(2, 2, bias=False)


@pytest.mark.parametrize("packed", [False, True])
def test_hook_replacing_a_submodule_never_leaves_the_live_child_stale(monkeypatch, packed):
    """A parent's post-load hook may replace its own submodule.

    Unpacked, the child is its own batch and is resolved against the live tree
    after the parent's hook ran. Packing puts parent and child into one batch,
    whose modules are resolved once before the parent's hook runs. Either the
    live child receives the published bytes or the install is rejected; it
    must never succeed with the bytes committed into a detached module.
    """
    model = nn.Module()
    model.layer = _Parent()
    names = frozenset(dict(model.named_parameters()))

    class Info:
        def __init__(self, layer):
            self.kernel_tensors = (dict(layer.named_parameters(recurse=False)), {})

        def reset(self):
            self.kernel_tensors = None

    def initialize(target):
        for layer in (target.layer, target.layer.child):
            layerwise.LAYERWISE_INFO[layer] = Info(layer)
            for name, parameter in list(layer.named_parameters(recurse=False)):
                setattr(layer, name, nn.Parameter(torch.empty_like(parameter, device="meta")))

    _install_fake_vllm(monkeypatch, initialize)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    quant_base = sys.modules["vllm.model_executor.layers.quantization.base_config"].QuantizeMethodBase

    def commit(layer, info):
        for name, original in info.kernel_tensors[0].items():
            original.data.copy_(getattr(layer, name))
            setattr(layer, name, original)

    monkeypatch.setattr(layerwise, "_copy_and_restore_kernel_tensors", commit)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)

    class ReplacesChild(quant_base):
        def process_weights_after_loading(self, layer):
            layer.child = nn.Linear(2, 2, bias=False)

    model.layer.quant_method = ReplacesChild()
    parent = {"layer.weight": torch.full((2, 2), 1.0)}
    child = {"layer.child.weight": torch.full((2, 2), 2.0)}

    def batches():
        if packed:
            yield {**parent, **child}
        else:
            yield parent
            yield child

    installer = _VllmInstaller(
        model=model, vllm_config=object(), model_config=object(), device=torch.device("cpu")
    )
    try:
        installer.install_streaming(PreparedStreamingTensors(batches, names, {}))
    except IncompleteRefit:
        return
    assert torch.equal(model.layer.child.weight, child["layer.child.weight"]), (
        "install succeeded but the live child never received its published bytes"
    )


def test_streaming_install_error_is_not_replaced_by_a_failed_prefetch_drain(monkeypatch):
    """install_streaming abandons the transfer with close(), not throw().

    The generator therefore sees GeneratorExit even while an install error is
    propagating, so any drain policy that keys on how the generator was left
    has to be applied by the consumer, which is the only side that knows.
    """
    import ctypes

    import modelexpress_rl.inference.nixl_staged_transfer as transfer_module
    from modelexpress.refit.reshard.slice_plan import Shard
    from modelexpress.refit.reshard.transfer_plan import SourceInfo
    from modelexpress.refit.reshard.types import CaptureResult, RecordedCopy

    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    _install_fake_vllm(monkeypatch, lambda model: None)

    source_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    source = SourceInfo(
        global_shape=(4,),
        dtype=torch.float32,
        elsize=4,
        shards=[Shard((0,), (4,), "source", source_tensor.data_ptr(), 4)],
    )
    copies = [
        RecordedCopy(
            src_name="w",
            op_chain=(),
            param_name=name,
            dest_offset=0,
            dest_shape=(4,),
            dest_stride=(1,),
            dest_dtype=torch.float32,
        )
        for name in ("a.weight", "b.weight")
    ]
    layout = {c.param_name: (c.dest_shape, c.dest_dtype) for c in copies}
    planned = transfer_module._bounded_batches(
        CaptureResult(copies=copies), layout, {"w": source}, 512
    )
    drained = []

    class Transport:
        def post_reads(self, descriptors):
            return descriptors

        def await_reads(self, posted):
            if drained:
                raise RuntimeError("injected prefetch drain failure")
            drained.append(True)
            for d in posted:
                ctypes.memmove(d.dst_addr, d.src_addr, d.nbytes)

    prepared = transfer_module._PreparedBoundedTransfer(planned, {"w": source}, Transport())
    transfer = object.__new__(transfer_module._NixlStagedTransfer)
    transfer._closed = False
    transfer._active = prepared
    transfer._device = torch.device("cpu")
    transfer._device_id = 0
    transfer._staging_arenas = [torch.empty(512, dtype=torch.uint8) for _ in range(2)]

    class Owner(nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4, dtype=dtype))

    model = nn.Module()
    model.a = Owner(torch.float64)  # the first batch cannot be committed
    model.b = Owner(torch.float32)
    names = frozenset(dict(model.named_parameters()))
    installer = _VllmInstaller(
        model=model, vllm_config=object(), model_config=object(), device=torch.device("cpu")
    )
    with pytest.raises(IncompleteRefit, match="no compatible live storage"):
        installer.install_streaming(
            PreparedStreamingTensors(lambda: transfer.iter_bounded(prepared, {}), names, {})
        )
