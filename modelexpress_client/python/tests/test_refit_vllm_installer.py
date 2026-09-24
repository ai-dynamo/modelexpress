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
from modelexpress_rl.inference.engines.vllm.installer import (
    _copy_received,
    _standard_module_access,
    _standard_parameter_writes,
    _VllmInstaller,
)
from modelexpress_rl.inference.plan import (
    PreparedCheckpointArtifact,
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
        "vllm.model_executor.layers.attention": ModuleType(
            "vllm.model_executor.layers.attention"
        ),
        "vllm.model_executor.model_loader.reload.meta": ModuleType(
            "vllm.model_executor.model_loader.reload.meta"
        ),
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

    def initialize_with_metadata(model):
        initialize(model)
        for layer in model.modules():
            if layer not in layerwise.LAYERWISE_INFO and any(
                tensor.is_meta for tensor in layer.parameters(recurse=False)
            ):
                layerwise.LAYERWISE_INFO[layer] = SimpleNamespace(
                    kernel_tensors=None,
                    reset=lambda: None,
                )

    def materialize_layer(layer, _info):
        for name, value in (*layer._parameters.items(), *layer._buffers.items()):
            if value is None or not value.is_meta:
                continue
            tensor = torch.empty_strided(value.shape, value.stride(), dtype=value.dtype)
            if name in layer._parameters:
                tensor = nn.Parameter(tensor, requires_grad=False)
            tensor.__dict__.update(value.__dict__)
            setattr(layer, name, tensor)

    modules[
        "vllm.model_executor.model_loader.reload.meta"
    ].materialize_layer = materialize_layer
    modules["vllm.model_executor.layers.attention"].is_deferred_attention_layer = (
        lambda layer: getattr(layer, "deferred_attention", False)
    )
    layerwise.initialize_layerwise_reload = initialize_with_metadata
    layerwise.finalize_layerwise_reload = lambda _model, _config: None
    layerwise._copy_and_restore_kernel_tensors = lambda _layer, _info: None

    def process(layer, info):
        materialize_layer(layer, info)
        quant_method = getattr(layer, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            quant_method.process_weights_after_loading(layer)
            if hasattr(layer, "update_param_tp_status"):
                layer.update_param_tp_status()
        layerwise._copy_and_restore_kernel_tensors(layer, info)
        info.reset()

    layerwise._layerwise_process = process
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
        target.child = nn.Module()
        target.child.register_parameter(
            "orphan", nn.Parameter(torch.empty(1, device="meta"))
        )

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


def test_bounded_install_reports_transfer_and_apply_without_install_only_metric(
    monkeypatch,
):
    model = nn.Linear(2, 2, bias=False)
    values = torch.full_like(model.weight, 7.0)
    transfer_metrics = {}

    def batches():
        transfer_metrics.update(wire_s=0.25, bytes_received=16.0)
        yield {"weight": values}

    source = PreparedStreamingTensors(batches, frozenset({"weight"}), transfer_metrics)
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

    installer._reload(lambda _aliases: setattr(model, "workspace", torch.tensor([7.0])))

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
        received = {name: tensor.clone() for name, tensor in attn.named_buffers()}
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
    warm_runtime_install,
    attribute,
    value,
    message,
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
def test_installer_preserves_engine_derived_storage(
    monkeypatch,
    tmp_path,
    install_path,
    quantized,
):
    model = nn.Module()
    attention = nn.Module()
    attention.projection = nn.Linear(1, 1, bias=False)
    attention.derived_left = torch.zeros(1)
    attention.derived_right = torch.zeros(1)
    model.add_module("attention", attention)
    originals = {
        name: getattr(attention, name) for name in ("derived_left", "derived_right")
    }
    pointers = {name: tensor.data_ptr() for name, tensor in originals.items()}

    _install_fake_vllm(monkeypatch, lambda _model: None)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]

    def finalize(target, _config):
        # Simulate the engine's deferred post-load processing.
        value = target.attention.projection.weight.detach().flatten()
        target.attention.derived_left = value + 10
        target.attention.derived_right = value + 20

    layerwise.finalize_layerwise_reload = finalize

    class DefaultModelLoader:
        def __init__(self, _load_config):
            pass

        def load_weights(self, target, _model_config):
            target.attention.projection.weight.data.fill_(7)

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
        installer.install_tensors(
            {"attention.projection.weight": torch.tensor([[7.0]])}
        )
    else:
        installer.install_checkpoint(tmp_path)

    for name, expected in (("derived_left", 17), ("derived_right", 27)):
        actual = getattr(attention, name)
        assert actual is originals[name]
        assert actual.data_ptr() == pointers[name]
        assert actual.item() == expected
    assert synchronized == [torch.device("cpu")]


@pytest.mark.parametrize("change", ["missing", "shape", "dtype", "device", "storage"])
def test_reload_rejects_incompatible_graph_bound_tensor(monkeypatch, change):
    _install_fake_vllm(monkeypatch, lambda _model: None)
    model = nn.Module()
    model.derived = torch.zeros(2)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]

    def finalize(target, _config):
        if change == "missing":
            del target.derived
        elif change == "shape":
            target.derived = torch.ones(3)
        elif change == "dtype":
            target.derived = torch.ones(2, dtype=torch.float64)
        elif change == "device":
            target.derived = torch.empty(2, device="meta")
        else:
            target.derived.data = torch.ones(2)

    layerwise.finalize_layerwise_reload = finalize
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    with pytest.raises(IncompleteRefit, match="graph-bound tensor.*changed geometry"):
        installer._reload(lambda _aliases: None)


def test_reload_accepts_in_place_engine_post_load(monkeypatch):
    _install_fake_vllm(monkeypatch, lambda _model: None)
    model = nn.Module()
    model.derived = torch.zeros(2)
    original = model.derived
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    layerwise.finalize_layerwise_reload = lambda target, _config: target.derived.fill_(
        7
    )
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    installer._reload(lambda _aliases: None)
    assert model.derived is original
    assert torch.equal(model.derived, torch.full((2,), 7.0))


@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
@pytest.mark.parametrize("representation", ["tensor", "scalar"])
def test_reload_allows_native_host_bookkeeping_changes(
    monkeypatch, device, representation
):
    _install_fake_vllm(monkeypatch, lambda _model: None)
    model = nn.Module()
    model.host_helper = torch.tensor(1.0)
    original = model.host_helper
    observed_devices = []
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]

    def finalize(target, _config):
        # Native hooks allocate host bookkeeping without an explicit device.
        value = torch.tensor(7.0)
        observed_devices.append(value.device)
        target.host_helper = value if representation == "tensor" else value.item()

    layerwise.finalize_layerwise_reload = finalize
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device(device),
    )

    installer._reload(lambda _aliases: None)

    assert observed_devices == [torch.device("cpu")]
    assert model.host_helper is not original
    assert model.host_helper == 7.0
    assert isinstance(model.host_helper, torch.Tensor) == (representation == "tensor")


@pytest.mark.parametrize("override", ["copy", "torch_function"])
def test_receive_copy_does_not_expose_source_to_parameter_dispatch(override):
    observed = []

    class Parameter(nn.Parameter):
        def copy_(self, source, *args, **kwargs):
            observed.append(source)
            return super().copy_(source, *args, **kwargs)

        @classmethod
        def __torch_function__(cls, function, types, args=(), kwargs=None):
            if override == "torch_function" and function.__name__ == "copy_":
                observed.append(args[1])
            with torch._C.DisableTorchFunctionSubclass():
                return function(*args, **(kwargs or {}))

    target = Parameter(torch.zeros(2), requires_grad=False)
    source = torch.tensor([7.0, 11.0])
    _copy_received(target, source)
    assert not observed
    assert torch.equal(target, source)
    source.fill_(99)
    assert torch.equal(target, torch.tensor([7.0, 11.0]))


@pytest.mark.parametrize("kind", ["function", "dispatch"])
def test_receive_copy_rejects_ambient_dispatch_before_exposing_source(kind):
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    observed = []

    class FunctionMode(TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            observed.append(function)
            return function(*args, **(kwargs or {}))

    class DispatchMode(TorchDispatchMode):
        def __torch_dispatch__(self, function, types, args=(), kwargs=None):
            observed.append(function)
            return function(*args, **(kwargs or {}))

    target = nn.Parameter(torch.zeros(2), requires_grad=False)
    source = torch.ones(2)
    mode = FunctionMode() if kind == "function" else DispatchMode()
    with mode, pytest.raises(IncompleteRefit, match="active tensor dispatch modes"):
        _copy_received(target, source)
    assert not observed
    assert torch.equal(target, torch.zeros(2))


def test_materialization_preserves_intra_owner_alias_before_post_load(monkeypatch):
    model = nn.Module()
    model.weight = nn.Parameter(torch.zeros(2))
    model.tied = model.weight
    original = model.weight

    class Info:
        def __init__(self):
            self.kernel_tensors = ({"weight": original, "tied": original}, {})
            self.loaded_weights = []

        def reset(self):
            self.kernel_tensors = None

    def initialize(target):
        layerwise.LAYERWISE_INFO[target] = Info()
        target.weight = target.tied = nn.Parameter(torch.empty(2, device="meta"))

    _install_fake_vllm(monkeypatch, initialize)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    quant_base = sys.modules[
        "vllm.model_executor.layers.quantization.base_config"
    ].QuantizeMethodBase
    calls = []

    class Hook(quant_base):
        def process_weights_after_loading(self, layer):
            assert layer.tied is layer.weight
            assert torch.equal(layer.tied, torch.tensor([7.0, 11.0]))
            calls.append("post_load")

    def commit(layer, info):
        original.data.copy_(layer.weight)
        layer.weight = layer.tied = original

    model.quant_method = Hook()
    layerwise._copy_and_restore_kernel_tensors = commit
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    source = torch.tensor([7.0, 11.0])
    installer.install_streaming(
        PreparedStreamingTensors(
            lambda: iter([{"weight": source}]), frozenset({"weight"}), {}
        )
    )
    assert calls == ["post_load"]
    assert model.weight is model.tied is original
    assert torch.equal(model.weight, source)


def test_streaming_accepts_independent_empty_tensors(monkeypatch):
    _install_fake_vllm(monkeypatch, lambda _model: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    model = nn.Module()
    model.weight = nn.Parameter(torch.empty(0))
    source = torch.empty(0)
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    prepared = PreparedStreamingTensors(
        lambda: iter([{"weight": source}]), frozenset({"weight"}), {}
    )
    installer.install_streaming(prepared)
    assert model.weight.numel() == 0
    assert prepared.ownership.iterator is None
    assert not prepared.ownership.release_blocked


def test_streaming_rejects_aliasing_source_before_any_batch_copy(monkeypatch):
    _install_fake_vllm(monkeypatch, lambda _model: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    model = nn.Module()
    model.first = nn.Parameter(torch.zeros(2))
    model.second = nn.Parameter(torch.zeros(2))
    source = {"first": torch.ones(2), "second": model.second.detach()}
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    prepared = PreparedStreamingTensors(lambda: iter([source]), frozenset(source), {})
    with pytest.raises(IncompleteRefit, match="aliases live storage"):
        installer.install_streaming(prepared)
    assert torch.equal(model.first, torch.zeros(2))
    assert prepared.ownership.iterator is None
    assert not prepared.ownership.release_blocked


def test_streaming_uses_engine_processing_and_defers_attention(monkeypatch):
    model = nn.Module()
    model.first = nn.Linear(2, 2, bias=False)
    model.attention = nn.Module()
    model.attention.scale = nn.Parameter(torch.zeros(1))
    model.attention.deferred_attention = True
    model.attention.derived = torch.zeros(1)
    originals = dict(model.named_parameters())
    derived = model.attention.derived
    events = []
    retained = []
    arena = torch.empty(4)

    class Info:
        def __init__(self, layer):
            self.kernel_tensors = (dict(layer.named_parameters(recurse=False)), {})
            self.loaded_weights = []

        def reset(self):
            self.kernel_tensors = None

    def initialize(target):
        for layer in (target.first, target.attention):
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

    class Hook(quant_base):
        def process_weights_after_loading(self, layer):
            assert (
                layer.weight.untyped_storage().data_ptr()
                != arena.untyped_storage().data_ptr()
            )
            retained.append(layer.weight.detach())
            events.append("post_load")

    model.first.quant_method = Hook()
    model.first.update_param_tp_status = lambda: events.append("tp")

    def commit(layer, info):
        for name, original in info.kernel_tensors[0].items():
            original.data.copy_(getattr(layer, name))
            setattr(layer, name, original)
        events.append("commit_first" if layer is model.first else "commit_attention")

    def finalize(target, _config):
        assert layerwise.LAYERWISE_INFO[target.first].kernel_tensors is None
        info = layerwise.LAYERWISE_INFO[target.attention]
        assert info.kernel_tensors is not None
        target.attention.derived = (
            target.first.weight.sum().reshape(1) + target.attention.scale
        )
        info.reset()
        events.append("finalize")

    layerwise._copy_and_restore_kernel_tensors = commit
    layerwise.finalize_layerwise_reload = finalize
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    for value in (1, 3, 5):
        events.clear()

        def batches(value=value):
            arena.fill_(value)
            yield {"first.weight": arena.view(2, 2)}
            arena.fill_(value + 1)
            yield {"attention.scale": arena[:1]}
            arena.fill_(99)

        prepared = PreparedStreamingTensors(batches, frozenset(originals), {})
        recorder = RefitTimingRecorder(backend="test", version=str(value))
        with use_refit_timing(recorder):
            installer.install_streaming(prepared)
        metadata = recorder.as_dict()["stages"]["installation"]["metadata"]
        for key in ("materialization_s", "receive_copy_s", "post_load_processing_s"):
            assert metadata[key] >= 0
        assert events == [
            "post_load",
            "tp",
            "commit_first",
            "commit_attention",
            "finalize",
        ]
        assert prepared.ownership.iterator is None
        assert not prepared.ownership.release_blocked
        assert all(
            parameter is originals[name] for name, parameter in model.named_parameters()
        )
        assert model.attention.derived is derived
        assert model.attention.derived.item() == 5 * value + 1
    assert [tensor[0, 0].item() for tensor in retained] == [1, 3, 5]


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


def test_streaming_walks_live_owners_without_arena_scans(monkeypatch):
    """Keep owner validation live without recursively scanning tensor contents."""
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
        assert "retention_scan_s" not in metrics
        assert all(torch.equal(p, torch.ones(2, 2)) for p in model.parameters())
        return len(walks)

    # Extra batches only repeat the owner/completeness walk.
    assert walks_for(1) - walks_for(6) == 5


@pytest.mark.parametrize("mode", ["consume_then_remove", "new_module"])
def test_streaming_hooks_retain_owned_values_across_arena_reuse(monkeypatch, mode):
    """Post-load callbacks may retain engine values without retaining the arena."""
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
    installer.install_streaming(PreparedStreamingTensors(batches, names, {}))
    assert refilled
    assert model.first.weight.item() == 1
    if mode == "consume_then_remove":
        assert model.second.weight.item() == 3
    else:
        assert model.second.weight.item() == 2
        assert model.dynamic.stash.item() == 1
        assert (
            model.dynamic.stash.untyped_storage().data_ptr()
            != arena.untyped_storage().data_ptr()
        )


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
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
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
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
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


@pytest.mark.parametrize("helper", ["module_access", "parameter_writes"])
@pytest.mark.parametrize("field", ["_parameters", "_buffers", "_modules"])
@pytest.mark.parametrize("failure", [KeyError, StopIteration])
def test_alias_lookup_failures_preserve_order_and_retry_classification(
    helper, field, failure
):
    calls = []

    class LookupDict(dict):
        def get(self, name, default=None):
            calls.append(name)
            if name == field:
                raise failure("lookup failed")
            return super().get(name, default)

    attributes = LookupDict(_parameters={}, _buffers={}, _modules={})

    class Owner(nn.Module):
        @property
        def __dict__(self):
            return attributes

    owner = Owner()
    expected = RuntimeError if failure is StopIteration else failure
    with pytest.raises(expected) as captured:
        if helper == "module_access":
            _standard_module_access(owner, "child", set(), set())
        else:
            aliases = SimpleNamespace(writers=((owner, type(owner)),), attributes=())
            _standard_parameter_writes(aliases)
    fields = ["_parameters", "_buffers", "_modules"]
    assert calls == fields[: fields.index(field) + 1]
    if failure is StopIteration:
        assert isinstance(captured.value.__cause__, StopIteration)


@pytest.mark.parametrize("parameter_count", [1, 128])
def test_alias_validation_deduplicates_edges_but_checks_every_boundary(
    monkeypatch, parameter_count
):
    model = nn.Module()
    model.left = nn.Module()
    model.left.leaf = nn.Module()
    for index in range(parameter_count):
        model.left.leaf.register_parameter(
            f"weight_{index}", nn.Parameter(torch.zeros(1))
        )
    model.right = model.left
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    aliases = installer._parameter_aliases(model)
    reads = []
    original_getattr = getattr

    def counted(module, name, *default):
        if isinstance(module, nn.Module):
            reads.append((id(module), name))
        return original_getattr(module, name, *default)

    monkeypatch.setattr(
        sys.modules[_VllmInstaller.__module__], "getattr", counted, raising=False
    )
    installer._validate_alias_owners(aliases)
    installer._validate_alias_owners(aliases)
    expected = [(id(model), "left"), (id(model.left), "leaf"), (id(model), "right")]
    assert reads == expected * 2


def test_alias_restoration_writes_only_distinct_changed_slots(monkeypatch):
    writes = []
    model = nn.Module()
    model.a = nn.Module()
    model.a.weight = nn.Parameter(torch.ones(2))
    model.same_owner = model.a
    model.b = nn.Module()
    model.b.weight = model.a.weight
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    aliases = installer._parameter_aliases(model)
    original_setattr = setattr

    def counted(module, name, value):
        writes.append((id(module), value))
        original_setattr(module, name, value)

    monkeypatch.setattr(
        sys.modules[_VllmInstaller.__module__], "setattr", counted, raising=False
    )
    installer._restore_parameter_aliases(aliases)
    assert writes == []
    model.b.weight = nn.Parameter(torch.zeros(2))
    writes.clear()
    installer._restore_parameter_aliases(aliases)
    assert writes == [(id(model.b), model.a.weight)]
    assert model.same_owner.weight is model.b.weight is model.a.weight


@pytest.mark.parametrize(
    "custom",
    [
        "setattr",
        "register_parameter",
        "getattr",
        "getattribute",
        "descriptor",
        "instance_registration",
        "global_hook",
    ],
)
def test_alias_restoration_preserves_custom_parameter_assignment(monkeypatch, custom):
    class Layer(nn.Module):
        pass

    model = nn.Module()
    model.a = Layer()
    model.a.weight = nn.Parameter(torch.ones(1))
    model.same_owner = model.a
    model.b = Layer()
    model.b.weight = model.a.weight
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    aliases = installer._parameter_aliases(model)
    writes = []
    original_setattr = setattr

    def counted(module, name, value):
        writes.append((id(module), name))
        original_setattr(module, name, value)

    monkeypatch.setattr(
        sys.modules[_VllmInstaller.__module__], "setattr", counted, raising=False
    )
    if custom == "setattr":
        monkeypatch.setattr(
            Layer,
            "__setattr__",
            lambda self, name, value: nn.Module.__setattr__(self, name, value),
        )
    elif custom == "register_parameter":
        monkeypatch.setattr(
            Layer,
            "register_parameter",
            lambda self, name, value: nn.Module.register_parameter(self, name, value),
        )
    elif custom == "getattr":
        monkeypatch.setattr(
            Layer, "__getattr__", lambda self, name: nn.Module.__getattr__(self, name)
        )
    elif custom == "getattribute":
        monkeypatch.setattr(
            Layer,
            "__getattribute__",
            lambda self, name: object.__getattribute__(self, name),
        )
    elif custom == "descriptor":
        monkeypatch.setattr(
            Layer,
            "weight",
            property(lambda self: self._parameters["weight"]),
            raising=False,
        )
    elif custom == "instance_registration":
        model.a.register_parameter = lambda name, value: nn.Module.register_parameter(
            model.a, name, value
        )
    else:
        from torch.nn.modules.module import _global_parameter_registration_hooks

        monkeypatch.setitem(
            _global_parameter_registration_hooks,
            "test_alias",
            lambda module, name, value: value,
        )
    installer._restore_parameter_aliases(aliases)
    assert writes == [(id(model.a), "weight"), (id(model.b), "weight")]


@pytest.mark.parametrize("mutation", ["remove", "replace", "root"])
def test_alias_validation_guards_all_ancestors_before_using_cached_owner(mutation):
    model = nn.Module()
    model.branch = nn.Module()
    model.branch.leaf = nn.Linear(1, 1, bias=False)
    model.alias = model.branch.leaf
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    aliases = installer._parameter_aliases(model)
    if mutation == "remove":
        del model.branch
    elif mutation == "replace":
        replacement = nn.Module()
        replacement.leaf = model.alias
        model.branch = replacement
    else:
        replacement = nn.Module()
        replacement.branch = model.branch
        replacement.alias = model.alias
        installer._model = replacement
    with pytest.raises(IncompleteRefit, match="parameter alias (owner 'branch'|root)"):
        installer._restore_parameter_aliases(aliases)


def test_custom_get_submodule_preserves_repeated_path_fallback():
    calls = []

    class Model(nn.Module):
        def get_submodule(self, path):
            calls.append(path)
            return super().get_submodule(path)

    model = Model()
    model.a = nn.Linear(1, 1)
    model.b = model.a
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    aliases = installer._parameter_aliases(model)
    installer._restore_parameter_aliases(aliases)
    assert calls == ["a", "b", "a", "b"]


def test_base_module_get_submodule_override_uses_original_path_validation(monkeypatch):
    model = nn.Module()
    model.a = nn.Linear(1, 1, bias=False)
    model.b = model.a
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    aliases = installer._parameter_aliases(model)
    replacement = nn.Linear(1, 1, bias=False)
    monkeypatch.setattr(nn.Module, "get_submodule", lambda self, path: replacement)
    with pytest.raises(IncompleteRefit, match="parameter alias owner 'a'"):
        installer._validate_alias_owners(aliases)


@pytest.mark.parametrize(
    "method", ["__getattr__", "__getattribute__", "__setattr__", "register_parameter"]
)
def test_base_module_override_preserves_original_assignment_behavior(
    monkeypatch, method
):
    model = nn.Module()
    model.a = nn.Linear(1, 1, bias=False)
    model.b = model.a
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    aliases = installer._parameter_aliases(model)
    original_method = getattr(nn.Module, method)
    monkeypatch.setattr(
        nn.Module, method, lambda self, *args: original_method(self, *args)
    )
    writes = []
    original_setattr = setattr

    def counted(module, name, value):
        writes.append((id(module), name))
        original_setattr(module, name, value)

    monkeypatch.setattr(
        sys.modules[_VllmInstaller.__module__], "setattr", counted, raising=False
    )
    installer._restore_parameter_aliases(aliases)
    assert writes == [(id(model.a), "weight")]


@pytest.mark.parametrize("invalid_traversal", ["conflicting_path", "missing_prefix"])
def test_alias_validation_falls_back_for_nonstandard_module_traversal(
    monkeypatch, invalid_traversal
):
    model = nn.Module()
    model.branch = nn.Module()
    model.branch.leaf = nn.Linear(1, 1, bias=False)
    model.alias = model.branch.leaf
    leaf = model.branch.leaf
    if invalid_traversal == "conflicting_path":
        detached = nn.Linear(1, 1, bias=False)
        entries = [
            ("", model),
            ("branch", model.branch),
            ("branch.leaf", detached),
            ("alias", detached),
            ("branch.leaf", leaf),
            ("alias", leaf),
        ]
    else:
        entries = [("branch.leaf", leaf), ("alias", leaf)]
    monkeypatch.setattr(model, "named_modules", lambda **kwargs: iter(entries))
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    aliases = installer._parameter_aliases(model)
    if invalid_traversal == "conflicting_path":
        with pytest.raises(
            IncompleteRefit, match="parameter alias owner 'branch.leaf'"
        ):
            installer._restore_parameter_aliases(aliases)
    else:
        installer._restore_parameter_aliases(aliases)
        assert model.branch.leaf is model.alias


@pytest.mark.parametrize(
    "accessor", ["getattr", "getattribute", "descriptor", "introduced_after_capture"]
)
def test_alias_validation_preserves_dynamic_attribute_lookup(monkeypatch, accessor):
    class Parent(nn.Module):
        pass

    def dynamic_getattr(self, name):
        redirect = object.__getattribute__(self, "__dict__").get("redirect")
        if name == "leaf" and redirect is not None:
            return redirect
        return nn.Module.__getattr__(self, name)

    if accessor == "getattr":
        Parent.__getattr__ = dynamic_getattr
    elif accessor == "getattribute":

        def dynamic_getattribute(self, name):
            redirect = object.__getattribute__(self, "__dict__").get("redirect")
            if name == "leaf" and redirect is not None:
                return redirect
            return object.__getattribute__(self, name)

        Parent.__getattribute__ = dynamic_getattribute
    elif accessor == "descriptor":
        Parent.leaf = property(
            lambda self: self.__dict__.get("redirect", self._modules["leaf"])
        )
    model = nn.Module()
    model.parent = Parent()
    model.parent._modules["leaf"] = nn.Linear(1, 1, bias=False)
    model.alias = model.parent._modules["leaf"]
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    aliases = installer._parameter_aliases(model)
    installer._restore_parameter_aliases(aliases)
    if accessor == "introduced_after_capture":
        monkeypatch.setattr(Parent, "__getattr__", dynamic_getattr)
    object.__setattr__(model.parent, "redirect", nn.Linear(1, 1, bias=False))
    with pytest.raises(IncompleteRefit, match="parameter alias owner 'parent.leaf'"):
        installer._restore_parameter_aliases(aliases)


@pytest.mark.parametrize("path", [".a", ".", "..", "a.", "a..b"])
def test_alias_validation_rejects_custom_paths_with_empty_components(monkeypatch, path):
    model = nn.Module()
    model.a = nn.Linear(1, 1, bias=False)
    model.alias = model.a
    monkeypatch.setattr(
        model,
        "named_modules",
        lambda **kwargs: iter((("", model), (path, model.a), ("alias", model.a))),
    )
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    aliases = installer._parameter_aliases(model)
    with pytest.raises(IncompleteRefit, match="parameter alias owner"):
        installer._validate_alias_owners(aliases)


def test_alias_validation_rejects_conflicting_children_on_one_physical_edge(
    monkeypatch,
):
    model = nn.Module()
    model.a = nn.Module()
    model.a.child = nn.Linear(1, 1, bias=False)
    model.b = model.a
    detached = nn.Linear(1, 1, bias=False)
    detached.weight = model.a.child.weight
    entries = [
        ("", model),
        ("a", model.a),
        ("a.child", model.a.child),
        ("b", model.b),
        ("b.child", detached),
    ]
    monkeypatch.setattr(model, "named_modules", lambda **kwargs: iter(entries))
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    aliases = installer._parameter_aliases(model)
    with pytest.raises(IncompleteRefit, match="parameter alias owner 'b.child'"):
        installer._validate_alias_owners(aliases)


@pytest.mark.parametrize("local_materialization", [False, True])
def test_alias_mutation_fails_before_a_later_hook_can_hide_it(
    monkeypatch, local_materialization
):
    model = nn.Module()
    model.first = nn.Linear(1, 1, bias=False)
    model.second = nn.Linear(1, 1, bias=False)
    model.branch = nn.Module()
    model.branch.leaf = nn.Linear(1, 1, bias=False)
    model.alias = model.branch.leaf
    _install_fake_vllm(monkeypatch, lambda _model: None)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    quant_base = sys.modules[
        "vllm.model_executor.layers.quantization.base_config"
    ].QuantizeMethodBase
    hooks = []

    class Mutate(quant_base):
        def process_weights_after_loading(self, _layer):
            model.alias = nn.Linear(1, 1, bias=False)
            hooks.append("mutate")

    class Restore(quant_base):
        def process_weights_after_loading(self, _layer):
            model.alias = model.branch.leaf
            hooks.append("restore")

    model.first.quant_method = Mutate()
    model.second.quant_method = Restore()
    for layer in model.modules():
        layerwise.LAYERWISE_INFO[layer] = SimpleNamespace(
            kernel_tensors=(dict(layer.named_parameters(recurse=False)), {}),
            loaded_weights=[],
            reset=lambda: None,
        )
    if local_materialization:
        for info in layerwise.LAYERWISE_INFO.values():
            info.restore_device = torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    with pytest.raises(IncompleteRefit, match="parameter alias owner 'alias'"):
        installer.install_tensors(
            {name: torch.ones_like(value) for name, value in model.named_parameters()}
        )
    assert hooks == ["mutate"]


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
        assert "retention_batch_scans" not in prepared.transfer_metrics
        assert "retention_final_scans" not in prepared.transfer_metrics
        installer.capture(manifest)
        assert model.capture_calls == 4


class _Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))
        self.child = nn.Linear(2, 2, bias=False)


@pytest.mark.parametrize("packed", [False, True])
def test_hook_replacing_a_submodule_never_leaves_the_live_child_stale(
    monkeypatch, packed
):
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
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    try:
        installer.install_streaming(PreparedStreamingTensors(batches, names, {}))
    except IncompleteRefit:
        return
    assert torch.equal(model.layer.child.weight, child["layer.child.weight"]), (
        "install succeeded but the live child never received its published bytes"
    )


def test_streaming_install_error_is_not_replaced_by_a_failed_prefetch_drain(
    monkeypatch,
):
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

    prepared = transfer_module._PreparedBoundedTransfer(
        planned, {"w": source}, Transport()
    )
    transfer = object.__new__(transfer_module._NixlStagedTransfer)
    transfer._descriptor_cache = None
    transfer._workspace_generation = 0
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
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    with pytest.raises(IncompleteRefit, match="no compatible live storage"):
        installer.install_streaming(
            PreparedStreamingTensors(
                lambda: transfer.iter_bounded(prepared, {}), names, {}
            )
        )


@pytest.mark.parametrize("failure", [None, "source", "drain"])
def test_streaming_reuses_initial_aliases_only_within_reload(monkeypatch, failure):
    import gc
    import weakref

    from modelexpress_rl.inference.engines.vllm import installer as api

    _install_fake_vllm(monkeypatch, lambda model: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    model = nn.Module()
    model.first = model.first_alias = nn.Linear(2, 2, bias=False)
    model.second = model.second_alias = nn.Linear(2, 2, bias=False)
    names = frozenset(dict(model.named_parameters()))
    plans = []
    reload_plans = []
    original = api._select_parameter_aliases

    def observe(model, groups, modules, initial_aliases):
        plan = original(model, groups, modules, initial_aliases)
        assert plan is initial_aliases
        reload_plans.append(weakref.ref(initial_aliases))
        plans.append(weakref.ref(plan))
        return plan

    monkeypatch.setattr(api, "_select_parameter_aliases", observe)
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )

    def batches():
        yield {"first.weight": torch.ones(2, 2)}
        yield {"second.weight": torch.full((2, 2), 2.0)}
        if failure == "source":
            raise RuntimeError("source failed after completed batch")

    drains = []

    def observe_drain():
        drains.append(True)
        if len(drains) % 3:
            assert plans[-1]() is reload_plans[-1]()
            assert reload_plans[-1]() is not None
        if failure == "drain" and len(drains) == 3:
            raise RuntimeError("drain failed after completed batch")

    monkeypatch.setattr(installer, "_drain_streaming", observe_drain)
    prepared = PreparedStreamingTensors(batches, names, {})
    if failure:
        with pytest.raises(RuntimeError, match="failed after completed batch"):
            installer.install_streaming(prepared)
    else:
        installer.install_streaming(prepared)
        assert all(plan() is None for plan in plans)
        installer.install_streaming(PreparedStreamingTensors(batches, names, {}))
    gc.collect()
    assert len(plans) == (2 if failure else 4)
    assert all(plan() is None for plan in plans)
    assert all(plan() is None for plan in reload_plans)
    assert torch.equal(model.first.weight, torch.ones(2, 2))
    assert torch.equal(model.second.weight, torch.full((2, 2), 2.0))


def test_streaming_checks_fresh_owner_coverage_each_batch(monkeypatch):
    from modelexpress_rl.inference.engines.vllm import installer as api

    _install_fake_vllm(monkeypatch, lambda model: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    model = nn.Module()
    for name in ("first", "second", "third"):
        layer = nn.Linear(2, 2, bias=False)
        setattr(model, name, layer)
        setattr(model, f"{name}_alias", layer)
    names = frozenset(dict(model.named_parameters()))
    original = api._select_parameter_aliases
    reused = []

    def observe(model, groups, modules, initial_aliases):
        plan = original(model, groups, modules, initial_aliases)
        reused.append(plan is initial_aliases)
        return plan

    monkeypatch.setattr(api, "_select_parameter_aliases", observe)

    def batches():
        yield {"first.weight": torch.ones(2, 2)}
        yield {"second.weight": torch.ones(2, 2)}
        model.third.bias = nn.Parameter(torch.zeros(2))
        yield {"third.weight": torch.ones(2, 2)}

    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    with pytest.raises(IncompleteRefit, match="batch splits an owning module"):
        installer.install_streaming(PreparedStreamingTensors(batches, names, {}))
    assert reused == [True, True]


@pytest.mark.parametrize("owner_kind", ["unrelated", "new_alias", "initial_alias"])
def test_streaming_alias_plans_preserve_destructor_boundaries(monkeypatch, owner_kind):
    import gc
    import weakref

    from modelexpress_rl.inference.engines.vllm import installer as api

    original = api._select_parameter_aliases

    def run(reuse):
        trace = []
        with monkeypatch.context() as patch:
            _install_fake_vllm(patch, lambda model: None)
            patch.setattr(torch.cuda, "synchronize", lambda device: None)
            if not reuse:
                patch.setattr(
                    api,
                    "_select_parameter_aliases",
                    lambda model, groups, modules, _initial: (
                        api._compile_parameter_aliases(model, groups, modules)
                    ),
                )
            else:
                patch.setattr(api, "_select_parameter_aliases", original)
            model = nn.Module()
            model.first = model.first_alias = nn.Linear(2, 2, bias=False)
            model.second = model.second_alias = nn.Linear(2, 2, bias=False)
            root = weakref.ref(model)

            class Removed(nn.Module):
                def __del__(self):
                    trace.append("destructor")
                    live = root()
                    if live is not None:
                        live.second.extra = nn.Parameter(torch.zeros(2))

            if owner_kind != "new_alias":
                model.removed = Removed()
                if owner_kind == "initial_alias":
                    model.removed.weight = model.first.weight
            names = frozenset(dict(model.named_parameters()))

            def batches():
                if owner_kind == "new_alias":
                    model.removed = Removed()
                    model.removed.weight = model.first.weight
                yield {"first.weight": torch.ones(2, 2)}
                trace.append("detach")
                del model.removed
                trace.append("second-yield")
                yield {"second.weight": torch.ones(2, 2)}

            installer = _VllmInstaller(
                model=model,
                vllm_config=object(),
                model_config=object(),
                device=torch.device("cpu"),
            )
            try:
                installer.install_streaming(
                    PreparedStreamingTensors(batches, names, {})
                )
            except IncompleteRefit as error:
                outcome = str(error)
            else:
                outcome = "accepted"
            gc.collect()
            return outcome, trace, tuple(dict(model.named_parameters()))

    uncached, reused = run(False), run(True)
    assert reused == uncached
    assert reused[0] != "accepted"
    if owner_kind != "initial_alias":
        assert "batch splits an owning module" in reused[0]
        assert reused[1] == ["detach", "destructor", "second-yield"]


@pytest.mark.parametrize("case", ["no_alias", "load_time_tie", "runtime_alias", "mixed"])
@pytest.mark.parametrize("runtime_first", [False, True])
def test_checkpoint_reload_distinguishes_wna16_runtime_aliases(
    monkeypatch, tmp_path, case, runtime_first
):
    model = nn.Module()
    for name in ("w13", "w2"):
        packed = nn.Parameter(torch.zeros(2, 2), requires_grad=False)
        setattr(model, f"{name}_weight_packed", packed)
        if case in ("load_time_tie", "mixed"):
            setattr(model, f"{name}_tied", packed)
    metadata = dict(model._parameters)
    if case in ("runtime_alias", "mixed"):
        for name in ("w13", "w2"):
            setattr(model, f"{name}_weight", getattr(model, f"{name}_weight_packed"))
        if runtime_first:
            model._parameters = dict(reversed(list(model._parameters.items())))
    originals = dict(model._parameters)
    pointers = {name: value.data_ptr() for name, value in originals.items()}

    def initialize(target):
        target._parameters = {
            name: nn.Parameter(torch.empty_like(value), requires_grad=False)
            for name, value in metadata.items()
        }

    _install_fake_vllm(monkeypatch, initialize)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    layerwise.LAYERWISE_INFO[model] = SimpleNamespace(
        restore_metadata=(metadata, {}), kernel_tensors=None
    )
    loaded = []

    class DefaultModelLoader:
        def __init__(self, _config):
            pass

        def load_weights(self, target, _config):
            assert set(target._parameters) == set(metadata)
            for name in ("w13", "w2"):
                packed = getattr(target, f"{name}_weight_packed")
                if case in ("load_time_tie", "mixed"):
                    assert getattr(target, f"{name}_tied") is packed
                packed.data.fill_(7)
            loaded.append(True)

    def finalize(target, _config):
        for name in ("w13", "w2"):
            if case in ("runtime_alias", "mixed"):
                setattr(target, f"{name}_weight", getattr(target, f"{name}_weight_packed"))
        for name, value in originals.items():
            value.data.copy_(getattr(target, name))
        target._parameters = dict(originals)

    layerwise.finalize_layerwise_reload = finalize
    sys.modules[
        "vllm.model_executor.model_loader.default_loader"
    ].DefaultModelLoader = DefaultModelLoader
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    installer = _VllmInstaller(
        model=model,
        vllm_config=SimpleNamespace(
            quant_config=object(), load_config=SimpleNamespace(load_format="modelexpress")
        ),
        model_config=SimpleNamespace(model="/launch", revision="main"),
        device=torch.device("cpu"),
    )
    installer.install_checkpoint(tmp_path)
    assert loaded == [True]
    for name, original in originals.items():
        assert getattr(model, name) is original
        assert original.data_ptr() == pointers[name]
        assert torch.equal(original, torch.full((2, 2), 7.0))


@pytest.mark.parametrize("recorded", [False, True])
def test_load_time_alias_restoration_does_not_ignore_missing_ties(recorded):
    model = nn.Module()
    model.weight = nn.Parameter(torch.zeros(2))
    model.tied = model.weight
    installer = object.__new__(_VllmInstaller)
    installer._model = model
    info = {model: SimpleNamespace(restore_metadata=(dict(model._parameters), {}))}
    aliases = installer._load_time_parameter_aliases(
        installer._parameter_aliases(model), info if recorded else {}
    )
    del model.tied
    with pytest.raises(AttributeError):
        installer._restore_parameter_aliases(aliases)


def test_streamed_checkpoint_uses_graph_safe_reload_and_closes_iterator(monkeypatch):
    from unittest.mock import Mock
    from modelexpress_rl.inference.streaming_checkpoint import StreamedCheckpoint
    from modelexpress_rl.inference.receiver import _S3Version
    from modelexpress_rl.train import WeightPayloadFormat

    events = []
    _install_fake_vllm(monkeypatch, lambda model: events.append("initialize"))
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    layerwise.finalize_layerwise_reload = lambda model, config: events.append(
        "finalize"
    )
    weight_utils = ModuleType("vllm.model_executor.model_loader.weight_utils")
    distributed = ModuleType("vllm.distributed")
    distributed.get_world_group = lambda: SimpleNamespace(local_rank=1)
    distributed.get_pipeline_model_parallel_world_size = lambda: 1
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.model_loader.weight_utils", weight_utils
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: events.append("sync"))
    monkeypatch.setenv("MX_MS_DISTRIBUTED", "1")

    def iterator(files, use_tqdm_on_load, is_distributed):
        assert files == ["s3://bucket/v2/model.safetensors"]
        assert is_distributed
        try:
            events.append("stream")
            yield "weight", torch.tensor([7.0, 8.0])
        finally:
            events.append("close")

    weight_utils.runai_safetensors_weights_iterator = iterator
    model = nn.Module()
    model.register_parameter(
        "weight", nn.Parameter(torch.zeros(2), requires_grad=False)
    )
    address = model.weight.data_ptr()

    def load(weights):
        for name, tensor in weights:
            events.append("load")
            model.weight.data.copy_(tensor)

    model.load_weights = load
    stream = StreamedCheckpoint(
        version=_S3Version(
            "v2",
            None,
            WeightPayloadFormat.FULL_HF_CHECKPOINT,
            "s3://bucket/v2/model.safetensors.index.json",
        ),
        index_data=b"{}",
        weight_map={"weight": "model.safetensors"},
        tensor_metadata={"weight": {"dtype": "F32", "shape": [2], "byte_size": 8}},
        store=Mock(),
    )
    installer = _VllmInstaller(
        model=model,
        vllm_config=SimpleNamespace(
            load_config=SimpleNamespace(use_tqdm_on_load=False),
            parallel_config=SimpleNamespace(tensor_parallel_size=4),
        ),
        model_config=object(),
        device=torch.device("cpu"),
    )
    from pathlib import Path

    installer.install(
        PreparedCheckpointArtifact(
            PreparedCheckpoint("v2", Path("/unused"), {}, streaming=stream)
        )
    )
    assert events == ["initialize", "stream", "load", "close", "finalize", "sync"]
    assert model.weight.data_ptr() == address
    assert torch.equal(model.weight, torch.tensor([7.0, 8.0]))
    assert stream._writer is None
    assert stream._complete
