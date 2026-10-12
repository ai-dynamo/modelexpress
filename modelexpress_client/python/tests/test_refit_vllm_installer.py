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
    _EngineLoadLayout,
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


def _retain_test_layout(
    installer, *, parameters=None, groups=None
) -> _EngineLoadLayout:
    """Give installer tests the explicit load-time plan production captures."""
    from modelexpress.refit.reshard.types import CaptureResult
    from modelexpress_rl.inference.engines.vllm import installer as api

    model = installer._model
    named_parameters = dict(model.named_parameters())
    if parameters is None:
        parameters = {
            name: (tuple(parameter.shape), parameter.dtype)
            for name, parameter in named_parameters.items()
        }
    if groups is None:
        owners = {}
        for name in parameters:
            path, _, leaf = name.rpartition(".")
            owner = model.get_submodule(path)
            owners.setdefault(owner, []).append((name, leaf))
        groups = tuple((owner, tuple(values)) for owner, values in owners.items())
    aliases = installer._parameter_aliases(model)
    runtime_storages = frozenset(
        tensor.untyped_storage().data_ptr()
        for tensor in (*model.parameters(), *model.buffers())
        if tensor.device.type != "meta"
    )
    installer._engine_load_layout = api._EngineLoadLayout(
        CaptureResult(),
        parameters,
        groups,
        aliases,
        aliases,
        installer._graph_tensors(),
        runtime_storages,
    )
    return installer._engine_load_layout


def _install_streaming(
    installer: _VllmInstaller, prepared: PreparedStreamingTensors
) -> None:
    if installer._engine_load_layout is None:
        _retain_test_layout(installer)
    installer.install_streaming(prepared)


def test_installer_resolves_load_time_parameters_after_layerwise_reload(
    monkeypatch,
) -> None:
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
    _retain_test_layout(
        installer,
        parameters={"weight": ((1,), torch.float32)},
        groups=((model, (("weight", "weight"),)),),
    )

    installer._process_and_commit({"weight": torch.tensor([7.0])})

    assert model.weight.item() == 7.0


def test_installer_rejects_parameters_left_on_meta(monkeypatch) -> None:
    model = nn.Module()
    model.weight = nn.Parameter(torch.zeros(1))
    model.child = nn.Module()
    model.child.register_parameter("orphan", nn.Parameter(torch.zeros(1)))

    def initialize(target):
        target.weight = nn.Parameter(torch.empty(1, device="meta"))
        target.child.orphan = nn.Parameter(torch.empty(1, device="meta"))

    _install_fake_vllm(monkeypatch, initialize)
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    _retain_test_layout(
        installer,
        parameters={"weight": ((1,), torch.float32)},
        groups=((model, (("weight", "weight"),)),
                (model.child, (("child.orphan", "orphan"),))),
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
    _retain_test_layout(installer)

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
        _retain_test_layout(installer)

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
    _install_streaming(installer,
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
    _install_streaming(installer, prepared)
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
        _install_streaming(installer, prepared)
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
            _install_streaming(installer, prepared)
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
            _install_streaming(installer, prepared)
        assert torch.equal(model[1].weight, before_second)
    else:
        _install_streaming(installer, prepared)
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

    _install_streaming(installer,
        PreparedStreamingTensors(batches, frozenset(layout), {})
    )
    assert model.embedding.weight is model.lm_head.weight
    assert model.embedding.weight.data_ptr() == address
    assert torch.equal(model.lm_head.weight, torch.full((2, 2), 7.0))


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
    _install_streaming(installer, PreparedStreamingTensors(batches, names, {}))
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


def test_streaming_rejects_incomplete_transfer_for_captured_layout(
    monkeypatch,
) -> None:
    _install_fake_vllm(monkeypatch, lambda model: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    model = nn.Module()
    model.gate = nn.Linear(2, 2)
    model.experts = nn.Linear(2, 2)
    model.experts.bias = model.gate.bias
    names = frozenset(dict(model.named_parameters()))

    def batches():
        yield {"experts.weight": torch.ones(2, 2)}

    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    with pytest.raises(IncompleteRefit, match="ended before every parameter"):
        _install_streaming(installer, PreparedStreamingTensors(batches, names, {}))


@pytest.mark.parametrize("replace_alias_owner", [False])
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
    _install_streaming(installer, PreparedStreamingTensors(batches, names, {}))
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


@pytest.mark.parametrize("invalid_traversal", ["missing_prefix"])
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


def test_retained_install_plan_preserves_tied_parameters(
    monkeypatch,
) -> None:
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
    for value in (7.0, 11.0, -3.0):

        def batches(value=value):
            yield {"embedding.weight": torch.full((2, 2), value)}

        prepared = PreparedStreamingTensors(batches, frozenset(layout), {})
        _install_streaming(installer, prepared)
        assert model.embedding.weight is model.lm_head.weight
        assert model.embedding.weight.data_ptr() == address
        assert torch.equal(model.lm_head.weight, torch.full((2, 2), value))
        installer.capture(manifest)
    assert model.capture_calls == 1


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
        CaptureResult(copies=copies), layout, {"w": source}, 512,
        source_slots={"source": "slot-source"},
    )
    resolved = transfer_module._ResolvedSources(
        {"w": source},
        {"source": "source"},
        {"source": 0},
        {"source": b"metadata"},
    )
    source_binding = transfer_module._SourceBinding(
        resolved,
        {"slot-source": "source"},
        (("slot-source", "source", source.shards[0].addr),),
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
        planned, {"w": source}, Transport(), source_binding
    )
    transfer = object.__new__(transfer_module._NixlStagedTransfer)
    transfer._weight_update_plan = None
    transfer._closed = False
    transfer._active = prepared
    transfer._device = torch.device("cpu")
    transfer._device_id = 0
    transfer._staging_arenas = [torch.empty(512, dtype=torch.uint8) for _ in range(2)]
    from tests.test_refit_nixl_staged_transfer import _bind_test_execution
    _bind_test_execution(transfer, prepared)

    class Owner(nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4, dtype=dtype))

    model = nn.Module()
    model.a = Owner(torch.float32)
    model.b = Owner(torch.float32)
    names = frozenset(dict(model.named_parameters()))
    installer = _VllmInstaller(
        model=model,
        vllm_config=object(),
        model_config=object(),
        device=torch.device("cpu"),
    )
    def fail_copy(target, source) -> None:
        raise IncompleteRefit("injected installation failure")

    monkeypatch.setattr(
        "modelexpress_rl.inference.engines.vllm.installer._copy_received", fail_copy
    )
    with pytest.raises(IncompleteRefit, match="injected installation failure"):
        _install_streaming(installer,
            PreparedStreamingTensors(
                lambda: transfer.iter_bounded(prepared, {}), names, {}
            )
        )


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


@pytest.mark.parametrize("invalid", ["missing", "type", "shape", "dtype", "device"])
def test_install_validates_graph_buffers_against_retained_layout(monkeypatch, invalid) -> None:
    _install_fake_vllm(monkeypatch, lambda model: None)
    model = nn.Module()
    model.weight = nn.Parameter(torch.zeros(2))
    model.state = torch.zeros(2)
    installer = _VllmInstaller(
        model=model, vllm_config=object(), model_config=object(), device=torch.device("cpu")
    )
    _retain_test_layout(installer)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    if invalid == "missing":
        del model.state
    elif invalid == "type":
        model.state = "not a tensor"
    elif invalid == "shape":
        model.state = torch.ones(3)
    elif invalid == "dtype":
        model.state = torch.ones(2, dtype=torch.float64)
    else:
        model.state = torch.ones(2, device="meta")
    for _ in range(2):
        with pytest.raises(IncompleteRefit, match="graph buffer"):
            installer.install_tensors({"weight": torch.ones(2)})


def test_install_retains_graph_buffer_address_across_cold_and_warm_updates(monkeypatch):
    _install_fake_vllm(monkeypatch, lambda model: None)
    model = nn.Module()
    model.weight = nn.Parameter(torch.zeros(2))
    model.state = torch.zeros(2)
    installer = _VllmInstaller(
        model=model, vllm_config=object(), model_config=object(), device=torch.device("cpu")
    )
    _retain_test_layout(installer)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    value = 0.0
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]
    layerwise.finalize_layerwise_reload = lambda _model, _config: setattr(
        model, "state", torch.full((2,), value)
    )
    original = model.state
    address = original.data_ptr()
    for value in (3.0, 7.0):
        installer.install_tensors({"weight": torch.full((2,), value)})
        assert model.state is original
        assert model.state.data_ptr() == address
        assert torch.equal(model.state, torch.full((2,), value))
        assert torch.equal(model.weight, torch.full((2,), value))


@pytest.mark.parametrize("runtime_device", ["cpu", "meta"])
def test_install_checks_finalized_runtime_parameters(monkeypatch, runtime_device):
    model = nn.Module()
    model.weight_packed = nn.Parameter(torch.zeros(2))

    def initialize(model):
        if hasattr(model, "weight"):
            model.weight_packed = model.weight
            del model.weight

    _install_fake_vllm(monkeypatch, initialize)
    installer = _VllmInstaller(
        model=model, vllm_config=object(), model_config=object(), device=torch.device("cpu")
    )
    _retain_test_layout(installer)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device: None)
    layerwise = sys.modules["vllm.model_executor.model_loader.reload.layerwise"]

    def finalize(model, _config):
        model.weight = nn.Parameter(model.weight_packed.to(runtime_device))
        del model.weight_packed

    layerwise.finalize_layerwise_reload = finalize
    for value in (3.0, 7.0):
        tensors = {"weight_packed": torch.full((2,), value)}
        if runtime_device == "meta":
            with pytest.raises(IncompleteRefit, match="left parameters on the meta device"):
                installer.install_tensors(tensors)
        else:
            installer.install_tensors(tensors)
            assert torch.equal(model.weight, tensors["weight_packed"])
