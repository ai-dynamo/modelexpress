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
    _VllmInstaller,
)
from modelexpress_rl.inference.plan import (
    PreparedCheckpointArtifact,
    PreparedEngineTensors,
    PreparedRuntimeTensors,
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
    assert recorder.as_dict()["stages"]["post_install"]["count"] == 1


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
    staged = type(
        "Staged", (), {"tensors": live, "metrics": {"bytes_received": 0}}
    )()

    metrics = installer.install(PreparedRuntimeTensors(staged=staged))

    assert torch.equal(live["weight"], torch.tensor([7.0, 8.0]))
    assert torch.equal(live["runtime_buffer"], torch.tensor([9.0]))
    assert {name: tensor.data_ptr() for name, tensor in live.items()} == original_pointers
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
