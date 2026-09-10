# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from modelexpress_rl.inference.engines.sglang import (
    SglangGeneratorContext,
    _create_sglang_engine_runtime,
)
from modelexpress_rl.inference.engines.sglang.installer import _SglangInstaller
from modelexpress_rl.inference.plan import PreparedCheckpointArtifact
from modelexpress_rl.inference.receiver import (
    PreparedCheckpoint,
    ReceiverInstallError,
)


def _runner(tmp_path):
    checkpoint = tmp_path / "launch"
    return SimpleNamespace(
        model=object(),
        device="cpu",
        model_config=SimpleNamespace(
            model_path=str(checkpoint),
            revision=None,
            dtype=torch.float32,
        ),
        server_args=SimpleNamespace(
            download_dir=None,
            model_loader_extra_config=None,
        ),
    )


def _install_sglang_modules(monkeypatch, loader=None, setup_error=None):
    class DefaultModelLoader:
        pass

    if loader is None:
        loader = DefaultModelLoader()
        loader._get_weights_iterator = Mock(return_value=iter([]))
        loader.load_weights_and_postprocess = Mock()
    else:
        DefaultModelLoader = type(loader)

    modules = {
        name: ModuleType(name)
        for name in (
            "sglang",
            "sglang.srt",
            "sglang.srt.configs",
            "sglang.srt.configs.load_config",
            "sglang.srt.model_loader",
            "sglang.srt.model_loader.loader",
            "sglang.srt.model_loader.utils",
        )
    }
    modules["sglang.srt.configs.load_config"].LoadConfig = lambda **values: (
        SimpleNamespace(**values)
    )
    modules["sglang.srt.configs.load_config"].LoadFormat = SimpleNamespace(
        SAFETENSORS="safetensors"
    )
    loader_module = modules["sglang.srt.model_loader.loader"]
    loader_module.DefaultModelLoader = DefaultModelLoader
    loader_module.get_model_loader = (
        Mock(side_effect=setup_error)
        if setup_error is not None
        else lambda *_args: loader
    )
    modules["sglang.srt.model_loader.utils"].set_default_torch_dtype = lambda _dtype: (
        nullcontext()
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return loader


def _prepared(tmp_path):
    checkpoint = PreparedCheckpoint("target-a", tmp_path / "prepared", {})
    return PreparedCheckpointArtifact(checkpoint)


def test_sglang_engine_runtime_exposes_checkpoint_installer(tmp_path):
    runner = _runner(tmp_path)

    runtime = _create_sglang_engine_runtime(SglangGeneratorContext(runner))

    assert runtime.model_name == runner.model_config.model_path
    assert isinstance(runtime.installer, _SglangInstaller)
    assert runtime.full_tensor is None


def test_sglang_install_uses_the_prepared_checkpoint(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    installer = _SglangInstaller(runner)
    loader = _install_sglang_modules(monkeypatch)

    installer.install(_prepared(tmp_path))

    source = loader._get_weights_iterator.call_args.args[0]
    assert Path(source.model_or_path) == tmp_path / "prepared"
    loader.load_weights_and_postprocess.assert_called_once()


def test_sglang_install_rejects_unsupported_prepared_artifact(tmp_path):
    installer = _SglangInstaller(_runner(tmp_path))

    with pytest.raises(TypeError, match="requires a prepared checkpoint"):
        installer.install(object())


@pytest.mark.parametrize(
    ("setup_error", "load_error", "message"),
    [
        (RuntimeError("setup failed"), None, "setup failed"),
        (None, RuntimeError("load failed"), "load failed"),
    ],
)
def test_sglang_install_wraps_errors(
    tmp_path,
    monkeypatch,
    setup_error,
    load_error,
    message,
):
    runner = _runner(tmp_path)
    installer = _SglangInstaller(runner)

    class Loader:
        def _get_weights_iterator(self, _source):
            return iter([])

        def load_weights_and_postprocess(self, *_args):
            if load_error is not None:
                raise load_error

    _install_sglang_modules(monkeypatch, Loader(), setup_error)

    with pytest.raises(ReceiverInstallError, match=message):
        installer.install(_prepared(tmp_path))


class _LoadableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Parameter(torch.zeros(2, 3, dtype=torch.bfloat16))
        self.second = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, value in weights:
            params[name].weight_loader(params[name], value)


def _live_runner():
    return SimpleNamespace(
        model=_LoadableModel(),
        device="cpu",
        model_config=SimpleNamespace(
            model_path="Qwen/Qwen3-0.6B",
            dtype=torch.bfloat16,
            quantization=None,
            architectures=["Qwen3ForCausalLM"],
        ),
        server_args=SimpleNamespace(),
    )


def test_full_tensor_capture_does_not_change_live_weights(monkeypatch):
    from modelexpress_rl.inference.engines.sglang.layout import SglangFullTensorLayout

    runner = _live_runner()
    layout = SglangFullTensorLayout(runner)
    monkeypatch.setattr(
        "modelexpress_rl.inference.engines.sglang.layout._sglang_default_weight_loader",
        lambda: lambda param, value: param.copy_(value),
    )
    capture, params = layout.capture(
        [("first", torch.bfloat16, (2, 3)), ("second", torch.bfloat16, (3,))]
    )
    assert set(params) == {copy.param_name for copy in capture.copies}
    assert all(torch.count_nonzero(value) == 0 for value in runner.model.parameters())
    with pytest.raises(Exception, match="exact destination"):
        layout.capture([("first", torch.bfloat16, (2, 3))])


def test_full_tensor_install_validates_all_inputs_before_writing():
    from modelexpress_rl.inference.engines.sglang.layout import SglangFullTensorLayout
    from modelexpress_rl.inference.plan import PreparedEngineTensors

    runner = _live_runner()
    installer = _SglangInstaller(runner, layout=SglangFullTensorLayout(runner))
    pointers = [param.data_ptr() for param in runner.model.parameters()]
    tensors = {
        name: torch.ones_like(param) for name, param in runner.model.named_parameters()
    }
    tensors["second"] = torch.ones(4, dtype=torch.bfloat16)
    artifact = PreparedEngineTensors(SimpleNamespace(tensors=tensors))
    with pytest.raises(ReceiverInstallError, match="layout mismatch"):
        installer.install(artifact)
    assert torch.count_nonzero(runner.model.first) == 0
    tensors["second"] = torch.ones_like(runner.model.second)
    installer.install(artifact)
    assert all(torch.all(param == 1) for param in runner.model.parameters())
    assert pointers == [param.data_ptr() for param in runner.model.parameters()]


def test_partial_install_failure_permanently_fences_installer(monkeypatch):
    from modelexpress_rl.inference.engines.sglang.layout import SglangFullTensorLayout
    from modelexpress_rl.inference.plan import PreparedEngineTensors

    runner = _live_runner()
    installer = _SglangInstaller(runner, layout=SglangFullTensorLayout(runner))
    tensors = {
        name: torch.ones_like(param) for name, param in runner.model.named_parameters()
    }
    artifact = PreparedEngineTensors(SimpleNamespace(tensors=tensors))
    original = torch.Tensor.copy_

    def copy(destination, source, *args, **kwargs):
        if destination is runner.model.second:
            raise RuntimeError("injected second copy failure")
        return original(destination, source, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    with pytest.raises(RuntimeError, match="second copy failure"):
        installer.install(artifact)
    assert torch.all(runner.model.first == 1)
    assert torch.all(runner.model.second == 0)
    monkeypatch.setattr(torch.Tensor, "copy_", original)
    with pytest.raises(ReceiverInstallError, match="poisoned"):
        installer.install(artifact)


@pytest.mark.parametrize("case", ["quantized", "lora", "hidden", "alias", "storage"])
def test_full_tensor_rejects_unsupported_storage(case):
    from modelexpress_rl.inference.engines.sglang.layout import SglangFullTensorLayout

    runner = _live_runner()
    if case == "quantized":
        runner.model_config.quantization = "fp8"
    elif case == "lora":
        runner.server_args.enable_lora = True
    elif case == "hidden":
        runner.model.hidden = torch.zeros(1)
    elif case == "alias":
        runner.model.alias = runner.model.first
    if case == "storage":
        layout = SglangFullTensorLayout(runner)
        runner.model.first.data = torch.ones_like(runner.model.first)
        with pytest.raises(Exception, match="storage changed"):
            layout.validate_storage()
    else:
        with pytest.raises(Exception):
            SglangFullTensorLayout(runner)


def test_worker_fences_apply_failure_and_releases_handle(monkeypatch):
    from modelexpress_rl.inference.engines.sglang import worker

    handle = Mock(metrics={})
    client = Mock()
    client.stage_weight.return_value = handle
    client.apply_weight.side_effect = RuntimeError("partial install")
    monkeypatch.setattr(
        worker.ModelExpressGeneratorClient, "initialize", lambda config: client
    )
    binding = worker.SglangLiveRefit(_live_runner(), model_name="qwen")
    failed = binding.update(version_id="v1", training_step=1)
    assert not failed["success"] and failed["receiver_poisoned"]
    handle.release.assert_called_once()
    again = binding.update(version_id="v2", training_step=2)
    assert not again["success"] and again["receiver_poisoned"]
    client.stage_weight.assert_called_once()


def test_worker_allows_retry_after_nonmutating_stage_failure(monkeypatch):
    from modelexpress_rl.inference.engines.sglang import worker

    handle = Mock(metrics={})
    client = Mock()
    client.stage_weight.side_effect = [RuntimeError("transfer failed"), handle]
    client.apply_weight.return_value = {}
    monkeypatch.setattr(
        worker.ModelExpressGeneratorClient, "initialize", lambda config: client
    )
    binding = worker.SglangLiveRefit(_live_runner(), model_name="qwen")
    assert not binding.update(version_id="v1", training_step=1)["receiver_poisoned"]
    assert binding.update(version_id="v1", training_step=1)["success"]
    assert binding.update(version_id="v1", training_step=1)["success"]
    assert not binding.update(version_id="other", training_step=1)["success"]
    assert client.stage_weight.call_count == 2


@pytest.mark.parametrize("modern", [True, False])
def test_worker_timing_uses_runner_parallel_state_rank(monkeypatch, modern):
    from modelexpress_rl.inference.engines.sglang import worker

    runner = _live_runner()
    if modern:
        runner.ps = SimpleNamespace(tp_rank=1)
    else:
        runner.tp_rank = 1
    client = Mock()
    client.stage_weight.return_value = Mock(metrics={})
    client.apply_weight.return_value = {}
    monkeypatch.setattr(worker.ModelExpressGeneratorClient, "initialize", lambda config: client)
    result = worker.SglangLiveRefit(runner, model_name="qwen").update(
        version_id="v1", training_step=1
    )
    assert result["success"]
    assert result["timing"]["rank"] == 1
