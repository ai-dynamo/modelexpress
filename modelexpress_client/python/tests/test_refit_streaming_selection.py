# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from modelexpress_rl.inference import runtime
from modelexpress_rl.inference.adapter import (
    GeneratorSource,
    GeneratorTransferInputs,
    NixlGeneratorSource,
)
from modelexpress_rl.inference.engines.vllm.installer import _VllmInstaller
from modelexpress_rl.inference.plan import (
    PreparedDirectGroupTensors,
    PreparedStreamingTensors,
    TrainerUpdateSource,
)
from modelexpress_rl.train import WeightPayloadFormat


def setup_method(monkeypatch, installer_type=_VllmInstaller):
    model = torch.nn.Linear(2, 2)
    installer = object.__new__(installer_type)
    installer._model = model
    installer._vllm_config = SimpleNamespace(quant_config=None)
    events = []
    values = {name: torch.ones_like(p) for name, p in model.named_parameters()}
    layout = {name: (tuple(p.shape), p.dtype) for name, p in model.named_parameters()}

    class Transfer:
        def __init__(self, **kwargs):
            pass

        def prepare(self, **kwargs):
            events.append("prepare")
            return SimpleNamespace(
                metrics={}, batches=[SimpleNamespace(layouts=(dict(layout),))]
            )

        def iter_bounded(self, prepared, metrics):
            events.append("read")
            yield {name: value.clone() for name, value in values.items()}
            events.append("drained")

        def reset_workspace(self):
            events.append("reset")

    monkeypatch.setattr(runtime, "_NixlStagedTransfer", Transfer)
    method = runtime._create_load_time_tensor_method(
        capability=SimpleNamespace(device_id=0, device="cpu", capture_layout=None),
        worker_id="receiver",
        installer=installer,
    )
    source = TrainerUpdateSource(
        GeneratorTransferInputs(
            version_id="v:1",
            base_version_id=None,
            layout_signature="",
            payload_format=WeightPayloadFormat.FULL_TENSOR,
            sources=(
                GeneratorSource(
                    "slot",
                    "trainer",
                    "digest",
                    NixlGeneratorSource("trainer:19000", b"manifest", "structure"),
                ),
            ),
        )
    )
    return model, installer, method, source, events, values, layout


def test_runtime_selects_direct_before_read_and_preserves_three_updates(monkeypatch):
    model, installer, method, source, events, values, _ = setup_method(monkeypatch)
    original = {name: (p, p.data_ptr()) for name, p in model.named_parameters()}
    for version in range(3):
        events.clear()
        before = {name: p.clone() for name, p in model.named_parameters()}
        for tensor in values.values():
            tensor.fill_(version + 5)
        prepared = method.prepare_streaming(
            version=SimpleNamespace(version_id=f"v:{version}"),
            source=source,
            max_staging_bytes=512,
        )
        assert type(prepared) is PreparedDirectGroupTensors
        assert prepared.version_id == f"v:{version}"
        assert prepared.source is method._active_streamed
        assert prepared.metrics["direct_install_selected"] == 1
        assert prepared.metrics["streaming_selection_s"] >= 0
        assert events == ["prepare"]
        assert all(torch.equal(p, before[name]) for name, p in model.named_parameters())
        with method.installation_context(prepared):
            installer.install(prepared)
        method.release(prepared)
        assert events == ["prepare", "read", "drained"]
        assert method._active_direct is None and method._active_streamed is None
        for name, param in model.named_parameters():
            assert param is original[name][0] and param.data_ptr() == original[name][1]
            assert torch.equal(param, values[name])


@pytest.mark.parametrize("reason", ["shape", "dtype", "helper", "quantized"])
def test_engine_declines_whole_source_before_read(monkeypatch, reason):
    model, installer, method, source, events, _, layout = setup_method(monkeypatch)
    if reason == "shape":
        layout["weight"] = ((4, 1), torch.float32)
    elif reason == "dtype":
        layout["weight"] = ((2, 2), torch.bfloat16)
    elif reason == "helper":
        model.helper = object()
    else:
        installer._vllm_config.quant_config = object()
    before = {name: p.clone() for name, p in model.named_parameters()}
    prepared = method.prepare_streaming(
        version=SimpleNamespace(version_id="v:1"), source=source, max_staging_bytes=512
    )
    assert type(prepared) is PreparedStreamingTensors
    assert prepared is method._active_streamed
    assert prepared.metrics["direct_install_selected"] == 0
    assert events == ["prepare"]
    assert all(torch.equal(p, before[name]) for name, p in model.named_parameters())
    method.release(prepared)


@pytest.mark.parametrize(
    "staging_device,staging_buffers", [("cpu", 1), ("cuda", 2), ("cpu", 2)]
)
@pytest.mark.parametrize("glm_direct", [False, True])
def test_unsupported_receive_mode_declines_or_rejects_before_read(
    monkeypatch, staging_device, staging_buffers, glm_direct
):
    monkeypatch.setenv("MX_REFIT_GLM_DIRECT", str(int(glm_direct)))
    model, _, method, source, events, _, _ = setup_method(monkeypatch)
    before = {name: p.clone() for name, p in model.named_parameters()}
    arguments = {
        "version": SimpleNamespace(version_id="v:1"),
        "source": source,
        "max_staging_bytes": 512,
        "staging_device": staging_device,
        "staging_buffers": staging_buffers,
    }
    if glm_direct:
        with pytest.raises(ValueError, match="one CUDA receive arena"):
            method.prepare_streaming(**arguments)
        assert events == ["prepare", "reset"]
        assert method._active_streamed is None and method._active_direct is None
    else:
        prepared = method.prepare_streaming(**arguments)
        assert type(prepared) is PreparedStreamingTensors
        assert prepared is method._active_streamed
        assert prepared.metrics["direct_install_selected"] == 0
        assert events == ["prepare"]
        method.release(prepared)
    assert all(torch.equal(p, before[name]) for name, p in model.named_parameters())


@pytest.mark.parametrize("failure", ["exception", "foreign_source", "invalid_artifact"])
def test_selection_failure_resets_unread_workspace_and_allows_retry(
    monkeypatch, failure
):
    class Installer(_VllmInstaller):
        fail = True

        def prepare_streaming_artifact(self, **kwargs):
            if self.fail:
                if failure == "exception":
                    raise RuntimeError("selection failed")
                if failure == "foreign_source":
                    return PreparedStreamingTensors(lambda: iter(()), frozenset(), {})
                return None
            return super().prepare_streaming_artifact(**kwargs)

    _, installer, method, source, events, _, _ = setup_method(monkeypatch, Installer)
    version = SimpleNamespace(version_id="v:1")
    error = RuntimeError if failure == "exception" else ValueError
    with pytest.raises(error):
        method.prepare_streaming(version=version, source=source, max_staging_bytes=512)
    assert events == ["prepare", "reset"]
    assert method._active_streamed is None and method._active_direct is None
    installer.fail = False
    prepared = method.prepare_streaming(
        version=version, source=source, max_staging_bytes=512
    )
    assert type(prepared) is PreparedDirectGroupTensors
    assert events == ["prepare", "reset", "prepare"]
    method.release(prepared)
