# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from modelexpress.engines.sglang.refit.receiver import (
    SglangReshardReceiver,
    sglang_layout_signature,
)
from modelexpress.engines.sglang.refit.worker import (
    SglangRefitRequest,
    run_sglang_live_refit,
)
from modelexpress.refit.reshard.receiver import ReshardReceiver, ReshardTopologyChanged


class _LoadableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Parameter(torch.zeros(2, 3, dtype=torch.bfloat16))
        self.second = nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, value in weights:
            params[name].weight_loader(params[name], value)


def _receiver(model=None, registry=None):
    model = model or _LoadableModel()
    receiver = object.__new__(SglangReshardReceiver)
    receiver._model = model
    receiver._tensor_registry = registry or dict(model.named_parameters())
    receiver._model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        quantization=None,
        architectures=["Qwen3ForCausalLM"],
    )
    receiver._poisoned = False
    return receiver


def test_sglang_capture_uses_real_loader_and_requires_whole_model(monkeypatch):
    model = _LoadableModel()
    receiver = _receiver(model)

    def default_weight_loader(param, value):
        param.copy_(value)

    monkeypatch.setattr(
        "modelexpress.engines.sglang.refit.receiver._sglang_default_weight_loader",
        lambda: default_weight_loader,
    )
    capture, layout = receiver._capture(
        [
            ("first", torch.bfloat16, (2, 3)),
            ("second", torch.bfloat16, (3,)),
        ]
    )

    assert {copy.param_name for copy in capture.copies} == {"first", "second"}
    assert layout == {
        "first": ((2, 3), torch.bfloat16),
        "second": ((3,), torch.bfloat16),
    }
    assert torch.count_nonzero(model.first) == 0


def test_sglang_capture_rejects_partial_and_non_bf16(monkeypatch):
    receiver = _receiver()
    monkeypatch.setattr(
        "modelexpress.engines.sglang.refit.receiver._sglang_default_weight_loader",
        lambda: lambda param, value: param.copy_(value),
    )

    with pytest.raises(Exception, match="exact destination"):
        receiver._capture([("first", torch.bfloat16, (2, 3))])
    with pytest.raises(Exception, match="full BF16"):
        receiver._capture(
            [
                ("first", torch.float32, (2, 3)),
                ("second", torch.bfloat16, (3,)),
            ]
        )


def test_sglang_runtime_rejects_quantized_lora_and_hidden_tensors():
    model = _LoadableModel()
    receiver = _receiver(model)
    receiver._model_config.quantization = "fp8"
    with pytest.raises(Exception, match="quantized/FP8"):
        receiver._validate_runtime()

    receiver._model_config.quantization = None
    receiver._model_config.enable_lora = True
    with pytest.raises(Exception, match="LoRA"):
        receiver._validate_runtime()

    receiver._model_config.enable_lora = False
    receiver._tensor_registry["hidden"] = torch.zeros(1, dtype=torch.bfloat16)
    with pytest.raises(Exception, match="hidden tensors"):
        receiver._validate_runtime()

    del receiver._tensor_registry["hidden"]
    model.register_buffer("rotary_inv_freq", torch.ones(2), persistent=False)
    receiver._tensor_registry["rotary_inv_freq"] = model.rotary_inv_freq
    receiver._validate_runtime()


def test_sglang_install_preserves_storage_and_copies_all_parameters():
    model = _LoadableModel()
    receiver = _receiver(model)
    pointers = {name: param.data_ptr() for name, param in model.named_parameters()}
    buffers = {
        "first": torch.full_like(model.first, 4),
        "second": torch.full_like(model.second, 7),
    }

    receiver._install(buffers)

    assert torch.equal(model.first, buffers["first"])
    assert torch.equal(model.second, buffers["second"])
    assert {
        name: param.data_ptr() for name, param in model.named_parameters()
    } == pointers
    assert not receiver.poisoned


def test_layout_signature_changes_with_destination_geometry():
    first = _LoadableModel()
    second = nn.Linear(3, 4, bias=False, dtype=torch.bfloat16)

    assert sglang_layout_signature(first) == sglang_layout_signature(first)
    assert sglang_layout_signature(first) != sglang_layout_signature(second)


def test_receiver_close_releases_topology_scoped_buffers():
    receiver = object.__new__(ReshardReceiver)
    receiver._manager = SimpleNamespace(shutdown=lambda: None)
    receiver._transport = object()
    receiver._plan = object()
    receiver._source_signature = ("old",)
    receiver._recv_buffers = {"param": torch.zeros(1)}
    receiver._param_ptr = {"param": 1}
    receiver._staging = {"param": torch.zeros(1)}
    receiver._staging_ptr = {"param": 2}
    receiver._full_staging = {"source": torch.zeros(1)}
    receiver._full_staging_ptr = {"source": 3}

    receiver.close()

    assert receiver._transport is None
    assert receiver._plan is None
    assert receiver._source_signature is None
    assert not receiver._recv_buffers
    assert not receiver._param_ptr
    assert not receiver._staging
    assert not receiver._staging_ptr
    assert not receiver._full_staging
    assert not receiver._full_staging_ptr


def test_request_contract_rejects_unimplemented_fields_and_partial_groups():
    with pytest.raises(ValueError, match="unsupported.*cohort_generation"):
        SglangRefitRequest.from_mapping(
            {"target_training_step": 4, "cohort_generation": 9}
        )
    with pytest.raises(ValueError, match="logical_group='model'"):
        SglangRefitRequest(4, logical_group="layers.0").validate()


def test_receiver_factory_uses_dedicated_agent_not_startup_manager(monkeypatch):
    from modelexpress.engines.sglang.refit import worker as worker_mod

    model = _LoadableModel()
    startup_manager = object()
    client = SimpleNamespace(server_url="mx:8001")
    context = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16, quantization=None),
        identity=SimpleNamespace(model_name="qwen"),
        mx_client=client,
        global_rank=3,
        device_id=1,
        target_device=torch.device("cpu"),
    )
    state = SimpleNamespace(
        model=model,
        tensors=dict(model.named_parameters()),
        nixl_manager=startup_manager,
        context=context,
    )
    captured = {}

    class FakeReceiver:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(worker_mod, "get_sglang_loader_state", lambda _device: state)
    monkeypatch.setattr(worker_mod, "SglangReshardReceiver", FakeReceiver)
    worker_mod._receivers.clear()
    worker_mod._receiver_configs.clear()

    worker_mod._receiver_for(
        device_id=1,
        num_trainer_sources=2,
        listen_port=19001,
        timeout=4,
    )

    assert captured["listen_port"] == 19001
    assert captured["mx_client"] is client
    assert "manager" not in captured


def test_worker_entrypoint_is_monotonic_and_idempotent(monkeypatch):
    from modelexpress.engines.sglang.refit import worker as worker_mod

    class FakeReceiver:
        _plan = None
        poisoned = False
        layout_signature = "layout"

        def __init__(self):
            self.calls = []

        def update_weights(self, step, timeout):
            self.calls.append((step, timeout))
            self._plan = object()
            return {"step": step, "bytes_received": 12}

    fake = FakeReceiver()
    monkeypatch.setattr(worker_mod, "_receiver_for", lambda **_kwargs: fake)
    worker_mod._installed_versions.clear()
    worker_mod._locks.clear()

    first = run_sglang_live_refit(
        SglangRefitRequest(8, expected_layout_signature="layout"),
        device_id=0,
        num_trainer_sources=2,
        listen_port=19000,
        timeout=3,
    )
    duplicate = run_sglang_live_refit(
        {"target_training_step": 8},
        device_id=0,
        num_trainer_sources=2,
        listen_port=19000,
        timeout=3,
    )
    stale = run_sglang_live_refit(
        {"target_training_step": 7},
        device_id=0,
        num_trainer_sources=2,
        listen_port=19000,
        timeout=3,
    )

    assert first.success and first.installed_training_step == 8
    assert duplicate.success and duplicate.metrics == {"idempotent": True}
    assert not stale.success and "version rollback" in stale.error
    assert fake.calls == [(8, 3)]
    assert first.timing["backend"] == "sglang-reshard-nixl"


def test_worker_replans_once_before_reading_changed_topology(monkeypatch):
    from modelexpress.engines.sglang.refit import worker as worker_mod

    class ChangedReceiver:
        _plan = object()
        poisoned = False
        layout_signature = "layout"

        def __init__(self):
            self.closed = 0

        def update_weights(self, _step, timeout):
            raise ReshardTopologyChanged(f"changed at timeout {timeout}")

        def close(self):
            self.closed += 1

    class ReplacementReceiver:
        _plan = None
        poisoned = False
        layout_signature = "layout"

        def update_weights(self, step, timeout):
            self._plan = object()
            return {"step": step, "timeout": timeout}

    changed = ChangedReceiver()
    replacement = ReplacementReceiver()
    receivers = iter((changed, replacement))
    monkeypatch.setattr(worker_mod, "_receiver_for", lambda **_kwargs: next(receivers))
    worker_mod._receivers.clear()
    worker_mod._receiver_configs.clear()
    worker_mod._installed_versions.clear()
    worker_mod._locks.clear()

    response = run_sglang_live_refit(
        {"target_training_step": 10},
        device_id=0,
        num_trainer_sources=2,
        listen_port=19000,
        timeout=4,
    )

    assert response.success
    assert response.installed_training_step == 10
    assert changed.closed == 1
    assert response.metrics == {"step": 10, "timeout": 4}


def test_worker_reports_poisoned_install_failure(monkeypatch):
    from modelexpress.engines.sglang.refit import worker as worker_mod

    class FakeReceiver:
        _plan = object()
        poisoned = False
        layout_signature = "layout"

        def update_weights(self, _step, timeout):
            self.poisoned = True
            raise RuntimeError(f"copy failed at timeout {timeout}")

    fake = FakeReceiver()
    monkeypatch.setattr(worker_mod, "_receiver_for", lambda **_kwargs: fake)
    worker_mod._installed_versions.clear()
    worker_mod._locks.clear()

    response = run_sglang_live_refit(
        {"target_training_step": 9},
        device_id=1,
        num_trainer_sources=2,
        listen_port=19001,
        timeout=4,
    )

    assert not response.success
    assert response.receiver_poisoned
    assert response.installed_training_step is None
    assert "copy failed" in response.error
