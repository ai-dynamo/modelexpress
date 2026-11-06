# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang's VMM load envelope uses the shared arena lifecycle."""

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from modelexpress.engines.sglang import adapter, registration
from modelexpress.engines.sglang.loader import MxModelLoader
from modelexpress.adapter import StrategyFailed
from modelexpress.load_strategy import execute_load_strategies
from modelexpress.load_strategy import base as strategy_base
from modelexpress.vmm import runtime
from tests.test_sglang_loader import _device_config, _load_config, _model_config


@pytest.fixture
def load_args(monkeypatch, mock_accelerator_backend_cls):
    monkeypatch.setenv("MX_VMM_ARENA", "1")
    monkeypatch.setattr(
        adapter, "accelerator_backend_for", lambda device: mock_accelerator_backend_cls()
    )
    config_module = ModuleType("sglang.srt.configs.device_config")
    config_module.DeviceConfig = lambda device, gpu_id: _device_config(
        device=device, gpu_id=gpu_id
    )
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    return dict(
        model_config=_model_config(),
        load_config=_load_config(
            load_format="remote_instance",
            remote_instance_weight_loader_backend="modelexpress",
        ),
        device="cpu",
        gpu_id=0,
        memory_saver_adapter=SimpleNamespace(enabled=False),
        is_draft_worker=False,
    )


def test_plugin_registers_once_when_requested(monkeypatch):
    hooks = ModuleType("sglang.srt.plugins.hook_registry")
    hooks.HookType = SimpleNamespace(AROUND="around")
    hooks.HookRegistry = SimpleNamespace(register=Mock())
    monkeypatch.setitem(sys.modules, hooks.__name__, hooks)
    monkeypatch.setattr(registration, "_registered", False)
    monkeypatch.delenv("MX_VMM_ARENA", raising=False)
    registration.register()
    hooks.HookRegistry.register.assert_not_called()
    monkeypatch.setenv("MX_VMM_ARENA", "1")
    registration.register()
    registration.register()
    hooks.HookRegistry.register.assert_called_once_with(
        "sglang.srt.model_executor.model_runner_components.load_model_utils."
        "load_model_with_memory_saver",
        registration._load_with_vmm,
        hook_type="around",
    )


@pytest.mark.parametrize("failure", [False, True])
def test_arena_encloses_initialization_and_chain(load_args, monkeypatch, failure):
    events = []
    arena = object()
    contexts = []

    @contextmanager
    def enter(ctx):
        contexts.append(ctx)
        ctx.vmm_arena = arena
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    monkeypatch.setattr(registration, "maybe_enter_vmm_arena", enter)
    monkeypatch.setattr(registration, "log_arena_post_load", Mock())

    def chain(self, model, ctx):
        assert ctx is contexts[0]
        assert ctx.vmm_arena is arena
        events.append("chain")
        if failure:
            raise RuntimeError("load failed")
        return model

    monkeypatch.setattr(MxModelLoader, "_run_nixl_load", chain)

    def original(**kwargs):
        events.append("initialize")
        assert contexts[0].vmm_arena is arena
        return MxModelLoader(kwargs["load_config"]).load_model(
            model=torch.nn.Linear(2, 2),
            model_config=kwargs["model_config"],
            device_config=_device_config(),
        )

    if failure:
        with pytest.raises(RuntimeError, match="load failed"):
            registration._load_with_vmm(original, **load_args)
    else:
        assert isinstance(registration._load_with_vmm(original, **load_args), torch.nn.Module)
    assert events == ["enter", "initialize", "chain", "exit"]
    fresh = adapter.build_sglang_load_context(
        load_args["load_config"], load_args["model_config"], _device_config()
    )
    assert fresh is not contexts[0]
    assert fresh.vmm_arena is None


@pytest.mark.parametrize("bypass", ["disabled", "backend", "format", "transport", "memory_saver"])
def test_unrelated_loads_keep_the_original_allocator(load_args, monkeypatch, bypass, caplog):
    if bypass == "disabled":
        monkeypatch.delenv("MX_VMM_ARENA", raising=False)
    elif bypass == "backend":
        load_args["load_config"].remote_instance_weight_loader_backend = "nccl"
    elif bypass == "format":
        load_args["load_config"].load_format = "auto"
    elif bypass == "transport":
        load_args["load_config"].modelexpress_transport = "transfer_engine"
    else:
        load_args["memory_saver_adapter"].enabled = True
    enter = Mock()
    monkeypatch.setattr(registration, "maybe_enter_vmm_arena", enter)
    original = Mock(return_value=object())
    assert registration._load_with_vmm(original, **load_args) is original.return_value
    original.assert_called_once_with(**load_args)
    enter.assert_not_called()
    if bypass == "memory_saver":
        assert "memory saver" in caplog.text


def test_draft_keeps_the_main_arena(load_args, monkeypatch):
    main_arena = object()
    monkeypatch.setattr(runtime, "_vmm_arenas", {0: main_arena})
    load_args["model_config"].is_draft_model = True
    load_args["is_draft_worker"] = True

    def original(**kwargs):
        ctx = adapter.build_sglang_load_context(
            kwargs["load_config"], kwargs["model_config"], _device_config()
        )
        assert ctx.p2p_role == "draft"
        assert ctx.vmm_arena is None
        return "draft"

    assert registration._load_with_vmm(original, **load_args) == "draft"
    assert runtime._vmm_arenas == {0: main_arena}


def test_missing_arena_extension_uses_normal_registration(load_args, monkeypatch):
    monkeypatch.setattr(
        adapter, "accelerator_backend_for",
        lambda device: SimpleNamespace(supports_vmm=lambda: True),
    )
    from modelexpress.vmm import hook

    monkeypatch.setattr(hook, "ARENA_AVAILABLE", False)

    def original(**kwargs):
        ctx = adapter.build_sglang_load_context(
            kwargs["load_config"], kwargs["model_config"], _device_config()
        )
        assert ctx.vmm_arena is None
        return "normal"

    assert registration._load_with_vmm(original, **load_args) == "normal"


@pytest.mark.parametrize("failure", [False, True])
def test_shared_lifecycle_retains_successful_arena_and_closes_failed_load(
    load_args, monkeypatch, mock_accelerator_backend_cls, failure,
):
    from modelexpress.vmm import backend, hook

    monkeypatch.setattr(runtime, "_vmm_arenas", {})
    monkeypatch.setattr(hook, "ARENA_AVAILABLE", True)
    monkeypatch.setattr(backend, "CudaVmmBackend", Mock())
    monkeypatch.setattr(
        adapter, "accelerator_backend_for",
        lambda device: mock_accelerator_backend_cls(torch_device_type="cpu", vmm=True),
    )
    arena = SimpleNamespace(
        base=1024, total_bytes=4096, granularity=256, mapped_bytes=512,
        live_allocation_count=1, registered_range=lambda: (1024, 512), close=Mock(),
    )
    monkeypatch.setattr("modelexpress.vmm.arena.VmmArena", Mock(return_value=arena))
    active = []

    @contextmanager
    def use_arena(value, device):
        active.append(value)
        try:
            yield
        finally:
            active.pop()

    monkeypatch.setattr(hook, "use_arena", use_arena)
    manager = Mock(tensor_descriptors={})
    monkeypatch.setattr(strategy_base, "is_nixl_available", lambda: True)
    monkeypatch.setattr(strategy_base, "_metadata_publication_configured", lambda ctx: True)

    def original(**kwargs):
        assert active == [arena]
        model = torch.nn.Linear(2, 2)
        ctx = adapter.build_sglang_load_context(
            kwargs["load_config"], kwargs["model_config"], _device_config()
        )
        assert ctx.vmm_arena is arena
        ctx.nixl_manager = manager
        strategy_base.register_tensors(model, ctx)
        manager.register_arena.assert_called_once_with(arena, ctx.tensors)
        assert set(ctx.tensors) == {"weight", "bias"}
        manager.register_tensors.assert_not_called()
        if failure:
            raise RuntimeError("initialization failed")
        return model

    if failure:
        with pytest.raises(RuntimeError, match="initialization failed"):
            registration._load_with_vmm(original, **load_args)
        assert runtime._vmm_arenas == {}
        arena.close.assert_called_once()
    else:
        assert isinstance(registration._load_with_vmm(original, **load_args), torch.nn.Module)
        assert runtime._vmm_arenas == {0: arena}
        arena.close.assert_not_called()
    assert active == []


def test_mutated_strategy_retry_stays_in_allocation_scope(load_args, monkeypatch):
    active = []

    @contextmanager
    def enter(ctx):
        active.append(ctx)
        try:
            yield
        finally:
            active.pop()

    monkeypatch.setattr(registration, "maybe_enter_vmm_arena", enter)
    monkeypatch.setattr(registration, "log_arena_post_load", Mock())
    monkeypatch.setattr("modelexpress.load_strategy.publish_source_if_supported", Mock())
    attempts = []

    def failed_load(result, ctx):
        assert active == [ctx]
        attempts.append("failed")
        raise StrategyFailed("partial receive", mutated=True)

    def reinitialize(self, result):
        assert active and active[0].adapter is self
        attempts.append("reinitialize")
        result.model.weight = torch.nn.Parameter(torch.zeros_like(result.model.weight))
        return result

    def fallback_load(result, ctx):
        assert active == [ctx]
        attempts.append("fallback")
        with torch.no_grad():
            result.model.weight.fill_(7)
        return result

    monkeypatch.setattr(adapter.SglangAdapter, "reinit_for_retry", reinitialize)
    strategies = [
        SimpleNamespace(
            name=name, is_available=lambda ctx: True, prepare=Mock(),
            finalize=Mock(), abort=Mock(), rollback=Mock(), load=load,
        )
        for name, load in (("failed", failed_load), ("fallback", fallback_load))
    ]

    def original(**kwargs):
        model = torch.nn.Linear(2, 2)
        ctx = adapter.build_sglang_load_context(
            kwargs["load_config"], kwargs["model_config"], _device_config()
        )
        return execute_load_strategies(model, ctx, strategies)

    loaded = registration._load_with_vmm(original, **load_args)
    assert attempts == ["failed", "reinitialize", "fallback"]
    assert torch.all(loaded.weight == 7)
    assert active == []


def test_context_is_not_reused_for_another_model_or_device(load_args):
    def original(**kwargs):
        current = adapter.build_sglang_load_context(
            kwargs["load_config"], kwargs["model_config"], _device_config()
        )
        other_model = adapter.build_sglang_load_context(
            kwargs["load_config"], _model_config(), _device_config()
        )
        other_device = adapter.build_sglang_load_context(
            kwargs["load_config"], kwargs["model_config"], _device_config(gpu_id=1)
        )
        assert current is not other_model
        assert current is not other_device
        assert other_device.device_id == 1

    registration._load_with_vmm(original, **load_args)
