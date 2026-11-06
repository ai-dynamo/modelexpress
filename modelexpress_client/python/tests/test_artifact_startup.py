# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Artifact publishers must observe completed engine startup, not loaded weights."""

import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

from modelexpress.engines import startup
from modelexpress.engines.sglang import artifacts as sglang_artifacts
from modelexpress.engines.sglang import startup as sglang_startup
from modelexpress.engines.vllm import artifacts as vllm_artifacts
from modelexpress.engines.vllm import startup as vllm_startup
from modelexpress.metadata import artifact_lifecycle
from modelexpress.metadata.artifact_transfer import ArtifactCacheRoot


@pytest.fixture(autouse=True)
def isolate_gates(monkeypatch):
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "1")
    monkeypatch.delenv("MX_ARTIFACT_READY_URL", raising=False)
    monkeypatch.delenv("MX_DISABLE_PATCHES", raising=False)
    for engine in (vllm_startup, sglang_startup):
        monkeypatch.setattr(engine, "gate", startup.StartupGate())


@pytest.mark.parametrize("engine", [vllm_artifacts, sglang_artifacts])
def test_artifacts_wait_for_startup_and_cache_settling(engine, monkeypatch, tmp_path):
    signal = vllm_startup if engine is vllm_artifacts else sglang_startup
    signal.gate.prepare()
    root = tmp_path / "cache"
    root.mkdir()
    (root / "kernel.so").write_bytes(b"compiled")
    roots = (ArtifactCacheRoot("primary", root, root),)
    clock = [100.0]
    monkeypatch.setattr(artifact_lifecycle.time, "monotonic", lambda: clock[0])

    def unexpected_http(*args):
        pytest.fail("startup hook must replace HTTP polling")

    name = "vllm" if engine is vllm_artifacts else "sglang"
    monkeypatch.setattr(engine, f"_{name}_health_ready", unexpected_http)
    ready = getattr(engine, f"_{name}_artifact_ready_fn")(roots)
    assert ready() is False
    clock[0] += 100
    assert ready() is False
    signal.gate.complete(lambda: None)
    assert ready() is False
    clock[0] += artifact_lifecycle.CACHE_SETTLE_SECS
    assert ready() is True
    (root / "kernel.so").write_bytes(b"updated compilation")
    assert ready() is False


@pytest.mark.parametrize("engine", [vllm_artifacts, sglang_artifacts])
@pytest.mark.parametrize("explicit", [False, True])
def test_artifacts_keep_http_override_and_unsupported_fallback(
    engine, explicit, monkeypatch, tmp_path
):
    signal = vllm_startup if engine is vllm_artifacts else sglang_startup
    if explicit:
        signal.gate.prepare()
        signal.gate.complete(lambda: None)
        monkeypatch.setenv("MX_ARTIFACT_READY_URL", "http://custom/ready")
    root = tmp_path / "cache"
    root.mkdir()
    (root / "kernel.so").write_bytes(b"compiled")
    calls = []
    name = "vllm" if engine is vllm_artifacts else "sglang"
    monkeypatch.setattr(
        engine, f"_{name}_health_ready", lambda ctx: calls.append(ctx) or False
    )
    ready = getattr(engine, f"_{name}_artifact_ready_fn")(
        (ArtifactCacheRoot("primary", root, root),), ctx=None
    )
    assert ready() is False
    assert calls == [None]


def test_gate_waits_for_other_writers():
    gate = startup.StartupGate()
    gate.prepare()
    entered = threading.Event()
    release = threading.Event()

    def synchronize():
        entered.set()
        assert release.wait(timeout=5)

    thread = threading.Thread(target=gate.complete, args=(synchronize,))
    thread.start()
    try:
        assert entered.wait(timeout=5)
        assert gate.is_ready() is False
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert gate.is_ready() is True


def test_failed_barrier_does_not_release_publisher():
    gate = startup.StartupGate()
    gate.prepare()

    def failed():
        raise RuntimeError("peer failed startup")

    gate.complete(failed)
    assert gate.is_ready() is False


def _worker_module(monkeypatch, warmup):
    module = ModuleType("vllm.v1.worker.gpu_worker")
    module.Worker = type("Worker", (), {"compile_or_warm_up_model": warmup})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.Worker


def test_vllm_wrapper_preserves_result_and_waits_for_world(monkeypatch):
    calls = []
    result = object()

    def warmup(self, value, *, option):
        assert not vllm_startup.gate.is_ready()
        calls.append((value, option))
        return result

    worker = _worker_module(monkeypatch, warmup)
    monkeypatch.setattr(vllm_startup, "_synchronize", lambda: calls.append("barrier"))
    assert vllm_startup.install_startup_hook(SimpleNamespace()) is True
    assert vllm_startup.install_startup_hook(SimpleNamespace()) is True
    assert worker().compile_or_warm_up_model(7, option="test") is result
    assert calls == [(7, "test"), "barrier"]
    assert vllm_startup.gate.is_ready() is True


def test_vllm_failed_warmup_never_signals_or_enters_barrier(monkeypatch):
    def warmup(self):
        raise ValueError("compilation failed")

    worker = _worker_module(monkeypatch, warmup)
    monkeypatch.setattr(vllm_startup, "_synchronize", lambda: pytest.fail("barrier"))
    vllm_startup.install_startup_hook(SimpleNamespace())
    with pytest.raises(ValueError, match="compilation failed"):
        worker().compile_or_warm_up_model()
    assert vllm_startup.gate.is_ready() is False


def test_vllm_repeated_warmup_clears_previous_signal(monkeypatch):
    def warmup(self, fail=False):
        if fail:
            raise ValueError("second warmup failed")

    worker = _worker_module(monkeypatch, warmup)
    monkeypatch.setattr(vllm_startup, "_synchronize", lambda: None)
    vllm_startup.install_startup_hook(SimpleNamespace())
    instance = worker()
    instance.compile_or_warm_up_model()
    assert vllm_startup.gate.is_ready()
    with pytest.raises(ValueError, match="second warmup failed"):
        instance.compile_or_warm_up_model(fail=True)
    assert not vllm_startup.gate.is_ready()


def test_vllm_reload_keeps_completed_startup(monkeypatch):
    worker = _worker_module(monkeypatch, lambda self: None)
    monkeypatch.setattr(vllm_startup, "_synchronize", lambda: None)
    vllm_startup.install_startup_hook(SimpleNamespace())
    worker().compile_or_warm_up_model()
    assert vllm_startup.gate.is_ready()
    assert vllm_startup.install_startup_hook(SimpleNamespace(is_reload=True))
    assert vllm_startup.gate.is_ready()


@pytest.mark.parametrize(
    "setting,value",
    [
        ("MX_DISABLE_PATCHES", "1"),
        ("MX_ARTIFACT_READY_URL", "http://custom/ready"),
        ("MX_ARTIFACT_TRANSFER", "0"),
    ],
)
def test_vllm_hook_respects_opt_out(monkeypatch, setting, value):
    worker = _worker_module(monkeypatch, lambda self: None)
    original = worker.compile_or_warm_up_model
    monkeypatch.setenv(setting, value)
    assert vllm_startup.install_startup_hook(SimpleNamespace()) is False
    assert worker.compile_or_warm_up_model is original
    assert vllm_startup.gate.available is False


def test_vllm_resolves_configured_worker(monkeypatch):
    module = ModuleType("custom_mx_worker")
    module.Worker = type("Worker", (), {"compile_or_warm_up_model": lambda self: 42})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config = SimpleNamespace(parallel_config=SimpleNamespace(worker_cls="custom_mx_worker.Worker"))
    ctx = SimpleNamespace(adapter=SimpleNamespace(vllm_config=config))
    monkeypatch.setattr(vllm_startup, "_synchronize", lambda: None)
    assert vllm_startup.install_startup_hook(ctx)
    assert module.Worker().compile_or_warm_up_model() == 42
    assert vllm_startup.gate.is_ready()


def test_vllm_unsupported_worker_retains_fallback(monkeypatch):
    module = ModuleType("vllm.v1.worker.gpu_worker")
    module.Worker = type("Worker", (), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert vllm_startup.install_startup_hook(SimpleNamespace()) is False
    assert vllm_startup.gate.available is False


def test_sglang_registration_wraps_scheduler_once(monkeypatch):
    hooks = []
    registry = ModuleType("sglang.srt.plugins.hook_registry")
    registry.HookType = SimpleNamespace(AROUND="around")
    registry.HookRegistry = SimpleNamespace(register=lambda *a, **kw: hooks.append((a, kw)))
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    monkeypatch.setattr(sglang_startup, "_registered", False)
    calls = []
    monkeypatch.setattr(sglang_startup, "_synchronize", lambda: calls.append("barrier"))
    sglang_startup.register_startup_hook()
    sglang_startup.register_startup_hook()
    assert len(hooks) == 1
    (target, hook), options = hooks[0]
    assert target == "sglang.srt.managers.scheduler.Scheduler.__init__"
    assert options == {"hook_type": "around"}

    def initialize(self):
        assert sglang_startup.gate.available
        assert not sglang_startup.gate.is_ready()
        calls.append("init")

    assert hook(initialize, object()) is None
    assert calls == ["init", "barrier"]
    assert sglang_startup.gate.is_ready() is True


def test_sglang_failed_init_never_signals(monkeypatch):
    monkeypatch.setattr(sglang_startup, "_synchronize", lambda: pytest.fail("barrier"))

    def initialize(self):
        raise RuntimeError("capture failed")

    with pytest.raises(RuntimeError, match="capture failed"):
        sglang_startup._scheduler_startup(initialize, object())
    assert sglang_startup.gate.available
    assert not sglang_startup.gate.is_ready()


@pytest.mark.parametrize(
    "setting,value",
    [
        ("MX_DISABLE_PATCHES", "1"),
        ("MX_ARTIFACT_READY_URL", "http://custom/ready"),
        ("MX_ARTIFACT_TRANSFER", "0"),
    ],
)
def test_sglang_hook_respects_opt_out(monkeypatch, setting, value):
    monkeypatch.setattr(sglang_startup, "_registered", False)
    monkeypatch.setenv(setting, value)
    sglang_startup.register_startup_hook()
    assert not sglang_startup._registered
    assert not sglang_startup.gate.available
