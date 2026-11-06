# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native startup callbacks gate artifacts unless HTTP is explicitly selected."""

import asyncio
import gc
import sys
from types import ModuleType, SimpleNamespace

import pytest

from modelexpress.engines import startup
from modelexpress.engines.sglang import artifacts as sglang_artifacts
from modelexpress.engines.sglang import startup as sglang_startup
from modelexpress.engines.vllm import artifacts as vllm_artifacts
from modelexpress.engines.vllm import startup as vllm_startup
from modelexpress.metadata import artifact_lifecycle
from modelexpress.metadata.artifact_transfer import ArtifactCacheRoot


class _Config:
    pass


@pytest.fixture(autouse=True)
def isolate_startup(monkeypatch):
    monkeypatch.setenv("MX_ARTIFACT_TRANSFER", "1")
    monkeypatch.delenv("MX_ARTIFACT_READY_URL", raising=False)
    monkeypatch.setattr(vllm_startup, "_gates", {})
    monkeypatch.setattr(sglang_startup, "gate", startup.StartupGate())


def _context():
    return SimpleNamespace(adapter=SimpleNamespace(vllm_config=_Config()))


def _workers(count=2):
    contexts = [_context() for _ in range(count)]
    workers = []
    for ctx in contexts:
        assert vllm_startup.prepare_artifact_gate(ctx)
        worker = vllm_startup.ArtifactStartupWorkerExtension()
        worker.vllm_config = ctx.adapter.vllm_config
        workers.append(worker)
    return workers, contexts


@pytest.mark.parametrize("configured", [None, "", " ", "\t\n"])
@pytest.mark.parametrize("engine", [vllm_artifacts, sglang_artifacts])
def test_blank_url_defaults_to_startup_gate(engine, configured, monkeypatch, tmp_path):
    if configured is not None:
        monkeypatch.setenv("MX_ARTIFACT_READY_URL", configured)
    if engine is vllm_artifacts:
        workers, contexts = _workers(1)
        ctx = contexts[0]
        complete = workers[0].mx_artifact_startup_complete
        name = "vllm"
    else:
        sglang_startup.gate.prepare()
        ctx = None
        complete = lambda: sglang_startup.gate.complete(lambda: None)
        name = "sglang"
    root = tmp_path / "cache"
    root.mkdir()
    (root / "kernel.so").write_bytes(b"compiled")
    clock = [100.0]
    monkeypatch.setattr(artifact_lifecycle.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        engine, f"_{name}_health_ready", lambda *args: pytest.fail("HTTP probe")
    )
    ready = getattr(engine, f"_{name}_artifact_ready_fn")(
        (ArtifactCacheRoot("primary", root, root),), ctx=ctx
    )
    assert not ready()
    clock[0] += 100
    assert not ready()
    complete()
    assert not ready()
    clock[0] += artifact_lifecycle.CACHE_SETTLE_SECS
    assert ready()
    (root / "kernel.so").write_bytes(b"new compilation")
    assert not ready()


@pytest.mark.parametrize("engine", [vllm_artifacts, sglang_artifacts])
def test_explicit_url_overrides_completed_startup(engine, monkeypatch, tmp_path):
    workers, contexts = _workers(1)
    workers[0].mx_artifact_startup_complete()
    sglang_startup.gate.prepare()
    sglang_startup.gate.complete(lambda: None)
    monkeypatch.setenv("MX_ARTIFACT_READY_URL", "http://custom/ready")
    name = "vllm" if engine is vllm_artifacts else "sglang"
    calls = []
    monkeypatch.setattr(engine, f"_{name}_health_ready", lambda ctx: calls.append(ctx) or False)
    ready = getattr(engine, f"_{name}_artifact_ready_fn")((), ctx=contexts[0])
    assert not ready()
    assert calls == [contexts[0]]


def test_native_endpoint_callback_signals_workers_without_patches(monkeypatch):
    monkeypatch.setenv("MX_DISABLE_PATCHES", "1")
    workers, contexts = _workers()
    calls = []

    class EngineClient:
        async def collective_rpc(self, method, *, timeout):
            calls.append((method, timeout))
            return [getattr(worker, method)() for worker in workers]

    assert not any(ctx.artifact_ready_fn() for ctx in contexts)
    plugin = vllm_startup.ArtifactStartupPlugin()
    plugin.attach_router(object())
    asyncio.run(plugin.init_state(EngineClient(), object(), object()))
    assert calls[0][0] == "mx_artifact_startup_complete"
    assert all(ctx.artifact_ready_fn() for ctx in contexts)
    asyncio.run(plugin.init_state(EngineClient(), object(), object()))
    assert all(ctx.artifact_ready_fn() for ctx in contexts)


def test_native_callback_failure_leaves_publishers_closed():
    workers, contexts = _workers()

    class EngineClient:
        async def collective_rpc(self, method, *, timeout):
            raise RuntimeError("worker extension not installed")

    asyncio.run(vllm_startup.ArtifactStartupPlugin().init_state(EngineClient(), None, None))
    assert not any(ctx.artifact_ready_fn() for ctx in contexts)


@pytest.mark.parametrize("configured", [None, "", " \t"])
def test_endpoint_uses_startup_only_for_blank_url(configured, monkeypatch):
    workers, contexts = _workers(1)
    if configured is not None:
        monkeypatch.setenv("MX_ARTIFACT_READY_URL", configured)

    class EngineClient:
        async def collective_rpc(self, method, *, timeout):
            return [getattr(workers[0], method)()]

    asyncio.run(vllm_startup.ArtifactStartupPlugin().init_state(EngineClient(), None, None))
    assert contexts[0].artifact_ready_fn()


def test_explicit_url_does_not_require_worker_extension(monkeypatch):
    monkeypatch.setenv("MX_ARTIFACT_READY_URL", "http://custom/ready")

    class EngineClient:
        async def collective_rpc(self, *args, **kwargs):
            pytest.fail("explicit HTTP override must not require startup RPC")

    asyncio.run(vllm_startup.ArtifactStartupPlugin().init_state(EngineClient(), None, None))


def test_render_server_has_no_startup_signal():
    asyncio.run(vllm_startup.ArtifactStartupPlugin().init_state(None, None, None))


def test_engine_instances_and_reload_are_isolated():
    workers, contexts = _workers()
    workers[0].mx_artifact_startup_complete()
    assert contexts[0].artifact_ready_fn()
    assert not contexts[1].artifact_ready_fn()
    reloaded = SimpleNamespace(adapter=contexts[0].adapter)
    assert vllm_startup.prepare_artifact_gate(reloaded)
    assert reloaded.artifact_ready_fn()


def test_gate_lifetime_tracks_native_config():
    import weakref

    workers, contexts = _workers(1)
    config = contexts[0].adapter.vllm_config
    config_ref = weakref.ref(config)
    gate_ref = weakref.ref(contexts[0].artifact_ready_fn.__self__)
    contexts.clear()
    gc.collect()
    reloaded = SimpleNamespace(adapter=SimpleNamespace(vllm_config=config))
    assert vllm_startup.prepare_artifact_gate(reloaded)
    workers[0].mx_artifact_startup_complete()
    assert reloaded.artifact_ready_fn()
    del workers, reloaded, config
    gc.collect()
    assert config_ref() is None
    assert gate_ref() is None


def test_missing_startup_signal_never_falls_back_to_http(monkeypatch):
    workers, contexts = _workers(1)
    monkeypatch.setattr(vllm_artifacts, "_vllm_health_ready", lambda *a: pytest.fail("HTTP"))
    assert not vllm_artifacts._vllm_artifact_ready_fn((), ctx=contexts[0])()


def test_external_launcher_requires_explicit_readiness():
    ctx = _context()
    ctx.adapter.vllm_config.parallel_config = SimpleNamespace(
        distributed_executor_backend="external_launcher"
    )
    assert not vllm_startup.prepare_artifact_gate(ctx)


def test_sglang_registration_uses_native_hook_registry(monkeypatch):
    hooks = []
    registry = ModuleType("sglang.srt.plugins.hook_registry")
    registry.HookType = SimpleNamespace(AROUND="around")
    registry.HookRegistry = SimpleNamespace(register=lambda *a, **kw: hooks.append((a, kw)))
    monkeypatch.setitem(sys.modules, registry.__name__, registry)
    monkeypatch.setattr(sglang_startup, "_registered", False)
    monkeypatch.delenv("MX_DISABLE_PATCHES", raising=False)
    monkeypatch.setattr(sglang_startup, "_synchronize", lambda: None)
    sglang_startup.register_startup_hook()
    sglang_startup.register_startup_hook()
    assert len(hooks) == 1
    (target, hook), options = hooks[0]
    assert target == "sglang.srt.managers.scheduler.Scheduler.__init__"
    assert options == {"hook_type": "around"}
    hook(lambda self: None, object())
    assert sglang_startup.gate.is_ready()
