# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang memory releases stop and refill ModelExpress-loaded weights."""

import sys
from types import ModuleType, SimpleNamespace

import torch

from modelexpress.engines.sglang import loader as loader_mod
from modelexpress.engines.sglang import registration


def test_register_refills_weights_around_sglang_memory_release(monkeypatch):
    registered = []
    for name in ("sglang", "sglang.srt", "sglang.srt.plugins"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    hooks = ModuleType("sglang.srt.plugins.hook_registry")
    hooks.HookType = SimpleNamespace(BEFORE="before", AFTER="after")
    hooks.HookRegistry = SimpleNamespace(
        register=lambda target, fn, hook_type: registered.append(
            (target.rsplit(".", 1)[1], fn, hook_type)
        )
    )
    monkeypatch.setitem(sys.modules, hooks.__name__, hooks)
    constants = ModuleType("sglang.srt.constants")
    constants.GPU_MEMORY_TYPE_WEIGHTS = "weights"
    monkeypatch.setitem(sys.modules, constants.__name__, constants)
    monkeypatch.setattr(registration, "_registered", False)

    registration.register()
    registration.register()
    assert [(name, hook_type) for name, _, hook_type in registered] == [
        ("release_memory_occupation", "before"),
        ("resume_memory_occupation", "after"),
    ]
    (_, before_release, _), (_, after_resume, _) = registered

    calls = []
    monkeypatch.setattr(loader_mod, "release_weights", lambda d: calls.append(("release", d)))
    monkeypatch.setattr(loader_mod, "refill_weights", lambda d: calls.append(("refill", d)))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        torch.distributed, "barrier", lambda group: calls.append(("barrier", group))
    )
    manager = SimpleNamespace(tp_cpu_group="tp")

    before_release(manager, SimpleNamespace(tags=["kv_cache"]))
    after_resume(None, manager, SimpleNamespace(tags=["kv_cache"]))
    assert calls == []

    before_release(manager, SimpleNamespace(tags=None))
    after_resume(None, manager, SimpleNamespace(tags=["weights", "kv_cache"]))
    assert calls == [("release", 3), ("refill", 3), ("barrier", "tp")]
