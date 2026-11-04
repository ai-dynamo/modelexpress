# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang installs the RL loader through its replacement hook."""

import sys
from types import ModuleType

import pytest
import torch.nn as nn

from tests.test_sglang_loader import (
    _device_config,
    _load_config,
    _model_config,
)
from modelexpress_rl.inference.engines.sglang.loader import RLMxModelLoader


def test_register_replaces_base_loader_only_for_rl(monkeypatch):
    from modelexpress_rl.inference.engines.sglang import registration

    calls = []
    for name in ("sglang", "sglang.srt", "sglang.srt.plugins"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    hooks = ModuleType("sglang.srt.plugins.hook_registry")
    hooks.HookType = type("HookType", (), {"REPLACE": "replace"})
    hooks.HookRegistry = type(
        "HookRegistry",
        (),
        {"register": staticmethod(lambda *args, **kwargs: calls.append((args, kwargs)))},
    )
    monkeypatch.setitem(sys.modules, hooks.__name__, hooks)
    monkeypatch.setattr(registration, "_registered", False)
    monkeypatch.setattr(registration, "configure_engine_logging", lambda engine: None)

    monkeypatch.setenv("MX_LOAD_STRATEGY_CHAIN", "INFERENCE")
    registration.register()
    assert calls == []

    monkeypatch.setenv("MX_LOAD_STRATEGY_CHAIN", "RL")
    registration.register()
    registration.register()
    assert calls == [
        (("modelexpress.engines.sglang.loader.MxModelLoader", RLMxModelLoader),
         {"hook_type": hooks.HookType.REPLACE})
    ]


def test_sglang_rl_chain_rejects_draft(monkeypatch):
    monkeypatch.setenv("MX_LOAD_STRATEGY_CHAIN", "RL")
    loader = RLMxModelLoader(_load_config(modelexpress_transport="nixl"))
    with pytest.raises(ValueError, match="speculative draft"):
        loader.load_model(
            model=nn.Linear(2, 2),
            model_config=_model_config(is_draft_model=True),
            device_config=_device_config(),
        )
