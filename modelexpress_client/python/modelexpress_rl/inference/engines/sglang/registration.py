# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed SGLang plugin for ModelExpress RL."""

from modelexpress import envs
from .. import configure_engine_logging

_registered = False


def register() -> None:
    """Select the RL loader through SGLang's native class replacement hook."""
    global _registered
    configure_engine_logging("sglang")
    if _registered or envs.MX_LOAD_STRATEGY_CHAIN != "RL":
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from .loader import RLMxModelLoader

    HookRegistry.register(
        "modelexpress.engines.sglang.loader.MxModelLoader",
        RLMxModelLoader,
        hook_type=HookType.REPLACE,
    )
    _registered = True
