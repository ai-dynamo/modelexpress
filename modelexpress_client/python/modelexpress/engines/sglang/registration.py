# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enter the VMM arena before SGLang constructs the model."""

import logging

from ... import envs
from ...vmm.runtime import log_arena_post_load, maybe_enter_vmm_arena
from .adapter import _active_load_context, build_sglang_load_context

logger = logging.getLogger(__name__)
_registered = False


def register() -> None:
    """Install the optional allocation envelope through SGLang's plugin API."""
    global _registered
    if _registered or not envs.MX_VMM_ARENA:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.model_executor.model_runner_components.load_model_utils."
        "load_model_with_memory_saver",
        _load_with_vmm,
        hook_type=HookType.AROUND,
    )
    _registered = True


def _load_with_vmm(original, **kwargs):
    """Share the arena's context with the inner MX loader and reset on exit."""
    load_config = kwargs["load_config"]
    load_format = getattr(load_config, "load_format", None)
    backend = getattr(load_config, "remote_instance_weight_loader_backend", None)
    if (
        not envs.MX_VMM_ARENA
        or getattr(load_format, "value", load_format) != "remote_instance"
        or getattr(backend, "value", backend) != "modelexpress"
        or getattr(load_config, "modelexpress_transport", "nixl") != "nixl"
    ):
        return original(**kwargs)
    if kwargs["memory_saver_adapter"].enabled:
        logger.warning(
            "MX_VMM_ARENA=1 with SGLang memory saver is unsupported; "
            "keeping SGLang's allocator and normal NIXL registration"
        )
        return original(**kwargs)

    from sglang.srt.configs.device_config import DeviceConfig

    ctx = build_sglang_load_context(
        load_config,
        kwargs["model_config"],
        DeviceConfig(kwargs["device"], kwargs["gpu_id"]),
    )
    ctx.p2p_role = "draft" if kwargs["is_draft_worker"] else "main"
    token = _active_load_context.set(ctx)
    try:
        with maybe_enter_vmm_arena(ctx):
            result = original(**kwargs)
        if ctx.p2p_role == "main":
            log_arena_post_load(ctx)
        return result
    finally:
        _active_load_context.reset(token)
