# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed SGLang plugin for ModelExpress."""

from __future__ import annotations

import logging

from ... import configure_sglang_logging, envs
from .p2p_metadata_sync import before_server_args_post_init, sync_server_args_metadata

logger = logging.getLogger("modelexpress.engines.sglang.registration")

_registered = False


def register() -> None:
    """Register ModelExpress SGLang hooks via SGLang's native plugin framework."""
    global _registered
    configure_sglang_logging()
    if (
        _registered
        or envs.MX_DISABLE_PATCHES
        or not envs.MX_SGLANG_P2P_METADATA_SYNC
    ):
        return

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs.__post_init__",
        before_server_args_post_init,
        hook_type=HookType.BEFORE,
    )

    try:
        from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook

        @register_resolution_hook("handle_model_source_paths")
        def _handle_model_source_paths(server_args, previous):
            sync_server_args_metadata(server_args)
            previous(server_args)

    except Exception:
        logger.debug(
            "Resolution hooks unavailable; relying on ServerArgs.__post_init__ hook"
        )

    _registered = True
