# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Signal artifact publishers after SGLang scheduler initialization."""

from ... import envs
from ..startup import StartupGate

gate = StartupGate()
_registered = False


def _synchronize() -> None:
    from sglang.srt.distributed.parallel_state import get_world_group

    get_world_group().barrier()


def _scheduler_startup(original, self, *args, **kwargs):
    gate.prepare()
    result = original(self, *args, **kwargs)
    gate.complete(_synchronize)
    return result


def register_startup_hook() -> None:
    """Register through SGLang's native, process-local plugin hook registry."""
    global _registered
    if (
        _registered
        or not envs.MX_ARTIFACT_TRANSFER
        or envs.MX_DISABLE_PATCHES
        or envs.MX_ARTIFACT_READY_URL.strip()
    ):
        return
    try:
        from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    except ImportError:
        return
    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler.__init__",
        _scheduler_startup,
        hook_type=HookType.AROUND,
    )
    _registered = True
