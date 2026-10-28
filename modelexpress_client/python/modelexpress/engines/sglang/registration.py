# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed SGLang plugin that refills released weights over ModelExpress P2P.

SGLang's ``release_memory_occupation`` discards the weights it loaded through
ModelExpress, and ``resume_memory_occupation`` maps fresh memory behind the
same addresses. The hooks stop serving the weights before the release and
receive them again from a peer after the resume, in each scheduler process.
"""

from __future__ import annotations

import torch

_WEIGHT_UPDATER = (
    "sglang.srt.managers.scheduler_components.weight_updater."
    "SchedulerWeightUpdaterManager"
)

_registered = False


def register() -> None:
    global _registered
    if _registered:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        f"{_WEIGHT_UPDATER}.release_memory_occupation",
        _before_release,
        hook_type=HookType.BEFORE,
    )
    HookRegistry.register(
        f"{_WEIGHT_UPDATER}.resume_memory_occupation",
        _after_resume,
        hook_type=HookType.AFTER,
    )
    _registered = True


def _covers_weights(recv_req) -> bool:
    from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS

    return not recv_req.tags or GPU_MEMORY_TYPE_WEIGHTS in recv_req.tags


def _before_release(manager, recv_req) -> None:
    if _covers_weights(recv_req):
        from .loader import release_weights

        release_weights(torch.cuda.current_device())


def _after_resume(result, manager, recv_req) -> None:
    if _covers_weights(recv_req):
        from .loader import refill_weights

        refill_weights(torch.cuda.current_device())
        # Only the entry rank replies to the resume, so without this the
        # caller could see it finish while other ranks are still receiving.
        torch.distributed.barrier(manager.tp_cpu_group)
