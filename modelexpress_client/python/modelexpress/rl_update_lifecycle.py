# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-agnostic lifecycle hooks around RL weight updates."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, AsyncIterable, Callable
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any, TypeVar

logger = logging.getLogger("modelexpress.rl_update_lifecycle")

_T = TypeVar("_T")
_Hook = Callable[[], Any]


@dataclass(frozen=True)
class RlWeightUpdateLifecycleHooks:
    """Optional rollout-engine hooks around an RL weight update."""

    pause_generation: _Hook | None = None
    flush_cache: _Hook | None = None
    resume_generation: _Hook | None = None

    def __post_init__(self) -> None:
        for name in ("pause_generation", "flush_cache", "resume_generation"):
            hook = getattr(self, name)
            if hook is not None and not callable(hook):
                raise ValueError(f"{name} hook must be callable")

    @property
    def enabled(self) -> bool:
        return any(
            hook is not None
            for hook in (
                self.pause_generation,
                self.flush_cache,
                self.resume_generation,
            )
        )


async def iter_weight_update_lifecycle(
    items: AsyncIterable[_T],
    *,
    hooks: RlWeightUpdateLifecycleHooks | None = None,
) -> AsyncGenerator[_T, None]:
    """Yield update items inside pause/refit/flush/resume hooks.

    The caller owns the actual refit by consuming the yielded items. Resume runs
    in ``finally`` so a failed receive, failed refit, or cancellation does not
    leave generation paused.
    """
    active_hooks = hooks or RlWeightUpdateLifecycleHooks()
    if not active_hooks.enabled:
        async for item in items:
            yield item
        return

    primary_exc: BaseException | None = None
    try:
        await _call_hook(active_hooks.pause_generation)
        async for item in items:
            yield item
        await _call_hook(active_hooks.flush_cache)
    except BaseException as exc:
        primary_exc = exc
        raise
    finally:
        try:
            await _call_hook(active_hooks.resume_generation)
        except Exception:
            if primary_exc is None:
                raise
            logger.warning(
                "ModelExpress RL resume hook failed after weight-update error",
                exc_info=True,
            )


async def _call_hook(hook: _Hook | None) -> None:
    if hook is None:
        return
    result = hook()
    if isawaitable(result):
        await result
