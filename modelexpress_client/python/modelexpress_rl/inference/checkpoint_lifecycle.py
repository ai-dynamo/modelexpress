# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-owned serving controls for collective checkpoint installation."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from typing import Any

from .checkpoint_transaction import CheckpointTransactionError


@dataclass(frozen=True)
class CheckpointCollectiveContext:
    """Required framework integration for partial checkpoint mode.

    ``safe_point`` must pause and drain inference on entry, resume only on
    successful exit, and never resume after ``fence``. ``fence`` must stop
    inference and publication until coordinated worker recovery, including
    after a transport timeout. These are serving controls, not Python flags.

    ``all_gather(value, timeout_seconds)`` must include every worker rank in
    deterministic rank order and enforce the supplied finite deadline. Use a
    dedicated control transport; do not interleave unrelated collectives.
    Every rank must use the same configuration and enter each update together.
    """

    world_size: int
    all_gather: Callable[[object, float], tuple[object, ...]]
    safe_point: Callable[[], AbstractContextManager]
    fence: Callable[[], None]
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if type(self.world_size) is not int or self.world_size < 1:
            raise ValueError("checkpoint collective world_size must be positive")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError(
                "checkpoint collective timeout must be finite and positive"
            )
        for name in ("all_gather", "safe_point", "fence"):
            if not callable(getattr(self, name)):
                raise TypeError(f"checkpoint collective {name} must be callable")

    def gather(self, value: object) -> tuple[object, ...]:
        return self.all_gather(value, self.timeout_seconds)


def run_checkpoint_lifecycle(
    context: CheckpointCollectiveContext,
    *,
    serving_version: str | None,
    target_version: str,
    validate: Callable[[], None],
    unpublish: Callable[[], None],
    install: Callable[[], Any],
    publish: Callable[[], None],
    fence: Callable[[], None],
) -> Any:
    """Keep inference paused through collective install, cleanup and publication."""

    def phase(name: str, action: Callable[[], Any]) -> Any:
        error = None
        result = None
        try:
            result = action()
        except BaseException as failure:  # noqa: BLE001 - all ranks must fence
            error = failure
        state = ("lifecycle", name, serving_version, target_version, error is None)
        peers = context.gather(state)
        if len(peers) != context.world_size or any(peer != state for peer in peers):
            raise CheckpointTransactionError(
                f"checkpoint lifecycle disagreement during {name}"
            ) from error
        if error is not None:
            raise CheckpointTransactionError(
                f"checkpoint lifecycle failed during {name}"
            ) from error
        return result

    stack = ExitStack()
    try:
        phase("validated", validate)
        phase("paused", lambda: stack.enter_context(context.safe_point()))
        phase("unpublished", unpublish)
        result = phase("installed", install)
        phase("published", publish)
        phase("resumed", stack.close)
        return result
    except BaseException:
        # Fence before exit; a context must not suppress a failed transaction.
        try:
            fence()
        finally:
            stack.__exit__(*sys.exc_info())
        raise
