# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refit cycle ownership for the RL generator path.

:mod:`modelexpress.refit.timing` owns the record, the stage vocabulary and the
recording helpers. This module owns only the thing that was missing on this
path: a cycle. The recorder is populated while one is active, so with nothing
activating it the RL client produced no stages at all, and a framework asking
where a refit spent its time could only see the total.

That gap was measurable from outside. Instrumenting a refit at the framework
boundary attributed about 18% of it, because the boundary can only see
``stage``, ``apply`` and ``release`` -- three calls, one of which contains the
wire transfer, the reconstruction, post-load processing and the copy into kernel
storage, all charged together as "install".

A cycle here spans two client calls, ``stage_weight`` and ``apply_weight``, with
the caller's own work in between at a safe point of its choosing. So it cannot be
one ``with`` block; the recorder is created when staging starts, carried on the
staged handle, and re-activated for the apply. Its ``e2e_ms`` is therefore
staging through install, including the caller's pause, which is why the stage
durations are what to read for transport cost.

Layers contributing stages inside a cycle should reach for
:func:`modelexpress.refit.timing.refit_span` directly rather than anything here.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

from modelexpress.refit.timing import (
    RefitTimingRecorder,
    current_refit_timing,
    use_refit_timing,
)

from . import envs

BACKEND = "rl_generator"


def start_cycle(
    *,
    version_id: str,
    rank: int | None = None,
    backend: str = BACKEND,
) -> RefitTimingRecorder | None:
    """Open a recorder for one generator refit, or ``None`` when disabled.

    Returning ``None`` rather than a dummy keeps the disabled path free of the
    recorder entirely, and every consumer here already has to handle the absence
    of a cycle, since lower layers are also reachable from callers that never
    started one.
    """
    if not envs.MX_REFIT_TIMING:
        return None
    if current_refit_timing() is not None:
        # A caller driving its own cycle wins. Nesting a second recorder would
        # split one refit across two records and leave both looking incomplete.
        return None
    return RefitTimingRecorder(backend=backend, version=version_id, rank=rank)


@contextlib.contextmanager
def active(recorder: RefitTimingRecorder | None) -> Iterator[None]:
    """Make ``recorder`` visible to the layers that contribute stages."""
    if recorder is None:
        yield
        return
    with use_refit_timing(recorder):
        yield


def emit(
    recorder: RefitTimingRecorder | None, logger: logging.Logger
) -> dict[str, Any] | None:
    """Emit the cycle's record, once, if there is one.

    Idempotent in the recorder, which is what lets the client call this from
    both the apply and the release path: whichever ends the cycle reports it,
    and a refit that failed before applying is still reported rather than
    vanishing.
    """
    if recorder is not None:
        return recorder.emit(logger)
    return None


__all__ = [
    "BACKEND",
    "active",
    "emit",
    "start_cycle",
]
