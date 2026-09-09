# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalized refit timing for the RL generator path.

:mod:`modelexpress.refit.timing` already defines the record and its stage
vocabulary. What was missing is a cycle owner on this path: the recorder is only
populated while one is active, so with nothing activating it the RL client
produced no stages at all, and a framework asking where a refit spent its time
could only see the total.

That gap was measurable from outside. Instrumenting a refit at the framework
boundary attributed about 18% of it, because the boundary can only see
``stage``, ``apply`` and ``release`` -- three calls, one of which contains the
wire transfer, the reconstruction, post-load processing and the copy into kernel
storage, all charged together as "install". The durations were being measured
anyway: the staging path times its own wire and reconstruct phases and puts them
on the staged handle. They just had nowhere to go.

A cycle here spans two client calls, ``stage_weight`` and ``apply_weight``, with
the caller's own work in between at a safe point of its choosing. So it cannot be
one ``with`` block; the recorder is created when staging starts, carried on the
staged handle, and re-activated for the apply. Its ``e2e_ms`` is therefore
staging through install, including the caller's pause, which is why the stage
durations are what to read for transport cost.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any

from modelexpress.refit.timing import (
    RefitTimingRecorder,
    add_refit_bytes,
    current_refit_timing,
    refit_span,
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


def record_measured(
    stage: str,
    seconds: float,
    *,
    metadata: dict[str, Any] | None = None,
    accumulate_metadata: bool = False,
) -> None:
    """Attribute an already-measured duration to a normalized stage.

    The staging path measures the wire read and the reconstruction itself,
    around code that has to be timed from the inside to be timed at all. Those
    numbers are handed here rather than re-measured from outside, where they
    would be bundled together.
    """
    recorder = current_refit_timing()
    if recorder is not None:
        recorder.add_duration(
            stage,
            max(0.0, float(seconds)),
            metadata=metadata,
            accumulate_metadata=accumulate_metadata,
        )


def record_bytes(count: int) -> None:
    """Record payload bytes so the record carries throughput, not just time."""
    add_refit_bytes(int(count))


def record_cold(cold: bool) -> None:
    """Mark whether this cycle had to build a transfer plan or reused one.

    The distinction dominates ``transfer_planning``: a reused plan makes it
    almost free, so a mean over both is a number that describes neither.
    """
    recorder = current_refit_timing()
    if recorder is not None:
        recorder.set_cold(cold)


__all__ = [
    "BACKEND",
    "active",
    "emit",
    "record_bytes",
    "record_cold",
    "record_measured",
    "refit_span",
    "start_cycle",
]
