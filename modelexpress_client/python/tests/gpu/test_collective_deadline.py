# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bounded lane wait, on a device.

MX_NCCL_REFIT_TRANSFER_TIMEOUT_S is only worth anything if the wait it bounds
can actually be interrupted. A plain stream.synchronize() cannot be bounded
from Python, so the deadline records a CUDA event and polls it - and that
polling path needs a device, which means the unit suite cannot reach it. What
runs on a CPU box is the arming, the expiry, the abort and the fallback; every
one of those can be green while the poll itself is broken.

The allow arm is the one that matters here. A deny arm needs nothing external
and red-tests cleanly; admitting requires the event to record and query on a
real stream, and if _synchronize_bounded returned False on a GPU box then every
deadline in production would silently degrade to the blocking wait it was
written to replace, with no test anywhere going red.
"""

from __future__ import annotations

import time

import pytest
import torch

from modelexpress_rl.collective.comm import LaneCommunicator

pytestmark = pytest.mark.gpu

#: Long enough that a blocking wait is unmistakable against the deadline below.
SPIN_CYCLES = 20_000_000_000
DEADLINE_S = 0.2


def _requirements() -> str | None:
    if not torch.cuda.is_available():
        return "needs a CUDA device"
    return None


@pytest.fixture
def lane() -> LaneCommunicator:
    """A lane on its own stream. synchronize never touches the communicator."""
    stream = torch.cuda.Stream()
    live = LaneCommunicator(
        object(), rank=0, world_size=1, stream=stream, device=torch.cuda.current_device()
    )
    yield live
    # A test that times out leaves its kernel running. Drain it, or the next
    # test inherits the spin and its timing assertions become nonsense.
    torch.cuda.synchronize()


@pytest.mark.skipif(_requirements() is not None, reason=_requirements() or "")
def test_the_bound_actually_applies_on_a_real_stream(lane) -> None:
    """The allow arm: the poll runs rather than falling back.

    False here means every transfer deadline silently becomes a blocking wait.
    """
    assert lane._synchronize_bounded(30.0) is True


@pytest.mark.skipif(_requirements() is not None, reason=_requirements() or "")
def test_work_that_overruns_raises_rather_than_blocking(lane) -> None:
    with torch.cuda.stream(lane.stream):
        torch.cuda._sleep(SPIN_CYCLES)

    started = time.monotonic()
    with pytest.raises(TimeoutError) as excinfo:
        lane.synchronize(timeout_s=DEADLINE_S)
    elapsed = time.monotonic() - started

    assert "did not finish its enqueued work" in str(excinfo.value)
    # The point of the event poll: it returns ON the deadline. A blocking
    # synchronize would have returned only once the kernel finished, which is
    # the failure this whole path exists to prevent.
    assert elapsed < DEADLINE_S * 10, (
        f"returned after {elapsed:.2f}s for a {DEADLINE_S}s deadline; that is a "
        "blocking wait wearing a deadline's clothes"
    )


@pytest.mark.skipif(_requirements() is not None, reason=_requirements() or "")
def test_work_that_finishes_inside_its_budget_returns_cleanly(lane) -> None:
    """Positive control: the deadline must not fire on healthy work."""
    with torch.cuda.stream(lane.stream):
        torch.cuda._sleep(1_000_000)
    lane.synchronize(timeout_s=30.0)


@pytest.mark.skipif(_requirements() is not None, reason=_requirements() or "")
def test_an_unbounded_wait_still_waits(lane) -> None:
    """Without a deadline the behavior is the blocking synchronize, unchanged."""
    with torch.cuda.stream(lane.stream):
        torch.cuda._sleep(100_000_000)
    lane.synchronize()
    assert lane.stream.query() is True
