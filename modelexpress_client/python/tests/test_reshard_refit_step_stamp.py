# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the publisher step stamp on the rendezvous blob.

The stamp answers a question the receiver previously had to guess at: does the shard
table I just discovered describe the step I am refitting, or the one before it? It
matters because the table carries the per-shard digests the verify gate compares
against, so a table one step behind turns correctly delivered bytes into a reported
corruption - which cost two runs and a day on 2026-07-30.

What is worth testing here is not that a field round-trips. It is that the stamp is
*optional in both directions* (a fleet is upgraded one image at a time, and neither
side may assume the other has it), that a missing stamp is read as "unknown" rather
than as step 0, and that the receiver's staleness delta does not quietly depend on the
publisher's counter agreeing with its own.
"""

from __future__ import annotations

import json

from modelexpress.refit.reshard.receiver import ReshardReceiver
from modelexpress.refit.reshard.rendezvous import (
    unwrap_rendezvous_blob,
    unwrap_rendezvous_blob_with_step,
    wrap_rendezvous_blob,
)


def _blob(step=None, tensors=None):
    return wrap_rendezvous_blob(b"meta", "agent-r0", "h:1", tensors or [], step)


# ------------------------------------------------------------------- the wire format
def test_the_stamp_round_trips():
    _m, _n, _e, _t, step = unwrap_rendezvous_blob_with_step(_blob(step=7))

    assert step == 7


def test_an_unstamped_blob_reads_as_unknown_not_step_zero():
    """The distinction the staleness check rests on. Read as 0, an unstamped publisher
    would look permanently behind and every one of its shards would be excused."""
    _m, _n, _e, _t, step = unwrap_rendezvous_blob_with_step(_blob())

    assert step is None


def test_the_stamp_is_omitted_rather_than_nulled_when_absent():
    """So a receiver too old to know the key sees a blob it would have produced."""
    payload = json.loads(_blob().decode("utf-8"))

    assert "publisher_step" not in payload


def test_step_zero_is_carried_not_dropped():
    """0 is a legitimate step and must not be confused with 'no stamp'."""
    _m, _n, _e, _t, step = unwrap_rendezvous_blob_with_step(_blob(step=0))

    assert step == 0


def test_the_old_unwrap_keeps_its_arity():
    """Callers that predate the stamp must be unaffected."""
    agent_meta, name, endpoint, tensors = unwrap_rendezvous_blob(_blob(step=3))

    assert (agent_meta, name, endpoint, tensors) == (b"meta", "agent-r0", "h:1", [])


def test_a_new_receiver_reads_an_old_publishers_blob():
    """Forward compatibility: no publisher_step key at all."""
    payload = json.loads(_blob().decode("utf-8"))
    payload.pop("publisher_step", None)

    _m, _n, _e, _t, step = unwrap_rendezvous_blob_with_step(
        json.dumps(payload).encode("utf-8")
    )

    assert step is None


def test_an_old_receiver_ignores_a_new_publishers_stamp():
    """Backward compatibility: unwrap must not reject the extra key."""
    assert unwrap_rendezvous_blob(_blob(step=11))[1] == "agent-r0"


# --------------------------------------------------- the receiver's staleness delta
class _Receiver:
    """Just the bookkeeping under test, borrowed off the real class so the test cannot
    drift from it, without a cluster behind it."""

    _note_publisher_steps = ReshardReceiver._note_publisher_steps


def test_the_first_refit_flags_nothing():
    """There is no previous refit yet, so nothing can be shown to have lagged."""
    r = _Receiver()

    r._note_publisher_steps({"r0": 1, "r1": 1}, step=1)

    assert r._stale_sessions == frozenset()


def test_two_discoveries_in_one_refit_flag_nothing():
    """The hardware defect this keys on.

    A refit discovers twice - at prepare, and again for the fresh table - and both see
    the same publication, so no stamp has advanced between them. A per-call delta reads
    that as every publisher lagging. Run gate-stepstamp-v14 flagged all 16 at step 1,
    the step whose comparison is cleanest, where a real mismatch would then have been
    excused as unattributable.
    """
    r = _Receiver()

    r._note_publisher_steps({"r0": 1, "r1": 1}, step=1)
    r._note_publisher_steps({"r0": 1, "r1": 1}, step=1)

    assert r._stale_sessions == frozenset(), (
        "a second discovery within one refit is not a lagging publisher"
    )


def test_repeated_discoveries_in_one_refit_stay_idempotent():
    r = _Receiver()
    r._note_publisher_steps({"r0": 1}, step=1)
    r._note_publisher_steps({"r0": 2}, step=2)

    for _ in range(4):
        r._note_publisher_steps({"r0": 2}, step=2)

    assert r._stale_sessions == frozenset()


def test_a_publisher_that_advanced_is_not_stale():
    r = _Receiver()
    r._note_publisher_steps({"r0": 1}, step=1)

    r._note_publisher_steps({"r0": 2}, step=2)

    assert r._stale_sessions == frozenset()


def test_a_publisher_that_did_not_advance_is_stale():
    """The v12c step-2 situation: a new refit, a table still describing the old step."""
    r = _Receiver()
    r._note_publisher_steps({"r0": 1}, step=1)

    r._note_publisher_steps({"r0": 1}, step=2)

    assert r._stale_sessions == frozenset({"r0"})


def test_only_the_lagging_publisher_is_flagged():
    """The whole point: partial propagation is localised, not globalised."""
    r = _Receiver()
    r._note_publisher_steps({"lagging": 1, "current": 1}, step=1)

    r._note_publisher_steps({"lagging": 1, "current": 2}, step=2)

    assert r._stale_sessions == frozenset({"lagging"})


def test_an_unstamped_publisher_is_never_called_stale():
    """An absent stamp is not evidence of staleness."""
    r = _Receiver()
    r._note_publisher_steps({"r0": None}, step=1)

    r._note_publisher_steps({"r0": None}, step=2)

    assert r._stale_sessions == frozenset()


def test_an_unstamped_publisher_is_not_counted_as_stamping():
    """``stamps_seen`` keys on this, and must not be fooled by a None."""
    r = _Receiver()

    r._note_publisher_steps({"r0": None}, step=1)

    assert not r._publisher_steps


def test_a_publisher_appearing_late_is_not_stale():
    """It has no baseline of its own, so there is nothing to compare."""
    r = _Receiver()
    r._note_publisher_steps({"r0": 1}, step=1)

    r._note_publisher_steps({"r0": 2, "r1": 2}, step=2)

    assert r._stale_sessions == frozenset()


def test_a_publisher_going_backwards_is_stale():
    """The heartbeat-resurrection signature: the table reverts to an earlier step."""
    r = _Receiver()
    r._note_publisher_steps({"r0": 5}, step=1)

    r._note_publisher_steps({"r0": 3}, step=2)

    assert r._stale_sessions == frozenset({"r0"})


def test_the_delta_does_not_assume_the_counters_agree():
    """The receiver's refit step and the publisher's version are separate counters.

    A publisher stamping 100, 101 while the receiver refits 1, 2 must read as healthy;
    an implementation comparing the stamp against the receiver's own step would call
    every one of these stale.
    """
    r = _Receiver()
    r._note_publisher_steps({"r0": 100}, step=1)

    r._note_publisher_steps({"r0": 101}, step=2)

    assert r._stale_sessions == frozenset()


def test_staleness_does_not_persist_once_the_publisher_catches_up():
    r = _Receiver()
    r._note_publisher_steps({"r0": 1}, step=1)
    r._note_publisher_steps({"r0": 1}, step=2)
    assert r._stale_sessions == frozenset({"r0"})

    r._note_publisher_steps({"r0": 2}, step=3)

    assert r._stale_sessions == frozenset()


def test_a_skipped_refit_number_does_not_invent_staleness():
    """Steps are not guaranteed contiguous; a gap means no baseline, not a lag."""
    r = _Receiver()
    r._note_publisher_steps({"r0": 1}, step=1)

    r._note_publisher_steps({"r0": 1}, step=5)

    assert r._stale_sessions == frozenset()
