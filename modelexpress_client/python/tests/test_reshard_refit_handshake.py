# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The P2P metadata handshake must be bounded overall, retry transient failures,
say who it is talking to, and dial only the peers it needs.

Two distinct production failures motivate the bounds:
  * a dead publisher still advertised in the catalog previously hung the whole
    refit for the refit timeout, with no indication of which peer stalled;
  * a *live* publisher can be listening yet transiently unable to accept while it
    is busy publishing, which must not be fatal on the first dial.

The peer selection is a scaling concern rather than a failure: the handshake
resolves remote memory registrations for reads, so dialing a trainer this rank
never reads from buys nothing, and unnarrowed that is a dial per
receiver-trainer pair.
"""

import logging
import types

import pytest

from modelexpress import envs
from modelexpress.refit.reshard.receiver import (
    handshake_endpoints_for_plan,
    handshake_with_peers,
)
from modelexpress.refit.reshard.slice_plan import PullSegment
from modelexpress.refit.reshard.transfer_plan import TransferPlan


class _Manager:
    """Records dials. ``fail_until`` makes a peer fail its first N attempts."""

    def __init__(self, always_fail=(), fail_until=None):
        self.always_fail = set(always_fail)
        self.fail_until = dict(fail_until or {})
        self.calls = []

    def fetch_remote_and_wait(self, agent, host, port, timeout_seconds=None):
        self.calls.append((agent, host, port, timeout_seconds))
        if agent in self.always_fail:
            raise ConnectionRefusedError("peer is gone")
        if self.fail_until.get(agent, 0) > 0:
            self.fail_until[agent] -= 1
            raise TimeoutError(f"timed out after {timeout_seconds}s")

    def dials(self, agent):
        return [c for c in self.calls if c[0] == agent]


def _endpoints(n):
    return {f"trainer-r{i}": f"10.0.0.{i}:9999" for i in range(n)}


def test_every_peer_is_dialed_once_when_all_are_healthy():
    manager = _Manager()

    handshake_with_peers(manager, _endpoints(3), 300.0)

    assert [c[0] for c in manager.calls] == ["trainer-r0", "trainer-r1", "trainer-r2"]


def test_a_single_dial_is_capped_by_the_attempt_timeout_not_the_budget():
    """The whole point of the split: a 300 s budget must not become a 300 s dial."""
    manager = _Manager()

    handshake_with_peers(manager, _endpoints(2), 300.0, attempt_timeout=20.0)

    assert {c[3] for c in manager.calls} == {20.0}


def test_attempt_timeout_is_clamped_to_the_remaining_budget():
    manager = _Manager()

    handshake_with_peers(manager, _endpoints(1), 5.0, attempt_timeout=20.0)

    # Derived from the live remaining budget, so it lands just under 5 s.
    assert manager.calls[0][3] == pytest.approx(5.0, abs=0.1)


def test_a_sub_second_remainder_does_not_overrun_the_budget():
    """A one-second floor on the dial timeout would let the last attempt run past
    the total budget, which is the one thing this function has to hold."""
    manager = _Manager()

    handshake_with_peers(manager, _endpoints(1), 0.4, attempt_timeout=20.0)

    assert manager.calls[0][3] <= 0.4


def test_endpoint_is_split_into_host_and_port():
    manager = _Manager()

    handshake_with_peers(manager, {"trainer-r0": "10.0.0.7:9999"}, 300.0)

    assert manager.calls[0][:3] == ("trainer-r0", "10.0.0.7", 9999)


def test_ipv6_style_endpoint_splits_on_the_last_colon():
    manager = _Manager()

    handshake_with_peers(manager, {"trainer-r0": "fd00::1:9999"}, 300.0)

    assert manager.calls[0][1:3] == ("fd00::1", 9999)


def test_a_transiently_unreachable_peer_is_retried_and_succeeds():
    """A busy trainer that drops the first SYNs must not fail the refit."""
    manager = _Manager(fail_until={"trainer-r1": 2})

    handshake_with_peers(manager, _endpoints(3), 300.0, attempt_timeout=1.0)

    assert len(manager.dials("trainer-r1")) == 3


def test_a_failing_peer_is_deferred_so_others_are_not_blocked():
    """The old code aborted on peer 1 of 16 and never dialed the other 15."""
    manager = _Manager(fail_until={"trainer-r0": 1})

    handshake_with_peers(manager, _endpoints(3), 300.0, attempt_timeout=1.0)

    order = [c[0] for c in manager.calls]
    # r0 fails, r1 and r2 are tried before r0 is revisited.
    assert order[0] == "trainer-r0"
    assert order.index("trainer-r1") < order.index("trainer-r0", 1)
    assert order.index("trainer-r2") < order.index("trainer-r0", 1)


def test_budget_exhaustion_names_outstanding_peers_and_counts_progress():
    manager = _Manager(always_fail={"trainer-r2"})

    with pytest.raises(RuntimeError) as excinfo:
        handshake_with_peers(manager, _endpoints(3), 2.0, attempt_timeout=0.5)

    message = str(excinfo.value)
    assert "2 of 3 peer(s) answered" in message
    assert "trainer-r2@10.0.0.2:9999" in message
    assert "attempt(s)" in message
    assert "ConnectionRefusedError" in message


def test_healthy_peers_are_not_redialled_after_another_peer_fails():
    """Re-handshaking a peer that already answered would waste the budget."""
    manager = _Manager(always_fail={"trainer-r1"})

    with pytest.raises(RuntimeError):
        handshake_with_peers(manager, _endpoints(3), 2.0, attempt_timeout=0.5)

    assert len(manager.dials("trainer-r0")) == 1
    assert len(manager.dials("trainer-r2")) == 1


def test_no_peers_is_not_an_error():
    manager = _Manager()

    handshake_with_peers(manager, {}, 300.0)

    assert manager.calls == []


def test_a_malformed_endpoint_fails_only_its_own_peer():
    """One unparseable catalog entry must not abort the other peers' handshake -
    the same fault isolation a refused connection already gets."""
    manager = _Manager()
    endpoints = {"trainer-r0": "10.0.0.0-no-port", "trainer-r1": "10.0.0.1:9999"}

    with pytest.raises(RuntimeError) as excinfo:
        handshake_with_peers(manager, endpoints, 1.0, attempt_timeout=0.5)

    message = str(excinfo.value)
    assert "1 of 2 peer(s) answered" in message
    assert "trainer-r0@10.0.0.0-no-port" in message
    assert manager.dials("trainer-r1")


def test_handshake_bounds_come_from_the_env_registry(monkeypatch):
    monkeypatch.setenv("MX_RESHARD_HANDSHAKE_ATTEMPT_S", "3.5")

    assert envs.MX_RESHARD_HANDSHAKE_ATTEMPT_S == 3.5


def test_a_non_positive_bound_falls_back_to_the_default(monkeypatch, caplog):
    """Zero or negative is not a smaller bound but an absent one, which reinstates
    the hang the bound exists to prevent. The documented default is kept instead."""
    monkeypatch.setenv("MX_RESHARD_HANDSHAKE_TIMEOUT_S", "0")

    with caplog.at_level(logging.WARNING, logger="modelexpress.envs"):
        assert envs.MX_RESHARD_HANDSHAKE_TIMEOUT_S == 900.0

    assert "must be a finite positive number" in caplog.text


def test_an_unparseable_bound_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("MX_RESHARD_HANDSHAKE_BACKOFF_S", "soon")

    assert envs.MX_RESHARD_HANDSHAKE_BACKOFF_S == 2.0


@pytest.mark.parametrize("raw", ["inf", "-inf", "nan"])
def test_a_non_finite_bound_falls_back_to_the_default(monkeypatch, raw):
    """`float()` takes these happily and each one defeats the deadline: `now >=
    inf` is never true, and every comparison against `nan` is false. Both leave
    the handshake unbounded, which is what the bound is here to prevent."""
    monkeypatch.setenv("MX_RESHARD_HANDSHAKE_TIMEOUT_S", raw)

    assert envs.MX_RESHARD_HANDSHAKE_TIMEOUT_S == 900.0


# --------------------------------------------------- narrowing the peer set


def _segment(session):
    return PullSegment(
        session=session, src_addr=0, param_name="w", dst_byte=0, nbytes=1024
    )


def _session_to_agent(n):
    return {f"s{i}": f"trainer-r{i}" for i in range(n)}


def test_only_the_trainers_the_plan_reads_from_are_dialed():
    """Four trainers were discovered, the plan reads from two, so two are dialed."""
    plan = TransferPlan(segments=[_segment("s1"), _segment("s3")])

    narrowed = handshake_endpoints_for_plan(plan, _session_to_agent(4), _endpoints(4))

    assert set(narrowed) == {"trainer-r1", "trainer-r3"}


def test_the_narrowed_endpoints_keep_their_addresses():
    plan = TransferPlan(segments=[_segment("s2")])

    narrowed = handshake_endpoints_for_plan(plan, _session_to_agent(4), _endpoints(4))

    assert narrowed == {"trainer-r2": "10.0.0.2:9999"}


def test_reads_are_counted_from_every_phase_of_the_plan():
    """Segments land in three places - straight into live params, into
    dtype-conversion staging, and into full-pull staging. Missing any one of them
    would drop a peer the plan genuinely reads from, and the failure would surface
    later as an unresolvable address in prep_xfer_dlist."""
    plan = TransferPlan(
        segments=[_segment("s0")],
        converts=[types.SimpleNamespace(segments=[_segment("s1")])],
        full_pulls=[types.SimpleNamespace(segments=[_segment("s2")])],
    )

    assert plan.sessions() == {"s0", "s1", "s2"}

    narrowed = handshake_endpoints_for_plan(plan, _session_to_agent(4), _endpoints(4))

    assert set(narrowed) == {"trainer-r0", "trainer-r1", "trainer-r2"}


def test_a_planned_trainer_with_no_endpoint_fails_closed():
    """Silently skipping it would defer the failure to prep_xfer_dlist, which
    cannot say which peer it was missing metadata for."""
    plan = TransferPlan(segments=[_segment("s0"), _segment("s3")])
    endpoints = {"trainer-r0": "10.0.0.0:9999"}

    with pytest.raises(RuntimeError) as excinfo:
        handshake_endpoints_for_plan(plan, _session_to_agent(4), endpoints)

    assert "trainer-r3" in str(excinfo.value)


def test_an_empty_plan_dials_nobody():
    narrowed = handshake_endpoints_for_plan(
        TransferPlan(), _session_to_agent(4), _endpoints(4)
    )

    assert narrowed == {}
