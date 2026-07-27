# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The P2P metadata handshake must be bounded overall, retry transient failures,
and say who it is talking to.

Two distinct production failures motivate these:
  * a dead publisher still advertised in the catalog previously hung the whole
    refit for the refit timeout, with no indication of which peer stalled;
  * a *live* publisher can be listening yet transiently unable to accept while it
    is busy publishing, which must not be fatal on the first dial.
"""

import pytest

from modelexpress.refit.reshard.receiver import handshake_with_peers


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
