# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""NixlReshardTransport dispatch logic - no NIXL, no GPU.

A stub manager records post/await/execute calls so we can assert the transport
groups descriptors per session into one batch, resolves session -> remote agent
+ device, forms the right (remote_addr, local_addr, nbytes, device) ranges, and
posts every peer's batch before waiting on any of them. The actual RDMA is
exercised by the on-cluster smoke harness.

Run: pytest tests/test_reshard_refit_nixl_transport.py
"""

from dataclasses import dataclass

import pytest

from modelexpress.refit.reshard.transport import (
    NixlReshardTransport,
    ReadDescriptor,
)


@dataclass
class _StubPosted:
    remote_agent_name: str
    total_bytes: int
    num_ranges: int


class _StubManager:
    """Records the call sequence so ordering can be asserted, not just counts."""

    def __init__(self, fail_post_on: str | None = None):
        self.calls = []  # (remote_agent_name, ranges, mem_type, timeout)
        self.events = []  # ("post", agent) / ("await", n) / ("execute", agent)
        self.released = 0
        self._fail_post_on = fail_post_on

    def post_read_batch(self, remote_agent_name, ranges, mem_type=None):
        if remote_agent_name == self._fail_post_on:
            raise RuntimeError(f"prep failed for {remote_agent_name}")
        self.calls.append((remote_agent_name, list(ranges), mem_type, None))
        self.events.append(("post", remote_agent_name))
        return _StubPosted(
            remote_agent_name=remote_agent_name,
            total_bytes=sum(n for (_r, _l, n, _d) in ranges),
            num_ranges=len(ranges),
        )

    def await_read_batches(self, posted, timeout_seconds=None):
        batches = [p for p in posted if p is not None]
        self.events.append(("await", len(batches)))
        self.released += len(batches)
        return (
            sum(p.total_bytes for p in batches),
            sum(p.num_ranges for p in batches),
            0.0,
        )

    def execute_read_batch(
        self, remote_agent_name, ranges, mem_type=None, timeout_seconds=None
    ):
        self.calls.append((remote_agent_name, list(ranges), mem_type, timeout_seconds))
        self.events.append(("execute", remote_agent_name))
        total = sum(n for (_r, _l, n, _d) in ranges)
        return total, len(ranges), 0.0


def _two_session_descriptors():
    return [
        ReadDescriptor(session="sA", src_addr=1000, dst_addr=10, nbytes=16),
        ReadDescriptor(session="sB", src_addr=2000, dst_addr=20, nbytes=8),
        ReadDescriptor(session="sA", src_addr=1016, dst_addr=26, nbytes=16),
    ]


def _transport(mgr, **kwargs):
    return NixlReshardTransport(
        manager=mgr,
        session_to_agent={"sA": "trainer-agent-A", "sB": "trainer-agent-B"},
        session_to_device={"sA": 3, "sB": 5},
        mem_type="VRAM",
        timeout_seconds=30.0,
        **kwargs,
    )


def test_groups_per_session_and_resolves_agent_device():
    mgr = _StubManager()
    _transport(mgr).read(_two_session_descriptors())

    # One batched READ per session.
    assert len(mgr.calls) == 2
    by_agent = {c[0]: c for c in mgr.calls}

    _a_agent, a_ranges, a_mem, _a_timeout = by_agent["trainer-agent-A"]
    assert a_mem == "VRAM"
    # (remote_addr, local_addr, nbytes, remote_device_id); device 3 for sA.
    assert a_ranges == [(1000, 10, 16, 3), (1016, 26, 16, 3)]

    _b_agent, b_ranges, _m, _t = by_agent["trainer-agent-B"]
    assert b_ranges == [(2000, 20, 8, 5)]


def test_stats_accumulate_across_sessions():
    mgr = _StubManager()
    transport = _transport(mgr)
    transport.read(_two_session_descriptors())

    assert transport.bytes_moved == 16 + 16 + 8
    assert transport.reads_issued == 3


def test_all_peers_posted_before_any_wait():
    """The point of the post/wait split: N peers in flight, not one at a time."""
    mgr = _StubManager()
    _transport(mgr).read(_two_session_descriptors())

    kinds = [kind for kind, _ in mgr.events]
    assert kinds == ["post", "post", "await"], mgr.events
    # Nothing was drained peer-by-peer.
    assert "execute" not in kinds


def test_every_posted_batch_is_awaited():
    mgr = _StubManager()
    _transport(mgr).read(_two_session_descriptors())

    posted = sum(1 for kind, _ in mgr.events if kind == "post")
    assert mgr.released == posted


def test_serial_env_restores_one_peer_at_a_time(monkeypatch):
    monkeypatch.setenv("MX_RESHARD_SERIAL_READS", "1")
    mgr = _StubManager()
    transport = _transport(mgr)
    transport.read(_two_session_descriptors())

    # Drains each peer inline; never uses the post/await path.
    assert [kind for kind, _ in mgr.events] == ["execute", "execute"]
    assert transport.bytes_moved == 16 + 16 + 8
    assert transport.reads_issued == 3
    # Serial mode is the only path that forwards the timeout per batch.
    assert all(call[3] == 30.0 for call in mgr.calls)


def test_post_failure_drains_already_posted_batches():
    """A mid-loop post failure must not leak the handles already in flight."""
    mgr = _StubManager(fail_post_on="trainer-agent-B")
    with pytest.raises(RuntimeError, match="prep failed"):
        _transport(mgr).read(_two_session_descriptors())

    # The successful post was awaited (and therefore released) on the way out.
    assert ("await", 1) in mgr.events
    assert mgr.released == 1


def test_drain_failure_does_not_mask_post_failure(monkeypatch):
    mgr = _StubManager(fail_post_on="trainer-agent-B")

    def _boom(posted, timeout_seconds=None):
        raise RuntimeError("drain blew up")

    monkeypatch.setattr(mgr, "await_read_batches", _boom)

    # The original post failure surfaces, not the cleanup failure.
    with pytest.raises(RuntimeError, match="prep failed"):
        _transport(mgr).read(_two_session_descriptors())


def test_no_descriptors_is_a_noop():
    mgr = _StubManager()
    transport = _transport(mgr)
    transport.read([])

    assert mgr.events == []
    assert transport.bytes_moved == 0
    assert transport.reads_issued == 0


def test_missing_agent_raises():
    transport = NixlReshardTransport(manager=_StubManager(), session_to_agent={})
    with pytest.raises(KeyError):
        transport.read(
            [ReadDescriptor(session="unknown", src_addr=0, dst_addr=0, nbytes=4)]
        )


def test_missing_device_mapping_raises():
    transport = NixlReshardTransport(
        manager=_StubManager(),
        session_to_agent={"s": "agent"},
    )
    with pytest.raises(KeyError, match="no remote device id"):
        transport.read(
            [ReadDescriptor(session="s", src_addr=100, dst_addr=200, nbytes=4)]
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
