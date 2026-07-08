# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""NixlReshardTransport dispatch logic - no NIXL, no GPU.

A stub manager records execute_read_batch calls so we can assert the transport
groups descriptors per session into one batch, resolves session -> remote agent
+ device, and forms the right (remote_addr, local_addr, nbytes, device) ranges.
The actual RDMA is exercised by the on-cluster smoke harness.

Run: pytest tests/test_reshard_refit_nixl_transport.py
"""

import pytest

from modelexpress.reshard_refit.transport import NixlReshardTransport
from modelexpress.reshard_refit.transport import ReadDescriptor


class _StubManager:
    def __init__(self):
        self.calls = []  # (remote_agent_name, ranges, mem_type, timeout)

    def execute_read_batch(self, remote_agent_name, ranges, mem_type=None, timeout_seconds=None):
        self.calls.append((remote_agent_name, list(ranges), mem_type, timeout_seconds))
        total = sum(n for (_r, _l, n, _d) in ranges)
        return total, len(ranges), 0.0


def test_groups_per_session_and_resolves_agent_device():
    mgr = _StubManager()
    transport = NixlReshardTransport(
        manager=mgr,
        session_to_agent={"sA": "trainer-agent-A", "sB": "trainer-agent-B"},
        session_to_device={"sA": 3, "sB": 5},
        mem_type="VRAM",
        timeout_seconds=30.0,
    )

    descriptors = [
        ReadDescriptor(session="sA", src_addr=1000, dst_addr=10, nbytes=16),
        ReadDescriptor(session="sB", src_addr=2000, dst_addr=20, nbytes=8),
        ReadDescriptor(session="sA", src_addr=1016, dst_addr=26, nbytes=16),
    ]
    transport.read(descriptors)

    # One batched READ per session.
    assert len(mgr.calls) == 2
    by_agent = {c[0]: c for c in mgr.calls}

    a_agent, a_ranges, a_mem, a_timeout = by_agent["trainer-agent-A"]
    assert a_mem == "VRAM" and a_timeout == 30.0
    # (remote_addr, local_addr, nbytes, remote_device_id); device 3 for sA.
    assert a_ranges == [(1000, 10, 16, 3), (1016, 26, 16, 3)]

    _b_agent, b_ranges, _m, _t = by_agent["trainer-agent-B"]
    assert b_ranges == [(2000, 20, 8, 5)]

    # Stats accumulate across sessions.
    assert transport.bytes_moved == 16 + 16 + 8
    assert transport.reads_issued == 3


def test_missing_agent_raises():
    transport = NixlReshardTransport(manager=_StubManager(), session_to_agent={})
    with pytest.raises(KeyError):
        transport.read([ReadDescriptor(session="unknown", src_addr=0, dst_addr=0, nbytes=4)])


def test_device_defaults_to_zero():
    mgr = _StubManager()
    transport = NixlReshardTransport(manager=mgr, session_to_agent={"s": "agent"})  # no device map
    transport.read([ReadDescriptor(session="s", src_addr=100, dst_addr=200, nbytes=4)])
    assert mgr.calls[0][1] == [(100, 200, 4, 0)]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
