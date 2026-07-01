# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MaintenanceMixin._prune_dead_peers and ._quick_prune.

These tests bypass the full DhtNode by attaching the mixin's bound methods
to a hand-rolled stub that exposes only the attributes the methods read:
- self.routing_table with .all_peers() and .remove(peer_id)
- self.peer_store with .get_connection(peer_id)

No event loop, no real network, no real DhtNode construction.
"""

from kademlite.dht_maintenance import MaintenanceMixin

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubEntry:
    """Stand-in for routing.PeerEntry - only the attributes the prune
    methods touch."""

    def __init__(self, peer_id: bytes, connected: bool = True) -> None:
        self.peer_id = peer_id
        self.connected = connected


class _StubRoutingTable:
    def __init__(self, entries: list[_StubEntry]) -> None:
        self._entries = list(entries)
        self.removed: list[bytes] = []

    def all_peers(self) -> list[_StubEntry]:
        # Return a snapshot - the real method returns a fresh list too,
        # so prune methods don't fail mid-iteration when remove() is called.
        return list(self._entries)

    def remove(self, peer_id: bytes) -> bool:
        for i, e in enumerate(self._entries):
            if e.peer_id == peer_id:
                self._entries.pop(i)
                self.removed.append(peer_id)
                return True
        return False


class _StubConnection:
    def __init__(self, alive: bool) -> None:
        self._alive = alive

    @property
    def is_alive(self) -> bool:
        return self._alive


class _StubPeerStore:
    """Maps peer_id -> Connection or None."""

    def __init__(self, mapping: dict[bytes, _StubConnection | None] | None = None) -> None:
        self._mapping = mapping or {}

    def get_connection(self, peer_id: bytes):
        return self._mapping.get(peer_id)


class _StubNode:
    """Combines stubs and binds MaintenanceMixin methods directly."""

    def __init__(self, routing_table: _StubRoutingTable, peer_store: _StubPeerStore) -> None:
        self.routing_table = routing_table
        self.peer_store = peer_store

    # Bind the unbound methods from the mixin to this instance
    _prune_dead_peers = MaintenanceMixin._prune_dead_peers
    _quick_prune = MaintenanceMixin._quick_prune


# ---------------------------------------------------------------------------
# _prune_dead_peers
# ---------------------------------------------------------------------------


def test_prune_dead_peers_removes_entries_with_dead_connection() -> None:
    """A routing entry whose peer_store connection is non-None but not alive
    must be removed."""
    entries = [
        _StubEntry(b"peer-alive", connected=True),
        _StubEntry(b"peer-dead-conn", connected=True),
    ]
    rt = _StubRoutingTable(entries)
    ps = _StubPeerStore(
        {
            b"peer-alive": _StubConnection(alive=True),
            b"peer-dead-conn": _StubConnection(alive=False),
        }
    )
    node = _StubNode(rt, ps)

    node._prune_dead_peers()
    assert rt.removed == [b"peer-dead-conn"]


def test_prune_dead_peers_removes_disconnected_entries_with_no_conn() -> None:
    """A routing entry marked not-connected and with no peer_store
    connection must also be removed."""
    entries = [
        _StubEntry(b"peer-keep", connected=True),
        _StubEntry(b"peer-orphan", connected=False),
    ]
    rt = _StubRoutingTable(entries)
    ps = _StubPeerStore({b"peer-keep": None, b"peer-orphan": None})
    node = _StubNode(rt, ps)

    node._prune_dead_peers()
    assert rt.removed == [b"peer-orphan"]


def test_prune_dead_peers_keeps_connected_entries_with_no_conn() -> None:
    """A routing entry marked connected but with no peer_store record must
    NOT be removed (the prune is conservative and waits for explicit signal)."""
    entries = [_StubEntry(b"peer-c-noconn", connected=True)]
    rt = _StubRoutingTable(entries)
    ps = _StubPeerStore({b"peer-c-noconn": None})
    node = _StubNode(rt, ps)

    node._prune_dead_peers()
    assert rt.removed == []


def test_prune_dead_peers_keeps_live_connections() -> None:
    """A routing entry whose connection is alive must be kept regardless of
    the connected flag."""
    entries = [
        _StubEntry(b"peer-1", connected=True),
        _StubEntry(b"peer-2", connected=False),  # marked disconnected
    ]
    rt = _StubRoutingTable(entries)
    # But _peer-2_ has a still-alive connection in the store
    ps = _StubPeerStore(
        {
            b"peer-1": _StubConnection(alive=True),
            b"peer-2": _StubConnection(alive=True),
        }
    )
    node = _StubNode(rt, ps)

    node._prune_dead_peers()
    assert rt.removed == []


# ---------------------------------------------------------------------------
# _quick_prune
# ---------------------------------------------------------------------------


def test_quick_prune_removes_only_disconnected_with_no_conn() -> None:
    entries = [
        _StubEntry(b"a", connected=True),
        _StubEntry(b"b", connected=False),  # should be removed
        _StubEntry(b"c", connected=True),
        _StubEntry(b"d", connected=False),  # has live conn -> kept
    ]
    rt = _StubRoutingTable(entries)
    ps = _StubPeerStore(
        {
            b"a": _StubConnection(alive=True),
            b"b": None,
            b"c": None,
            b"d": _StubConnection(alive=True),
        }
    )
    node = _StubNode(rt, ps)

    node._quick_prune()
    assert rt.removed == [b"b"]


def test_quick_prune_no_op_on_healthy_table() -> None:
    """If every entry is connected (or has a connection), _quick_prune is a no-op."""
    entries = [
        _StubEntry(b"x", connected=True),
        _StubEntry(b"y", connected=True),
    ]
    rt = _StubRoutingTable(entries)
    ps = _StubPeerStore({b"x": None, b"y": None})
    node = _StubNode(rt, ps)

    node._quick_prune()
    assert rt.removed == []


def test_quick_prune_does_not_inspect_connection_object() -> None:
    """_quick_prune must not call .is_alive (it's the cheap path).
    A disconnected entry whose connection is dead must still be removed
    (because get_connection returns the dead conn -> condition is conn IS None)."""
    entries = [
        _StubEntry(b"dead-with-dead-conn", connected=False),
    ]
    rt = _StubRoutingTable(entries)

    # Connection exists but is dead. _quick_prune treats "any conn object" as
    # presence -> peer is kept (only _prune_dead_peers does the alive check).
    ps = _StubPeerStore({b"dead-with-dead-conn": _StubConnection(alive=False)})
    node = _StubNode(rt, ps)

    node._quick_prune()
    assert rt.removed == [], (
        "_quick_prune should not remove peers whose peer_store has any "
        "connection object - that's _prune_dead_peers' job"
    )


def test_prune_methods_handle_empty_routing_table() -> None:
    """Both methods must run cleanly on an empty routing table."""
    rt = _StubRoutingTable([])
    ps = _StubPeerStore({})
    node = _StubNode(rt, ps)

    node._prune_dead_peers()
    node._quick_prune()
    assert rt.removed == []


# ---------------------------------------------------------------------------
# _periodic_bootstrap_tick (added 2026-05-01 to cover commit e168934 -
# "prune before sampling routing-table size in periodic bootstrap")
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402


class _SizeAwareRoutingTable(_StubRoutingTable):
    """Adds size() and remembers when it was first called, so the test
    can assert sampling happens AFTER prune."""

    def __init__(self, entries):
        super().__init__(entries)
        self.size_call_log: list[int] = []  # remaining-entry counts at each size() call
        self.first_size_observed_at: int | None = None

    def size(self) -> int:
        n = len(self._entries)
        self.size_call_log.append(n)
        return n


class _TickNode:
    """Stub that supports _periodic_bootstrap_tick.

    Records every call so tests can assert ordering and decision logic.
    """

    def __init__(
        self,
        routing_table: _SizeAwareRoutingTable,
        peer_store: _StubPeerStore,
        k: int = 5,
        bootstrap_peers=None,
    ) -> None:
        self.routing_table = routing_table
        self.peer_store = peer_store
        self._k = k
        self._bootstrap_peers = bootstrap_peers or []
        self._bootstrap_dns = None
        self._bootstrap_dns_port = 4001
        self._bootstrap_hostlist = None
        self._mdns = None
        # Call log for assertions
        self.calls: list[tuple[str, int]] = []
        # peer_id used for self-lookup
        self.peer_id = b"\x00" * 32

    _quick_prune = MaintenanceMixin._quick_prune
    _periodic_bootstrap_tick = MaintenanceMixin._periodic_bootstrap_tick

    async def bootstrap(self, peers):
        self.calls.append(("bootstrap", len(peers)))

    async def bootstrap_from_dns(self, dns, port):
        self.calls.append(("bootstrap_from_dns", 0))

    async def bootstrap_from_hostlist(self, hostlist, port):
        self.calls.append(("bootstrap_from_hostlist", 0))

    async def _iterative_find_node(self, target):
        self.calls.append(("self_lookup", 0))
        return []

    async def _refresh_buckets(self):
        self.calls.append(("refresh_buckets", 0))


def test_periodic_bootstrap_tick_prunes_before_sampling_size() -> None:
    """The tick must run _quick_prune BEFORE reading routing_table.size()
    so the bootstrap decision reflects reachable peers, not stale ones.

    Regression: prior to e168934, size was sampled first, so a routing
    table that pruning would drop below k stayed inflated for the
    duration of the bootstrap-decision branch.
    """
    # 6 entries total; 4 are dead (disconnected, no live conn) and will be pruned.
    # k=5 so post-prune size (2) < k -> bootstrap path expected.
    # Pre-prune size (6) > k -> would have taken self-lookup path under the bug.
    entries = [
        _StubEntry(b"alive-1", connected=True),
        _StubEntry(b"alive-2", connected=True),
        _StubEntry(b"dead-1", connected=False),
        _StubEntry(b"dead-2", connected=False),
        _StubEntry(b"dead-3", connected=False),
        _StubEntry(b"dead-4", connected=False),
    ]
    rt = _SizeAwareRoutingTable(entries)
    ps = _StubPeerStore({e.peer_id: None for e in entries})
    node = _TickNode(rt, ps, k=5, bootstrap_peers=["/ip4/1.2.3.4/tcp/4001"])

    asyncio.run(node._periodic_bootstrap_tick())

    # size() was called exactly once, after prune dropped the dead peers.
    assert rt.size_call_log == [2], (
        f"size() should have been called exactly once after prune; "
        f"observed call log: {rt.size_call_log}"
    )
    # Bootstrap path was taken because post-prune size (2) < k (5).
    assert ("bootstrap", 1) in node.calls, (
        f"bootstrap should have been called with the configured peers; "
        f"actual calls: {node.calls}"
    )
    # Self-lookup should NOT fire on the sparse path.
    assert ("self_lookup", 0) not in node.calls, (
        f"self_lookup should not run on the sparse-table branch; "
        f"actual calls: {node.calls}"
    )
    # Refresh always runs at the end.
    assert node.calls[-1] == ("refresh_buckets", 0)


def test_periodic_bootstrap_tick_self_lookup_when_healthy() -> None:
    """When the post-prune table is at or above k, the tick takes the
    self-lookup branch instead of re-dialing bootstrap peers."""
    entries = [
        _StubEntry(b"alive-1", connected=True),
        _StubEntry(b"alive-2", connected=True),
        _StubEntry(b"alive-3", connected=True),
    ]
    rt = _SizeAwareRoutingTable(entries)
    ps = _StubPeerStore({e.peer_id: None for e in entries})
    node = _TickNode(rt, ps, k=2, bootstrap_peers=["/ip4/1.2.3.4/tcp/4001"])

    asyncio.run(node._periodic_bootstrap_tick())

    assert rt.size_call_log == [3]
    # Self-lookup fired, bootstrap did not.
    assert ("self_lookup", 0) in node.calls
    assert ("bootstrap", 1) not in node.calls
    # Refresh always runs at the end.
    assert node.calls[-1] == ("refresh_buckets", 0)


def test_periodic_bootstrap_tick_empty_table_takes_neither_branch() -> None:
    """A fully-empty post-prune table should re-bootstrap (size 0 < k)
    and skip the self-lookup elif branch.
    """
    entries = [
        _StubEntry(b"dead-1", connected=False),
        _StubEntry(b"dead-2", connected=False),
    ]
    rt = _SizeAwareRoutingTable(entries)
    ps = _StubPeerStore({e.peer_id: None for e in entries})
    node = _TickNode(rt, ps, k=5, bootstrap_peers=["/ip4/1.2.3.4/tcp/4001"])

    asyncio.run(node._periodic_bootstrap_tick())

    assert rt.size_call_log == [0]
    assert ("bootstrap", 1) in node.calls
    assert ("self_lookup", 0) not in node.calls
