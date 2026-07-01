# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v0.3.2 dead-peer-propagation regression tests.

Covers the two v0.3.2 fixes for the ghost-storm pattern that fresh
joiners hit when bootstrapping against a long-running responder whose
routing table has accumulated peers it has marked unreachable:

A. Responder-side filter
   ``KadHandler._closest_peers_encoded`` filters peers that the
   responder has marked disconnected out of FIND_NODE and GET_VALUE
   response peer lists, so they stop infecting fresh joiners with dead
   routing-table entries. v0.3.2 used a ``connected_only=True`` filter
   for this; v0.3.3 narrows the same filter to ``dial_verified_only=True``
   (the test helper ``mark_dial_verified``-promotes the "connected"
   peers so the original assertions still hold). The test contract
   - "disconnected entries are excluded from response peer lists" -
   is identical across both versions.

B. Bootstrap self-FIND_NODE walk skip
   ``BootstrapMixin._bootstrap_dial_peers`` and ``_dial_ips`` skip the
   post-dial ``_iterative_find_node(self.peer_id)`` walk when the dial
   phase already populated the routing table to ``>= self._k`` entries.
   The walk is opportunistic keyspace coverage, not a correctness
   requirement for the immediate operation that follows; against
   responders whose tables include unreachable peers, walking can pay
   one ``DIAL_TIMEOUT`` per ghost in alpha-sized batches.
"""

from unittest.mock import AsyncMock

import pytest

from kademlite.crypto import Ed25519Identity
from kademlite.dht import DhtNode
from kademlite.kad_handler import KadHandler
from kademlite.kademlia import (
    MSG_FIND_NODE,
    MSG_GET_VALUE,
    decode_kad_message,
    decode_peer,
    encode_kad_message,
)
from kademlite.routing import RoutingTable

# ---------------------------------------------------------------------------
# Fix A: responder-side connected_only filter
# ---------------------------------------------------------------------------


def _make_peer_id(seed: int) -> bytes:
    return seed.to_bytes(32, "big")


def _build_handler_with_mixed_peers(
    n_connected: int, n_disconnected: int, k: int = 20
) -> tuple[KadHandler, set[bytes], set[bytes]]:
    """Build a KadHandler whose routing table has both connected and
    disconnected peers. Returns (handler, connected_ids, disconnected_ids).

    The "connected" peers are explicitly ``mark_dial_verified`` so they
    pass the v0.3.3 responder filter. The v0.3.2 contract these tests
    document - "disconnected entries are excluded from FIND_NODE /
    GET_VALUE response peer lists" - is preserved under the stricter
    v0.3.3 contract because ``mark_disconnected`` clears the
    ``dial_verified`` bit on a previously-verified peer.
    """
    local = _make_peer_id(0)
    rt = RoutingTable(local, k=k)
    connected_ids: set[bytes] = set()
    disconnected_ids: set[bytes] = set()
    seed = 1
    for _ in range(n_connected):
        pid = _make_peer_id(seed)
        rt.add_or_update(pid, [b"/c"])
        rt.mark_dial_verified(pid)
        connected_ids.add(pid)
        seed += 1
    for _ in range(n_disconnected):
        pid = _make_peer_id(seed)
        rt.add_or_update(pid, [b"/d"])
        rt.mark_dial_verified(pid)
        rt.mark_disconnected(pid)
        disconnected_ids.add(pid)
        seed += 1
    return KadHandler(rt, k=k), connected_ids, disconnected_ids


def test_closest_peers_encoded_excludes_disconnected():
    """The internal helper must filter to connected peers only."""
    handler, connected, disconnected = _build_handler_with_mixed_peers(5, 5)
    encoded = handler._closest_peers_encoded(b"/some/key")
    returned = {decode_peer(p)["id"] for p in encoded}
    assert returned == connected
    assert not (returned & disconnected)


def test_find_node_response_excludes_disconnected_peers():
    """FIND_NODE response decoded off the wire excludes dead entries."""
    handler, connected, disconnected = _build_handler_with_mixed_peers(8, 7)
    request = encode_kad_message(MSG_FIND_NODE, key=b"/lookup/target")
    request_msg = decode_kad_message(request)
    response_bytes = handler._handle_find_node(request_msg)
    response = decode_kad_message(response_bytes)
    closer_ids = {p["id"] for p in response["closer_peers"]}
    assert closer_ids.issubset(connected)
    assert not (closer_ids & disconnected)


def test_get_value_no_record_response_excludes_disconnected_peers():
    """GET_VALUE with no local record falls through to closer peers and
    must apply the same filter."""
    handler, connected, disconnected = _build_handler_with_mixed_peers(8, 7)
    request = encode_kad_message(MSG_GET_VALUE, key=b"/missing/key")
    request_msg = decode_kad_message(request)
    response_bytes = handler._handle_get_value(request_msg)
    response = decode_kad_message(response_bytes)
    closer_ids = {p["id"] for p in response["closer_peers"]}
    assert closer_ids.issubset(connected)
    assert not (closer_ids & disconnected)


def test_responder_filter_does_not_drop_connected_peers():
    """The filter only excludes disconnected peers - all connected peers
    remain reachable in the response (subject to k-cap)."""
    handler, connected, _ = _build_handler_with_mixed_peers(
        n_connected=10, n_disconnected=10, k=20
    )
    encoded = handler._closest_peers_encoded(b"/key")
    returned = {decode_peer(p)["id"] for p in encoded}
    # All 10 connected peers fit under k=20 even with the filter
    assert returned == connected


def test_responder_filter_with_no_disconnected_peers_unchanged():
    """When the routing table has only connected peers, the filter is a
    no-op: all peers (up to k) are returned."""
    handler, connected, _ = _build_handler_with_mixed_peers(
        n_connected=15, n_disconnected=0, k=20
    )
    encoded = handler._closest_peers_encoded(b"/key")
    returned = {decode_peer(p)["id"] for p in encoded}
    assert returned == connected


def test_responder_filter_with_all_disconnected_returns_empty():
    """When every routing-table entry is marked disconnected, the
    response carries an empty closer-peers list. Fresh joiners still
    progress (they fall back to whatever they have locally), but they
    don't inherit ghost entries."""
    handler, _, disconnected = _build_handler_with_mixed_peers(
        n_connected=0, n_disconnected=10, k=20
    )
    encoded = handler._closest_peers_encoded(b"/key")
    assert encoded == []


# ---------------------------------------------------------------------------
# Fix B: bootstrap self-FIND_NODE walk skip when RT already at k
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_dial_peers_skips_walk_when_rt_at_k(monkeypatch):
    """When the routing table is already at >= k entries (e.g. populated
    by a prior bootstrap or a side-channel), the post-dial self-walk is
    redundant and must be skipped."""
    node = DhtNode(identity=Ed25519Identity.generate(), k=4)
    walk_called = {"count": 0}

    async def fake_walk(target):
        walk_called["count"] += 1
        return []

    monkeypatch.setattr(node, "_iterative_find_node", fake_walk)

    # Pre-populate routing table to >= k. Use peer IDs distinct from
    # local so add_or_update accepts them.
    for i in range(1, 6):  # 5 entries, k=4 -> condition `< k` is False
        node.routing_table.add_or_update(_make_peer_id(i), [b"/seed"])

    # Empty peers list: skips the dial loop entirely, drops straight to
    # the post-loop walk decision. The walk must NOT fire because
    # rt_size (5) >= k (4).
    await node._bootstrap_dial_peers([])
    assert walk_called["count"] == 0


@pytest.mark.asyncio
async def test_bootstrap_dial_peers_runs_walk_when_rt_sparse(monkeypatch):
    """When the routing table is below k after the dial phase, the walk
    is still necessary keyspace coverage and must run."""
    node = DhtNode(identity=Ed25519Identity.generate(), k=20)
    walk_called = {"count": 0}

    async def fake_walk(target):
        walk_called["count"] += 1
        return []

    monkeypatch.setattr(node, "_iterative_find_node", fake_walk)

    # Pre-populate to < k (5 entries with k=20).
    for i in range(1, 6):
        node.routing_table.add_or_update(_make_peer_id(i), [b"/seed"])

    await node._bootstrap_dial_peers([])
    assert walk_called["count"] == 1


@pytest.mark.asyncio
async def test_bootstrap_dial_peers_skips_walk_when_rt_empty(monkeypatch):
    """When all dials failed and the routing table is empty, the walk
    has nothing to iterate over. Both the v0.3.1 ``size() > 0`` guard
    and the v0.3.2 ``0 < size < k`` guard skip it; this test pins the
    behaviour in case the upper bound regresses."""
    node = DhtNode(identity=Ed25519Identity.generate(), k=20)
    walk_called = {"count": 0}

    async def fake_walk(target):
        walk_called["count"] += 1
        return []

    monkeypatch.setattr(node, "_iterative_find_node", fake_walk)

    # Routing table starts empty.
    assert node.routing_table.size() == 0
    await node._bootstrap_dial_peers([])
    assert walk_called["count"] == 0


@pytest.mark.asyncio
async def test_dial_ips_skips_walk_when_rt_at_k(monkeypatch):
    """The DNS / SLURM dial path shares the same skip rule. We feed it a
    non-empty IP list so the post-dial walk decision actually runs, but
    mock ``dial`` to fail synchronously so the test does not depend on
    real network reachability."""
    node = DhtNode(identity=Ed25519Identity.generate(), k=4)
    walk_called = {"count": 0}

    async def fake_walk(target):
        walk_called["count"] += 1
        return []

    monkeypatch.setattr(node, "_iterative_find_node", fake_walk)

    # Force the dial to raise immediately - no network I/O.
    async def fake_dial(*args, **kwargs):
        raise OSError("mocked dial failure")

    monkeypatch.setattr("kademlite.dht_bootstrap.dial", fake_dial)

    # Pre-populate to >= k connected entries.
    for i in range(1, 6):
        node.routing_table.add_or_update(_make_peer_id(i), [b"/seed"])

    await node._dial_ips(["192.0.2.1"], 4001, "TEST")

    # All dials failed, so connected==0 and no new entries were added,
    # but pre-populated reachable rt_size==5 >= k==4 -> walk skipped.
    assert walk_called["count"] == 0


@pytest.mark.asyncio
async def test_dial_ips_walk_skip_uses_connected_only_size(monkeypatch):
    """``_dial_ips`` (DNS / SLURM bootstrap) shares the same skip rule
    as ``_bootstrap_dial_peers`` and must use the same connected-only
    count. A table holding ``k`` stale ``connected=False`` entries plus
    one live entry must NOT suppress the self-walk: live coverage is
    only one peer wide, the walk is needed to expand it."""
    node = DhtNode(identity=Ed25519Identity.generate(), k=4)
    walk = AsyncMock(return_value=[])
    monkeypatch.setattr(node, "_iterative_find_node", walk)

    async def fake_dial(*args, **kwargs):
        raise OSError("mocked dial failure")

    monkeypatch.setattr("kademlite.dht_bootstrap.dial", fake_dial)

    # 4 stale entries + 1 live entry. Total size==5 >= k, but reachable
    # size==1 < k -> walk should run.
    for i in range(1, 5):
        pid = _make_peer_id(i)
        node.routing_table.add_or_update(pid, [b"/dead"])
        node.routing_table.mark_disconnected(pid)
    node.routing_table.add_or_update(_make_peer_id(99), [b"/live"])

    await node._dial_ips(["192.0.2.1"], 4001, "TEST")
    walk.assert_awaited_once()


@pytest.mark.asyncio
async def test_bootstrap_walk_skip_uses_connected_only_size(monkeypatch):
    """The walk-skip predicate counts connected peers, not total RT size.
    A table holding ``k`` stale ``connected=False`` entries must NOT
    suppress the self-walk: the responder filter would have hidden those
    entries from us anyway, so the walk is needed to discover live peers."""
    node = DhtNode(identity=Ed25519Identity.generate(), k=4)
    walk = AsyncMock(return_value=[])
    monkeypatch.setattr(node, "_iterative_find_node", walk)

    # 5 entries in the table, all marked disconnected. Total size==5 >= k,
    # but reachable size==0. Walk must run because there is no live coverage.
    for i in range(1, 6):
        pid = _make_peer_id(i)
        node.routing_table.add_or_update(pid, [b"/dead"])
        node.routing_table.mark_disconnected(pid)

    # Need at least one live peer for the walk to be useful (the
    # ``0 < rt_size`` guard skips an empty live table). Add a single
    # connected entry.
    node.routing_table.add_or_update(_make_peer_id(99), [b"/live"])

    await node._bootstrap_dial_peers([])
    walk.assert_awaited_once()


def test_routing_table_size_connected_only():
    """``RoutingTable.size(connected_only=True)`` counts only reachable peers."""
    rt = RoutingTable(_make_peer_id(0), k=20)
    for i in range(1, 6):
        rt.add_or_update(_make_peer_id(i), [b"/c"])
    for i in range(6, 9):
        pid = _make_peer_id(i)
        rt.add_or_update(pid, [b"/d"])
        rt.mark_disconnected(pid)
    assert rt.size() == 8
    assert rt.size(connected_only=True) == 5
    assert rt.size(connected_only=False) == 8


@pytest.mark.asyncio
async def test_bootstrap_dial_peers_marks_recovered_peer_connected(monkeypatch):
    """A peer that previously existed in the routing table marked
    disconnected must be marked connected again after a successful
    bootstrap re-dial. ``add_or_update`` alone does not reset the flag,
    and the v0.3.2 responder filter would otherwise leave the recovered
    peer invisible to other nodes asking us for closer-peer hints."""
    node = DhtNode(identity=Ed25519Identity.generate(), k=20)

    # Pre-seed a peer entry and mark it disconnected, simulating a peer
    # we knew about and lost.
    recovered_peer = _make_peer_id(42)
    node.routing_table.add_or_update(recovered_peer, [b"/old"])
    node.routing_table.mark_disconnected(recovered_peer)
    assert node.routing_table.find(recovered_peer).connected is False

    # Stub out the dial path so the bootstrap call resolves successfully.
    class _FakeConn:
        remote_peer_id = recovered_peer

    async def fake_get_or_dial(_pid):
        return _FakeConn()

    async def fake_identify(_conn):
        return []

    async def fake_walk(_target):
        # The post-dial walk would otherwise dial recovered_peer,
        # fail (no real network), and re-mark it disconnected. Skip it
        # to isolate the mark_connected behaviour under test.
        return []

    monkeypatch.setattr(node.peer_store, "get_or_dial", fake_get_or_dial)
    monkeypatch.setattr(node, "_perform_identify", fake_identify)
    monkeypatch.setattr(node, "_iterative_find_node", fake_walk)
    monkeypatch.setattr(node, "_setup_kad_handler", lambda _conn: None)
    monkeypatch.setattr(node, "_setup_identify_handler", lambda _conn: None)
    monkeypatch.setattr(node, "_setup_identify_push_handler", lambda _conn: None)

    # Patch the multiaddr parsers so the bootstrap path accepts an
    # opaque addr string and resolves it to ``recovered_peer``.
    monkeypatch.setattr(
        "kademlite.dht_bootstrap._parse_peer_multiaddr",
        lambda _addr: (recovered_peer, "192.0.2.1", 4001),
    )
    monkeypatch.setattr(
        "kademlite.dht_bootstrap.parse_multiaddr_string",
        lambda _addr: b"/ip4/192.0.2.1/tcp/4001",
    )

    await node._bootstrap_dial_peers(["/ip4/192.0.2.1/tcp/4001/p2p/IGNORED"])

    entry = node.routing_table.find(recovered_peer)
    assert entry is not None
    assert entry.connected is True


@pytest.mark.asyncio
async def test_bootstrap_walk_skip_threshold_is_strict_less_than_k():
    """Boundary check: at exactly rt_size == k, the walk is skipped.
    At rt_size == k - 1, the walk runs."""
    # At-threshold: skip
    node_a = DhtNode(identity=Ed25519Identity.generate(), k=4)
    walk_a = AsyncMock(return_value=[])
    node_a._iterative_find_node = walk_a
    for i in range(1, 5):  # exactly k=4 entries
        node_a.routing_table.add_or_update(_make_peer_id(i), [b"/seed"])
    await node_a._bootstrap_dial_peers([])
    walk_a.assert_not_awaited()

    # Just below: run
    node_b = DhtNode(identity=Ed25519Identity.generate(), k=4)
    walk_b = AsyncMock(return_value=[])
    node_b._iterative_find_node = walk_b
    for i in range(1, 4):  # k-1 = 3 entries
        node_b.routing_table.add_or_update(_make_peer_id(i), [b"/seed"])
    await node_b._bootstrap_dial_peers([])
    walk_b.assert_awaited_once()


# ---------------------------------------------------------------------------
# Fix C: requester-side seed filter (mirror of Fix A on the lookup side)
#
# ``_iterative_find_node`` and ``_iterative_get_value`` seed their candidate
# set from ``self.routing_table.closest_peers(target, self._k)``. Without
# ``connected_only=True`` on that seed, the closest ``k`` entries can be
# fully populated by stale ``connected=False`` rows; ``_is_peer_reachable``
# then skips every one of them in ``fill()``, no task is dispatched, and
# connected candidates that sit just past the closest-``k`` window are
# never reached. Mirror the responder-side filter on the requester so the
# same one-line flip closes the same defect on both code paths.
# ---------------------------------------------------------------------------


def _seed_connected(node: DhtNode, peer_id: bytes, addrs: list[bytes]) -> None:
    """Add a peer to the routing table and explicitly mark it reachable."""
    node.routing_table.add_or_update(peer_id, addrs)
    node.routing_table.mark_connected(peer_id)


def _seed_disconnected(node: DhtNode, peer_id: bytes, addrs: list[bytes]) -> None:
    """Add a peer to the routing table and mark it unreachable, simulating
    a stale entry from a previously-live peer that has since dropped."""
    node.routing_table.add_or_update(peer_id, addrs)
    node.routing_table.mark_disconnected(peer_id)


def _xor_closest_n(peer_ids: list[bytes], target: bytes) -> list[bytes]:
    """Return ``peer_ids`` sorted by XOR distance to ``target`` (closest
    first). Mirrors the ordering ``RoutingTable.closest_peers`` applies."""
    from kademlite.routing import kad_key, xor_distance

    target_kad = kad_key(target)
    return sorted(peer_ids, key=lambda p: xor_distance(kad_key(p), target_kad))


@pytest.mark.asyncio
async def test_iterative_get_value_skips_stale_seeds_to_reach_live_peer():
    """The closest ``k`` entries to the key are all marked disconnected;
    one connected peer holds the value but sits just past that window.

    Without the requester-side filter on the seed, the lookup never
    reaches the connected peer because the seed exhausts the closest-``k``
    window with stale entries that ``fill()`` then skips. With the
    filter, the seed pulls the closest ``k`` *reachable* entries,
    which includes the live value holder, and the value is returned.
    """
    k = 5
    node = DhtNode(rpc_timeout=5.0, dial_timeout=5.0, alpha=3, k=k)
    lookup_key = b"/some/lookup-key"

    # Build k+1 peer IDs and order them by XOR closeness to the key.
    candidates = [_make_peer_id(100 + i) for i in range(k + 1)]
    ordered = _xor_closest_n(candidates, lookup_key)

    # Closest k -> stale; the one at position k -> live and holds value.
    for pid in ordered[:k]:
        _seed_disconnected(node, pid, [b"/dead"])
    live_holder = ordered[k]
    _seed_connected(node, live_holder, [b"/live"])

    target_value = b"requester-side-fix-works"

    async def stub_get_value_single(peer_id, addrs, key):
        if peer_id == live_holder:
            return target_value, None, []
        # Stale peers should never be queried (filtered at seed),
        # but if they are, simulate a dial failure.
        return None, None, []

    node._get_value_single = stub_get_value_single  # type: ignore[assignment]

    result = await node._iterative_get_value(lookup_key)
    assert result == target_value


@pytest.mark.asyncio
async def test_iterative_find_node_skips_stale_seeds_to_reach_live_peer():
    """Symmetric of the GET test for the FIND_NODE path: the closest
    ``k`` entries are stale, the connected peer-at-``k`` is the only
    one the lookup can talk to. Result must contain the live peer."""
    k = 5
    node = DhtNode(rpc_timeout=5.0, dial_timeout=5.0, alpha=3, k=k)
    target = b"/some/find-target"

    candidates = [_make_peer_id(200 + i) for i in range(k + 1)]
    ordered = _xor_closest_n(candidates, target)

    for pid in ordered[:k]:
        _seed_disconnected(node, pid, [b"/dead"])
    live_peer = ordered[k]
    _seed_connected(node, live_peer, [b"/live"])

    async def stub_find_node_single(peer_id, addrs, lookup_target):
        # Live peer responds with no closer peers (terminal).
        return []

    node._find_node_single = stub_find_node_single  # type: ignore[assignment]

    result = await node._iterative_find_node(target)
    result_ids = {pid for pid, _ in result}
    assert live_peer in result_ids


@pytest.mark.asyncio
async def test_iterative_get_value_returns_none_when_only_stale_in_table():
    """When the routing table holds *only* stale entries and no live
    peers, the lookup must short-circuit cleanly with ``None`` rather
    than spin trying to dispatch unreachable peers. The seed filter
    is what produces this shape: ``closest_peers(connected_only=True)``
    returns ``[]``, the early-return path fires."""
    node = DhtNode(rpc_timeout=5.0, dial_timeout=5.0, alpha=3, k=5)
    lookup_key = b"/lookup"

    for i in range(5):
        _seed_disconnected(node, _make_peer_id(300 + i), [b"/dead"])

    queries = []

    async def stub_get_value_single(peer_id, addrs, key):
        queries.append(peer_id)
        return None, None, []

    node._get_value_single = stub_get_value_single  # type: ignore[assignment]

    result = await node._iterative_get_value(lookup_key)
    assert result is None
    # Critical: no peer queries ever fired. Pre-fix, the lookup would
    # have populated peer_map with the dead seeds and looped through
    # ``fill()`` skipping each one - no dispatched tasks, but more work.
    # Post-fix, the early-return on empty seed is the cheaper path.
    assert queries == []


@pytest.mark.asyncio
async def test_iterative_find_node_returns_empty_when_only_stale_in_table():
    """Symmetric early-return check on the FIND_NODE path."""
    node = DhtNode(rpc_timeout=5.0, dial_timeout=5.0, alpha=3, k=5)

    for i in range(5):
        _seed_disconnected(node, _make_peer_id(400 + i), [b"/dead"])

    queries = []

    async def stub_find_node_single(peer_id, addrs, target):
        queries.append(peer_id)
        return []

    node._find_node_single = stub_find_node_single  # type: ignore[assignment]

    result = await node._iterative_find_node(b"/target")
    assert result == []
    assert queries == []


@pytest.mark.asyncio
async def test_iterative_lookups_unaffected_when_all_peers_connected():
    """When every routing-table entry is connected, the requester-side
    filter is a no-op and the lookup behaves exactly as before. Pins
    the no-regression contract."""
    k = 5
    node = DhtNode(rpc_timeout=5.0, dial_timeout=5.0, alpha=3, k=k)
    lookup_key = b"/key"

    candidates = [_make_peer_id(500 + i) for i in range(k)]
    ordered = _xor_closest_n(candidates, lookup_key)
    for pid in ordered:
        _seed_connected(node, pid, [b"/live"])
    holder = ordered[0]
    target_value = b"v"

    async def stub_get_value_single(peer_id, addrs, key):
        if peer_id == holder:
            return target_value, None, []
        return None, None, []

    node._get_value_single = stub_get_value_single  # type: ignore[assignment]

    result = await node._iterative_get_value(lookup_key)
    assert result == target_value
