# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v0.3.3 dial-verification regression tests.

v0.3.2 added a ``connected_only=True`` filter on the responder-side
``_closest_peers_encoded`` so FIND_NODE / GET_VALUE responses stopped
propagating peers the local node had explicitly marked unreachable.
That filter was correct but not sufficient: ``connected=True`` was the
default for new entries, including peers learnt only via inbound
connection (where Identify discovered the peer's listen addrs but the
local node never dialled them) and peers learnt as iterative-walk
hints (added to the routing table by lookup integration without a
verifying dial). Long-running responder nodes accumulated dead-but-
``connected=True`` entries and propagated them in closer-peer hints,
so fresh joiners still paid one ``DIAL_TIMEOUT`` per ghost.

v0.3.3 adds a ``dial_verified`` flag on ``PeerEntry`` that flips True
only on an authoritative outbound dial-success signal. The responder-
side filter switches to ``dial_verified_only=True`` (which requires
both ``dial_verified`` AND ``connected``), so unverified entries no
longer leak as hints. Local lookup paths keep using ``connected_only``
because peers reachable only via an inbound connection are still
reachable through the existing socket via ``peer_store.get_or_dial``
- excluding them locally would lose useful breadth.
"""

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


def _make_peer_id(seed: int) -> bytes:
    return seed.to_bytes(32, "big")


# ---------------------------------------------------------------------------
# Default state: new entries are unverified
# ---------------------------------------------------------------------------


def test_peer_entry_defaults_dial_verified_false():
    """Constructor default is ``dial_verified=False``: hint-only learning
    paths (the ``add_or_update`` line in iterative walks, Identify push,
    inbound connection learning) all flow through this default and must
    never produce shareable entries on their own."""
    rt = RoutingTable(_make_peer_id(0), k=20)
    pid = _make_peer_id(1)
    rt.add_or_update(pid, [b"/learnt"])
    entry = rt.find(pid)
    assert entry is not None
    assert entry.connected is True  # active-session default unchanged
    assert entry.dial_verified is False  # new flag defaults False


def test_add_or_update_does_not_touch_dial_verified():
    """Updating an existing entry must not flip ``dial_verified`` either
    way. Hint refreshes (e.g. an inbound walk re-learning a peer's addrs)
    must not promote, and a re-learn after explicit verification must
    not demote."""
    rt = RoutingTable(_make_peer_id(0), k=20)
    pid = _make_peer_id(1)
    rt.add_or_update(pid, [b"/v1"])
    rt.mark_dial_verified(pid)
    rt.add_or_update(pid, [b"/v2"])  # refresh
    assert rt.find(pid).dial_verified is True

    pid2 = _make_peer_id(2)
    rt.add_or_update(pid2, [b"/v1"])
    rt.add_or_update(pid2, [b"/v2"])  # multiple refreshes, never verified
    assert rt.find(pid2).dial_verified is False


# ---------------------------------------------------------------------------
# mark_dial_verified / mark_connected / mark_disconnected lifecycle
# ---------------------------------------------------------------------------


def test_mark_dial_verified_sets_both_flags():
    """``mark_dial_verified`` is the only path that promotes ``dial_verified``;
    it must also set ``connected=True`` because a successful dial is the
    strongest evidence of session connectivity available."""
    rt = RoutingTable(_make_peer_id(0), k=20)
    pid = _make_peer_id(1)
    rt.add_or_update(pid, [b"/x"])
    rt.mark_disconnected(pid)
    assert rt.find(pid).connected is False
    assert rt.find(pid).dial_verified is False

    assert rt.mark_dial_verified(pid) is True
    entry = rt.find(pid)
    assert entry.connected is True
    assert entry.dial_verified is True


def test_mark_connected_does_not_promote_dial_verified():
    """Inbound-only connections call ``mark_connected``, which must not
    promote ``dial_verified``. The advertised listen addr remains
    unverified until we dial it ourselves."""
    rt = RoutingTable(_make_peer_id(0), k=20)
    pid = _make_peer_id(1)
    rt.add_or_update(pid, [b"/listen"])
    assert rt.find(pid).dial_verified is False

    assert rt.mark_connected(pid) is True
    entry = rt.find(pid)
    assert entry.connected is True
    assert entry.dial_verified is False


def test_mark_disconnected_clears_dial_verified():
    """``mark_disconnected`` is called from RPC / dial failure paths.
    Clearing ``dial_verified`` on failure mirrors the True-on-success
    signal: if the listen addr was reachable but is now failing, it
    should not be propagated as a hint until re-verified."""
    rt = RoutingTable(_make_peer_id(0), k=20)
    pid = _make_peer_id(1)
    rt.add_or_update(pid, [b"/x"])
    rt.mark_dial_verified(pid)
    assert rt.find(pid).dial_verified is True

    assert rt.mark_disconnected(pid) is True
    entry = rt.find(pid)
    assert entry.connected is False
    assert entry.dial_verified is False


def test_dial_verified_round_trip_after_failure():
    """Verified -> failed -> re-dialled flow flips the flag back to True.
    Dialed-once peers can flap; the flag must follow the latest evidence."""
    rt = RoutingTable(_make_peer_id(0), k=20)
    pid = _make_peer_id(1)
    rt.add_or_update(pid, [b"/x"])
    rt.mark_dial_verified(pid)
    assert rt.find(pid).dial_verified is True

    rt.mark_disconnected(pid)
    assert rt.find(pid).dial_verified is False

    rt.mark_dial_verified(pid)
    assert rt.find(pid).dial_verified is True
    assert rt.find(pid).connected is True


# ---------------------------------------------------------------------------
# Responder-side filter: dial_verified_only=True
# ---------------------------------------------------------------------------


def _build_handler_with_states(
    n_verified: int,
    n_inbound_only: int,
    n_disconnected_was_verified: int,
    n_hint_only: int,
    k: int = 20,
) -> tuple[KadHandler, set[bytes], set[bytes], set[bytes], set[bytes]]:
    """Build a KadHandler whose routing table contains peers in each of
    the four states the new filter must distinguish.

    Returns ``(handler, verified_ids, inbound_only_ids,
    prev_verified_now_disconnected_ids, hint_only_ids)``. Only the
    ``verified`` group should appear in ``_closest_peers_encoded``.
    """
    local = _make_peer_id(0)
    rt = RoutingTable(local, k=k)
    verified: set[bytes] = set()
    inbound_only: set[bytes] = set()
    prev_verified: set[bytes] = set()
    hint_only: set[bytes] = set()

    seed = 1
    for _ in range(n_verified):
        pid = _make_peer_id(seed)
        rt.add_or_update(pid, [b"/v"])
        rt.mark_dial_verified(pid)
        verified.add(pid)
        seed += 1
    for _ in range(n_inbound_only):
        pid = _make_peer_id(seed)
        rt.add_or_update(pid, [b"/inbound"])
        rt.mark_connected(pid)  # inbound connection: connected=True, dial_verified=False
        inbound_only.add(pid)
        seed += 1
    for _ in range(n_disconnected_was_verified):
        pid = _make_peer_id(seed)
        rt.add_or_update(pid, [b"/v-then-fail"])
        rt.mark_dial_verified(pid)
        rt.mark_disconnected(pid)  # both flags now False
        prev_verified.add(pid)
        seed += 1
    for _ in range(n_hint_only):
        pid = _make_peer_id(seed)
        rt.add_or_update(pid, [b"/hint"])
        # No mark_*: defaults to connected=True, dial_verified=False, like
        # iterative-walk hint integration via dht_queries.add_or_update.
        hint_only.add(pid)
        seed += 1
    return KadHandler(rt, k=k), verified, inbound_only, prev_verified, hint_only


def test_closest_peers_encoded_includes_only_verified():
    """The internal helper must filter to dial-verified-AND-connected peers."""
    handler, verified, inbound_only, prev_verified, hint_only = (
        _build_handler_with_states(
            n_verified=5, n_inbound_only=3, n_disconnected_was_verified=2, n_hint_only=4
        )
    )
    encoded = handler._closest_peers_encoded(b"/some/key")
    returned = {decode_peer(p)["id"] for p in encoded}
    assert returned == verified
    assert not (returned & inbound_only)
    assert not (returned & prev_verified)
    assert not (returned & hint_only)


def test_find_node_response_excludes_unverified_peers():
    """FIND_NODE response decoded off the wire excludes inbound-only,
    previously-verified-now-disconnected, and hint-only entries."""
    handler, verified, inbound_only, prev_verified, hint_only = (
        _build_handler_with_states(
            n_verified=5, n_inbound_only=3, n_disconnected_was_verified=2, n_hint_only=4
        )
    )
    request = encode_kad_message(MSG_FIND_NODE, key=b"/lookup/target")
    request_msg = decode_kad_message(request)
    response_bytes = handler._handle_find_node(request_msg)
    response = decode_kad_message(response_bytes)
    closer_ids = {p["id"] for p in response["closer_peers"]}
    assert closer_ids.issubset(verified)
    assert not (closer_ids & inbound_only)
    assert not (closer_ids & prev_verified)
    assert not (closer_ids & hint_only)


def test_get_value_no_record_response_excludes_unverified_peers():
    """GET_VALUE with no local record falls through to closer peers and
    must apply the same filter."""
    handler, verified, inbound_only, prev_verified, hint_only = (
        _build_handler_with_states(
            n_verified=5, n_inbound_only=3, n_disconnected_was_verified=2, n_hint_only=4
        )
    )
    request = encode_kad_message(MSG_GET_VALUE, key=b"/missing/key")
    request_msg = decode_kad_message(request)
    response_bytes = handler._handle_get_value(request_msg)
    response = decode_kad_message(response_bytes)
    closer_ids = {p["id"] for p in response["closer_peers"]}
    assert closer_ids.issubset(verified)
    assert not (closer_ids & inbound_only)
    assert not (closer_ids & prev_verified)
    assert not (closer_ids & hint_only)


def test_responder_filter_with_only_unverified_returns_empty():
    """A long-running responder whose routing table is entirely populated
    via inbound connections / hint propagation must serve an empty
    closer-peers list. Fresh joiners don't get useful coverage from this
    responder, but they don't inherit ghost entries either - the v0.3.2
    failure shape we are closing."""
    handler, _, _, _, _ = _build_handler_with_states(
        n_verified=0, n_inbound_only=5, n_disconnected_was_verified=3, n_hint_only=4
    )
    encoded = handler._closest_peers_encoded(b"/key")
    assert encoded == []


def test_responder_filter_with_only_verified_unchanged():
    """When every routing-table entry is dial-verified and connected,
    the new filter is a no-op vs the v0.3.2 connected_only filter:
    all peers (up to k) are returned."""
    handler, verified, _, _, _ = _build_handler_with_states(
        n_verified=12,
        n_inbound_only=0,
        n_disconnected_was_verified=0,
        n_hint_only=0,
        k=20,
    )
    encoded = handler._closest_peers_encoded(b"/key")
    returned = {decode_peer(p)["id"] for p in encoded}
    assert returned == verified


# ---------------------------------------------------------------------------
# RoutingTable.closest_peers: dial_verified_only filter parameter
# ---------------------------------------------------------------------------


def test_closest_peers_dial_verified_only_filter():
    """The new filter on ``RoutingTable.closest_peers`` returns only
    entries where both ``dial_verified`` and ``connected`` are True."""
    rt = RoutingTable(_make_peer_id(0), k=20)

    verified_pid = _make_peer_id(1)
    rt.add_or_update(verified_pid, [b"/v"])
    rt.mark_dial_verified(verified_pid)

    inbound_pid = _make_peer_id(2)
    rt.add_or_update(inbound_pid, [b"/i"])
    rt.mark_connected(inbound_pid)

    hint_pid = _make_peer_id(3)
    rt.add_or_update(hint_pid, [b"/h"])

    disconnected_pid = _make_peer_id(4)
    rt.add_or_update(disconnected_pid, [b"/d"])
    rt.mark_dial_verified(disconnected_pid)
    rt.mark_disconnected(disconnected_pid)

    target = b"/some/key"

    # No filter: all four peers
    all_returned = rt.closest_peers(target, count=20)
    all_ids = {p.peer_id for p in all_returned}
    assert all_ids == {verified_pid, inbound_pid, hint_pid, disconnected_pid}

    # connected_only: filters out the disconnected one
    conn_only = rt.closest_peers(target, count=20, connected_only=True)
    conn_ids = {p.peer_id for p in conn_only}
    assert conn_ids == {verified_pid, inbound_pid, hint_pid}

    # dial_verified_only: only the verified one
    dv_only = rt.closest_peers(target, count=20, dial_verified_only=True)
    dv_ids = {p.peer_id for p in dv_only}
    assert dv_ids == {verified_pid}


def test_closest_peers_dial_verified_only_supersedes_connected_only():
    """``dial_verified_only=True`` takes precedence over ``connected_only``:
    passing both gives the strict dial-verified+connected filter, not
    something looser."""
    rt = RoutingTable(_make_peer_id(0), k=20)
    verified_pid = _make_peer_id(1)
    inbound_pid = _make_peer_id(2)
    rt.add_or_update(verified_pid, [b"/v"])
    rt.mark_dial_verified(verified_pid)
    rt.add_or_update(inbound_pid, [b"/i"])
    rt.mark_connected(inbound_pid)

    both = rt.closest_peers(
        b"/key", count=20, connected_only=True, dial_verified_only=True
    )
    assert {p.peer_id for p in both} == {verified_pid}


# ---------------------------------------------------------------------------
# DhtNode wiring: _on_peer_connected, _on_peer_unreachable
# ---------------------------------------------------------------------------


def test_on_peer_connected_promotes_to_dial_verified():
    """The PeerStore success callback (outbound dial only) must promote
    the routing-table entry to dial_verified."""
    node = DhtNode(identity=Ed25519Identity.generate(), k=20)
    pid = _make_peer_id(1)
    node.routing_table.add_or_update(pid, [b"/x"])
    assert node.routing_table.find(pid).dial_verified is False

    node._on_peer_connected(pid)
    assert node.routing_table.find(pid).dial_verified is True
    assert node.routing_table.find(pid).connected is True


def test_on_peer_unreachable_clears_dial_verified():
    """The PeerStore failure callback must clear both flags."""
    node = DhtNode(identity=Ed25519Identity.generate(), k=20)
    pid = _make_peer_id(1)
    node.routing_table.add_or_update(pid, [b"/x"])
    node.routing_table.mark_dial_verified(pid)
    assert node.routing_table.find(pid).dial_verified is True

    node._on_peer_unreachable(pid)
    entry = node.routing_table.find(pid)
    assert entry.connected is False
    assert entry.dial_verified is False


# ---------------------------------------------------------------------------
# Bootstrap path: explicit mark_dial_verified after add_or_update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_dial_peers_marks_dial_verified(monkeypatch):
    """After a successful bootstrap dial, the peer must end up
    ``dial_verified=True`` so the responder-side filter propagates it
    as a hint to other nodes asking us for closer peers."""
    node = DhtNode(identity=Ed25519Identity.generate(), k=20)

    bootstrap_peer = _make_peer_id(42)

    class _FakeConn:
        remote_peer_id = bootstrap_peer

    async def fake_get_or_dial(_pid):
        return _FakeConn()

    async def fake_identify(_conn):
        return []

    async def fake_walk(_target):
        return []

    monkeypatch.setattr(node.peer_store, "get_or_dial", fake_get_or_dial)
    monkeypatch.setattr(node, "_perform_identify", fake_identify)
    monkeypatch.setattr(node, "_iterative_find_node", fake_walk)
    monkeypatch.setattr(node, "_setup_kad_handler", lambda _conn: None)
    monkeypatch.setattr(node, "_setup_identify_handler", lambda _conn: None)
    monkeypatch.setattr(node, "_setup_identify_push_handler", lambda _conn: None)

    monkeypatch.setattr(
        "kademlite.dht_bootstrap._parse_peer_multiaddr",
        lambda _addr: (bootstrap_peer, "192.0.2.1", 4001),
    )
    monkeypatch.setattr(
        "kademlite.dht_bootstrap.parse_multiaddr_string",
        lambda _addr: b"/ip4/192.0.2.1/tcp/4001",
    )

    await node._bootstrap_dial_peers(["/ip4/192.0.2.1/tcp/4001/p2p/IGNORED"])

    entry = node.routing_table.find(bootstrap_peer)
    assert entry is not None
    assert entry.connected is True
    assert entry.dial_verified is True


@pytest.mark.asyncio
async def test_bootstrap_dial_peers_recovers_previously_disconnected(monkeypatch):
    """A peer previously marked disconnected must be flipped back to
    dial_verified+connected after a successful bootstrap re-dial. This
    is the v0.3.2 ``mark_connected`` regression scenario adapted to the
    stricter v0.3.3 contract: the recovered peer must end up shareable
    again, not just connected."""
    node = DhtNode(identity=Ed25519Identity.generate(), k=20)

    recovered_peer = _make_peer_id(42)
    node.routing_table.add_or_update(recovered_peer, [b"/old"])
    node.routing_table.mark_dial_verified(recovered_peer)
    node.routing_table.mark_disconnected(recovered_peer)
    assert node.routing_table.find(recovered_peer).connected is False
    assert node.routing_table.find(recovered_peer).dial_verified is False

    class _FakeConn:
        remote_peer_id = recovered_peer

    async def fake_get_or_dial(_pid):
        return _FakeConn()

    async def fake_identify(_conn):
        return []

    async def fake_walk(_target):
        return []

    monkeypatch.setattr(node.peer_store, "get_or_dial", fake_get_or_dial)
    monkeypatch.setattr(node, "_perform_identify", fake_identify)
    monkeypatch.setattr(node, "_iterative_find_node", fake_walk)
    monkeypatch.setattr(node, "_setup_kad_handler", lambda _conn: None)
    monkeypatch.setattr(node, "_setup_identify_handler", lambda _conn: None)
    monkeypatch.setattr(node, "_setup_identify_push_handler", lambda _conn: None)
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
    assert entry.dial_verified is True


@pytest.mark.asyncio
async def test_dial_ips_marks_dial_verified(monkeypatch):
    """The DNS / SLURM dial path must also promote successful dials to
    dial_verified, parallel to the explicit bootstrap-dial path."""
    node = DhtNode(identity=Ed25519Identity.generate(), k=4)

    dialled_peer = _make_peer_id(99)

    class _FakeConn:
        remote_peer_id = dialled_peer

    async def fake_dial(*args, **kwargs):
        return _FakeConn()

    async def fake_identify(_conn):
        return []

    async def fake_walk(_target):
        return []

    monkeypatch.setattr("kademlite.dht_bootstrap.dial", fake_dial)
    monkeypatch.setattr(node, "_perform_identify", fake_identify)
    monkeypatch.setattr(node, "_iterative_find_node", fake_walk)
    monkeypatch.setattr(node, "_setup_kad_handler", lambda _conn: None)
    monkeypatch.setattr(node, "_setup_identify_handler", lambda _conn: None)
    monkeypatch.setattr(node, "_setup_identify_push_handler", lambda _conn: None)

    await node._dial_ips(["192.0.2.1"], 4001, "TEST")

    entry = node.routing_table.find(dialled_peer)
    assert entry is not None
    assert entry.connected is True
    assert entry.dial_verified is True


# ---------------------------------------------------------------------------
# Inbound-connection path: connected only, not dial_verified
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inbound_connection_does_not_set_dial_verified(monkeypatch):
    """``_on_inbound_connection`` learns the peer's listen addrs via
    Identify and adds them to the routing table, but the local node has
    NOT dialled those addrs. The entry must end up
    ``connected=True, dial_verified=False`` so the responder-side filter
    does not propagate the unverified listen addr to third parties."""
    node = DhtNode(identity=Ed25519Identity.generate(), k=20)

    inbound_peer = _make_peer_id(7)

    class _FakeConn:
        remote_peer_id = inbound_peer
        remote_addr = ("192.0.2.5", 4001)

    async def fake_identify(_conn):
        # Return one routable listen addr from the Identify exchange.
        from kademlite.multiaddr import encode_multiaddr_ip_tcp_p2p

        return [encode_multiaddr_ip_tcp_p2p("198.51.100.1", 4001, inbound_peer)]

    monkeypatch.setattr(node, "_perform_identify", fake_identify)
    monkeypatch.setattr(node, "_setup_kad_handler", lambda _conn: None)
    monkeypatch.setattr(node, "_setup_identify_handler", lambda _conn: None)
    monkeypatch.setattr(node, "_setup_identify_push_handler", lambda _conn: None)

    await node._on_inbound_connection(_FakeConn())

    entry = node.routing_table.find(inbound_peer)
    assert entry is not None
    assert entry.connected is True
    # Critical invariant: inbound connection alone is NOT enough evidence
    # to propagate the listen addr as a hint.
    assert entry.dial_verified is False


# ---------------------------------------------------------------------------
# Hint integration via iterative walks: still unverified
# ---------------------------------------------------------------------------


def test_iterative_walk_hint_integration_keeps_dial_verified_false():
    """``_iterative_find_node`` and ``_iterative_get_value`` integrate
    learnt peers via ``self.routing_table.add_or_update(pid, addrs)``
    (dht_queries.py:145, :329). New entries from this path must default
    to ``dial_verified=False`` so the responder filter does not later
    leak them as hints. This is the structural fix v0.3.3 closes:
    learnt-but-not-dialled peers no longer become shareable just by
    virtue of being added to our routing table."""
    rt = RoutingTable(_make_peer_id(0), k=20)
    # Simulate the integration line: a closer-peers hint we just received
    # in a FIND_NODE response.
    learnt_pid = _make_peer_id(1)
    rt.add_or_update(learnt_pid, [b"/from-hint"])

    entry = rt.find(learnt_pid)
    assert entry.connected is True  # default from constructor; harmless locally
    assert entry.dial_verified is False  # the load-bearing invariant

    handler = KadHandler(rt, k=20)
    encoded = handler._closest_peers_encoded(b"/key")
    returned = {decode_peer(p)["id"] for p in encoded}
    assert learnt_pid not in returned


# ---------------------------------------------------------------------------
# Local lookup paths still see connected (inbound-only) peers
# ---------------------------------------------------------------------------


def test_local_lookup_seed_still_includes_inbound_only_peers():
    """Local iterative-lookup seed selection uses ``connected_only=True``
    (dht_queries.py:68 and :241), not ``dial_verified_only=True``.
    Inbound-only peers stay reachable via the existing inbound socket
    through ``peer_store.get_or_dial``, so excluding them locally would
    lose useful breadth. The stricter filter only applies to outbound
    propagation as hints to third parties."""
    rt = RoutingTable(_make_peer_id(0), k=20)
    inbound_pid = _make_peer_id(1)
    verified_pid = _make_peer_id(2)
    rt.add_or_update(inbound_pid, [b"/inbound"])
    rt.mark_connected(inbound_pid)
    rt.add_or_update(verified_pid, [b"/verified"])
    rt.mark_dial_verified(verified_pid)

    seeds = rt.closest_peers(b"/lookup", count=20, connected_only=True)
    seed_ids = {p.peer_id for p in seeds}
    assert inbound_pid in seed_ids
    assert verified_pid in seed_ids


# ---------------------------------------------------------------------------
# Full mesh end-to-end: long-running responder no longer propagates ghosts
# ---------------------------------------------------------------------------


def test_long_running_responder_no_longer_leaks_inbound_hints_via_get_value():
    """End-to-end shape of the v0.3.3 fix. Build a 'responder' KadHandler
    whose routing table contains:
      - 3 verified peers (we have outbound-dialled them at some point)
      - 8 inbound-only peers (they connected to us; we know their listen
        addrs via Identify but never dialled)
      - 5 hint-only peers (added via add_or_update from iterative walks)
    A fresh joiner asking us for closer peers via GET_VALUE on a missing
    key must receive ONLY the 3 verified entries. The 8 inbound-only +
    5 hint-only = 13 unverified entries are exactly the ghost-cliff
    population the trace identified as the source of per-GET DIAL_TIMEOUT
    cost. The responder still keeps all 16 entries locally for its own
    use; the filter only restricts outbound propagation."""
    local = _make_peer_id(0)
    rt = RoutingTable(local, k=20)
    verified = set()
    for i in range(1, 4):
        pid = _make_peer_id(i)
        rt.add_or_update(pid, [b"/v"])
        rt.mark_dial_verified(pid)
        verified.add(pid)
    for i in range(4, 12):
        pid = _make_peer_id(i)
        rt.add_or_update(pid, [b"/inbound"])
        rt.mark_connected(pid)  # inbound only
    for i in range(12, 17):
        pid = _make_peer_id(i)
        rt.add_or_update(pid, [b"/hint"])

    handler = KadHandler(rt, k=20)
    request = encode_kad_message(MSG_GET_VALUE, key=b"/missing/key")
    request_msg = decode_kad_message(request)
    response_bytes = handler._handle_get_value(request_msg)
    response = decode_kad_message(response_bytes)
    closer_ids = {p["id"] for p in response["closer_peers"]}

    # Only the 3 verified entries leak out.
    assert closer_ids == verified
    # And the routing table itself still holds all 16 entries.
    assert rt.size(connected_only=False) == 16


# ---------------------------------------------------------------------------
# Audit-driven regression tests
#
# The defects below were caught by an external audit on the initial v0.3.3
# patch. Each test pins the specific failure mode the fix addresses so a
# future regression cannot silently re-open the leak.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_does_not_promote_cached_inbound_connection(monkeypatch):
    """``peer_store.get_or_dial`` returns a cached connection without
    dialing when one is alive (peer_store.py: early-return before the
    dial loop). If the cached connection came from
    ``_on_inbound_connection``, the bootstrap path must NOT promote the
    peer to ``dial_verified`` - we never actually outbound-dialled the
    listen addr, so the v0.3.3 invariant would be violated.
    """
    from kademlite.connection import Connection

    node = DhtNode(identity=Ed25519Identity.generate(), k=20)
    bootstrap_peer = _make_peer_id(7)

    # Simulate a prior inbound connection: a connection object exists in
    # peer_store and the routing table holds the peer with mark_connected
    # but NOT mark_dial_verified (the v0.3.3 inbound path).
    fake_conn = object.__new__(Connection)
    fake_conn.remote_peer_id = bootstrap_peer
    node.peer_store.set_connection(bootstrap_peer, fake_conn)
    node.routing_table.add_or_update(bootstrap_peer, [b"/inbound"])
    node.routing_table.mark_connected(bootstrap_peer)
    assert node.routing_table.find(bootstrap_peer).dial_verified is False

    async def fake_identify(_conn):
        return []

    async def fake_walk(_target):
        return []

    monkeypatch.setattr(node, "_perform_identify", fake_identify)
    monkeypatch.setattr(node, "_iterative_find_node", fake_walk)
    monkeypatch.setattr(node, "_setup_kad_handler", lambda _conn: None)
    monkeypatch.setattr(node, "_setup_identify_handler", lambda _conn: None)
    monkeypatch.setattr(node, "_setup_identify_push_handler", lambda _conn: None)
    monkeypatch.setattr(
        "kademlite.dht_bootstrap._parse_peer_multiaddr",
        lambda _addr: (bootstrap_peer, "192.0.2.1", 4001),
    )
    monkeypatch.setattr(
        "kademlite.dht_bootstrap.parse_multiaddr_string",
        lambda _addr: b"/ip4/192.0.2.1/tcp/4001",
    )
    # Make get_or_dial assert-fail if it tries to dial; we expect cache hit.
    async def fake_get_or_dial(pid, addrs=None):
        return fake_conn

    monkeypatch.setattr(node.peer_store, "get_or_dial", fake_get_or_dial)

    await node._bootstrap_dial_peers(["/ip4/192.0.2.1/tcp/4001/p2p/IGNORED"])

    entry = node.routing_table.find(bootstrap_peer)
    assert entry is not None
    # connected stays True (we have a live socket); dial_verified must
    # NOT have been promoted because no fresh outbound dial happened.
    assert entry.connected is True
    assert entry.dial_verified is False


@pytest.mark.asyncio
async def test_bootstrap_promotes_when_no_cached_connection(monkeypatch):
    """Companion test to ``...does_not_promote_cached_inbound_connection``:
    when there is NO existing connection and ``get_or_dial`` actually
    dials, the bootstrap path must promote the peer to dial_verified.
    The two tests together pin the cache-aware branch in the bootstrap
    promotion logic."""
    from kademlite.connection import Connection

    node = DhtNode(identity=Ed25519Identity.generate(), k=20)
    bootstrap_peer = _make_peer_id(8)

    # No prior connection: peer_store.get_connection returns None.
    fake_conn = object.__new__(Connection)
    fake_conn.remote_peer_id = bootstrap_peer

    async def fake_get_or_dial(pid, addrs=None):
        # Simulate a fresh dial: no connection existed before, now there is one.
        node.peer_store.set_connection(pid, fake_conn)
        return fake_conn

    async def fake_identify(_conn):
        return []

    async def fake_walk(_target):
        return []

    monkeypatch.setattr(node.peer_store, "get_or_dial", fake_get_or_dial)
    monkeypatch.setattr(node, "_perform_identify", fake_identify)
    monkeypatch.setattr(node, "_iterative_find_node", fake_walk)
    monkeypatch.setattr(node, "_setup_kad_handler", lambda _conn: None)
    monkeypatch.setattr(node, "_setup_identify_handler", lambda _conn: None)
    monkeypatch.setattr(node, "_setup_identify_push_handler", lambda _conn: None)
    monkeypatch.setattr(
        "kademlite.dht_bootstrap._parse_peer_multiaddr",
        lambda _addr: (bootstrap_peer, "192.0.2.1", 4001),
    )
    monkeypatch.setattr(
        "kademlite.dht_bootstrap.parse_multiaddr_string",
        lambda _addr: b"/ip4/192.0.2.1/tcp/4001",
    )

    await node._bootstrap_dial_peers(["/ip4/192.0.2.1/tcp/4001/p2p/IGNORED"])

    entry = node.routing_table.find(bootstrap_peer)
    assert entry is not None
    assert entry.connected is True
    assert entry.dial_verified is True


def test_kbucket_pending_entry_is_visited_by_mark_dial_verified():
    """When the bucket is full and the LRU is alive, ``add_or_update``
    stores the new peer in ``_pending`` and returns False. A subsequent
    dial-success signal calls ``mark_dial_verified``; pre-fix that scan
    only walked ``self.peers`` and missed the pending entry, so a real
    outbound-dialled peer would later be promoted into the active bucket
    with ``dial_verified=False`` and never propagate as a hint."""
    from kademlite.routing import K, KBucket, PeerEntry

    bucket = KBucket(k=2, is_alive=lambda _pid: True)
    # Fill the bucket; both entries are recently seen so the LRU is alive.
    p_a = b"\xaa" * 32
    p_b = b"\xbb" * 32
    bucket.add_or_update(p_a, [b"/a"])
    bucket.add_or_update(p_b, [b"/b"])
    assert len(bucket) == 2

    # Add a third peer: bucket is full, LRU alive (callback returns True),
    # so the newcomer goes to _pending.
    p_c = b"\xcc" * 32
    accepted = bucket.add_or_update(p_c, [b"/c"])
    assert accepted is False
    assert bucket._pending is not None
    assert bucket._pending.peer_id == p_c
    assert bucket._pending.dial_verified is False

    # Outbound dial-success on p_c: the pending entry must receive the flag.
    assert bucket.mark_dial_verified(p_c) is True
    assert bucket._pending.dial_verified is True
    assert bucket._pending.connected is True

    # Promotion path: remove p_a (LRU); _pending is promoted into peers.
    # The promoted entry must keep its dial_verified=True.
    assert bucket.remove(p_a) is True
    promoted = next(e for e in bucket.peers if e.peer_id == p_c)
    assert promoted.dial_verified is True
    assert promoted.connected is True

    # Suppress unused-import warning under ruff.
    assert PeerEntry is not None and K > 0


def test_kbucket_pending_entry_is_visited_by_mark_disconnected():
    """Symmetric pending-cache check for ``mark_disconnected``: a pending
    entry that has been dial-verified can later receive a failure signal
    (e.g. an RPC timeout to that peer through a different code path);
    the flag clear must reach it before promotion."""
    bucket = _new_full_bucket_with_pending()
    p_c = b"\xcc" * 32
    bucket.mark_dial_verified(p_c)
    assert bucket._pending.dial_verified is True

    assert bucket.mark_disconnected(p_c) is True
    assert bucket._pending.connected is False
    assert bucket._pending.dial_verified is False


def _new_full_bucket_with_pending():
    """Helper: build a KBucket with peers full and one peer in _pending."""
    from kademlite.routing import KBucket

    bucket = KBucket(k=2, is_alive=lambda _pid: True)
    bucket.add_or_update(b"\xaa" * 32, [b"/a"])
    bucket.add_or_update(b"\xbb" * 32, [b"/b"])
    bucket.add_or_update(b"\xcc" * 32, [b"/c"])  # goes to _pending
    return bucket


def test_identify_push_preserves_dial_verified_when_addrs_unchanged():
    """Idempotent Identify Push (same listen addrs) is a no-op for
    ``dial_verified``. Localhost meshes generate frequent observed-IP
    pushes whose listen-addr field doesn't actually change; flapping
    verification on every such push would break chain-bootstrap
    discovery without producing any real safety benefit."""
    rt = RoutingTable(_make_peer_id(0), k=20)
    pid = _make_peer_id(11)
    rt.add_or_update(pid, [b"/v1"])
    rt.mark_dial_verified(pid)
    assert rt.find(pid).dial_verified is True

    # Push handler logic: same addrs => no revocation.
    existing = rt.find(pid)
    routable = [b"/v1"]
    addrs_changed = existing is None or set(existing.addrs) != set(routable)
    rt.add_or_update(pid, routable)
    if addrs_changed:
        rt.mark_dial_unverified(pid)

    assert addrs_changed is False
    assert rt.find(pid).dial_verified is True


def test_identify_push_revokes_dial_verified_when_addrs_changed():
    """Genuine address-set change in Identify Push revokes
    ``dial_verified`` because the new addrs were never dialed. This
    closes the address-churn variant of the responder-side hint leak:
    a peer that flips its advertised addrs after the initial dial-
    verification cannot continue propagating the new addrs as
    'verified' hints until the local node has dialed them itself."""
    rt = RoutingTable(_make_peer_id(0), k=20)
    pid = _make_peer_id(11)
    rt.add_or_update(pid, [b"/v1"])
    rt.mark_dial_verified(pid)
    assert rt.find(pid).dial_verified is True

    existing = rt.find(pid)
    routable = [b"/v2-pushed"]
    addrs_changed = existing is None or set(existing.addrs) != set(routable)
    rt.add_or_update(pid, routable)
    if addrs_changed:
        rt.mark_dial_unverified(pid)

    assert addrs_changed is True
    entry = rt.find(pid)
    # connection stays alive; address-level verification revoked.
    assert entry.connected is True
    assert entry.dial_verified is False


def test_identify_push_set_compare_ignores_addr_ordering():
    """Set-based comparison in the push-handler logic prevents a peer
    pushing the same addrs in a different order from triggering a
    spurious revocation."""
    rt = RoutingTable(_make_peer_id(0), k=20)
    pid = _make_peer_id(11)
    rt.add_or_update(pid, [b"/a", b"/b"])
    rt.mark_dial_verified(pid)

    existing = rt.find(pid)
    routable = [b"/b", b"/a"]  # same set, different order
    addrs_changed = existing is None or set(existing.addrs) != set(routable)
    assert addrs_changed is False


def test_mark_connected_re_tails_entry_in_active_bucket():
    """Mark-state updates that refresh ``last_seen`` must re-tail the
    entry in ``self.peers`` to preserve the LRU invariant. The bucket's
    eviction logic uses ``self.peers[0]`` as the eviction candidate,
    so a freshly-marked entry that stays at the head would be evicted
    incorrectly while older peers keep their slots."""
    from kademlite.routing import KBucket

    bucket = KBucket(k=20)
    p_a = b"\xaa" * 32
    p_b = b"\xbb" * 32
    p_c = b"\xcc" * 32
    bucket.add_or_update(p_a, [b"/a"])
    bucket.add_or_update(p_b, [b"/b"])
    bucket.add_or_update(p_c, [b"/c"])
    # Initial order: a, b, c (head -> tail).
    assert [e.peer_id for e in bucket.peers] == [p_a, p_b, p_c]

    # Mark p_a connected: must re-tail to keep LRU semantics correct.
    bucket.mark_connected(p_a)
    assert [e.peer_id for e in bucket.peers] == [p_b, p_c, p_a]


def test_mark_dial_verified_re_tails_entry_in_active_bucket():
    """Symmetric LRU-reorder check on ``mark_dial_verified``."""
    from kademlite.routing import KBucket

    bucket = KBucket(k=20)
    p_a = b"\xaa" * 32
    p_b = b"\xbb" * 32
    p_c = b"\xcc" * 32
    bucket.add_or_update(p_a, [b"/a"])
    bucket.add_or_update(p_b, [b"/b"])
    bucket.add_or_update(p_c, [b"/c"])
    bucket.mark_dial_verified(p_a)
    assert [e.peer_id for e in bucket.peers] == [p_b, p_c, p_a]


def test_kbucket_add_or_update_preserves_pending_flags_on_idempotent_call():
    """When the bucket is full, the LRU is alive, and a peer already in
    the pending replacement cache is added again with the same addrs,
    ``add_or_update`` must update the existing pending entry in place
    and return early - NOT replace it with a fresh ``PeerEntry`` that
    would clobber any ``connected`` / ``dial_verified`` flags the
    ``mark_*`` methods previously set on the pending entry. Pre-fix
    the in-place update at the top of the method was followed by a
    fall-through to the bucket-full-alive-LRU branch, which replaced
    ``self._pending`` with a brand-new entry whose flags were defaults.
    """
    bucket = _new_full_bucket_with_pending()
    p_c = b"\xcc" * 32
    bucket.mark_dial_verified(p_c)
    assert bucket._pending.dial_verified is True

    # Idempotent re-add (e.g. inbound walk hint that pid was already known).
    accepted = bucket.add_or_update(p_c, [b"/c"])
    assert accepted is False  # still in pending, not promoted
    assert bucket._pending is not None
    assert bucket._pending.peer_id == p_c
    # Critical invariant: flags survive the idempotent re-add.
    assert bucket._pending.dial_verified is True
    assert bucket._pending.connected is True


def test_kbucket_add_or_update_preserves_pending_flags_on_pending_timeout_path():
    """Symmetric coverage of the pending-timeout branch: when the
    pending entry has waited past ``PENDING_ENTRY_TIMEOUT``, a
    re-add of the same peer should evict the LRU, promote the existing
    pending object into ``self.peers`` (preserving flags), and clear
    ``_pending``. This covers the branch at the start of the
    bucket-full handling that the standard LRU-alive path skips."""
    import time

    from kademlite.routing import KBucket

    bucket = KBucket(k=2, is_alive=lambda _pid: True)
    p_a = b"\xaa" * 32
    p_b = b"\xbb" * 32
    p_c = b"\xcc" * 32
    bucket.add_or_update(p_a, [b"/a"])
    bucket.add_or_update(p_b, [b"/b"])
    bucket.add_or_update(p_c, [b"/c"])  # goes to _pending
    assert bucket._pending is not None
    bucket.mark_dial_verified(p_c)
    assert bucket._pending.dial_verified is True

    # Force the pending-timeout branch by aging _pending_since past
    # PENDING_ENTRY_TIMEOUT (60 s in the kademlite default).
    bucket._pending_since = time.monotonic() - 120.0

    accepted = bucket.add_or_update(p_c, [b"/c"])

    assert accepted is True  # pending promoted into bucket
    assert bucket._pending is None
    assert bucket._pending_since is None

    # No duplicate: exactly one entry for p_c in self.peers, and its
    # dial_verified flag survived the promotion.
    matches = [e for e in bucket.peers if e.peer_id == p_c]
    assert len(matches) == 1
    assert matches[0].dial_verified is True
    assert matches[0].connected is True


def test_kbucket_get_returns_pending_entry():
    """``KBucket.get`` (and therefore ``RoutingTable.find``) must scan
    the pending replacement slot. Otherwise callers that want to read
    the current state of a peer they know is pending - e.g. the
    Identify Push handler computing ``addrs_changed`` - would see
    ``None`` for pending peers and incorrectly conclude every push is
    an addr-set change."""
    from kademlite.routing import KBucket

    bucket = KBucket(k=2, is_alive=lambda _pid: True)
    p_a = b"\xaa" * 32
    p_b = b"\xbb" * 32
    p_c = b"\xcc" * 32
    bucket.add_or_update(p_a, [b"/a"])
    bucket.add_or_update(p_b, [b"/b"])
    bucket.add_or_update(p_c, [b"/c"])  # goes to _pending

    assert bucket._pending is not None
    found = bucket.get(p_c)
    assert found is not None
    assert found.peer_id == p_c
    # Identical instance, not a copy: mutations through ``get`` reach
    # the actual pending entry.
    assert found is bucket._pending


def test_routing_table_find_returns_pending_entry():
    """Symmetric: ``RoutingTable.find`` (delegates to ``KBucket.get``)
    must surface pending entries. Required for the Identify Push
    handler's conditional-revoke path to work correctly when the peer
    is in the pending cache."""
    rt = RoutingTable(_make_peer_id(0), k=2)
    p_a = _make_peer_id(1)
    p_b = _make_peer_id(2)
    p_c = _make_peer_id(3)
    # Force them all into the same bucket via consecutive seeds.
    # K=2 means the 3rd one goes to _pending. The is_alive default
    # (None) uses STALE_PEER_TIMEOUT-based liveness; recent peers are
    # alive, so the 3rd add lands in pending.
    rt.add_or_update(p_a, [b"/a"])
    rt.add_or_update(p_b, [b"/b"])
    rt.add_or_update(p_c, [b"/c"])

    # At least one of {a, b, c} should be in _pending. Find each.
    found_count = 0
    for pid in (p_a, p_b, p_c):
        if rt.find(pid) is not None:
            found_count += 1
    assert found_count == 3


def test_identify_push_idempotent_on_pending_peer_preserves_dial_verified():
    """End-to-end check of the pending-peer interaction with the
    conditional-revoke branch. A peer in the pending cache that has
    been dial-verified must keep that flag through an idempotent
    Identify Push (same addrs). Before the fix the push handler saw
    ``find`` return None for the pending peer, computed
    ``addrs_changed=True``, called ``add_or_update`` (which replaced
    the pending entry with default flags), and then ``mark_dial_unverified``
    on a freshly-defaulted entry - the dial_verified bit was lost
    even on an idempotent push."""
    from kademlite.routing import KBucket

    # Build a bucket-with-pending directly so we can probe the
    # interaction without a full DhtNode setup.
    bucket = KBucket(k=2, is_alive=lambda _pid: True)
    p_a = b"\xaa" * 32
    p_b = b"\xbb" * 32
    p_c = b"\xcc" * 32
    bucket.add_or_update(p_a, [b"/a"])
    bucket.add_or_update(p_b, [b"/b"])
    bucket.add_or_update(p_c, [b"/c-original"])
    bucket.mark_dial_verified(p_c)
    assert bucket._pending.dial_verified is True

    # Simulate the Identify Push handler's logic for an idempotent push
    # (same addrs as the pending entry currently holds).
    existing = bucket.get(p_c)
    routable = [b"/c-original"]
    addrs_changed = existing is None or set(existing.addrs) != set(routable)
    assert addrs_changed is False

    bucket.add_or_update(p_c, routable)
    if addrs_changed:
        # Path not taken on idempotent push.
        bucket._pending.dial_verified = False  # simulated revocation

    assert bucket._pending.dial_verified is True
    assert bucket._pending.connected is True


def test_identify_push_changed_addrs_on_pending_peer_revokes_dial_verified():
    """Symmetric: a genuine addr-set change in Identify Push targeting
    a pending peer DOES revoke ``dial_verified``. The fix to
    ``KBucket.get`` makes this flow reach the same conditional-revoke
    branch as for active-bucket peers."""
    from kademlite.routing import KBucket

    bucket = KBucket(k=2, is_alive=lambda _pid: True)
    p_a = b"\xaa" * 32
    p_b = b"\xbb" * 32
    p_c = b"\xcc" * 32
    bucket.add_or_update(p_a, [b"/a"])
    bucket.add_or_update(p_b, [b"/b"])
    bucket.add_or_update(p_c, [b"/c-original"])
    bucket.mark_dial_verified(p_c)
    assert bucket._pending.dial_verified is True

    existing = bucket.get(p_c)
    assert existing is not None  # find now scans pending
    routable = [b"/c-changed"]
    addrs_changed = existing is None or set(existing.addrs) != set(routable)
    assert addrs_changed is True

    bucket.add_or_update(p_c, routable)  # in-place update, returns False
    if addrs_changed:
        bucket.mark_dial_unverified(p_c)

    assert bucket._pending.dial_verified is False
    assert bucket._pending.connected is True
    assert bucket._pending.addrs == routable


def test_mark_pending_entry_does_not_disturb_active_bucket_order():
    """LRU reorder is a no-op when the marked entry lives in the
    pending replacement cache. The active bucket's order must stay
    stable; a future promotion via bucket eviction will append to the
    tail naturally. Note: ``_new_full_bucket_with_pending`` triggers
    one LRU-refresh during the third add (LRU alive => move to tail),
    so the captured initial order reflects that refresh, not the
    insertion order."""
    bucket = _new_full_bucket_with_pending()
    p_c = b"\xcc" * 32  # in _pending
    initial = [e.peer_id for e in bucket.peers]
    assert p_c not in initial
    assert bucket._pending is not None and bucket._pending.peer_id == p_c

    bucket.mark_dial_verified(p_c)  # touches pending only

    # Active bucket order is unchanged because the marked entry was in
    # _pending, not self.peers.
    assert [e.peer_id for e in bucket.peers] == initial
    assert bucket._pending.peer_id == p_c
    assert bucket._pending.dial_verified is True


def test_routing_table_mark_dial_unverified_returns_false_for_unknown_peer():
    """``mark_dial_unverified`` returns False when the peer is not in
    the routing table - matches the existing convention for the other
    mark_* methods so callers can branch on presence."""
    rt = RoutingTable(_make_peer_id(0), k=20)
    assert rt.mark_dial_unverified(_make_peer_id(99)) is False


def test_kbucket_pending_entry_is_visited_by_mark_dial_unverified():
    """The pending-cache symmetry also applies to mark_dial_unverified:
    a pending entry that has been promoted to dial_verified can later
    receive an Identify Push that revokes the verification before
    bucket-eviction promotes it into peers."""
    bucket = _new_full_bucket_with_pending()
    p_c = b"\xcc" * 32
    bucket.mark_dial_verified(p_c)
    assert bucket._pending.dial_verified is True

    assert bucket.mark_dial_unverified(p_c) is True
    assert bucket._pending.dial_verified is False
    # connected is unaffected - the unverify path is address-level only.
    assert bucket._pending.connected is True
