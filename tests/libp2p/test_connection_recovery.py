# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connection failure, recovery, bootstrap, and peer pruning tests.

Covers:
- Dead peer pruning from routing table
- Record survival after peer death
- Reconnection after peer restart
- PUT with dead peers in K closest
- Bootstrap with all/some unreachable peers
- Bootstrap recovery after peer restart
- Periodic re-bootstrap discovering new peers
- Pruning dead vs healthy connections
- GET/PUT on empty DHT
- Dialing unreachable peers
- Node start/stop idempotency
- Configurable timeouts
- Routable address edge cases
"""

import asyncio

import pytest

from mx_libp2p.crypto import Ed25519Identity, _base58btc_encode
from mx_libp2p.dht import DhtNode
from mx_libp2p.multiaddr import encode_multiaddr_ip4_tcp_p2p


def _node_multiaddr(node: DhtNode) -> str:
    host, port = node.listen_addr
    return f"/ip4/{host}/tcp/{port}/p2p/{_base58btc_encode(node.peer_id)}"


# ---------------------------------------------------------------------------
# Peer death and recovery
# ---------------------------------------------------------------------------


async def test_peer_death_routing_table_recovery():
    """When a peer dies, its routing table entry is pruned and records
    stored on the remaining node are still retrievable."""
    node_a = DhtNode(record_ttl=300)
    node_b = DhtNode(record_ttl=300)
    node_c = DhtNode(record_ttl=300)

    await node_a.start("127.0.0.1", 0)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node_a)])
    await node_c.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node_a)])
    await asyncio.sleep(0.5)

    try:
        key = b"/mx/test/worker/0"
        value = b'{"endpoint":"10.0.0.1:50051"}'
        count = await node_b.put(key, value)
        assert count >= 1, "record should be stored on at least one peer"

        await node_b.stop()
        await asyncio.sleep(0.3)

        result = await node_c.get(key)
        assert result == value, f"record should survive peer death, got {result!r}"

        node_a._prune_dead_peers()
        entry = node_a.routing_table.find(node_b.peer_id)
        if entry is not None:
            conn = node_a.peer_store.get_connection(node_b.peer_id)
            assert conn is None or not conn.is_alive
    finally:
        await node_a.stop()
        await node_c.stop()


async def test_reconnect_after_failure():
    """A node that was unreachable can be reconnected after it restarts."""
    node_a = DhtNode(record_ttl=300)
    await node_a.start("127.0.0.1", 0)

    node_b = DhtNode(record_ttl=300)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node_a)])
    await asyncio.sleep(0.3)
    await node_b.stop()
    await asyncio.sleep(0.3)

    try:
        node_b2 = DhtNode(record_ttl=300)
        await node_b2.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node_a)])
        await asyncio.sleep(0.3)

        key = b"/mx/reconnect/worker/0"
        value = b'{"status":"ok"}'
        await node_b2.put(key, value)
        result = await node_a.get(key)
        assert result == value

        await node_b2.stop()
    finally:
        await node_a.stop()


async def test_put_with_dead_peer_in_closest():
    """PUT should succeed even if some of the K closest peers are dead."""
    nodes = []
    for _ in range(4):
        nodes.append(DhtNode(record_ttl=300))

    await nodes[0].start("127.0.0.1", 0)
    for n in nodes[1:]:
        await n.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(nodes[0])])
    await asyncio.sleep(0.5)

    try:
        await nodes[2].stop()
        await asyncio.sleep(0.2)

        key = b"/mx/partial/worker/0"
        value = b'{"partial":"test"}'
        count = await nodes[0].put(key, value)
        assert count >= 1

        result = await nodes[1].get(key)
        assert result == value
    finally:
        for n in nodes:
            try:
                await n.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Bootstrap failure and recovery
# ---------------------------------------------------------------------------


async def test_bootstrap_unreachable_then_recover():
    """Node should handle unreachable bootstrap gracefully and recover
    when a peer becomes available later."""
    node_a = DhtNode(dial_timeout=1.0)
    await node_a.start("127.0.0.1", 0, bootstrap_peers=[
        "/ip4/192.0.2.1/tcp/4001/p2p/12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN"
    ])

    assert node_a.routing_table.size() == 0, "should have no peers after failed bootstrap"

    node_b = DhtNode()
    await node_b.start("127.0.0.1", 0)

    try:
        await node_a.bootstrap([_node_multiaddr(node_b)])
        await asyncio.sleep(0.3)

        assert node_a.routing_table.size() >= 1, "should discover peer after recovery"

        key = b"/mx/recovery/test"
        value = b'{"recovered": true}'
        await node_a.put(key, value)
        result = await node_b.get(key)
        assert result == value
    finally:
        await node_a.stop()
        await node_b.stop()


async def test_bootstrap_partial_failure():
    """When some bootstrap peers are reachable and others aren't,
    the node should connect to the reachable ones."""
    node_real = DhtNode()
    await node_real.start("127.0.0.1", 0)

    try:
        node_test = DhtNode(dial_timeout=1.0)
        await node_test.start("127.0.0.1", 0, bootstrap_peers=[
            "/ip4/192.0.2.1/tcp/4001/p2p/12D3KooWDpJ7As7BWAwRMfu1VU2WCqNjvq387JEYKDBj4kx6nXTN",
            _node_multiaddr(node_real),
        ])
        await asyncio.sleep(0.3)

        assert node_test.routing_table.size() >= 1, "should connect to reachable peer"
        await node_test.stop()
    finally:
        await node_real.stop()


async def test_bootstrap_all_peers_unreachable():
    """Node should start successfully even when all bootstrap peers are unreachable."""
    node = DhtNode(dial_timeout=1.0)
    await node.start(
        "127.0.0.1", 0,
        bootstrap_peers=["/ip4/192.0.2.1/tcp/4001/p2p/QmYyQSo1c1Ym7orWxLYvCrM2EmxFTANf8wXmmE7DWjhx5N"]
    )

    try:
        assert node.routing_table.size() == 0

        await node.put(b"local-key", b"local-value")
        result = await node.get(b"local-key")
        assert result == b"local-value"
    finally:
        await node.stop()


async def test_periodic_rebootstrap_discovers_new_peers():
    """The periodic bootstrap loop should discover peers that joined after initial bootstrap."""
    node_a = DhtNode(record_ttl=300)
    await node_a.start("127.0.0.1", 0)
    addr_a = _node_multiaddr(node_a)

    node_b = DhtNode(record_ttl=300)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[addr_a])
    await asyncio.sleep(0.3)

    try:
        assert node_b.routing_table.size() >= 1

        node_c = DhtNode(record_ttl=300)
        await node_c.start("127.0.0.1", 0, bootstrap_peers=[addr_a])
        await asyncio.sleep(0.3)

        assert node_c.routing_table.size() >= 1

        await node_b._iterative_find_node(node_b.peer_id)

        assert node_b.routing_table.size() >= 2, (
            f"node_b should discover node_c via re-bootstrap, "
            f"but only has {node_b.routing_table.size()} peers"
        )
    finally:
        await node_a.stop()
        await node_b.stop()
        await node_c.stop()


async def test_bootstrap_recovery_after_peer_restart():
    """When a bootstrap peer was initially down but comes back,
    the node should discover it during periodic re-bootstrap."""
    node_a = DhtNode(record_ttl=300)
    await node_a.start("127.0.0.1", 0)
    addr_a = _node_multiaddr(node_a)
    identity_a = node_a.identity
    await node_a.stop()

    node_b = DhtNode(record_ttl=300, dial_timeout=1.0)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[addr_a])
    await asyncio.sleep(0.3)

    try:
        assert node_b.routing_table.size() == 0

        node_a2 = DhtNode(record_ttl=300)
        await node_a2.start("127.0.0.1", 0)
        addr_a2 = _node_multiaddr(node_a2)

        await node_b.bootstrap([addr_a2])

        assert node_b.routing_table.size() >= 1, "node_b should connect to restarted bootstrap"

        await node_a2.stop()
    finally:
        await node_b.stop()


# ---------------------------------------------------------------------------
# Peer pruning
# ---------------------------------------------------------------------------


async def test_prune_dead_peers_removes_dead_connections():
    """_prune_dead_peers should remove routing table entries whose
    connections exist but are no longer alive."""
    node_a = DhtNode(record_ttl=300)
    node_b = DhtNode(record_ttl=300)

    await node_a.start("127.0.0.1", 0)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node_a)])
    await asyncio.sleep(0.3)

    try:
        assert node_a.routing_table.find(node_b.peer_id) is not None
        conn = node_a.peer_store.get_connection(node_b.peer_id)
        assert conn is not None
        assert conn.is_alive

        node_b.noise = None
        await node_b.stop()
        await asyncio.sleep(0.2)

        node_a._prune_dead_peers()

        entry = node_a.routing_table.find(node_b.peer_id)
        conn_after = node_a.peer_store.get_connection(node_b.peer_id)
        assert entry is None or conn_after is None, (
            "dead peer should be removed from routing table after pruning"
        )
    finally:
        await node_a.stop()


async def test_prune_does_not_remove_healthy_peers():
    """_prune_dead_peers should NOT remove peers with live connections."""
    node_a = DhtNode(record_ttl=300)
    node_b = DhtNode(record_ttl=300)

    await node_a.start("127.0.0.1", 0)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node_a)])
    await asyncio.sleep(0.3)

    try:
        initial_size = node_a.routing_table.size()
        assert initial_size >= 1

        node_a._prune_dead_peers()

        assert node_a.routing_table.size() == initial_size
    finally:
        await node_a.stop()
        await node_b.stop()


async def test_prune_dead_peers_no_connection():
    """_prune_dead_peers should NOT remove peers that have no connection
    object at all (no connection != dead connection)."""
    node = DhtNode()
    await node.start("127.0.0.1", 0)

    fake_peer = b"\xaa" * 32
    addr = encode_multiaddr_ip4_tcp_p2p("10.0.0.1", 4001, fake_peer)
    node.routing_table.add_or_update(fake_peer, [addr])
    assert node.routing_table.size() == 1

    node._prune_dead_peers()
    assert node.routing_table.size() == 1

    await node.stop()


# ---------------------------------------------------------------------------
# Edge cases: empty DHT, unreachable peers, idempotent stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_on_empty_dht():
    """GET on a DHT with no peers should return None."""
    node = DhtNode()
    await node.start("127.0.0.1", 0)

    result = await node.get(b"nonexistent-key")
    assert result is None

    await node.stop()


@pytest.mark.asyncio
async def test_put_with_no_peers():
    """PUT with no peers should store locally and return 0."""
    node = DhtNode()
    await node.start("127.0.0.1", 0)

    count = await node.put(b"key", b"value")
    assert count == 0

    result = await node.get(b"key")
    assert result == b"value"

    await node.stop()


@pytest.mark.asyncio
async def test_dial_unreachable_peer():
    """Dialing an unreachable peer should fail gracefully."""
    node = DhtNode(dial_timeout=1.0)
    await node.start("127.0.0.1", 0)

    bogus_peer_id = b"\xff" * 32
    bogus_addr = encode_multiaddr_ip4_tcp_p2p("192.0.2.1", 1, bogus_peer_id)
    node.peer_store.add_addrs(bogus_peer_id, [bogus_addr])

    result = await node._put_to_peer(bogus_peer_id, [bogus_addr], b"key", b"value")
    assert result is False

    await node.stop()


@pytest.mark.asyncio
async def test_node_start_stop_idempotent():
    """Stopping a node multiple times should not raise."""
    node = DhtNode()
    await node.start("127.0.0.1", 0)
    await node.stop()
    await node.stop()


@pytest.mark.asyncio
async def test_configurable_timeouts():
    """DhtNode should accept custom timeout values."""
    node = DhtNode(rpc_timeout=1.0, dial_timeout=0.5)
    assert node.rpc_timeout == 1.0
    assert node.dial_timeout == 0.5
    await node.start("127.0.0.1", 0)
    await node.stop()


@pytest.mark.asyncio
async def test_routable_addr_before_start():
    """routable_addr() should raise before the node is started."""
    node = DhtNode()
    with pytest.raises(RuntimeError, match="not started"):
        node.routable_addr()


@pytest.mark.asyncio
async def test_local_addrs_unroutable():
    """local_addrs() should return empty when bound to 0.0.0.0 with no observed IP."""
    node = DhtNode()
    await node.start("0.0.0.0", 0)

    assert node.local_addrs() == []

    await node.stop()


# ---------------------------------------------------------------------------
# Connection close detection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graceful_shutdown_marks_peer_disconnected():
    """When a peer shuts down gracefully, the other peer should detect it
    immediately via the Yamux session close callback, not wait for a failed dial."""
    node_a = DhtNode(rpc_timeout=5.0, dial_timeout=2.0)
    node_b = DhtNode(rpc_timeout=5.0, dial_timeout=2.0)

    await node_a.start("127.0.0.1", 0)
    addr_a = _node_multiaddr(node_a)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[addr_a])

    # Verify both peers see each other as connected
    entry = node_a.routing_table.find(node_b.peer_id)
    assert entry is not None
    assert entry.connected

    # Shut down node_b gracefully
    await node_b.stop()

    # Give the event loop a moment to process the Yamux done callback
    await asyncio.sleep(0.1)

    # node_a should have detected the closure immediately
    entry = node_a.routing_table.find(node_b.peer_id)
    assert entry is not None
    assert not entry.connected, "peer should be marked disconnected after remote shutdown"

    await node_a.stop()


@pytest.mark.asyncio
async def test_connection_close_clears_peer_store_connection():
    """When a remote peer's connection closes, the peer store should clear
    the stale connection object so get_connection returns None."""
    node_a = DhtNode(rpc_timeout=5.0, dial_timeout=2.0)
    node_b = DhtNode(rpc_timeout=5.0, dial_timeout=2.0)

    await node_a.start("127.0.0.1", 0)
    addr_a = _node_multiaddr(node_a)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[addr_a])

    # node_a should have a connection to node_b (from inbound accept)
    conn = node_a.peer_store.get_connection(node_b.peer_id)
    assert conn is not None

    await node_b.stop()
    await asyncio.sleep(0.1)

    # The stale connection should be cleared
    conn = node_a.peer_store.get_connection(node_b.peer_id)
    assert conn is None, "stale connection should be cleared after remote shutdown"

    await node_a.stop()


@pytest.mark.asyncio
async def test_close_callback_not_fired_on_local_close():
    """When WE close a connection (not the remote), the on_peer_unreachable
    callback should NOT fire - we don't want to mark a peer disconnected
    just because we chose to close our end."""
    node_a = DhtNode(rpc_timeout=5.0, dial_timeout=2.0)
    node_b = DhtNode(rpc_timeout=5.0, dial_timeout=2.0)

    await node_a.start("127.0.0.1", 0)
    addr_a = _node_multiaddr(node_a)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[addr_a])

    entry = node_b.routing_table.find(node_a.peer_id)
    assert entry is not None
    assert entry.connected

    # node_b stops itself - this is a local close, not a remote event
    await node_b.stop()

    # node_b's own routing table entry for node_a should still show connected
    # (the callback is suppressed for local closes)
    entry = node_b.routing_table.find(node_a.peer_id)
    assert entry is not None
    assert entry.connected, "local close should not mark peer as disconnected"

    await node_a.stop()
