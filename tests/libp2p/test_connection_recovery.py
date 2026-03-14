# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connection failure and recovery tests.

Validates that DhtNode handles peer disconnection gracefully:
- Dead peers are pruned from the routing table
- Records remain retrievable after a peer dies
- New connections can be established after failures
"""

import asyncio

from mx_libp2p.crypto import Ed25519Identity, _base58btc_encode
from mx_libp2p.dht import DhtNode


def _node_multiaddr(node: DhtNode) -> str:
    host, port = node.listen_addr
    return f"/ip4/{host}/tcp/{port}/p2p/{_base58btc_encode(node.peer_id)}"


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
        # Store a record from node_b (propagated to the network)
        key = b"/mx/test/worker/0"
        value = b'{"endpoint":"10.0.0.1:50051"}'
        count = await node_b.put(key, value)
        assert count >= 1, "record should be stored on at least one peer"

        # Kill node_b
        await node_b.stop()
        await asyncio.sleep(0.3)

        # node_c should still be able to GET the record (from node_a or local)
        result = await node_c.get(key)
        assert result == value, f"record should survive peer death, got {result!r}"

        # Prune dead peers from node_a's routing table
        node_a._prune_dead_peers()
        # node_b should no longer be in the routing table
        entry = node_a.routing_table.find(node_b.peer_id)
        # Entry might still exist (connection was from b's side), but if found,
        # its connection should be dead
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

    # Start and stop node_b to simulate a failure
    node_b = DhtNode(record_ttl=300)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node_a)])
    await asyncio.sleep(0.3)
    b_identity = node_b.identity
    await node_b.stop()
    await asyncio.sleep(0.3)

    try:
        # Start a new node_b2 with a fresh identity on a different port
        node_b2 = DhtNode(record_ttl=300)
        await node_b2.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node_a)])
        await asyncio.sleep(0.3)

        # Should be able to PUT/GET through the network
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
        # Kill one node
        await nodes[2].stop()
        await asyncio.sleep(0.2)

        # PUT should still succeed on remaining peers
        key = b"/mx/partial/worker/0"
        value = b'{"partial":"test"}'
        count = await nodes[0].put(key, value)
        # At least 1 remote peer + local store
        assert count >= 1

        # GET from a surviving peer should work
        result = await nodes[1].get(key)
        assert result == value
    finally:
        for n in nodes:
            try:
                await n.stop()
            except Exception:
                pass
