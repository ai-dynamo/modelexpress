# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DHT cluster tests: multi-node PUT/GET, bidirectional exchange, replication.

Covers:
- Two Python nodes: direct PUT/GET
- Three Python nodes in a line: multi-hop FIND_NODE + GET
- Five Python nodes: full DHT cluster PUT/GET with batch records
- Bidirectional PUT/GET between two nodes
- Record TTL expiry in a cluster
- PUT replication to K closest peers (XOR distance verification)
"""

import asyncio
import json
import logging
import time

from mx_libp2p.crypto import Ed25519Identity, _base58btc_encode
from mx_libp2p.dht import DhtNode
from mx_libp2p.routing import xor_distance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


async def make_node(
    bootstrap_peers: list[str] | None = None,
    record_ttl: float = 3600,
    republish_interval: float = 3600,
) -> DhtNode:
    """Create and start a DhtNode on localhost with a random port."""
    node = DhtNode(
        record_ttl=record_ttl,
        republish_interval=republish_interval,
    )
    await node.start("127.0.0.1", 0, bootstrap_peers=bootstrap_peers)
    return node


def node_multiaddr(node: DhtNode) -> str:
    """Get the multiaddr string for a running node."""
    host, port = node.listen_addr
    peer_id_b58 = _base58btc_encode(node.peer_id)
    return f"/ip4/{host}/tcp/{port}/p2p/{peer_id_b58}"


async def test_two_nodes_direct():
    """Test: Two nodes, direct PUT from A, GET from B."""
    log.info("=== test_two_nodes_direct ===")

    node_a = await make_node()
    addr_a = node_multiaddr(node_a)
    node_b = await make_node(bootstrap_peers=[addr_a])

    await asyncio.sleep(0.5)

    key = b"/test/hello"
    value = b"world"

    count = await node_a.put(key, value)
    log.info(f"node_a.put: stored on {count} peers")

    result = await node_b.get(key)
    assert result == value, f"expected {value!r}, got {result!r}"
    log.info("PASS: node_b.get returned correct value")

    await node_a.stop()
    await node_b.stop()


async def test_three_nodes_line():
    """Test: Three nodes A <-> B <-> C. A stores, C retrieves via B."""
    log.info("=== test_three_nodes_line ===")

    node_a = await make_node()
    addr_a = node_multiaddr(node_a)

    node_b = await make_node(bootstrap_peers=[addr_a])
    addr_b = node_multiaddr(node_b)

    node_c = await make_node(bootstrap_peers=[addr_b])

    await asyncio.sleep(0.5)

    key = b"/test/multihop"
    value = b"found-via-routing"

    node_a.kad_handler.put_local(key, value)
    await node_a.put(key, value)

    result = await node_c.get(key)
    assert result == value, f"expected {value!r}, got {result!r}"
    log.info("PASS: node_c.get returned correct value (multi-hop)")

    await node_a.stop()
    await node_b.stop()
    await node_c.stop()


async def test_five_node_cluster():
    """Test: Five nodes forming a mini-DHT. PUT from node 0, GET from node 4."""
    log.info("=== test_five_node_cluster ===")

    nodes: list[DhtNode] = []

    node_0 = await make_node()
    nodes.append(node_0)

    for i in range(1, 5):
        addr = node_multiaddr(nodes[i - 1])
        node = await make_node(bootstrap_peers=[addr])
        nodes.append(node)

    await asyncio.sleep(1.0)

    for i, node in enumerate(nodes):
        log.info(f"node_{i} routing table: {node.routing_table.size()} peers")

    key = b"/cluster/test"
    value = json.dumps({"message": "hello from cluster"}).encode()
    count = await nodes[0].put(key, value)
    log.info(f"node_0.put: stored on {count} peers")

    result = await nodes[4].get(key)
    assert result == value, f"expected {value!r}, got {result!r}"
    log.info("PASS: node_4.get returned correct value")

    for i in range(10):
        k = f"/batch/{i}".encode()
        v = f"value-{i}".encode()
        await nodes[i % 5].put(k, v)

    for i in range(10):
        k = f"/batch/{i}".encode()
        expected = f"value-{i}".encode()
        result = await nodes[(i + 3) % 5].get(k)
        assert result == expected, f"key {k!r}: expected {expected!r}, got {result!r}"

    log.info("PASS: 10 batch records stored and retrieved across cluster")

    for node in nodes:
        await node.stop()


async def test_record_ttl():
    """Test: Records expire after TTL."""
    log.info("=== test_record_ttl ===")

    node_a = await make_node(record_ttl=1.0, republish_interval=3600)
    addr_a = node_multiaddr(node_a)
    node_b = await make_node(
        bootstrap_peers=[addr_a], record_ttl=1.0, republish_interval=3600
    )

    await asyncio.sleep(0.5)

    key = b"/test/expiry"
    value = b"ephemeral"

    await node_a.put(key, value)

    result = await node_b.get(key)
    assert result == value, f"expected {value!r} immediately, got {result!r}"

    await asyncio.sleep(1.5)

    node_a.kad_handler.remove_expired(1.0)
    node_b.kad_handler.remove_expired(1.0)

    node_a._originated_records.clear()

    result = await node_b.get(key)
    assert result is None, f"expected None after TTL, got {result!r}"
    log.info("PASS: record expired after TTL")

    await node_a.stop()
    await node_b.stop()


async def test_bidirectional_put_get():
    """Test: Both nodes can PUT and GET to/from each other."""
    log.info("=== test_bidirectional_put_get ===")

    node_a = await make_node()
    addr_a = node_multiaddr(node_a)
    node_b = await make_node(bootstrap_peers=[addr_a])

    await asyncio.sleep(0.5)

    await node_a.put(b"/from/a", b"hello-from-a")
    result = await node_b.get(b"/from/a")
    assert result == b"hello-from-a", f"A->B failed: {result!r}"

    await node_b.put(b"/from/b", b"hello-from-b")
    result = await node_a.get(b"/from/b")
    assert result == b"hello-from-b", f"B->A failed: {result!r}"

    log.info("PASS: bidirectional PUT/GET works")

    await node_a.stop()
    await node_b.stop()


async def test_put_stores_on_closest_peers():
    """PUT should store the record on the K closest peers to the key."""
    nodes = []
    node_a = DhtNode(record_ttl=300)
    await node_a.start("127.0.0.1", 0)
    nodes.append(node_a)

    for _ in range(4):
        n = DhtNode(record_ttl=300)
        await n.start("127.0.0.1", 0, bootstrap_peers=[node_multiaddr(node_a)])
        nodes.append(n)

    await asyncio.sleep(0.5)

    try:
        key = b"/mx/repl-test/worker/0"
        value = b'{"replication":"test"}'
        count = await node_a.put(key, value)

        assert count >= 1, f"PUT only reached {count} peers"

        nodes_with_record = []
        for i, n in enumerate(nodes):
            local = n.kad_handler.get_local(key)
            if local is not None:
                nodes_with_record.append(i)

        assert 0 in nodes_with_record, "originator should have the record"
        assert len(nodes_with_record) >= 2, (
            f"record should be on at least 2 nodes, "
            f"but only on nodes {nodes_with_record}"
        )

        distances = []
        for i, n in enumerate(nodes):
            d = xor_distance(n.peer_id, key)
            distances.append((d, i))
        distances.sort()

        closest_indices = {idx for _, idx in distances[:count + 1]}
        record_indices = set(nodes_with_record)
        overlap = closest_indices & record_indices
        assert len(overlap) >= 1, (
            f"records should be on closest peers. "
            f"Closest: {closest_indices}, have record: {record_indices}"
        )
    finally:
        for n in nodes:
            await n.stop()
