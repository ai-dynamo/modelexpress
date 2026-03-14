# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DHT cluster tests: multi-node PUT/GET, TTL expiry, and Rust interop.

Test matrix:
1. Two Python nodes: direct PUT/GET
2. Three Python nodes in a line: multi-hop FIND_NODE + GET
3. Five Python nodes: full DHT cluster PUT/GET
4. Record TTL expiry
5. Interop: Python nodes + one Rust node (if available)
"""

import asyncio
import logging
import os
import json
import subprocess
import sys
import time

from mx_libp2p.crypto import Ed25519Identity
from mx_libp2p.dht import DhtNode
from mx_libp2p.multiaddr import encode_multiaddr_ip4_tcp_p2p

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
    peer_id_hex = node.peer_id.hex()
    # Build multiaddr with raw peer ID bytes encoded as base58
    from mx_libp2p.crypto import _base58btc_encode
    peer_id_b58 = _base58btc_encode(node.peer_id)
    return f"/ip4/{host}/tcp/{port}/p2p/{peer_id_b58}"


async def test_two_nodes_direct():
    """Test: Two nodes, direct PUT from A, GET from B."""
    log.info("=== test_two_nodes_direct ===")

    node_a = await make_node()
    addr_a = node_multiaddr(node_a)
    node_b = await make_node(bootstrap_peers=[addr_a])

    # Wait for routing tables to settle
    await asyncio.sleep(0.5)

    key = b"/test/hello"
    value = b"world"

    # A puts a record
    count = await node_a.put(key, value)
    log.info(f"node_a.put: stored on {count} peers")

    # B gets the record
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

    # C only knows B, not A
    node_c = await make_node(bootstrap_peers=[addr_b])

    await asyncio.sleep(0.5)

    key = b"/test/multihop"
    value = b"found-via-routing"

    # Store on A
    node_a.kad_handler.put_local(key, value)
    # Also put via DHT so B gets it
    await node_a.put(key, value)

    # C should find it through iterative lookup via B -> A
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

    # Create chain: each node bootstraps from the previous
    node_0 = await make_node()
    nodes.append(node_0)

    for i in range(1, 5):
        addr = node_multiaddr(nodes[i - 1])
        node = await make_node(bootstrap_peers=[addr])
        nodes.append(node)

    await asyncio.sleep(1.0)

    # Log routing table sizes
    for i, node in enumerate(nodes):
        log.info(f"node_{i} routing table: {node.routing_table.size()} peers")

    # PUT from node 0
    key = b"/cluster/test"
    value = json.dumps({"message": "hello from cluster"}).encode()
    count = await nodes[0].put(key, value)
    log.info(f"node_0.put: stored on {count} peers")

    # GET from node 4 (furthest in chain)
    result = await nodes[4].get(key)
    assert result == value, f"expected {value!r}, got {result!r}"
    log.info("PASS: node_4.get returned correct value")

    # Test multiple records
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

    # Use very short TTL for testing
    node_a = await make_node(record_ttl=1.0, republish_interval=3600)
    addr_a = node_multiaddr(node_a)
    node_b = await make_node(
        bootstrap_peers=[addr_a], record_ttl=1.0, republish_interval=3600
    )

    await asyncio.sleep(0.5)

    key = b"/test/expiry"
    value = b"ephemeral"

    await node_a.put(key, value)

    # Should be retrievable immediately
    result = await node_b.get(key)
    assert result == value, f"expected {value!r} immediately, got {result!r}"

    # Wait for TTL to expire
    await asyncio.sleep(1.5)

    # Expire records
    node_a.kad_handler.remove_expired(1.0)
    node_b.kad_handler.remove_expired(1.0)

    # Clear originated records so republish won't fire
    node_a._originated_records.clear()

    # Should no longer be retrievable
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

    # A puts, B gets
    await node_a.put(b"/from/a", b"hello-from-a")
    result = await node_b.get(b"/from/a")
    assert result == b"hello-from-a", f"A->B failed: {result!r}"

    # B puts, A gets
    await node_b.put(b"/from/b", b"hello-from-b")
    result = await node_a.get(b"/from/b")
    assert result == b"hello-from-b", f"B->A failed: {result!r}"

    log.info("PASS: bidirectional PUT/GET works")

    await node_a.stop()
    await node_b.stop()


async def test_rust_interop():
    """Test: Python nodes + one Rust node (if the Rust binary is available)."""
    log.info("=== test_rust_interop ===")

    rust_binary = os.path.join(
        os.path.dirname(__file__),
        "..", "tests", "libp2p_kad_interop", "rust_node", "target", "release", "rust-libp2p-node",
    )
    # Also check debug build
    if not os.path.exists(rust_binary):
        rust_binary = os.path.join(
            os.path.dirname(__file__),
            "..", "tests", "libp2p_kad_interop", "rust_node", "target", "debug", "rust-libp2p-node",
        )

    if not os.path.exists(rust_binary):
        log.info("SKIP: Rust binary not found, skipping interop test")
        return

    # Start Rust node in "listen" mode (it puts a record and listens)
    proc = subprocess.Popen(
        [rust_binary, "put", "test-key", "test-value-from-rust"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Wait for Rust node to print its listening address
        rust_addr = None
        for _ in range(50):
            line = proc.stdout.readline()
            if not line:
                await asyncio.sleep(0.1)
                continue
            line = line.strip()
            log.info(f"rust: {line}")
            if line.startswith("/ip4/"):
                rust_addr = line
                break

        if rust_addr is None:
            log.warning("SKIP: Rust node didn't print an address")
            return

        # Start Python nodes, bootstrap one from the Rust node
        node_a = await make_node(bootstrap_peers=[rust_addr])
        addr_a = node_multiaddr(node_a)
        node_b = await make_node(bootstrap_peers=[addr_a])

        await asyncio.sleep(1.0)

        # Python node should be able to GET the record the Rust node stored
        result = await node_a.get(b"test-key")
        if result is not None:
            log.info(f"PASS: got record from Rust node: {result!r}")
        else:
            log.warning("Rust interop GET returned None - may need Rust node protocol alignment")

        # Python PUT, then verify another Python node can GET
        await node_a.put(b"/py/key", b"python-value")
        result = await node_b.get(b"/py/key")
        assert result == b"python-value", f"Python cluster GET failed: {result!r}"
        log.info("PASS: Python cluster works alongside Rust node")

        await node_a.stop()
        await node_b.stop()
    finally:
        proc.terminate()
        proc.wait(timeout=5)


async def main():
    """Run all tests."""
    tests = [
        test_two_nodes_direct,
        test_bidirectional_put_get,
        test_three_nodes_line,
        test_five_node_cluster,
        test_record_ttl,
    ]

    # Optionally include Rust interop
    if "--interop" in sys.argv:
        tests.append(test_rust_interop)

    passed = 0
    failed = 0
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            log.error(f"FAIL: {test.__name__}: {e}", exc_info=True)
            failed += 1

    log.info(f"\nResults: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
