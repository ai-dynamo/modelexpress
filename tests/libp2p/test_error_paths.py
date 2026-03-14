# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for error paths and edge cases in the DHT."""

import asyncio
import pytest

from mx_libp2p.crypto import Ed25519Identity
from mx_libp2p.dht import DhtNode


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

    # But the value should be stored locally
    result = await node.get(b"key")
    assert result == b"value"

    await node.stop()


@pytest.mark.asyncio
async def test_dial_unreachable_peer():
    """Dialing an unreachable peer should fail gracefully."""
    node = DhtNode(dial_timeout=1.0)
    await node.start("127.0.0.1", 0)

    # Try to put_to_peer with a bogus address
    from mx_libp2p.multiaddr import encode_multiaddr_ip4_tcp_p2p
    bogus_peer_id = b"\xff" * 32
    bogus_addr = encode_multiaddr_ip4_tcp_p2p("192.0.2.1", 1, bogus_peer_id)
    node.peer_store.add_addrs(bogus_peer_id, [bogus_addr])

    result = await node._put_to_peer(bogus_peer_id, [bogus_addr], b"key", b"value")
    assert result is False

    await node.stop()


@pytest.mark.asyncio
async def test_record_ttl_expiry():
    """Records should expire after TTL."""
    node = DhtNode(record_ttl=0.1)
    await node.start("127.0.0.1", 0)

    await node.put(b"key", b"value")

    # Should be available immediately
    assert await node.get(b"key") == b"value"

    # Wait for TTL to expire
    await asyncio.sleep(0.2)

    # Local record should be expired (no peers to find it remotely)
    result = await node.get(b"key")
    assert result is None

    await node.stop()


@pytest.mark.asyncio
async def test_node_start_stop_idempotent():
    """Stopping a node multiple times should not raise."""
    node = DhtNode()
    await node.start("127.0.0.1", 0)
    await node.stop()
    await node.stop()  # Should not raise


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

    # No observed IP yet, bound to wildcard -> no routable addrs
    assert node.local_addrs() == []

    await node.stop()


@pytest.mark.asyncio
async def test_prune_dead_peers():
    """_prune_dead_peers should remove entries with dead connections."""
    node = DhtNode()
    await node.start("127.0.0.1", 0)

    # Manually add a peer to routing table (no actual connection)
    fake_peer = b"\xaa" * 32
    from mx_libp2p.multiaddr import encode_multiaddr_ip4_tcp_p2p
    addr = encode_multiaddr_ip4_tcp_p2p("10.0.0.1", 4001, fake_peer)
    node.routing_table.add_or_update(fake_peer, [addr])
    assert node.routing_table.size() == 1

    # Prune - peer has no connection, should NOT be removed
    # (get_connection returns None for unknown peers, which means "no connection",
    # not "dead connection" - we only remove peers whose connection object exists
    # but is_alive is False)
    node._prune_dead_peers()
    # Peer remains because get_connection returns None (no connection object at all)
    assert node.routing_table.size() == 1

    await node.stop()
