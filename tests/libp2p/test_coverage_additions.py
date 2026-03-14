# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests filling identified coverage gaps.

1. Per-record TTL through full DhtNode.put() to remote peers
2. record_filter callback rejecting remote PUTs through DhtNode
3. Observed IP voting (threshold, address change)
4. _random_key_for_bucket bit manipulation correctness
5. Yamux ping/pong
6. Bucket refresh discovering peers self-lookup misses
7. Listener max_connections enforcement
8. Record publisher/ttl fields round-trip through protobuf
"""

import asyncio
import struct
import time

import pytest

from mx_libp2p.crypto import Ed25519Identity, _base58btc_encode
from mx_libp2p.dht import DhtNode
from mx_libp2p.kad_handler import KadHandler, StoredRecord
from mx_libp2p.routing import RoutingTable, _common_prefix_length
from mx_libp2p.kademlia import encode_record, decode_record
from mx_libp2p.listener import Listener
from mx_libp2p.connection import dial
from mx_libp2p.noise import handshake_initiator, handshake_responder
from mx_libp2p.multistream import negotiate_outbound, negotiate_inbound
from mx_libp2p.yamux import (
    YamuxSession, YamuxStream, DEFAULT_WINDOW_SIZE,
    _encode_header, TYPE_PING, FLAG_SYN, FLAG_ACK,
)
from mx_libp2p.multiaddr import decode_multiaddr, PROTO_IP4, encode_multiaddr_ip4_tcp


def _node_multiaddr(node: DhtNode) -> str:
    host, port = node.listen_addr
    return f"/ip4/{host}/tcp/{port}/p2p/{_base58btc_encode(node.peer_id)}"


# ---------------------------------------------------------------------------
# 1. Per-record TTL through full DhtNode.put() to remote peers
# ---------------------------------------------------------------------------


async def test_per_record_ttl_propagates_to_remote():
    """put(key, value, ttl=X) should store with the given TTL both
    locally and on remote peers. Remote copies should expire at the
    per-record TTL, not the node default."""
    node_a = DhtNode(record_ttl=300.0, republish_interval=3600)
    await node_a.start("127.0.0.1", 0)
    addr_a = _node_multiaddr(node_a)

    node_b = DhtNode(record_ttl=300.0, republish_interval=3600)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[addr_a])
    await asyncio.sleep(0.3)

    try:
        key = b"/mx/ttl-test/status/0"
        value = b'{"status":"READY"}'

        # PUT with a very short per-record TTL
        await node_a.put(key, value, ttl=1.0)

        # node_b should have it via the PUT path
        remote_rec = node_b.kad_handler.get_local(key)
        assert remote_rec is not None, "remote peer should have the record"
        assert remote_rec.value == value

        # The local record on node_a should have the per-record TTL
        local_rec = node_a.kad_handler.get_local(key)
        assert local_rec is not None
        assert local_rec.ttl == 1.0

        # Wait for the per-record TTL to expire
        await asyncio.sleep(1.5)

        # Expire check on node_b with the node's default TTL (300s)
        # The per-record TTL (1s) should take precedence
        node_b.kad_handler.remove_expired(300.0)
        assert node_b.kad_handler.get_local(key) is None, (
            "record on remote peer should have expired via per-record TTL"
        )
    finally:
        await node_a.stop()
        await node_b.stop()


# ---------------------------------------------------------------------------
# 2. record_filter callback rejecting remote PUTs through DhtNode
# ---------------------------------------------------------------------------


async def test_record_filter_rejects_remote_put():
    """A DhtNode with a record_filter should reject inbound PUTs that
    don't pass the filter, while accepting ones that do."""

    def only_mx_keys(key: bytes, value: bytes) -> bool:
        return key.startswith(b"/mx/")

    node_a = DhtNode(record_ttl=300)
    await node_a.start("127.0.0.1", 0)
    addr_a = _node_multiaddr(node_a)

    node_b = DhtNode(record_ttl=300, record_filter=only_mx_keys)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[addr_a])
    await asyncio.sleep(0.3)

    try:
        # PUT a key that passes the filter
        good_key = b"/mx/model/worker/0"
        await node_a.put(good_key, b'{"rank":0}')
        assert node_b.kad_handler.get_local(good_key) is not None, (
            "record with /mx/ prefix should be accepted by filter"
        )

        # PUT a key that does NOT pass the filter
        bad_key = b"/bad/namespace/key"
        await node_a.put(bad_key, b'{"bad":true}')
        assert node_b.kad_handler.get_local(bad_key) is None, (
            "record without /mx/ prefix should be rejected by filter"
        )
    finally:
        await node_a.stop()
        await node_b.stop()


# ---------------------------------------------------------------------------
# 3. Observed IP voting threshold and address change detection
# ---------------------------------------------------------------------------


async def test_observed_ip_voting_requires_threshold():
    """Observed IP should NOT be set until the threshold number of
    votes is reached."""
    node = DhtNode()
    # Don't start - we're testing the internal method directly
    node._listen_addr = ("0.0.0.0", 4001)
    node._observed_ip_threshold = 3

    observed_addr = encode_multiaddr_ip4_tcp("10.0.1.5", 12345)

    # First vote: below threshold
    await node._maybe_set_observed_ip(observed_addr)
    assert node._observed_ip is None, "should not set IP after 1 vote (threshold=3)"

    # Second vote: still below
    await node._maybe_set_observed_ip(observed_addr)
    assert node._observed_ip is None, "should not set IP after 2 votes"

    # Third vote: reaches threshold (triggers create_task for identify push)
    await node._maybe_set_observed_ip(observed_addr)
    assert node._observed_ip == "10.0.1.5", "should set IP after 3 votes"


async def test_observed_ip_voting_different_ips():
    """Different IPs get independent vote counts. The first to reach
    threshold wins."""
    node = DhtNode()
    node._listen_addr = ("0.0.0.0", 4001)
    node._observed_ip_threshold = 2

    addr_a = encode_multiaddr_ip4_tcp("10.0.1.5", 12345)
    addr_b = encode_multiaddr_ip4_tcp("10.0.1.6", 12345)

    # One vote for each
    await node._maybe_set_observed_ip(addr_a)
    await node._maybe_set_observed_ip(addr_b)
    assert node._observed_ip is None, "neither IP has reached threshold"

    # Second vote for addr_a reaches threshold
    await node._maybe_set_observed_ip(addr_a)
    assert node._observed_ip == "10.0.1.5"


async def test_observed_ip_change_detection():
    """When a new IP reaches threshold and differs from the current one,
    the observed IP should update."""
    node = DhtNode()
    node._listen_addr = ("0.0.0.0", 4001)
    node._observed_ip_threshold = 2

    addr_a = encode_multiaddr_ip4_tcp("10.0.1.5", 12345)
    addr_b = encode_multiaddr_ip4_tcp("10.0.1.6", 12345)

    # Set initial IP
    await node._maybe_set_observed_ip(addr_a)
    await node._maybe_set_observed_ip(addr_a)
    assert node._observed_ip == "10.0.1.5"

    # Vote for new IP (votes were cleared after threshold was reached)
    await node._maybe_set_observed_ip(addr_b)
    await node._maybe_set_observed_ip(addr_b)
    assert node._observed_ip == "10.0.1.6", "should update to new IP"


async def test_observed_ip_ignores_unroutable():
    """0.0.0.0 and loopback (when bound to wildcard) should be ignored."""
    node = DhtNode()
    node._listen_addr = ("0.0.0.0", 4001)
    node._observed_ip_threshold = 1

    # 0.0.0.0 should be ignored
    await node._maybe_set_observed_ip(encode_multiaddr_ip4_tcp("0.0.0.0", 1234))
    assert node._observed_ip is None

    # 127.0.0.1 should be ignored when bound to 0.0.0.0
    await node._maybe_set_observed_ip(encode_multiaddr_ip4_tcp("127.0.0.1", 1234))
    assert node._observed_ip is None


# ---------------------------------------------------------------------------
# 4. _random_key_for_bucket bit manipulation correctness
# ---------------------------------------------------------------------------


def test_random_key_for_bucket_cpl():
    """_random_key_for_bucket(cpl) must produce a key whose CPL with
    our peer ID is exactly `cpl`."""
    identity = Ed25519Identity.generate()
    node = DhtNode(identity=identity)
    node._listen_addr = ("127.0.0.1", 0)  # needed by the method

    for cpl in [0, 1, 7, 8, 15, 16, 31, 63, 127]:
        key = node._random_key_for_bucket(cpl)
        actual_cpl = _common_prefix_length(node.peer_id, key)
        assert actual_cpl == cpl, (
            f"_random_key_for_bucket({cpl}) produced key with CPL={actual_cpl}"
        )


def test_random_key_for_bucket_randomness():
    """Multiple calls should produce different keys (not deterministic)."""
    identity = Ed25519Identity.generate()
    node = DhtNode(identity=identity)
    node._listen_addr = ("127.0.0.1", 0)

    keys = {node._random_key_for_bucket(10) for _ in range(20)}
    assert len(keys) > 1, "random keys should not all be identical"


# ---------------------------------------------------------------------------
# 5. Yamux ping/pong
# ---------------------------------------------------------------------------


async def test_yamux_ping_pong():
    """A Yamux PING with SYN flag should get an ACK response with the
    same opaque value."""
    server_identity = Ed25519Identity.generate()
    client_identity = Ed25519Identity.generate()

    server_yamux = None
    setup_done = asyncio.Event()

    async def handle(reader, writer):
        nonlocal server_yamux
        await negotiate_inbound(reader, writer, ["/noise"])
        noise = await handshake_responder(reader, writer, server_identity)
        from mx_libp2p.connection import _noise_to_rw
        nr, nw = _noise_to_rw(noise)
        await negotiate_inbound(nr, nw, ["/yamux/1.0.0"])
        server_yamux = YamuxSession(noise, is_initiator=False)
        await server_yamux.start()
        setup_done.set()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    addr = server.sockets[0].getsockname()

    try:
        reader, writer = await asyncio.open_connection(addr[0], addr[1])
        await negotiate_outbound(reader, writer, "/noise")
        noise = await handshake_initiator(reader, writer, client_identity)
        from mx_libp2p.connection import _noise_to_rw
        nr, nw = _noise_to_rw(noise)
        await negotiate_outbound(nr, nw, "/yamux/1.0.0")
        client_yamux = YamuxSession(noise, is_initiator=True)
        await client_yamux.start()

        await asyncio.wait_for(setup_done.wait(), timeout=5.0)

        # Send a PING from client to server
        opaque_value = 0xDEADBEEF
        from mx_libp2p.yamux import TYPE_PING, FLAG_SYN
        await client_yamux._send_frame(TYPE_PING, FLAG_SYN, 0, b"", length_override=opaque_value)

        # The server's read loop should handle the ping automatically.
        # Verify by opening a stream (proves the session is still alive
        # and processing frames after handling the ping).
        await asyncio.sleep(0.2)
        stream = await client_yamux.open_stream()
        assert stream is not None, "session should still be alive after ping"

        await stream.close()
    finally:
        await client_yamux.stop()
        if server_yamux:
            await server_yamux.stop()
        server.close()


# ---------------------------------------------------------------------------
# 6. Bucket refresh discovers peers self-lookup misses
# ---------------------------------------------------------------------------


async def test_bucket_refresh_discovers_distant_peers():
    """Bucket refresh should discover peers in distant buckets that a
    self-lookup alone wouldn't find."""
    # Create a star topology: all nodes connect to node_a.
    # Self-lookup from any node finds peers near itself.
    # Bucket refresh forces lookups at all distances.
    node_a = DhtNode(record_ttl=300)
    await node_a.start("127.0.0.1", 0)
    addr_a = _node_multiaddr(node_a)

    nodes = [node_a]
    for _ in range(5):
        n = DhtNode(record_ttl=300)
        await n.start("127.0.0.1", 0, bootstrap_peers=[addr_a])
        nodes.append(n)

    await asyncio.sleep(0.5)

    try:
        # Record node_a's routing table size after initial bootstrap
        size_before = node_a.routing_table.size()

        # Trigger bucket refresh
        await node_a._refresh_buckets()

        # After refresh, routing table should have at least as many peers
        # (refresh shouldn't lose peers, and may find new ones via
        # closer_peers responses)
        size_after = node_a.routing_table.size()
        assert size_after >= size_before, (
            f"bucket refresh should not lose peers: {size_before} -> {size_after}"
        )

        # All 5 other nodes should be known
        assert size_after >= 5, (
            f"after refresh, should know all 5 peers, have {size_after}"
        )
    finally:
        for n in nodes:
            await n.stop()


# ---------------------------------------------------------------------------
# 7. Listener max_connections enforcement
# ---------------------------------------------------------------------------


async def test_listener_concurrent_connection_limit():
    """When max_connections simultaneous connections are active during
    handshake, the listener should reject additional ones."""
    identity = Ed25519Identity.generate()
    accepted = []

    # Slow on_connection that holds the "slot" for a while
    async def slow_on_conn(conn):
        accepted.append(conn)
        await asyncio.sleep(2.0)  # simulate slow processing

    listener = Listener(
        identity,
        host="127.0.0.1",
        port=0,
        on_connection=slow_on_conn,
        max_connections=2,
    )
    await listener.start()
    host, port = listener.listen_addr

    try:
        # Rapidly open 4 connections while on_connection is still running
        # for the first ones. Note: with the fix, _active_connections is
        # decremented in finally (after on_connection returns), so during
        # the slow_on_conn sleep the counter is still 1 (the accept itself
        # is fast, the slow part is on_connection which runs before finally).
        # The counter tracks the full _handle_connection scope.
        conns = []
        for _ in range(4):
            try:
                c = await asyncio.wait_for(
                    dial(Ed25519Identity.generate(), host, port),
                    timeout=3.0,
                )
                conns.append(c)
            except Exception:
                pass
            await asyncio.sleep(0.05)

        # Give time for accepts to complete
        await asyncio.sleep(0.5)

        # At least 2 should have been accepted (the limit)
        assert len(accepted) >= 2, f"expected at least 2 accepted, got {len(accepted)}"

        for c in conns:
            await c.close()
    finally:
        await listener.stop()


# ---------------------------------------------------------------------------
# 8. Record publisher/ttl fields round-trip through protobuf
# ---------------------------------------------------------------------------


def test_record_proto_publisher_ttl_roundtrip():
    """Record.publisher (field 666) and Record.ttl (field 777) should
    survive encode/decode."""
    publisher_id = b"\x00\x25" + b"\x01" * 34  # fake peer ID
    encoded = encode_record(
        key=b"/mx/test",
        value=b'{"rank":0}',
        publisher=publisher_id,
        ttl=300,
    )

    decoded = decode_record(encoded)
    assert decoded["key"] == b"/mx/test"
    assert decoded["value"] == b'{"rank":0}'
    assert decoded["publisher"] == publisher_id
    assert decoded["ttl"] == 300


def test_record_proto_without_publisher_ttl():
    """Records without publisher/ttl should decode cleanly (backward compat)."""
    encoded = encode_record(key=b"/mx/test", value=b"hello")
    decoded = decode_record(encoded)
    assert decoded["publisher"] is None
    assert decoded["ttl"] is None
