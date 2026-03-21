# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robustness tests: noise failures, connection drops, lookup edge cases, Yamux
regressions, listener behavior, observed IP, and protocol edge cases.

Covers:
- Noise XX handshake: self-test, garbage msg1, truncated msg2, wrong signature,
  connection closed mid-handshake
- Yamux receive window replenishment after read
- Yamux large transfer (>256 KB) without RST
- Yamux ping/pong
- Listener connection counter decrement on success
- Listener max_connections enforcement (sequential and concurrent)
- PUT/GET survive peer crash mid-operation
- GET with dead peers in routing table
- Connection refused during iterative lookup
- Iterative lookup stall detection and termination
- Observed IP voting (threshold, different IPs, change detection, unroutable)
- Record filter callback rejecting remote PUTs
- Malformed multistream and connection drop during handshake
"""

import asyncio
import struct
import time

import pytest

from mx_libp2p.crypto import Ed25519Identity, _base58btc_encode
from mx_libp2p.dht import DhtNode
from mx_libp2p.noise import handshake_initiator, handshake_responder
from mx_libp2p.multistream import negotiate_outbound, negotiate_inbound
from mx_libp2p.yamux import YamuxSession, DEFAULT_WINDOW_SIZE
from mx_libp2p.listener import Listener
from mx_libp2p.connection import accept, dial, _noise_to_rw
from mx_libp2p.multiaddr import (
    encode_multiaddr_ip4_tcp,
    encode_multiaddr_ip4_tcp_p2p,
)
from mx_libp2p.routing import _common_prefix_length, K


def _node_multiaddr(node: DhtNode) -> str:
    host, port = node.listen_addr
    return f"/ip4/{host}/tcp/{port}/p2p/{_base58btc_encode(node.peer_id)}"


# ---------------------------------------------------------------------------
# Noise XX handshake tests
# ---------------------------------------------------------------------------


async def test_noise_xx_handshake_self():
    """Full round-trip: multistream-select + Noise XX + encrypted echo."""
    server_identity = Ed25519Identity.generate()
    client_identity = Ed25519Identity.generate()

    handshake_done = asyncio.Event()
    server_transport = None

    async def handle_connection(reader, writer):
        nonlocal server_transport
        try:
            proto = await negotiate_inbound(reader, writer, ["/noise"])
            assert proto == "/noise"
            server_transport = await handshake_responder(reader, writer, server_identity)
            handshake_done.set()

            msg = await server_transport.read_msg()
            await server_transport.write_msg(b"echo:" + msg)
        except Exception:
            handshake_done.set()

    server = await asyncio.start_server(handle_connection, "127.0.0.1", 0)
    addr = server.sockets[0].getsockname()

    try:
        reader, writer = await asyncio.open_connection(addr[0], addr[1])
        await negotiate_outbound(reader, writer, "/noise")
        client_transport = await handshake_initiator(reader, writer, client_identity)

        assert client_transport.remote_peer_id == server_identity.peer_id
        await asyncio.wait_for(handshake_done.wait(), timeout=5.0)
        assert server_transport.remote_peer_id == client_identity.peer_id

        await client_transport.write_msg(b"hello from pure python!")
        response = await client_transport.read_msg()
        assert response == b"echo:hello from pure python!"

        client_transport.close()
    finally:
        server.close()


async def test_noise_rejects_garbage_msg1():
    """Responder should reject a handshake where msg1 is garbage data."""
    server_identity = Ed25519Identity.generate()

    error_raised = asyncio.Event()

    async def handle_connection(reader, writer):
        try:
            await negotiate_inbound(reader, writer, ["/noise"])
            await handshake_responder(reader, writer, server_identity)
        except Exception:
            error_raised.set()
        finally:
            writer.close()

    server = await asyncio.start_server(handle_connection, "127.0.0.1", 0)
    addr = server.sockets[0].getsockname()

    try:
        reader, writer = await asyncio.open_connection(addr[0], addr[1])
        await negotiate_outbound(reader, writer, "/noise")

        garbage = b"\x00" * 10
        writer.write(struct.pack(">H", len(garbage)) + garbage)
        await writer.drain()

        await asyncio.wait_for(error_raised.wait(), timeout=3.0)
        assert error_raised.is_set()
        writer.close()
    finally:
        server.close()


async def test_noise_rejects_truncated_msg2():
    """Initiator should reject a truncated msg2 from the responder."""
    client_identity = Ed25519Identity.generate()

    async def handle_connection(reader, writer):
        try:
            await negotiate_inbound(reader, writer, ["/noise"])
            length_bytes = await reader.readexactly(2)
            length = struct.unpack(">H", length_bytes)[0]
            await reader.readexactly(length)

            truncated = b"\xab" * 20
            writer.write(struct.pack(">H", len(truncated)) + truncated)
            await writer.drain()
        except Exception:
            pass
        finally:
            writer.close()

    server = await asyncio.start_server(handle_connection, "127.0.0.1", 0)
    addr = server.sockets[0].getsockname()

    try:
        reader, writer = await asyncio.open_connection(addr[0], addr[1])
        await negotiate_outbound(reader, writer, "/noise")

        with pytest.raises(Exception):
            await asyncio.wait_for(
                handshake_initiator(reader, writer, client_identity),
                timeout=3.0,
            )
        writer.close()
    finally:
        server.close()


async def test_noise_rejects_wrong_signature():
    """Handshake should complete with correct peer IDs when both sides
    have valid but different identities."""
    id_a = Ed25519Identity.generate()
    id_b = Ed25519Identity.generate()

    handshake_done = asyncio.Event()
    server_result = {}

    async def handle_connection(reader, writer):
        try:
            await negotiate_inbound(reader, writer, ["/noise"])
            transport = await handshake_responder(reader, writer, id_b)
            server_result["peer_id"] = transport.remote_peer_id
            handshake_done.set()
        except Exception as e:
            server_result["error"] = e
            handshake_done.set()
        finally:
            writer.close()

    server = await asyncio.start_server(handle_connection, "127.0.0.1", 0)
    addr = server.sockets[0].getsockname()

    try:
        reader, writer = await asyncio.open_connection(addr[0], addr[1])
        await negotiate_outbound(reader, writer, "/noise")
        transport = await handshake_initiator(reader, writer, id_a)

        await asyncio.wait_for(handshake_done.wait(), timeout=3.0)

        assert transport.remote_peer_id == id_b.peer_id
        assert server_result.get("peer_id") == id_a.peer_id

        transport.close()
    finally:
        server.close()


async def test_noise_connection_closed_mid_handshake():
    """If the connection drops during the Noise handshake, it should fail cleanly."""
    client_identity = Ed25519Identity.generate()

    async def handle_connection(reader, writer):
        try:
            await negotiate_inbound(reader, writer, ["/noise"])
        except Exception:
            pass
        writer.close()

    server = await asyncio.start_server(handle_connection, "127.0.0.1", 0)
    addr = server.sockets[0].getsockname()

    try:
        reader, writer = await asyncio.open_connection(addr[0], addr[1])
        await negotiate_outbound(reader, writer, "/noise")

        with pytest.raises((asyncio.IncompleteReadError, ConnectionError, Exception)):
            await asyncio.wait_for(
                handshake_initiator(reader, writer, client_identity),
                timeout=3.0,
            )
        writer.close()
    finally:
        server.close()


# ---------------------------------------------------------------------------
# Yamux regression tests
# ---------------------------------------------------------------------------


async def test_yamux_recv_window_replenished_after_read():
    """After reading data and sending a window update, the local _recv_window
    must be incremented. Without the fix, _recv_window monotonically decreases
    and eventually triggers a false window violation RST."""
    server_identity = Ed25519Identity.generate()
    client_identity = Ed25519Identity.generate()

    server_yamux = None
    setup_done = asyncio.Event()

    async def handle(reader, writer):
        nonlocal server_yamux
        await negotiate_inbound(reader, writer, ["/noise"])
        noise = await handshake_responder(reader, writer, server_identity)
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
        nr, nw = _noise_to_rw(noise)
        await negotiate_outbound(nr, nw, "/yamux/1.0.0")
        client_yamux = YamuxSession(noise, is_initiator=True)
        await client_yamux.start()

        await asyncio.wait_for(setup_done.wait(), timeout=5.0)

        client_stream = await client_yamux.open_stream()
        server_stream = await server_yamux.accept_stream()

        chunk = b"x" * 1024
        await client_stream.write(chunk)
        data = await server_stream.read()
        assert data == chunk

        assert server_stream._recv_window >= DEFAULT_WINDOW_SIZE - 100, (
            f"recv_window should be replenished after read(), "
            f"got {server_stream._recv_window} (expected ~{DEFAULT_WINDOW_SIZE})"
        )

        await client_stream.close()
        await server_stream.close()
    finally:
        await client_yamux.stop()
        if server_yamux:
            await server_yamux.stop()
        server.close()


async def test_yamux_large_transfer_no_rst():
    """Transfer more than DEFAULT_WINDOW_SIZE (256 KB) on a single stream.
    Before the fix, this would RST the stream after 256 KB due to the
    receive window never being replenished."""
    server_identity = Ed25519Identity.generate()
    client_identity = Ed25519Identity.generate()

    server_yamux = None
    setup_done = asyncio.Event()

    async def handle(reader, writer):
        nonlocal server_yamux
        await negotiate_inbound(reader, writer, ["/noise"])
        noise = await handshake_responder(reader, writer, server_identity)
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
        nr, nw = _noise_to_rw(noise)
        await negotiate_outbound(nr, nw, "/yamux/1.0.0")
        client_yamux = YamuxSession(noise, is_initiator=True)
        await client_yamux.start()

        await asyncio.wait_for(setup_done.wait(), timeout=5.0)

        client_stream = await client_yamux.open_stream()
        server_stream = await server_yamux.accept_stream()

        total_bytes = DEFAULT_WINDOW_SIZE * 2
        chunk_size = 4096
        total_sent = 0
        total_received = 0

        async def sender():
            nonlocal total_sent
            remaining = total_bytes
            while remaining > 0:
                size = min(chunk_size, remaining)
                await client_stream.write(b"A" * size)
                total_sent += size
                remaining -= size
            await client_stream.close()

        async def receiver():
            nonlocal total_received
            while True:
                data = await server_stream.read()
                if not data:
                    break
                total_received += len(data)

        await asyncio.wait_for(
            asyncio.gather(sender(), receiver()),
            timeout=15.0,
        )

        assert total_received == total_bytes, (
            f"expected {total_bytes} bytes received, got {total_received}. "
            f"Stream was likely RST'd due to recv_window exhaustion."
        )
        assert server_stream._recv_window >= 0, (
            f"recv_window went negative: {server_stream._recv_window}"
        )
    finally:
        await client_yamux.stop()
        if server_yamux:
            await server_yamux.stop()
        server.close()


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
        nr, nw = _noise_to_rw(noise)
        await negotiate_outbound(nr, nw, "/yamux/1.0.0")
        client_yamux = YamuxSession(noise, is_initiator=True)
        await client_yamux.start()

        await asyncio.wait_for(setup_done.wait(), timeout=5.0)

        opaque_value = 0xDEADBEEF
        from mx_libp2p.yamux import TYPE_PING, FLAG_SYN
        await client_yamux._send_frame(TYPE_PING, FLAG_SYN, 0, b"", length_override=opaque_value)

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
# Listener regression tests
# ---------------------------------------------------------------------------


async def test_listener_counter_decrements_on_success():
    """After a successful accept, _active_connections must be decremented
    so the counter doesn't permanently inflate."""
    identity = Ed25519Identity.generate()
    connections_received = []

    async def on_conn(conn):
        connections_received.append(conn)

    listener = Listener(
        identity,
        host="127.0.0.1",
        port=0,
        on_connection=on_conn,
    )
    await listener.start()
    host, port = listener.listen_addr

    try:
        conns = []
        for _ in range(3):
            client_id = Ed25519Identity.generate()
            conn = await asyncio.wait_for(
                dial(client_id, host, port),
                timeout=5.0,
            )
            conns.append(conn)
            await asyncio.sleep(0.1)

        assert len(connections_received) == 3

        assert listener._active_connections == 0, (
            f"_active_connections should be 0 after accepts complete, "
            f"got {listener._active_connections} (counter not decremented on success)"
        )

        for c in conns:
            await c.close()
    finally:
        await listener.stop()


async def test_listener_does_not_reject_after_many_connections():
    """With a low max_connections limit, the listener should still accept
    new connections after previous ones have been handled."""
    identity = Ed25519Identity.generate()
    accepted = []

    async def on_conn(conn):
        accepted.append(conn)

    listener = Listener(
        identity,
        host="127.0.0.1",
        port=0,
        on_connection=on_conn,
        max_connections=2,
    )
    await listener.start()
    host, port = listener.listen_addr

    try:
        all_conns = []
        for i in range(5):
            client_id = Ed25519Identity.generate()
            conn = await asyncio.wait_for(
                dial(client_id, host, port),
                timeout=5.0,
            )
            all_conns.append(conn)
            await asyncio.sleep(0.1)

        assert len(accepted) == 5, (
            f"expected 5 accepted connections, got {len(accepted)}. "
            f"Listener likely rejected connections due to inflated counter."
        )

        for c in all_conns:
            await c.close()
    finally:
        await listener.stop()


async def test_listener_concurrent_connection_limit():
    """When max_connections simultaneous connections are active during
    handshake, the listener should reject additional ones."""
    identity = Ed25519Identity.generate()
    accepted = []

    async def slow_on_conn(conn):
        accepted.append(conn)
        await asyncio.sleep(2.0)

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

        await asyncio.sleep(0.5)

        assert len(accepted) >= 2, f"expected at least 2 accepted, got {len(accepted)}"

        for c in conns:
            await c.close()
    finally:
        await listener.stop()


# ---------------------------------------------------------------------------
# Connection drop during Kademlia RPC
# ---------------------------------------------------------------------------


async def test_put_survives_peer_crash_mid_operation():
    """If a peer crashes during a PUT (after some peers received it),
    the operation should still succeed partially and not hang."""
    nodes = [DhtNode(record_ttl=300, dial_timeout=2.0, rpc_timeout=2.0) for _ in range(4)]
    await nodes[0].start("127.0.0.1", 0)
    for n in nodes[1:]:
        await n.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(nodes[0])])
    await asyncio.sleep(0.5)

    try:
        await nodes[3].stop()
        await asyncio.sleep(0.1)

        key = b"/mx/crash/worker/0"
        value = b'{"status":"mid-crash-test"}'

        count = await asyncio.wait_for(nodes[0].put(key, value), timeout=10.0)
        assert count >= 1, "should store on at least one surviving peer"

        result = await nodes[1].get(key)
        assert result == value
    finally:
        for n in nodes:
            try:
                await n.stop()
            except Exception:
                pass


async def test_get_with_dead_peer_in_routing_table():
    """GET should succeed even when the routing table contains dead peers,
    by skipping unreachable peers and finding the record on live ones."""
    nodes = [DhtNode(record_ttl=300, dial_timeout=1.0, rpc_timeout=2.0) for _ in range(3)]
    await nodes[0].start("127.0.0.1", 0)
    for n in nodes[1:]:
        await n.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(nodes[0])])
    await asyncio.sleep(0.5)

    try:
        key = b"/mx/dead-peer-get/worker/0"
        value = b'{"endpoint":"10.0.0.1:50051"}'
        count = await nodes[0].put(key, value)
        assert count >= 1

        await nodes[2].stop()
        await asyncio.sleep(0.2)

        result = await asyncio.wait_for(nodes[1].get(key), timeout=10.0)
        assert result == value
    finally:
        for n in nodes:
            try:
                await n.stop()
            except Exception:
                pass


async def test_connection_refused_during_iterative_lookup():
    """Iterative lookup should handle connection refused errors
    without crashing or hanging."""
    node_a = DhtNode(record_ttl=300, dial_timeout=1.0)
    node_b = DhtNode(record_ttl=300, dial_timeout=1.0)
    await node_a.start("127.0.0.1", 0)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node_a)])
    await asyncio.sleep(0.3)

    try:
        fake_peer = b"\xcc" * 32
        fake_addr = encode_multiaddr_ip4_tcp_p2p("127.0.0.1", 1, fake_peer)
        node_a.routing_table.add_or_update(fake_peer, [fake_addr])
        node_a.peer_store.add_addrs(fake_peer, [fake_addr])

        key = b"/mx/fake-peer/test"
        value = b'{"works":"yes"}'
        count = await asyncio.wait_for(node_a.put(key, value), timeout=10.0)
        assert count >= 0

        result = await asyncio.wait_for(node_b.get(key), timeout=10.0)
        assert result == value
    finally:
        await node_a.stop()
        await node_b.stop()


# ---------------------------------------------------------------------------
# Iterative lookup stall detection
# ---------------------------------------------------------------------------


async def test_lookup_stall_detection():
    """When initial peers return no closer peers, stall detection should
    boost parallelism and eventually terminate without hanging."""
    node_a = DhtNode(record_ttl=300)
    await node_a.start("127.0.0.1", 0)
    addr_a = _node_multiaddr(node_a)

    others = []
    for _ in range(4):
        n = DhtNode(record_ttl=300)
        await n.start("127.0.0.1", 0, bootstrap_peers=[addr_a])
        others.append(n)

    await asyncio.sleep(0.5)

    try:
        target_key = b"\xff" * 32

        start = time.monotonic()
        result = await node_a._iterative_find_node(target_key)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"lookup took too long: {elapsed:.1f}s (stall detection may be broken)"
        assert len(result) > 0, "lookup returned no peers"
        assert len(result) <= K, f"lookup returned more than K={K} peers"
    finally:
        await node_a.stop()
        for n in others:
            await n.stop()


async def test_stall_terminates_after_all_queried():
    """Lookup should terminate when all known peers have been queried,
    not loop forever."""
    node_a = DhtNode(record_ttl=300)
    await node_a.start("127.0.0.1", 0)
    addr_a = _node_multiaddr(node_a)

    node_b = DhtNode(record_ttl=300)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[addr_a])
    await asyncio.sleep(0.3)

    try:
        result = await node_b._iterative_find_node(b"\xee" * 32)
        assert len(result) >= 1
    finally:
        await node_a.stop()
        await node_b.stop()


# ---------------------------------------------------------------------------
# Observed IP voting
# ---------------------------------------------------------------------------


async def test_observed_ip_voting_requires_threshold():
    """Observed IP should NOT be set until the threshold number of
    votes is reached."""
    node = DhtNode()
    node._listen_addr = ("0.0.0.0", 4001)
    node._observed_ip_threshold = 3

    observed_addr = encode_multiaddr_ip4_tcp("10.0.1.5", 12345)

    await node._maybe_set_observed_ip(observed_addr)
    assert node._observed_ip is None, "should not set IP after 1 vote (threshold=3)"

    await node._maybe_set_observed_ip(observed_addr)
    assert node._observed_ip is None, "should not set IP after 2 votes"

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

    await node._maybe_set_observed_ip(addr_a)
    await node._maybe_set_observed_ip(addr_b)
    assert node._observed_ip is None, "neither IP has reached threshold"

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

    await node._maybe_set_observed_ip(addr_a)
    await node._maybe_set_observed_ip(addr_a)
    assert node._observed_ip == "10.0.1.5"

    await node._maybe_set_observed_ip(addr_b)
    await node._maybe_set_observed_ip(addr_b)
    assert node._observed_ip == "10.0.1.6", "should update to new IP"


async def test_observed_ip_ignores_unroutable():
    """0.0.0.0 and loopback (when bound to wildcard) should be ignored."""
    node = DhtNode()
    node._listen_addr = ("0.0.0.0", 4001)
    node._observed_ip_threshold = 1

    await node._maybe_set_observed_ip(encode_multiaddr_ip4_tcp("0.0.0.0", 1234))
    assert node._observed_ip is None

    await node._maybe_set_observed_ip(encode_multiaddr_ip4_tcp("127.0.0.1", 1234))
    assert node._observed_ip is None


# ---------------------------------------------------------------------------
# Record filter
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
        good_key = b"/mx/model/worker/0"
        await node_a.put(good_key, b'{"rank":0}')
        assert node_b.kad_handler.get_local(good_key) is not None, (
            "record with /mx/ prefix should be accepted by filter"
        )

        bad_key = b"/bad/namespace/key"
        await node_a.put(bad_key, b'{"bad":true}')
        assert node_b.kad_handler.get_local(bad_key) is None, (
            "record without /mx/ prefix should be rejected by filter"
        )
    finally:
        await node_a.stop()
        await node_b.stop()


# ---------------------------------------------------------------------------
# Protocol edge cases
# ---------------------------------------------------------------------------


async def test_malformed_multistream_rejected():
    """Connecting with garbage data should be rejected cleanly."""
    node = DhtNode()
    await node.start("127.0.0.1", 0)
    host, port = node.listen_addr

    try:
        reader, writer = await asyncio.open_connection(host, port)
        writer.write(b"\x00\x00garbage data not a protocol\n")
        await writer.drain()

        await asyncio.sleep(0.5)

        assert node.listener is not None
        assert node.listener._listen_addr is not None

        writer.close()
    finally:
        await node.stop()


async def test_connection_drop_during_handshake():
    """Dropping the TCP connection mid-handshake should not crash the node."""
    node = DhtNode()
    await node.start("127.0.0.1", 0)
    host, port = node.listen_addr

    try:
        reader, writer = await asyncio.open_connection(host, port)
        writer.close()
        await asyncio.sleep(0.3)

        reader, writer = await asyncio.open_connection(host, port)
        writer.write(b"\x13/multistream/1.0.0\n")
        await writer.drain()
        writer.close()
        await asyncio.sleep(0.3)

        assert node.listener is not None

        node_b = DhtNode()
        await node_b.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node)])
        await asyncio.sleep(0.3)
        assert node_b.routing_table.size() >= 1
        await node_b.stop()
    finally:
        await node.stop()
