# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for critical bugs found during review.

Bug 1: Yamux receive window was never replenished after _send_window_update().
  The local _recv_window tracking monotonically decreased, causing streams to
  RST after transferring > DEFAULT_WINDOW_SIZE (256 KB) cumulative data.

Bug 2: Listener._active_connections was only decremented on handshake failure,
  not on success. After max_connections successful accepts, ALL new connections
  were rejected permanently.
"""

import asyncio

import pytest

from mx_libp2p.crypto import Ed25519Identity
from mx_libp2p.noise import handshake_initiator, handshake_responder
from mx_libp2p.multistream import negotiate_outbound, negotiate_inbound
from mx_libp2p.yamux import YamuxSession, DEFAULT_WINDOW_SIZE
from mx_libp2p.listener import Listener
from mx_libp2p.connection import accept, dial


# ---------------------------------------------------------------------------
# Bug 1: Yamux receive window replenishment
# ---------------------------------------------------------------------------


async def test_yamux_recv_window_replenished_after_read():
    """After reading data and sending a window update, the local _recv_window
    must be incremented. Without this fix, _recv_window monotonically decreases
    and eventually triggers a false window violation RST."""
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

        # Open a stream from client to server
        client_stream = await client_yamux.open_stream()
        server_stream = await server_yamux.accept_stream()

        # Send a chunk of data and read it on the other side
        chunk = b"x" * 1024
        await client_stream.write(chunk)
        data = await server_stream.read()
        assert data == chunk

        # After read(), _recv_window should be back to roughly DEFAULT_WINDOW_SIZE
        # (it was decremented by 1024 on receive, then incremented by 1024 on read)
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

        client_stream = await client_yamux.open_stream()
        server_stream = await server_yamux.accept_stream()

        # Transfer 512 KB total (2x the window size) in 4 KB chunks
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

        # Run sender and receiver concurrently with a timeout
        await asyncio.wait_for(
            asyncio.gather(sender(), receiver()),
            timeout=15.0,
        )

        assert total_received == total_bytes, (
            f"expected {total_bytes} bytes received, got {total_received}. "
            f"Stream was likely RST'd due to recv_window exhaustion."
        )
        # recv_window should still be positive (not driven to zero)
        assert server_stream._recv_window >= 0, (
            f"recv_window went negative: {server_stream._recv_window}"
        )
    finally:
        await client_yamux.stop()
        if server_yamux:
            await server_yamux.stop()
        server.close()


# ---------------------------------------------------------------------------
# Bug 2: Listener connection counter
# ---------------------------------------------------------------------------


async def test_listener_counter_decrements_on_success():
    """After a successful accept, _active_connections must be decremented
    so the counter doesn't permanently inflate. Before the fix, only failed
    handshakes decremented the counter."""
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
        # Make 3 successful connections
        conns = []
        for _ in range(3):
            client_id = Ed25519Identity.generate()
            conn = await asyncio.wait_for(
                dial(client_id, host, port),
                timeout=5.0,
            )
            conns.append(conn)
            # Small delay for the listener to process
            await asyncio.sleep(0.1)

        assert len(connections_received) == 3

        # The counter should NOT be stuck at 3. With the fix, it decrements
        # after each successful accept, so it should be 0 by now.
        assert listener._active_connections == 0, (
            f"_active_connections should be 0 after accepts complete, "
            f"got {listener._active_connections} (counter not decremented on success)"
        )

        # Clean up
        for c in conns:
            await c.close()
    finally:
        await listener.stop()


async def test_listener_does_not_reject_after_many_connections():
    """With a low max_connections limit, the listener should still accept
    new connections after previous ones have been handled. Before the fix,
    the counter grew without bound and hit the limit permanently."""
    identity = Ed25519Identity.generate()
    accepted = []

    async def on_conn(conn):
        accepted.append(conn)

    listener = Listener(
        identity,
        host="127.0.0.1",
        port=0,
        on_connection=on_conn,
        max_connections=2,  # Very low limit
    )
    await listener.start()
    host, port = listener.listen_addr

    try:
        # Make 5 sequential connections (more than max_connections)
        # Each should succeed because the counter resets after each accept
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
