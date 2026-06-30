# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the responder-side handshake timeout (commit ebf39db).

Covers the slowloris-style failure mode: a peer that opens TCP but
sends nothing (or only partial multistream bytes) used to occupy a
listener slot indefinitely. The fix bounds the multistream + Noise +
Yamux setup with a wall-clock timeout; these tests assert the fix
fires, releases the slot, and the listener stays healthy.
"""

import asyncio

import pytest  # noqa: F401

from kademlite.connection import dial
from kademlite.crypto import Ed25519Identity
from kademlite.listener import Listener


async def test_listener_drops_silent_peer_after_handshake_timeout() -> None:
    """A peer that opens TCP and sends nothing must be dropped after the
    configured handshake_timeout, and the listener slot must be released."""
    identity = Ed25519Identity.generate()
    listener = Listener(
        identity,
        host="127.0.0.1",
        port=0,
        handshake_timeout=0.05,  # tight window for fast tests
        max_connections=4,
    )
    await listener.start()
    host, port = listener.listen_addr

    try:
        # Open a raw TCP connection that says nothing.
        reader, writer = await asyncio.open_connection(host, port)
        # Wait until well past the handshake timeout. Listener should
        # have given up by now and decremented the active counter.
        await asyncio.sleep(0.25)

        assert listener._active_connections == 0, (
            f"_active_connections should return to 0 after silent peer is "
            f"dropped, got {listener._active_connections}"
        )

        # Cleanup the dangling client side.
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
    finally:
        await listener.stop()


async def test_listener_recovers_after_dropping_silent_peers() -> None:
    """After dropping multiple silent peers, the listener must still accept
    a legitimate handshake from a real client."""
    identity = Ed25519Identity.generate()
    accepted = []

    async def on_conn(conn):
        accepted.append(conn)

    listener = Listener(
        identity,
        host="127.0.0.1",
        port=0,
        handshake_timeout=0.05,
        max_connections=4,
        on_connection=on_conn,
    )
    await listener.start()
    host, port = listener.listen_addr

    try:
        # Three silent peers that just sit there.
        silent = []
        for _ in range(3):
            r, w = await asyncio.open_connection(host, port)
            silent.append((r, w))

        # Wait past the timeout for all three to be dropped.
        await asyncio.sleep(0.2)
        assert listener._active_connections == 0

        # Real client should still complete a handshake.
        client_id = Ed25519Identity.generate()
        conn = await asyncio.wait_for(dial(client_id, host, port), timeout=5.0)
        assert conn is not None
        assert listener._active_connections == 0  # accept completed cleanly
        assert len(accepted) == 1

        await conn.close()

        for _r, w in silent:
            w.close()
            try:
                await w.wait_closed()
            except Exception:
                pass
    finally:
        await listener.stop()


async def test_listener_does_not_drop_legitimate_handshake() -> None:
    """A normal client whose handshake completes well within the timeout
    must not be cut off, even if the timeout is tight."""
    identity = Ed25519Identity.generate()
    accepted = []

    async def on_conn(conn):
        accepted.append(conn)

    # 5 seconds should be plenty for a localhost handshake; the goal is
    # to verify the timeout doesn't fire on the happy path.
    listener = Listener(
        identity,
        host="127.0.0.1",
        port=0,
        handshake_timeout=5.0,
        on_connection=on_conn,
    )
    await listener.start()
    host, port = listener.listen_addr

    try:
        client_id = Ed25519Identity.generate()
        conn = await asyncio.wait_for(dial(client_id, host, port), timeout=5.0)
        assert conn is not None
        assert len(accepted) == 1
        # After accept completes, slot is released.
        await asyncio.sleep(0.05)
        assert listener._active_connections == 0
        await conn.close()
    finally:
        await listener.stop()
