# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-test: Noise XX handshake between two Python endpoints."""

import asyncio

from mx_libp2p.crypto import Ed25519Identity
from mx_libp2p.noise import handshake_initiator, handshake_responder
from mx_libp2p.multistream import negotiate_outbound, negotiate_inbound


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

        # Verify peer IDs match
        assert client_transport.remote_peer_id == server_identity.peer_id
        await asyncio.wait_for(handshake_done.wait(), timeout=5.0)
        assert server_transport.remote_peer_id == client_identity.peer_id

        # Echo test over encrypted transport
        await client_transport.write_msg(b"hello from pure python!")
        response = await client_transport.read_msg()
        assert response == b"echo:hello from pure python!"

        client_transport.close()
    finally:
        server.close()
