# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TCP listener for inbound libp2p connections.

Accepts inbound TCP connections and performs the responder-side handshake:
TCP -> Noise (responder) -> Yamux -> Connection.
"""

import asyncio
import logging
from typing import Callable, Awaitable

from .crypto import Ed25519Identity
from .connection import Connection, accept

log = logging.getLogger(__name__)


class Listener:
    """Listens for inbound libp2p connections on a TCP address."""

    # Default max concurrent inbound connections (0 = unlimited)
    DEFAULT_MAX_CONNECTIONS = 256

    def __init__(
        self,
        identity: Ed25519Identity,
        host: str = "0.0.0.0",
        port: int = 0,
        supported_protocols: list[str] | None = None,
        on_connection: Callable[[Connection], Awaitable[None]] | None = None,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
    ):
        self.identity = identity
        self.host = host
        self.port = port
        self.supported_protocols = supported_protocols
        self.on_connection = on_connection
        self.max_connections = max_connections
        self._server: asyncio.Server | None = None
        self._listen_addr: tuple[str, int] | None = None
        self._active_connections = 0

    @property
    def listen_addr(self) -> tuple[str, int] | None:
        """The (host, port) this listener is bound to, or None if not started."""
        return self._listen_addr

    async def start(self) -> tuple[str, int]:
        """Start listening. Returns the (host, port) we're bound to."""
        self._server = await asyncio.start_server(
            self._handle_connection, self.host, self.port
        )
        sock = self._server.sockets[0]
        self._listen_addr = sock.getsockname()[:2]
        log.info(f"listening on {self._listen_addr[0]}:{self._listen_addr[1]}")
        return self._listen_addr

    async def stop(self) -> None:
        """Stop accepting new connections."""
        if self._server:
            self._server.close()
            self._server = None
            self._listen_addr = None

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a single inbound TCP connection."""
        remote = writer.get_extra_info("peername")

        if self.max_connections > 0 and self._active_connections >= self.max_connections:
            log.warning(
                f"rejecting connection from {remote}: at limit "
                f"({self._active_connections}/{self.max_connections})"
            )
            writer.close()
            return

        self._active_connections += 1
        try:
            conn = await accept(
                self.identity, reader, writer,
                supported_protocols=self.supported_protocols,
            )
            log.info(f"accepted connection from {remote}, peer {conn.remote_peer_id.hex()[:16]}...")
            if self.on_connection:
                await self.on_connection(conn)
        except Exception as e:
            log.warning(f"failed to accept connection from {remote}: {e}")
            writer.close()
        finally:
            self._active_connections -= 1
