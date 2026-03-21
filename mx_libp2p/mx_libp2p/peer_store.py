# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Peer store: maps peer IDs to addresses and active connections.

Provides connection reuse - get_or_dial() returns an existing live connection
or establishes a new one.
"""

import asyncio
import logging
import socket
import struct
import time

from typing import Callable

from .crypto import Ed25519Identity
from .connection import Connection, dial
from .multiaddr import decode_multiaddr, PROTO_IP4, PROTO_IP6, PROTO_TCP, PROTO_DNS, PROTO_DNS4, PROTO_DNS6

log = logging.getLogger(__name__)

# Dial backoff parameters
BACKOFF_BASE = 1.0        # initial backoff in seconds
BACKOFF_MAX = 60.0        # maximum backoff in seconds
MAX_DIAL_FAILURES = 5     # remove address after this many consecutive failures


class AddrInfo:
    """Tracks an address with dial failure state for backoff."""

    __slots__ = ("addr", "failures", "backoff_until")

    def __init__(self, addr: bytes):
        self.addr = addr
        self.failures: int = 0
        self.backoff_until: float = 0.0

    def record_failure(self) -> None:
        self.failures += 1
        backoff = min(BACKOFF_BASE * (2 ** (self.failures - 1)), BACKOFF_MAX)
        self.backoff_until = time.monotonic() + backoff

    def record_success(self) -> None:
        self.failures = 0
        self.backoff_until = 0.0

    @property
    def is_backed_off(self) -> bool:
        return time.monotonic() < self.backoff_until

    @property
    def should_remove(self) -> bool:
        return self.failures >= MAX_DIAL_FAILURES


class PeerInfo:
    """Known information about a peer."""

    def __init__(self, peer_id: bytes, addrs: list[bytes] | None = None):
        self.peer_id = peer_id
        self.addr_infos: dict[bytes, AddrInfo] = {}
        if addrs:
            for addr in addrs:
                self.addr_infos[addr] = AddrInfo(addr)
        self.connection: Connection | None = None

    @property
    def addrs(self) -> list[bytes]:
        return [info.addr for info in self.addr_infos.values()]

    @addrs.setter
    def addrs(self, value: list[bytes]) -> None:
        new_infos: dict[bytes, AddrInfo] = {}
        for addr in value:
            existing = self.addr_infos.get(addr)
            new_infos[addr] = existing if existing else AddrInfo(addr)
        self.addr_infos = new_infos


class PeerStore:
    """Manages peer addresses and connection reuse."""

    # Default max outbound connections (0 = unlimited)
    DEFAULT_MAX_CONNECTIONS = 256

    def __init__(
        self,
        identity: Ed25519Identity,
        supported_protocols: list[str] | None = None,
        on_new_connection: Callable | None = None,
        on_peer_unreachable: Callable | None = None,
        on_peer_connected: Callable | None = None,
        max_connections: int = DEFAULT_MAX_CONNECTIONS,
    ):
        self.identity = identity
        self.supported_protocols = supported_protocols
        self.on_new_connection = on_new_connection
        self.on_peer_unreachable = on_peer_unreachable
        self.on_peer_connected = on_peer_connected
        self.max_connections = max_connections
        self._peers: dict[bytes, PeerInfo] = {}
        self._dial_locks: dict[bytes, asyncio.Lock] = {}

    def add_addrs(self, peer_id: bytes, addrs: list[bytes]) -> None:
        """Record addresses for a peer."""
        info = self._peers.get(peer_id)
        if info is None:
            info = PeerInfo(peer_id)
            self._peers[peer_id] = info
        for addr in addrs:
            if addr not in info.addr_infos:
                info.addr_infos[addr] = AddrInfo(addr)

    def replace_addrs(self, peer_id: bytes, addrs: list[bytes]) -> None:
        """Replace all addresses for a peer (clears stale addrs first)."""
        info = self._peers.get(peer_id)
        if info is None:
            info = PeerInfo(peer_id)
            self._peers[peer_id] = info
        info.addr_infos = {addr: AddrInfo(addr) for addr in addrs}

    def set_connection(self, peer_id: bytes, conn: Connection) -> None:
        """Register an active connection for a peer (e.g. from an inbound accept)."""
        info = self._peers.get(peer_id)
        if info is None:
            info = PeerInfo(peer_id)
            self._peers[peer_id] = info
        info.connection = conn

    def get_connection(self, peer_id: bytes) -> Connection | None:
        """Return the active connection for a peer, or None."""
        info = self._peers.get(peer_id)
        if info is None:
            return None
        conn = info.connection
        if conn is not None and not conn.is_alive:
            info.connection = None
            return None
        return conn

    def get_addrs(self, peer_id: bytes) -> list[bytes]:
        """Return known addresses for a peer."""
        info = self._peers.get(peer_id)
        return info.addrs if info else []

    async def get_or_dial(self, peer_id: bytes, addrs: list[bytes] | None = None) -> Connection:
        """Get existing connection or dial a new one.

        If addrs are provided, they're added to the peer's known addresses.
        Serializes dials to the same peer to avoid duplicate connections.
        """
        if addrs:
            self.add_addrs(peer_id, addrs)

        # Check existing connection
        conn = self.get_connection(peer_id)
        if conn is not None:
            return conn

        # Serialize dials per peer
        if peer_id not in self._dial_locks:
            self._dial_locks[peer_id] = asyncio.Lock()

        async with self._dial_locks[peer_id]:
            # Re-check after acquiring lock
            conn = self.get_connection(peer_id)
            if conn is not None:
                return conn

            # Check connection limit before dialing
            if self.max_connections > 0:
                active = sum(
                    1 for info in self._peers.values()
                    if info.connection is not None and info.connection.is_alive
                )
                if active >= self.max_connections:
                    raise ConnectionError(
                        f"connection limit reached ({active}/{self.max_connections})"
                    )

            # Dial with backoff
            info = self._peers.get(peer_id)
            if not info or not info.addr_infos:
                raise ConnectionError(f"no addresses known for peer {peer_id.hex()[:16]}...")

            last_err = None
            for addr, addr_info in list(info.addr_infos.items()):
                if addr_info.is_backed_off:
                    log.debug(f"skipping backed-off address for {peer_id.hex()[:16]}...")
                    continue
                host, port = _extract_ip_tcp(addr)
                if host is None:
                    continue
                try:
                    conn = await dial(
                        self.identity, host, port,
                        supported_protocols=self.supported_protocols,
                    )
                    addr_info.record_success()
                    self.set_connection(peer_id, conn)
                    if self.on_new_connection:
                        self.on_new_connection(conn)
                    if self.on_peer_connected:
                        self.on_peer_connected(peer_id)
                    return conn
                except Exception as e:
                    last_err = e
                    addr_info.record_failure()
                    log.debug(
                        f"dial to {host}:{port} failed (attempt {addr_info.failures}/"
                        f"{MAX_DIAL_FAILURES}): {e}"
                    )
                    if addr_info.should_remove:
                        del info.addr_infos[addr]
                        log.debug(f"removed address {host}:{port} after {MAX_DIAL_FAILURES} failures")

            if self.on_peer_unreachable:
                self.on_peer_unreachable(peer_id)
            raise ConnectionError(
                f"failed to dial peer {peer_id.hex()[:16]}...: {last_err}"
            )

    def connected_peers(self) -> list[tuple[bytes, "Connection"]]:
        """Return all peers with live connections."""
        result = []
        for info in self._peers.values():
            if info.connection is not None and info.connection.is_alive:
                result.append((info.peer_id, info.connection))
        return result

    def prune_stale(self) -> int:
        """Remove peer entries that have no live connection and no addresses.

        Returns the number of entries removed. Called periodically by the
        DhtNode republish loop to prevent unbounded memory growth.
        Also cleans up dial locks for peers no longer in the store.
        """
        to_remove = []
        for peer_id, info in self._peers.items():
            has_live_conn = info.connection is not None and info.connection.is_alive
            has_addrs = len(info.addr_infos) > 0
            if not has_live_conn and not has_addrs:
                to_remove.append(peer_id)
            # Also clear dead connection objects to free resources
            if info.connection is not None and not info.connection.is_alive:
                info.connection = None
        for peer_id in to_remove:
            del self._peers[peer_id]
        if to_remove:
            log.debug(f"pruned {len(to_remove)} stale peer entries")
        # Clean up dial locks for any peer no longer tracked
        stale_locks = [pid for pid in self._dial_locks if pid not in self._peers]
        for pid in stale_locks:
            del self._dial_locks[pid]
        return len(to_remove)

    async def close_all(self) -> None:
        """Close all active connections."""
        for info in self._peers.values():
            if info.connection is not None:
                try:
                    await info.connection.close()
                except Exception:
                    pass
                info.connection = None


def _extract_ip_tcp(addr: bytes) -> tuple[str | None, int | None]:
    """Extract (host, port) from a binary multiaddr.

    Handles /ip4/, /ip6/, /dns/, /dns4/, /dns6/ with /tcp/.
    Returns (None, None) if the multiaddr doesn't contain a supported
    address + TCP combination.
    """
    components = decode_multiaddr(addr)
    host = None
    port = None
    for code, data in components:
        if code == PROTO_IP4:
            host = socket.inet_ntoa(data)
        elif code == PROTO_IP6:
            host = socket.inet_ntop(socket.AF_INET6, data)
        elif code in (PROTO_DNS, PROTO_DNS4, PROTO_DNS6):
            host = data.decode("utf-8")
        elif code == PROTO_TCP:
            port = struct.unpack(">H", data)[0]
    if host and port:
        return host, port
    return None, None


# Keep the old name as an alias for backwards compatibility in tests
_extract_ip4_tcp = _extract_ip_tcp
