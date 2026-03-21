# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identify protocol integration for DhtNode (mixin).

Handles inbound/outbound Identify exchanges, Identify Push, and the
observed-IP voting mechanism that lets nodes discover their routable address.
"""

import asyncio
import logging
import socket

from .connection import Connection, IDENTIFY_PROTOCOL, IDENTIFY_PUSH_PROTOCOL
from .identify import encode_identify_msg, decode_identify_msg
from .kademlia import (
    KADEMLIA_PROTOCOL,
    _read_length_prefixed,
    _write_length_prefixed,
)
from .multiaddr import (
    decode_multiaddr,
    encode_multiaddr_ip4_tcp,
    PROTO_IP4,
    PROTO_IP6,
)
from .dht_utils import _filter_routable_addrs, _log_task_exception

log = logging.getLogger(__name__)


class IdentifyMixin:
    """Identify and Identify Push protocol handling for DhtNode."""

    def _setup_identify_handler(self, conn: Connection) -> None:
        """Register the Identify protocol handler on a connection."""
        q = conn.register_protocol(IDENTIFY_PROTOCOL)
        task = asyncio.create_task(self._dispatch_identify_streams(conn, q))
        self._track_task(task)

    async def _dispatch_identify_streams(self, conn: Connection, queue: asyncio.Queue) -> None:
        """Dispatch inbound Identify streams (we respond with our info)."""
        try:
            while True:
                stream, reader, writer = await queue.get()
                asyncio.create_task(self._handle_identify_stream(conn, stream, reader, writer))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.debug(f"identify dispatch error: {e}")

    async def _handle_identify_stream(
        self, conn: Connection, stream, reader, writer
    ) -> None:
        """Handle an inbound Identify stream: send our Identify message and close."""
        try:
            observed_addr = b""
            if conn.remote_addr:
                host, port = conn.remote_addr
                observed_addr = encode_multiaddr_ip4_tcp(host, port)

            msg = encode_identify_msg(
                identity=self.identity,
                listen_addrs=self.local_addrs(),
                observed_addr=observed_addr,
                protocols=[KADEMLIA_PROTOCOL, IDENTIFY_PROTOCOL, IDENTIFY_PUSH_PROTOCOL],
            )
            _write_length_prefixed(writer, msg)
            await writer.drain()
            await stream.close()
        except Exception as e:
            log.warning(f"identify handler error: {e}")

    def _setup_identify_push_handler(self, conn: Connection) -> None:
        """Register the Identify Push protocol handler on a connection."""
        q = conn.register_protocol(IDENTIFY_PUSH_PROTOCOL)
        task = asyncio.create_task(self._dispatch_identify_push_streams(conn, q))
        self._track_task(task)

    async def _dispatch_identify_push_streams(self, conn: Connection, queue: asyncio.Queue) -> None:
        """Dispatch inbound Identify Push streams."""
        try:
            while True:
                stream, reader, writer = await queue.get()
                asyncio.create_task(self._handle_identify_push_stream(conn, stream, reader, writer))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.debug(f"identify push dispatch error: {e}")

    async def _handle_identify_push_stream(
        self, conn: Connection, stream, reader, writer
    ) -> None:
        """Handle an inbound Identify Push stream: read the pushed message, update peer info."""
        try:
            data = await asyncio.wait_for(
                _read_length_prefixed(reader), timeout=5.0
            )
            info = decode_identify_msg(data)
            addrs = info.get("listen_addrs", [])
            agent = info.get("agent_version", "?")
            log.info(
                f"identify push from {conn.remote_peer_id.hex()[:16]}...: "
                f"agent={agent}, {len(addrs)} listen addr(s)"
            )

            routable = _filter_routable_addrs(addrs)
            if routable:
                self.routing_table.add_or_update(conn.remote_peer_id, routable)
                self.peer_store.replace_addrs(conn.remote_peer_id, routable)
        except Exception as e:
            log.debug(f"identify push handler error: {e}")

    async def _push_identify_to_all(self) -> None:
        """Push our updated Identify message to all connected peers (fire-and-forget)."""
        peers = self.peer_store.connected_peers()
        if not peers:
            return
        log.info(f"pushing updated identify to {len(peers)} connected peer(s)")
        for peer_id, conn in peers:
            try:
                stream, reader, writer = await asyncio.wait_for(
                    conn.open_stream(IDENTIFY_PUSH_PROTOCOL), timeout=5.0
                )
                msg = encode_identify_msg(
                    identity=self.identity,
                    listen_addrs=self.local_addrs(),
                    observed_addr=b"",
                    protocols=[KADEMLIA_PROTOCOL, IDENTIFY_PROTOCOL, IDENTIFY_PUSH_PROTOCOL],
                )
                _write_length_prefixed(writer, msg)
                await writer.drain()
                await stream.close()
            except Exception as e:
                log.debug(f"identify push to {peer_id.hex()[:16]}... failed: {e}")

    async def _perform_identify(self, conn: Connection) -> list[bytes]:
        """Open an outbound Identify stream and learn the remote peer's listen addrs.

        Returns a list of binary multiaddrs, or [] on failure.
        """
        try:
            stream, reader, writer = await asyncio.wait_for(
                conn.open_stream(IDENTIFY_PROTOCOL), timeout=5.0
            )
            data = await asyncio.wait_for(
                _read_length_prefixed(reader), timeout=5.0
            )
            info = decode_identify_msg(data)
            addrs = info.get("listen_addrs", [])
            agent = info.get("agent_version", "?")
            log.info(
                f"identify: peer {conn.remote_peer_id.hex()[:16]}... "
                f"agent={agent}, {len(addrs)} listen addr(s)"
            )

            # Extract observed_addr to learn our own routable IP
            await self._maybe_set_observed_ip(info.get("observed_addr", b""))

            return addrs
        except Exception as e:
            log.warning(f"identify exchange failed with {conn.remote_peer_id.hex()[:16]}...: {e}")
            return []

    async def _maybe_set_observed_ip(self, observed_addr: bytes) -> None:
        """Extract our IP from an Identify observed_addr and update votes.

        Uses multi-observer voting: an IP must be reported by at least
        _observed_ip_threshold distinct Identify exchanges before being
        accepted. If a new IP reaches threshold and differs from the current
        one, the observed IP is updated and an Identify Push is sent.

        Protected by _observed_ip_lock to prevent concurrent Identify
        exchanges from corrupting vote counts.
        """
        if not observed_addr:
            return

        try:
            components = decode_multiaddr(observed_addr)
        except Exception:
            return

        ip = None
        for code, data in components:
            if code == PROTO_IP4:
                ip = socket.inet_ntoa(data)
                break
            elif code == PROTO_IP6:
                ip = socket.inet_ntop(socket.AF_INET6, data)
                break

        if ip is None:
            return

        # Skip unroutable addresses
        if ip in ("0.0.0.0", "::"):
            return
        # Skip loopback only when bound to 0.0.0.0 (wildcard)
        if ip in ("127.0.0.1", "::1") and self._listen_addr and self._listen_addr[0] in ("0.0.0.0", "::"):
            return

        async with self._observed_ip_lock:
            self._observed_ip_votes[ip] = self._observed_ip_votes.get(ip, 0) + 1
            votes = self._observed_ip_votes[ip]

            if votes >= self._observed_ip_threshold and ip != self._observed_ip:
                old_ip = self._observed_ip
                self._observed_ip = ip
                # Reset votes so a future NAT change can be detected cleanly
                self._observed_ip_votes.clear()
                self._observed_ip_votes[ip] = votes
                if old_ip is None:
                    log.info(f"observed address confirmed: peers see us as {ip} ({votes} votes)")
                else:
                    log.info(f"observed address changed: {old_ip} -> {ip} ({votes} votes)")
                t = asyncio.create_task(self._push_identify_to_all())
                t.add_done_callback(_log_task_exception)
