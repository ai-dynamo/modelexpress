# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DhtNode: complete Kademlia DHT node orchestrator.

Combines listener, peer store, routing table, and Kademlia handler into a
single high-level interface:

    node = DhtNode(identity)
    await node.start("127.0.0.1", 0, bootstrap_peers=["/ip4/.../tcp/.../p2p/..."])
    await node.put(key, value)
    value = await node.get(key)
    await node.stop()
"""

import asyncio
import logging
import os
import socket
import struct
import time

from .crypto import Ed25519Identity
from .connection import Connection, dial, IDENTIFY_PROTOCOL, IDENTIFY_PUSH_PROTOCOL
from .listener import Listener
from .peer_store import PeerStore
from .routing import RoutingTable, K, ALPHA, xor_distance
from .kad_handler import KadHandler
from .kademlia import (
    KADEMLIA_PROTOCOL,
    MSG_FIND_NODE,
    MSG_GET_VALUE,
    MSG_PUT_VALUE,
    kad_get_value,
    kad_put_value,
    kad_find_node,
    encode_kad_message,
    decode_kad_message,
    encode_record,
    encode_peer,
    _read_length_prefixed,
    _write_length_prefixed,
)
from .identify import encode_identify_msg, decode_identify_msg
from .multiaddr import (
    encode_multiaddr_ip4_tcp,
    encode_multiaddr_ip4_tcp_p2p,
    encode_multiaddr_ip_tcp_p2p,
    parse_multiaddr_string,
    decode_multiaddr,
    PROTO_IP4,
    PROTO_IP6,
    PROTO_TCP,
    PROTO_P2P,
)

log = logging.getLogger(__name__)


def _log_task_exception(task: asyncio.Task) -> None:
    """Done callback that logs unhandled exceptions from fire-and-forget tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.warning(f"background task {task.get_name()} failed: {exc}", exc_info=exc)

# Default record TTL: 24 hours
DEFAULT_RECORD_TTL = 24 * 60 * 60
# Republish interval: 1 hour
DEFAULT_REPUBLISH_INTERVAL = 60 * 60
# Periodic bootstrap interval: 5 minutes (matches rust-libp2p)
BOOTSTRAP_INTERVAL = 5 * 60
# Replication happens every N republish cycles (~4 hours at default interval)
REPLICATION_CYCLE_INTERVAL = 4
# Timeout for individual Kademlia RPCs (dial + request + response)
RPC_TIMEOUT = 10.0
# Timeout for dial attempts
DIAL_TIMEOUT = 5.0
# Max rounds for iterative lookups
MAX_LOOKUP_ROUNDS = 10
# When a lookup stalls (no new closer peers), increase parallelism by this factor
STALL_PARALLELISM_BOOST = 2
# Max concurrent background queries for republish/replication
MAX_CONCURRENT_BACKGROUND_QUERIES = 10


class DhtNode:
    """A complete Kademlia DHT node.

    Manages: listening, connection reuse, routing table, local record store,
    iterative lookups, bootstrap, and background republishing.
    """

    def __init__(
        self,
        identity: Ed25519Identity | None = None,
        record_ttl: float = DEFAULT_RECORD_TTL,
        republish_interval: float = DEFAULT_REPUBLISH_INTERVAL,
        rpc_timeout: float = RPC_TIMEOUT,
        dial_timeout: float = DIAL_TIMEOUT,
        record_filter=None,
    ):
        """
        Args:
            identity: Ed25519 identity for this node (generated if None)
            record_ttl: default record TTL in seconds
            republish_interval: how often to republish originated records
            rpc_timeout: timeout for individual Kademlia RPCs
            dial_timeout: timeout for dial attempts
            record_filter: optional callable(key: bytes, value: bytes) -> bool.
                If provided, inbound PUT_VALUE records are only accepted when
                this returns True. Useful for key namespace or value schema validation.
        """
        self.identity = identity or Ed25519Identity.generate()
        self.record_ttl = record_ttl
        self.republish_interval = republish_interval
        self.rpc_timeout = rpc_timeout
        self.dial_timeout = dial_timeout
        self._observed_ip: str | None = None
        self._observed_ip_votes: dict[str, int] = {}  # ip -> vote count
        # Number of confirmations needed before accepting an observed IP.
        # Set to 1 for single-peer setups (e.g. tests), 2+ for production.
        self._observed_ip_threshold = 2
        self._observed_ip_lock = asyncio.Lock()

        self.peer_store = PeerStore(
            self.identity,
            supported_protocols=[KADEMLIA_PROTOCOL, IDENTIFY_PROTOCOL, IDENTIFY_PUSH_PROTOCOL],
            on_new_connection=self._on_outbound_connection,
        )
        # Routing table uses connection liveness to decide eviction
        self.routing_table = RoutingTable(
            self.identity.peer_id,
            is_alive=lambda pid: self.peer_store.get_connection(pid) is not None,
        )
        self.kad_handler = KadHandler(self.routing_table, record_filter=record_filter)
        self.listener: Listener | None = None
        self._listen_addr: tuple[str, int] | None = None
        self._republish_task: asyncio.Task | None = None
        self._bootstrap_task: asyncio.Task | None = None
        self._dispatch_tasks: set[asyncio.Task] = set()
        self._originated_records: dict[bytes, bytes] = {}  # key -> value (records WE originated)
        self._bootstrap_peers: list[str] = []

    @property
    def peer_id(self) -> bytes:
        return self.identity.peer_id

    @property
    def peer_id_short(self) -> str:
        return self.identity.peer_id.hex()[:16]

    @property
    def listen_addr(self) -> tuple[str, int] | None:
        return self._listen_addr

    def local_addrs(self) -> list[bytes]:
        """Return our listen addresses as binary multiaddrs.

        Uses observed_ip (from Identify) when available, otherwise the bound
        address. Filters out non-routable addresses (0.0.0.0, ::) so we never
        advertise them to peers via Identify.
        """
        if self._listen_addr is None:
            return []
        host, port = self._listen_addr
        if self._observed_ip:
            host = self._observed_ip
        if host in ("0.0.0.0", "::"):
            return []
        return [encode_multiaddr_ip_tcp_p2p(host, port, self.peer_id)]

    def routable_addr(self) -> tuple[str, int]:
        """Return (host, port) using the best available address.

        Priority: observed_ip (from Identify) > bound address.
        Raises RuntimeError if the node hasn't started yet.
        """
        if self._listen_addr is None:
            raise RuntimeError("node not started")
        host, port = self._listen_addr
        if self._observed_ip:
            host = self._observed_ip
        return host, port

    async def start(
        self,
        listen_host: str = "127.0.0.1",
        listen_port: int = 0,
        bootstrap_peers: list[str] | None = None,
        bootstrap_dns: str | None = None,
        bootstrap_dns_port: int = 4001,
    ) -> None:
        """Start the DHT node.

        Args:
            listen_host: IP to listen on
            listen_port: port to listen on (0 for random)
            bootstrap_peers: multiaddr strings of bootstrap nodes
            bootstrap_dns: hostname to resolve for peer discovery (e.g. a
                K8s headless Service). Each resolved IP is dialed on
                bootstrap_dns_port. The peer ID is discovered via Noise
                handshake, so no prior knowledge is needed. If both
                bootstrap_peers and bootstrap_dns are given, both are used.
            bootstrap_dns_port: port to use for DNS-discovered peers
        """
        # Save for periodic re-bootstrap
        self._bootstrap_peers = bootstrap_peers or []
        self._bootstrap_dns = bootstrap_dns
        self._bootstrap_dns_port = bootstrap_dns_port

        # Start listener
        self.listener = Listener(
            self.identity,
            host=listen_host,
            port=listen_port,
            supported_protocols=[KADEMLIA_PROTOCOL, IDENTIFY_PROTOCOL, IDENTIFY_PUSH_PROTOCOL],
            on_connection=self._on_inbound_connection,
        )
        self._listen_addr = await self.listener.start()
        log.info(f"DHT node {self.peer_id_short}... listening on {self._listen_addr}")

        # Bootstrap from explicit multiaddrs
        if self._bootstrap_peers:
            await self.bootstrap(self._bootstrap_peers)

        # Bootstrap from DNS discovery
        if self._bootstrap_dns:
            await self.bootstrap_from_dns(self._bootstrap_dns, self._bootstrap_dns_port)

        # Start background loops
        self._republish_task = asyncio.create_task(self._republish_loop())
        if self._bootstrap_peers or self._bootstrap_dns:
            self._bootstrap_task = asyncio.create_task(self._periodic_bootstrap_loop())

    async def stop(self) -> None:
        """Stop the DHT node."""
        for task in [self._republish_task, self._bootstrap_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Cancel all dispatch tasks (snapshot the set since done callbacks mutate it)
        tasks = list(self._dispatch_tasks)
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._dispatch_tasks.clear()

        # Close connections (unblocks yamux read loops), then stop listener.
        await self.peer_store.close_all()

        if self.listener:
            await self.listener.stop()

        log.info(f"DHT node {self.peer_id_short}... stopped")

    async def bootstrap(self, peers: list[str]) -> None:
        """Connect to bootstrap peers and perform a self-lookup to populate the routing table."""
        for addr_str in peers:
            try:
                peer_id, host, port = _parse_peer_multiaddr(addr_str)
                if peer_id is None:
                    log.warning(f"bootstrap addr missing peer ID: {addr_str}")
                    continue

                addr_bytes = parse_multiaddr_string(addr_str)
                self.peer_store.add_addrs(peer_id, [addr_bytes])
                conn = await self.peer_store.get_or_dial(peer_id)
                self.routing_table.add_or_update(peer_id, [addr_bytes])
                log.info(f"bootstrapped with peer {peer_id.hex()[:16]}...")

                # Register protocol handlers for this connection
                self._setup_kad_handler(conn)
                self._setup_identify_handler(conn)
                self._setup_identify_push_handler(conn)

                # Identify exchange: learn bootstrap's real addrs + our observed IP
                addrs = await self._perform_identify(conn)
                routable = _filter_routable_addrs(addrs)
                if routable:
                    self.routing_table.add_or_update(peer_id, routable)
                    self.peer_store.replace_addrs(peer_id, routable)

            except Exception as e:
                log.warning(f"failed to bootstrap with {addr_str}: {e}")

        # Self-lookup to discover nearby peers
        if self.routing_table.size() > 0:
            await self._iterative_find_node(self.peer_id)

    async def bootstrap_from_dns(self, hostname: str, port: int = 4001) -> None:
        """Discover and connect to peers by resolving a DNS hostname.

        Resolves the hostname (e.g. a K8s headless Service) to get IP
        addresses, dials each one, and learns peer IDs via the Noise
        handshake. No prior knowledge of peer IDs is needed.

        IPs that match our own listen address are skipped. Failed dials
        are silently ignored (the peer may not be up yet).
        """
        try:
            loop = asyncio.get_event_loop()
            infos = await loop.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        except socket.gaierror as e:
            log.warning(f"DNS resolution failed for {hostname}: {e}")
            return

        # Deduplicate IPs
        ips = list(dict.fromkeys(
            info[4][0] for info in infos
        ))

        # Filter out our own IPs (both listen and observed, since they can differ)
        my_ips = set()
        if self._listen_addr:
            my_ips.add(self._listen_addr[0])
        if self._observed_ip:
            my_ips.add(self._observed_ip)
        ips = [ip for ip in ips if ip not in my_ips]

        if not ips:
            log.info(f"DNS bootstrap: {hostname} resolved but no peers to dial (only self)")
            return

        log.info(f"DNS bootstrap: {hostname} resolved to {len(ips)} peer IP(s)")

        connected = 0
        for ip in ips:
            try:
                conn = await asyncio.wait_for(
                    dial(self.identity, ip, port,
                         supported_protocols=[KADEMLIA_PROTOCOL, IDENTIFY_PROTOCOL, IDENTIFY_PUSH_PROTOCOL]),
                    timeout=self.dial_timeout,
                )
                peer_id = conn.remote_peer_id
                addr_bytes = encode_multiaddr_ip_tcp_p2p(ip, port, peer_id)

                self.peer_store.set_connection(peer_id, conn)
                self.peer_store.add_addrs(peer_id, [addr_bytes])
                self.routing_table.add_or_update(peer_id, [addr_bytes])

                self._setup_kad_handler(conn)
                self._setup_identify_handler(conn)
                self._setup_identify_push_handler(conn)

                # Identify exchange for real addresses + observed IP
                addrs = await self._perform_identify(conn)
                routable = _filter_routable_addrs(addrs)
                if routable:
                    self.routing_table.add_or_update(peer_id, routable)
                    self.peer_store.replace_addrs(peer_id, routable)

                connected += 1
                log.info(f"DNS bootstrap: connected to {ip}:{port} (peer {peer_id.hex()[:16]}...)")
            except Exception as e:
                log.debug(f"DNS bootstrap: failed to dial {ip}:{port}: {e}")

        log.info(f"DNS bootstrap: connected to {connected}/{len(ips)} peers")

        # Self-lookup to discover more peers
        if self.routing_table.size() > 0:
            await self._iterative_find_node(self.peer_id)

    async def put(self, key: bytes, value: bytes, ttl: float | None = None) -> int:
        """Store a key-value record in the DHT.

        Finds K closest peers to the key and sends PUT_VALUE to all of them.
        Also stores locally. Returns the number of peers that accepted the record.

        Args:
            key: record key
            value: record value
            ttl: per-record TTL in seconds. If None, the node's default
                 record_ttl is used. Enables different lifetimes for
                 directory entries vs status heartbeats.
        """
        # Store locally (publisher=None means we originated it)
        self.kad_handler.put_local(key, value, publisher=None, ttl=ttl)
        self._originated_records[key] = value

        # Find closest peers
        closest = await self._iterative_find_node(key)
        if not closest:
            log.debug(f"put {key!r}: no peers found, stored locally only")
            return 0

        # PUT to all closest peers (include our peer ID as publisher for interop)
        ttl_secs = int(ttl) if ttl is not None else int(self.record_ttl)
        success_count = 0
        tasks = []
        for peer_id, addrs in closest:
            tasks.append(self._put_to_peer(peer_id, addrs, key, value,
                                           publisher=self.peer_id, ttl_secs=ttl_secs))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if r is True:
                success_count += 1

        log.info(f"put {key!r}: stored on {success_count}/{len(closest)} peers")
        return success_count

    async def get(self, key: bytes) -> bytes | None:
        """Retrieve a value from the DHT.

        Performs an iterative GET: walks progressively closer peers until
        a record is found or all closest peers have been queried.
        """
        # Check local store first (use per-record TTL if set, else node default)
        local = self.kad_handler.get_local(key)
        if local is not None:
            effective_ttl = local.ttl if local.ttl is not None else self.record_ttl
            if time.monotonic() - local.timestamp <= effective_ttl:
                return local.value

        # Iterative GET
        return await self._iterative_get_value(key)

    def remove(self, key: bytes) -> bool:
        """Remove a record from the local store and stop republishing it.

        The record will be removed locally immediately. Remote copies will
        expire naturally via their TTL. For immediate removal from the
        network, publish a tombstone (empty value or status=REMOVED) instead.

        Returns True if the record existed, False otherwise.
        """
        existed = key in self._originated_records or key in self.kad_handler.records
        self._originated_records.pop(key, None)
        if key in self.kad_handler.records:
            del self.kad_handler.records[key]
        if existed:
            log.info(f"removed record {key!r} (local + stopped republish)")
        return existed

    async def _on_inbound_connection(self, conn: Connection) -> None:
        """Handle a new inbound connection.

        Performs an Identify exchange to learn the remote peer's real listen
        addresses, avoiding the ephemeral TCP source port problem.
        """
        self.peer_store.set_connection(conn.remote_peer_id, conn)
        self._setup_kad_handler(conn)
        self._setup_identify_handler(conn)
        self._setup_identify_push_handler(conn)

        # Identify exchange: learn the remote peer's real listen addresses
        addrs = await self._perform_identify(conn)
        routable = _filter_routable_addrs(addrs)
        if routable:
            self.routing_table.add_or_update(conn.remote_peer_id, routable)
            self.peer_store.replace_addrs(conn.remote_peer_id, routable)
        elif conn.remote_addr:
            # Fallback: use the observed ephemeral address (old behavior)
            host, port = conn.remote_addr
            addr = encode_multiaddr_ip4_tcp_p2p(host, port, conn.remote_peer_id)
            self.routing_table.add_or_update(conn.remote_peer_id, [addr])

    def _on_outbound_connection(self, conn: Connection) -> None:
        """Called by PeerStore when a new outbound connection is dialled."""
        self._setup_kad_handler(conn)
        self._setup_identify_handler(conn)
        self._setup_identify_push_handler(conn)

    def _track_task(self, task: asyncio.Task) -> None:
        """Track a background task. Uses done callbacks for O(1) cleanup and error logging."""
        self._dispatch_tasks.add(task)
        task.add_done_callback(self._dispatch_tasks.discard)
        task.add_done_callback(_log_task_exception)

    def _setup_kad_handler(self, conn: Connection) -> None:
        """Register the Kademlia protocol handler on a connection.

        register_protocol() returns the existing queue if already registered,
        so this is safe to call multiple times without orphaning streams.
        """
        q = conn.register_protocol(KADEMLIA_PROTOCOL)
        task = asyncio.create_task(self._dispatch_kad_streams(q, conn.remote_peer_id))
        self._track_task(task)

    async def _dispatch_kad_streams(self, queue: asyncio.Queue, remote_peer_id: bytes) -> None:
        """Dispatch inbound Kademlia streams to the handler."""
        try:
            while True:
                stream, reader, writer = await queue.get()
                t = asyncio.create_task(
                    self.kad_handler.handle_stream(stream, reader, writer, sender=remote_peer_id)
                )
                t.add_done_callback(_log_task_exception)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.debug(f"kad dispatch error: {e}")

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

    async def _iterative_find_node(self, target: bytes) -> list[tuple[bytes, list[bytes]]]:
        """Iterative FIND_NODE lookup with stall detection.

        Returns list of (peer_id, addrs) for the K closest peers found.

        When no new closer peers are discovered in a round (stall), the
        parallelism is increased to query more peers simultaneously. This
        matches rust-libp2p's adaptive approach to query termination.
        """
        # Seed with locally known closest peers
        closest = self.routing_table.closest_peers(target, K)
        if not closest:
            return []

        queried: set[bytes] = set()
        queried.add(self.peer_id)  # don't query ourselves
        peer_map: dict[bytes, list[bytes]] = {}  # peer_id -> addrs

        for entry in closest:
            peer_map[entry.peer_id] = entry.addrs

        parallelism = ALPHA
        stall_count = 0

        for _round in range(MAX_LOOKUP_ROUNDS):
            # Sort by distance, pick unqueried peers up to current parallelism
            candidates = sorted(peer_map.keys(), key=lambda p: xor_distance(p, target))
            to_query = [p for p in candidates if p not in queried][:parallelism]

            if not to_query:
                break

            tasks = []
            for peer_id in to_query:
                queried.add(peer_id)
                tasks.append(self._find_node_single(peer_id, peer_map.get(peer_id, []), target))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            new_closer_found = False
            # Track the previous best distance for stall detection
            prev_best = min(
                (xor_distance(p, target) for p in peer_map),
                default=None,
            )

            for result in results:
                if isinstance(result, Exception):
                    continue
                for pid, addrs in result:
                    if pid not in peer_map and pid != self.peer_id:
                        peer_map[pid] = addrs
                        self.routing_table.add_or_update(pid, addrs)
                        self.peer_store.add_addrs(pid, addrs)
                        if prev_best is None or xor_distance(pid, target) < prev_best:
                            new_closer_found = True

            if not new_closer_found:
                stall_count += 1
                if stall_count >= 2:
                    # Two consecutive stalls: query is converged
                    break
                # First stall: boost parallelism to try harder
                parallelism = min(ALPHA * STALL_PARALLELISM_BOOST, K)
            else:
                stall_count = 0
                parallelism = ALPHA

        # Return K closest
        sorted_peers = sorted(peer_map.keys(), key=lambda p: xor_distance(p, target))
        return [(p, peer_map[p]) for p in sorted_peers[:K]]

    async def _find_node_single(
        self, peer_id: bytes, addrs: list[bytes], target: bytes
    ) -> list[tuple[bytes, list[bytes]]]:
        """Send FIND_NODE to a single peer, return discovered peers."""
        try:
            conn = await asyncio.wait_for(
                self.peer_store.get_or_dial(peer_id, addrs), timeout=self.dial_timeout
            )
            response = await asyncio.wait_for(
                kad_find_node(conn, target), timeout=self.rpc_timeout
            )
            if response is None:
                return []

            result = []
            for peer_info in response.get("closer_peers", []):
                pid = peer_info.get("id")
                paddrs = peer_info.get("addrs", [])
                if pid:
                    result.append((pid, paddrs))
            return result
        except Exception as e:
            log.debug(f"find_node to {peer_id.hex()[:16]}... failed: {e}", exc_info=True)
            return []

    async def _iterative_get_value(self, key: bytes) -> bytes | None:
        """Iterative GET_VALUE lookup.

        Walks peers progressively closer to the key, stopping when a record
        is found or all closest peers have been queried.
        """
        closest = self.routing_table.closest_peers(key, K)
        if not closest:
            return None

        queried: set[bytes] = set()
        queried.add(self.peer_id)
        peer_map: dict[bytes, list[bytes]] = {}

        for entry in closest:
            peer_map[entry.peer_id] = entry.addrs

        for _round in range(MAX_LOOKUP_ROUNDS):
            candidates = sorted(peer_map.keys(), key=lambda p: xor_distance(p, key))
            to_query = [p for p in candidates if p not in queried][:ALPHA]

            if not to_query:
                break

            tasks = []
            for peer_id in to_query:
                queried.add(peer_id)
                tasks.append(self._get_value_single(peer_id, peer_map.get(peer_id, []), key))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    continue
                value, ttl, new_peers = result
                if value is not None:
                    # Found the record - propagate per-record TTL from the wire
                    self.kad_handler.put_local(key, value, ttl=ttl)
                    return value
                for pid, addrs in new_peers:
                    if pid not in peer_map and pid != self.peer_id:
                        peer_map[pid] = addrs
                        self.routing_table.add_or_update(pid, addrs)
                        self.peer_store.add_addrs(pid, addrs)

        return None

    async def _get_value_single(
        self, peer_id: bytes, addrs: list[bytes], key: bytes
    ) -> tuple[bytes | None, float | None, list[tuple[bytes, list[bytes]]]]:
        """Send GET_VALUE to a single peer.

        Returns (value_or_none, ttl_or_none, list_of_closer_peers).
        The TTL is extracted from the record's wire format (protobuf field 777)
        so it can be propagated when caching the record locally.
        """
        try:
            conn = await asyncio.wait_for(
                self.peer_store.get_or_dial(peer_id, addrs), timeout=self.dial_timeout
            )
            response = await asyncio.wait_for(
                kad_get_value(conn, key), timeout=self.rpc_timeout
            )
            if response is None:
                return None, None, []

            # Check if response contains a record
            record = response.get("record")
            if record and record.get("value") is not None:
                wire_ttl = record.get("ttl")
                ttl = float(wire_ttl) if wire_ttl is not None else None
                return record["value"], ttl, []

            # Otherwise collect closer peers
            new_peers = []
            for peer_info in response.get("closer_peers", []):
                pid = peer_info.get("id")
                paddrs = peer_info.get("addrs", [])
                if pid:
                    new_peers.append((pid, paddrs))
            return None, None, new_peers
        except Exception as e:
            log.debug(f"get_value from {peer_id.hex()[:16]}... failed: {e}", exc_info=True)
            return None, None, []

    async def _put_to_peer(
        self, peer_id: bytes, addrs: list[bytes], key: bytes, value: bytes,
        publisher: bytes | None = None, ttl_secs: int | None = None,
    ) -> bool:
        """Send PUT_VALUE to a single peer."""
        try:
            conn = await asyncio.wait_for(
                self.peer_store.get_or_dial(peer_id, addrs), timeout=self.dial_timeout
            )
            await asyncio.wait_for(
                kad_put_value(conn, key, value, publisher=publisher, ttl=ttl_secs),
                timeout=self.rpc_timeout,
            )
            return True
        except Exception as e:
            log.debug(f"put_value to {peer_id.hex()[:16]}... failed: {e}", exc_info=True)
            return False

    def _prune_dead_peers(self) -> None:
        """Remove peers with dead connections from the routing table."""
        for entry in self.routing_table.all_peers():
            conn = self.peer_store.get_connection(entry.peer_id)
            if conn is not None and not conn.is_alive:
                self.routing_table.remove(entry.peer_id)

    async def _periodic_bootstrap_loop(self) -> None:
        """Periodically re-bootstrap to discover new peers and refresh buckets.

        Always runs a self-lookup (and re-dials bootstrap peers when sparse).
        This matches rust-libp2p behavior: periodic bootstrap is unconditional,
        not gated on routing table size.
        """
        try:
            while True:
                await asyncio.sleep(BOOTSTRAP_INTERVAL)
                size = self.routing_table.size()

                if size < K:
                    # Sparse table: re-dial bootstrap peers first
                    log.info(
                        f"periodic re-bootstrap: routing table has "
                        f"{size} peers (< K={K}), re-dialing bootstrap peers"
                    )
                    if self._bootstrap_peers:
                        await self.bootstrap(self._bootstrap_peers)
                    if self._bootstrap_dns:
                        await self.bootstrap_from_dns(self._bootstrap_dns, self._bootstrap_dns_port)
                elif size > 0:
                    # Table is healthy: just do a self-lookup to discover
                    # new nearby peers and refresh routing
                    log.debug(
                        f"periodic self-lookup: routing table has {size} peers"
                    )
                    await self._iterative_find_node(self.peer_id)

                # Bucket refresh: lookup a random key in each non-empty bucket
                # to discover peers we wouldn't find through self-lookup alone
                await self._refresh_buckets()
        except asyncio.CancelledError:
            pass

    async def _refresh_buckets(self) -> None:
        """Refresh routing table buckets by looking up a random key in each.

        For each non-empty bucket, generates a random peer ID at that bucket's
        distance and performs an iterative lookup. This discovers peers that a
        self-lookup alone would miss (peers in distant buckets with no natural
        traffic).
        """
        for i, bucket in enumerate(self.routing_table._buckets):
            if not bucket.peers:
                continue
            # Generate a random key at this bucket's CPL distance
            random_key = self._random_key_for_bucket(i)
            try:
                await self._iterative_find_node(random_key)
            except Exception as e:
                log.debug(f"bucket {i} refresh failed: {e}")

    def _random_key_for_bucket(self, cpl: int) -> bytes:
        """Generate a random peer ID that falls into the given bucket (CPL).

        The key shares `cpl` leading bits with our peer ID, then differs at
        bit `cpl`, and is random after that.
        """
        local = self.peer_id
        key = bytearray(os.urandom(len(local)))

        # Copy the first `cpl` full bytes
        full_bytes = cpl // 8
        for i in range(full_bytes):
            key[i] = local[i]

        # Handle the partial byte: copy top bits, flip the divergence bit
        remaining_bits = cpl % 8
        if full_bytes < len(local):
            byte_val = local[full_bytes]
            # The bit at position `remaining_bits` must differ from local
            flip_bit = 0x80 >> remaining_bits
            # XOR the original byte to flip exactly the divergence bit
            flipped = byte_val ^ flip_bit
            # Keep top (remaining_bits + 1) bits from flipped, rest random
            control_mask = ((0xFF << (8 - remaining_bits)) | flip_bit) & 0xFF
            key[full_bytes] = (flipped & control_mask) | (key[full_bytes] & ~control_mask)

        return bytes(key)

    async def _republish_loop(self) -> None:
        """Background loop: re-PUT originated records, replicate stored records, and expire old ones.

        Uses a semaphore to limit concurrent background queries, preventing
        a spike of outbound connections when republishing/replicating many records.
        """
        replication_counter = 0
        sem = asyncio.Semaphore(MAX_CONCURRENT_BACKGROUND_QUERIES)
        try:
            while True:
                await asyncio.sleep(self.republish_interval)

                # Prune dead peers from routing table
                self._prune_dead_peers()
                self.peer_store.prune_stale()

                # Expire old records
                self.kad_handler.remove_expired(self.record_ttl)

                # Republish records we originated (rate-limited)
                async def _republish(key, value):
                    async with sem:
                        await self.put(key, value)

                republish_tasks = []
                for key, value in list(self._originated_records.items()):
                    republish_tasks.append(asyncio.create_task(_republish(key, value)))
                if republish_tasks:
                    results = await asyncio.gather(*republish_tasks, return_exceptions=True)
                    for i, r in enumerate(results):
                        if isinstance(r, Exception):
                            log.debug(f"republish failed: {r}")

                # Replicate ALL stored records every N cycles to handle topology changes.
                # This includes records received from other peers, ensuring that if
                # the originator dies, copies survive on the K closest nodes.
                replication_counter += 1
                if replication_counter >= REPLICATION_CYCLE_INTERVAL:
                    replication_counter = 0

                    async def _replicate(key, value):
                        async with sem:
                            closest = await self._iterative_find_node(key)
                            for peer_id, addrs in closest:
                                await self._put_to_peer(peer_id, addrs, key, value)

                    replicate_tasks = []
                    for key, rec in list(self.kad_handler.records.items()):
                        if key not in self._originated_records:
                            replicate_tasks.append(asyncio.create_task(_replicate(key, rec.value)))
                    if replicate_tasks:
                        results = await asyncio.gather(*replicate_tasks, return_exceptions=True)
                        for r in results:
                            if isinstance(r, Exception):
                                log.debug(f"replicate failed: {r}")
        except asyncio.CancelledError:
            pass


def _filter_routable_addrs(addrs: list[bytes]) -> list[bytes]:
    """Filter out multiaddrs containing non-routable IP addresses (0.0.0.0, ::).

    Returns only addresses that are safe to store in routing tables and peer stores.
    """
    result = []
    for addr in addrs:
        try:
            components = decode_multiaddr(addr)
        except Exception:
            continue
        routable = True
        for code, data in components:
            if code == PROTO_IP4 and socket.inet_ntoa(data) == "0.0.0.0":
                routable = False
                break
            if code == PROTO_IP6 and socket.inet_ntop(socket.AF_INET6, data) == "::":
                routable = False
                break
        if routable:
            result.append(addr)
    return result


def _parse_peer_multiaddr(addr_str: str) -> tuple[bytes | None, str | None, int | None]:
    """Parse a multiaddr string and extract (peer_id, host, port).

    Supports formats:
        /ip4/<host>/tcp/<port>/p2p/<peer_id>
        /ip6/<host>/tcp/<port>/p2p/<peer_id>
        /dns/<host>/tcp/<port>/p2p/<peer_id>
        /dns4/<host>/tcp/<port>/p2p/<peer_id>
        /dns6/<host>/tcp/<port>/p2p/<peer_id>
    """
    addr_bytes = parse_multiaddr_string(addr_str)
    components = decode_multiaddr(addr_bytes)

    peer_id = None
    host = None
    port = None

    for code, data in components:
        if code == PROTO_IP4:
            host = socket.inet_ntoa(data)
        elif code == PROTO_IP6:
            host = socket.inet_ntop(socket.AF_INET6, data)
        elif code == PROTO_TCP:
            port = struct.unpack(">H", data)[0]
        elif code == PROTO_P2P:
            peer_id = data

    return peer_id, host, port
