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
import time

from .crypto import Ed25519Identity
from .connection import Connection, IDENTIFY_PROTOCOL, IDENTIFY_PUSH_PROTOCOL
from .listener import Listener
from .peer_store import PeerStore
from .routing import RoutingTable
from .kad_handler import KadHandler
from .kademlia import KADEMLIA_PROTOCOL
from .multiaddr import encode_multiaddr_ip4_tcp_p2p, encode_multiaddr_ip_tcp_p2p
from .dht_utils import _log_task_exception, _filter_routable_addrs
from .dht_identify import IdentifyMixin
from .dht_queries import QueryMixin
from .dht_bootstrap import BootstrapMixin
from .dht_maintenance import MaintenanceMixin
from .mdns import MdnsDiscovery
from .multiaddr import multiaddr_to_string

log = logging.getLogger(__name__)

# Default record TTL: 24 hours
DEFAULT_RECORD_TTL = 24 * 60 * 60
# Republish interval: 1 hour
DEFAULT_REPUBLISH_INTERVAL = 60 * 60
# Timeout for individual Kademlia RPCs (dial + request + response)
RPC_TIMEOUT = 10.0
# Timeout for dial attempts
DIAL_TIMEOUT = 5.0


class DhtNode(IdentifyMixin, QueryMixin, BootstrapMixin, MaintenanceMixin):
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
        self._bootstrap_hostlist: str | None = None
        self._mdns: MdnsDiscovery | None = None

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
        bootstrap_slurm: str | None = None,
        enable_mdns: bool | None = None,
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
            bootstrap_dns_port: port to use for DNS/SLURM-discovered peers
            bootstrap_slurm: SLURM compact hostlist (e.g. ``gpu[01-08]``).
                If None, auto-detected from ``SLURM_JOB_NODELIST`` env var.
                Each expanded hostname is resolved and dialed on
                bootstrap_dns_port.
            enable_mdns: enable mDNS peer discovery. If None (default),
                auto-detected: enabled when no other bootstrap mechanism
                is configured.
        """
        # Save for periodic re-bootstrap
        self._bootstrap_peers = bootstrap_peers or []
        self._bootstrap_dns = bootstrap_dns
        self._bootstrap_dns_port = bootstrap_dns_port

        # SLURM hostlist: explicit param or auto-detect from env
        if bootstrap_slurm is None:
            bootstrap_slurm = os.environ.get("SLURM_JOB_NODELIST")
        self._bootstrap_hostlist = bootstrap_slurm or None

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

        # Bootstrap from SLURM hostlist
        if self._bootstrap_hostlist:
            await self.bootstrap_from_hostlist(self._bootstrap_hostlist, self._bootstrap_dns_port)

        # Start background loops
        self._republish_task = asyncio.create_task(self._republish_loop())
        has_bootstrap = self._bootstrap_peers or self._bootstrap_dns or self._bootstrap_hostlist
        if has_bootstrap:
            self._bootstrap_task = asyncio.create_task(self._periodic_bootstrap_loop())

        # mDNS discovery: auto-enable when no explicit bootstrap is configured
        if enable_mdns is None:
            enable_mdns = not has_bootstrap
        if enable_mdns:
            from .crypto import _base58btc_encode
            self._mdns = MdnsDiscovery(
                peer_id=self.peer_id,
                peer_id_b58=_base58btc_encode(self.peer_id),
                listen_addrs=self._mdns_listen_addrs,
                on_peer_discovered=self._on_mdns_peer,
            )
            await self._mdns.start()
            # Also start periodic bootstrap loop for mDNS (to re-query on sparse table)
            if not self._bootstrap_task:
                self._bootstrap_task = asyncio.create_task(self._periodic_bootstrap_loop())

    async def stop(self) -> None:
        """Stop the DHT node."""
        if self._mdns:
            await self._mdns.stop()
            self._mdns = None

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

    # -- mDNS helpers ----------------------------------------------------------

    def _mdns_listen_addrs(self) -> list[str]:
        """Return multiaddr strings for mDNS announcements.

        Converts binary local_addrs() to strings. When bound on 0.0.0.0,
        falls back to _get_local_ips() to find the default interface IP.
        """
        addrs = self.local_addrs()
        if addrs:
            return [multiaddr_to_string(a) for a in addrs]

        # Fallback for 0.0.0.0: detect local IP
        if self._listen_addr:
            from .mdns import _get_local_ips
            from .crypto import _base58btc_encode
            host, port = self._listen_addr
            if host in ("0.0.0.0", "::"):
                local_ips = _get_local_ips()
                return [
                    f"/ip4/{ip}/tcp/{port}/p2p/{_base58btc_encode(self.peer_id)}"
                    for ip in local_ips
                ]
        return []

    async def _on_mdns_peer(self, multiaddr_str: str) -> None:
        """Callback when mDNS discovers a new peer."""
        from .dht_utils import _parse_peer_multiaddr
        try:
            peer_id, host, port = _parse_peer_multiaddr(multiaddr_str)
        except Exception as e:
            log.debug(f"mDNS: failed to parse discovered addr {multiaddr_str}: {e}")
            return

        if peer_id is None:
            return

        # Skip ourselves
        if peer_id == self.peer_id:
            return

        # Skip already-connected peers
        if self.peer_store.get_connection(peer_id) is not None:
            return

        log.info(f"mDNS: connecting to discovered peer {peer_id.hex()[:16]}...")
        await self.bootstrap([multiaddr_str])

    # -- Connection callbacks --------------------------------------------------

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

    # -- Task tracking ---------------------------------------------------------

    def _track_task(self, task: asyncio.Task) -> None:
        """Track a background task. Uses done callbacks for O(1) cleanup and error logging."""
        self._dispatch_tasks.add(task)
        task.add_done_callback(self._dispatch_tasks.discard)
        task.add_done_callback(_log_task_exception)

    # -- Kademlia stream dispatch ----------------------------------------------

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
