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

from .connection import IDENTIFY_PROTOCOL, IDENTIFY_PUSH_PROTOCOL, Connection
from .crypto import Ed25519Identity
from .dht_bootstrap import BootstrapMixin
from .dht_identify import IdentifyMixin
from .dht_maintenance import BOOTSTRAP_INTERVAL, MaintenanceMixin
from .dht_queries import QueryMixin
from .dht_utils import _filter_routable_addrs, _log_task_exception
from .kad_handler import KadHandler
from .kademlia import KADEMLIA_PROTOCOL
from .listener import Listener
from .mdns import MdnsDiscovery
from .multiaddr import (
    encode_multiaddr_ip4_tcp_p2p,
    encode_multiaddr_ip_tcp_p2p,
    multiaddr_to_string,
)
from .peer_store import PeerStore
from .routing import ALPHA, K, RoutingTable

log = logging.getLogger(__name__)

# Default record TTL: 24 hours
DEFAULT_RECORD_TTL = 24 * 60 * 60
# Republish interval: 1 hour
DEFAULT_REPUBLISH_INTERVAL = 60 * 60
# Timeout for individual Kademlia RPCs (request + response after the
# connection is up). ``_get_value_single`` and friends apply this as
# ``wait_for(kad_rpc, rpc_timeout)`` AFTER the dial has completed via
# the separate ``dial_timeout`` budget below, so this covers the
# protocol round-trip only, not the cold-handshake cost.
#
# v0.3.1: dropped 10.0 -> 2.0. The 10s default came from over-conservative
# margin assumptions; in practice a healthy peer responds in <200ms once
# the connection is established. With alpha=3 iterative lookups, a
# single ghost peer at 10s eats the full RPC_TIMEOUT before a round can
# advance. 2s is a generous floor for healthy peers under typical
# WAN+RTT and lets ghost peers fail fast. Override per-node via the
# rpc_timeout constructor argument when running on a high-latency overlay.
RPC_TIMEOUT = 2.0
# Timeout for dial attempts (TCP + Noise + yamux + connection setup).
# This is the cold-handshake budget; legitimate slow peers (high-RTT
# WAN, busy overlays) hit this before they hit RPC_TIMEOUT, so this is
# the knob most likely to need bumping in production deployments where
# the per-connection setup is heavyweight.
#
# v0.3.1: dropped 5.0 -> 1.0. A successful TCP+Noise+yamux dial against
# a reachable peer completes in <100ms on a typical LAN/cluster
# overlay; 1s leaves a 10x margin while keeping ghost peer failure
# fast. Override per-node via dial_timeout when running on a
# high-latency overlay or against geographically distant peers.
DIAL_TIMEOUT = 1.0


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
        k: int = K,
        alpha: int = ALPHA,
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
            k: replication factor and bucket size (default 20). Determines how
                many closest peers a record is replicated to and how many peers
                each k-bucket holds. Matches the libp2p kad-dht spec's `k`.
            alpha: parallelism factor for iterative lookups (default 3). Number
                of concurrent FIND_NODE / GET_VALUE requests per round. Matches
                the libp2p kad-dht spec's `alpha`.
        """
        # Reject bool explicitly: bool is a subclass of int in Python so
        # ``isinstance(True, int)`` is True and ``True > 0`` is True, which
        # would silently treat ``DhtNode(k=True)`` as ``k=1``. Also reject
        # floats so a misconfigured ``k=20.0`` fails fast at construction
        # rather than at routing-table comparison time.
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(f"k must be an int, got {type(k).__name__}")
        if isinstance(alpha, bool) or not isinstance(alpha, int):
            raise TypeError(f"alpha must be an int, got {type(alpha).__name__}")
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.identity = identity or Ed25519Identity.generate()
        self.record_ttl = record_ttl
        self.republish_interval = republish_interval
        self.rpc_timeout = rpc_timeout
        self.dial_timeout = dial_timeout
        self._k = k
        self._alpha = alpha
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
            on_peer_unreachable=self._on_peer_unreachable,
            on_peer_connected=self._on_peer_connected,
        )
        # Routing table uses connection liveness to decide eviction
        self.routing_table = RoutingTable(
            self.identity.peer_id,
            k=self._k,
            is_alive=lambda pid: self.peer_store.get_connection(pid) is not None,
        )
        self.kad_handler = KadHandler(
            self.routing_table, record_filter=record_filter, k=self._k
        )
        self.listener: Listener | None = None
        self._listen_addr: tuple[str, int] | None = None
        self._republish_task: asyncio.Task | None = None
        self._bootstrap_task: asyncio.Task | None = None
        self._dispatch_tasks: set[asyncio.Task] = set()
        self._originated_records: dict[bytes, bytes] = {}  # key -> value (records WE originated)
        self._bootstrap_peers: list[str] = []
        # DNS / SLURM bootstrap fields are set by start(); initialize
        # here to safe defaults so the no-arg bootstrap() trigger and
        # _periodic_bootstrap_tick can be called before start() without
        # AttributeError. A pre-start bootstrap() with no sources still
        # raises NoKnownPeersError as designed; the fields just exist.
        self._bootstrap_dns: str | None = None
        self._bootstrap_dns_port: int = 4001
        self._bootstrap_hostlist: str | None = None
        # Cadence for the periodic bootstrap loop. Set by start();
        # initialized here to the module default so attribute access
        # before start() (e.g. from _periodic_bootstrap_loop in a unit
        # test that bypasses start()) returns a sane value. None
        # disables the loop entirely.
        self._bootstrap_interval: float | None = BOOTSTRAP_INTERVAL
        self._mdns: MdnsDiscovery | None = None
        # mDNS-only nodes start with an empty routing table so the
        # bootstrap-time wait_until_routable refresh skips. Track
        # whether the one-shot per-CPL refresh has been triggered yet,
        # so the FIRST mDNS-discovered peer fires it once. Subsequent
        # mDNS discoveries hit the maintenance loop's normal refresh
        # cadence rather than running fresh per-CPL walks every time.
        self._mdns_routable_refresh_done: bool = False

    @property
    def peer_id(self) -> bytes:
        return self.identity.peer_id

    @property
    def peer_id_short(self) -> str:
        return self.identity.peer_id.hex()[:16]

    @property
    def k(self) -> int:
        """Replication factor and bucket size for this node."""
        return self._k

    @property
    def alpha(self) -> int:
        """Lookup parallelism for this node."""
        return self._alpha

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
        wait_until_routable: bool = True,
        bootstrap_interval: float | None = BOOTSTRAP_INTERVAL,
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
            wait_until_routable: when True (default), after bootstrap
                completes its self-lookup, run one round of per-CPL
                bucket refresh before returning so the routing table has
                seen distant-bucket peers. Matches the original Kademlia
                paper's join procedure and rust-libp2p's behavior. Cold
                consumers that PUT/GET immediately after start() see
                full-overlay convergence rather than a partially-populated
                neighborhood. Set False to skip the per-bucket walks (for
                tests or constrained startups where the maintenance loop's
                periodic refresh is acceptable).
            bootstrap_interval: cadence in seconds for the periodic
                bootstrap loop. Defaults to ``BOOTSTRAP_INTERVAL``
                (5 minutes). Pass a smaller value for tighter cadences
                (e.g. cold-start convergence on large fleets) or ``None``
                to disable the periodic loop entirely. Mirrors
                rust-libp2p's ``Config::set_periodic_bootstrap_interval``
                with its ``Option<Duration>`` semantics where ``None``
                disables the timer. The interval applies for the
                lifetime of this run; restart the node with a different
                value to change it.
        """
        # Validate bootstrap_interval before storing. asyncio.sleep
        # crashes on non-numeric values and busy-spins on 0/negative,
        # and bool is an int subclass so False would silently fall
        # through the None-check and hit the <= 0 path. Reject all of
        # those at the boundary with the same shape as __init__'s
        # k/alpha validation.
        if bootstrap_interval is not None:
            if isinstance(bootstrap_interval, bool) or not isinstance(
                bootstrap_interval, (int, float)
            ):
                raise TypeError(
                    f"bootstrap_interval must be float, int, or None, "
                    f"got {type(bootstrap_interval).__name__}"
                )
            if bootstrap_interval <= 0:
                raise ValueError(
                    f"bootstrap_interval must be > 0 or None, got {bootstrap_interval}"
                )

        # Reset run-scoped mDNS refresh gate so a stop/start cycle on
        # the same DhtNode instance re-fires the one-shot refresh on
        # its next first-mDNS-peer.
        self._mdns_routable_refresh_done = False

        # Save for periodic re-bootstrap
        self._bootstrap_peers = bootstrap_peers or []
        self._bootstrap_dns = bootstrap_dns
        self._bootstrap_dns_port = bootstrap_dns_port
        self._bootstrap_interval = bootstrap_interval

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

        # Per-CPL bucket refresh: matches the original Kademlia join
        # procedure (Maymounkov & Mazieres 2002) and rust-libp2p's
        # synchronous bootstrap. Without this, a cold consumer sees only
        # the closest-neighbor bucket plus whatever the self-lookup
        # discovered transitively; PUTs and GETs against keys in distant
        # buckets fail until the 5-minute maintenance loop fires its own
        # refresh. Skipped on opt-out (wait_until_routable=False) and on
        # the no-bootstrap path (mDNS-only nodes have nothing to refresh
        # against until peers are discovered).
        has_bootstrap = self._bootstrap_peers or self._bootstrap_dns or self._bootstrap_hostlist
        if wait_until_routable and has_bootstrap and self.routing_table.size() > 0:
            await self._refresh_buckets()

        # Start background loops. The periodic bootstrap loop is gated
        # on bootstrap_interval being non-None: passing
        # ``bootstrap_interval=None`` disables the loop entirely so no
        # task is spawned (libp2p-kad's Option<Duration> semantics).
        self._republish_task = asyncio.create_task(self._republish_loop())
        if has_bootstrap and self._bootstrap_interval is not None:
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
            # Also start periodic bootstrap loop for mDNS (to re-query on sparse table).
            # Same interval gate as above: bootstrap_interval=None disables the loop.
            if not self._bootstrap_task and self._bootstrap_interval is not None:
                self._bootstrap_task = asyncio.create_task(self._periodic_bootstrap_loop())

    async def wait_until_routable(self) -> None:
        """Run one round of per-CPL bucket refresh.

        Provides a caller-driven equivalent to the synchronous bootstrap
        finalization in rust-libp2p (which chains per-bucket walks into
        the bootstrap QueryId so it doesn't finish until the buckets fill)
        and go-libp2p's ``RefreshRoutingTable`` / ``ForceRefresh``. After
        this returns, the routing table has seen peers across all
        non-empty CPL buckets, not just the closest-neighbor bucket
        populated by the self-lookup.

        Useful when ``DhtNode.start(wait_until_routable=False)`` was used
        and the caller wants to verify routing readiness before issuing
        the first PUT/GET. Idempotent and cheap to call multiple times,
        though the per-bucket walks have nontrivial wall-time cost.
        """
        if self.routing_table.size() == 0:
            return
        await self._refresh_buckets()

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

        # PUT to all closest peers (include our peer ID as publisher for interop).
        #
        # v0.3.1: replication PUT needs the full per-peer acceptance count
        # before returning, so this loop drains all K tasks and the
        # overall wall time is still bounded by the slowest per-peer
        # budget. The win for PUT comes from the dropped RPC_TIMEOUT and
        # DIAL_TIMEOUT defaults (per-peer worst case from 15s to 3s),
        # not from the ``asyncio.wait`` FIRST_COMPLETED structure - that
        # structure here is consistency with the iterative-lookup paths
        # plus correct per-task cancellation propagation when the
        # caller cancels the PUT mid-fanout.
        ttl_secs = int(ttl) if ttl is not None else int(self.record_ttl)
        success_count = 0
        tasks: set[asyncio.Task] = set()
        for peer_id, addrs in closest:
            tasks.add(asyncio.create_task(
                self._put_to_peer(peer_id, addrs, key, value,
                                  publisher=self.peer_id, ttl_secs=ttl_secs)
            ))

        try:
            pending = tasks
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    if task.cancelled():
                        continue
                    exc = task.exception()
                    if exc is not None:
                        continue
                    if task.result() is True:
                        success_count += 1
        finally:
            # If the caller cancels the PUT mid-fanout, propagate the
            # cancel to per-peer tasks so they don't leak into the
            # background. ``asyncio.gather`` previously did this
            # implicitly; ``asyncio.wait`` does not. Drain via gather
            # to retrieve any final exceptions cleanly.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

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
            from .crypto import _base58btc_encode
            from .mdns import _get_local_ips
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

        # mDNS-only nodes skip the bootstrap-time per-CPL refresh
        # (because start() ran with an empty routing table). Now that
        # we have at least one peer, fire the routability refresh once
        # in the background so distant-bucket discovery doesn't have to
        # wait for the 5-min maintenance loop. Spawned as a tracked
        # background task rather than awaited inline so this callback
        # stays fast.
        if (
            not self._mdns_routable_refresh_done
            and self.routing_table.size() > 0
        ):
            self._mdns_routable_refresh_done = True
            log.info(
                "mDNS: first peer discovered, firing one-shot routability refresh"
            )
            self._track_task(asyncio.create_task(self._refresh_buckets()))

    # -- Connection callbacks --------------------------------------------------

    async def _on_inbound_connection(self, conn: Connection) -> None:
        """Handle a new inbound connection.

        Performs an Identify exchange to learn the remote peer's real listen
        addresses, avoiding the ephemeral TCP source port problem.
        """
        self.peer_store.set_connection(conn.remote_peer_id, conn)
        self.routing_table.mark_connected(conn.remote_peer_id)
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

    def _on_peer_unreachable(self, peer_id: bytes) -> None:
        """Called by PeerStore when all dial attempts to a peer fail."""
        if self.routing_table.mark_disconnected(peer_id):
            log.debug(f"marked peer {peer_id.hex()[:16]}... as disconnected (dial failed)")

    def _on_peer_connected(self, peer_id: bytes) -> None:
        """Called by PeerStore when a dial to a peer succeeds.

        Promotes the entry to ``dial_verified=True`` (also setting
        ``connected=True``). The PeerStore only invokes this callback
        from outbound-dial-success paths, which is the authoritative
        signal we need to trust the listen addr we have for this peer
        and propagate it as a hint in FIND_NODE / GET_VALUE responses.
        """
        if self.routing_table.mark_dial_verified(peer_id):
            log.debug(
                f"marked peer {peer_id.hex()[:16]}... as dial-verified (outbound dial ok)"
            )

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
                # Track the per-stream handler task so DhtNode.stop()
                # cancels it via _dispatch_tasks rather than leaving it
                # to outlive the connection.
                t = asyncio.create_task(
                    self.kad_handler.handle_stream(stream, reader, writer, sender=remote_peer_id)
                )
                self._track_task(t)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.debug(f"kad dispatch error: {e}")
