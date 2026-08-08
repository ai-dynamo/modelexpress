# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap providers for DhtNode (mixin).

Handles peer discovery from multiple sources: explicit multiaddrs, DNS
resolution (K8s headless services), and SLURM hostlist expansion.  All
IP-based providers share a common concurrent dial helper.
"""

import asyncio
import logging
import socket

from .connection import IDENTIFY_PROTOCOL, IDENTIFY_PUSH_PROTOCOL, dial
from .dht_utils import _filter_routable_addrs, _parse_peer_multiaddr
from .kademlia import KADEMLIA_PROTOCOL
from .multiaddr import (
    encode_multiaddr_ip_tcp_p2p,
    parse_multiaddr_string,
)
from .slurm import expand_hostlist

log = logging.getLogger(__name__)


class NoKnownPeersError(RuntimeError):
    """Raised by ``DhtNode.bootstrap()`` (no-arg form) when there is
    nothing to bootstrap against: the routing table is empty AND no
    bootstrap sources (explicit peers, DNS hostname, SLURM hostlist)
    are configured. Mirrors libp2p-kad's ``NoKnownPeers`` error type
    (see rust ``libp2p_kad::behaviour::NoKnownPeers``)."""


class BootstrapMixin:
    """Bootstrap providers for DhtNode."""

    async def bootstrap(self, peers: list[str] | None = None) -> None:
        """Trigger a bootstrap cycle.

        Two call shapes are supported:

        - ``await node.bootstrap()`` (no args) runs one on-demand
          bootstrap cycle equivalent to a single periodic-bootstrap
          tick: prune dead peers, re-dial the configured bootstrap
          sources when the routing table is below ``k``, otherwise
          self-lookup, then refresh per-CPL buckets. Mirrors
          rust-libp2p ``Behaviour::bootstrap``. If the routing table
          is empty AND no bootstrap sources are configured, raises
          :class:`NoKnownPeersError` (matching libp2p-kad's
          ``NoKnownPeers``).
        - ``await node.bootstrap(peers)`` with an explicit list of
          peer multiaddrs dials those peers, adds them to the routing
          table, and performs an Identify exchange. A post-dial
          self-lookup runs only when the routing table holds fewer than
          ``k`` reachable peers after the dials; once the table is at
          ``>= k`` connected entries the walk is opportunistic keyspace
          coverage and is skipped. This is the original kademlite
          ``bootstrap`` shape and stays available for callers that want
          to add seed peers without going through ``DhtNode.start``.

        The two shapes never both run in one call: when ``peers`` is
        ``None`` (or omitted) the on-demand-tick path runs; when
        ``peers`` is a (possibly empty) list, the dial-peers path runs.
        """
        if peers is None:
            await self._bootstrap_now()
            return
        await self._bootstrap_dial_peers(peers)

    async def _bootstrap_now(self) -> None:
        """One on-demand bootstrap cycle (libp2p-kad shape).

        Equivalent to one ``_periodic_bootstrap_tick`` body, plus an
        empty-state guard that raises :class:`NoKnownPeersError` when
        there is nothing to bootstrap against.
        """
        # Guard: matches libp2p-kad's NoKnownPeers semantic. If we have
        # neither a populated routing table nor any configured bootstrap
        # source to fall back on, the caller's request to bootstrap is
        # unfulfillable; surface that explicitly rather than running a
        # silent no-op tick. mDNS counts as a source: the tick fires
        # ``self._mdns.send_query()`` on a sparse table, which can
        # discover peers on the local segment without any other config.
        has_bootstrap_source = bool(
            self._bootstrap_peers
            or self._bootstrap_dns
            or self._bootstrap_hostlist
            or self._mdns
        )
        if self.routing_table.size() == 0 and not has_bootstrap_source:
            raise NoKnownPeersError(
                "cannot bootstrap: routing table is empty and no bootstrap "
                "sources (peers, DNS, SLURM hostlist, mDNS) are configured"
            )
        await self._periodic_bootstrap_tick()

    async def _bootstrap_dial_peers(self, peers: list[str]) -> None:
        """Connect to bootstrap peers and conditionally self-lookup to
        populate the routing table.

        After the explicit dial loop, a post-dial
        ``_iterative_find_node(self.peer_id)`` walk runs only when the
        routing table holds fewer than ``k`` reachable entries. When
        the dial phase already populated the table to ``>= k``
        connected peers, the walk is opportunistic keyspace coverage
        and is skipped to avoid paying ``DIAL_TIMEOUT`` per ghost
        against responders whose tables include unreachable peers.
        """
        for addr_str in peers:
            try:
                peer_id, host, port = _parse_peer_multiaddr(addr_str)
                if peer_id is None:
                    log.warning(f"bootstrap addr missing peer ID: {addr_str}")
                    continue

                addr_bytes = parse_multiaddr_string(addr_str)
                self.peer_store.add_addrs(peer_id, [addr_bytes])
                # Check whether we already have a cached connection to this
                # peer BEFORE calling ``get_or_dial``. ``get_or_dial`` returns
                # the cached connection without dialing if one is alive
                # (peer_store.py: ``get_or_dial`` early-returns at the
                # ``get_connection`` check), and that cached connection may
                # have been an inbound-only socket from
                # ``_on_inbound_connection``. Promoting a peer to
                # ``dial_verified`` based on a cached connection that we
                # never actually outbound-dialled would re-introduce the
                # leak v0.3.3 is closing.
                already_connected = self.peer_store.get_connection(peer_id) is not None
                conn = await asyncio.wait_for(
                    self.peer_store.get_or_dial(peer_id),
                    timeout=self.dial_timeout,
                )
                self.routing_table.add_or_update(peer_id, [addr_bytes])
                # Promotion semantics: only promote to ``dial_verified`` when
                # ``get_or_dial`` actually performed a fresh outbound dial.
                # When a cached connection was returned, fall back to
                # ``mark_connected`` (still resets ``connected=True`` for a
                # peer previously marked unreachable, matching the v0.3.2
                # mark_connected role) without claiming dial verification.
                # The ``_on_peer_connected`` callback fires synchronously
                # inside ``get_or_dial`` *before* ``add_or_update`` runs,
                # so the routing-table entry does not yet exist when the
                # callback fires; the explicit post-``add_or_update`` mark
                # is what actually lands the flag. This explicit-mark path
                # is therefore the load-bearing one and must distinguish
                # cache vs fresh dial.
                if already_connected:
                    self.routing_table.mark_connected(peer_id)
                else:
                    self.routing_table.mark_dial_verified(peer_id)
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

        # Self-lookup to discover nearby peers. Skip when the dial phase
        # already populated the routing table to >= k *reachable* entries:
        # the walk is opportunistic keyspace coverage at that point, not
        # a correctness requirement, and walking against responders whose
        # routing tables include unreachable peers can pay one
        # ``DIAL_TIMEOUT`` per ghost in alpha-sized batches before any
        # subsequent operation (GET_VALUE, PUT_VALUE) can run. Counting
        # connected-only avoids the corner case where the table holds
        # ``k`` stale ``connected=False`` entries from a prior session;
        # those should not suppress a fresh walk.
        rt_size = self.routing_table.size(connected_only=True)
        if 0 < rt_size < self._k:
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

        ips = list(dict.fromkeys(info[4][0] for info in infos))
        await self._dial_ips(ips, port, "DNS")

    async def bootstrap_from_hostlist(self, hostlist: str, port: int = 4001) -> None:
        """Discover and connect to peers by expanding a SLURM hostlist.

        Expands the compact hostlist notation (e.g. ``gpu[01-08]``), resolves
        each hostname via getaddrinfo, and dials them. Peer IDs are learned
        via the Noise handshake. Mirrors bootstrap_from_dns() behavior.

        Hostnames that resolve to our own listen address are skipped.
        """
        hostnames = expand_hostlist(hostlist)
        if not hostnames:
            log.warning(f"SLURM hostlist expansion produced no hosts: {hostlist!r}")
            return

        loop = asyncio.get_event_loop()
        ip_set: dict[str, None] = {}  # ordered dedup
        for hostname in hostnames:
            try:
                infos = await loop.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
                for info in infos:
                    ip_set.setdefault(info[4][0], None)
            except socket.gaierror as e:
                log.debug(f"SLURM bootstrap: failed to resolve {hostname}: {e}")

        await self._dial_ips(list(ip_set), port, "SLURM")

    async def _dial_ips(self, ips: list[str], port: int, label: str) -> None:
        """Dial a list of IPs concurrently with bounded parallelism.

        Shared implementation for DNS and SLURM bootstrap. Filters out our
        own IPs, dials up to ``k`` peers, learns peer IDs via Noise
        handshake, and performs an Identify exchange. A post-dial
        self-lookup runs only when the routing table holds fewer than
        ``k`` reachable entries after the dials; once the table is at
        ``>= k`` connected entries the walk is opportunistic keyspace
        coverage and is skipped (same rationale as
        ``_bootstrap_dial_peers``).
        """
        # Filter out our own IPs (both listen and observed, since they can differ)
        my_ips: set[str] = set()
        if self._listen_addr:
            my_ips.add(self._listen_addr[0])
        if self._observed_ip:
            my_ips.add(self._observed_ip)
        ips = [ip for ip in ips if ip not in my_ips]

        if not ips:
            log.info(f"{label} bootstrap: resolved but no peers to dial (only self)")
            return

        log.info(f"{label} bootstrap: resolved to {len(ips)} peer IP(s)")

        max_concurrent = 20
        sem = asyncio.Semaphore(max_concurrent)
        connected = 0
        lock = asyncio.Lock()

        async def dial_one(ip: str) -> bool:
            nonlocal connected
            # Atomically reserve a slot under the lock: check-and-reserve
            # in one critical section so concurrent dial_one tasks can't
            # all pass an early check and overshoot k. The reservation is
            # released on failure inside the dial path.
            async with lock:
                if connected >= self._k:
                    return False
                connected += 1
            async with sem:
                ok = False
                try:
                    conn = await asyncio.wait_for(
                        dial(
                            self.identity,
                            ip,
                            port,
                            supported_protocols=[
                                KADEMLIA_PROTOCOL,
                                IDENTIFY_PROTOCOL,
                                IDENTIFY_PUSH_PROTOCOL,
                            ],
                        ),
                        timeout=self.dial_timeout,
                    )
                    peer_id = conn.remote_peer_id
                    addr_bytes = encode_multiaddr_ip_tcp_p2p(ip, port, peer_id)

                    self.peer_store.set_connection(peer_id, conn)
                    self.peer_store.add_addrs(peer_id, [addr_bytes])
                    self.routing_table.add_or_update(peer_id, [addr_bytes])
                    # Successful outbound dial: promote to ``dial_verified``
                    # (which also sets ``connected=True``). Required so the
                    # responder-side filter (``dial_verified_only=True``)
                    # propagates this peer as a closer-peer hint to others.
                    # ``add_or_update`` alone does not touch the flags.
                    # See parallel comment in ``_bootstrap_dial_peers``.
                    self.routing_table.mark_dial_verified(peer_id)

                    self._setup_kad_handler(conn)
                    self._setup_identify_handler(conn)
                    self._setup_identify_push_handler(conn)

                    # Identify exchange for real addresses + observed IP
                    addrs = await self._perform_identify(conn)
                    routable = _filter_routable_addrs(addrs)
                    if routable:
                        self.routing_table.add_or_update(peer_id, routable)
                        self.peer_store.replace_addrs(peer_id, routable)

                    peer_short = peer_id.hex()[:16]
                    log.info(
                        f"{label} bootstrap: connected to {ip}:{port} (peer {peer_short}...)"
                    )
                    ok = True
                    return True
                except Exception as e:
                    log.debug(f"{label} bootstrap: failed to dial {ip}:{port}: {e}")
                    return False
                finally:
                    # Release the reserved slot on any non-success exit
                    # (regular Exception, asyncio.CancelledError, anything).
                    # asyncio.CancelledError is a BaseException, not Exception,
                    # so the except clause above wouldn't catch it.
                    if not ok:
                        async with lock:
                            connected -= 1

        await asyncio.gather(*(dial_one(ip) for ip in ips))
        log.info(f"{label} bootstrap: connected to {connected}/{len(ips)} peers")

        # Self-lookup to discover more peers. Skip when the dial phase
        # already populated the routing table to >= k *reachable* entries:
        # the walk is opportunistic keyspace coverage at that point, not
        # a correctness requirement, and walking against responders whose
        # routing tables include unreachable peers can pay one
        # ``DIAL_TIMEOUT`` per ghost in alpha-sized batches before any
        # subsequent operation (GET_VALUE, PUT_VALUE) can run. Counting
        # connected-only avoids the corner case where the table holds
        # ``k`` stale ``connected=False`` entries from a prior session;
        # those should not suppress a fresh walk.
        rt_size = self.routing_table.size(connected_only=True)
        if 0 < rt_size < self._k:
            await self._iterative_find_node(self.peer_id)
