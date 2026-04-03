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

from .connection import dial, IDENTIFY_PROTOCOL, IDENTIFY_PUSH_PROTOCOL
from .kademlia import KADEMLIA_PROTOCOL
from .multiaddr import (
    encode_multiaddr_ip_tcp_p2p,
    parse_multiaddr_string,
)
from .routing import K
from .slurm import expand_hostlist
from .dht_utils import _filter_routable_addrs, _parse_peer_multiaddr

log = logging.getLogger(__name__)


class BootstrapMixin:
    """Bootstrap providers for DhtNode."""

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
                conn = await asyncio.wait_for(
                    self.peer_store.get_or_dial(peer_id),
                    timeout=self.dial_timeout,
                )
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
        own IPs, dials up to K peers, learns peer IDs via Noise handshake,
        and performs Identify exchange + self-lookup.
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
            if connected >= K:
                return False
            async with sem:
                if connected >= K:
                    return False
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

                    async with lock:
                        connected += 1
                    log.info(f"{label} bootstrap: connected to {ip}:{port} (peer {peer_id.hex()[:16]}...)")
                    return True
                except Exception as e:
                    log.debug(f"{label} bootstrap: failed to dial {ip}:{port}: {e}")
                    return False

        await asyncio.gather(*(dial_one(ip) for ip in ips))
        log.info(f"{label} bootstrap: connected to {connected}/{len(ips)} peers")

        # Self-lookup to discover more peers
        if self.routing_table.size() > 0:
            await self._iterative_find_node(self.peer_id)
