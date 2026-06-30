# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Periodic maintenance loops for DhtNode (mixin).

Handles periodic re-bootstrap, bucket refresh, record republishing,
replication, and dead peer pruning.
"""

import asyncio
import logging
import os

log = logging.getLogger(__name__)

# Default periodic bootstrap interval: 5 minutes (matches rust-libp2p).
# Used as the default value of DhtNode.start(bootstrap_interval=...);
# pass a different float to override per-node, or None to disable the
# periodic loop entirely.
BOOTSTRAP_INTERVAL = 5 * 60
# Replication happens every N republish cycles (~4 hours at default interval)
REPLICATION_CYCLE_INTERVAL = 4
# Max concurrent background queries for republish/replication
MAX_CONCURRENT_BACKGROUND_QUERIES = 10


class MaintenanceMixin:
    """Periodic maintenance loops for DhtNode."""

    def _prune_dead_peers(self) -> None:
        """Remove peers with dead connections from the routing table."""
        for entry in self.routing_table.all_peers():
            conn = self.peer_store.get_connection(entry.peer_id)
            if conn is not None and not conn.is_alive:
                self.routing_table.remove(entry.peer_id)
            # Also remove peers marked disconnected with no active connection.
            # These were flagged by failed dials or explicit disconnect signals.
            elif not entry.connected and conn is None:
                self.routing_table.remove(entry.peer_id)

    def _quick_prune(self) -> None:
        """Remove routing table peers that are marked disconnected with no active connection.

        Lighter than _prune_dead_peers - no connection object inspection needed.
        Called during periodic bootstrap to keep the routing table clean between
        the longer republish cycles.
        """
        for entry in self.routing_table.all_peers():
            if not entry.connected and self.peer_store.get_connection(entry.peer_id) is None:
                self.routing_table.remove(entry.peer_id)

    async def _periodic_bootstrap_loop(self) -> None:
        """Periodically re-bootstrap to discover new peers and refresh buckets.

        Always runs a self-lookup (and re-dials bootstrap peers when sparse).
        This matches rust-libp2p behavior: periodic bootstrap is unconditional,
        not gated on routing table size. The actual work of one cycle is in
        ``_periodic_bootstrap_tick`` so tests can drive it deterministically
        without monkey-patching ``asyncio.sleep``.

        The cadence is read from ``self._bootstrap_interval`` once per
        iteration so a node configured with ``bootstrap_interval=None``
        does nothing (the loop exits immediately on entry). The interval
        is otherwise treated as immutable for the lifetime of the loop:
        runtime mutation is intentionally not supported (matches
        libp2p-kad, which only exposes the cadence at construction).
        """
        if self._bootstrap_interval is None:
            return
        try:
            while True:
                await asyncio.sleep(self._bootstrap_interval)
                await self._periodic_bootstrap_tick()
        except asyncio.CancelledError:
            pass

    async def _periodic_bootstrap_tick(self) -> None:
        """One iteration of the periodic-bootstrap loop body.

        Prunes the routing table, then either re-dials bootstrap peers
        (sparse table) or runs a self-lookup (healthy table), then
        refreshes buckets. Extracted from the loop body so unit tests
        can drive a single cycle without faking ``asyncio.sleep``.

        Inspection API: tests in this repository call this directly to
        assert prune-before-size ordering and bootstrap-decision logic.
        Production code should not call this except via the loop.
        """
        # Prune disconnected peers before sampling table size so the
        # sparse-table check reflects actual reachable peers. Reading
        # size first would race against the prune and skip the
        # recovery cycle when pruning drops us below k.
        self._quick_prune()
        size = self.routing_table.size()

        if size < self._k:
            # Sparse table: re-dial bootstrap peers first
            log.info(
                f"periodic re-bootstrap: routing table has "
                f"{size} peers (< k={self._k}), re-dialing bootstrap peers"
            )
            if self._bootstrap_peers:
                await self.bootstrap(self._bootstrap_peers)
            if self._bootstrap_dns:
                await self.bootstrap_from_dns(self._bootstrap_dns, self._bootstrap_dns_port)
            if self._bootstrap_hostlist:
                await self.bootstrap_from_hostlist(
                    self._bootstrap_hostlist, self._bootstrap_dns_port
                )
            if self._mdns:
                self._mdns.send_query()
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

    def _random_key_for_bucket(self, cpl: int, max_attempts: int = 65536) -> bytes:
        """Generate a random preimage whose Kad-keyspace CPL with our local
        peer ID matches the given bucket index.

        Because the Kad keyspace is reached through SHA-256, we cannot
        construct a preimage with a target keyspace CPL directly: we use
        rejection sampling. The expected attempts to hit bucket ``cpl`` is
        ``2^(cpl+1)``; for typical populated buckets (cpl <= ~12) this is
        cheap. For very deep buckets it may exhaust ``max_attempts``, in
        which case we fall back to a uniformly random key (still useful
        for discovery, just not bucket-targeted).
        """
        from .routing import _common_prefix_length, kad_key

        local_kad = kad_key(self.peer_id)
        for _ in range(max_attempts):
            candidate = os.urandom(32)
            if _common_prefix_length(local_kad, kad_key(candidate)) == cpl:
                return candidate
        # Couldn't hit the target bucket - fall back to a random key.
        log.debug(
            f"bucket refresh: rejection sampling exhausted at cpl={cpl} "
            f"after {max_attempts} attempts, using random key"
        )
        return os.urandom(32)

    async def _republish_loop(self) -> None:
        """Background loop: re-PUT originated records, replicate stored, and expire old ones.

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
                    for _i, r in enumerate(results):
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
