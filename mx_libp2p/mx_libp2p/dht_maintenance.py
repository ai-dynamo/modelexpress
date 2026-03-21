# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Periodic maintenance loops for DhtNode (mixin).

Handles periodic re-bootstrap, bucket refresh, record republishing,
replication, and dead peer pruning.
"""

import asyncio
import logging
import os

from .routing import K

log = logging.getLogger(__name__)

# Periodic bootstrap interval: 5 minutes (matches rust-libp2p)
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
                    if self._bootstrap_hostlist:
                        await self.bootstrap_from_hostlist(self._bootstrap_hostlist, self._bootstrap_dns_port)
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
