# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Iterative DHT lookup engine (mixin).

Implements iterative FIND_NODE and GET_VALUE with adaptive parallelism
(stall detection), plus single-peer PUT_VALUE.
"""

import asyncio
import logging

from .routing import K, ALPHA, xor_distance
from .kademlia import kad_find_node, kad_get_value, kad_put_value

log = logging.getLogger(__name__)

# Max rounds for iterative lookups
MAX_LOOKUP_ROUNDS = 10
# When a lookup stalls (no new closer peers), increase parallelism by this factor
STALL_PARALLELISM_BOOST = 2


class QueryMixin:
    """Iterative Kademlia lookups and single-peer RPCs for DhtNode."""

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
            self.routing_table.mark_disconnected(peer_id)
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
