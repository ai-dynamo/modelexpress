# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Iterative DHT lookup engine (mixin).

Implements iterative FIND_NODE and GET_VALUE with adaptive parallelism
(stall detection), plus single-peer PUT_VALUE.
"""

import asyncio
import logging

from .kademlia import kad_find_node, kad_get_value, kad_put_value
from .routing import kad_key, xor_distance

log = logging.getLogger(__name__)

# Max iterative lookup work, expressed as a budget of total per-peer
# task dispatches. v0.3.1 switched from a per-round "drain alpha tasks
# before advancing" model to a continuous alpha-refill walk; the bound
# carries over as ``MAX_LOOKUP_ROUNDS * alpha`` so the worst-case work
# stays comparable to the prior round-based limit.
MAX_LOOKUP_ROUNDS = 10
# When a lookup stalls (no new closer peers), increase parallelism by this factor
STALL_PARALLELISM_BOOST = 2


class QueryMixin:
    """Iterative Kademlia lookups and single-peer RPCs for DhtNode."""

    def _is_peer_reachable(self, peer_id: bytes) -> bool:
        """Check if a peer is worth dialing. Returns False for peers marked
        disconnected in the routing table (they'd just timeout)."""
        entry = self.routing_table.find(peer_id)
        if entry is None:
            return True  # unknown peer, worth trying
        return entry.connected

    async def _iterative_find_node(self, target: bytes) -> list[tuple[bytes, list[bytes]]]:
        """Iterative FIND_NODE lookup with stall detection.

        Returns list of (peer_id, addrs) for the k closest peers found.

        v0.3.1 (post-CodeRabbit): walks the candidate set with a
        continuous alpha-refill window rather than draining a per-round
        batch. As soon as a peer task completes, a new task is launched
        from the (possibly newly extended) candidate frontier, up to
        ``parallelism`` concurrent in-flight tasks. This means a slow
        ghost peer no longer barriers the round - the fast peers'
        responses immediately drive the next hop. Stall detection
        terminates the walk after ``2 * alpha`` task completions with no
        new closer peer; the first stall boosts parallelism by
        ``STALL_PARALLELISM_BOOST`` before the second-stall break.
        """
        # Seed with locally known closest reachable peers. Filtering to
        # ``connected_only=True`` here mirrors the responder-side filter
        # in ``KadHandler._closest_peers_encoded``: the same defect (dead
        # routing-table entries shadowing live ones in lookup paths)
        # exists on both ends of the request. Without this, if the
        # closest ``k`` entries in the local table are all marked
        # unreachable, ``_is_peer_reachable`` would skip every one of
        # them in ``fill()`` and the lookup would never reach connected
        # candidates that sit just past the closest-``k`` window. The
        # filter only excludes peers the local node has already marked
        # disconnected; new entries learned mid-lookup from inbound
        # responses default to ``connected=True`` and are still tried
        # once before failure marks them.
        closest = self.routing_table.closest_peers(target, self._k, connected_only=True)
        if not closest:
            return []

        queried: set[bytes] = set()
        queried.add(self.peer_id)  # don't query ourselves
        peer_map: dict[bytes, list[bytes]] = {}  # peer_id -> addrs
        # Cache kad_id alongside peer_map so the candidate sort and the
        # stall-detection distance comparison reuse sha256(peer_id)
        # instead of recomputing per peer per task completion. Populated
        # when peers are seeded AND when closer peers arrive in iterative
        # responses.
        peer_kad: dict[bytes, bytes] = {}

        for entry in closest:
            peer_map[entry.peer_id] = entry.addrs
            peer_kad[entry.peer_id] = entry.kad_id

        target_kad = kad_key(target)
        best_distance = min(
            (xor_distance(peer_kad[p], target_kad) for p in peer_map),
            default=None,
        )
        parallelism = self._alpha
        completions_since_progress = 0
        stall_threshold = self._alpha * 2
        stalled_once = False
        # ``MAX_LOOKUP_ROUNDS * alpha`` total per-peer dispatches. The
        # ``+ 1`` accounts for ``self.peer_id`` already being in
        # ``queried`` so the budget reflects actual peer queries.
        max_total_queries = MAX_LOOKUP_ROUNDS * self._alpha + 1
        suppress_fill = False  # set on stall/cap to drain in_flight without refill

        in_flight: set[asyncio.Task] = set()

        def fill() -> None:
            """Top up ``in_flight`` from the closest unqueried reachable
            peers until parallelism is reached or candidates exhaust."""
            if len(in_flight) >= parallelism:
                return
            candidates = sorted(
                peer_map.keys(),
                key=lambda p: xor_distance(peer_kad[p], target_kad),
            )
            for peer_id in candidates:
                if len(in_flight) >= parallelism:
                    return
                if len(queried) >= max_total_queries:
                    return
                if peer_id in queried or not self._is_peer_reachable(peer_id):
                    continue
                queried.add(peer_id)
                t = asyncio.create_task(
                    self._find_node_single(peer_id, peer_map[peer_id], target)
                )
                in_flight.add(t)

        fill()
        try:
            while in_flight:
                done, _ = await asyncio.wait(
                    in_flight, return_when=asyncio.FIRST_COMPLETED
                )
                in_flight.difference_update(done)

                progressed_this_batch = False
                for task in done:
                    completions_since_progress += 1
                    if task.cancelled():
                        continue
                    exc = task.exception()
                    if exc is not None:
                        continue
                    for pid, addrs in task.result():
                        if pid not in peer_map and pid != self.peer_id:
                            peer_map[pid] = addrs
                            peer_kad[pid] = kad_key(pid)
                            self.routing_table.add_or_update(pid, addrs)
                            self.peer_store.add_addrs(pid, addrs)
                            d = xor_distance(peer_kad[pid], target_kad)
                            if best_distance is None or d < best_distance:
                                best_distance = d
                                progressed_this_batch = True

                if progressed_this_batch:
                    completions_since_progress = 0
                    parallelism = self._alpha
                    stalled_once = False
                    suppress_fill = False
                elif completions_since_progress >= stall_threshold:
                    if not stalled_once:
                        # First stall: boost parallelism, reset counter,
                        # let the wider window pull in more candidates.
                        stalled_once = True
                        parallelism = min(
                            self._alpha * STALL_PARALLELISM_BOOST, self._k
                        )
                        completions_since_progress = 0
                    else:
                        # Second stall after boost: stop refilling. We
                        # still drain whatever is in_flight so a slow
                        # but productive peer still in the window can
                        # surface its result before we exit (its new
                        # closer peers may flip ``progressed_this_batch``
                        # back on, in which case the lookup resumes).
                        suppress_fill = True

                if len(queried) >= max_total_queries:
                    suppress_fill = True

                if not suppress_fill:
                    fill()
        finally:
            # Propagate parent cancellation (or natural termination) to
            # any peer tasks still in flight, then drain via gather so
            # CancelledError and any sibling exceptions are retrieved
            # cleanly. ``asyncio.gather`` previously did this implicitly;
            # ``asyncio.wait`` does not.
            for task in in_flight:
                if not task.done():
                    task.cancel()
            if in_flight:
                await asyncio.gather(*in_flight, return_exceptions=True)

        # Return k closest
        sorted_peers = sorted(peer_map.keys(), key=lambda p: xor_distance(peer_kad[p], target_kad))
        return [(p, peer_map[p]) for p in sorted_peers[:self._k]]

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

        v0.3.1 (post-CodeRabbit): mirrors ``_iterative_find_node``'s
        continuous alpha-refill walk - in_flight peer queries are
        topped up to ``parallelism`` as tasks complete, so a slow ghost
        peer never barriers the next hop. As soon as any task surfaces
        a value, the loop bails immediately and cancels the rest.
        Stall detection terminates the walk after ``2 * alpha`` task
        completions with no new closer peer, with a parallelism boost
        on the first stall to widen the query before giving up.
        """
        # See the rationale on the symmetric seed in
        # ``_iterative_find_node``: filter to reachable peers so the
        # closest-``k`` window is not exhausted by stale entries before
        # any connected candidate gets a chance to fire.
        closest = self.routing_table.closest_peers(key, self._k, connected_only=True)
        if not closest:
            return None

        queried: set[bytes] = set()
        queried.add(self.peer_id)
        peer_map: dict[bytes, list[bytes]] = {}
        # Cache kad_id alongside peer_map (see _iterative_find_node).
        peer_kad: dict[bytes, bytes] = {}

        for entry in closest:
            peer_map[entry.peer_id] = entry.addrs
            peer_kad[entry.peer_id] = entry.kad_id

        key_kad = kad_key(key)
        best_distance = min(
            (xor_distance(peer_kad[p], key_kad) for p in peer_map),
            default=None,
        )
        parallelism = self._alpha
        completions_since_progress = 0
        stall_threshold = self._alpha * 2
        stalled_once = False
        # ``+ 1`` accounts for ``self.peer_id`` in ``queried``.
        max_total_queries = MAX_LOOKUP_ROUNDS * self._alpha + 1
        suppress_fill = False
        value_found: bytes | None = None
        value_ttl: float | None = None

        in_flight: set[asyncio.Task] = set()

        def fill() -> None:
            if len(in_flight) >= parallelism:
                return
            candidates = sorted(
                peer_map.keys(),
                key=lambda p: xor_distance(peer_kad[p], key_kad),
            )
            for peer_id in candidates:
                if len(in_flight) >= parallelism:
                    return
                if len(queried) >= max_total_queries:
                    return
                if peer_id in queried or not self._is_peer_reachable(peer_id):
                    continue
                queried.add(peer_id)
                t = asyncio.create_task(
                    self._get_value_single(peer_id, peer_map[peer_id], key)
                )
                in_flight.add(t)

        fill()
        try:
            while in_flight:
                done, _ = await asyncio.wait(
                    in_flight, return_when=asyncio.FIRST_COMPLETED
                )
                in_flight.difference_update(done)

                progressed_this_batch = False
                # Inspect every ``done`` task even after a value is
                # captured, so any sibling task's exception is retrieved
                # in this loop instead of leaking past the early exit.
                for task in done:
                    completions_since_progress += 1
                    if task.cancelled():
                        continue
                    exc = task.exception()
                    if exc is not None:
                        continue
                    value, ttl, new_peers = task.result()
                    if value is not None:
                        if value_found is None:
                            value_found = value
                            value_ttl = ttl
                        # Keep iterating to retrieve sibling exceptions;
                        # don't continue integrating new peers though,
                        # we're about to bail.
                        continue
                    if value_found is not None:
                        # Already captured a value earlier in this batch;
                        # skip integrating new peers from siblings since
                        # we won't act on them.
                        continue
                    for pid, addrs in new_peers:
                        if pid not in peer_map and pid != self.peer_id:
                            peer_map[pid] = addrs
                            peer_kad[pid] = kad_key(pid)
                            self.routing_table.add_or_update(pid, addrs)
                            self.peer_store.add_addrs(pid, addrs)
                            d = xor_distance(peer_kad[pid], key_kad)
                            if best_distance is None or d < best_distance:
                                best_distance = d
                                progressed_this_batch = True

                if value_found is not None:
                    break

                if progressed_this_batch:
                    completions_since_progress = 0
                    parallelism = self._alpha
                    stalled_once = False
                    suppress_fill = False
                elif completions_since_progress >= stall_threshold:
                    if not stalled_once:
                        stalled_once = True
                        parallelism = min(
                            self._alpha * STALL_PARALLELISM_BOOST, self._k
                        )
                        completions_since_progress = 0
                    else:
                        # Second stall: stop refilling but keep draining
                        # in_flight so a slow but productive task (one
                        # holding the value, or one bringing a new
                        # closer peer that resumes the walk) still gets
                        # harvested before we exit.
                        suppress_fill = True

                if len(queried) >= max_total_queries:
                    suppress_fill = True

                if not suppress_fill:
                    fill()
        finally:
            for task in in_flight:
                if not task.done():
                    task.cancel()
            if in_flight:
                await asyncio.gather(*in_flight, return_exceptions=True)

        if value_found is not None:
            self.kad_handler.put_local(key, value_found, ttl=value_ttl)
            return value_found
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
            self.routing_table.mark_disconnected(peer_id)
            return None, None, []

    async def _put_to_peer(
        self, peer_id: bytes, addrs: list[bytes], key: bytes, value: bytes,
        publisher: bytes | None = None, ttl_secs: int | None = None,
    ) -> bool:
        """Send PUT_VALUE to a single peer. Skips peers marked disconnected."""
        if not self._is_peer_reachable(peer_id):
            return False
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
            self.routing_table.mark_disconnected(peer_id)
            log.debug(f"put_value to {peer_id.hex()[:16]}... failed: {e}", exc_info=True)
            return False
