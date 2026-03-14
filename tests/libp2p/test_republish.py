# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Republish loop correctness tests.

Validates that records survive past their TTL via the republish mechanism
and that the replication loop pushes records to new nodes.
"""

import asyncio
import time

from mx_libp2p.crypto import _base58btc_encode
from mx_libp2p.dht import DhtNode
from mx_libp2p.kad_handler import StoredRecord


def _node_multiaddr(node: DhtNode) -> str:
    host, port = node.listen_addr
    return f"/ip4/{host}/tcp/{port}/p2p/{_base58btc_encode(node.peer_id)}"


async def test_republish_refreshes_timestamp():
    """Originated records should be re-PUT during the republish loop,
    refreshing their timestamp so they don't expire."""
    node = DhtNode(
        record_ttl=5.0,          # records expire after 5s
        republish_interval=1.0,  # republish every 1s
    )
    await node.start("127.0.0.1", 0)

    try:
        key = b"/mx/repub/worker/0"
        value = b'{"test":"republish"}'
        await node.put(key, value)

        # Record should exist
        local = node.kad_handler.get_local(key)
        assert local is not None
        original_ts = local.timestamp

        # Wait for republish to fire (1s interval + margin)
        await asyncio.sleep(1.5)

        # Trigger expiry check - but record should have been refreshed
        # by the republish loop's put() call
        local_after = node.kad_handler.get_local(key)
        assert local_after is not None, "record should survive past one republish cycle"
        assert local_after.timestamp > original_ts, "timestamp should be refreshed"
    finally:
        await node.stop()


async def test_record_expires_without_republish():
    """A record that is NOT in originated_records should expire after TTL."""
    node = DhtNode(
        record_ttl=1.0,
        republish_interval=60.0,  # won't fire during this test
    )
    await node.start("127.0.0.1", 0)

    try:
        key = b"/mx/expire/worker/0"
        # Put directly into the store WITHOUT tracking as originated
        node.kad_handler.put_local(key, b'{"will":"expire"}')
        assert node.kad_handler.get_local(key) is not None

        # Manually backdate the record so it's past TTL
        rec = node.kad_handler._records[key]
        node.kad_handler._records[key] = StoredRecord(
            rec.value, time.monotonic() - 2.0, rec.publisher, rec.ttl
        )

        # Expire check should remove it
        removed = node.kad_handler.remove_expired(1.0)
        assert removed == 1
        assert node.kad_handler.get_local(key) is None
    finally:
        await node.stop()


async def test_per_record_ttl_respected():
    """Records with different TTLs should expire at different times."""
    node = DhtNode(
        record_ttl=60.0,         # default TTL
        republish_interval=60.0,  # won't fire
    )
    await node.start("127.0.0.1", 0)

    try:
        # Store two records with different TTLs
        key_short = b"/mx/short/status/0"
        key_long = b"/mx/long/worker/0"

        node.kad_handler.put_local(key_short, b'{"ttl":"short"}', ttl=1.0)
        node.kad_handler.put_local(key_long, b'{"ttl":"long"}', ttl=60.0)

        # Backdate both by 2 seconds
        now = time.monotonic()
        for key in [key_short, key_long]:
            rec = node.kad_handler._records[key]
            node.kad_handler._records[key] = StoredRecord(
                rec.value, now - 2.0, rec.publisher, rec.ttl
            )

        # Expire with default TTL of 60s
        removed = node.kad_handler.remove_expired(60.0)
        assert removed == 1  # only the short-TTL record should expire
        assert node.kad_handler.get_local(key_short) is None
        assert node.kad_handler.get_local(key_long) is not None
    finally:
        await node.stop()


async def test_republish_propagates_to_new_peer():
    """When a new peer joins, the republish loop should propagate
    originated records to it."""
    node_a = DhtNode(
        record_ttl=300.0,
        republish_interval=1.0,  # fast republish
    )
    await node_a.start("127.0.0.1", 0)

    try:
        # Store a record on node_a
        key = b"/mx/propagate/worker/0"
        value = b'{"test":"propagate"}'
        await node_a.put(key, value)

        # Wait a moment, then join a new peer
        await asyncio.sleep(0.5)
        node_b = DhtNode(record_ttl=300.0)
        await node_b.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node_a)])
        await asyncio.sleep(0.3)

        try:
            # Wait for republish to fire and push the record to node_b
            await asyncio.sleep(1.5)

            # node_b should now have the record (either from iterative GET
            # or from node_a's republish cycle)
            result = await node_b.get(key)
            assert result == value, f"record should propagate to new peer, got {result!r}"
        finally:
            await node_b.stop()
    finally:
        await node_a.stop()
