# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TTL, expiry, republish lifecycle, record deletion, and replication tests.

Covers:
- Republish loop refreshing record timestamps
- Records expiring without republish
- Per-record TTL respected (different TTLs expire independently)
- Per-record TTL propagation to remote peers via DhtNode.put()
- Republish propagation to new peers
- Replication loop pushing non-originated records
- Republish loop cancellation on node stop
- Record deletion (remove()) from local store and originated records
- Remove stops republish cycle
- Remove works for non-originated records
- Single-node TTL auto-expiry
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
        record_ttl=5.0,
        republish_interval=1.0,
    )
    await node.start("127.0.0.1", 0)

    try:
        key = b"/mx/repub/worker/0"
        value = b'{"test":"republish"}'
        await node.put(key, value)

        local = node.kad_handler.get_local(key)
        assert local is not None
        original_ts = local.timestamp

        await asyncio.sleep(1.5)

        local_after = node.kad_handler.get_local(key)
        assert local_after is not None, "record should survive past one republish cycle"
        assert local_after.timestamp > original_ts, "timestamp should be refreshed"
    finally:
        await node.stop()


async def test_record_expires_without_republish():
    """A record that is NOT in originated_records should expire after TTL."""
    node = DhtNode(
        record_ttl=1.0,
        republish_interval=60.0,
    )
    await node.start("127.0.0.1", 0)

    try:
        key = b"/mx/expire/worker/0"
        node.kad_handler.put_local(key, b'{"will":"expire"}')
        assert node.kad_handler.get_local(key) is not None

        rec = node.kad_handler._records[key]
        node.kad_handler._records[key] = StoredRecord(
            rec.value, time.monotonic() - 2.0, rec.publisher, rec.ttl
        )

        removed = node.kad_handler.remove_expired(1.0)
        assert removed == 1
        assert node.kad_handler.get_local(key) is None
    finally:
        await node.stop()


async def test_per_record_ttl_respected():
    """Records with different TTLs should expire at different times."""
    node = DhtNode(
        record_ttl=60.0,
        republish_interval=60.0,
    )
    await node.start("127.0.0.1", 0)

    try:
        key_short = b"/mx/short/status/0"
        key_long = b"/mx/long/worker/0"

        node.kad_handler.put_local(key_short, b'{"ttl":"short"}', ttl=1.0)
        node.kad_handler.put_local(key_long, b'{"ttl":"long"}', ttl=60.0)

        now = time.monotonic()
        for key in [key_short, key_long]:
            rec = node.kad_handler._records[key]
            node.kad_handler._records[key] = StoredRecord(
                rec.value, now - 2.0, rec.publisher, rec.ttl
            )

        removed = node.kad_handler.remove_expired(60.0)
        assert removed == 1
        assert node.kad_handler.get_local(key_short) is None
        assert node.kad_handler.get_local(key_long) is not None
    finally:
        await node.stop()


async def test_per_record_ttl_propagates_to_remote():
    """put(key, value, ttl=X) should store with the given TTL both
    locally and on remote peers. Remote copies should expire at the
    per-record TTL, not the node default."""
    node_a = DhtNode(record_ttl=300.0, republish_interval=3600)
    await node_a.start("127.0.0.1", 0)
    addr_a = _node_multiaddr(node_a)

    node_b = DhtNode(record_ttl=300.0, republish_interval=3600)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[addr_a])
    await asyncio.sleep(0.3)

    try:
        key = b"/mx/ttl-test/status/0"
        value = b'{"status":"READY"}'

        await node_a.put(key, value, ttl=1.0)

        remote_rec = node_b.kad_handler.get_local(key)
        assert remote_rec is not None, "remote peer should have the record"
        assert remote_rec.value == value

        local_rec = node_a.kad_handler.get_local(key)
        assert local_rec is not None
        assert local_rec.ttl == 1.0

        await asyncio.sleep(1.5)

        node_b.kad_handler.remove_expired(300.0)
        assert node_b.kad_handler.get_local(key) is None, (
            "record on remote peer should have expired via per-record TTL"
        )
    finally:
        await node_a.stop()
        await node_b.stop()


async def test_republish_propagates_to_new_peer():
    """When a new peer joins, the republish loop should propagate
    originated records to it."""
    node_a = DhtNode(
        record_ttl=300.0,
        republish_interval=1.0,
    )
    await node_a.start("127.0.0.1", 0)

    try:
        key = b"/mx/propagate/worker/0"
        value = b'{"test":"propagate"}'
        await node_a.put(key, value)

        await asyncio.sleep(0.5)
        node_b = DhtNode(record_ttl=300.0)
        await node_b.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node_a)])
        await asyncio.sleep(0.3)

        try:
            await asyncio.sleep(1.5)

            result = await node_b.get(key)
            assert result == value, f"record should propagate to new peer, got {result!r}"
        finally:
            await node_b.stop()
    finally:
        await node_a.stop()


async def test_replication_loop_pushes_non_originated_records():
    """The replication loop (every 4th republish cycle) should push records
    received from other peers to the K closest nodes."""
    node_a = DhtNode(record_ttl=300, republish_interval=0.5)
    await node_a.start("127.0.0.1", 0)
    addr_a = _node_multiaddr(node_a)

    node_b = DhtNode(record_ttl=300, republish_interval=0.5)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[addr_a])
    await asyncio.sleep(0.3)

    try:
        key = b"/mx/repl-cycle/worker/0"
        value = b'{"test":"replication-cycle"}'
        await node_a.put(key, value)

        assert node_b.kad_handler.get_local(key) is not None

        node_c = DhtNode(record_ttl=300, republish_interval=0.5)
        await node_c.start("127.0.0.1", 0, bootstrap_peers=[addr_a])
        await asyncio.sleep(0.3)

        try:
            await asyncio.sleep(3.0)

            result = await node_c.get(key)
            assert result == value, f"replication should propagate to node_c, got {result!r}"
        finally:
            await node_c.stop()
    finally:
        await node_a.stop()
        await node_b.stop()


async def test_republish_loop_stops_after_node_stop():
    """The republish loop should be cancelled when the node stops."""
    node = DhtNode(record_ttl=300, republish_interval=0.5)
    await node.start("127.0.0.1", 0)

    await node.put(b"key", b"value")
    assert node._republish_task is not None
    assert not node._republish_task.done()

    await node.stop()

    assert node._republish_task.done()


async def test_record_ttl_expiry_single_node():
    """Records should expire after TTL on a single node."""
    node = DhtNode(record_ttl=0.1)
    await node.start("127.0.0.1", 0)

    await node.put(b"key", b"value")

    assert await node.get(b"key") == b"value"

    await asyncio.sleep(0.2)

    result = await node.get(b"key")
    assert result is None

    await node.stop()


# --- Record deletion ---


async def test_remove_originated_record():
    """remove() should delete from local store and stop republishing."""
    node = DhtNode(record_ttl=300, republish_interval=0.5)
    await node.start("127.0.0.1", 0)

    try:
        key = b"/mx/del/worker/0"
        value = b'{"test":"delete"}'
        await node.put(key, value)

        assert await node.get(key) == value
        assert key in node._originated_records

        existed = node.remove(key)
        assert existed is True

        assert node.kad_handler.get_local(key) is None
        assert key not in node._originated_records

        assert await node.get(key) is None

        assert node.remove(key) is False
    finally:
        await node.stop()


async def test_remove_stops_republish():
    """After remove(), the record should NOT be republished in the next cycle."""
    node_a = DhtNode(record_ttl=300, republish_interval=0.5)
    await node_a.start("127.0.0.1", 0)
    addr_a = _node_multiaddr(node_a)

    node_b = DhtNode(record_ttl=300)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[addr_a])
    await asyncio.sleep(0.3)

    try:
        key = b"/mx/del-repub/worker/0"
        value = b'{"test":"delete-republish"}'
        await node_a.put(key, value)

        assert node_b.kad_handler.get_local(key) is not None

        node_a.remove(key)

        if key in node_b.kad_handler.records:
            del node_b.kad_handler.records[key]

        await asyncio.sleep(1.0)

        assert node_b.kad_handler.get_local(key) is None, (
            "record should not be republished after remove()"
        )
    finally:
        await node_a.stop()
        await node_b.stop()


async def test_remove_non_originated_record():
    """remove() should also work for records received from other peers."""
    node_a = DhtNode(record_ttl=300)
    await node_a.start("127.0.0.1", 0)
    addr_a = _node_multiaddr(node_a)

    node_b = DhtNode(record_ttl=300)
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[addr_a])
    await asyncio.sleep(0.3)

    try:
        key = b"/mx/del-remote/worker/0"
        value = b'{"test":"delete-remote"}'
        await node_a.put(key, value)
        assert node_b.kad_handler.get_local(key) is not None

        existed = node_b.remove(key)
        assert existed is True
        assert node_b.kad_handler.get_local(key) is None
    finally:
        await node_a.stop()
        await node_b.stop()
