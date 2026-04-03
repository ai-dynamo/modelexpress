# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for routing table, XOR distance, K-bucket eviction, and bucket refresh.

Covers:
- XOR distance arithmetic (identity, symmetry, single bit, padding)
- Common prefix length computation
- Leading zeros helper
- KBucket add/update/evict/remove with staleness and liveness callbacks
- RoutingTable self-rejection, closest_peers ordering and limits
- KadHandler record limits and expiry
- _random_key_for_bucket bit manipulation correctness
- Bucket refresh discovering distant peers
"""

import asyncio
import time

import pytest

from mx_libp2p.routing import (
    RoutingTable,
    KBucket,
    PeerEntry,
    xor_distance,
    _common_prefix_length,
    _leading_zeros,
    K,
    STALE_PEER_TIMEOUT,
)
from mx_libp2p.kad_handler import KadHandler, MAX_RECORDS, StoredRecord
from mx_libp2p.crypto import Ed25519Identity, _base58btc_encode
from mx_libp2p.dht import DhtNode


# --- XOR distance ---

def test_xor_distance_identical():
    a = b"\x00" * 32
    assert xor_distance(a, a) == 0


def test_xor_distance_opposite():
    a = b"\x00" * 32
    b = b"\xff" * 32
    assert xor_distance(a, b) == (1 << 256) - 1


def test_xor_distance_single_bit():
    a = b"\x00" * 32
    b = b"\x00" * 31 + b"\x01"
    assert xor_distance(a, b) == 1


def test_xor_distance_symmetry():
    a = b"\x12\x34" + b"\x00" * 30
    b = b"\x56\x78" + b"\x00" * 30
    assert xor_distance(a, b) == xor_distance(b, a)


def test_xor_distance_different_lengths():
    # Shorter is padded with zeros
    a = b"\x01"
    b = b"\x01\x00"
    assert xor_distance(a, b) == 0


# --- Common prefix length ---

def test_common_prefix_length_identical():
    a = b"\x00" * 32
    assert _common_prefix_length(a, a) == 256


def test_common_prefix_length_opposite():
    a = b"\x00" * 32
    b = b"\x80" + b"\x00" * 31
    assert _common_prefix_length(a, b) == 0


def test_common_prefix_length_one_bit():
    a = b"\x00" * 32
    b = b"\x00" * 31 + b"\x01"
    assert _common_prefix_length(a, b) == 255


def test_common_prefix_length_half():
    a = b"\x00" * 16 + b"\x80" + b"\x00" * 15
    b = b"\x00" * 32
    assert _common_prefix_length(a, b) == 128


# --- Leading zeros ---

def test_leading_zeros():
    assert _leading_zeros(0) == 8
    assert _leading_zeros(0x80) == 0
    assert _leading_zeros(0x40) == 1
    assert _leading_zeros(0x01) == 7
    assert _leading_zeros(0xFF) == 0


# --- KBucket ---

def test_kbucket_add_and_update():
    bucket = KBucket(k=3)
    assert bucket.add_or_update(b"\x01", [])
    assert bucket.add_or_update(b"\x02", [])
    assert bucket.add_or_update(b"\x03", [])
    assert len(bucket) == 3

    # Updating existing peer should succeed
    assert bucket.add_or_update(b"\x01", [b"new_addr"])
    assert len(bucket) == 3
    # Peer should be at the tail (most recently seen)
    assert bucket.peers[-1].peer_id == b"\x01"
    assert bucket.peers[-1].addrs == [b"new_addr"]


def test_kbucket_full_rejects_newcomer():
    bucket = KBucket(k=2)
    assert bucket.add_or_update(b"\x01", [])
    assert bucket.add_or_update(b"\x02", [])
    # Bucket full, LRU is recent -> reject newcomer
    assert not bucket.add_or_update(b"\x03", [])
    assert len(bucket) == 2


def test_kbucket_evicts_stale_peer(monkeypatch):
    bucket = KBucket(k=2)
    assert bucket.add_or_update(b"\x01", [])
    assert bucket.add_or_update(b"\x02", [])

    # Make LRU peer stale by backdating last_seen (extra margin for jitter)
    bucket.peers[0].last_seen = time.monotonic() - STALE_PEER_TIMEOUT * 2

    # Now newcomer should evict the stale LRU
    assert bucket.add_or_update(b"\x03", [])
    assert len(bucket) == 2
    peer_ids = [p.peer_id for p in bucket.peers]
    assert b"\x01" not in peer_ids
    assert b"\x03" in peer_ids


def test_kbucket_liveness_callback():
    alive_peers = {b"\x01"}

    def is_alive(pid):
        return pid in alive_peers

    bucket = KBucket(k=2, is_alive=is_alive)
    assert bucket.add_or_update(b"\x01", [])
    assert bucket.add_or_update(b"\x02", [])

    # LRU is \x01, which is alive -> reject newcomer
    assert not bucket.add_or_update(b"\x03", [])

    # Mark \x01 as dead
    alive_peers.discard(b"\x01")

    # Now newcomer should evict \x01
    # \x02 is now LRU, \x01 was moved to tail by the previous check
    # Actually: after the rejected add, \x01 was moved to tail. So \x02 is LRU.
    # \x02 is not in alive_peers, so it gets evicted.
    assert bucket.add_or_update(b"\x03", [])
    assert len(bucket) == 2


def test_kbucket_remove():
    bucket = KBucket(k=3)
    bucket.add_or_update(b"\x01", [])
    bucket.add_or_update(b"\x02", [])
    assert bucket.remove(b"\x01")
    assert not bucket.remove(b"\x99")  # not found
    assert len(bucket) == 1


# --- RoutingTable ---

def test_routing_table_no_self():
    local_id = b"\x00" * 32
    rt = RoutingTable(local_id)
    assert not rt.add_or_update(local_id, [])
    assert rt.size() == 0


def test_routing_table_closest_peers():
    local_id = b"\x00" * 32
    rt = RoutingTable(local_id)

    # Add some peers at known distances
    peer1 = b"\x00" * 31 + b"\x01"  # distance 1
    peer2 = b"\x00" * 31 + b"\x02"  # distance 2
    peer3 = b"\x80" + b"\x00" * 31  # distance 2^255

    rt.add_or_update(peer1, [])
    rt.add_or_update(peer2, [])
    rt.add_or_update(peer3, [])

    # Closest to local_id should be peer1, then peer2, then peer3
    target = b"\x00" * 32
    closest = rt.closest_peers(target, 3)
    assert len(closest) == 3
    assert closest[0].peer_id == peer1
    assert closest[1].peer_id == peer2
    assert closest[2].peer_id == peer3


def test_routing_table_closest_peers_limited():
    local_id = b"\x00" * 32
    rt = RoutingTable(local_id)
    for i in range(10):
        pid = b"\x00" * 31 + bytes([i + 1])
        rt.add_or_update(pid, [])

    closest = rt.closest_peers(local_id, 3)
    assert len(closest) == 3


# --- KadHandler record limits ---

def test_kad_handler_max_records():
    rt = RoutingTable(b"\x00" * 32)
    handler = KadHandler(rt, max_records=3)

    assert handler.put_local(b"k1", b"v1")
    assert handler.put_local(b"k2", b"v2")
    assert handler.put_local(b"k3", b"v3")
    # Store is full, new key should evict the furthest record (not reject)
    assert handler.put_local(b"k4", b"v4")
    assert len(handler.records) == 3  # still at max
    assert handler.get_local(b"k4") is not None  # new record stored
    # Updating existing key should still work
    assert handler.put_local(b"k1", b"v1_updated")


def test_kad_handler_record_expiry():
    rt = RoutingTable(b"\x00" * 32)
    handler = KadHandler(rt)

    handler.put_local(b"k1", b"v1")
    handler.put_local(b"k2", b"v2")

    # Backdate one record using StoredRecord with a monotonic timestamp in the past
    handler._records[b"k1"] = StoredRecord(b"v1", time.monotonic() - 100)

    removed = handler.remove_expired(50)
    assert removed == 1
    assert handler.get_local(b"k1") is None
    assert handler.get_local(b"k2") is not None


# --- _random_key_for_bucket bit manipulation ---


def test_random_key_for_bucket_cpl():
    """_random_key_for_bucket(cpl) must produce a key whose CPL with
    our peer ID is exactly `cpl`."""
    identity = Ed25519Identity.generate()
    node = DhtNode(identity=identity)
    node._listen_addr = ("127.0.0.1", 0)

    for cpl in [0, 1, 7, 8, 15, 16, 31, 63, 127]:
        key = node._random_key_for_bucket(cpl)
        actual_cpl = _common_prefix_length(node.peer_id, key)
        assert actual_cpl == cpl, (
            f"_random_key_for_bucket({cpl}) produced key with CPL={actual_cpl}"
        )


def test_random_key_for_bucket_randomness():
    """Multiple calls should produce different keys (not deterministic)."""
    identity = Ed25519Identity.generate()
    node = DhtNode(identity=identity)
    node._listen_addr = ("127.0.0.1", 0)

    keys = {node._random_key_for_bucket(10) for _ in range(20)}
    assert len(keys) > 1, "random keys should not all be identical"


# --- Bucket refresh ---


def _node_multiaddr(node: DhtNode) -> str:
    host, port = node.listen_addr
    return f"/ip4/{host}/tcp/{port}/p2p/{_base58btc_encode(node.peer_id)}"


async def test_bucket_refresh_discovers_distant_peers():
    """Bucket refresh should discover peers in distant buckets that a
    self-lookup alone wouldn't find."""
    node_a = DhtNode(record_ttl=300)
    await node_a.start("127.0.0.1", 0)
    addr_a = _node_multiaddr(node_a)

    nodes = [node_a]
    for _ in range(5):
        n = DhtNode(record_ttl=300)
        await n.start("127.0.0.1", 0, bootstrap_peers=[addr_a])
        nodes.append(n)

    await asyncio.sleep(0.5)

    try:
        size_before = node_a.routing_table.size()

        await node_a._refresh_buckets()

        size_after = node_a.routing_table.size()
        assert size_after >= size_before, (
            f"bucket refresh should not lose peers: {size_before} -> {size_after}"
        )

        assert size_after >= 5, (
            f"after refresh, should know all 5 peers, have {size_after}"
        )
    finally:
        for n in nodes:
            await n.stop()
