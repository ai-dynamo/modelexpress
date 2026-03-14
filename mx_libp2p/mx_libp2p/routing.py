# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kademlia routing table with XOR distance metric.

Reference: https://github.com/libp2p/specs/blob/master/kad-dht/README.md

The routing table organizes peers into K-buckets based on XOR distance from
the local node's peer ID. Each bucket holds up to K peers (default 20).
Buckets are indexed by the number of leading zero bits in the XOR distance,
giving 256 buckets for SHA-256-based peer IDs.
"""

import time
import logging
from typing import Callable

log = logging.getLogger(__name__)

# Kademlia parameters
K = 20  # max peers per bucket
ALPHA = 3  # parallelism factor for iterative lookups

# Peers not seen within this window are candidates for eviction when the
# bucket is full and a liveness checker is not provided.
STALE_PEER_TIMEOUT = 300.0  # 5 minutes

# How long a pending replacement entry waits before being promoted
# unconditionally. Matches rust-libp2p's default pending_timeout.
PENDING_ENTRY_TIMEOUT = 60.0  # seconds


class PeerEntry:
    """A peer in the routing table."""

    __slots__ = ("peer_id", "addrs", "last_seen")

    def __init__(self, peer_id: bytes, addrs: list[bytes], last_seen: float | None = None):
        self.peer_id = peer_id
        self.addrs = addrs
        self.last_seen = last_seen or time.monotonic()


def xor_distance(a: bytes, b: bytes) -> int:
    """Compute XOR distance between two peer IDs as an integer."""
    # Pad to same length
    max_len = max(len(a), len(b))
    a_padded = a.ljust(max_len, b"\x00")
    b_padded = b.ljust(max_len, b"\x00")
    return int.from_bytes(bytes(x ^ y for x, y in zip(a_padded, b_padded)), "big")


def _common_prefix_length(a: bytes, b: bytes) -> int:
    """Number of leading zero bits in XOR(a, b). This determines the bucket index."""
    max_len = max(len(a), len(b))
    a_padded = a.ljust(max_len, b"\x00")
    b_padded = b.ljust(max_len, b"\x00")
    bits = 0
    for x, y in zip(a_padded, b_padded):
        xor_byte = x ^ y
        if xor_byte == 0:
            bits += 8
        else:
            bits += _leading_zeros(xor_byte)
            break
    return bits


def _leading_zeros(byte: int) -> int:
    """Count leading zero bits in a byte."""
    if byte == 0:
        return 8
    count = 0
    mask = 0x80
    while (byte & mask) == 0:
        count += 1
        mask >>= 1
    return count


class KBucket:
    """A single K-bucket holding up to K peers, ordered by last-seen (LRU).

    When the bucket is full and a new peer wants in, the standard Kademlia
    approach is to check whether the least-recently-seen (LRU) peer is still
    alive before evicting it. This prevents Sybil attacks and preserves
    long-lived stable peers.

    The bucket accepts an optional ``is_alive`` callback (provided by the
    RoutingTable owner). When present, the LRU peer is only evicted if the
    callback reports it as dead. When absent, peers not seen within
    ``STALE_PEER_TIMEOUT`` are evicted, and recently-seen peers are kept.

    A single-slot replacement cache holds one pending peer. When a bucket peer
    is removed (confirmed dead or explicitly evicted), the pending peer is
    automatically promoted into the bucket. This matches rust-libp2p's
    replacement cache behavior.
    """

    def __init__(self, k: int = K, is_alive: Callable[[bytes], bool] | None = None):
        self.k = k
        self.peers: list[PeerEntry] = []
        self._is_alive = is_alive
        self._pending: PeerEntry | None = None  # replacement cache (1 slot)
        self._pending_since: float | None = None  # when the pending entry was stored

    def __len__(self) -> int:
        return len(self.peers)

    def add_or_update(self, peer_id: bytes, addrs: list[bytes]) -> bool:
        """Add a peer or move it to the tail (most recently seen).

        Returns True if the peer was added/updated, False if the bucket
        is full and the LRU peer is still considered alive. When rejected,
        the peer is stored in the replacement cache for later promotion.
        """
        for i, entry in enumerate(self.peers):
            if entry.peer_id == peer_id:
                # Move to tail (most recently seen)
                entry.addrs = addrs
                entry.last_seen = time.monotonic()
                self.peers.append(self.peers.pop(i))
                return True

        # Also update pending cache if this peer is already there
        if self._pending and self._pending.peer_id == peer_id:
            self._pending.addrs = addrs
            self._pending.last_seen = time.monotonic()

        if len(self.peers) < self.k:
            self.peers.append(PeerEntry(peer_id, addrs))
            return True

        # Bucket full - check if the LRU peer should be evicted
        lru = self.peers[0]

        # Check if a pending entry has timed out (waited long enough to be
        # promoted unconditionally, matching rust-libp2p's pending_timeout).
        if (
            self._pending is not None
            and self._pending_since is not None
            and time.monotonic() - self._pending_since >= PENDING_ENTRY_TIMEOUT
        ):
            evicted = self.peers.pop(0)
            log.debug(
                f"evicted LRU peer {evicted.peer_id.hex()[:16]}... from bucket "
                f"(pending entry timed out after {PENDING_ENTRY_TIMEOUT}s)"
            )
            self.peers.append(self._pending)
            self._pending = PeerEntry(peer_id, addrs)
            self._pending_since = time.monotonic()
            return False  # the new peer is now pending, not yet in bucket

        if self._is_alive is not None:
            if self._is_alive(lru.peer_id):
                # LRU is alive: refresh it, reject the newcomer but cache it
                lru.last_seen = time.monotonic()
                self.peers.append(self.peers.pop(0))
                self._pending = PeerEntry(peer_id, addrs)
                self._pending_since = time.monotonic()
                return False
        else:
            age = time.monotonic() - lru.last_seen
            if age < STALE_PEER_TIMEOUT:
                # Recently seen: keep it, cache the newcomer
                self._pending = PeerEntry(peer_id, addrs)
                self._pending_since = time.monotonic()
                return False

        # LRU is stale or confirmed dead: evict and add the new peer
        evicted = self.peers.pop(0)
        log.debug(f"evicted stale peer {evicted.peer_id.hex()[:16]}... from bucket")
        self.peers.append(PeerEntry(peer_id, addrs))
        return True

    def remove(self, peer_id: bytes) -> bool:
        """Remove a peer from the bucket. Returns True if found.

        If there is a pending replacement, it is automatically promoted.
        """
        for i, entry in enumerate(self.peers):
            if entry.peer_id == peer_id:
                self.peers.pop(i)
                # Promote pending replacement if available
                if self._pending is not None:
                    self.peers.append(self._pending)
                    log.debug(
                        f"promoted pending peer {self._pending.peer_id.hex()[:16]}... "
                        f"into bucket (replacing {peer_id.hex()[:16]}...)"
                    )
                    self._pending = None
                    self._pending_since = None
                return True
        # Also clear pending if it matches the removed peer
        if self._pending and self._pending.peer_id == peer_id:
            self._pending = None
            self._pending_since = None
        return False

    def get(self, peer_id: bytes) -> PeerEntry | None:
        """Find a peer in this bucket."""
        for entry in self.peers:
            if entry.peer_id == peer_id:
                return entry
        return None

    def all_peers(self) -> list[PeerEntry]:
        """Return all peers in this bucket."""
        return list(self.peers)


class RoutingTable:
    """Kademlia routing table: 256 K-buckets indexed by XOR distance prefix length."""

    def __init__(
        self,
        local_peer_id: bytes,
        k: int = K,
        is_alive: Callable[[bytes], bool] | None = None,
    ):
        self.local_peer_id = local_peer_id
        self.k = k
        self._buckets = [KBucket(k, is_alive=is_alive) for _ in range(256)]

    def _bucket_index(self, peer_id: bytes) -> int:
        """Determine which bucket a peer belongs in."""
        cpl = _common_prefix_length(self.local_peer_id, peer_id)
        return min(cpl, 255)

    def add_or_update(self, peer_id: bytes, addrs: list[bytes]) -> bool:
        """Add or update a peer in the routing table.

        Does not add self.
        """
        if peer_id == self.local_peer_id:
            return False
        idx = self._bucket_index(peer_id)
        return self._buckets[idx].add_or_update(peer_id, addrs)

    def remove(self, peer_id: bytes) -> bool:
        """Remove a peer from the routing table."""
        idx = self._bucket_index(peer_id)
        return self._buckets[idx].remove(peer_id)

    def find(self, peer_id: bytes) -> PeerEntry | None:
        """Look up a specific peer."""
        idx = self._bucket_index(peer_id)
        return self._buckets[idx].get(peer_id)

    def closest_peers(self, target: bytes, count: int = K) -> list[PeerEntry]:
        """Return up to `count` peers closest to `target` by XOR distance."""
        all_peers = []
        for bucket in self._buckets:
            all_peers.extend(bucket.all_peers())

        all_peers.sort(key=lambda p: xor_distance(p.peer_id, target))
        return all_peers[:count]

    def size(self) -> int:
        """Total number of peers in the routing table."""
        return sum(len(b) for b in self._buckets)

    def all_peers(self) -> list[PeerEntry]:
        """Return all peers across all buckets."""
        result = []
        for bucket in self._buckets:
            result.extend(bucket.all_peers())
        return result
