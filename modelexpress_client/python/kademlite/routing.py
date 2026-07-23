# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kademlia routing table with XOR distance metric.

Reference: https://github.com/libp2p/specs/blob/master/kad-dht/README.md

The routing table organizes peers into K-buckets based on XOR distance from
the local node's peer ID, computed in the Kademlia keyspace. Per the libp2p
kad-dht spec, the metric is XOR(sha256(a), sha256(b)) - peer IDs and record
keys are hashed into a 32-byte keyspace before XOR. Use ``kad_key()`` to
transform an identifier into its keyspace representation; ``xor_distance``
and ``_common_prefix_length`` operate on the resulting raw bytes.

Each bucket holds up to K peers (default 20). Buckets are indexed by the
number of leading zero bits in the XOR distance, giving 256 buckets.
"""

import hashlib
import logging
import random
import time
from collections.abc import Callable

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
    """A peer in the routing table.

    Two booleans track reachability evidence on independent axes:

    - ``connected``: true while we have an active session connectivity to
      this peer (inbound or outbound). Cleared when a dial fails. Used by
      local lookup paths to pick peers we can route through right now.
    - ``dial_verified``: true only after we have personally dialled this
      peer's advertised listen addrs and the dial succeeded. Cleared on
      dial failure. Used by the responder-side hint filter so we never
      propagate addresses we have not verified are dialable from third
      parties. An inbound-only connection proves the peer can reach us;
      it does not prove the peer's listen addr is reachable from anyone
      else, so ``dial_verified`` stays false in that case.

    The two flags give four distinct states (verified+connected,
    verified+disconnected, unverified+connected, unverified+disconnected)
    and the responder filter requires both bits set.
    """

    __slots__ = ("peer_id", "kad_id", "addrs", "last_seen", "connected", "dial_verified")

    def __init__(self, peer_id: bytes, addrs: list[bytes], last_seen: float | None = None):
        self.peer_id = peer_id
        # Precompute the Kad-keyspace identifier (sha256(peer_id)) once at
        # construction so distance comparisons in closest_peers / iterative
        # lookups don't re-hash on every call. peer_id never changes for an
        # entry's lifetime, so the cache is safe.
        self.kad_id = kad_key(peer_id)
        self.addrs = addrs
        self.last_seen = last_seen or time.monotonic()
        self.connected: bool = True
        # New entries default to unverified. Promotion to ``dial_verified``
        # requires an authoritative outbound dial-success signal. Hint-only
        # peers learned from FIND_NODE / GET_VALUE responses, peers learned
        # via inbound connection, and peers whose listen addrs we have only
        # heard about all start (and stay) at False until we dial them.
        self.dial_verified: bool = False


def kad_key(identifier: bytes) -> bytes:
    """Hash a peer ID or record key into the Kademlia keyspace.

    Per the libp2p kad-dht spec, distance between two identifiers is
    ``XOR(sha256(a), sha256(b))``, not raw XOR of the underlying bytes.
    This wraps the SHA-256 transformation so call sites that compute
    routing distances or bucket indices can stay in the keyspace.

    Matches rust-libp2p's ``kbucket::Key`` and go-libp2p-kbucket's
    ``XORKeySpace.Key`` transformations.
    """
    return hashlib.sha256(identifier).digest()


def xor_distance(a: bytes, b: bytes) -> int:
    """Compute XOR distance between two raw byte strings as an integer.

    This is the bare XOR-distance primitive operating on whatever bytes
    are passed in. Most call sites should pass keyspace-transformed
    identifiers via ``kad_key()`` to match the kad-dht spec.
    """
    # Pad to same length
    max_len = max(len(a), len(b))
    a_padded = a.ljust(max_len, b"\x00")
    b_padded = b.ljust(max_len, b"\x00")
    return int.from_bytes(bytes(x ^ y for x, y in zip(a_padded, b_padded, strict=True)), "big")


def _common_prefix_length(a: bytes, b: bytes) -> int:
    """Number of leading zero bits in XOR(a, b). This determines the bucket index."""
    max_len = max(len(a), len(b))
    a_padded = a.ljust(max_len, b"\x00")
    b_padded = b.ljust(max_len, b"\x00")
    bits = 0
    for x, y in zip(a_padded, b_padded, strict=True):
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

        When the peer is already in the pending replacement cache, the
        existing pending entry is updated in place (preserving its
        ``connected`` / ``dial_verified`` flags), and is promoted into
        the active bucket as a single existing object - never replaced
        with a fresh ``PeerEntry`` - so flag state from earlier
        ``mark_*`` calls survives.
        """
        for i, entry in enumerate(self.peers):
            if entry.peer_id == peer_id:
                # Move to tail (most recently seen)
                entry.addrs = addrs
                entry.last_seen = time.monotonic()
                self.peers.append(self.peers.pop(i))
                return True

        # If this peer is already in the pending replacement cache, update
        # the existing entry in place. The promotion branches below check
        # ``pending_match`` and promote the existing pending object into
        # ``self.peers`` rather than creating a new ``PeerEntry``, so any
        # flags set on the pending entry by ``mark_*`` calls survive.
        pending_match = (
            self._pending is not None and self._pending.peer_id == peer_id
        )
        if pending_match:
            self._pending.addrs = addrs
            self._pending.last_seen = time.monotonic()

        if len(self.peers) < self.k:
            if pending_match:
                # Promote the existing pending entry to preserve flags.
                self.peers.append(self._pending)
                self._pending = None
                self._pending_since = None
            else:
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
            if pending_match:
                # Newcomer is the pending entry being promoted. Clear the
                # pending slot rather than replace it with a fresh entry
                # for the same peer_id.
                self.peers.append(self._pending)
                self._pending = None
                self._pending_since = None
                return True
            self.peers.append(self._pending)
            self._pending = PeerEntry(peer_id, addrs)
            self._pending_since = time.monotonic()
            return False  # the new peer is now pending, not yet in bucket

        # If the LRU peer is disconnected, evict it immediately without liveness check.
        # Dead peers should never block live ones from entering the routing table.
        if not lru.connected:
            evicted = self.peers.pop(0)
            log.debug(
                f"evicted disconnected peer {evicted.peer_id.hex()[:16]}... from bucket"
            )
            if pending_match:
                self.peers.append(self._pending)
                self._pending = None
                self._pending_since = None
            else:
                self.peers.append(PeerEntry(peer_id, addrs))
            return True

        if self._is_alive is not None:
            if self._is_alive(lru.peer_id):
                # LRU is alive: refresh it, reject the newcomer but cache it.
                lru.last_seen = time.monotonic()
                self.peers.append(self.peers.pop(0))
                if not pending_match:
                    # Only allocate a fresh pending entry when the newcomer
                    # is not already pending; an in-place pending update
                    # already happened above.
                    self._pending = PeerEntry(peer_id, addrs)
                    self._pending_since = time.monotonic()
                return False
        else:
            age = time.monotonic() - lru.last_seen
            # Jitter prevents synchronized eviction when traffic stops
            jittered_timeout = STALE_PEER_TIMEOUT * random.uniform(0.8, 1.2)
            if age < jittered_timeout:
                # Recently seen: keep it, cache the newcomer
                if not pending_match:
                    self._pending = PeerEntry(peer_id, addrs)
                    self._pending_since = time.monotonic()
                return False

        # LRU is stale or confirmed dead: evict and add the new peer
        evicted = self.peers.pop(0)
        log.debug(f"evicted stale peer {evicted.peer_id.hex()[:16]}... from bucket")
        if pending_match:
            self.peers.append(self._pending)
            self._pending = None
            self._pending_since = None
        else:
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

    def _find_entry(self, peer_id: bytes) -> PeerEntry | None:
        """Locate a peer entry across both the active bucket and the
        pending replacement cache.

        Mark-state mutations must reach the pending entry too: a peer
        can be successfully outbound-dialled while the bucket is full,
        which lands the new entry in ``_pending`` (``add_or_update``
        returns False). When a future bucket eviction promotes the
        pending entry into ``peers`` (``remove`` -> promotion at
        ``self.peers.append(self._pending)``), the entry's flags are
        whatever they were last left at. If the dial-success mark missed
        ``_pending`` it would never set ``dial_verified``, and the now-
        active peer would stay invisible to the responder filter despite
        having been authoritatively dial-verified.
        """
        for entry in self.peers:
            if entry.peer_id == peer_id:
                return entry
        if self._pending is not None and self._pending.peer_id == peer_id:
            return self._pending
        return None

    def mark_disconnected(self, peer_id: bytes) -> bool:
        """Mark a peer as disconnected. Returns True if found.

        Also clears ``dial_verified``: a failed dial means the listen addr
        we hold for this peer is no longer trustworthy as a hint to give
        to other nodes. The flag is reinstated on a future successful
        dial via ``mark_dial_verified``. Searches both the active bucket
        and the pending replacement slot.
        """
        entry = self._find_entry(peer_id)
        if entry is None:
            return False
        entry.connected = False
        entry.dial_verified = False
        return True

    def _move_to_tail(self, entry: PeerEntry) -> None:
        """Reposition ``entry`` to the tail of ``self.peers`` if present.

        The bucket uses list order as its LRU queue: ``self.peers[0]``
        is the eviction candidate, ``self.peers[-1]`` is the most
        recently seen. Mark-state updates that refresh ``last_seen``
        must also re-tail the entry in ``self.peers`` to keep the LRU
        invariant intact; otherwise a freshly reconnected or
        dial-verified peer can stay at the head and be evicted while
        an older peer at the tail keeps its slot. No-op if the entry
        lives in the pending replacement cache (not in ``self.peers``);
        a future promotion into ``peers`` from ``_pending`` will append
        to the tail naturally.
        """
        for i, current in enumerate(self.peers):
            if current is entry:
                self.peers.append(self.peers.pop(i))
                return

    def mark_connected(self, peer_id: bytes) -> bool:
        """Mark a peer as connected. Returns True if found.

        Sets ``connected=True`` only. Does NOT touch ``dial_verified``,
        which requires the stronger signal of a successful outbound dial
        to this peer's listen addrs (see ``mark_dial_verified``).
        Searches both the active bucket and the pending replacement slot,
        and re-tails the entry in the active bucket so the LRU queue
        order tracks the refreshed ``last_seen``.
        """
        entry = self._find_entry(peer_id)
        if entry is None:
            return False
        entry.connected = True
        entry.last_seen = time.monotonic()
        self._move_to_tail(entry)
        return True

    def mark_dial_verified(self, peer_id: bytes) -> bool:
        """Mark a peer as dial-verified AND connected. Returns True if found.

        Called when an outbound dial to this peer's listen addrs has
        succeeded. The dial is the only authoritative signal that the
        listen addr we are holding actually works, so propagating it as
        a hint to third parties is now safe. Searches both the active
        bucket and the pending replacement slot - a peer can be in
        ``_pending`` if the bucket was full at insertion time, and the
        flag must follow the entry through later promotion. Re-tails
        the entry in the active bucket on success so the LRU queue
        order tracks the refreshed ``last_seen``.
        """
        entry = self._find_entry(peer_id)
        if entry is None:
            return False
        entry.connected = True
        entry.dial_verified = True
        entry.last_seen = time.monotonic()
        self._move_to_tail(entry)
        return True

    def mark_dial_unverified(self, peer_id: bytes) -> bool:
        """Clear the ``dial_verified`` flag on an entry while leaving the
        ``connected`` flag and the entry's position untouched. Returns
        True if found.

        Used by code paths that learn a new set of listen addrs for a
        peer without dialing them - notably the Identify Push handler,
        where the peer pushes updated addrs over an existing connection.
        Pre-pushed addrs that we previously dial-verified are no longer
        the addrs we hold, so the verification must be revoked until a
        future outbound dial lands. Searches both the active bucket and
        the pending replacement slot for symmetry with the other
        ``mark_*`` methods.
        """
        entry = self._find_entry(peer_id)
        if entry is None:
            return False
        entry.dial_verified = False
        return True

    def get(self, peer_id: bytes) -> PeerEntry | None:
        """Find a peer in this bucket.

        Scans both the active bucket and the pending replacement slot.
        A pending entry is still "in" the routing table from the
        caller's perspective - it has flags (``connected``,
        ``dial_verified``, ``last_seen``) that mutators update via
        ``_find_entry``, and a future bucket eviction will promote it
        into ``self.peers``. Returning it from ``get`` keeps lookup
        consistent with the mutator side; otherwise ``find`` callers
        miss pending peers entirely (e.g. the Identify Push handler
        would treat an idempotent push to a pending peer as a brand-
        new addr set and revoke ``dial_verified``).
        """
        return self._find_entry(peer_id)

    def all_peers(self) -> list[PeerEntry]:
        """Return all peers in this bucket.

        Returns active-bucket entries only; pending replacement-cache
        entries are not yet promoted and are excluded so callers like
        ``RoutingTable.closest_peers`` and ``RoutingTable.size`` stay
        on the previous semantic. ``get`` / ``find`` are the points
        where pending visibility matters; iteration paths do not.
        """
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
        self._local_kad_id = kad_key(local_peer_id)
        self.k = k
        self._buckets = [KBucket(k, is_alive=is_alive) for _ in range(256)]

    def _bucket_index(self, peer_id: bytes) -> int:
        """Determine which bucket a peer belongs in.

        Uses the kad-dht keyspace metric: bucket = CPL of XOR over the
        SHA-256 hashes of the local and remote peer IDs.
        """
        cpl = _common_prefix_length(self._local_kad_id, kad_key(peer_id))
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

    def mark_disconnected(self, peer_id: bytes) -> bool:
        """Mark a peer as disconnected in the routing table. Returns True if found.

        Also clears ``dial_verified`` (see ``KBucket.mark_disconnected``).
        """
        idx = self._bucket_index(peer_id)
        return self._buckets[idx].mark_disconnected(peer_id)

    def mark_connected(self, peer_id: bytes) -> bool:
        """Mark a peer as connected in the routing table. Returns True if found.

        Sets ``connected=True`` only. Does NOT promote to ``dial_verified``.
        Use ``mark_dial_verified`` for that signal.
        """
        idx = self._bucket_index(peer_id)
        return self._buckets[idx].mark_connected(peer_id)

    def mark_dial_verified(self, peer_id: bytes) -> bool:
        """Mark a peer as dial-verified (and connected). Returns True if found.

        Called from outbound-dial-success paths only. The dial is the
        authoritative signal that the listen addr we hold for this peer
        is actually dialable, so the entry becomes safe to propagate as
        a hint in FIND_NODE / GET_VALUE responses.
        """
        idx = self._bucket_index(peer_id)
        return self._buckets[idx].mark_dial_verified(peer_id)

    def mark_dial_unverified(self, peer_id: bytes) -> bool:
        """Clear the ``dial_verified`` flag without changing
        ``connected`` or the entry's position. Returns True if found.

        Used when we accept a new set of listen addrs for an existing
        peer without dialing them (e.g. an inbound Identify Push). The
        previously-dialled addrs no longer match what we have, so the
        verification claim must be retracted until the new addrs are
        themselves dialed.
        """
        idx = self._bucket_index(peer_id)
        return self._buckets[idx].mark_dial_unverified(peer_id)

    def closest_peers(
        self,
        target: bytes,
        count: int = K,
        connected_only: bool = False,
        dial_verified_only: bool = False,
    ) -> list[PeerEntry]:
        """Return up to ``count`` peers closest to ``target`` in the Kad keyspace.

        Distance is XOR(sha256(peer_id), sha256(target)) per the kad-dht spec.

        ``connected_only=True`` filters to peers whose ``connected`` flag is
        set. Used by local lookup paths so candidate selection stays inside
        peers we can currently route through.

        ``dial_verified_only=True`` is stricter: it requires both
        ``dial_verified`` AND ``connected``. Used by the responder-side
        hint filter (``KadHandler._closest_peers_encoded``) so we never
        propagate addresses to third parties that we have not verified
        are dialable. ``dial_verified_only`` implies ``connected_only``;
        passing both is equivalent to passing only ``dial_verified_only``.
        """
        all_peers = []
        for bucket in self._buckets:
            all_peers.extend(bucket.all_peers())

        if dial_verified_only:
            # Strictest filter: both bits must be set. The dial-verified
            # bit alone is not enough because a peer that was dial-verified
            # but has since been marked disconnected (failed RPC) is no
            # longer trustworthy as a hint either; ``mark_disconnected``
            # clears both flags, so this gates against post-failure stale
            # entries that have not yet been re-verified.
            all_peers = [p for p in all_peers if p.dial_verified and p.connected]
        elif connected_only:
            all_peers = [p for p in all_peers if p.connected]

        target_kad = kad_key(target)
        # PeerEntry caches kad_id at construction so the sort key skips the
        # per-call sha256 of every peer_id; matters at N >> K.
        all_peers.sort(key=lambda p: xor_distance(p.kad_id, target_kad))
        return all_peers[:count]

    def size(self, connected_only: bool = False) -> int:
        """Total number of peers in the routing table.

        When ``connected_only`` is True, returns the count of peers whose
        ``connected`` flag is set. Useful for callers that need to know
        whether the table has enough *reachable* entries to satisfy a
        coverage threshold; the unfiltered count includes entries that
        the local node has marked unreachable but not yet evicted.
        """
        if not connected_only:
            return sum(len(b) for b in self._buckets)
        return sum(1 for b in self._buckets for p in b.peers if p.connected)

    def all_peers(self) -> list[PeerEntry]:
        """Return all peers across all buckets."""
        result = []
        for bucket in self._buckets:
            result.extend(bucket.all_peers())
        return result
