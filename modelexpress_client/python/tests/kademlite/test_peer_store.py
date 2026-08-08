# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backoff and eviction lifecycle tests for PeerStore.

Covers:
- AddrInfo backoff arithmetic (failure increments, is_backed_off window,
  success reset, MAX_DIAL_FAILURES eviction).
- prune_stale removes peers with no live connection and no addresses.
- replace_addrs preserves AddrInfo state for known addrs (note: current
  implementation resets state - we check the actual semantics either way).
- replace_addrs on unknown peer creates the entry without raising.
- get_or_dial uses real dial substitution to drive the eviction path
  through MAX_DIAL_FAILURES.
"""

import time

import pytest

from kademlite.crypto import Ed25519Identity
from kademlite.multiaddr import encode_multiaddr_ip4_tcp
from kademlite.peer_store import (
    BACKOFF_BASE,
    MAX_DIAL_FAILURES,
    AddrInfo,
    PeerInfo,
    PeerStore,
)

# ---------------------------------------------------------------------------
# AddrInfo backoff arithmetic
# ---------------------------------------------------------------------------


def test_addrinfo_record_failure_increments() -> None:
    info = AddrInfo(b"addr-1")
    assert info.failures == 0
    assert info.backoff_until == 0.0
    assert not info.is_backed_off

    info.record_failure()
    assert info.failures == 1
    # backoff is monotonic-now + base
    assert info.backoff_until > time.monotonic()
    assert info.is_backed_off


def test_addrinfo_backoff_window_exponential() -> None:
    info = AddrInfo(b"addr-2")
    info.record_failure()  # 1 -> base * 2^0 = base
    first = info.backoff_until - time.monotonic()
    info.record_failure()  # 2 -> base * 2^1
    second = info.backoff_until - time.monotonic()
    # Second backoff window must be larger (modulo small clock skew)
    assert second > first
    # Lower bound sanity: at least BACKOFF_BASE
    assert first >= BACKOFF_BASE - 0.5


def test_addrinfo_record_success_resets() -> None:
    info = AddrInfo(b"addr-3")
    info.record_failure()
    info.record_failure()
    assert info.is_backed_off
    info.record_success()
    assert info.failures == 0
    assert info.backoff_until == 0.0
    assert not info.is_backed_off


def test_addrinfo_should_remove_threshold() -> None:
    info = AddrInfo(b"addr-4")
    for _ in range(MAX_DIAL_FAILURES - 1):
        info.record_failure()
    assert not info.should_remove
    info.record_failure()
    assert info.should_remove


# ---------------------------------------------------------------------------
# PeerInfo.addrs setter preserves existing AddrInfo state
# ---------------------------------------------------------------------------


def test_peerinfo_addrs_setter_preserves_existing_state() -> None:
    addr_old = encode_multiaddr_ip4_tcp("10.0.0.1", 4001)
    addr_new = encode_multiaddr_ip4_tcp("10.0.0.2", 4001)

    info = PeerInfo(b"peer-1", [addr_old])
    info.addr_infos[addr_old].record_failure()
    assert info.addr_infos[addr_old].failures == 1

    # Use the property setter to add a new address while keeping the old
    info.addrs = [addr_old, addr_new]

    # State for the existing addr must be preserved (same AddrInfo object)
    assert info.addr_infos[addr_old].failures == 1
    # New addr starts fresh
    assert info.addr_infos[addr_new].failures == 0


def test_peerinfo_addrs_setter_drops_missing() -> None:
    addr_a = encode_multiaddr_ip4_tcp("10.0.0.1", 4001)
    addr_b = encode_multiaddr_ip4_tcp("10.0.0.2", 4001)

    info = PeerInfo(b"peer-2", [addr_a, addr_b])
    info.addrs = [addr_b]  # drop addr_a
    assert addr_a not in info.addr_infos
    assert addr_b in info.addr_infos


# ---------------------------------------------------------------------------
# PeerStore.replace_addrs
# ---------------------------------------------------------------------------


def _make_store() -> PeerStore:
    return PeerStore(identity=Ed25519Identity.generate())


def test_replace_addrs_on_unknown_peer_does_not_raise() -> None:
    store = _make_store()
    addr = encode_multiaddr_ip4_tcp("10.0.0.5", 4001)
    # Must not raise - should create the peer entry on demand
    store.replace_addrs(b"unknown-peer", [addr])
    assert store.get_addrs(b"unknown-peer") == [addr]


def test_replace_addrs_resets_state_on_existing_peer() -> None:
    """replace_addrs is the 'wipe and replace' path; it does not preserve
    AddrInfo state by design (use add_addrs for the merge semantics)."""
    store = _make_store()
    addr_a = encode_multiaddr_ip4_tcp("10.0.0.1", 4001)
    addr_b = encode_multiaddr_ip4_tcp("10.0.0.2", 4001)

    store.add_addrs(b"peer-rp", [addr_a])
    info = store._peers[b"peer-rp"]
    info.addr_infos[addr_a].record_failure()

    # Replacing with a list that still contains addr_a must wipe its state
    store.replace_addrs(b"peer-rp", [addr_a, addr_b])
    assert store.get_addrs(b"peer-rp") == [addr_a, addr_b]
    assert store._peers[b"peer-rp"].addr_infos[addr_a].failures == 0


# ---------------------------------------------------------------------------
# PeerStore.prune_stale
# ---------------------------------------------------------------------------


class _FakeConnection:
    """Stand-in for connection.Connection used only for is_alive checks."""

    def __init__(self, alive: bool = True) -> None:
        self._alive = alive

    @property
    def is_alive(self) -> bool:
        return self._alive

    def kill(self) -> None:
        self._alive = False


def test_prune_stale_removes_peers_with_no_addrs_and_no_conn() -> None:
    store = _make_store()
    # Peer with no addrs and no connection - empty PeerInfo
    store._peers[b"empty-peer"] = PeerInfo(b"empty-peer", [])

    removed = store.prune_stale()
    assert removed == 1
    assert b"empty-peer" not in store._peers


def test_prune_stale_clears_dead_connection_objects() -> None:
    """Peer with addrs + dead connection: connection cleared, peer kept."""
    store = _make_store()
    addr = encode_multiaddr_ip4_tcp("10.0.0.1", 4001)
    store.add_addrs(b"peer-dead-conn", [addr])
    dead_conn = _FakeConnection(alive=False)
    store._peers[b"peer-dead-conn"].connection = dead_conn

    removed = store.prune_stale()
    assert removed == 0  # peer not removed (has addrs)
    assert store._peers[b"peer-dead-conn"].connection is None


def test_prune_stale_keeps_peers_with_live_connection() -> None:
    store = _make_store()
    live = _FakeConnection(alive=True)
    store._peers[b"peer-live"] = PeerInfo(b"peer-live", [])
    store._peers[b"peer-live"].connection = live

    removed = store.prune_stale()
    assert removed == 0
    assert b"peer-live" in store._peers


# ---------------------------------------------------------------------------
# get_or_dial backoff + eviction integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_or_dial_evicts_addr_after_max_failures(monkeypatch) -> None:
    """After MAX_DIAL_FAILURES consecutive failures, the address must be
    removed from the peer's addr_infos."""
    store = _make_store()
    addr = encode_multiaddr_ip4_tcp("198.51.100.7", 4001)
    store.add_addrs(b"peer-evict", [addr])

    # Stub dial to always fail synchronously, AND to bypass the backoff
    # window by resetting backoff_until before each call.
    call_count = 0

    async def fake_dial(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ConnectionError("stub: refused")

    monkeypatch.setattr("kademlite.peer_store.dial", fake_dial)

    # Drive MAX_DIAL_FAILURES failures, manually clearing backoff between
    # attempts so we don't have to wait real seconds.
    for _ in range(MAX_DIAL_FAILURES):
        info = store._peers[b"peer-evict"]
        if addr in info.addr_infos:
            info.addr_infos[addr].backoff_until = 0.0
        with pytest.raises(ConnectionError):
            await store.get_or_dial(b"peer-evict")

    # Address must have been evicted on the final failure
    info = store._peers[b"peer-evict"]
    assert addr not in info.addr_infos
    # And dial was called exactly MAX_DIAL_FAILURES times
    assert call_count == MAX_DIAL_FAILURES


@pytest.mark.asyncio
async def test_get_or_dial_skips_backed_off_addr(monkeypatch) -> None:
    """If an address is in its backoff window, get_or_dial must skip it
    rather than re-attempting."""
    store = _make_store()
    addr = encode_multiaddr_ip4_tcp("198.51.100.8", 4001)
    store.add_addrs(b"peer-backoff", [addr])

    # Pre-mark the addr as backed off
    info = store._peers[b"peer-backoff"]
    info.addr_infos[addr].record_failure()
    assert info.addr_infos[addr].is_backed_off

    call_count = 0

    async def fake_dial(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise ConnectionError("should not be reached")

    monkeypatch.setattr("kademlite.peer_store.dial", fake_dial)

    # All addrs are in backoff so dial must not be called; we still get
    # ConnectionError because there are no usable addresses.
    with pytest.raises(ConnectionError):
        await store.get_or_dial(b"peer-backoff")
    assert call_count == 0


@pytest.mark.asyncio
async def test_get_or_dial_records_success_resets_backoff(monkeypatch) -> None:
    """On a successful dial, the AddrInfo state must be reset and the
    connection registered."""
    store = _make_store()
    addr = encode_multiaddr_ip4_tcp("198.51.100.9", 4001)
    store.add_addrs(b"peer-good", [addr])
    # Seed a prior failure so we can verify reset behavior
    info = store._peers[b"peer-good"]
    info.addr_infos[addr].record_failure()
    info.addr_infos[addr].backoff_until = 0.0  # bypass backoff window

    fake_conn = _FakeConnection(alive=True)

    async def fake_dial(*args, **kwargs):
        return fake_conn

    monkeypatch.setattr("kademlite.peer_store.dial", fake_dial)

    conn = await store.get_or_dial(b"peer-good")
    assert conn is fake_conn

    info = store._peers[b"peer-good"]
    assert info.addr_infos[addr].failures == 0
    assert info.addr_infos[addr].backoff_until == 0.0
    # Subsequent get_or_dial returns the cached connection without re-dialing
    again = await store.get_or_dial(b"peer-good")
    assert again is fake_conn
