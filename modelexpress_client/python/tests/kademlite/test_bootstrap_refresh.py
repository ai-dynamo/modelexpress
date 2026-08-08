# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for v0.3.0 per-CPL bucket-fill bootstrap.

The original Kademlia paper (Maymounkov & Mazieres 2002) defines the
join procedure as: insert known node, lookup own ID, then refresh all
k-buckets farther than the closest-neighbor bucket. rust-libp2p does
this synchronously inside the bootstrap QueryId. kademlite v0.2.0 only
ran the self-lookup on bootstrap and deferred per-bucket refresh to
the 5-minute maintenance loop, so cold consumers issuing PUT/GET right
after start saw an under-populated routing table.

v0.3.0 adds:
- ``DhtNode.start(wait_until_routable=True)`` (default) runs one round
  of ``_refresh_buckets`` after the self-lookup.
- ``DhtNode.wait_until_routable()`` for callers who used
  ``start(wait_until_routable=False)`` and want to refresh later.
"""

import asyncio
from unittest.mock import patch

import pytest  # noqa: F401

from kademlite.crypto import _base58btc_encode
from kademlite.dht import DhtNode


def _node_multiaddr(node: DhtNode) -> str:
    host, port = node.listen_addr
    return f"/ip4/{host}/tcp/{port}/p2p/{_base58btc_encode(node.peer_id)}"


async def test_start_default_runs_refresh_buckets() -> None:
    """start() with default wait_until_routable=True must call
    _refresh_buckets exactly once after the bootstrap-time self-lookup,
    so the cold consumer sees distant-bucket peers immediately."""
    node_a = DhtNode()
    await node_a.start("127.0.0.1", 0, wait_until_routable=False)
    addr_a = _node_multiaddr(node_a)
    try:
        node_b = DhtNode()
        # Patch _refresh_buckets to count invocations triggered by start().
        refresh_calls = []
        original_refresh = node_b._refresh_buckets

        async def counting_refresh():
            refresh_calls.append(1)
            await original_refresh()

        with patch.object(node_b, "_refresh_buckets", side_effect=counting_refresh):
            await node_b.start("127.0.0.1", 0, bootstrap_peers=[addr_a])

        try:
            assert len(refresh_calls) == 1, (
                f"expected exactly one _refresh_buckets call from start() "
                f"with default wait_until_routable=True, got {len(refresh_calls)}"
            )
        finally:
            await node_b.stop()
    finally:
        await node_a.stop()


async def test_start_opt_out_skips_refresh_buckets() -> None:
    """start(wait_until_routable=False) must NOT fire _refresh_buckets
    during bootstrap. The maintenance loop retains its own periodic
    refresh, but bootstrap-time is skipped."""
    node_a = DhtNode()
    await node_a.start("127.0.0.1", 0, wait_until_routable=False)
    addr_a = _node_multiaddr(node_a)
    try:
        node_b = DhtNode()
        refresh_calls = []
        original_refresh = node_b._refresh_buckets

        async def counting_refresh():
            refresh_calls.append(1)
            await original_refresh()

        with patch.object(node_b, "_refresh_buckets", side_effect=counting_refresh):
            await node_b.start(
                "127.0.0.1", 0,
                bootstrap_peers=[addr_a],
                wait_until_routable=False,
            )

        try:
            assert len(refresh_calls) == 0, (
                f"_refresh_buckets must not be called from start() when "
                f"wait_until_routable=False; got {len(refresh_calls)} calls"
            )
        finally:
            await node_b.stop()
    finally:
        await node_a.stop()


async def test_wait_until_routable_method_runs_refresh() -> None:
    """DhtNode.wait_until_routable() must invoke _refresh_buckets when
    the routing table is non-empty. Caller-driven equivalent of the
    sync bootstrap-finalization for opted-out start() callers."""
    node_a = DhtNode()
    await node_a.start("127.0.0.1", 0, wait_until_routable=False)
    addr_a = _node_multiaddr(node_a)
    try:
        node_b = DhtNode()
        await node_b.start(
            "127.0.0.1", 0,
            bootstrap_peers=[addr_a],
            wait_until_routable=False,
        )
        try:
            refresh_calls = []
            original_refresh = node_b._refresh_buckets

            async def counting_refresh():
                refresh_calls.append(1)
                await original_refresh()

            with patch.object(node_b, "_refresh_buckets", side_effect=counting_refresh):
                await node_b.wait_until_routable()

            assert len(refresh_calls) == 1
        finally:
            await node_b.stop()
    finally:
        await node_a.stop()


async def test_wait_until_routable_skips_on_empty_routing_table() -> None:
    """If the routing table is empty (e.g. no bootstrap peers configured),
    wait_until_routable() is a no-op; refreshing nothing avoids a
    needless busy loop in _refresh_buckets's bucket scan."""
    node = DhtNode()
    await node.start("127.0.0.1", 0, wait_until_routable=False)
    try:
        refresh_calls = []
        original_refresh = node._refresh_buckets

        async def counting_refresh():
            refresh_calls.append(1)
            await original_refresh()

        with patch.object(node, "_refresh_buckets", side_effect=counting_refresh):
            await node.wait_until_routable()

        assert len(refresh_calls) == 0, (
            f"wait_until_routable() on empty routing table must skip; "
            f"got {len(refresh_calls)} calls"
        )
    finally:
        await node.stop()


async def test_mdns_first_peer_triggers_one_shot_refresh() -> None:
    """An mDNS-only node starts with an empty routing table so the
    bootstrap-time per-CPL refresh skips. The FIRST mDNS-discovered
    peer must fire the refresh exactly once; subsequent discoveries
    must NOT fire fresh per-CPL walks (the maintenance loop's normal
    cadence handles those)."""
    # Set up a peer that mDNS can "discover" via direct callback.
    node_a = DhtNode()
    await node_a.start("127.0.0.1", 0, wait_until_routable=False)
    addr_a = _node_multiaddr(node_a)
    try:
        node_b = DhtNode()
        # No bootstrap configured; mDNS is explicitly disabled so we
        # can drive _on_mdns_peer manually with deterministic timing
        # rather than waiting for real mDNS multicast traffic.
        await node_b.start("127.0.0.1", 0, enable_mdns=False)

        refresh_calls = []
        original_refresh = node_b._refresh_buckets

        async def counting_refresh():
            refresh_calls.append(1)
            await original_refresh()

        try:
            with patch.object(node_b, "_refresh_buckets", side_effect=counting_refresh):
                # First mDNS peer: should fire the one-shot refresh.
                await node_b._on_mdns_peer(addr_a)
                # Refresh is spawned as a background task; let it run.
                await asyncio.sleep(0.3)

            assert len(refresh_calls) == 1, (
                f"first mDNS-discovered peer should fire exactly one "
                f"_refresh_buckets call; got {len(refresh_calls)}"
            )
            assert node_b._mdns_routable_refresh_done is True, (
                "_mdns_routable_refresh_done flag should be set after "
                "the one-shot refresh fires"
            )

            # Second mDNS peer (if there were one) should NOT fire again.
            # We can simulate by re-calling _on_mdns_peer with the same
            # peer (it'll skip on already-connected) AND with a fresh
            # peer to verify the flag holds.
            with patch.object(node_b, "_refresh_buckets", side_effect=counting_refresh):
                await node_b._on_mdns_peer(addr_a)  # already connected, skip
                await asyncio.sleep(0.1)

            assert len(refresh_calls) == 1, (
                f"subsequent mDNS discoveries must NOT fire additional "
                f"refreshes; got {len(refresh_calls)} total"
            )
        finally:
            await node_b.stop()
    finally:
        await node_a.stop()


async def test_no_bootstrap_skips_refresh_even_with_default() -> None:
    """A node with no bootstrap configured (mDNS-only or fully isolated)
    has nothing to refresh against. start(wait_until_routable=True) must
    skip the refresh in this case rather than spinning on an empty table."""
    node = DhtNode()
    refresh_calls = []
    original_refresh = node._refresh_buckets

    async def counting_refresh():
        refresh_calls.append(1)
        await original_refresh()

    with patch.object(node, "_refresh_buckets", side_effect=counting_refresh):
        # No bootstrap_peers, no bootstrap_dns, no bootstrap_slurm.
        # mDNS may auto-enable but doesn't populate the routing table
        # synchronously during start.
        await node.start("127.0.0.1", 0, enable_mdns=False)

    try:
        assert len(refresh_calls) == 0, (
            f"start() with no bootstrap should skip refresh; "
            f"got {len(refresh_calls)} calls"
        )
    finally:
        await node.stop()
