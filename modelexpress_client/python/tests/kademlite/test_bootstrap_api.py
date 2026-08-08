# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the libp2p-kad-style bootstrap API additions.

Two capabilities, mirroring rust-libp2p-kad's surface:

- ``DhtNode.start(bootstrap_interval=...)`` configures the periodic
  bootstrap loop's cadence (matches ``Config::set_periodic_bootstrap_interval``).
  ``bootstrap_interval=None`` disables the loop entirely.
- ``DhtNode.bootstrap()`` (no args) triggers one bootstrap cycle on
  demand (matches ``Behaviour::bootstrap``). Raises
  ``NoKnownPeersError`` when there's nothing to bootstrap against.

Original ``DhtNode.bootstrap(peers)`` shape is preserved as a
back-compat call form that dispatches to ``_bootstrap_dial_peers``.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from kademlite import NoKnownPeersError
from kademlite.crypto import _base58btc_encode
from kademlite.dht import DhtNode
from kademlite.dht_maintenance import BOOTSTRAP_INTERVAL


def _node_multiaddr(node: DhtNode) -> str:
    host, port = node.listen_addr
    return f"/ip4/{host}/tcp/{port}/p2p/{_base58btc_encode(node.peer_id)}"


# ---------------------------------------------------------------------------
# bootstrap_interval (Patch 1)
# ---------------------------------------------------------------------------


async def test_default_bootstrap_interval_matches_module_constant() -> None:
    """When start() is called without bootstrap_interval, the instance
    attribute reflects the module-level default. Preserves the v0.3.0
    behavior where the cadence was hardcoded to BOOTSTRAP_INTERVAL."""
    node = DhtNode()
    await node.start("127.0.0.1", 0, enable_mdns=False)
    try:
        assert node._bootstrap_interval == BOOTSTRAP_INTERVAL
    finally:
        await node.stop()


async def test_bootstrap_interval_custom_value_stored() -> None:
    """A custom bootstrap_interval passed to start() is stored on the
    instance and used by the periodic loop."""
    node = DhtNode()
    await node.start("127.0.0.1", 0, enable_mdns=False, bootstrap_interval=12.5)
    try:
        assert node._bootstrap_interval == 12.5
    finally:
        await node.stop()


async def test_bootstrap_interval_drives_periodic_tick_cadence() -> None:
    """The periodic bootstrap loop fires _periodic_bootstrap_tick at the
    configured interval. Use a tight interval so the test wall-time
    stays small while still observing multiple firings."""
    node_a = DhtNode()
    await node_a.start("127.0.0.1", 0, wait_until_routable=False, enable_mdns=False)
    addr_a = _node_multiaddr(node_a)
    try:
        # Patch the tick before start() so the loop sees the patched
        # version on its first sleep wakeup.
        node_b = DhtNode()
        tick_event = asyncio.Event()
        tick_count = 0

        async def counting_tick() -> None:
            nonlocal tick_count
            tick_count += 1
            tick_event.set()

        node_b._periodic_bootstrap_tick = counting_tick  # type: ignore[method-assign]
        await node_b.start(
            "127.0.0.1",
            0,
            bootstrap_peers=[addr_a],
            wait_until_routable=False,
            bootstrap_interval=0.1,
            enable_mdns=False,
        )
        try:
            # Wait for the first tick. With a 0.1s interval we expect
            # the first firing within ~0.2s; allow generous slack so
            # CI scheduling jitter doesn't flake the test.
            await asyncio.wait_for(tick_event.wait(), timeout=2.0)
            assert tick_count >= 1, (
                f"expected at least one tick at 0.1s cadence; got {tick_count}"
            )
        finally:
            await node_b.stop()
    finally:
        await node_a.stop()


async def test_bootstrap_interval_none_disables_periodic_loop() -> None:
    """bootstrap_interval=None disables the periodic bootstrap loop
    entirely: no task is spawned and _periodic_bootstrap_tick is
    never called even after waiting longer than the previous default
    cadence would have allowed."""
    node_a = DhtNode()
    await node_a.start("127.0.0.1", 0, wait_until_routable=False, enable_mdns=False)
    addr_a = _node_multiaddr(node_a)
    try:
        node_b = DhtNode()
        tick_count = 0

        async def counting_tick() -> None:
            nonlocal tick_count
            tick_count += 1

        node_b._periodic_bootstrap_tick = counting_tick  # type: ignore[method-assign]
        await node_b.start(
            "127.0.0.1",
            0,
            bootstrap_peers=[addr_a],
            wait_until_routable=False,
            bootstrap_interval=None,
            enable_mdns=False,
        )
        try:
            # No bootstrap task should have been spawned. Wait long
            # enough that any reasonable default-cadence loop would
            # have fired at least once.
            await asyncio.sleep(0.5)
            assert node_b._bootstrap_task is None, (
                "bootstrap_interval=None must not spawn a periodic loop task"
            )
            assert tick_count == 0, (
                f"bootstrap_interval=None must not fire ticks; got {tick_count}"
            )
        finally:
            await node_b.stop()
    finally:
        await node_a.stop()


async def test_bootstrap_interval_none_with_mdns_also_disables_loop() -> None:
    """The mDNS branch of start() also creates the periodic loop (so
    mDNS-only nodes can re-query on a sparse table). It must honor
    bootstrap_interval=None the same way as the explicit-bootstrap
    branch. Patches MdnsDiscovery so the branch runs without opening
    multicast sockets."""
    node = DhtNode()
    tick_count = 0

    async def counting_tick() -> None:
        nonlocal tick_count
        tick_count += 1

    node._periodic_bootstrap_tick = counting_tick  # type: ignore[method-assign]
    fake_mdns = AsyncMock()
    fake_mdns.start = AsyncMock(return_value=None)
    fake_mdns.stop = AsyncMock(return_value=None)

    with patch("kademlite.dht.MdnsDiscovery", return_value=fake_mdns):
        # No explicit bootstrap_peers => has_bootstrap is False =>
        # enable_mdns auto-resolves to True, taking the mDNS branch.
        # bootstrap_interval=None must still suppress task creation
        # in that branch (the gate we're verifying).
        await node.start(
            "127.0.0.1",
            0,
            bootstrap_interval=None,
        )
        try:
            await asyncio.sleep(0.2)
            assert node._mdns is fake_mdns, (
                "mDNS branch must have run: _mdns should be set to the "
                "patched MdnsDiscovery instance"
            )
            assert node._bootstrap_task is None, (
                "mDNS branch must skip task creation when "
                "bootstrap_interval=None"
            )
            assert tick_count == 0
        finally:
            await node.stop()


async def test_public_bootstrap_with_mdns_enabled_does_not_raise() -> None:
    """mDNS counts as a bootstrap source: an mDNS-enabled node with an
    empty routing table can call bootstrap() without NoKnownPeersError,
    because the underlying tick will fire mDNS.send_query() to discover
    local peers. Without this, mDNS-only deployments could not use the
    on-demand trigger."""
    node = DhtNode()
    fake_mdns = AsyncMock()
    fake_mdns.start = AsyncMock(return_value=None)
    fake_mdns.stop = AsyncMock(return_value=None)
    fake_mdns.send_query = lambda: None

    with patch("kademlite.dht.MdnsDiscovery", return_value=fake_mdns):
        await node.start(
            "127.0.0.1",
            0,
            # No explicit bootstrap; mDNS auto-enables.
            bootstrap_interval=None,
        )
        try:
            assert node._mdns is fake_mdns
            assert node.routing_table.size() == 0
            # Must NOT raise NoKnownPeersError: mDNS is a valid source.
            await node.bootstrap()
            # send_query is called from _periodic_bootstrap_tick on the
            # sparse-table branch; verify the tick reached that path.
        finally:
            await node.stop()


# ---------------------------------------------------------------------------
# DhtNode.bootstrap() public trigger (Patch 2)
# ---------------------------------------------------------------------------


async def test_public_bootstrap_runs_one_tick() -> None:
    """node.bootstrap() with no args runs the equivalent of one
    periodic bootstrap cycle. Verified by patching
    _periodic_bootstrap_tick and counting invocations."""
    node_a = DhtNode()
    await node_a.start("127.0.0.1", 0, wait_until_routable=False, enable_mdns=False)
    addr_a = _node_multiaddr(node_a)
    try:
        node_b = DhtNode()
        await node_b.start(
            "127.0.0.1",
            0,
            bootstrap_peers=[addr_a],
            wait_until_routable=False,
            # Disable the periodic loop so its ticks don't race with
            # the on-demand call we're measuring.
            bootstrap_interval=None,
            enable_mdns=False,
        )
        try:
            tick_calls = 0

            async def counting_tick() -> None:
                nonlocal tick_calls
                tick_calls += 1

            node_b._periodic_bootstrap_tick = counting_tick  # type: ignore[method-assign]
            await node_b.bootstrap()
            assert tick_calls == 1, (
                f"bootstrap() should run exactly one tick; got {tick_calls}"
            )
        finally:
            await node_b.stop()
    finally:
        await node_a.stop()


async def test_public_bootstrap_raises_no_known_peers_when_empty() -> None:
    """node.bootstrap() raises NoKnownPeersError when the routing
    table is empty AND no bootstrap sources are configured. Mirrors
    rust-libp2p-kad's NoKnownPeers Err return."""
    node = DhtNode()
    # No bootstrap_peers, no bootstrap_dns, no bootstrap_slurm.
    # mDNS disabled so the routing table is genuinely empty.
    await node.start("127.0.0.1", 0, enable_mdns=False)
    try:
        with pytest.raises(NoKnownPeersError):
            await node.bootstrap()
    finally:
        await node.stop()


async def test_public_bootstrap_runs_when_bootstrap_sources_configured() -> None:
    """If the routing table is empty but bootstrap_peers were
    configured at start(), bootstrap() should NOT raise; the tick
    will re-dial those peers. Matches rust-libp2p, where having
    seed peers in the routing table (via add_address) prevents
    NoKnownPeers."""
    node_a = DhtNode()
    await node_a.start("127.0.0.1", 0, wait_until_routable=False, enable_mdns=False)
    addr_a = _node_multiaddr(node_a)
    try:
        node_b = DhtNode()
        await node_b.start(
            "127.0.0.1",
            0,
            bootstrap_peers=[addr_a],
            wait_until_routable=False,
            bootstrap_interval=None,
            enable_mdns=False,
        )
        try:
            # bootstrap_peers were saved at start(), so even if the
            # routing table is empty here this call must not raise.
            tick_calls = 0

            async def counting_tick() -> None:
                nonlocal tick_calls
                tick_calls += 1

            node_b._periodic_bootstrap_tick = counting_tick  # type: ignore[method-assign]
            # Force-empty the routing table to test the fall-through
            # path (configured sources but no live peers).
            for entry in list(node_b.routing_table.all_peers()):
                node_b.routing_table.remove(entry.peer_id)
            assert node_b.routing_table.size() == 0

            await node_b.bootstrap()
            assert tick_calls == 1
        finally:
            await node_b.stop()
    finally:
        await node_a.stop()


async def test_public_bootstrap_before_start_raises_no_known_peers() -> None:
    """Calling bootstrap() before start() must surface NoKnownPeersError
    (the stable empty-state error), not AttributeError. _bootstrap_dns
    and _bootstrap_dns_port are initialized in __init__ to safe defaults
    so the has_bootstrap_source check works pre-start."""
    node = DhtNode()
    # No start(), no listener, routing table empty, no sources configured.
    # The has_bootstrap_source check in _bootstrap_now reads
    # _bootstrap_peers / _bootstrap_dns / _bootstrap_hostlist - all three
    # need init defaults for this not to AttributeError.
    with pytest.raises(NoKnownPeersError):
        await node.bootstrap()


# ---------------------------------------------------------------------------
# bootstrap_interval validation
# ---------------------------------------------------------------------------


async def test_bootstrap_interval_zero_rejected() -> None:
    """bootstrap_interval=0 would busy-spin in asyncio.sleep. Reject
    at the boundary with ValueError."""
    node = DhtNode()
    with pytest.raises(ValueError):
        await node.start("127.0.0.1", 0, enable_mdns=False, bootstrap_interval=0)


async def test_bootstrap_interval_negative_rejected() -> None:
    """Negative bootstrap_interval has no sane meaning. Reject with
    ValueError."""
    node = DhtNode()
    with pytest.raises(ValueError):
        await node.start("127.0.0.1", 0, enable_mdns=False, bootstrap_interval=-1.5)


async def test_bootstrap_interval_bool_rejected() -> None:
    """bool is an int subclass in Python. Both True and False would
    slip past the None-check; True is technically 1.0 (legal but
    almost certainly a bug) and False is 0 (would busy-spin). Reject
    explicitly with TypeError, matching __init__'s k/alpha pattern."""
    node = DhtNode()
    with pytest.raises(TypeError):
        await node.start("127.0.0.1", 0, enable_mdns=False, bootstrap_interval=True)
    node2 = DhtNode()
    with pytest.raises(TypeError):
        await node2.start("127.0.0.1", 0, enable_mdns=False, bootstrap_interval=False)


async def test_bootstrap_interval_non_numeric_rejected() -> None:
    """A string or other non-numeric value must raise TypeError, not
    fall through to asyncio.sleep where it would crash with a less
    helpful TypeError from the sleep call itself."""
    node = DhtNode()
    with pytest.raises(TypeError):
        await node.start("127.0.0.1", 0, enable_mdns=False, bootstrap_interval="60")  # type: ignore[arg-type]


async def test_bootstrap_interval_int_accepted() -> None:
    """Plain int (not bool) is a valid duration in seconds. The
    boundary check accepts both int and float; only bool is rejected."""
    node = DhtNode()
    await node.start("127.0.0.1", 0, enable_mdns=False, bootstrap_interval=30)
    try:
        assert node._bootstrap_interval == 30
    finally:
        await node.stop()


async def test_public_bootstrap_with_peers_preserves_dial_behavior() -> None:
    """Back-compat: node.bootstrap([addrs]) still dials those peers
    and adds them to the routing table. The kademlite-original call
    shape stays available alongside the new no-arg form."""
    node_a = DhtNode()
    await node_a.start("127.0.0.1", 0, wait_until_routable=False, enable_mdns=False)
    addr_a = _node_multiaddr(node_a)
    try:
        node_b = DhtNode()
        # No bootstrap_peers at start: we'll dial via the explicit
        # bootstrap(peers) call instead.
        await node_b.start(
            "127.0.0.1",
            0,
            wait_until_routable=False,
            bootstrap_interval=None,
            enable_mdns=False,
        )
        try:
            assert node_b.routing_table.size() == 0
            await node_b.bootstrap([addr_a])
            assert node_b.routing_table.size() >= 1, (
                "bootstrap(peers) must populate the routing table "
                "(back-compat with the v0.3.0 dial-peers shape)"
            )
        finally:
            await node_b.stop()
    finally:
        await node_a.stop()
