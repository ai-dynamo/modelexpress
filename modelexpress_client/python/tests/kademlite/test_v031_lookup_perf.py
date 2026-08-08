# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""v0.3.1 lookup-performance regression tests.

Covers the three primary v0.3.1 fixes:

A. ``asyncio.wait`` FIRST_COMPLETED loops in ``_iterative_get_value``,
   ``_iterative_find_node``, and the PUT-fanout, replacing the prior
   ``asyncio.gather`` round-barriers. A GET that finds a value on a fast
   peer must return immediately without waiting on a slow ghost peer.

B. RPC_TIMEOUT (10.0 -> 2.0) and DIAL_TIMEOUT (5.0 -> 1.0) defaults so a
   single ghost peer no longer eats 15s before the round can advance.

C. Stall-detection in ``_iterative_get_value`` mirroring the existing
   logic in ``_iterative_find_node``: two consecutive rounds with no new
   closer peers terminate the lookup early instead of burning the
   remaining ``MAX_LOOKUP_ROUNDS``.
"""

import asyncio
import time

from kademlite.dht import DIAL_TIMEOUT, RPC_TIMEOUT, DhtNode
from kademlite.routing import kad_key, xor_distance

# ---------------------------------------------------------------------------
# Fix B: timeout default regression guards
# ---------------------------------------------------------------------------


def test_rpc_timeout_default_is_2_seconds():
    """v0.3.1 dropped RPC_TIMEOUT from 10.0 to 2.0."""
    assert RPC_TIMEOUT == 2.0


def test_dial_timeout_default_is_1_second():
    """v0.3.1 dropped DIAL_TIMEOUT from 5.0 to 1.0."""
    assert DIAL_TIMEOUT == 1.0


def test_dhtnode_inherits_new_defaults():
    """A no-arg DhtNode picks up the new module-level defaults."""
    node = DhtNode()
    assert node.rpc_timeout == 2.0
    assert node.dial_timeout == 1.0


def test_dhtnode_overrides_still_work():
    """The constructor knobs remain overridable for high-latency overlays."""
    node = DhtNode(rpc_timeout=10.0, dial_timeout=5.0)
    assert node.rpc_timeout == 10.0
    assert node.dial_timeout == 5.0


# ---------------------------------------------------------------------------
# Helpers for Fix A / C: in-process iterative-lookup harness
# ---------------------------------------------------------------------------


def _seed_peer(node: DhtNode, peer_id: bytes, addrs: list[bytes]) -> None:
    """Inject a peer into the routing table without dialling.

    The ``_iterative_*`` methods seed candidates from the routing table,
    so this is enough to drive them in unit tests with stubbed RPCs.
    """
    node.routing_table.add_or_update(peer_id, addrs)
    node.routing_table.mark_connected(peer_id)


def _make_peer_id(seed: int) -> bytes:
    """Deterministic 32-byte peer ID from an integer seed."""
    return seed.to_bytes(32, "big")


def _xor_sorted_seed_peers(
    node: DhtNode, key: bytes, n_peers: int, seed_offset: int = 100
) -> list[bytes]:
    """Generate ``n_peers`` deterministic peer IDs, sort them by XOR
    distance to ``kad_key(key)``, and seed them into the routing table
    in that order.

    Returns the peer IDs sorted by XOR-closeness to the lookup key (so
    ``result[0]`` is the closest, ``result[n-1]`` is the furthest).
    Tests that depend on "the Nth closest peer" must use this rather
    than relying on insertion order, because ``_iterative_*`` methods
    select candidates by ``xor_distance(kad_key(peer_id), kad_key(key))``
    and SHA-256 destroys any correlation between numeric seed and XOR
    distance.
    """
    candidates = [_make_peer_id(seed_offset + i) for i in range(n_peers)]
    key_kad = kad_key(key)
    candidates.sort(key=lambda p: xor_distance(kad_key(p), key_kad))
    for i, pid in enumerate(candidates):
        _seed_peer(node, pid, [f"/p{i}".encode()])
    return candidates


# ---------------------------------------------------------------------------
# Fix A: GET returns on first responder, doesn't barrier on slow peer
# ---------------------------------------------------------------------------


async def test_get_returns_on_first_responder_not_slow_peer():
    """One peer returns the value at 50ms; a sibling hangs for >1s.

    Prior behavior (gather-based) waited for the slow peer to time out
    before the value-found check fired. v0.3.1's FIRST_COMPLETED loop
    must return immediately on the fast responder.
    """
    node = DhtNode(rpc_timeout=5.0, dial_timeout=5.0, alpha=3, k=20)
    fast_peer = _make_peer_id(1)
    slow_peer = _make_peer_id(2)
    medium_peer = _make_peer_id(3)
    _seed_peer(node, fast_peer, [b"/fast"])
    _seed_peer(node, slow_peer, [b"/slow"])
    _seed_peer(node, medium_peer, [b"/medium"])

    target_value = b"hello v0.3.1"

    async def stub_get_value_single(peer_id, addrs, key):
        if peer_id == fast_peer:
            await asyncio.sleep(0.05)
            return target_value, None, []
        if peer_id == medium_peer:
            await asyncio.sleep(0.10)
            return None, None, []
        # Slow peer: simulate a ghost that hangs forever (cancelled by
        # the FIRST_COMPLETED early-return path).
        await asyncio.sleep(10.0)
        return None, None, []

    node._get_value_single = stub_get_value_single  # type: ignore[assignment]

    t0 = time.monotonic()
    result = await node._iterative_get_value(b"some-key")
    elapsed = time.monotonic() - t0

    assert result == target_value
    # Must return well before the slow peer's 10s hang. Generous bound
    # for CI variance, but tight enough to fail if the round barriers.
    assert elapsed < 1.0, f"GET took {elapsed:.2f}s; expected <1s with FIRST_COMPLETED"


async def test_get_cancels_pending_tasks_after_value_found():
    """When the fast peer surfaces a value, slow peers are cancelled
    rather than left running. Keeps the event loop tidy and doesn't
    leak background work past the lookup."""
    node = DhtNode(rpc_timeout=5.0, dial_timeout=5.0, alpha=3, k=20)
    fast_peer = _make_peer_id(1)
    slow_peer = _make_peer_id(2)
    _seed_peer(node, fast_peer, [b"/fast"])
    _seed_peer(node, slow_peer, [b"/slow"])

    cancelled = asyncio.Event()

    async def stub_get_value_single(peer_id, addrs, key):
        if peer_id == fast_peer:
            await asyncio.sleep(0.01)
            return b"v", None, []
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return None, None, []

    node._get_value_single = stub_get_value_single  # type: ignore[assignment]
    result = await node._iterative_get_value(b"k")
    assert result == b"v"
    # Yield once so the cancellation propagates.
    await asyncio.sleep(0.01)
    assert cancelled.is_set(), "slow peer task should be cancelled after value found"


async def test_get_value_single_honors_dial_timeout_in_production():
    """End-to-end test of the production ``_get_value_single`` dial-timeout
    path: stub ``peer_store.get_or_dial`` to hang and verify the configured
    ``dial_timeout`` fires, the peer is marked disconnected, and the
    lookup terminates well inside the budget. Exercises the real
    ``asyncio.wait_for`` inside ``_get_value_single`` (not a re-stubbed
    one), so the test fails if the dial_timeout plumbing breaks.

    The lookup is wrapped in ``asyncio.wait_for`` with a generous
    wall-clock cap so a regression that breaks the dial_timeout path
    surfaces as a bounded ``TimeoutError`` instead of hanging the suite
    indefinitely (pytest-timeout is not in deps)."""
    node = DhtNode(rpc_timeout=5.0, dial_timeout=0.15, alpha=2, k=20)
    ghost = _make_peer_id(1)
    _seed_peer(node, ghost, [b"/g"])

    async def hanging_get_or_dial(peer_id, addrs):
        await asyncio.Event().wait()  # never returns

    node.peer_store.get_or_dial = hanging_get_or_dial  # type: ignore[assignment]

    t0 = time.monotonic()
    # Outer cap: 2s is comfortably past the 0.15s dial_timeout but
    # short enough that a regression fails fast.
    result = await asyncio.wait_for(node._iterative_get_value(b"k"), timeout=2.0)
    elapsed = time.monotonic() - t0

    assert result is None
    # The single seeded peer's dial fires once for ~0.15s, gets marked
    # disconnected, then no further candidates -> lookup terminates.
    assert elapsed < 0.6, f"dial timeout not honored: {elapsed:.2f}s"
    entry = node.routing_table.find(ghost)
    assert entry is not None and entry.connected is False, (
        "_get_value_single should mark a hung-dial peer as disconnected"
    )


# ---------------------------------------------------------------------------
# Fix C: stall detection in _iterative_get_value
# ---------------------------------------------------------------------------


async def test_get_stall_detection_terminates_early():
    """When peers consistently respond with no new closer peers and no
    value, the alpha-refill walk terminates after the second stall
    rather than burning the full ``MAX_LOOKUP_ROUNDS * alpha`` budget.

    With alpha=2 and stall_threshold = 2*alpha = 4, the walk fires
    4 no-progress completions -> first stall -> boost parallelism to
    alpha*BOOST=4 -> 4 more no-progress completions -> second stall ->
    break. Total cap on this configuration: 8 calls. Without stall
    detection, the cap would be ``MAX_LOOKUP_ROUNDS * alpha = 20``."""
    node = DhtNode(rpc_timeout=5.0, dial_timeout=5.0, alpha=2, k=20)
    seeded = []
    for i in range(20):
        pid = _make_peer_id(100 + i)
        _seed_peer(node, pid, [f"/p{i}".encode()])
        seeded.append(pid)

    call_count = 0

    async def stub_get_value_single(peer_id, addrs, key):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.001)
        return None, None, []

    node._get_value_single = stub_get_value_single  # type: ignore[assignment]

    result = await node._iterative_get_value(b"k")
    assert result is None
    # Stall caps the walk at alpha*2 (pre-boost) + alpha*2*BOOST (post-boost)
    # = 2*2 + 4*2 = 12 in the worst overlap. Looser bound here for
    # CI variance and to allow the next-fill window to finish in flight.
    assert call_count <= 12, f"stall detection didn't fire: {call_count} calls"
    # Must terminate well before the MAX_LOOKUP_ROUNDS*alpha budget would
    # have allowed (20 with alpha=2).
    assert call_count < 20


# ---------------------------------------------------------------------------
# PUT-fanout structure under FIRST_COMPLETED
# ---------------------------------------------------------------------------


async def test_put_fanout_tallies_as_responses_arrive():
    """The PUT path uses FIRST_COMPLETED to tally per-peer acceptances
    as they arrive. Wall time is still bounded by the slowest task -
    PUT needs the full success count, so the loop drains all K - but
    the structure preserves correct cancellation propagation when the
    caller cancels the PUT mid-fanout (see the dedicated test below).
    The latency win for PUT comes from the dropped DIAL_TIMEOUT and
    RPC_TIMEOUT defaults, not the loop structure."""
    node = DhtNode(rpc_timeout=5.0, dial_timeout=5.0, alpha=3, k=4)
    peers = [_make_peer_id(200 + i) for i in range(4)]
    for i, pid in enumerate(peers):
        _seed_peer(node, pid, [f"/peer{i}".encode()])

    # Stub _put_to_peer with varying delays. Three respond in <50ms;
    # one is slow but eventually returns False.
    async def stub_put_to_peer(peer_id, addrs, key, value, publisher=None, ttl_secs=None):
        idx = peers.index(peer_id)
        if idx == 3:
            await asyncio.sleep(0.5)
            return False
        await asyncio.sleep(0.01 * (idx + 1))
        return True

    node._put_to_peer = stub_put_to_peer  # type: ignore[assignment]
    # Stub _iterative_find_node so put() doesn't try to do a real lookup.
    async def stub_find_node(target):
        return [(pid, [f"/peer{i}".encode()]) for i, pid in enumerate(peers)]

    node._iterative_find_node = stub_find_node  # type: ignore[assignment]

    t0 = time.monotonic()
    n = await node.put(b"k", b"v", ttl=60)
    elapsed = time.monotonic() - t0

    # All four peers responded; three accepted.
    assert n == 3
    # The loop still drains all tasks (we need the full success count),
    # so wall time is bounded by the slowest. This test mainly guards
    # the loop structure: if we wired up early-exit-on-K-success in a
    # future revision, this assertion would relax.
    assert elapsed < 1.5, f"PUT took {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# Codex audit follow-ups
# ---------------------------------------------------------------------------


async def test_get_stall_boost_widens_query_before_giving_up():
    """The GET stall-detection's parallelism boost must actually widen
    the dispatch frontier so peers beyond the no-boost reach get
    queried. Otherwise a value living past the pre-stall frontier
    returns ``None``.

    Setup: alpha=3 with the candidate set sized so the boost is
    structurally required. Under the alpha-refill walk at alpha=3 with
    stall_threshold=6, the no-boost dispatch frontier covers ~9 peers
    (3 initial + ~6 refilled before the stall fires, then the in_flight
    drain covers the last few without launching new tasks). Placing 13
    peers and putting the value-holder at sorted-XOR position 10 means
    a no-boost walk would terminate at ~9 peers queried and miss the
    value; the boost to parallelism = ``alpha * STALL_PARALLELISM_BOOST``
    = 6 dispatches the next batch and reaches peer 10.

    Peer roles are pinned to actual XOR order against the lookup key,
    not seed order, because SHA-256 destroys any correlation between
    numeric seed and XOR distance.
    """
    lookup_key = b"k"
    node = DhtNode(rpc_timeout=5.0, dial_timeout=5.0, alpha=3, k=20)
    peers = _xor_sorted_seed_peers(node, lookup_key, n_peers=13, seed_offset=300)

    value_holder = peers[10]  # 11th-closest in actual XOR order
    target_value = b"deep value"

    queried_peers: list[bytes] = []

    async def stub_get_value_single(peer_id, addrs, key):
        queried_peers.append(peer_id)
        await asyncio.sleep(0.001)
        if peer_id == value_holder:
            return target_value, None, []
        return None, None, []

    node._get_value_single = stub_get_value_single  # type: ignore[assignment]

    result = await node._iterative_get_value(lookup_key)

    holder_state = "queried" if value_holder in queried_peers else "NOT queried"
    assert result == target_value, (
        f"value not found: queried {len(queried_peers)} peers; "
        f"value_holder at index 10 was {holder_state}; "
        "boost path is missing or broken"
    )
    # Sanity: the boost must have run (we needed to dispatch beyond the
    # no-boost frontier of ~9 peers to reach index 10).
    assert value_holder in queried_peers
    assert queried_peers.index(value_holder) >= 9, (
        "value_holder was queried within the no-boost frontier; "
        "test setup does not actually exercise the boost path"
    )


async def test_find_node_propagates_caller_cancellation_to_peer_tasks():
    """If the parent lookup is cancelled mid-walk, ALL pending per-peer
    tasks must be cancelled - not just the one that happened to fire
    its except clause first. ``asyncio.gather`` used to propagate
    cancellation implicitly to every submitted awaitable; the
    FIRST_COMPLETED + ``in_flight`` set pattern needs explicit
    cancellation in the ``finally`` clause to match.

    The test seeds two peers, dispatches both via the alpha-refill walk,
    cancels the parent, and asserts BOTH peer tasks recorded a
    CancelledError - not just one.
    """
    node = DhtNode(rpc_timeout=5.0, dial_timeout=5.0, alpha=2, k=20)
    a = _make_peer_id(1)
    b = _make_peer_id(2)
    _seed_peer(node, a, [b"/a"])
    _seed_peer(node, b, [b"/b"])

    cancelled: set[bytes] = set()
    started: set[bytes] = set()
    both_started = asyncio.Event()

    async def hanging_find_single(peer_id, addrs, target):
        started.add(peer_id)
        if {a, b} <= started:
            both_started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.add(peer_id)
            raise
        return []

    node._find_node_single = hanging_find_single  # type: ignore[assignment]

    parent = asyncio.create_task(node._iterative_find_node(b"target"))
    # Wait for BOTH children to be running before cancelling, so we know
    # both should be cancelled by the propagation.
    await both_started.wait()
    parent.cancel()
    try:
        await parent
    except asyncio.CancelledError:
        pass
    # Yield once so the cancellation drain runs.
    await asyncio.sleep(0.01)
    assert cancelled == {a, b}, (
        f"find_node parent cancel should propagate to all in-flight peer "
        f"tasks; got {cancelled!r}, expected {{a, b}}"
    )


async def test_get_does_not_cancel_slow_in_flight_value_holder_on_stall():
    """When stall fires, in-flight tasks must NOT be cancelled - the
    closest peer in the alpha window may be slow but holds the value.
    Cancelling it for "convergence" produces a false negative.

    Reproduction shape: alpha=2, the closest peer (by actual XOR order)
    is slow (200ms) but holds the value; several other peers respond
    fast with no progress, driving the stall counter past threshold
    while the slow peer is still in flight. With the bug, the finally
    clause cancels the slow peer and returns None. Fixed code lets
    in_flight drain without cancelling, so the value comes through.

    Peer roles pinned to actual XOR order via ``_xor_sorted_seed_peers``
    so the slow value-holder is guaranteed to occupy one of the initial
    alpha=2 dispatch slots.
    """
    lookup_key = b"k"
    node = DhtNode(rpc_timeout=5.0, dial_timeout=5.0, alpha=2, k=20)
    peers = _xor_sorted_seed_peers(node, lookup_key, n_peers=8, seed_offset=400)
    slow_holder = peers[0]  # closest peer in actual XOR order

    target_value = b"slow-but-found"

    async def stub_get_value_single(peer_id, addrs, key):
        if peer_id == slow_holder:
            await asyncio.sleep(0.2)
            return target_value, None, []
        await asyncio.sleep(0.001)
        return None, None, []

    node._get_value_single = stub_get_value_single  # type: ignore[assignment]

    t0 = time.monotonic()
    result = await node._iterative_get_value(lookup_key)
    elapsed = time.monotonic() - t0

    assert result == target_value, (
        "stall-cancellation regression: slow in-flight value-holder was "
        "cancelled instead of allowed to drain"
    )
    # Total time bounded by slow_holder's 200ms hang plus small overhead.
    assert elapsed < 0.6, f"got result but took too long: {elapsed:.2f}s"


async def test_get_alpha_refill_advances_past_slow_ghost_in_multi_hop():
    """Codex/CodeRabbit headline check: a multi-hop GET must not pay the
    per-round drain on a slow ghost peer in the alpha window.

    Setup: alpha=3 with three seeded peers. Two are "fast routers" that
    respond at ~30ms with a pointer to a final "value holder" peer; the
    third is a ghost that hangs for 5 seconds. Under the prior
    drain-the-batch model, the round would barrier on the ghost before
    the value-holder peer is queried, so total time would be >= ghost
    hang. Under the alpha-refill walk, a fast router returns, its new
    candidate (the value-holder) gets refilled into the in-flight window
    immediately, the value comes back fast, and the lookup terminates
    well inside the ghost's hang.
    """
    node = DhtNode(rpc_timeout=5.0, dial_timeout=5.0, alpha=3, k=20)
    fast_a = _make_peer_id(1)
    fast_b = _make_peer_id(2)
    ghost = _make_peer_id(3)
    holder = _make_peer_id(4)  # "discovered" by fast routers, holds value

    _seed_peer(node, fast_a, [b"/fa"])
    _seed_peer(node, fast_b, [b"/fb"])
    _seed_peer(node, ghost, [b"/g"])
    # holder is NOT seeded; it must be discovered through the routers.

    target_value = b"under-budget"

    async def stub_get_value_single(peer_id, addrs, key):
        if peer_id == fast_a or peer_id == fast_b:
            await asyncio.sleep(0.03)
            return None, None, [(holder, [b"/h"])]
        if peer_id == holder:
            await asyncio.sleep(0.03)
            return target_value, None, []
        # ghost
        await asyncio.sleep(5.0)
        return None, None, []

    node._get_value_single = stub_get_value_single  # type: ignore[assignment]

    t0 = time.monotonic()
    result = await node._iterative_get_value(b"k")
    elapsed = time.monotonic() - t0

    assert result == target_value
    # Hop 1 (~30ms) + refill + hop 2 (~30ms) = ~60-100ms.
    # Must be far under the ghost's 5s hang, AND under Pixel's 1s SLO.
    assert elapsed < 0.5, (
        f"multi-hop GET took {elapsed:.2f}s; alpha-refill is not "
        f"advancing past the in-flight ghost"
    )


async def test_put_propagates_caller_cancellation_to_peer_tasks():
    """Codex audit medium #2 sibling: PUT-fanout cancellation propagation."""
    node = DhtNode(rpc_timeout=5.0, dial_timeout=5.0, alpha=3, k=4)
    peers = [_make_peer_id(400 + i) for i in range(3)]
    for i, pid in enumerate(peers):
        _seed_peer(node, pid, [f"/p{i}".encode()])

    cancelled_count = 0
    started = asyncio.Event()

    async def hanging_put_to_peer(peer_id, addrs, key, value, publisher=None, ttl_secs=None):
        nonlocal cancelled_count
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled_count += 1
            raise
        return False

    node._put_to_peer = hanging_put_to_peer  # type: ignore[assignment]

    async def stub_find_node(target):
        return [(pid, [f"/p{i}".encode()]) for i, pid in enumerate(peers)]

    node._iterative_find_node = stub_find_node  # type: ignore[assignment]

    parent = asyncio.create_task(node.put(b"k", b"v"))
    await started.wait()
    parent.cancel()
    try:
        await parent
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0.01)
    assert cancelled_count == len(peers), (
        f"PUT parent cancel should propagate to all {len(peers)} peer tasks; "
        f"only {cancelled_count} cancelled"
    )
