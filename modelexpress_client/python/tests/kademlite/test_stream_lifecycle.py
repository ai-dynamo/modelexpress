# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource-release tests for the v0.2.0 stream-cleanup commits.

These tests assert observable resource release - specifically, that
``YamuxSession.live_streams_count`` returns to a baseline after a
fault path - rather than mocking ``stream.close()`` to verify the
line was called. Mocked-close tests are performative coverage; they
go green while the transport still leaks in production.

Coverage targets:

- 5493aa4: outbound Kad RPCs (kad_get_value/put/find_node) close their
  stream when the response read times out or is cancelled.
- 6630070, 44cea2e: outbound identify pull/push close on every path.
- be51fee: Connection.open_stream closes the YamuxStream when
  multistream negotiation raises (Exception or CancelledError).
- 7350dea: inbound _negotiate_inbound_stream closes the stream when
  negotiation fails or no handler claims the protocol.
- fae4027: inbound identify response/push handlers close their stream
  in finally, even when the body raises.
"""

import asyncio
from unittest.mock import patch

import pytest  # noqa: F401

from kademlite.connection import Connection
from kademlite.crypto import _base58btc_encode
from kademlite.dht import DhtNode
from kademlite.kademlia import (
    KADEMLIA_PROTOCOL,
    kad_find_node,
    kad_get_value,
    kad_put_value,
)


def _node_multiaddr(node: DhtNode) -> str:
    host, port = node.listen_addr
    return f"/ip4/{host}/tcp/{port}/p2p/{_base58btc_encode(node.peer_id)}"


# ---------------------------------------------------------------------------
# Outbound Kad RPC stream cleanup (commit 5493aa4)
# ---------------------------------------------------------------------------


async def _two_connected_nodes():
    """Helper: spin up two DhtNodes, bootstrap b to a, return (a, b).

    Caller is responsible for stopping both.
    """
    node_a = DhtNode()
    await node_a.start("127.0.0.1", 0)
    node_b = DhtNode()
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node_a)])
    # Allow bootstrap and identify to settle.
    await asyncio.sleep(0.2)
    return node_a, node_b


def _conn_to(src: DhtNode, dst: DhtNode) -> Connection:
    """Get the Connection object on ``src`` that talks to ``dst``."""
    conn = src.peer_store.get_connection(dst.peer_id)
    if conn is None:
        raise AssertionError(
            f"no connection from {_base58btc_encode(src.peer_id)[:16]} to "
            f"{_base58btc_encode(dst.peer_id)[:16]}; bootstrap may not have settled"
        )
    return conn


async def test_kad_get_value_releases_stream_on_failure() -> None:
    """When the remote handler doesn't send a response, kad_get_value
    fails (timeout, EOF, or cancellation); the outbound stream must be
    released either way. The diagnostic is live_streams_count returning
    to baseline, not which exception was raised."""
    node_a, node_b = await _two_connected_nodes()
    try:
        conn = _conn_to(node_a, node_b)
        baseline = conn.yamux.live_streams_count

        # Make node_b's KadHandler hang forever after accepting the stream
        # so it never writes a response. node_a's read either times out or
        # sees EOF when its own cleanup races - either is fine.
        hang_event = asyncio.Event()  # never set

        async def hang(*args, **kwargs):
            await hang_event.wait()

        with patch.object(node_b.kad_handler, "handle_stream", new=hang):
            with pytest.raises((asyncio.TimeoutError, asyncio.IncompleteReadError)):
                await asyncio.wait_for(
                    kad_get_value(conn, b"/test/key"), timeout=0.2
                )

        await asyncio.sleep(0.05)
        assert conn.yamux.live_streams_count == baseline, (
            f"outbound stream leaked after failed RPC: live_streams_count "
            f"went from {baseline} -> {conn.yamux.live_streams_count} "
            f"(ids: {conn.yamux.live_stream_ids})"
        )
    finally:
        await node_a.stop()
        await node_b.stop()


async def test_kad_put_value_releases_stream_on_failure() -> None:
    """Same shape for kad_put_value."""
    node_a, node_b = await _two_connected_nodes()
    try:
        conn = _conn_to(node_a, node_b)
        baseline = conn.yamux.live_streams_count

        hang_event = asyncio.Event()

        async def hang(*args, **kwargs):
            await hang_event.wait()

        with patch.object(node_b.kad_handler, "handle_stream", new=hang):
            with pytest.raises((asyncio.TimeoutError, asyncio.IncompleteReadError)):
                await asyncio.wait_for(
                    kad_put_value(conn, b"/test/key", b"value"), timeout=0.2
                )

        await asyncio.sleep(0.05)
        assert conn.yamux.live_streams_count == baseline, (
            f"kad_put_value leaked stream: {baseline} -> "
            f"{conn.yamux.live_streams_count}"
        )
    finally:
        await node_a.stop()
        await node_b.stop()


async def test_kad_find_node_releases_stream_on_failure() -> None:
    """Same shape for kad_find_node."""
    node_a, node_b = await _two_connected_nodes()
    try:
        conn = _conn_to(node_a, node_b)
        baseline = conn.yamux.live_streams_count

        hang_event = asyncio.Event()

        async def hang(*args, **kwargs):
            await hang_event.wait()

        with patch.object(node_b.kad_handler, "handle_stream", new=hang):
            with pytest.raises((asyncio.TimeoutError, asyncio.IncompleteReadError)):
                await asyncio.wait_for(
                    kad_find_node(conn, b"target"), timeout=0.2
                )

        await asyncio.sleep(0.05)
        assert conn.yamux.live_streams_count == baseline
    finally:
        await node_a.stop()
        await node_b.stop()


async def test_repeated_kad_rpc_failures_do_not_grow_stream_count() -> None:
    """Sustained failed RPCs over the same connection must not accumulate
    streams - the per-call cleanup must be reliable, not just one-shot.

    This is the ratchet test: even if a single failure leaks half a stream
    or some bookkeeping slop, 20 consecutive failures would surface the
    leak as monotonic growth in live_streams_count.
    """
    node_a, node_b = await _two_connected_nodes()
    try:
        conn = _conn_to(node_a, node_b)
        baseline = conn.yamux.live_streams_count

        hang_event = asyncio.Event()

        async def hang(*args, **kwargs):
            await hang_event.wait()

        with patch.object(node_b.kad_handler, "handle_stream", new=hang):
            for _ in range(20):
                try:
                    await asyncio.wait_for(
                        kad_get_value(conn, b"/probe"), timeout=0.05
                    )
                except Exception:
                    pass  # any failure is fine; we care about cleanup

        await asyncio.sleep(0.1)
        assert conn.yamux.live_streams_count == baseline, (
            f"20 failed kad_get_value calls leaked streams: baseline "
            f"{baseline} -> {conn.yamux.live_streams_count} "
            f"(ids: {conn.yamux.live_stream_ids})"
        )
    finally:
        await node_a.stop()
        await node_b.stop()


# ---------------------------------------------------------------------------
# Outbound negotiation failure (commit be51fee)
# ---------------------------------------------------------------------------


async def test_open_stream_closes_yamux_stream_on_negotiation_exception() -> None:
    """Connection.open_stream must close the YamuxStream when
    negotiate_outbound raises a regular Exception, not leak it."""
    node_a, node_b = await _two_connected_nodes()
    try:
        conn = _conn_to(node_a, node_b)
        baseline = conn.yamux.live_streams_count

        with patch(
            "kademlite.connection.negotiate_outbound",
            side_effect=RuntimeError("boom during negotiation"),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                await conn.open_stream("/some/proto")

        await asyncio.sleep(0.05)
        assert conn.yamux.live_streams_count == baseline, (
            f"failed negotiation leaked yamux stream: {baseline} -> "
            f"{conn.yamux.live_streams_count}"
        )
    finally:
        await node_a.stop()
        await node_b.stop()


async def test_open_stream_closes_yamux_stream_on_cancellation() -> None:
    """Connection.open_stream must catch BaseException (specifically
    CancelledError) and close the YamuxStream before re-raising."""
    node_a, node_b = await _two_connected_nodes()
    try:
        conn = _conn_to(node_a, node_b)
        baseline = conn.yamux.live_streams_count

        # Pass the exception CLASS as side_effect so AsyncMock instantiates
        # and raises it at await-time. Passing an async-callable side_effect
        # is unreliable here: AsyncMock can return the unawaited coroutine
        # object instead of raising, which would silently fail to exercise
        # the cancellation path under test.
        with patch(
            "kademlite.connection.negotiate_outbound",
            side_effect=asyncio.CancelledError,
        ):
            with pytest.raises(asyncio.CancelledError):
                await conn.open_stream("/some/proto")

        await asyncio.sleep(0.05)
        assert conn.yamux.live_streams_count == baseline, (
            f"cancelled negotiation leaked yamux stream: {baseline} -> "
            f"{conn.yamux.live_streams_count}"
        )

        # Connection still usable for a real stream.
        stream, reader, writer = await conn.open_stream(KADEMLIA_PROTOCOL)
        await stream.close()
        await asyncio.sleep(0.05)
        assert conn.yamux.live_streams_count == baseline
    finally:
        await node_a.stop()
        await node_b.stop()


# ---------------------------------------------------------------------------
# Inbound stream cleanup (commit 7350dea)
# ---------------------------------------------------------------------------


class _FakeYamuxStream:
    """Minimal stand-in for YamuxStream that tracks close() calls."""

    def __init__(self, stream_id: int = 1) -> None:
        self.stream_id = stream_id
        self.close_calls = 0
        self._closed = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def close(self) -> None:
        self.close_calls += 1
        self._closed = True


async def test_negotiate_inbound_stream_closes_on_negotiation_failure() -> None:
    """If negotiate_inbound raises, the inbound YamuxStream must be closed
    even though it was already accepted (no caller will close it for us)."""
    node_a = DhtNode()
    await node_a.start("127.0.0.1", 0)
    node_b = DhtNode()
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node_a)])
    await asyncio.sleep(0.2)
    try:
        conn = _conn_to(node_a, node_b)
        fake = _FakeYamuxStream(stream_id=42)

        with patch(
            "kademlite.connection.negotiate_inbound",
            side_effect=RuntimeError("negotiate boom"),
        ):
            await conn._negotiate_inbound_stream(fake)

        assert fake.close_calls == 1, (
            f"inbound stream must be closed on negotiation failure; "
            f"close_calls={fake.close_calls}"
        )
        assert fake.is_closed
    finally:
        await node_a.stop()
        await node_b.stop()


async def test_negotiate_inbound_stream_closes_on_unknown_protocol() -> None:
    """If multistream resolves to a protocol with no registered handler,
    the stream must be closed rather than silently dropped."""
    node_a = DhtNode()
    await node_a.start("127.0.0.1", 0)
    node_b = DhtNode()
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node_a)])
    await asyncio.sleep(0.2)
    try:
        conn = _conn_to(node_a, node_b)
        fake = _FakeYamuxStream(stream_id=43)

        with patch(
            "kademlite.connection.negotiate_inbound",
            return_value="/no/such/protocol",
        ):
            await conn._negotiate_inbound_stream(fake)

        assert fake.close_calls == 1
        assert fake.is_closed
    finally:
        await node_a.stop()
        await node_b.stop()


async def test_negotiate_inbound_stream_closes_when_queue_full() -> None:
    """If the handler queue rejects the dispatch (e.g. full), the inbound
    stream must be closed - we are still the last party who can clean it up."""
    node_a = DhtNode()
    await node_a.start("127.0.0.1", 0)
    node_b = DhtNode()
    await node_b.start("127.0.0.1", 0, bootstrap_peers=[_node_multiaddr(node_a)])
    await asyncio.sleep(0.2)
    try:
        conn = _conn_to(node_a, node_b)
        fake = _FakeYamuxStream(stream_id=44)

        # Replace one of the registered handler queues with a full queue
        # to simulate dispatch failure.
        full_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        full_queue.put_nowait("dummy")  # now put_nowait will raise QueueFull
        conn._protocol_handlers["/test/full"] = full_queue

        with patch(
            "kademlite.connection.negotiate_inbound",
            return_value="/test/full",
        ):
            await conn._negotiate_inbound_stream(fake)

        assert fake.close_calls == 1, (
            f"inbound stream must be closed when handler queue is full; "
            f"close_calls={fake.close_calls}"
        )
        assert fake.is_closed
    finally:
        await node_a.stop()
        await node_b.stop()


# ---------------------------------------------------------------------------
# Inbound identify handlers (commit fae4027)
# ---------------------------------------------------------------------------


class _FakeStreamWriter:
    """Stand-in for asyncio.StreamWriter that can be told to fail on drain."""

    def __init__(self, fail_drain: bool = False, fail_write: bool = False) -> None:
        self.writes: list[bytes] = []
        self.fail_drain = fail_drain
        self.fail_write = fail_write

    def write(self, data: bytes) -> None:
        if self.fail_write:
            raise RuntimeError("write boom")
        self.writes.append(data)

    async def drain(self) -> None:
        if self.fail_drain:
            raise RuntimeError("drain boom")


class _FakeStreamReader:
    """Stand-in for asyncio.StreamReader that returns a fixed payload or raises."""

    def __init__(
        self,
        payload: bytes | None = None,
        raise_exc: BaseException | None = None,
    ) -> None:
        self.payload = payload
        self.raise_exc = raise_exc
        self._consumed = False

    async def readexactly(self, n: int) -> bytes:
        if self.raise_exc is not None:
            raise self.raise_exc
        if self._consumed:
            return b""
        self._consumed = True
        return (self.payload or b"")[:n]


async def test_handle_identify_stream_closes_on_writer_failure() -> None:
    """The inbound identify response handler must close its stream even
    when writer.drain() raises mid-response."""
    node = DhtNode()
    await node.start("127.0.0.1", 0)
    try:
        fake_stream = _FakeYamuxStream(stream_id=99)
        fake_reader = _FakeStreamReader(payload=b"")
        fake_writer = _FakeStreamWriter(fail_drain=True)

        # Conn stub - only conn.remote_addr is read by the handler.
        class _StubConn:
            remote_addr = ("127.0.0.1", 4001)
            remote_peer_id = b"\x01" * 32

        await node._handle_identify_stream(_StubConn(), fake_stream, fake_reader, fake_writer)

        assert fake_stream.close_calls == 1, (
            f"identify response stream must be closed on writer failure; "
            f"close_calls={fake_stream.close_calls}"
        )
    finally:
        await node.stop()


async def test_handle_identify_push_stream_closes_on_decode_failure() -> None:
    """The inbound identify push handler must close its stream even when
    the payload decode raises."""
    node = DhtNode()
    await node.start("127.0.0.1", 0)
    try:
        fake_stream = _FakeYamuxStream(stream_id=100)
        # Reader that raises during read - covers the early-failure path.
        fake_reader = _FakeStreamReader(raise_exc=RuntimeError("decode boom"))
        fake_writer = _FakeStreamWriter()

        class _StubConn:
            remote_peer_id = b"\x02" * 32

        await node._handle_identify_push_stream(_StubConn(), fake_stream, fake_reader, fake_writer)

        assert fake_stream.close_calls == 1, (
            f"identify push stream must be closed on read failure; "
            f"close_calls={fake_stream.close_calls}"
        )
    finally:
        await node.stop()
