# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Yamux protocol-level unit tests using a fake in-memory transport pair.

Covers:
- read_exact() across multiple data frames sends a single batched window update
  (not per-frame).
- read() after stream close returns EOF without raising.
- read_exact() after stream close raises ConnectionError.
- A data frame larger than DEFAULT_WINDOW_SIZE causes the stream to RST and close.
- accept_stream() raises ConnectionError after GO_AWAY.
- GO_AWAY frame stops the read loop and unblocks pending acceptors.

The transport interface yamux requires is just write_msg(bytes) and
read_msg() -> bytes. We wire two FakeTransport instances together in memory.
"""

import asyncio

import pytest

from kademlite.yamux import (
    DEFAULT_WINDOW_SIZE,
    FLAG_RST,
    FLAG_SYN,
    HEADER_SIZE,
    TYPE_DATA,
    TYPE_GO_AWAY,
    TYPE_WINDOW_UPDATE,
    YamuxSession,
    _decode_header,
    _encode_header,
)


class FakeTransport:
    """In-memory bidirectional transport pair.

    A FakeTransport pair is created via FakeTransport.pair(). Each side has
    write_msg(bytes) and read_msg() -> bytes; writes on one side appear as
    reads on the peer's side.
    """

    def __init__(self) -> None:
        self._inbox: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.peer: FakeTransport | None = None
        self.closed = False

    @classmethod
    def pair(cls) -> tuple["FakeTransport", "FakeTransport"]:
        a = cls()
        b = cls()
        a.peer = b
        b.peer = a
        return a, b

    async def write_msg(self, data: bytes) -> None:
        if self.peer is None or self.peer.closed:
            raise ConnectionError("peer closed")
        self.peer._inbox.put_nowait(bytes(data))

    async def read_msg(self) -> bytes:
        msg = await self._inbox.get()
        if msg is None:
            raise ConnectionError("transport closed")
        return msg

    def close(self) -> None:
        self.closed = True
        self._inbox.put_nowait(None)


async def _make_session_pair() -> tuple[YamuxSession, YamuxSession, FakeTransport, FakeTransport]:
    """Create a connected pair of YamuxSessions running over a FakeTransport pair."""
    t_init, t_resp = FakeTransport.pair()
    initiator = YamuxSession(t_init, is_initiator=True)
    responder = YamuxSession(t_resp, is_initiator=False)
    await initiator.start()
    await responder.start()
    return initiator, responder, t_init, t_resp


# ---------------------------------------------------------------------------
# Window update batching in read_exact
# ---------------------------------------------------------------------------


async def test_read_exact_batches_window_update_across_frames() -> None:
    """read_exact() consuming N bytes split across multiple DATA frames must
    emit a SINGLE window update (not one per frame). We instrument
    _send_window_update to count its calls."""
    initiator, responder, _, _ = await _make_session_pair()
    try:
        client_stream = await initiator.open_stream()
        # Wait for responder to register the stream
        server_stream = await asyncio.wait_for(responder.accept_stream(), timeout=2.0)

        # Instrument _send_window_update to count invocations
        update_calls: list[int] = []
        original = server_stream._send_window_update

        async def counting_update(delta: int) -> None:
            update_calls.append(delta)
            await original(delta)

        server_stream._send_window_update = counting_update

        # Send three small chunks that together total 600 bytes
        chunks = [b"a" * 200, b"b" * 200, b"c" * 200]
        for chunk in chunks:
            await client_stream.write(chunk)

        # Wait for all three frames to arrive at the responder
        for _ in range(20):
            if server_stream._recv_buffer.qsize() >= 3:
                break
            await asyncio.sleep(0.02)

        total = sum(len(c) for c in chunks)
        data = await asyncio.wait_for(server_stream.read_exact(total), timeout=2.0)
        assert data == b"".join(chunks)

        # Allow the window update frame to be sent
        await asyncio.sleep(0.05)

        # read_exact must batch into exactly one window update, not one per
        # data frame consumed.
        assert len(update_calls) == 1, (
            f"expected exactly 1 batched window update, got {len(update_calls)}: "
            f"{update_calls}"
        )
        assert update_calls[0] == total, (
            f"batched delta should equal total bytes consumed ({total}), "
            f"got {update_calls[0]}"
        )

        # And the local window must have been restored to its starting value
        # (initial DEFAULT_WINDOW_SIZE - total + total = DEFAULT_WINDOW_SIZE).
        assert server_stream._recv_window == DEFAULT_WINDOW_SIZE
    finally:
        await initiator.stop()
        await responder.stop()


# ---------------------------------------------------------------------------
# Closed-stream read behavior
# ---------------------------------------------------------------------------


async def test_read_after_close_returns_eof() -> None:
    """Once a stream is closed locally, read() returns empty bytes (EOF)."""
    initiator, responder, _, _ = await _make_session_pair()
    try:
        client_stream = await initiator.open_stream()
        await asyncio.wait_for(responder.accept_stream(), timeout=2.0)
        await client_stream.close()

        result = await client_stream.read()
        assert result == b""
    finally:
        await initiator.stop()
        await responder.stop()


async def test_read_exact_after_close_raises() -> None:
    """read_exact() after the stream is closed must raise ConnectionError."""
    initiator, responder, _, _ = await _make_session_pair()
    try:
        client_stream = await initiator.open_stream()
        await asyncio.wait_for(responder.accept_stream(), timeout=2.0)
        await client_stream.close()

        with pytest.raises(ConnectionError):
            await client_stream.read_exact(5)
    finally:
        await initiator.stop()
        await responder.stop()


# ---------------------------------------------------------------------------
# Window violation -> RST
# ---------------------------------------------------------------------------


async def test_oversized_data_frame_resets_stream() -> None:
    """If a peer delivers a DATA frame whose size exceeds the receive window,
    the stream must close and the recipient sends RST."""
    # We do this by directly manipulating one side: open a stream, then
    # synthesize a DATA frame that violates the window.
    initiator, responder, _, _ = await _make_session_pair()
    try:
        client_stream = await initiator.open_stream()
        server_stream = await asyncio.wait_for(responder.accept_stream(), timeout=2.0)

        # Synthesize an oversized DATA frame and inject it into responder
        # by sending it from the initiator side using a raw frame.
        oversized = DEFAULT_WINDOW_SIZE + 1
        payload = b"x" * oversized
        await initiator._send_frame(
            TYPE_DATA, 0, client_stream.stream_id, payload
        )

        # Give the responder time to process
        await asyncio.sleep(0.1)

        # The server-side stream must be closed
        assert server_stream._closed is True
        # And reading from it should return EOF
        result = await asyncio.wait_for(server_stream.read(), timeout=1.0)
        assert result == b""
    finally:
        await initiator.stop()
        await responder.stop()


# ---------------------------------------------------------------------------
# GO_AWAY semantics
# ---------------------------------------------------------------------------


async def test_accept_stream_raises_after_go_away() -> None:
    """If a session's read loop sees GO_AWAY, accept_stream() raises
    ConnectionError to unblock anyone waiting for an inbound stream."""
    initiator, responder, _, _ = await _make_session_pair()
    try:
        # Park an acceptor on the responder side
        accept_task = asyncio.create_task(responder.accept_stream())

        # Give the accept_stream call a moment to actually block
        await asyncio.sleep(0.05)

        # Initiator sends GO_AWAY
        await initiator._send_frame(TYPE_GO_AWAY, 0, 0, b"", length_override=0)

        # The waiting acceptor must raise ConnectionError
        with pytest.raises(ConnectionError):
            await asyncio.wait_for(accept_task, timeout=1.0)
    finally:
        await initiator.stop()
        await responder.stop()


async def test_go_away_stops_read_loop() -> None:
    """After GO_AWAY, the responder's read loop should exit and is_alive
    must become False."""
    initiator, responder, _, _ = await _make_session_pair()
    try:
        assert responder.is_alive
        await initiator._send_frame(TYPE_GO_AWAY, 0, 0, b"", length_override=0)
        # Read loop should terminate shortly
        for _ in range(20):
            if not responder.is_alive:
                break
            await asyncio.sleep(0.05)
        assert not responder.is_alive
    finally:
        await initiator.stop()
        await responder.stop()


# ---------------------------------------------------------------------------
# Header encoding round-trip (covers _encode_header / _decode_header)
# ---------------------------------------------------------------------------


def test_header_roundtrip() -> None:
    """_encode_header / _decode_header are inverses."""
    header = _encode_header(TYPE_WINDOW_UPDATE, FLAG_SYN, 7, 256 * 1024)
    assert len(header) == HEADER_SIZE
    version, type_, flags, stream_id, length = _decode_header(header)
    assert version == 0
    assert type_ == TYPE_WINDOW_UPDATE
    assert flags == FLAG_SYN
    assert stream_id == 7
    assert length == 256 * 1024


def test_header_rst_flag_roundtrip() -> None:
    header = _encode_header(TYPE_DATA, FLAG_RST, 3, 0)
    _, _, flags, sid, _ = _decode_header(header)
    assert flags == FLAG_RST
    assert sid == 3
