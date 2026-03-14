# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Yamux stream multiplexer.

Reference: https://github.com/hashicorp/yamux/blob/master/spec.md

Yamux multiplexes multiple logical streams over a single connection.
Each frame has a 12-byte header:

    Version (1) | Type (1) | Flags (2) | StreamID (4) | Length (4)

Types:
    0x00 = Data
    0x01 = Window Update
    0x02 = Ping
    0x03 = Go Away

Flags:
    0x01 = SYN (new stream)
    0x02 = ACK (acknowledge stream)
    0x04 = FIN (half-close)
    0x08 = RST (reset stream)

Flow control: each stream has a receive window (default 256 KB).
The sender must not send more data than the receiver's window allows.
Window updates increase the available window.
"""

import asyncio
import struct
import logging

log = logging.getLogger(__name__)

# Yamux constants
YAMUX_VERSION = 0
HEADER_SIZE = 12
DEFAULT_WINDOW_SIZE = 256 * 1024  # 256 KB
MAX_STREAM_WINDOW = 16 * 1024 * 1024  # 16 MB

# Frame types
TYPE_DATA = 0x00
TYPE_WINDOW_UPDATE = 0x01
TYPE_PING = 0x02
TYPE_GO_AWAY = 0x03

# Flags
FLAG_SYN = 0x0001
FLAG_ACK = 0x0002
FLAG_FIN = 0x0004
FLAG_RST = 0x0008


def _encode_header(type_: int, flags: int, stream_id: int, length: int) -> bytes:
    return struct.pack(">BBHII", YAMUX_VERSION, type_, flags, stream_id, length)


def _decode_header(data: bytes) -> tuple[int, int, int, int, int]:
    """Returns (version, type, flags, stream_id, length)."""
    return struct.unpack(">BBHII", data)


class YamuxStream:
    """A single multiplexed stream within a Yamux session."""

    def __init__(self, session: "YamuxSession", stream_id: int):
        self.session = session
        self.stream_id = stream_id
        self._recv_buffer = asyncio.Queue()
        self._recv_window = DEFAULT_WINDOW_SIZE
        self._send_window = DEFAULT_WINDOW_SIZE
        self._send_window_cond = asyncio.Condition()
        self._closed = False
        self._remote_closed = False
        self._read_excess = bytearray()  # leftover bytes from read_exact

    async def read(self) -> bytes:
        """Read the next chunk of data from this stream.

        Returns empty bytes on EOF.
        """
        if self._closed:
            return b""
        try:
            data = await self._recv_buffer.get()
            if data is None:
                return b""  # EOF
            # Send window update to peer
            await self._send_window_update(len(data))
            return data
        except asyncio.CancelledError:
            return b""

    async def read_exact(self, n: int) -> bytes:
        """Read exactly n bytes from this stream.

        Excess bytes are stored in a local buffer (not pushed back into the
        async queue) to preserve ordering. Batches the window update into a
        single frame instead of one per chunk.
        """
        buf = bytearray()
        total_consumed = 0
        # Drain any leftover from a previous read_exact first
        if self._read_excess:
            taken = bytes(self._read_excess[:n])
            self._read_excess = self._read_excess[len(taken):]
            buf.extend(taken)
        while len(buf) < n:
            # Read raw from the buffer without sending per-chunk window updates
            if self._closed:
                raise ConnectionError(f"stream {self.stream_id} closed, need {n - len(buf)} more bytes")
            try:
                data = await self._recv_buffer.get()
                if data is None:
                    raise ConnectionError(f"stream {self.stream_id} closed, need {n - len(buf)} more bytes")
                total_consumed += len(data)
                buf.extend(data)
            except asyncio.CancelledError:
                raise ConnectionError(f"stream {self.stream_id} cancelled, need {n - len(buf)} more bytes")
        if len(buf) > n:
            self._read_excess.extend(buf[n:])
        # Send a single batched window update for all consumed data
        if total_consumed > 0:
            await self._send_window_update(total_consumed)
        return bytes(buf[:n])

    async def write(self, data: bytes) -> None:
        """Write data to this stream."""
        if self._closed:
            raise ConnectionError(f"stream {self.stream_id} is closed")
        offset = 0
        while offset < len(data):
            # Wait for send window using Condition for atomic wait-and-check.
            # This eliminates the race between Event.clear() and Event.wait()
            # that could lose wakeups under concurrent writes.
            async with self._send_window_cond:
                while self._send_window <= 0:
                    if self._closed:
                        raise ConnectionError(f"stream {self.stream_id} was reset")
                    await self._send_window_cond.wait()
                if self._closed:
                    raise ConnectionError(f"stream {self.stream_id} was reset")

            chunk_size = min(len(data) - offset, self._send_window, MAX_STREAM_WINDOW)
            chunk = data[offset : offset + chunk_size]
            await self.session._send_frame(TYPE_DATA, 0, self.stream_id, chunk)
            self._send_window -= chunk_size
            offset += chunk_size

    async def _send_window_update(self, delta: int) -> None:
        """Send a window update to the peer and replenish our local receive window."""
        self._recv_window += delta
        await self.session._send_frame(TYPE_WINDOW_UPDATE, 0, self.stream_id, b"", length_override=delta)

    def _recv_data(self, data: bytes) -> None:
        """Called by the session when data arrives for this stream.

        Enforces the receive window: if the remote sends more data than
        the window allows, the stream is reset.
        """
        self._recv_window -= len(data)
        if self._recv_window < 0:
            log.warning(
                f"yamux stream {self.stream_id}: receive window violated "
                f"(window={self._recv_window + len(data)}, received={len(data)})"
            )
            self._closed = True
            self._recv_buffer.put_nowait(None)
            # Send RST to inform the remote side
            asyncio.get_running_loop().create_task(
                self.session._send_frame(TYPE_DATA, FLAG_RST, self.stream_id, b"")
            )
            return
        self._recv_buffer.put_nowait(data)

    def _recv_window_update(self, delta: int) -> None:
        """Called by the session when a window update arrives.

        Uses a fire-and-forget task to acquire the Condition lock and notify
        waiters, since this is called from the synchronous read loop.
        """
        self._send_window += delta
        # Schedule the notification on the event loop (Condition.notify requires
        # the lock, which is async)
        asyncio.get_running_loop().create_task(self._notify_send_window())

    async def _notify_send_window(self) -> None:
        """Notify waiters that the send window has been updated."""
        async with self._send_window_cond:
            self._send_window_cond.notify_all()

    def _recv_fin(self) -> None:
        """Called when the remote side half-closes."""
        self._remote_closed = True
        self._recv_buffer.put_nowait(None)

    def _recv_rst(self) -> None:
        """Called when the remote side resets."""
        self._closed = True
        self._recv_buffer.put_nowait(None)
        # Wake any writers blocked on send window so they can see _closed
        asyncio.get_running_loop().create_task(self._notify_send_window())

    async def close(self) -> None:
        """Half-close this stream (send FIN)."""
        if not self._closed:
            await self.session._send_frame(TYPE_DATA, FLAG_FIN, self.stream_id, b"")
            self._closed = True


class YamuxSession:
    """Yamux multiplexer session.

    Wraps a NoiseTransport (or any read_msg/write_msg interface) and provides
    multiplexed streams on top.
    """

    def __init__(self, transport, is_initiator: bool):
        """Create a new Yamux session.

        Args:
            transport: must have read_msg() -> bytes and write_msg(bytes) methods
            is_initiator: True if this side initiated the connection (uses odd stream IDs)
        """
        self.transport = transport
        self.is_initiator = is_initiator
        self._streams: dict[int, YamuxStream] = {}
        self._next_stream_id = 1 if is_initiator else 2  # odd=initiator, even=responder
        self._incoming_streams: asyncio.Queue[YamuxStream] = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._running = False
        self._run_task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the session's background frame reader."""
        self._running = True
        self._run_task = asyncio.create_task(self._read_loop())

    @property
    def is_alive(self) -> bool:
        """Check if the session's read loop is still running."""
        return self._run_task is not None and not self._run_task.done()

    async def stop(self) -> None:
        """Stop the session gracefully.

        Sends a GO_AWAY frame to inform the remote side, then cancels
        the read loop. If the GO_AWAY fails (transport already closed),
        we proceed with shutdown anyway.
        """
        self._running = False
        try:
            await self._send_frame(TYPE_GO_AWAY, 0, 0, b"", length_override=0)
        except Exception:
            pass
        if self._run_task:
            self._run_task.cancel()
            try:
                await self._run_task
            except (asyncio.CancelledError, Exception):
                pass

    async def open_stream(self) -> YamuxStream:
        """Open a new outbound stream."""
        stream_id = self._next_stream_id
        self._next_stream_id += 2

        stream = YamuxStream(self, stream_id)
        self._streams[stream_id] = stream

        # Send SYN with a window update to advertise our receive window
        await self._send_frame(TYPE_WINDOW_UPDATE, FLAG_SYN, stream_id, b"", length_override=DEFAULT_WINDOW_SIZE)
        return stream

    async def accept_stream(self) -> YamuxStream:
        """Wait for and accept an incoming stream.

        Raises ConnectionError if the session received GO_AWAY.
        """
        stream = await self._incoming_streams.get()
        if stream is None:
            raise ConnectionError("yamux session received GO_AWAY")
        return stream

    async def _send_frame(
        self, type_: int, flags: int, stream_id: int, data: bytes, length_override: int | None = None
    ) -> None:
        """Send a Yamux frame over the transport."""
        length = length_override if length_override is not None else len(data)
        header = _encode_header(type_, flags, stream_id, length)
        async with self._write_lock:
            await self.transport.write_msg(header + data)

    async def _read_loop(self) -> None:
        """Background task that reads and dispatches Yamux frames."""
        try:
            while self._running:
                frame = await self.transport.read_msg()
                if len(frame) < HEADER_SIZE:
                    log.warning(f"yamux: short frame ({len(frame)} bytes)")
                    continue

                version, type_, flags, stream_id, length = _decode_header(frame[:HEADER_SIZE])
                payload = frame[HEADER_SIZE:]

                if version != YAMUX_VERSION:
                    log.warning(f"yamux: unexpected version {version}")
                    continue

                if type_ == TYPE_PING:
                    await self._handle_ping(flags, length)
                elif type_ == TYPE_GO_AWAY:
                    log.info("yamux: received GO_AWAY")
                    self._running = False
                    # Unblock any coroutine waiting in accept_stream()
                    self._incoming_streams.put_nowait(None)
                    break
                else:
                    await self._handle_stream_frame(type_, flags, stream_id, length, payload)

        except asyncio.CancelledError:
            pass
        except ConnectionError:
            log.debug("yamux: connection closed")
        except asyncio.IncompleteReadError:
            log.debug("yamux: connection EOF")
        except Exception as e:
            log.warning(f"yamux: read loop error: {e}")

    async def _handle_ping(self, flags: int, opaque: int) -> None:
        """Handle a ping frame. If SYN, send ACK response."""
        if flags & FLAG_SYN:
            await self._send_frame(TYPE_PING, FLAG_ACK, 0, b"", length_override=opaque)

    async def _handle_stream_frame(
        self, type_: int, flags: int, stream_id: int, length: int, payload: bytes
    ) -> None:
        """Handle a data or window-update frame for a stream."""
        # SYN flag: new incoming stream
        if flags & FLAG_SYN:
            if stream_id not in self._streams:
                stream = YamuxStream(self, stream_id)
                self._streams[stream_id] = stream
                # Send ACK with window size
                await self._send_frame(
                    TYPE_WINDOW_UPDATE, FLAG_ACK, stream_id, b"", length_override=DEFAULT_WINDOW_SIZE
                )
                self._incoming_streams.put_nowait(stream)

        stream = self._streams.get(stream_id)
        if stream is None:
            log.debug(f"yamux: frame for unknown stream {stream_id}")
            return

        if type_ == TYPE_DATA and payload:
            stream._recv_data(payload)
        elif type_ == TYPE_WINDOW_UPDATE:
            stream._recv_window_update(length)

        # Handle FIN/RST
        if flags & FLAG_FIN:
            stream._recv_fin()
        if flags & FLAG_RST:
            stream._recv_rst()
            self._streams.pop(stream_id, None)
