# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Connection establishment: TCP + Noise + Yamux.

The libp2p connection sequence for a dialer:
1. TCP connect
2. Multistream-select -> /noise
3. Noise XX handshake (mutual authentication)
4. Multistream-select -> /yamux/1.0.0
5. Yamux session established
6. Open streams for sub-protocols (Identify, Kademlia, etc.)

For a listener, the same sequence but in the responding role.
"""

import asyncio
import logging
from typing import Callable

from .crypto import Ed25519Identity
from .multistream import negotiate_outbound, negotiate_inbound
from .noise import NoiseTransport, handshake_initiator, handshake_responder
from .yamux import YamuxSession, YamuxStream

log = logging.getLogger(__name__)

NOISE_PROTOCOL = "/noise"
YAMUX_PROTOCOL = "/yamux/1.0.0"
IDENTIFY_PROTOCOL = "/ipfs/id/1.0.0"
IDENTIFY_PUSH_PROTOCOL = "/ipfs/id/push/1.0.0"


class Connection:
    """A fully established libp2p connection (TCP + Noise + Yamux)."""

    def __init__(
        self,
        identity: Ed25519Identity,
        noise: NoiseTransport,
        yamux: YamuxSession,
        remote_peer_id: bytes,
        local_addr: tuple[str, int] | None = None,
        remote_addr: tuple[str, int] | None = None,
    ):
        self.identity = identity
        self.noise = noise
        self.yamux = yamux
        self.remote_peer_id = remote_peer_id
        self.local_addr = local_addr
        self.remote_addr = remote_addr
        self._protocol_handlers: dict[str, asyncio.Queue] = {}
        self._inbound_task: asyncio.Task | None = None
        self._closing = False  # True when we initiate close, to suppress the callback
        self._on_closed: Callable[[bytes], None] | None = None

    def register_protocol(self, protocol_id: str) -> asyncio.Queue:
        """Register a protocol for inbound streams. Returns a queue that will
        receive (stream, protocol_id) tuples when the remote opens a stream
        for this protocol.

        If already registered, returns the existing queue to avoid orphaning
        streams that arrived between connection setup and this call.
        """
        existing = self._protocol_handlers.get(protocol_id)
        if existing is not None:
            return existing
        q: asyncio.Queue = asyncio.Queue()
        self._protocol_handlers[protocol_id] = q
        return q

    async def start_inbound_handler(self) -> None:
        """Start handling inbound streams (protocol negotiation + dispatch)."""
        self._inbound_task = asyncio.create_task(self._handle_inbound_streams())

    async def _handle_inbound_streams(self) -> None:
        """Accept inbound Yamux streams, negotiate protocol, dispatch."""
        try:
            while True:
                stream = await self.yamux.accept_stream()
                asyncio.create_task(self._negotiate_inbound_stream(stream))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"inbound handler error: {e}")

    async def _negotiate_inbound_stream(self, stream: YamuxStream) -> None:
        """Negotiate protocol for an inbound stream and dispatch it."""
        try:
            reader, writer = _stream_to_rw(stream)
            supported = list(self._protocol_handlers.keys())
            protocol = await negotiate_inbound(reader, writer, supported)
            log.debug(f"inbound stream {stream.stream_id}: negotiated {protocol}")
            q = self._protocol_handlers.get(protocol)
            if q:
                q.put_nowait((stream, reader, writer))
        except Exception as e:
            log.warning(f"inbound stream negotiation failed: {e}")

    async def open_stream(self, protocol_id: str) -> tuple[YamuxStream, asyncio.StreamReader, asyncio.StreamWriter]:
        """Open a new outbound stream and negotiate the given protocol."""
        stream = await self.yamux.open_stream()
        reader, writer = _stream_to_rw(stream)
        await negotiate_outbound(reader, writer, protocol_id)
        return stream, reader, writer

    def set_on_closed(self, callback: Callable[[bytes], None]) -> None:
        """Register a callback fired when the remote end closes the connection.

        The callback receives the remote peer ID. It is NOT called when we
        initiate the close ourselves (via close()), only when the session
        ends due to remote GO_AWAY, TCP EOF, or transport error.
        """
        self._on_closed = callback
        if self.yamux._run_task is not None:
            self.yamux._run_task.add_done_callback(self._on_session_done)

    def _on_session_done(self, task: asyncio.Task) -> None:
        """Done callback on the Yamux read loop task."""
        if self._closing:
            return
        if self._on_closed is not None:
            try:
                self._on_closed(self.remote_peer_id)
            except Exception as e:
                log.warning(f"on_closed callback error: {e}")

    @property
    def is_alive(self) -> bool:
        """Check if the connection is still usable."""
        return self.yamux.is_alive

    async def close(self) -> None:
        self._closing = True
        if self._inbound_task:
            self._inbound_task.cancel()
        # Close the underlying transport first so the yamux read loop unblocks
        self.noise.close()
        await self.yamux.stop()


async def accept(
    identity: Ed25519Identity,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    supported_protocols: list[str] | None = None,
) -> Connection:
    """Accept an inbound libp2p connection (responder side).

    Performs the same handshake sequence as dial() but in the responding role:
    1. Multistream-select inbound -> /noise
    2. Noise XX handshake (responder)
    3. Multistream-select inbound -> /yamux/1.0.0 (over Noise)
    4. Yamux session (responder side, even stream IDs)
    """
    remote_addr = writer.get_extra_info("peername")
    local_addr = writer.get_extra_info("sockname")
    log.info(f"accepting connection from {remote_addr}")

    # 1. Multistream-select -> /noise (responder)
    await negotiate_inbound(reader, writer, [NOISE_PROTOCOL])
    log.debug("negotiated /noise (responder)")

    # 2. Noise XX handshake (responder)
    noise = await handshake_responder(reader, writer, identity)
    log.info(f"noise handshake complete (responder), remote peer: {noise.remote_peer_id.hex()[:16]}...")

    # 3. Multistream-select -> /yamux/1.0.0 (over Noise transport, responder)
    noise_reader, noise_writer = _noise_to_rw(noise)
    await negotiate_inbound(noise_reader, noise_writer, [YAMUX_PROTOCOL])
    log.debug("negotiated /yamux/1.0.0 (responder)")

    # 4. Start Yamux session (responder uses even stream IDs)
    yamux = YamuxSession(noise, is_initiator=False)
    await yamux.start()

    conn = Connection(
        identity=identity,
        noise=noise,
        yamux=yamux,
        remote_peer_id=noise.remote_peer_id,
        local_addr=local_addr,
        remote_addr=remote_addr,
    )

    # Register default protocols
    if supported_protocols is None:
        supported_protocols = [IDENTIFY_PROTOCOL, IDENTIFY_PUSH_PROTOCOL]
    for proto in supported_protocols:
        conn.register_protocol(proto)

    await conn.start_inbound_handler()
    return conn


def _stream_to_rw(stream: YamuxStream) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Wrap a YamuxStream as asyncio StreamReader/StreamWriter-like objects.

    We use a lightweight adapter since multistream-select expects the
    standard asyncio stream interface.
    """
    return YamuxStreamReader(stream), YamuxStreamWriter(stream)


class YamuxStreamReader:
    """Adapts a YamuxStream to the asyncio.StreamReader interface (subset)."""

    def __init__(self, stream: YamuxStream):
        self._stream = stream
        self._buffer = bytearray()

    async def readexactly(self, n: int) -> bytes:
        while len(self._buffer) < n:
            chunk = await self._stream.read()
            if not chunk:
                raise asyncio.IncompleteReadError(bytes(self._buffer), n)
            self._buffer.extend(chunk)
        result = bytes(self._buffer[:n])
        self._buffer = self._buffer[n:]
        return result

    async def read(self, n: int = -1) -> bytes:
        if self._buffer:
            if n < 0:
                result = bytes(self._buffer)
                self._buffer.clear()
                return result
            result = bytes(self._buffer[:n])
            self._buffer = self._buffer[n:]
            return result
        return await self._stream.read()


class YamuxStreamWriter:
    """Adapts a YamuxStream to the asyncio.StreamWriter interface (subset)."""

    def __init__(self, stream: YamuxStream):
        self._stream = stream
        self._pending = bytearray()

    def write(self, data: bytes) -> None:
        self._pending.extend(data)

    async def drain(self) -> None:
        if self._pending:
            await self._stream.write(bytes(self._pending))
            self._pending.clear()


async def dial(
    identity: Ed25519Identity,
    host: str,
    port: int,
    supported_protocols: list[str] | None = None,
) -> Connection:
    """Establish a libp2p connection to a remote peer.

    Args:
        identity: our Ed25519 identity
        host: remote IP address
        port: remote TCP port
        supported_protocols: protocols to register for inbound streams

    Returns:
        A fully established Connection with Yamux session running.
    """
    log.info(f"dialing {host}:{port}")

    # 1. TCP connect
    reader, writer = await asyncio.open_connection(host, port)
    local_addr = writer.get_extra_info("sockname")
    remote_addr = (host, port)

    # 2. Multistream-select -> /noise
    await negotiate_outbound(reader, writer, NOISE_PROTOCOL)
    log.debug("negotiated /noise")

    # 3. Noise XX handshake
    noise = await handshake_initiator(reader, writer, identity)
    log.info(f"noise handshake complete, remote peer: {noise.remote_peer_id.hex()[:16]}...")

    # 4. Multistream-select -> /yamux/1.0.0 (over Noise transport)
    # After Noise, all communication is encrypted. We need to negotiate
    # Yamux over the Noise transport using its read_msg/write_msg.
    noise_reader, noise_writer = _noise_to_rw(noise)
    await negotiate_outbound(noise_reader, noise_writer, YAMUX_PROTOCOL)
    log.debug("negotiated /yamux/1.0.0")

    # 5. Start Yamux session
    yamux = YamuxSession(noise, is_initiator=True)
    await yamux.start()

    conn = Connection(
        identity=identity,
        noise=noise,
        yamux=yamux,
        remote_peer_id=noise.remote_peer_id,
        local_addr=local_addr,
        remote_addr=remote_addr,
    )

    # Register default protocols
    if supported_protocols is None:
        supported_protocols = [IDENTIFY_PROTOCOL, IDENTIFY_PUSH_PROTOCOL]
    for proto in supported_protocols:
        conn.register_protocol(proto)

    await conn.start_inbound_handler()

    return conn


class _NoiseStreamReader:
    """Adapts NoiseTransport to StreamReader interface for multistream negotiation."""

    def __init__(self, noise: NoiseTransport):
        self._noise = noise
        self._buffer = bytearray()

    async def readexactly(self, n: int) -> bytes:
        while len(self._buffer) < n:
            chunk = await self._noise.read_msg()
            if not chunk:
                raise asyncio.IncompleteReadError(bytes(self._buffer), n)
            self._buffer.extend(chunk)
        result = bytes(self._buffer[:n])
        self._buffer = self._buffer[n:]
        return result


class _NoiseStreamWriter:
    """Adapts NoiseTransport to StreamWriter interface for multistream negotiation."""

    def __init__(self, noise: NoiseTransport):
        self._noise = noise
        self._pending = bytearray()

    def write(self, data: bytes) -> None:
        self._pending.extend(data)

    async def drain(self) -> None:
        if self._pending:
            await self._noise.write_msg(bytes(self._pending))
            self._pending.clear()


def _noise_to_rw(noise: NoiseTransport) -> tuple[_NoiseStreamReader, _NoiseStreamWriter]:
    return _NoiseStreamReader(noise), _NoiseStreamWriter(noise)
