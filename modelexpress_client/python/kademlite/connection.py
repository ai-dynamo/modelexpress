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

from .crypto import Ed25519Identity
from .multistream import negotiate_inbound, negotiate_outbound
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
        # Track per-stream negotiation/handler tasks spawned by the
        # inbound dispatcher. Without tracking, Connection.close()
        # cancels only _inbound_task and leaves the per-stream tasks
        # running against a torn-down yamux session.
        self._stream_tasks: set[asyncio.Task] = set()

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
                t = asyncio.create_task(self._negotiate_inbound_stream(stream))
                self._stream_tasks.add(t)
                # Self-clean from the set on completion so finished
                # tasks aren't held forever.
                t.add_done_callback(self._stream_tasks.discard)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"inbound handler error: {e}")

    async def _negotiate_inbound_stream(self, stream: YamuxStream) -> None:
        """Negotiate protocol for an inbound stream and dispatch it.

        On a successful negotiation the stream is handed off to a protocol
        handler queue and that handler owns the close. Any failure path -
        negotiation error, cancellation, unknown protocol with no
        registered handler, full queue - closes the stream here so it
        doesn't outlive this method.
        """
        dispatched = False
        try:
            reader, writer = _stream_to_rw(stream)
            supported = list(self._protocol_handlers.keys())
            protocol = await negotiate_inbound(reader, writer, supported)
            log.debug(f"inbound stream {stream.stream_id}: negotiated {protocol}")
            q = self._protocol_handlers.get(protocol)
            if q is None:
                log.warning(
                    f"inbound stream {stream.stream_id}: no handler for "
                    f"negotiated protocol {protocol}"
                )
            else:
                q.put_nowait((stream, reader, writer))
                dispatched = True
        except Exception as e:
            log.warning(f"inbound stream negotiation failed: {e}")
        finally:
            if not dispatched:
                # Use the cancel-safe close helper so a cancellation in
                # the caller's task can't interrupt the close mid-await
                # and leave the stream live.
                from .kademlia import _close_stream_quietly
                await _close_stream_quietly(stream)

    async def open_stream(
        self, protocol_id: str
    ) -> tuple[YamuxStream, asyncio.StreamReader, asyncio.StreamWriter]:
        """Open a new outbound stream and negotiate the given protocol.

        If multistream-select negotiation fails or is cancelled after the
        YamuxStream has been opened, close the stream before propagating
        so a half-opened stream doesn't leak on the connection. The
        caller never sees the stream object on a failed negotiation, so
        only this method can clean it up.
        """
        stream = await self.yamux.open_stream()
        try:
            reader, writer = _stream_to_rw(stream)
            await negotiate_outbound(reader, writer, protocol_id)
        except BaseException:
            # Use the cancel-safe close helper: caller's cancellation
            # could otherwise interrupt the close mid-await and leak
            # the half-opened yamux stream.
            from .kademlia import _close_stream_quietly
            await _close_stream_quietly(stream)
            raise
        return stream, reader, writer

    @property
    def is_alive(self) -> bool:
        """Check if the connection is still usable."""
        return self.yamux.is_alive

    async def close(self) -> None:
        # Cancel the dispatcher first so it stops spawning new
        # _negotiate_inbound_stream tasks while we're tearing down.
        # Wait for the dispatcher to actually quiesce before snapshotting
        # _stream_tasks - otherwise a task spawned after our snapshot
        # could escape the cancel-and-gather sweep below.
        if self._inbound_task:
            self._inbound_task.cancel()
            try:
                await self._inbound_task
            except (asyncio.CancelledError, Exception):
                pass
        # Cancel any in-flight per-stream negotiation/handler tasks so
        # they don't outlive the yamux session. Each will hit
        # CancelledError on its current await; their finally clauses
        # close the stream via _close_stream_quietly.
        for t in list(self._stream_tasks):
            t.cancel()
        if self._stream_tasks:
            try:
                await asyncio.gather(*self._stream_tasks, return_exceptions=True)
            except Exception as e:
                log.debug(f"stream task gather raised during close: {e}")
        # Stop yamux BEFORE closing noise so the GO_AWAY frame can ship
        # through the still-alive noise transport - that gives the remote
        # peer a polite shutdown signal instead of a TCP half-close.
        # yamux.stop cancels its read_loop task which interrupts the
        # noise.read_msg await; the read loop terminates via
        # CancelledError without needing the transport to be closed.
        # (The reverse order also works mechanically because
        # yamux._send_frame's GO_AWAY is wrapped in try/except, but the
        # frame is silently dropped when noise is already closed.)
        try:
            await self.yamux.stop()
        except Exception as e:
            log.debug(f"yamux.stop during Connection.close raised: {e}")
        try:
            self.noise.close()
        except Exception as e:
            log.debug(f"noise.close during Connection.close raised: {e}")


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

    # Same cleanup-on-failure shape as dial(): track allocated resources
    # so a mid-handshake failure (Listener timeout cancellation, peer
    # closing connection during Noise, yamux.start exception) doesn't
    # leak TCP, Noise state, or yamux background tasks.
    noise = None
    yamux = None
    try:
        # 1. Multistream-select -> /noise (responder)
        await negotiate_inbound(reader, writer, [NOISE_PROTOCOL])
        log.debug("negotiated /noise (responder)")

        # 2. Noise XX handshake (responder)
        noise = await handshake_responder(reader, writer, identity)
        remote_short = noise.remote_peer_id.hex()[:16]
        log.info(f"noise handshake complete (responder), remote peer: {remote_short}...")

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
    except BaseException:
        # accept() doesn't own the TCP writer (the listener does), so
        # we don't close it here - the listener's _handle_connection
        # closes the writer in its finally block on every non-success
        # path (including external cancellation).
        await _cleanup_partial_handshake(None, noise, yamux)
        raise


async def _cleanup_partial_handshake(
    writer: asyncio.StreamWriter | None,
    noise: NoiseTransport | None,
    yamux: YamuxSession | None,
) -> None:
    """Best-effort teardown of resources allocated mid-handshake.

    Used by dial()'s except-BaseException path (and the equivalent in
    accept()) to release the TCP socket, Noise state, and yamux
    background task when a handshake step raises after earlier steps
    succeeded. Each cleanup is wrapped in its own try/except so a
    failure in one step doesn't mask the original exception or skip
    later cleanup steps.

    Writer ownership: dial() owns the TCP writer it allocated via
    open_connection, so it passes the writer here for cleanup.
    accept() does NOT own the writer (the listener does), so it
    passes writer=None - the listener's _handle_connection cleans up
    the TCP side itself.

    Order is reverse of allocation: yamux (which has a background
    task that needs cancellation) first, then noise (which holds
    encrypted state and an open transport), then writer (the raw TCP
    socket). Yamux.stop sends GO_AWAY which needs the noise transport
    alive, so noise must outlive yamux teardown.

    Cancellation safety: dial() / accept() catch BaseException to also
    invoke this helper on caller-task cancellation. If the cancellation
    lands during one of our awaits (yamux.stop, writer.wait_closed),
    we catch CancelledError, finish the remaining cleanup steps, and
    re-raise CancelledError at the end so the caller's cancellation
    contract still holds. Without this, a cancellation mid-yamux.stop
    would skip noise.close and writer.close - the exact leak the helper
    exists to prevent.
    """
    # Capture the first CancelledError instance (not just the fact)
    # so we preserve the original cancellation message/context that
    # Task.cancel(msg) attaches on Python 3.11+. Re-raising a fresh
    # CancelledError() would drop that metadata.
    cancelled_during_cleanup: asyncio.CancelledError | None = None

    if yamux is not None:
        try:
            await yamux.stop()
        except asyncio.CancelledError as exc:
            cancelled_during_cleanup = cancelled_during_cleanup or exc
            log.debug("yamux.stop cancelled during partial-handshake cleanup; continuing teardown")
        except Exception as e:
            log.debug(f"yamux.stop during partial-handshake cleanup raised: {e}")

    if noise is not None:
        # noise.close is synchronous; no cancellation surface here.
        try:
            noise.close()
        except Exception as e:
            log.debug(f"noise.close during partial-handshake cleanup raised: {e}")

    if writer is not None:
        try:
            writer.close()
            # Await wait_closed so the transport actually drains before
            # the function returns; otherwise the FD can stay open long
            # enough to trigger ResourceWarnings or flaky FD-exhaustion
            # in tests. Best-effort: failures here just mean the OS
            # cleans up later.
            await writer.wait_closed()
        except asyncio.CancelledError as exc:
            cancelled_during_cleanup = cancelled_during_cleanup or exc
            log.debug("writer.wait_closed cancelled during partial-handshake cleanup")
        except Exception as e:
            log.debug(f"writer.close during partial-handshake cleanup raised: {e}")

    if cancelled_during_cleanup is not None:
        # Surface the original cancellation (with its message/context)
        # now that all cleanup steps have run.
        raise cancelled_during_cleanup


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

    # Track resources allocated mid-handshake for cleanup-on-failure.
    # If anything raises after a step succeeds (multistream timeout,
    # yamux.start exception, asyncio.CancelledError from a higher-layer
    # wait_for, etc.), we tear down what was already constructed before
    # propagating - so callers don't leak TCP sockets, Noise state, or
    # background yamux tasks on failed dials.
    writer = None
    noise = None
    yamux = None
    try:
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
    except BaseException:
        # Tear down whatever was constructed so callers don't leak.
        # Each cleanup is best-effort: log and continue rather than
        # mask the original exception.
        await _cleanup_partial_handshake(writer, noise, yamux)
        raise


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
