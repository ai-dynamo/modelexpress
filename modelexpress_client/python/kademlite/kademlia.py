# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kademlia DHT protocol (/ipfs/kad/1.0.0).

Reference: https://github.com/libp2p/specs/blob/master/kad-dht/README.md

Wire protocol uses protobuf messages sent as length-prefixed frames
over a Yamux stream: <uvarint-length><protobuf-message>
"""

import asyncio
import logging

from .crypto import _encode_uvarint
from .proto.dht_pb2 import Message as MessageProto
from .proto.dht_pb2 import Record as RecordProto

log = logging.getLogger(__name__)

# Kademlia message types (mirrors MessageProto.MessageType enum)
MSG_PUT_VALUE = 0
MSG_GET_VALUE = 1
MSG_ADD_PROVIDER = 2
MSG_GET_PROVIDERS = 3
MSG_FIND_NODE = 4
MSG_PING = 5

KADEMLIA_PROTOCOL = "/ipfs/kad/1.0.0"

# Maximum Kademlia message size. rust-libp2p defaults to 16 KB.
# We match this to avoid sending messages that Rust peers will reject.
MAX_KAD_MESSAGE_SIZE = 16 * 1024


def encode_record(
    key: bytes,
    value: bytes,
    publisher: bytes | None = None,
    ttl: int | None = None,
) -> bytes:
    """Encode a Kademlia Record protobuf.

    Args:
        publisher: peer ID of the record's original publisher (rust-libp2p field 666)
        ttl: record TTL in seconds (rust-libp2p field 777)
    """
    rec = RecordProto()
    rec.key = key
    rec.value = value
    if publisher:
        rec.publisher = publisher
    if ttl is not None:
        rec.ttl = ttl
    return rec.SerializeToString()


def decode_record(data: bytes) -> dict:
    """Decode a Kademlia Record protobuf."""
    rec = RecordProto()
    rec.ParseFromString(data)
    return {
        "key": rec.key or None,
        "value": rec.value or None,
        "time_received": rec.timeReceived or None,
        "publisher": rec.publisher or None,
        "ttl": rec.ttl if rec.ttl else None,
    }


def encode_peer(peer_id: bytes, addrs: list[bytes]) -> bytes:
    """Encode a Kademlia Peer protobuf (embedded in Message)."""
    peer = MessageProto.Peer()
    peer.id = peer_id
    for addr in addrs:
        peer.addrs.append(addr)
    return peer.SerializeToString()


def decode_peer(data: bytes) -> dict:
    """Decode a Kademlia Peer protobuf."""
    peer = MessageProto.Peer()
    peer.ParseFromString(data)
    return {
        "id": peer.id or None,
        "addrs": list(peer.addrs),
        "connection": peer.connection,
    }


def encode_kad_message(
    type_: int,
    key: bytes | None = None,
    record: bytes | None = None,
    closer_peers: list[bytes] | None = None,
) -> bytes:
    """Encode a Kademlia Message protobuf.

    Args:
        type_: message type (MSG_PUT_VALUE, MSG_GET_VALUE, etc.)
        key: the lookup key
        record: pre-serialized Record protobuf bytes (for PUT/GET_VALUE)
        closer_peers: list of pre-serialized Peer protobuf bytes
    """
    msg = MessageProto()
    msg.type = type_
    if key is not None:
        msg.key = key
    if record is not None:
        msg.record.ParseFromString(record)
    if closer_peers:
        for peer_bytes in closer_peers:
            peer = msg.closerPeers.add()
            peer.ParseFromString(peer_bytes)
    msg.clusterLevelRaw = 0
    return msg.SerializeToString()


def decode_kad_message(data: bytes) -> dict:
    """Decode a Kademlia Message protobuf."""
    msg = MessageProto()
    msg.ParseFromString(data)

    record = None
    if msg.HasField("record"):
        record = {
            "key": msg.record.key or None,
            "value": msg.record.value or None,
            "publisher": msg.record.publisher or None,
            "ttl": msg.record.ttl if msg.record.ttl else None,
        }

    closer_peers = []
    for peer in msg.closerPeers:
        closer_peers.append({
            "id": peer.id or None,
            "addrs": list(peer.addrs),
            "connection": peer.connection,
        })

    return {
        "type": msg.type,
        "key": msg.key or None,
        "record": record,
        "closer_peers": closer_peers,
        "cluster_level": msg.clusterLevelRaw,
    }


async def _read_length_prefixed(reader) -> bytes:
    """Read a uvarint-length-prefixed message from a stream reader."""
    result = 0
    shift = 0
    while True:
        byte_data = await reader.readexactly(1)
        b = byte_data[0]
        result |= (b & 0x7F) << shift
        if b & 0x80 == 0:
            break
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")
    if result == 0:
        return b""
    if result > MAX_KAD_MESSAGE_SIZE:
        raise ValueError(f"Kademlia message too large: {result} bytes")
    return await reader.readexactly(result)


def _write_length_prefixed(writer, data: bytes) -> None:
    """Write a uvarint-length-prefixed message to a stream writer."""
    writer.write(_encode_uvarint(len(data)) + data)


async def _close_stream_quietly(stream) -> None:
    """Best-effort stream close that survives caller-task cancellation.

    Cancellation safety: spawns the close as a background task and shields
    it from the caller's cancellation. If the caller's task is cancelled
    mid-close, the close completes in the background and CancelledError
    re-raises to honor the cancellation contract. On Python 3.11+ a bare
    ``await stream.close()`` inside a finally block can be interrupted by
    a pending cancellation before FIN/RST is sent, leaving the yamux
    stream live; the shield + ensure_future pattern prevents that leak.

    Errors raised by ``stream.close()`` itself are logged at debug level.
    Used in cleanup paths of outbound Kad RPCs, identify handlers, and
    inbound stream negotiation - anywhere the close itself might race
    with cancellation.
    """
    close_task = asyncio.ensure_future(stream.close())

    def _consume_close_exc(t: asyncio.Task) -> None:
        # When the caller's task is cancelled while shield() is awaiting
        # close_task, shield re-raises CancelledError immediately and we
        # never await close_task ourselves. Without this callback, a
        # subsequent stream.close() exception would surface as an
        # asyncio "Task exception was never retrieved" warning. Consume
        # and log it instead.
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            log.debug(f"stream close raised in background: {exc}")

    close_task.add_done_callback(_consume_close_exc)

    try:
        await asyncio.shield(close_task)
    except asyncio.CancelledError:
        # Caller is cancelling us. close_task continues running outside
        # our task scope (the done-callback above will log any failure);
        # let the cancellation propagate.
        raise
    except Exception as e:
        log.debug(f"stream close raised during cleanup: {e}")


async def kad_get_value(conn, key: bytes) -> dict | None:
    """Send a GET_VALUE request over a new Kademlia stream."""
    stream, reader, writer = await conn.open_stream(KADEMLIA_PROTOCOL)
    try:
        request = encode_kad_message(MSG_GET_VALUE, key=key)
        _write_length_prefixed(writer, request)
        await writer.drain()

        response_data = await _read_length_prefixed(reader)
        return decode_kad_message(response_data)
    finally:
        await _close_stream_quietly(stream)


async def kad_put_value(
    conn, key: bytes, value: bytes,
    publisher: bytes | None = None, ttl: int | None = None,
) -> dict | None:
    """Send a PUT_VALUE request over a new Kademlia stream."""
    stream, reader, writer = await conn.open_stream(KADEMLIA_PROTOCOL)
    try:
        record = encode_record(key, value, publisher=publisher, ttl=ttl)
        request = encode_kad_message(MSG_PUT_VALUE, key=key, record=record)
        _write_length_prefixed(writer, request)
        await writer.drain()

        response_data = await _read_length_prefixed(reader)
        return decode_kad_message(response_data)
    finally:
        await _close_stream_quietly(stream)


async def kad_find_node(conn, key: bytes) -> dict | None:
    """Send a FIND_NODE request over a new Kademlia stream."""
    stream, reader, writer = await conn.open_stream(KADEMLIA_PROTOCOL)
    try:
        request = encode_kad_message(MSG_FIND_NODE, key=key)
        _write_length_prefixed(writer, request)
        await writer.drain()

        response_data = await _read_length_prefixed(reader)
        return decode_kad_message(response_data)
    finally:
        await _close_stream_quietly(stream)
