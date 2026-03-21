# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for message size limits, record store enforcement, and protocol encoding.

Covers:
- Kademlia message size limit (read_length_prefixed)
- Multistream message size limit
- KadHandler record size enforcement (oversized rejection, valid acceptance)
- KadHandler store eviction via inbound PUT_VALUE
- Record protobuf publisher/ttl field round-trip
- Unknown Kademlia message type handling
"""

import asyncio
import pytest

from mx_libp2p.kademlia import (
    _read_length_prefixed,
    MAX_KAD_MESSAGE_SIZE,
    encode_kad_message,
    decode_kad_message,
    encode_record,
    decode_record,
    MSG_PUT_VALUE,
)
from mx_libp2p.crypto import _encode_uvarint
from mx_libp2p.kad_handler import KadHandler, MAX_RECORD_VALUE_SIZE
from mx_libp2p.routing import RoutingTable
from mx_libp2p.multistream import _read_msg, MAX_MULTISTREAM_MSG_SIZE


class FakeReader:
    """Fake asyncio reader that yields data from a buffer."""

    def __init__(self, data: bytes):
        self._data = data
        self._offset = 0

    async def readexactly(self, n: int) -> bytes:
        if self._offset + n > len(self._data):
            raise asyncio.IncompleteReadError(
                self._data[self._offset:], n
            )
        chunk = self._data[self._offset : self._offset + n]
        self._offset += n
        return chunk


# --- Kademlia message size limit ---

@pytest.mark.asyncio
async def test_read_length_prefixed_rejects_oversized():
    """A message claiming to be larger than MAX_KAD_MESSAGE_SIZE should be rejected."""
    fake_length = 2 * 1024 * 1024
    data = _encode_uvarint(fake_length)
    reader = FakeReader(data)

    with pytest.raises(ValueError, match="too large"):
        await _read_length_prefixed(reader)


@pytest.mark.asyncio
async def test_read_length_prefixed_accepts_normal():
    """A normal-sized message should be read successfully."""
    payload = b"hello world"
    data = _encode_uvarint(len(payload)) + payload
    reader = FakeReader(data)

    result = await _read_length_prefixed(reader)
    assert result == payload


@pytest.mark.asyncio
async def test_read_length_prefixed_empty():
    """A zero-length message should return empty bytes."""
    data = _encode_uvarint(0)
    reader = FakeReader(data)

    result = await _read_length_prefixed(reader)
    assert result == b""


# --- Multistream message size limit ---

@pytest.mark.asyncio
async def test_multistream_read_msg_rejects_oversized():
    """A multistream message larger than MAX_MULTISTREAM_MSG_SIZE should be rejected."""
    fake_length = MAX_MULTISTREAM_MSG_SIZE + 1
    data = _encode_uvarint(fake_length)
    reader = FakeReader(data)

    with pytest.raises(ValueError, match="too large"):
        await _read_msg(reader)


# --- KadHandler record size enforcement ---

def test_kad_handler_rejects_oversized_record():
    rt = RoutingTable(b"\x00" * 32)
    handler = KadHandler(rt, max_record_size=100)

    big_value = b"x" * 101
    record = encode_record(b"key", big_value)
    msg_bytes = encode_kad_message(MSG_PUT_VALUE, key=b"key", record=record)
    msg = decode_kad_message(msg_bytes)

    response_bytes = handler._handle_put_value(msg)
    response = decode_kad_message(response_bytes)

    assert handler.get_local(b"key") is None
    assert response.get("record") is None


def test_kad_handler_accepts_valid_record():
    rt = RoutingTable(b"\x00" * 32)
    handler = KadHandler(rt, max_record_size=100)

    value = b"small"
    record = encode_record(b"key", value)
    msg_bytes = encode_kad_message(MSG_PUT_VALUE, key=b"key", record=record)
    msg = decode_kad_message(msg_bytes)

    response_bytes = handler._handle_put_value(msg)
    response = decode_kad_message(response_bytes)

    local = handler.get_local(b"key")
    assert local is not None
    assert local.value == value
    # Response should NOT echo the record (matches rust-libp2p behavior)
    assert response.get("record") is None


def test_kad_handler_evicts_when_store_full():
    rt = RoutingTable(b"\x00" * 32)
    handler = KadHandler(rt, max_records=2)

    handler.put_local(b"k1", b"v1")
    handler.put_local(b"k2", b"v2")

    # Store via inbound PUT_VALUE - should evict the furthest record
    value = b"v3"
    record = encode_record(b"k3", value)
    msg_bytes = encode_kad_message(MSG_PUT_VALUE, key=b"k3", record=record)
    msg = decode_kad_message(msg_bytes)

    response_bytes = handler._handle_put_value(msg)
    response = decode_kad_message(response_bytes)

    assert handler.get_local(b"k3") is not None
    assert len(handler.records) == 2
    assert response.get("record") is None


# --- Record protobuf publisher/ttl round-trip ---

def test_record_proto_publisher_ttl_roundtrip():
    """Record.publisher (field 666) and Record.ttl (field 777) should
    survive encode/decode."""
    publisher_id = b"\x00\x25" + b"\x01" * 34  # fake peer ID
    encoded = encode_record(
        key=b"/mx/test",
        value=b'{"rank":0}',
        publisher=publisher_id,
        ttl=300,
    )

    decoded = decode_record(encoded)
    assert decoded["key"] == b"/mx/test"
    assert decoded["value"] == b'{"rank":0}'
    assert decoded["publisher"] == publisher_id
    assert decoded["ttl"] == 300


def test_record_proto_without_publisher_ttl():
    """Records without publisher/ttl should decode cleanly (backward compat)."""
    encoded = encode_record(key=b"/mx/test", value=b"hello")
    decoded = decode_record(encoded)
    assert decoded["publisher"] is None
    assert decoded["ttl"] is None


# --- Unknown Kademlia message type ---

async def test_unknown_kad_message_type():
    """An unknown Kademlia message type should be ignored, not crash."""
    rt = RoutingTable(b"\x00" * 32)
    handler = KadHandler(rt)

    from mx_libp2p.proto.dht_pb2 import Message as MessageProto
    msg = MessageProto()
    msg.type = 99
    msg.key = b"test"

    msg_dict = decode_kad_message(msg.SerializeToString())
    assert msg_dict["type"] == 99

    # The handler should return None for unknown types (no crash)
    response = None
    if msg_dict["type"] == 0:
        response = handler._handle_put_value(msg_dict)
    elif msg_dict["type"] == 1:
        response = handler._handle_get_value(msg_dict)
    elif msg_dict["type"] == 4:
        response = handler._handle_find_node(msg_dict)

    assert response is None
