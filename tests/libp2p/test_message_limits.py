# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for message size limits and record store enforcement."""

import asyncio
import pytest

from mx_libp2p.kademlia import _read_length_prefixed, MAX_KAD_MESSAGE_SIZE
from mx_libp2p.crypto import _encode_uvarint
from mx_libp2p.kad_handler import KadHandler, MAX_RECORD_VALUE_SIZE
from mx_libp2p.routing import RoutingTable
from mx_libp2p.kademlia import (
    encode_kad_message,
    decode_kad_message,
    encode_record,
    MSG_PUT_VALUE,
)
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
    # Encode a varint claiming 2 MB
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

    # Build a PUT_VALUE message with oversized value
    big_value = b"x" * 101
    record = encode_record(b"key", big_value)
    msg_bytes = encode_kad_message(MSG_PUT_VALUE, key=b"key", record=record)
    msg = decode_kad_message(msg_bytes)

    response_bytes = handler._handle_put_value(msg)
    response = decode_kad_message(response_bytes)

    # Record should not be stored
    assert handler.get_local(b"key") is None
    # Response should not contain a record (rejection signal)
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

    # Record should be stored
    local = handler.get_local(b"key")
    assert local is not None
    assert local.value == value
    # Response should NOT echo the record (matches rust-libp2p behavior)
    assert response.get("record") is None


def test_kad_handler_evicts_when_store_full():
    rt = RoutingTable(b"\x00" * 32)
    handler = KadHandler(rt, max_records=2)

    # Fill the store
    handler.put_local(b"k1", b"v1")
    handler.put_local(b"k2", b"v2")

    # Store via inbound PUT_VALUE - should evict the furthest record
    value = b"v3"
    record = encode_record(b"k3", value)
    msg_bytes = encode_kad_message(MSG_PUT_VALUE, key=b"k3", record=record)
    msg = decode_kad_message(msg_bytes)

    response_bytes = handler._handle_put_value(msg)
    response = decode_kad_message(response_bytes)

    # New key should be stored (evicted furthest)
    assert handler.get_local(b"k3") is not None
    assert len(handler.records) == 2  # still at max
    # Response confirms storage (no record echoed, matching rust-libp2p)
    assert response.get("record") is None
