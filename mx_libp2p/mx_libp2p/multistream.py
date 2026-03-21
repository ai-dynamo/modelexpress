# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multistream-select 1.0.0 protocol negotiation.

Reference: https://github.com/multiformats/multistream-select

Every libp2p connection and sub-stream begins with multistream-select negotiation.
Messages are length-prefixed with unsigned varint encoding, newline-terminated.

Wire format:
    <varint-length><protocol-id>\n

The initiator (dialer) sends the multistream header, then proposes a protocol.
The listener echoes the header, then echoes the protocol if it supports it,
or sends "na" to reject.
"""

import asyncio

from .crypto import _encode_uvarint

MULTISTREAM_PROTOCOL_ID = "/multistream/1.0.0"

# Maximum multistream-select message size (64 KB)
MAX_MULTISTREAM_MSG_SIZE = 64 * 1024


async def _read_uvarint(reader: asyncio.StreamReader) -> int:
    """Read an unsigned varint from the stream."""
    result = 0
    shift = 0
    while True:
        byte = await reader.readexactly(1)
        b = byte[0]
        result |= (b & 0x7F) << shift
        if b & 0x80 == 0:
            return result
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _encode_msg(protocol_id: str) -> bytes:
    """Encode a multistream-select message: varint-length + protocol + newline."""
    payload = protocol_id.encode("utf-8") + b"\n"
    return _encode_uvarint(len(payload)) + payload


async def _read_msg(reader: asyncio.StreamReader) -> str:
    """Read a multistream-select message and return the protocol string."""
    length = await _read_uvarint(reader)
    if length > MAX_MULTISTREAM_MSG_SIZE:
        raise ValueError(f"multistream message too large: {length} bytes")
    data = await reader.readexactly(length)
    # Strip trailing newline
    if data.endswith(b"\n"):
        data = data[:-1]
    return data.decode("utf-8")


async def negotiate_outbound(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    protocol_id: str,
) -> None:
    """Negotiate a protocol as the dialer (initiator).

    Sends the multistream header + desired protocol in a single write,
    then reads back the header echo and protocol confirmation.

    Raises ValueError if the remote rejects the protocol.
    """
    # Send header + protocol together (libp2p optimistic send)
    msg = _encode_msg(MULTISTREAM_PROTOCOL_ID) + _encode_msg(protocol_id)
    writer.write(msg)
    await writer.drain()

    # Read header echo
    header = await _read_msg(reader)
    if header != MULTISTREAM_PROTOCOL_ID:
        raise ValueError(f"unexpected multistream header: {header!r}")

    # Read protocol confirmation
    response = await _read_msg(reader)
    if response == "na":
        raise ValueError(f"remote rejected protocol: {protocol_id}")
    if response != protocol_id:
        raise ValueError(f"protocol mismatch: expected {protocol_id!r}, got {response!r}")


async def negotiate_inbound(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    supported_protocols: list[str],
) -> str:
    """Negotiate a protocol as the listener.

    Reads the multistream header and echoes it, then reads the proposed protocol.
    If we support it, echo it back. Otherwise send "na".

    Returns the negotiated protocol ID.
    Raises ValueError if no common protocol is found.
    """
    # Read and echo header
    header = await _read_msg(reader)
    if header != MULTISTREAM_PROTOCOL_ID:
        raise ValueError(f"unexpected multistream header: {header!r}")

    writer.write(_encode_msg(MULTISTREAM_PROTOCOL_ID))
    await writer.drain()

    # Read proposed protocol
    proposed = await _read_msg(reader)

    if proposed in supported_protocols:
        writer.write(_encode_msg(proposed))
        await writer.drain()
        return proposed

    # Reject
    writer.write(_encode_msg("na"))
    await writer.drain()
    raise ValueError(f"no common protocol: remote proposed {proposed!r}, we support {supported_protocols}")
