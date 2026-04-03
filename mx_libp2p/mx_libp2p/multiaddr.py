# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multiaddr encoding/decoding.

Reference: https://github.com/multiformats/multiaddr

A multiaddr is a self-describing network address format.
Each component is: <protocol-code-varint><address-bytes>

Common protocols:
    /ip4/1.2.3.4     -> 0x04 + 4 bytes
    /tcp/1234         -> 0x06 + 2 bytes (big-endian)
    /p2p/<peer-id>    -> 0xa503 (varint) + varint-length + peer-id-bytes
"""

import socket
import struct

from .crypto import _encode_uvarint, _base58btc_decode, _base58btc_encode

# Protocol codes
PROTO_IP4 = 0x04
PROTO_TCP = 0x06
PROTO_UDP = 0x0111
PROTO_IP6 = 0x29
PROTO_DNS = 0x35
PROTO_DNS4 = 0x36
PROTO_DNS6 = 0x37
PROTO_P2P = 0x01A5

# Protocols with fixed-size address data (not length-prefixed)
_FIXED_SIZE_PROTOCOLS = {
    PROTO_IP4: 4,
    PROTO_TCP: 2,
    PROTO_UDP: 2,
    PROTO_IP6: 16,
}

# Protocols with length-prefixed address data
_LENGTH_PREFIXED_PROTOCOLS = {PROTO_P2P, PROTO_DNS, PROTO_DNS4, PROTO_DNS6}


def _decode_uvarint(data: bytes, offset: int = 0) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = data[offset]
        result |= (b & 0x7F) << shift
        offset += 1
        if b & 0x80 == 0:
            return result, offset
        shift += 7


def encode_multiaddr_ip4_tcp(ip: str, port: int) -> bytes:
    """Encode /ip4/<ip>/tcp/<port> as binary multiaddr."""
    result = bytearray()
    # /ip4/
    result.extend(_encode_uvarint(PROTO_IP4))
    result.extend(socket.inet_aton(ip))
    # /tcp/
    result.extend(_encode_uvarint(PROTO_TCP))
    result.extend(struct.pack(">H", port))
    return bytes(result)


def encode_multiaddr_ip4_tcp_p2p(ip: str, port: int, peer_id: bytes) -> bytes:
    """Encode /ip4/<ip>/tcp/<port>/p2p/<peer-id> as binary multiaddr."""
    result = bytearray()
    result.extend(encode_multiaddr_ip4_tcp(ip, port))
    # /p2p/
    result.extend(_encode_uvarint(PROTO_P2P))
    result.extend(_encode_uvarint(len(peer_id)))
    result.extend(peer_id)
    return bytes(result)


def encode_multiaddr_ip6_tcp(ip: str, port: int) -> bytes:
    """Encode /ip6/<ip>/tcp/<port> as binary multiaddr."""
    result = bytearray()
    result.extend(_encode_uvarint(PROTO_IP6))
    result.extend(socket.inet_pton(socket.AF_INET6, ip))
    result.extend(_encode_uvarint(PROTO_TCP))
    result.extend(struct.pack(">H", port))
    return bytes(result)


def encode_multiaddr_ip_tcp_p2p(ip: str, port: int, peer_id: bytes) -> bytes:
    """Encode /ip4/ or /ip6/ + /tcp/<port>/p2p/<peer-id>, auto-detecting IP version."""
    result = bytearray()
    if ":" in ip:
        result.extend(encode_multiaddr_ip6_tcp(ip, port))
    else:
        result.extend(encode_multiaddr_ip4_tcp(ip, port))
    result.extend(_encode_uvarint(PROTO_P2P))
    result.extend(_encode_uvarint(len(peer_id)))
    result.extend(peer_id)
    return bytes(result)


def decode_multiaddr(data: bytes) -> list[tuple[int, bytes]]:
    """Decode a binary multiaddr into a list of (protocol_code, address_bytes) tuples."""
    components = []
    offset = 0
    while offset < len(data):
        code, offset = _decode_uvarint(data, offset)
        if code in _FIXED_SIZE_PROTOCOLS:
            size = _FIXED_SIZE_PROTOCOLS[code]
            addr = data[offset : offset + size]
            offset += size
            components.append((code, addr))
        elif code in _LENGTH_PREFIXED_PROTOCOLS:
            length, offset = _decode_uvarint(data, offset)
            addr = data[offset : offset + length]
            offset += length
            components.append((code, addr))
        else:
            raise ValueError(
                f"unsupported multiaddr protocol code: 0x{code:x}. "
                f"Cannot determine address size for unknown protocols."
            )
    return components


def multiaddr_to_string(data: bytes) -> str:
    """Convert binary multiaddr to human-readable string."""
    parts = []
    for code, addr in decode_multiaddr(data):
        if code == PROTO_IP4:
            parts.append(f"/ip4/{socket.inet_ntoa(addr)}")
        elif code == PROTO_IP6:
            parts.append(f"/ip6/{socket.inet_ntop(socket.AF_INET6, addr)}")
        elif code == PROTO_TCP:
            parts.append(f"/tcp/{struct.unpack('>H', addr)[0]}")
        elif code == PROTO_UDP:
            parts.append(f"/udp/{struct.unpack('>H', addr)[0]}")
        elif code == PROTO_P2P:
            parts.append(f"/p2p/{_base58btc_encode(addr)}")
        elif code == PROTO_DNS:
            parts.append(f"/dns/{addr.decode('utf-8')}")
        elif code == PROTO_DNS4:
            parts.append(f"/dns4/{addr.decode('utf-8')}")
        elif code == PROTO_DNS6:
            parts.append(f"/dns6/{addr.decode('utf-8')}")
        else:
            parts.append(f"/{code}/{addr.hex()}")
    return "".join(parts)


def parse_multiaddr_string(s: str) -> bytes:
    """Parse a human-readable multiaddr string to binary.

    Supports: /ip4/, /ip6/, /dns/, /dns4/, /dns6/, /tcp/, /udp/, /p2p/
    """
    parts = s.strip("/").split("/")
    result = bytearray()
    i = 0
    while i < len(parts):
        proto = parts[i]
        i += 1
        if proto == "ip4":
            result.extend(_encode_uvarint(PROTO_IP4))
            result.extend(socket.inet_aton(parts[i]))
            i += 1
        elif proto == "ip6":
            result.extend(_encode_uvarint(PROTO_IP6))
            result.extend(socket.inet_pton(socket.AF_INET6, parts[i]))
            i += 1
        elif proto in ("dns", "dns4", "dns6"):
            code = {"dns": PROTO_DNS, "dns4": PROTO_DNS4, "dns6": PROTO_DNS6}[proto]
            result.extend(_encode_uvarint(code))
            hostname = parts[i].encode("utf-8")
            result.extend(_encode_uvarint(len(hostname)))
            result.extend(hostname)
            i += 1
        elif proto == "tcp":
            result.extend(_encode_uvarint(PROTO_TCP))
            result.extend(struct.pack(">H", int(parts[i])))
            i += 1
        elif proto == "udp":
            result.extend(_encode_uvarint(PROTO_UDP))
            result.extend(struct.pack(">H", int(parts[i])))
            i += 1
        elif proto == "p2p":
            result.extend(_encode_uvarint(PROTO_P2P))
            peer_id = _base58btc_decode(parts[i])
            result.extend(_encode_uvarint(len(peer_id)))
            result.extend(peer_id)
            i += 1
        else:
            raise ValueError(f"unsupported multiaddr protocol: {proto}")
    return bytes(result)
