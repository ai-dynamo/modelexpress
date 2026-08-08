# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round-trip parse/encode tests for multiaddr strings.

Covers:
- ip4/tcp, ip4/tcp/p2p
- ip6/tcp (with and without /p2p suffix)
- dns / dns4 / dns6 + tcp + optional /p2p
- udp variants
- bare /p2p suffix on top of an IP/TCP base
- unsupported-protocol cases that must raise ValueError
"""

import pytest

from kademlite.crypto import Ed25519Identity, _base58btc_encode
from kademlite.multiaddr import (
    PROTO_DNS,
    PROTO_DNS4,
    PROTO_DNS6,
    PROTO_IP4,
    PROTO_IP6,
    PROTO_P2P,
    PROTO_TCP,
    PROTO_UDP,
    decode_multiaddr,
    multiaddr_to_string,
    parse_multiaddr_string,
)


def _peer_b58() -> str:
    """Generate a real base58btc peer ID string for use in /p2p/ components."""
    return _base58btc_encode(Ed25519Identity.generate().peer_id)


@pytest.fixture(scope="module")
def peer_b58() -> str:
    return _peer_b58()


# Strings without /p2p suffix - parameterized so parametrize doesn't need a fixture
ROUNDTRIP_STATIC = [
    # ip4 + tcp
    "/ip4/127.0.0.1/tcp/4001",
    "/ip4/0.0.0.0/tcp/0",
    "/ip4/192.168.1.42/tcp/65535",
    # ip6 + tcp (no /p2p)
    "/ip6/::1/tcp/4001",
    "/ip6/2001:db8::1/tcp/9000",
    "/ip6/fe80::1/tcp/12345",
    # dns variants without /p2p
    "/dns/example.com/tcp/443",
    "/dns4/host.example.org/tcp/4001",
    "/dns6/v6.example.com/tcp/4001",
    # udp
    "/ip4/10.0.0.1/udp/53",
    "/ip6/::1/udp/4242",
    # tcp without an address (just to confirm component is preserved by the
    # binary form even if practically unusual)
    "/tcp/1234",
]


@pytest.mark.parametrize("ma_str", ROUNDTRIP_STATIC)
def test_multiaddr_roundtrip_static(ma_str: str) -> None:
    """parse -> re-encode -> re-parse must be idempotent for static strings."""
    encoded = parse_multiaddr_string(ma_str)
    decoded = multiaddr_to_string(encoded)
    assert decoded == ma_str

    # Second round to confirm idempotence
    re_encoded = parse_multiaddr_string(decoded)
    assert re_encoded == encoded


@pytest.mark.parametrize("base", [
    "/ip4/127.0.0.1/tcp/4001",
    "/ip6/::1/tcp/4001",
    "/dns/example.com/tcp/443",
    "/dns4/host.example.org/tcp/4001",
    "/dns6/v6.example.com/tcp/4001",
])
def test_multiaddr_roundtrip_with_p2p(base: str, peer_b58: str) -> None:
    """Same matrix, with a /p2p/<peer-id> suffix appended."""
    ma_str = f"{base}/p2p/{peer_b58}"
    encoded = parse_multiaddr_string(ma_str)
    decoded = multiaddr_to_string(encoded)
    assert decoded == ma_str

    re_encoded = parse_multiaddr_string(decoded)
    assert re_encoded == encoded


def test_multiaddr_decode_components_ip4_tcp_p2p(peer_b58: str) -> None:
    """Sanity: the decoded component list has the expected protocol codes."""
    ma_str = f"/ip4/127.0.0.1/tcp/4001/p2p/{peer_b58}"
    encoded = parse_multiaddr_string(ma_str)
    components = decode_multiaddr(encoded)
    codes = [c for c, _ in components]
    assert codes == [PROTO_IP4, PROTO_TCP, PROTO_P2P]


def test_multiaddr_decode_components_ip6_tcp() -> None:
    encoded = parse_multiaddr_string("/ip6/::1/tcp/4001")
    components = decode_multiaddr(encoded)
    codes = [c for c, _ in components]
    assert codes == [PROTO_IP6, PROTO_TCP]


def test_multiaddr_decode_components_dns_variants() -> None:
    for proto, expected_code in (
        ("dns", PROTO_DNS),
        ("dns4", PROTO_DNS4),
        ("dns6", PROTO_DNS6),
    ):
        encoded = parse_multiaddr_string(f"/{proto}/example.com/tcp/443")
        codes = [c for c, _ in decode_multiaddr(encoded)]
        assert codes == [expected_code, PROTO_TCP]


def test_multiaddr_decode_udp() -> None:
    encoded = parse_multiaddr_string("/ip4/10.0.0.1/udp/53")
    codes = [c for c, _ in decode_multiaddr(encoded)]
    assert codes == [PROTO_IP4, PROTO_UDP]


@pytest.mark.parametrize("ma_str", [
    "/totallybogusproto/foo",
    "/ip4/127.0.0.1/tcp/4001/garbage/x",
    "/quic/9000",
])
def test_parse_unsupported_protocol_raises(ma_str: str) -> None:
    """Unsupported protocol names in parse_multiaddr_string must raise ValueError."""
    with pytest.raises(ValueError):
        parse_multiaddr_string(ma_str)


def test_decode_unsupported_protocol_raises() -> None:
    """A binary multiaddr with an unknown protocol code must raise ValueError."""
    # 0x99 is not in either the fixed-size or length-prefixed protocol tables.
    bogus = bytes([0x99, 0x00, 0x00, 0x00])
    with pytest.raises(ValueError, match="unsupported"):
        decode_multiaddr(bogus)
