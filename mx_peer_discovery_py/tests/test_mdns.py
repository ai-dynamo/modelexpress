# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mDNS peer discovery tests: DNS wire format, parameterization, construction.

Covers:
- DNS name encoding/decoding (with pointer compression)
- Query and response packet construction with parameterized service name
- Roundtrip: build response -> parse -> extract (host, port, txt)
- TXT dict encoding and round-trip with arbitrary keys
- Service-name scoping: query for service A is not confused with service B
- Handcrafted-packet parse to confirm wire format compatibility
- MdnsDiscovery construction with defaults and with custom service/instance
"""

import struct

from mx_peer_discovery.mdns import (
    CLASS_CACHE_FLUSH,
    DEFAULT_SERVICE_NAME,
    TXT_KEY_HOST,
    TXT_KEY_PORT,
    MdnsDiscovery,
    _build_query,
    _build_response,
    _decode_dns_name,
    _encode_dns_name,
    _encode_txt_rdata,
    _extract_peers_from_packet,
    _is_query_for_service,
    _parse_txt_records,
    _txt_entries_to_dict,
)


# -- DNS name encoding/decoding -----------------------------------------------


def test_encode_dns_name():
    encoded = _encode_dns_name("_mx-peer._tcp.local")
    # Labels: 8 "_mx-peer" | 4 "_tcp" | 5 "local" | 0 (null terminator)
    assert encoded == (
        b"\x08_mx-peer"
        b"\x04_tcp"
        b"\x05local"
        b"\x00"
    )


def test_encode_dns_name_single_label():
    assert _encode_dns_name("localhost") == b"\x09localhost\x00"


def test_decode_dns_name():
    raw = b"\x08_mx-peer\x04_tcp\x05local\x00"
    name, offset = _decode_dns_name(raw, 0)
    assert name == "_mx-peer._tcp.local"
    assert offset == len(raw)


def test_decode_dns_name_pointer_compression():
    """Decode with 0xC0XX pointer compression."""
    full = b"\x08_mx-peer\x04_tcp\x05local\x00"
    data = full + b"\xc0\x00"
    name, offset = _decode_dns_name(data, len(full))
    assert name == "_mx-peer._tcp.local"
    assert offset == len(full) + 2


# -- Query building -----------------------------------------------------------


def test_build_query_default_service():
    packet = _build_query(DEFAULT_SERVICE_NAME)
    assert len(packet) >= 12
    _id, flags, qdcount, ancount, _ns, _ar = struct.unpack("!HHHHHH", packet[:12])
    assert flags == 0x0000
    assert qdcount == 1
    assert ancount == 0
    assert _is_query_for_service(packet, DEFAULT_SERVICE_NAME)


def test_build_query_custom_service():
    service = "_custom._tcp.local"
    packet = _build_query(service)
    assert _is_query_for_service(packet, service)
    # A query for a different service name must not match.
    assert not _is_query_for_service(packet, "_other._tcp.local")


def test_is_query_for_service_rejects_response():
    # Response packets should never be detected as queries
    txt = {TXT_KEY_HOST: "10.0.0.1", TXT_KEY_PORT: "4001"}
    packet = _build_response("peer", DEFAULT_SERVICE_NAME, txt, 360)
    assert not _is_query_for_service(packet, DEFAULT_SERVICE_NAME)


# -- Response building + roundtrip -------------------------------------------


def test_build_response_basic():
    txt = {TXT_KEY_HOST: "192.168.1.5", TXT_KEY_PORT: "4001"}
    packet = _build_response("mypeer", DEFAULT_SERVICE_NAME, txt, 360)

    _id, flags, qdcount, ancount, _ns, arcount = struct.unpack("!HHHHHH", packet[:12])
    assert flags == 0x8400  # response + authoritative
    assert qdcount == 0
    assert ancount == 1
    assert arcount == 1

    peers = _extract_peers_from_packet(packet, DEFAULT_SERVICE_NAME)
    assert len(peers) == 1
    host, port, txt_dict, instance, ttl = peers[0]
    assert host == "192.168.1.5"
    assert port == 4001
    assert txt_dict == txt
    assert instance == "mypeer"
    assert ttl == 360


def test_roundtrip_multiple_txt_keys():
    txt = {
        TXT_KEY_HOST: "10.0.0.1",
        TXT_KEY_PORT: "4001",
        "role": "worker",
        "zone": "us-west",
    }
    packet = _build_response("nodeA", DEFAULT_SERVICE_NAME, txt, 120)
    peers = _extract_peers_from_packet(packet, DEFAULT_SERVICE_NAME)
    assert len(peers) == 1
    host, port, txt_dict, _inst, _ttl = peers[0]
    assert host == "10.0.0.1"
    assert port == 4001
    assert txt_dict == txt


def test_missing_host_port_defaults_to_empty_zero():
    """A peer that forgot host/port TXT keys still parses but gets defaults."""
    txt = {"role": "orphan"}
    packet = _build_response("peer", DEFAULT_SERVICE_NAME, txt, 60)
    peers = _extract_peers_from_packet(packet, DEFAULT_SERVICE_NAME)
    assert len(peers) == 1
    host, port, txt_dict, _inst, _ttl = peers[0]
    assert host == ""
    assert port == 0
    assert txt_dict == {"role": "orphan"}


def test_non_numeric_port_falls_back_to_zero():
    txt = {TXT_KEY_HOST: "h", TXT_KEY_PORT: "not-a-number"}
    packet = _build_response("peer", DEFAULT_SERVICE_NAME, txt, 60)
    peers = _extract_peers_from_packet(packet, DEFAULT_SERVICE_NAME)
    assert len(peers) == 1
    assert peers[0][1] == 0


def test_service_name_scoping():
    """A packet announcing service A must not show up when parsing for service B."""
    service_a = "_service-a._tcp.local"
    service_b = "_service-b._tcp.local"
    txt = {TXT_KEY_HOST: "10.1.1.1", TXT_KEY_PORT: "7000"}
    packet = _build_response("peerA", service_a, txt, 120)

    assert len(_extract_peers_from_packet(packet, service_a)) == 1
    assert _extract_peers_from_packet(packet, service_b) == []


def test_cache_flush_bit_present():
    """Responses must set the cache-flush class bit."""
    txt = {TXT_KEY_HOST: "1.2.3.4", TXT_KEY_PORT: "5"}
    packet = _build_response("peer", DEFAULT_SERVICE_NAME, txt, 120)
    assert struct.pack("!H", CLASS_CACHE_FLUSH) in packet[12:]


# -- TXT record helpers -------------------------------------------------------


def test_txt_dict_entries_roundtrip():
    entries = ["key=value", "another=entry", "dnsaddr=/ip4/1.2.3.4/tcp/5/p2p/QmFoo"]
    rdata = _encode_txt_rdata(entries)
    assert _parse_txt_records(rdata) == entries


def test_txt_entries_to_dict_skips_malformed():
    """Entries without '=' or with empty keys are dropped."""
    entries = ["host=1.2.3.4", "no-equals-here", "=empty-key", "good=yes"]
    d = _txt_entries_to_dict(entries)
    assert d == {"host": "1.2.3.4", "good": "yes"}


def test_txt_entries_to_dict_value_with_equals():
    """Equal signs beyond the first one are preserved in the value."""
    d = _txt_entries_to_dict(["url=https://example.com/?a=1&b=2"])
    assert d == {"url": "https://example.com/?a=1&b=2"}


# -- Handcrafted packet parse (wire-format compatibility) --------------------


def test_parse_handcrafted_packet():
    """Parse a byte-level handcrafted packet to validate the wire format.

    This asserts the parser accepts the exact structure we claim to emit:
    PTR + TXT with cache-flush class, flags 0x8400, key=value TXT entries.
    """
    service_name = "_mx-peer._tcp.local"
    service_encoded = _encode_dns_name(service_name)
    fqdn = _encode_dns_name(f"testpeer.{service_name}")

    txt_bytes = bytearray()
    for entry in ("host=10.0.0.5", "port=9000", "role=leader"):
        eb = entry.encode("utf-8")
        txt_bytes.append(len(eb))
        txt_bytes.extend(eb)

    header = struct.pack("!HHHHHH", 0, 0x8400, 0, 1, 0, 1)
    ptr_record = (
        service_encoded
        + struct.pack("!HHI", 12, 0x8001, 300)  # PTR, cache-flush, TTL=300
        + struct.pack("!H", len(fqdn))
        + fqdn
    )
    txt_record = (
        fqdn
        + struct.pack("!HHI", 16, 0x8001, 300)
        + struct.pack("!H", len(txt_bytes))
        + bytes(txt_bytes)
    )
    packet = header + ptr_record + txt_record

    peers = _extract_peers_from_packet(packet, service_name)
    assert len(peers) == 1
    host, port, txt_dict, instance, ttl = peers[0]
    assert host == "10.0.0.5"
    assert port == 9000
    assert txt_dict == {"host": "10.0.0.5", "port": "9000", "role": "leader"}
    assert instance == "testpeer"
    assert ttl == 300


# -- MdnsDiscovery construction ----------------------------------------------


def test_construct_default_service():
    """Can build a discovery instance with default service name."""

    async def on_peer(host, port, txt):
        pass

    def supplier():
        return {TXT_KEY_HOST: "127.0.0.1", TXT_KEY_PORT: "1234"}

    mdns = MdnsDiscovery(on_peer, supplier)
    assert mdns.service_name == DEFAULT_SERVICE_NAME
    assert isinstance(mdns.instance_name, str)
    assert len(mdns.instance_name) > 0


def test_construct_custom_service_and_instance():
    async def on_peer(host, port, txt):
        pass

    def supplier():
        return {}

    mdns = MdnsDiscovery(
        on_peer,
        supplier,
        service_name="_custom._tcp.local",
        instance_name="stable-id",
    )
    assert mdns.service_name == "_custom._tcp.local"
    assert mdns.instance_name == "stable-id"
