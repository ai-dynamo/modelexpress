# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""mDNS peer discovery for libp2p nodes.

Implements the libp2p mDNS discovery spec: multicast DNS queries and responses
on 224.0.0.251:5353 using the _p2p._udp.local service name. Wire-compatible
with rust-libp2p's libp2p-mdns crate.

IPv4-only for now. IPv6 can be added later.
"""

import asyncio
import logging
import random
import socket
import string
import struct
import time
from collections.abc import Awaitable, Callable

log = logging.getLogger(__name__)

MDNS_ADDR = "224.0.0.251"
MDNS_PORT = 5353
SERVICE_NAME = "_p2p._udp.local"
MAX_PACKET_SIZE = 8932

# DNS record types
TYPE_PTR = 12
TYPE_TXT = 16

# DNS class: IN with cache-flush bit set
CLASS_CACHE_FLUSH = 0x8001
CLASS_IN = 0x0001


# -- DNS wire format helpers --------------------------------------------------


def _encode_dns_name(name: str) -> bytes:
    """Encode a DNS name as length-prefixed labels, null-terminated."""
    result = bytearray()
    for label in name.split("."):
        encoded = label.encode("utf-8")
        result.append(len(encoded))
        result.extend(encoded)
    result.append(0)
    return bytes(result)


def _decode_dns_name(data: bytes, offset: int) -> tuple[str, int]:
    """Decode a DNS name, handling 0xC0XX pointer compression."""
    labels = []
    seen_offsets: set[int] = set()
    jump_target = -1
    while offset < len(data):
        if offset in seen_offsets:
            raise ValueError("circular pointer in DNS name")
        seen_offsets.add(offset)
        length = data[offset]
        if length == 0:
            offset += 1
            break
        if (length & 0xC0) == 0xC0:
            # Pointer compression
            if jump_target < 0:
                jump_target = offset + 2
            pointer = struct.unpack("!H", data[offset : offset + 2])[0] & 0x3FFF
            offset = pointer
            continue
        offset += 1
        labels.append(data[offset : offset + length].decode("utf-8"))
        offset += length
    if jump_target >= 0:
        offset = jump_target
    return ".".join(labels), offset


def _build_query() -> bytes:
    """Build a PTR query for _p2p._udp.local."""
    # Header: ID=0, flags=0x0000, QDCOUNT=1, ANCOUNT=0, NSCOUNT=0, ARCOUNT=0
    header = struct.pack("!HHHHHH", 0, 0x0000, 1, 0, 0, 0)
    # Question: QNAME + QTYPE(PTR=12) + QCLASS(IN=1)
    question = _encode_dns_name(SERVICE_NAME) + struct.pack("!HH", TYPE_PTR, CLASS_IN)
    return header + question


def _encode_txt_rdata(entries: list[str]) -> bytes:
    """Encode TXT RDATA: each entry is a length-prefixed string."""
    result = bytearray()
    for entry in entries:
        encoded = entry.encode("utf-8")
        if len(encoded) > 255:
            log.warning(f"TXT entry too long ({len(encoded)} bytes), truncating")
            encoded = encoded[:255]
        result.append(len(encoded))
        result.extend(encoded)
    return bytes(result)


def _build_response(peer_name: str, multiaddr_strings: list[str], ttl: int) -> bytes:
    """Build a DNS response with PTR answer + TXT additional section.

    PTR answer: _p2p._udp.local -> <peer_name>._p2p._udp.local
    TXT additional: dnsaddr=<multiaddr> for each address
    """
    fqdn = f"{peer_name}.{SERVICE_NAME}"
    service_encoded = _encode_dns_name(SERVICE_NAME)
    fqdn_encoded = _encode_dns_name(fqdn)

    # Header: ID=0, flags=0x8400 (response, authoritative), QD=0, AN=1, NS=0, AR=1
    header = struct.pack("!HHHHHH", 0, 0x8400, 0, 1, 0, 1)

    # PTR answer record
    ptr_rdata = fqdn_encoded
    ptr_record = (
        service_encoded
        + struct.pack("!HHI", TYPE_PTR, CLASS_CACHE_FLUSH, ttl)
        + struct.pack("!H", len(ptr_rdata))
        + ptr_rdata
    )

    # TXT additional record with dnsaddr entries
    txt_entries = [f"dnsaddr={addr}" for addr in multiaddr_strings]
    txt_rdata = _encode_txt_rdata(txt_entries)
    txt_record = (
        fqdn_encoded
        + struct.pack("!HHI", TYPE_TXT, CLASS_CACHE_FLUSH, ttl)
        + struct.pack("!H", len(txt_rdata))
        + txt_rdata
    )

    packet = header + ptr_record + txt_record
    if len(packet) > MAX_PACKET_SIZE:
        log.warning(
            f"mDNS response packet exceeds {MAX_PACKET_SIZE} bytes "
            f"({len(packet)} bytes), peers may not parse it"
        )
    return packet


def _parse_txt_records(rdata: bytes) -> list[str]:
    """Split TXT RDATA into individual strings."""
    entries = []
    offset = 0
    while offset < len(rdata):
        length = rdata[offset]
        offset += 1
        if offset + length > len(rdata):
            break
        entries.append(rdata[offset : offset + length].decode("utf-8", errors="replace"))
        offset += length
    return entries


def _extract_peers_from_packet(data: bytes) -> list[tuple[str, int]]:
    """Parse a DNS packet and extract discovered peer multiaddrs.

    Returns list of (multiaddr_string, ttl) pairs.
    """
    if len(data) < 12:
        return []

    _id, flags, qdcount, ancount, _nscount, arcount = struct.unpack(
        "!HHHHHH", data[:12]
    )

    # Only process responses (QR bit set)
    if not (flags & 0x8000):
        return []

    offset = 12

    # Skip questions
    for _ in range(qdcount):
        _name, offset = _decode_dns_name(data, offset)
        offset += 4

    # Parse answer records - look for PTR records pointing to our service
    ptr_names: dict[str, int] = {}  # target name -> TTL
    for _ in range(ancount):
        if offset >= len(data):
            break
        name, offset = _decode_dns_name(data, offset)
        if offset + 10 > len(data):
            break
        rtype, rclass, rttl, rdlength = struct.unpack("!HHIH", data[offset : offset + 10])
        offset += 10
        rdata = data[offset : offset + rdlength]
        offset += rdlength
        if rtype == TYPE_PTR and name.lower() == SERVICE_NAME:
            target, _ = _decode_dns_name(rdata, 0)
            ptr_names[target.lower()] = rttl

    # Skip NS records (nscount)
    # _nscount was parsed from the header but we already skipped it in offset tracking
    # Actually we need to skip authority section too
    # Re-parse from the point after answers to skip authority
    for _ in range(_nscount):
        if offset >= len(data):
            break
        _name, offset = _decode_dns_name(data, offset)
        if offset + 10 > len(data):
            break
        _rt, _rc, _rttl, rdlength = struct.unpack("!HHIH", data[offset : offset + 10])
        offset += 10 + rdlength

    # Parse additional records - look for TXT records matching PTR targets
    results: list[tuple[str, int]] = []
    for _ in range(arcount):
        if offset >= len(data):
            break
        name, offset = _decode_dns_name(data, offset)
        if offset + 10 > len(data):
            break
        rtype, _rclass, rttl, rdlength = struct.unpack("!HHIH", data[offset : offset + 10])
        offset += 10
        rdata = data[offset : offset + rdlength]
        offset += rdlength
        if rtype == TYPE_TXT and name.lower() in ptr_names:
            ttl = ptr_names[name.lower()]
            for entry in _parse_txt_records(rdata):
                if entry.startswith("dnsaddr="):
                    results.append((entry[len("dnsaddr="):], ttl))

    return results


def _is_query_for_service(data: bytes) -> bool:
    """Check if this DNS packet is a query for _p2p._udp.local."""
    if len(data) < 12:
        return False
    _id, flags, qdcount = struct.unpack("!HHH", data[:6])
    # Must be a query (QR bit not set)
    if flags & 0x8000:
        return False
    offset = 12
    for _ in range(qdcount):
        if offset >= len(data):
            break
        name, offset = _decode_dns_name(data, offset)
        if offset + 4 > len(data):
            break
        qtype, _qclass = struct.unpack("!HH", data[offset : offset + 4])
        offset += 4
        if qtype == TYPE_PTR and name.lower() == SERVICE_NAME:
            return True
    return False


def _get_local_ips() -> list[str]:
    """Get local IP addresses using the UDP connect trick.

    Connects a UDP socket to the mDNS multicast address to find which
    interface the OS would use, then returns that IP.
    """
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        try:
            s.connect((MDNS_ADDR, MDNS_PORT))
            ip = s.getsockname()[0]
            if ip and ip != "0.0.0.0":
                ips.append(ip)
        finally:
            s.close()
    except OSError:
        pass
    return ips


# -- Transport ----------------------------------------------------------------


class _MdnsProtocol(asyncio.DatagramProtocol):
    """UDP protocol for mDNS multicast traffic."""

    def __init__(self, on_datagram: Callable[[bytes, tuple[str, int]], None]):
        self._on_datagram = on_datagram
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self._on_datagram(data, addr)

    def error_received(self, exc: Exception) -> None:
        log.debug(f"mDNS socket error: {exc}")

    def connection_lost(self, exc: Exception | None) -> None:
        pass


# -- MdnsDiscovery ------------------------------------------------------------


class MdnsDiscovery:
    """mDNS-based peer discovery for libp2p nodes.

    Sends periodic PTR queries for _p2p._udp.local and responds with our
    own addresses. Discovered peers are passed to on_peer_discovered callback.
    """

    def __init__(
        self,
        peer_id: bytes,
        peer_id_b58: str,
        listen_addrs: Callable[[], list[str]],
        on_peer_discovered: Callable[[str], Awaitable[None]],
        query_interval: float = 300.0,
        ttl: int = 360,
    ):
        self._peer_id = peer_id
        self._peer_id_b58 = peer_id_b58
        self._listen_addrs = listen_addrs
        self._on_peer_discovered = on_peer_discovered
        self._query_interval = query_interval
        self._ttl = ttl

        # Random peer name (NOT the peer ID) - 32 lowercase alphanumeric chars
        self._peer_name = "".join(random.choices(string.ascii_lowercase + string.digits, k=32))

        self._discovered: dict[str, float] = {}  # multiaddr -> monotonic expiry
        self._transport: asyncio.DatagramTransport | None = None
        self._protocol: _MdnsProtocol | None = None
        self._query_task: asyncio.Task | None = None
        self._sock: socket.socket | None = None

    async def start(self) -> None:
        """Start mDNS discovery. Fails gracefully if socket bind fails."""
        loop = asyncio.get_event_loop()

        try:
            # Create the UDP socket manually for multicast setup
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, "SO_REUSEPORT"):
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except OSError:
                    pass
            sock.bind(("", MDNS_PORT))

            # Join multicast group
            mreq = struct.pack(
                "4s4s",
                socket.inet_aton(MDNS_ADDR),
                socket.inet_aton("0.0.0.0"),
            )
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_LOOP, 1)
            sock.setblocking(False)
            self._sock = sock

            transport, protocol = await loop.create_datagram_endpoint(
                lambda: _MdnsProtocol(self._on_datagram),
                sock=sock,
            )
            self._transport = transport  # type: ignore[assignment]
            self._protocol = protocol  # type: ignore[assignment]

        except OSError as e:
            log.warning(f"mDNS: failed to bind multicast socket: {e} - continuing without mDNS")
            return

        log.info(f"mDNS: started discovery (peer_name={self._peer_name[:8]}...)")

        # Send initial query
        self._send_packet(_build_query())

        # Start query loop
        self._query_task = asyncio.create_task(self._query_loop())

    async def stop(self) -> None:
        """Stop mDNS discovery and clean up."""
        if self._query_task:
            self._query_task.cancel()
            try:
                await self._query_task
            except asyncio.CancelledError:
                pass
            self._query_task = None

        if self._sock:
            try:
                mreq = struct.pack(
                    "4s4s",
                    socket.inet_aton(MDNS_ADDR),
                    socket.inet_aton("0.0.0.0"),
                )
                self._sock.setsockopt(socket.IPPROTO_IP, socket.IP_DROP_MEMBERSHIP, mreq)
            except OSError:
                pass

        if self._transport:
            self._transport.close()
            self._transport = None

        self._protocol = None
        self._sock = None
        log.info("mDNS: stopped")

    def send_query(self) -> None:
        """Send an mDNS query. Can be called externally to trigger re-discovery."""
        self._send_packet(_build_query())

    def _send_packet(self, data: bytes) -> None:
        """Send a packet to the mDNS multicast group."""
        if self._transport:
            try:
                self._transport.sendto(data, (MDNS_ADDR, MDNS_PORT))
            except OSError as e:
                log.debug(f"mDNS: failed to send packet: {e}")

    async def _query_loop(self) -> None:
        """Periodic query loop with exponential backoff at startup."""
        try:
            # Exponential backoff: 0.5s, 1s, 2s, 4s, ... up to query_interval
            delay = 0.5
            while True:
                await asyncio.sleep(delay)
                self._send_packet(_build_query())
                self._expire_discovered()
                if delay < self._query_interval:
                    delay = min(delay * 2, self._query_interval)
        except asyncio.CancelledError:
            pass

    def _expire_discovered(self) -> None:
        """Remove expired entries from the discovered set."""
        now = time.monotonic()
        expired = [addr for addr, expiry in self._discovered.items() if expiry <= now]
        for addr in expired:
            del self._discovered[addr]

    def _on_datagram(self, data: bytes, addr: tuple[str, int]) -> None:
        """Handle incoming mDNS packet (called from protocol)."""
        try:
            if _is_query_for_service(data):
                self._send_response()
                return

            peers = _extract_peers_from_packet(data)
            for multiaddr_str, ttl in peers:
                # Filter out our own peer ID
                if f"/p2p/{self._peer_id_b58}" in multiaddr_str:
                    continue

                now = time.monotonic()
                expiry = now + (ttl if ttl > 0 else self._ttl)

                if multiaddr_str in self._discovered and self._discovered[multiaddr_str] > now:
                    # Already known and not expired - just update expiry
                    self._discovered[multiaddr_str] = expiry
                    continue

                self._discovered[multiaddr_str] = expiry
                log.info(f"mDNS: discovered peer {multiaddr_str}")

                # Schedule the async callback
                loop = asyncio.get_event_loop()
                loop.create_task(self._safe_callback(multiaddr_str))

        except Exception as e:
            log.debug(f"mDNS: error handling packet from {addr}: {e}")

    async def _safe_callback(self, multiaddr_str: str) -> None:
        """Invoke the discovery callback with error handling."""
        try:
            await self._on_peer_discovered(multiaddr_str)
        except Exception as e:
            log.debug(f"mDNS: peer discovery callback failed for {multiaddr_str}: {e}")

    def _send_response(self) -> None:
        """Send our mDNS response announcing our addresses."""
        addrs = self._listen_addrs()
        if not addrs:
            return
        packet = _build_response(self._peer_name, addrs, self._ttl)
        self._send_packet(packet)
