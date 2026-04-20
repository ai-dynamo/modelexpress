# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Substrate-agnostic mDNS peer discovery.

Implements RFC 6762 (mDNS) and RFC 6763 (DNS-SD) PTR + TXT service discovery
over UDP multicast on 224.0.0.251:5353. Service name, instance identifier,
and TXT record contents are all parameterized at the API level so this module
can drive discovery for arbitrary peer networks.

Wire format:
- PTR query asking "who offers <service_name>?"
- PTR + TXT response announcing "<instance>.<service_name>" with TXT entries
- TXT entries are encoded as ``key=value`` strings; the caller chooses keys.
- Two TXT keys carry special meaning for the :class:`MdnsDiscovery` return
  shape: ``host`` and ``port``. The caller sets them when announcing; the
  discovery callback receives them parsed out into the ``(host, port, txt)``
  tuple. Entries without ``=`` or with empty keys are ignored.

IPv4-only for now. IPv6 can be added later without breaking the wire format.
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
DEFAULT_SERVICE_NAME = "_mx-peer._tcp.local"
MAX_PACKET_SIZE = 8932

# DNS record types
TYPE_PTR = 12
TYPE_TXT = 16

# DNS class: IN with cache-flush bit set
CLASS_CACHE_FLUSH = 0x8001
CLASS_IN = 0x0001

# Reserved TXT keys used to transport (host, port) out-of-band from the
# otherwise free-form TXT dictionary. Chosen to match common DNS-SD practice.
TXT_KEY_HOST = "host"
TXT_KEY_PORT = "port"


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


def _build_query(service_name: str) -> bytes:
    """Build a PTR query for ``service_name``."""
    # Header: ID=0, flags=0x0000, QDCOUNT=1, ANCOUNT=0, NSCOUNT=0, ARCOUNT=0
    header = struct.pack("!HHHHHH", 0, 0x0000, 1, 0, 0, 0)
    # Question: QNAME + QTYPE(PTR=12) + QCLASS(IN=1)
    question = _encode_dns_name(service_name) + struct.pack("!HH", TYPE_PTR, CLASS_IN)
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


def _txt_dict_to_entries(txt: dict[str, str]) -> list[str]:
    """Serialize a TXT dict into ``key=value`` strings."""
    entries = []
    for key, value in txt.items():
        if not key:
            continue
        entries.append(f"{key}={value}")
    return entries


def _build_response(
    instance_name: str,
    service_name: str,
    txt: dict[str, str],
    ttl: int,
) -> bytes:
    """Build a DNS response with PTR answer + TXT additional section.

    PTR answer: ``service_name`` -> ``<instance_name>.<service_name>``
    TXT additional: one ``key=value`` per entry in ``txt``
    """
    fqdn = f"{instance_name}.{service_name}"
    service_encoded = _encode_dns_name(service_name)
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

    # TXT additional record
    txt_entries = _txt_dict_to_entries(txt)
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


def _txt_entries_to_dict(entries: list[str]) -> dict[str, str]:
    """Parse ``key=value`` TXT entries into a dict.

    Entries without ``=`` or with an empty key are skipped. Later duplicates
    overwrite earlier ones (matches common DNS-SD consumer behavior).
    """
    out: dict[str, str] = {}
    for entry in entries:
        eq = entry.find("=")
        if eq <= 0:
            continue
        out[entry[:eq]] = entry[eq + 1 :]
    return out


def _extract_peers_from_packet(
    data: bytes,
    service_name: str,
) -> list[tuple[str, int, dict[str, str], str, int]]:
    """Parse a DNS packet and extract announcements for ``service_name``.

    Returns a list of ``(host, port, txt_records, instance_name, ttl)``
    tuples. ``host`` defaults to empty string and ``port`` to 0 if the
    announcement did not include reserved TXT keys ``host`` / ``port``.
    """
    if len(data) < 12:
        return []

    _id, flags, qdcount, ancount, nscount, arcount = struct.unpack(
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

    service_lower = service_name.lower()
    # target FQDN (lowercased) -> (TTL, instance_label)
    ptr_names: dict[str, tuple[int, str]] = {}
    for _ in range(ancount):
        if offset >= len(data):
            break
        name, offset = _decode_dns_name(data, offset)
        if offset + 10 > len(data):
            break
        rtype, _rclass, rttl, rdlength = struct.unpack("!HHIH", data[offset : offset + 10])
        offset += 10
        rdata = data[offset : offset + rdlength]
        offset += rdlength
        if rtype == TYPE_PTR and name.lower() == service_lower:
            target, _ = _decode_dns_name(rdata, 0)
            instance_label = target.split(".", 1)[0] if "." in target else target
            ptr_names[target.lower()] = (rttl, instance_label)

    # Skip authority section
    for _ in range(nscount):
        if offset >= len(data):
            break
        _name, offset = _decode_dns_name(data, offset)
        if offset + 10 > len(data):
            break
        _rt, _rc, _rttl, rdlength = struct.unpack("!HHIH", data[offset : offset + 10])
        offset += 10 + rdlength

    # Parse additional records - look for TXT records matching PTR targets
    results: list[tuple[str, int, dict[str, str], str, int]] = []
    for _ in range(arcount):
        if offset >= len(data):
            break
        name, offset = _decode_dns_name(data, offset)
        if offset + 10 > len(data):
            break
        rtype, _rclass, _rttl, rdlength = struct.unpack("!HHIH", data[offset : offset + 10])
        offset += 10
        rdata = data[offset : offset + rdlength]
        offset += rdlength
        if rtype == TYPE_TXT and name.lower() in ptr_names:
            ptr_ttl, instance_label = ptr_names[name.lower()]
            txt_entries = _parse_txt_records(rdata)
            txt_dict = _txt_entries_to_dict(txt_entries)
            host = txt_dict.get(TXT_KEY_HOST, "")
            port_str = txt_dict.get(TXT_KEY_PORT, "")
            try:
                port = int(port_str) if port_str else 0
            except ValueError:
                port = 0
            results.append((host, port, txt_dict, instance_label, ptr_ttl))

    return results


def _is_query_for_service(data: bytes, service_name: str) -> bool:
    """Check if this DNS packet is a query for ``service_name``."""
    if len(data) < 12:
        return False
    _id, flags, qdcount = struct.unpack("!HHH", data[:6])
    # Must be a query (QR bit not set)
    if flags & 0x8000:
        return False
    offset = 12
    service_lower = service_name.lower()
    for _ in range(qdcount):
        if offset >= len(data):
            break
        name, offset = _decode_dns_name(data, offset)
        if offset + 4 > len(data):
            break
        qtype, _qclass = struct.unpack("!HH", data[offset : offset + 4])
        offset += 4
        if qtype == TYPE_PTR and name.lower() == service_lower:
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


def _random_instance_name() -> str:
    """Generate a random 32-char lowercase alphanumeric instance label."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=32))


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


#: Callback signature invoked when a new peer is discovered.
#: Arguments: ``(host, port, txt_records)``.
PeerDiscoveryCallback = Callable[[str, int, dict[str, str]], Awaitable[None]]

#: Supplier for the TXT records to announce in our own responses. Called
#: fresh each time we answer a query so addresses can change over the
#: lifetime of the discovery instance.
TxtSupplier = Callable[[], dict[str, str]]


class MdnsDiscovery:
    """mDNS-based peer discovery, parameterized by service and instance name.

    Sends periodic PTR queries for ``service_name`` and responds with our own
    TXT announcement. Discovered peers are passed to ``on_peer_discovered``
    as ``(host, port, txt_records)`` tuples. The caller's own announcements
    are filtered out based on the instance label, not the source IP, because
    loopback multicast echoes our own packets.

    :param on_peer_discovered: Async callback invoked for each newly
        discovered peer.
    :param txt_supplier: Callable returning the TXT record dict to announce.
        The dict should include reserved keys ``host`` and ``port`` so peers
        can reach us; other keys pass through unchanged.
    :param service_name: mDNS service name (e.g. ``"_mx-peer._tcp.local"``).
    :param instance_name: Label used to prefix the FQDN. Defaults to a
        random 32-char string. Must be unique per process; callers that want
        to identify themselves stably can pass a deterministic value.
    :param query_interval: Upper bound (seconds) on the periodic re-query
        cadence. The query loop uses exponential backoff starting at 0.5s.
    :param ttl: TTL (seconds) placed on our own PTR/TXT records.
    """

    def __init__(
        self,
        on_peer_discovered: PeerDiscoveryCallback,
        txt_supplier: TxtSupplier,
        *,
        service_name: str = DEFAULT_SERVICE_NAME,
        instance_name: str | None = None,
        query_interval: float = 300.0,
        ttl: int = 360,
    ):
        self._on_peer_discovered = on_peer_discovered
        self._txt_supplier = txt_supplier
        self._service_name = service_name
        self._instance_name = instance_name or _random_instance_name()
        self._query_interval = query_interval
        self._ttl = ttl

        # (host, port) -> monotonic expiry. Simple dedup keyed on addr pair;
        # callers that need per-TXT dedup can do so in the callback.
        self._discovered: dict[tuple[str, int], float] = {}
        self._transport: asyncio.DatagramTransport | None = None
        self._protocol: _MdnsProtocol | None = None
        self._query_task: asyncio.Task | None = None
        self._sock: socket.socket | None = None

    @property
    def service_name(self) -> str:
        return self._service_name

    @property
    def instance_name(self) -> str:
        return self._instance_name

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

        log.info(
            f"mDNS: started discovery (service={self._service_name} "
            f"instance={self._instance_name[:8]}...)"
        )

        # Send initial query
        self._send_packet(_build_query(self._service_name))

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
        self._send_packet(_build_query(self._service_name))

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
                self._send_packet(_build_query(self._service_name))
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
            if _is_query_for_service(data, self._service_name):
                self._send_response()
                return

            peers = _extract_peers_from_packet(data, self._service_name)
            for host, port, txt_records, instance_label, ttl in peers:
                # Filter out our own announcements by instance label.
                # Loopback multicast echoes packets back to the sender.
                if instance_label == self._instance_name:
                    continue
                # Skip entries that lack host/port - nothing we can do with them
                if not host or port <= 0:
                    continue

                key = (host, port)
                now = time.monotonic()
                expiry = now + (ttl if ttl > 0 else self._ttl)

                if key in self._discovered and self._discovered[key] > now:
                    # Already known and not expired - just update expiry
                    self._discovered[key] = expiry
                    continue

                self._discovered[key] = expiry
                log.info(f"mDNS: discovered peer {host}:{port}")

                # Schedule the async callback
                loop = asyncio.get_event_loop()
                loop.create_task(self._safe_callback(host, port, txt_records))

        except Exception as e:
            log.debug(f"mDNS: error handling packet from {addr}: {e}")

    async def _safe_callback(
        self, host: str, port: int, txt_records: dict[str, str]
    ) -> None:
        """Invoke the discovery callback with error handling."""
        try:
            await self._on_peer_discovered(host, port, txt_records)
        except Exception as e:
            log.debug(
                f"mDNS: peer discovery callback failed for {host}:{port}: {e}"
            )

    def _send_response(self) -> None:
        """Send our mDNS response announcing our TXT records."""
        txt = self._txt_supplier()
        if not txt:
            return
        packet = _build_response(
            self._instance_name, self._service_name, txt, self._ttl
        )
        self._send_packet(packet)
