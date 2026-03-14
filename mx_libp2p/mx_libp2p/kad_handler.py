# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kademlia inbound request handler.

Dispatches incoming Kademlia stream requests:
- FIND_NODE: return closest peers from the routing table
- GET_VALUE: return local record or closest peers
- PUT_VALUE: store record locally
"""

import asyncio
import time
import logging

from .routing import RoutingTable, xor_distance
from .multiaddr import encode_multiaddr_ip4_tcp_p2p
from .kademlia import (
    MSG_PUT_VALUE,
    MSG_GET_VALUE,
    MSG_FIND_NODE,
    MSG_PING,
    KADEMLIA_PROTOCOL,
    encode_kad_message,
    decode_kad_message,
    encode_record,
    encode_peer,
    _read_length_prefixed,
    _write_length_prefixed,
)

log = logging.getLogger(__name__)

# Maximum record value size accepted from remote peers.
# Must fit within MAX_KAD_MESSAGE_SIZE (16 KB) with protobuf framing overhead.
# Local puts (from our own application) are not subject to this limit.
MAX_RECORD_VALUE_SIZE = 14 * 1024

# Maximum number of records stored locally (matches rust-libp2p default).
MAX_RECORDS = 1024


class StoredRecord:
    """A record stored in the local Kademlia record store.

    Attributes:
        value: the record payload
        timestamp: monotonic time when the record was stored/updated
        publisher: peer ID of the node that originated this record, or None
                   if we are the originator (local put)
        ttl: per-record time-to-live in seconds, or None to use the node's
             default TTL. Enables different lifetimes for directory entries
             vs status heartbeats.
    """
    __slots__ = ("value", "timestamp", "publisher", "ttl")

    def __init__(self, value: bytes, timestamp: float, publisher: bytes | None = None, ttl: float | None = None):
        self.value = value
        self.timestamp = timestamp
        self.publisher = publisher
        self.ttl = ttl


class KadHandler:
    """Handles inbound Kademlia protocol requests."""

    def __init__(
        self,
        routing_table: RoutingTable,
        max_record_size: int = MAX_RECORD_VALUE_SIZE,
        max_records: int = MAX_RECORDS,
        record_filter=None,
    ):
        """
        Args:
            routing_table: the node's routing table
            max_record_size: max bytes for inbound record values
            max_records: max records in the local store
            record_filter: optional callable(key: bytes, value: bytes) -> bool.
                If provided, inbound PUT_VALUE records are only accepted when
                this returns True. Enables application-level validation (e.g.
                key namespace checks, value schema validation).
        """
        self.routing_table = routing_table
        self._records: dict[bytes, StoredRecord] = {}  # key -> StoredRecord
        self._max_record_size = max_record_size
        self._max_records = max_records
        self._record_filter = record_filter

    @property
    def records(self) -> dict[bytes, StoredRecord]:
        return self._records

    def put_local(self, key: bytes, value: bytes, publisher: bytes | None = None, ttl: float | None = None) -> bool:
        """Store a record locally. Returns True if stored.

        When the store is full and the key is new, evicts the record
        whose key is furthest from our node's peer ID (matching
        rust-libp2p's MemoryStore behavior). This ensures we preferentially
        keep records we're responsible for in the DHT.

        Args:
            key: record key
            value: record value
            publisher: peer ID of the originator, or None for locally originated
            ttl: per-record TTL in seconds, or None for node default
        """
        if key not in self._records and len(self._records) >= self._max_records:
            self._evict_furthest(key)
        self._records[key] = StoredRecord(value, time.monotonic(), publisher, ttl)
        return True

    def _evict_furthest(self, incoming_key: bytes) -> None:
        """Evict the record whose key is furthest from our peer ID.

        If the incoming key is further than all existing keys, it is not
        stored (the caller should handle this, but in practice we always
        evict because the incoming record is presumably closer to us or
        equally important).
        """
        local_id = self.routing_table.local_peer_id
        furthest_key = None
        furthest_dist = -1
        for k in self._records:
            d = xor_distance(k, local_id)
            if d > furthest_dist:
                furthest_dist = d
                furthest_key = k
        if furthest_key is not None:
            del self._records[furthest_key]
            log.debug(f"evicted furthest record {furthest_key!r} to make room")

    def get_local(self, key: bytes) -> StoredRecord | None:
        """Retrieve a local record, or None if not found."""
        return self._records.get(key)

    def remove_expired(self, default_max_age: float) -> int:
        """Remove records older than their TTL. Returns count removed.

        Each record uses its own TTL if set, otherwise falls back to
        default_max_age. This enables different lifetimes for directory
        entries vs status heartbeats.
        """
        now = time.monotonic()
        expired = []
        for k, rec in self._records.items():
            max_age = rec.ttl if rec.ttl is not None else default_max_age
            if now - rec.timestamp > max_age:
                expired.append(k)
        for k in expired:
            del self._records[k]
        return len(expired)

    def _closest_peers_encoded(self, target: bytes, count: int = 20) -> list[bytes]:
        """Return closest peers as encoded Peer protobufs."""
        entries = self.routing_table.closest_peers(target, count)
        result = []
        for entry in entries:
            addrs = entry.addrs if entry.addrs else []
            result.append(encode_peer(entry.peer_id, addrs))
        return result

    # Timeout for reading an inbound request (prevents slow-peer resource exhaustion)
    INBOUND_READ_TIMEOUT = 10.0

    async def handle_stream(self, stream, reader, writer, sender: bytes | None = None) -> None:
        """Handle a single inbound Kademlia stream.

        Reads one request, sends one response, then the stream is done.

        Args:
            sender: peer ID of the remote peer that opened this stream.
                Used to track record publisher for replication decisions.
        """
        try:
            request_data = await asyncio.wait_for(
                _read_length_prefixed(reader), timeout=self.INBOUND_READ_TIMEOUT
            )
            if not request_data:
                return
            msg = decode_kad_message(request_data)
            msg_type = msg.get("type")

            if msg_type == MSG_FIND_NODE:
                response = self._handle_find_node(msg)
            elif msg_type == MSG_GET_VALUE:
                response = self._handle_get_value(msg)
            elif msg_type == MSG_PUT_VALUE:
                response = self._handle_put_value(msg, sender=sender)
            elif msg_type == MSG_PING:
                response = self._handle_ping(msg)
            else:
                # Unknown message type (e.g. ADD_PROVIDER, GET_PROVIDERS).
                # Return closer peers as a graceful fallback per Kademlia spec.
                log.debug(f"unknown kad message type {msg_type}, returning closer peers")
                key = msg.get("key", b"")
                closer = self._closest_peers_encoded(key)
                response = encode_kad_message(msg_type, key=key, closer_peers=closer)

            _write_length_prefixed(writer, response)
            await writer.drain()
        except asyncio.IncompleteReadError:
            pass
        except TimeoutError:
            log.debug(f"kad handler: inbound read timed out after {self.INBOUND_READ_TIMEOUT}s")
        except Exception as e:
            log.debug(f"kad handler error: {e}", exc_info=True)
        finally:
            try:
                await stream.close()
            except Exception:
                pass

    def _handle_ping(self, msg: dict) -> bytes:
        """Handle PING: echo back a PING response.

        rust-libp2p sends PING messages for peer liveness probing.
        The response is simply a PING message echoed back.
        """
        return encode_kad_message(MSG_PING)

    def _handle_find_node(self, msg: dict) -> bytes:
        """Handle FIND_NODE: return closest peers to the requested key."""
        key = msg.get("key", b"")
        closer = self._closest_peers_encoded(key)
        return encode_kad_message(MSG_FIND_NODE, key=key, closer_peers=closer)

    def _handle_get_value(self, msg: dict) -> bytes:
        """Handle GET_VALUE: return record if we have it, otherwise closest peers."""
        key = msg.get("key", b"")
        local = self.get_local(key)
        if local is not None:
            record = encode_record(key, local.value)
            return encode_kad_message(MSG_GET_VALUE, key=key, record=record)
        else:
            closer = self._closest_peers_encoded(key)
            return encode_kad_message(MSG_GET_VALUE, key=key, closer_peers=closer)

    def _handle_put_value(self, msg: dict, sender: bytes | None = None) -> bytes:
        """Handle PUT_VALUE: store record locally.

        Rejects records that are oversized, when the store is full, or when
        the record_filter callback rejects the key/value pair.

        Args:
            sender: peer ID of the node that sent this PUT_VALUE. Tracked as
                the record's publisher for replication decisions.
        """
        key = msg.get("key", b"")
        record = msg.get("record")
        if record and record.get("value") is not None:
            value = record["value"]
            if len(value) > self._max_record_size:
                log.warning(
                    f"rejecting oversized record for key {key!r}: "
                    f"{len(value)} bytes > {self._max_record_size} limit"
                )
                return encode_kad_message(MSG_PUT_VALUE, key=key)
            if self._record_filter is not None and not self._record_filter(key, value):
                log.debug(f"record filter rejected key {key!r}")
                return encode_kad_message(MSG_PUT_VALUE, key=key)
            if key not in self._records and len(self._records) >= self._max_records:
                self._evict_furthest(key)
            # Preserve per-record TTL from the wire (rust-libp2p field 777)
            wire_ttl = record.get("ttl")
            record_ttl = float(wire_ttl) if wire_ttl is not None else None
            self._records[key] = StoredRecord(value, time.monotonic(), publisher=sender, ttl=record_ttl)
            log.debug(f"stored record for key {key!r} ({len(value)} bytes, ttl={record_ttl})")
            # Confirm storage by echoing just the key (no record).
            # rust-libp2p does not echo the record in PUT_VALUE responses.
            return encode_kad_message(MSG_PUT_VALUE, key=key)
        return encode_kad_message(MSG_PUT_VALUE, key=key)
