# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateless utility functions for the DHT layer."""

import asyncio
import logging
import socket
import struct

from .multiaddr import (
    decode_multiaddr,
    parse_multiaddr_string,
    PROTO_IP4,
    PROTO_IP6,
    PROTO_TCP,
    PROTO_P2P,
)

log = logging.getLogger(__name__)


def _log_task_exception(task: asyncio.Task) -> None:
    """Done callback that logs unhandled exceptions from fire-and-forget tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        log.warning(f"background task {task.get_name()} failed: {exc}", exc_info=exc)


def _filter_routable_addrs(addrs: list[bytes]) -> list[bytes]:
    """Filter out multiaddrs containing non-routable IP addresses (0.0.0.0, ::).

    Returns only addresses that are safe to store in routing tables and peer stores.
    """
    result = []
    for addr in addrs:
        try:
            components = decode_multiaddr(addr)
        except Exception:
            continue
        routable = True
        for code, data in components:
            if code == PROTO_IP4 and socket.inet_ntoa(data) == "0.0.0.0":
                routable = False
                break
            if code == PROTO_IP6 and socket.inet_ntop(socket.AF_INET6, data) == "::":
                routable = False
                break
        if routable:
            result.append(addr)
    return result


def _parse_peer_multiaddr(addr_str: str) -> tuple[bytes | None, str | None, int | None]:
    """Parse a multiaddr string and extract (peer_id, host, port).

    Supports formats:
        /ip4/<host>/tcp/<port>/p2p/<peer_id>
        /ip6/<host>/tcp/<port>/p2p/<peer_id>
        /dns/<host>/tcp/<port>/p2p/<peer_id>
        /dns4/<host>/tcp/<port>/p2p/<peer_id>
        /dns6/<host>/tcp/<port>/p2p/<peer_id>
    """
    addr_bytes = parse_multiaddr_string(addr_str)
    components = decode_multiaddr(addr_bytes)

    peer_id = None
    host = None
    port = None

    for code, data in components:
        if code == PROTO_IP4:
            host = socket.inet_ntoa(data)
        elif code == PROTO_IP6:
            host = socket.inet_ntop(socket.AF_INET6, data)
        elif code == PROTO_TCP:
            port = struct.unpack(">H", data)[0]
        elif code == PROTO_P2P:
            peer_id = data

    return peer_id, host, port
