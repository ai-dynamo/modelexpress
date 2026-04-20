# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for mx_peer_discovery.dns."""

import asyncio
import socket
from unittest.mock import AsyncMock, patch

from mx_peer_discovery.dns import (
    filter_own_ips,
    resolve_hostlist,
    resolve_hostname,
)


def _mk_getaddrinfo_result(ip: str, port: int) -> tuple:
    """Build a single getaddrinfo tuple: (family, type, proto, canonname, sockaddr)."""
    return (socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, port))


def _patch_loop_getaddrinfo(*, return_value=None, side_effect=None):
    """Patch getaddrinfo on the currently running event loop.

    The abstract-class method isn't used at runtime: the concrete loop
    class (SelectorEventLoop, ProactorEventLoop, uvloop.Loop, ...) has its
    own implementation. Patching the loop instance directly is the most
    portable way to intercept the call.
    """
    loop = asyncio.get_event_loop()
    kwargs: dict = {"new_callable": AsyncMock}
    if return_value is not None:
        kwargs["return_value"] = return_value
    if side_effect is not None:
        kwargs["side_effect"] = side_effect
    return patch.object(loop, "getaddrinfo", **kwargs)


# -- resolve_hostname ---------------------------------------------------------


async def test_resolve_hostname_returns_unique_ordered_ips():
    with _patch_loop_getaddrinfo(
        return_value=[
            _mk_getaddrinfo_result("10.0.0.1", 4001),
            _mk_getaddrinfo_result("10.0.0.2", 4001),
            _mk_getaddrinfo_result("10.0.0.1", 4001),  # dup
            _mk_getaddrinfo_result("10.0.0.3", 4001),
        ]
    ):
        ips = await resolve_hostname("svc.cluster.local", 4001)
    assert ips == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]


async def test_resolve_hostname_swallows_gaierror():
    with _patch_loop_getaddrinfo(
        side_effect=socket.gaierror(-2, "Name does not resolve"),
    ):
        ips = await resolve_hostname("nonexistent.invalid", 4001)
    assert ips == []


async def test_resolve_hostname_empty_result():
    with _patch_loop_getaddrinfo(return_value=[]):
        ips = await resolve_hostname("svc.cluster.local", 4001)
    assert ips == []


# -- resolve_hostlist ---------------------------------------------------------


async def test_resolve_hostlist_expands_and_resolves():
    call_map = {
        "gpu01": [_mk_getaddrinfo_result("10.0.0.1", 4001)],
        "gpu02": [_mk_getaddrinfo_result("10.0.0.2", 4001)],
        "gpu03": [_mk_getaddrinfo_result("10.0.0.3", 4001)],
    }

    async def fake_getaddrinfo(host, port, **kwargs):
        return call_map[host]

    with _patch_loop_getaddrinfo(side_effect=fake_getaddrinfo):
        ips = await resolve_hostlist("gpu[01-03]", 4001)
    assert ips == ["10.0.0.1", "10.0.0.2", "10.0.0.3"]


async def test_resolve_hostlist_deduplicates_across_hosts():
    async def fake_getaddrinfo(host, port, **kwargs):
        # All hosts resolve to the same two IPs
        return [
            _mk_getaddrinfo_result("10.0.0.1", port),
            _mk_getaddrinfo_result("10.0.0.2", port),
        ]

    with _patch_loop_getaddrinfo(side_effect=fake_getaddrinfo):
        ips = await resolve_hostlist("gpu[01-03]", 4001)
    assert ips == ["10.0.0.1", "10.0.0.2"]


async def test_resolve_hostlist_skips_unresolvable_hosts():
    async def fake_getaddrinfo(host, port, **kwargs):
        if host == "gpu2":
            raise socket.gaierror(-2, "Name does not resolve")
        return [_mk_getaddrinfo_result(f"10.0.0.{host[-1]}", port)]

    with _patch_loop_getaddrinfo(side_effect=fake_getaddrinfo):
        ips = await resolve_hostlist("gpu[1-3]", 4001)
    assert ips == ["10.0.0.1", "10.0.0.3"]


async def test_resolve_hostlist_empty_input():
    ips = await resolve_hostlist("", 4001)
    assert ips == []


async def test_resolve_hostlist_all_fail():
    async def fake_getaddrinfo(host, port, **kwargs):
        raise socket.gaierror(-2, "Name does not resolve")

    with _patch_loop_getaddrinfo(side_effect=fake_getaddrinfo):
        ips = await resolve_hostlist("gpu[1-3]", 4001)
    assert ips == []


# -- filter_own_ips -----------------------------------------------------------


def test_filter_own_ips_removes_matching():
    ips = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    own = {"10.0.0.2"}
    assert filter_own_ips(ips, own) == ["10.0.0.1", "10.0.0.3"]


def test_filter_own_ips_preserves_order():
    ips = ["10.0.0.3", "10.0.0.1", "10.0.0.2"]
    own = {"10.0.0.1"}
    assert filter_own_ips(ips, own) == ["10.0.0.3", "10.0.0.2"]


def test_filter_own_ips_empty_own_set():
    ips = ["10.0.0.1", "10.0.0.2"]
    assert filter_own_ips(ips, set()) == ips


def test_filter_own_ips_empty_list():
    assert filter_own_ips([], {"10.0.0.1"}) == []


def test_filter_own_ips_all_self():
    ips = ["10.0.0.1", "10.0.0.2"]
    own = {"10.0.0.1", "10.0.0.2"}
    assert filter_own_ips(ips, own) == []
