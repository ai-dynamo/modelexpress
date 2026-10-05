# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL's socket metadata API requires numeric worker addresses."""

import socket
from unittest.mock import patch

import pytest

from modelexpress.nixl_transfer import NixlTransferManager


class RecordingAgent:
    def __init__(self):
        self.fetches = []

    def fetch_remote_metadata(self, name, address, port):
        self.fetches.append((name, address, port))

    def check_remote_metadata(self, name):
        return any(peer == name for peer, _, _ in self.fetches)


@pytest.fixture
def manager():
    result = NixlTransferManager(agent_name="target", device_id=0)
    result._agent = RecordingAgent()
    return result


@pytest.mark.parametrize(
    "family,address,sockaddr",
    [
        (socket.AF_INET, "192.0.2.10", ("192.0.2.10", 5555)),
        (socket.AF_INET6, "2001:db8::10", ("2001:db8::10", 5555, 0, 0)),
        (socket.AF_INET6, "fe80::10%7", ("fe80::10", 5555, 0, 7)),
        (socket.AF_INET6, "fe80::10%eth0", ("fe80::10%eth0", 5555, 0, 7)),
    ],
)
def test_fetch_resolves_worker_hostname(manager, family, address, sockaddr):
    answers = [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr)]
    with patch("socket.getaddrinfo", return_value=answers) as resolve:
        manager.fetch_remote_and_wait("source", "source.example.test", 5555)

    resolve.assert_called_once_with(
        "source.example.test", 5555, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
    )
    assert manager._agent.fetches == [("source", address, 5555)]
    assert manager._remote_agents == {"source": (address, 5555)}


@pytest.mark.parametrize(
    "host,address",
    [
        ("192.0.2.10", "192.0.2.10"),
        ("2001:db8::10", "2001:db8::10"),
        ("[2001:db8::10]", "2001:db8::10"),
        ("fe80::10%eth0", "fe80::10%eth0"),
        ("[fe80::10%3]", "fe80::10%3"),
    ],
)
def test_fetch_accepts_numeric_addresses_without_dns(manager, host, address):
    with patch("socket.getaddrinfo", side_effect=AssertionError("DNS must not be used")):
        manager.fetch_remote_and_wait("source", host, 5555)

    assert manager._agent.fetches == [("source", address, 5555)]
    assert manager._remote_agents == {"source": (address, 5555)}


@pytest.mark.parametrize("first_family", [socket.AF_INET, socket.AF_INET6])
def test_fetch_preserves_resolver_order(manager, first_family):
    ipv4 = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("192.0.2.10", 5555))
    ipv6 = (
        socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
        ("2001:db8::10", 5555, 0, 0),
    )
    answers = [ipv4, ipv6] if first_family == socket.AF_INET else [ipv6, ipv4]
    expected = "192.0.2.10" if first_family == socket.AF_INET else "2001:db8::10"
    with patch("socket.getaddrinfo", return_value=answers):
        manager.fetch_remote_and_wait("source", "source.example.test", 5555)

    assert manager._agent.fetches == [("source", expected, 5555)]


def test_failed_dns_lookup_fails_before_nixl_fetch(manager):
    with patch("socket.getaddrinfo", side_effect=socket.gaierror("Name not known")):
        with pytest.raises(RuntimeError, match="resolve NIXL metadata host.*peer.invalid"):
            manager.fetch_remote_and_wait("source", "peer.invalid", 5555)

    assert manager._agent.fetches == []
    assert manager._remote_agents == {}


def test_empty_dns_result_fails_before_nixl_fetch(manager):
    with patch("socket.getaddrinfo", return_value=[]):
        with pytest.raises(RuntimeError, match="IPv4 or IPv6.*peer.invalid"):
            manager.fetch_remote_and_wait("source", "peer.invalid", 5555)

    assert manager._agent.fetches == []
    assert manager._remote_agents == {}
