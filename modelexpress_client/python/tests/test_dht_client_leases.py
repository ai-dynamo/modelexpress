# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the K8s-Lease self-organizing bootstrap path in MxDhtClient.

Everything external is mocked: no real apiserver, no real network, no real
DhtNode. The tests exercise env parsing, the ``_advertise_host`` helper, and
the anchor/worker branches of ``_bootstrap_via_leases`` (plus its
advertise-host guard). The real ``kademlite.multiaddr`` helpers run unmocked -
they are pure and work fine on synthetic peer_id bytes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modelexpress.dht_client import MxDhtClient


# ---------------------------------------------------------------------------
# Env parsing
# ---------------------------------------------------------------------------


def test_bootstrap_leases_from_env(monkeypatch):
    monkeypatch.setenv("MX_DHT_BOOTSTRAP_LEASES", "mx-dht-anchor")
    monkeypatch.delenv("MX_DHT_LEASE_NAMESPACE", raising=False)
    client = MxDhtClient()
    assert client._bootstrap_leases == "mx-dht-anchor"
    assert client._lease_namespace is None


def test_lease_namespace_from_env(monkeypatch):
    monkeypatch.delenv("MX_DHT_BOOTSTRAP_LEASES", raising=False)
    monkeypatch.setenv("MX_DHT_LEASE_NAMESPACE", "foo")
    client = MxDhtClient()
    assert client._lease_namespace == "foo"
    assert client._bootstrap_leases is None


def test_lease_env_unset_yields_none(monkeypatch):
    monkeypatch.delenv("MX_DHT_BOOTSTRAP_LEASES", raising=False)
    monkeypatch.delenv("MX_DHT_LEASE_NAMESPACE", raising=False)
    client = MxDhtClient()
    assert client._bootstrap_leases is None
    assert client._lease_namespace is None


def test_lease_kwargs_override_env(monkeypatch):
    monkeypatch.setenv("MX_DHT_BOOTSTRAP_LEASES", "env-prefix")
    monkeypatch.setenv("MX_DHT_LEASE_NAMESPACE", "env-ns")
    client = MxDhtClient(
        bootstrap_leases="kwarg-prefix",
        lease_namespace="kwarg-ns",
    )
    assert client._bootstrap_leases == "kwarg-prefix"
    assert client._lease_namespace == "kwarg-ns"


# ---------------------------------------------------------------------------
# _advertise_host
# ---------------------------------------------------------------------------


def test_advertise_host_prefers_pod_ip(monkeypatch):
    monkeypatch.setenv("POD_IP", "10.0.0.5")
    client = MxDhtClient()
    assert client._advertise_host("192.168.1.2") == "10.0.0.5"


def test_advertise_host_falls_back_to_listen_host(monkeypatch):
    monkeypatch.delenv("POD_IP", raising=False)
    client = MxDhtClient()
    assert client._advertise_host("192.168.1.2") == "192.168.1.2"


def test_advertise_host_none_for_wildcard(monkeypatch):
    monkeypatch.delenv("POD_IP", raising=False)
    client = MxDhtClient()
    assert client._advertise_host("0.0.0.0") is None
    assert client._advertise_host("::") is None
    assert client._advertise_host("") is None


# ---------------------------------------------------------------------------
# _bootstrap_via_leases
# ---------------------------------------------------------------------------


def _make_node() -> MagicMock:
    """Fake DhtNode with just the surface _bootstrap_via_leases touches."""
    node = MagicMock()
    node.peer_id = b"\x00" * 31 + b"\x05"
    node.listen_addr = ("0.0.0.0", 4001)
    node.bootstrap = AsyncMock()
    return node


def _make_coordinator_instance(won: bool, converged: bool = True) -> MagicMock:
    """Fake LeaseCoordinator instance with async methods stubbed."""
    inst = MagicMock()
    inst.slot_for = MagicMock(return_value=5)
    inst.claim = AsyncMock(return_value=won)
    inst.renew = AsyncMock(return_value=True)
    inst.anchor_multiaddrs = AsyncMock(
        return_value=["/ip4/10.0.0.9/tcp/4001/p2p/anchorpeer"]
    )
    inst.wait_all_converged = AsyncMock(return_value=converged)
    return inst


@pytest.mark.asyncio
async def test_bootstrap_via_leases_anchor_path(monkeypatch):
    monkeypatch.setenv("POD_IP", "10.0.0.5")
    client = MxDhtClient(bootstrap_leases="mx-dht-anchor")
    client._node = _make_node()

    coordinator = _make_coordinator_instance(won=True)
    coordinator_cls = MagicMock(return_value=coordinator)

    with patch("kademlite.k8s_lease.LeaseCoordinator", coordinator_cls):
        await client._bootstrap_via_leases()

    try:
        coordinator.claim.assert_awaited()
        # Winning anchor flags itself converged on the initial renew.
        coordinator.renew.assert_awaited()
        _, kwargs = coordinator.renew.await_args
        assert kwargs.get("converged") is True
        # Anchor set was dialed via the node's bootstrap.
        client._node.bootstrap.assert_awaited()
        # A renew loop task was scheduled.
        assert client._lease_task is not None
    finally:
        if client._lease_task is not None:
            client._lease_task.cancel()


@pytest.mark.asyncio
async def test_bootstrap_via_leases_worker_path(monkeypatch):
    monkeypatch.setenv("POD_IP", "10.0.0.5")
    client = MxDhtClient(bootstrap_leases="mx-dht-anchor")
    client._node = _make_node()

    coordinator = _make_coordinator_instance(won=False, converged=True)
    coordinator_cls = MagicMock(return_value=coordinator)

    with patch("kademlite.k8s_lease.LeaseCoordinator", coordinator_cls):
        await client._bootstrap_via_leases()

    coordinator.claim.assert_awaited()
    coordinator.wait_all_converged.assert_awaited()
    # Worker does not renew (it never held the lease).
    coordinator.renew.assert_not_awaited()
    # No anchor renew loop for a worker.
    assert client._lease_task is None
    # But it still bootstraps against the available anchors.
    client._node.bootstrap.assert_awaited()


@pytest.mark.asyncio
async def test_bootstrap_via_leases_skips_without_advertise_host(monkeypatch):
    monkeypatch.delenv("POD_IP", raising=False)
    client = MxDhtClient(bootstrap_leases="mx-dht-anchor")
    node = _make_node()
    node.listen_addr = ("0.0.0.0", 4001)  # wildcard -> no advertise host
    client._node = node

    coordinator_cls = MagicMock()

    with patch("kademlite.k8s_lease.LeaseCoordinator", coordinator_cls):
        await client._bootstrap_via_leases()

    # No advertisable address -> return early, never construct a coordinator
    # and never touch the node's bootstrap.
    coordinator_cls.assert_not_called()
    client._node.bootstrap.assert_not_awaited()
    assert client._lease_task is None
