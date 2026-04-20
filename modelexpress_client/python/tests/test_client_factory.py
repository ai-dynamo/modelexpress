# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for modelexpress.client_factory.create_metadata_client()."""

from unittest.mock import patch

from modelexpress.client import MxClient
from modelexpress.client_factory import create_metadata_client
from modelexpress.peer_direct_client import PeerDirectMetadataClient


def test_peer_direct_backend_returns_peer_direct_client(monkeypatch):
    monkeypatch.setenv("MX_METADATA_BACKEND", "peer-direct")
    monkeypatch.setenv("MX_PEER_DISCOVERY_SUBSTRATE", "static")
    client = create_metadata_client()
    try:
        assert isinstance(client, PeerDirectMetadataClient)
    finally:
        client.close()


def test_peer_direct_substrate_defaults_to_mdns(monkeypatch):
    monkeypatch.setenv("MX_METADATA_BACKEND", "peer-direct")
    monkeypatch.delenv("MX_PEER_DISCOVERY_SUBSTRATE", raising=False)
    # We check by verifying the resulting object is a PeerDirectMetadataClient;
    # substrate-level assertions live in test_peer_direct_client.py.
    with patch(
        "modelexpress.client_factory.PeerDirectMetadataClient"
    ) as MockClient:
        create_metadata_client()
    MockClient.assert_called_once_with(substrate="mdns")


def test_unset_backend_returns_mxclient(monkeypatch):
    monkeypatch.delenv("MX_METADATA_BACKEND", raising=False)
    with patch("modelexpress.client_factory.MxClient") as MockMx:
        create_metadata_client()
    MockMx.assert_called_once_with()


def test_legacy_redis_backend_returns_mxclient(monkeypatch):
    monkeypatch.setenv("MX_METADATA_BACKEND", "redis")
    with patch("modelexpress.client_factory.MxClient") as MockMx:
        create_metadata_client()
    MockMx.assert_called_once_with()


def test_legacy_kubernetes_backend_returns_mxclient(monkeypatch):
    monkeypatch.setenv("MX_METADATA_BACKEND", "kubernetes")
    with patch("modelexpress.client_factory.MxClient") as MockMx:
        create_metadata_client()
    MockMx.assert_called_once_with()


def test_unknown_backend_falls_back_to_mxclient_with_warning(monkeypatch, caplog):
    import logging

    monkeypatch.setenv("MX_METADATA_BACKEND", "flux-capacitor")
    caplog.set_level(logging.WARNING, logger="modelexpress.client_factory")
    with patch("modelexpress.client_factory.MxClient") as MockMx:
        create_metadata_client()
    MockMx.assert_called_once_with()
    assert any(
        "Unknown MX_METADATA_BACKEND" in record.message
        for record in caplog.records
    )


def test_backend_value_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("MX_METADATA_BACKEND", "PEER-DIRECT")
    monkeypatch.setenv("MX_PEER_DISCOVERY_SUBSTRATE", "static")
    client = create_metadata_client()
    try:
        assert isinstance(client, PeerDirectMetadataClient)
    finally:
        client.close()


def test_backend_value_strips_whitespace(monkeypatch):
    monkeypatch.setenv("MX_METADATA_BACKEND", "  peer-direct  ")
    monkeypatch.setenv("MX_PEER_DISCOVERY_SUBSTRATE", "static")
    client = create_metadata_client()
    try:
        assert isinstance(client, PeerDirectMetadataClient)
    finally:
        client.close()
