# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PeerDirectMetadataClient.

The static substrate is used throughout so nothing on the host's multicast
network is touched. ``MX_PEER_ENDPOINTS`` is cleared in a fixture to
guarantee no peer endpoints leak in from the ambient environment.
"""

from __future__ import annotations

import pytest

from modelexpress import p2p_pb2
from modelexpress.peer_direct_client import (
    PeerDirectMetadataClient,
    _FakeGetMetadataResponse,
    _FakeListSourcesResponse,
    _compute_mx_source_id,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip substrate-related env vars so tests are hermetic."""
    for var in (
        "MX_PEER_DISCOVERY_SUBSTRATE",
        "MX_PEER_ENDPOINTS",
        "MX_PEER_IP",
        "MX_WORKER_GRPC_PORT",
        "MX_WORKER_DEVICE_ID",
    ):
        monkeypatch.delenv(var, raising=False)


class TestPublishMetadata:
    def test_publish_metadata_computes_source_id(self):
        identity = p2p_pb2.SourceIdentity(
            model_name="test-publish-model",
            backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
            tensor_parallel_size=1,
        )
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=0,
            status=p2p_pb2.SOURCE_STATUS_READY,
            metadata_endpoint="127.0.0.1:5555",
            agent_name="test-agent",
        )

        # port=0 lets the kernel pick a free port so parallel test runs
        # don't collide on the default 6555.
        client = PeerDirectMetadataClient(substrate="static", peers="", port=0)
        try:
            source_id = client.publish_metadata(identity, worker, "worker-0")
        finally:
            client.close()

        expected = _compute_mx_source_id(identity)
        assert source_id == expected
        assert len(source_id) == 16


class TestListSources:
    def test_list_sources_empty_when_no_peers(self):
        client = PeerDirectMetadataClient(substrate="static", peers="")
        try:
            response = client.list_sources()
        finally:
            client.close()

        assert isinstance(response, _FakeListSourcesResponse)
        assert response.instances == []

    def test_list_sources_respects_identity_filter(self):
        """With no peers, identity filter should still return empty
        rather than raising on the filter branch."""
        client = PeerDirectMetadataClient(substrate="static", peers="")
        try:
            identity = p2p_pb2.SourceIdentity(model_name="filter-target")
            response = client.list_sources(identity=identity)
        finally:
            client.close()
        assert response.instances == []


class TestGetMetadata:
    def test_get_metadata_found_false_when_source_unknown(self):
        client = PeerDirectMetadataClient(substrate="static", peers="")
        try:
            response = client.get_metadata("0123456789abcdef", "worker-0")
        finally:
            client.close()
        assert isinstance(response, _FakeGetMetadataResponse)
        assert response.found is False


class TestUpdateStatus:
    def test_update_status_noop_for_unknown_source(self):
        """Peer-direct mode returns True for non-owned sources: there's
        no central store to fail against."""
        client = PeerDirectMetadataClient(substrate="static", peers="")
        try:
            result = client.update_status(
                "0123456789abcdef",
                "worker-0",
                0,
                p2p_pb2.SOURCE_STATUS_READY,
            )
        finally:
            client.close()
        assert result is True


class TestClose:
    def test_close_is_idempotent(self):
        client = PeerDirectMetadataClient(substrate="static", peers="")
        client.close()
        # A second close must not raise.
        client.close()

    def test_close_without_start_is_idempotent(self):
        """Close on a never-used client (no _ensure_started() call) must
        not raise."""
        client = PeerDirectMetadataClient(substrate="static", peers="")
        client.close()

    def test_use_after_close_raises(self):
        client = PeerDirectMetadataClient(substrate="static", peers="")
        client.close()
        with pytest.raises(RuntimeError, match="closed"):
            client.publish_metadata(
                p2p_pb2.SourceIdentity(model_name="m"),
                p2p_pb2.WorkerMetadata(),
                "w0",
            )


class TestConstructor:
    def test_rejects_unknown_substrate(self):
        with pytest.raises(ValueError, match="unsupported substrate"):
            PeerDirectMetadataClient(substrate="carrier-pigeon")

    def test_substrate_defaults_to_mdns_when_env_unset(self):
        client = PeerDirectMetadataClient()
        try:
            assert client._substrate == "mdns"
        finally:
            client.close()

    def test_substrate_reads_env(self, monkeypatch):
        monkeypatch.setenv("MX_PEER_DISCOVERY_SUBSTRATE", "static")
        client = PeerDirectMetadataClient()
        try:
            assert client._substrate == "static"
        finally:
            client.close()
