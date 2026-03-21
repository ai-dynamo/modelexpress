# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DhtMetadataClient and DHT backend selection in the loader."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from modelexpress import p2p_pb2
from modelexpress.dht_client import (
    DhtMetadataClient,
    _FakeGetMetadataResponse,
    _FakeListSourcesResponse,
    _FakeSourceInstanceRef,
    _compute_mx_source_id,
    _json_to_worker_metadata,
    _worker_to_json,
    _worker_key,
    _worker_directory_key,
    _instances_key,
    _attrs_key,
    _sources_key,
)


# ---------------------------------------------------------------------------
# Source ID computation
# ---------------------------------------------------------------------------


class TestComputeSourceId:
    def test_deterministic(self):
        identity = p2p_pb2.SourceIdentity(model_name="test-model")
        id1 = _compute_mx_source_id(identity)
        id2 = _compute_mx_source_id(identity)
        assert id1 == id2
        assert len(id1) == 16

    def test_case_insensitive(self):
        id1 = _compute_mx_source_id(p2p_pb2.SourceIdentity(model_name="My-Model"))
        id2 = _compute_mx_source_id(p2p_pb2.SourceIdentity(model_name="my-model"))
        assert id1 == id2

    def test_different_models_different_ids(self):
        id1 = _compute_mx_source_id(p2p_pb2.SourceIdentity(model_name="model-a"))
        id2 = _compute_mx_source_id(p2p_pb2.SourceIdentity(model_name="model-b"))
        assert id1 != id2


# ---------------------------------------------------------------------------
# Key format helpers
# ---------------------------------------------------------------------------


class TestKeyFormat:
    def test_worker_key(self):
        assert _worker_key("abc123", "wid1", 0) == b"/mx/abc123/wid1/0"
        assert _worker_key("abc123", "wid1", 3) == b"/mx/abc123/wid1/3"

    def test_worker_directory_key(self):
        assert _worker_directory_key("abc123", "wid1") == b"/mx/abc123/wid1/workers"

    def test_instances_key(self):
        assert _instances_key("abc123") == b"/mx/abc123/instances"

    def test_attrs_key(self):
        assert _attrs_key("abc123") == b"/mx/abc123/attrs"

    def test_sources_key(self):
        assert _sources_key() == b"/mx/_sources"


# ---------------------------------------------------------------------------
# JSON serialization round-trips
# ---------------------------------------------------------------------------


class TestJsonSerialization:
    def test_worker_to_json_nixl(self):
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=0,
            metadata_endpoint="10.0.0.1:5555",
            agent_name="mx-auto-worker0-abc",
            worker_grpc_endpoint="10.0.0.1:5556",
            status=p2p_pb2.SOURCE_STATUS_READY,
            updated_at=1234567890,
        )
        result = _worker_to_json(worker)
        assert result["worker_rank"] == 0
        assert result["backend_type"] == "nixl"
        assert result["metadata_endpoint"] == "10.0.0.1:5555"
        assert result["agent_name"] == "mx-auto-worker0-abc"
        assert result["worker_grpc_endpoint"] == "10.0.0.1:5556"
        assert "tensors" not in result
        assert result["status"] == p2p_pb2.SOURCE_STATUS_READY

    def test_worker_to_json_transfer_engine(self):
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=1,
            transfer_engine_session_id="10.0.0.1:12345",
            status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
        )
        result = _worker_to_json(worker)
        assert result["backend_type"] == "transfer_engine"
        assert result["transfer_engine_session_id"] == "10.0.0.1:12345"

    def test_worker_to_json_none_backend(self):
        worker = p2p_pb2.WorkerMetadata(worker_rank=0)
        result = _worker_to_json(worker)
        assert result["backend_type"] == "none"

    def test_json_roundtrip(self):
        """worker -> JSON -> bytes -> JSON -> WorkerMetadata preserves all fields."""
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=2,
            metadata_endpoint="host:5557",
            agent_name="agent-2",
            worker_grpc_endpoint="host:5558",
            status=p2p_pb2.SOURCE_STATUS_READY,
            updated_at=9999,
        )
        json_dict = _worker_to_json(worker)
        encoded = json.dumps(json_dict).encode()
        decoded = json.loads(encoded)
        back = _json_to_worker_metadata(decoded)

        assert back.worker_rank == 2
        assert back.metadata_endpoint == "host:5557"
        assert back.agent_name == "agent-2"
        assert back.worker_grpc_endpoint == "host:5558"
        assert back.status == p2p_pb2.SOURCE_STATUS_READY
        assert back.updated_at == 9999

    def test_json_to_worker_metadata_missing_fields(self):
        """Partial JSON (missing optional fields) still deserializes."""
        record = {"worker_rank": 0}
        result = _json_to_worker_metadata(record)
        assert result.worker_rank == 0
        assert result.metadata_endpoint == ""
        assert result.agent_name == ""
        assert result.worker_grpc_endpoint == ""
        assert result.status == 0

    def test_json_schema_has_no_tensors(self):
        """DHT JSON must not contain tensor data (served via WorkerService gRPC)."""
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=0,
            metadata_endpoint="10.0.0.1:5555",
            agent_name="agent-0",
            worker_grpc_endpoint="10.0.0.1:5556",
            status=p2p_pb2.SOURCE_STATUS_READY,
        )
        result = _worker_to_json(worker)
        assert "tensors" not in result, "tensor manifests must not be in DHT records"

    def test_json_schema_has_worker_grpc_endpoint(self):
        """DHT JSON must include worker_grpc_endpoint for tensor manifest fetching."""
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=0,
            metadata_endpoint="10.0.0.1:5555",
            worker_grpc_endpoint="10.0.0.1:5556",
        )
        result = _worker_to_json(worker)
        assert "worker_grpc_endpoint" in result
        assert result["worker_grpc_endpoint"] == "10.0.0.1:5556"

    def test_json_record_size_under_dht_limit(self):
        """DHT records must stay well under the 16KB Kademlia message limit."""
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=0,
            metadata_endpoint="10.0.0.1:5555",
            agent_name="mx-auto-worker0-" + "a" * 64,
            worker_grpc_endpoint="10.0.0.1:5556",
            transfer_engine_session_id="10.0.0.1:12345",
            status=p2p_pb2.SOURCE_STATUS_READY,
            updated_at=1234567890000,
        )
        record = _worker_to_json(worker)
        encoded = json.dumps(record).encode()
        assert len(encoded) < 1024, (
            f"DHT worker record is {len(encoded)} bytes, expected <1KB "
            f"(tensor manifests must not be embedded)"
        )

    def test_worker_to_json_matches_proto_schema(self):
        """All WorkerMetadata proto fields (except reserved) appear in JSON."""
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=3,
            metadata_endpoint="host:5555",
            agent_name="agent-3",
            worker_grpc_endpoint="host:5556",
            transfer_engine_session_id="host:12345",
            status=p2p_pb2.SOURCE_STATUS_READY,
            updated_at=42,
        )
        result = _worker_to_json(worker)
        expected_keys = {
            "worker_rank", "backend_type", "metadata_endpoint",
            "agent_name", "worker_grpc_endpoint",
            "transfer_engine_session_id", "status", "updated_at",
        }
        assert set(result.keys()) == expected_keys, (
            f"JSON keys {set(result.keys())} != expected {expected_keys}"
        )


# ---------------------------------------------------------------------------
# Duck-typed response objects
# ---------------------------------------------------------------------------


class TestFakeResponses:
    def test_get_metadata_found(self):
        worker = p2p_pb2.WorkerMetadata(worker_rank=0)
        resp = _FakeGetMetadataResponse(
            found=True, worker=worker, mx_source_id="abc", worker_id="wid1",
        )
        assert resp.found is True
        assert resp.worker.worker_rank == 0
        assert resp.mx_source_id == "abc"

    def test_get_metadata_not_found(self):
        resp = _FakeGetMetadataResponse(found=False)
        assert resp.found is False
        assert resp.worker is None

    def test_list_sources_response(self):
        ref = _FakeSourceInstanceRef("sid", "wid", "model", 0)
        resp = _FakeListSourcesResponse(instances=[ref])
        assert len(resp.instances) == 1
        assert resp.instances[0].mx_source_id == "sid"
        assert resp.instances[0].worker_rank == 0


# ---------------------------------------------------------------------------
# DhtMetadataClient with mocked DhtNode
# ---------------------------------------------------------------------------


def _make_mock_node():
    """Create a mock DhtNode with async put/get/stop."""
    node = AsyncMock()
    node.put = AsyncMock(return_value=1)
    node.get = AsyncMock(return_value=None)
    node.stop = AsyncMock()
    node.start = AsyncMock()
    return node


def _make_client(mock_node):
    """Create a DhtMetadataClient with a pre-injected mock node."""
    import asyncio
    import threading

    client = DhtMetadataClient.__new__(DhtMetadataClient)
    client._record_ttl = 300
    client._lock = threading.Lock()

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    client._loop = loop
    client._thread = thread
    client._node = mock_node
    client._started = True
    client._listen_addr = "127.0.0.1:4001"
    client._bootstrap_peers_str = ""
    client._bootstrap_dns = None
    return client


class TestPublishMetadata:
    def test_publish_returns_source_id(self):
        node = _make_mock_node()
        client = _make_client(node)

        identity = p2p_pb2.SourceIdentity(model_name="test-model")
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=0,
            metadata_endpoint="10.0.0.1:5555",
            agent_name="agent-0",
            status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
        )
        source_id = client.publish_metadata(identity, worker, "wid1")
        assert isinstance(source_id, str)
        assert len(source_id) == 16

        # Should have put: worker record, worker directory, instances, attrs, sources = 5 puts
        assert node.put.call_count == 5
        client.close()

    def test_publish_consistent_source_id(self):
        node = _make_mock_node()
        client = _make_client(node)

        identity = p2p_pb2.SourceIdentity(model_name="test-model")
        worker = p2p_pb2.WorkerMetadata(worker_rank=0)
        id1 = client.publish_metadata(identity, worker, "wid1")
        id2 = client.publish_metadata(identity, worker, "wid2")
        assert id1 == id2  # same identity -> same source_id
        client.close()


class TestGetMetadata:
    def test_get_not_found(self):
        node = _make_mock_node()
        node.get = AsyncMock(return_value=None)
        client = _make_client(node)

        resp = client.get_metadata("abc123", "wid1")
        assert resp.found is False
        client.close()

    def test_get_with_worker(self):
        node = _make_mock_node()
        source_id = "abc123"
        worker_id = "wid1"

        directory = json.dumps({"ranks": [0], "updated_at": 1000}).encode()
        w0 = json.dumps({
            "worker_rank": 0, "metadata_endpoint": "10.0.0.1:5555",
            "agent_name": "agent-0", "tensors": [], "status": 2, "updated_at": 1000,
        }).encode()

        async def mock_get(key):
            if key == _worker_directory_key(source_id, worker_id):
                return directory
            if key == _worker_key(source_id, worker_id, 0):
                return w0
            return None

        node.get = AsyncMock(side_effect=mock_get)
        client = _make_client(node)

        resp = client.get_metadata(source_id, worker_id)
        assert resp.found is True
        assert resp.worker.worker_rank == 0
        assert resp.worker.metadata_endpoint == "10.0.0.1:5555"
        assert resp.mx_source_id == source_id
        assert resp.worker_id == worker_id
        client.close()


class TestListSources:
    def test_list_empty(self):
        node = _make_mock_node()
        node.get = AsyncMock(return_value=None)
        client = _make_client(node)

        resp = client.list_sources()
        assert resp.instances == []
        client.close()

    def test_list_with_filter(self):
        node = _make_mock_node()
        source_id = "abc123"
        identity = p2p_pb2.SourceIdentity(model_name="test-model")

        sources = json.dumps({"source_ids": [source_id], "updated_at": 1000}).encode()
        attrs = json.dumps({"model_name": "test-model", "updated_at": 1000}).encode()
        instances = json.dumps({"worker_ids": ["wid1"], "updated_at": 1000}).encode()
        directory = json.dumps({"ranks": [0], "updated_at": 1000}).encode()
        w0_ready = json.dumps({
            "worker_rank": 0, "status": p2p_pb2.SOURCE_STATUS_READY,
            "tensors": [], "updated_at": 1000,
        }).encode()

        computed_sid = _compute_mx_source_id(identity)

        async def mock_get(key):
            if key == _sources_key():
                return json.dumps({"source_ids": [computed_sid], "updated_at": 1000}).encode()
            if key == _attrs_key(computed_sid):
                return attrs
            if key == _instances_key(computed_sid):
                return instances
            if key == _worker_directory_key(computed_sid, "wid1"):
                return directory
            if key == _worker_key(computed_sid, "wid1", 0):
                return w0_ready
            return None

        node.get = AsyncMock(side_effect=mock_get)
        client = _make_client(node)

        resp = client.list_sources(
            identity=identity,
            status_filter=p2p_pb2.SOURCE_STATUS_READY,
        )
        assert len(resp.instances) == 1
        assert resp.instances[0].worker_rank == 0
        assert resp.instances[0].model_name == "test-model"
        client.close()


class TestUpdateStatus:
    def test_update_status_success(self):
        node = _make_mock_node()
        existing = json.dumps({
            "worker_rank": 0, "metadata_endpoint": "10.0.0.1:5555",
            "agent_name": "agent-0", "tensors": [], "status": 1, "updated_at": 1000,
        }).encode()
        node.get = AsyncMock(return_value=existing)
        client = _make_client(node)

        result = client.update_status("sid", "wid1", 0, p2p_pb2.SOURCE_STATUS_READY)
        assert result is True

        put_call = node.put.call_args_list[0]
        updated = json.loads(put_call[0][1])
        assert updated["status"] == p2p_pb2.SOURCE_STATUS_READY
        assert updated["updated_at"] > 1000
        client.close()

    def test_update_status_not_found(self):
        node = _make_mock_node()
        node.get = AsyncMock(return_value=None)
        client = _make_client(node)

        result = client.update_status("sid", "wid1", 99, p2p_pb2.SOURCE_STATUS_READY)
        assert result is False
        client.close()


# ---------------------------------------------------------------------------
# Loader backend selection
# ---------------------------------------------------------------------------


class TestLoaderBackendSelection:
    """Test that _create_metadata_client picks the right client."""

    def test_default_is_mx_client(self):
        from modelexpress.vllm_loader import _create_metadata_client
        from modelexpress.client import MxClient

        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("MX_METADATA_BACKEND", None)
            client = _create_metadata_client()
            assert isinstance(client, MxClient)

    def test_dht_backend_selection(self):
        from modelexpress.vllm_loader import _create_metadata_client

        with patch.dict("os.environ", {"MX_METADATA_BACKEND": "dht"}):
            with patch("modelexpress.dht_client._get_dht_node_class"):
                client = _create_metadata_client()
                assert isinstance(client, DhtMetadataClient)

    def test_redis_backend_uses_mx_client(self):
        from modelexpress.vllm_loader import _create_metadata_client
        from modelexpress.client import MxClient

        with patch.dict("os.environ", {"MX_METADATA_BACKEND": "redis"}):
            client = _create_metadata_client()
            assert isinstance(client, MxClient)
