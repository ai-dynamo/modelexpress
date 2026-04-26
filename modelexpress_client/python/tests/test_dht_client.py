# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MxDhtClient and DHT factory dispatch."""

from __future__ import annotations

import asyncio
import json

import pytest

from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from modelexpress.client_factory import create_metadata_client
from modelexpress.dht_client import (
    MxDhtClient,
    _decode_worker_pointer,
    _encode_worker_pointer,
    _key_for,
)
from modelexpress.metadata import _is_p2p_metadata_enabled
from modelexpress.source_id import compute_mx_source_id


def _base_identity() -> p2p_pb2.SourceIdentity:
    return p2p_pb2.SourceIdentity(
        mx_version="0.3.0",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name="deepseek-ai/DeepSeek-V3",
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        expert_parallel_size=0,
        dtype="bfloat16",
        quantization="",
        revision="abc123",
    )


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["dht", "kademlia", "DHT", "Kademlia"])
def test_factory_dht_aliases_return_dht_client(monkeypatch, value):
    monkeypatch.setenv("MX_METADATA_BACKEND", value)
    client = create_metadata_client(worker_rank=2)
    try:
        assert isinstance(client, MxDhtClient)
        assert client._worker_rank == 2
    finally:
        client.close()


def test_factory_default_is_not_dht(monkeypatch):
    monkeypatch.delenv("MX_METADATA_BACKEND", raising=False)
    assert isinstance(create_metadata_client(), MxClient)


def test_factory_unknown_backend_lists_dht_aliases(monkeypatch):
    monkeypatch.setenv("MX_METADATA_BACKEND", "bogus")
    with pytest.raises(ValueError, match="Unknown MX_METADATA_BACKEND") as exc_info:
        create_metadata_client()
    msg = str(exc_info.value)
    assert "dht" in msg
    assert "kademlia" in msg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_key_for_format():
    assert _key_for("abcdef0123456789", 3) == b"/mx/abcdef0123456789/rank/3"


def test_encode_decode_pointer_roundtrip():
    worker = p2p_pb2.WorkerMetadata(
        worker_rank=4,
        worker_grpc_endpoint="10.0.0.7:6559",
        metadata_endpoint="10.0.0.7:7000",
        agent_name="agent-rank-4",
    )
    raw = _encode_worker_pointer(worker, "worker-xyz")
    decoded = _decode_worker_pointer(raw)
    assert decoded == {
        "worker_id": "worker-xyz",
        "worker_rank": 4,
        "worker_grpc_endpoint": "10.0.0.7:6559",
        "metadata_endpoint": "10.0.0.7:7000",
        "agent_name": "agent-rank-4",
    }


def test_decode_pointer_missing_required_fields():
    raw = json.dumps({"worker_id": "x"}).encode("utf-8")
    with pytest.raises(ValueError, match="missing required fields"):
        _decode_worker_pointer(raw)


def test_decode_pointer_invalid_json():
    with pytest.raises(json.JSONDecodeError):
        _decode_worker_pointer(b"not json")


def test_parse_listen_host_and_port():
    assert MxDhtClient._parse_listen("10.0.0.1:4001") == ("10.0.0.1", 4001)


def test_parse_listen_bare_host():
    assert MxDhtClient._parse_listen("0.0.0.0") == ("0.0.0.0", 0)


def test_parse_listen_empty_host_yields_wildcard():
    assert MxDhtClient._parse_listen(":4001") == ("0.0.0.0", 4001)


# ---------------------------------------------------------------------------
# Construction / configuration
# ---------------------------------------------------------------------------


def test_construct_defaults(monkeypatch):
    for var in (
        "MX_DHT_LISTEN", "MX_DHT_BOOTSTRAP_PEERS", "MX_DHT_BOOTSTRAP_DNS",
        "MX_DHT_BOOTSTRAP_SLURM", "SLURM_JOB_NODELIST",
        "MX_DHT_BOOTSTRAP_PORT", "MX_DHT_RECORD_TTL",
        "MX_DHT_GET_RETRIES", "MX_DHT_GET_BACKOFF_SECONDS",
    ):
        monkeypatch.delenv(var, raising=False)
    client = MxDhtClient()
    try:
        assert client._listen_addr == "0.0.0.0:0"
        assert client._bootstrap_peers == []
        assert client._bootstrap_dns is None
        assert client._bootstrap_slurm is None
        assert client._bootstrap_port == 4001
        assert client._record_ttl == 24 * 60 * 60
        assert client._max_retries == 5
        assert client._backoff_seconds == 0.5
        assert client._worker_rank is None
    finally:
        client.close()


def test_construct_from_env(monkeypatch):
    monkeypatch.setenv("MX_DHT_LISTEN", "0.0.0.0:4001")
    monkeypatch.setenv(
        "MX_DHT_BOOTSTRAP_PEERS",
        "/ip4/10.0.0.1/tcp/4001/p2p/Qm1, /ip4/10.0.0.2/tcp/4001/p2p/Qm2",
    )
    monkeypatch.setenv("MX_DHT_BOOTSTRAP_DNS", "mx-dht-headless")
    monkeypatch.setenv("MX_DHT_BOOTSTRAP_PORT", "4002")
    monkeypatch.setenv("MX_DHT_RECORD_TTL", "3600")
    monkeypatch.setenv("MX_DHT_GET_RETRIES", "8")
    monkeypatch.setenv("MX_DHT_GET_BACKOFF_SECONDS", "0.1")
    client = MxDhtClient()
    try:
        assert client._listen_addr == "0.0.0.0:4001"
        assert client._bootstrap_peers == [
            "/ip4/10.0.0.1/tcp/4001/p2p/Qm1",
            "/ip4/10.0.0.2/tcp/4001/p2p/Qm2",
        ]
        assert client._bootstrap_dns == "mx-dht-headless"
        assert client._bootstrap_port == 4002
        assert client._record_ttl == 3600
        assert client._max_retries == 8
        assert client._backoff_seconds == 0.1
    finally:
        client.close()


def test_slurm_explicit_env_takes_priority_over_auto(monkeypatch):
    monkeypatch.setenv("MX_DHT_BOOTSTRAP_SLURM", "node[01-04]")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "auto[10-12]")
    client = MxDhtClient()
    try:
        assert client._bootstrap_slurm == "node[01-04]"
    finally:
        client.close()


def test_slurm_falls_back_to_slurm_job_nodelist(monkeypatch):
    monkeypatch.delenv("MX_DHT_BOOTSTRAP_SLURM", raising=False)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "compute[001-064]")
    client = MxDhtClient()
    try:
        assert client._bootstrap_slurm == "compute[001-064]"
    finally:
        client.close()


def test_slurm_kwarg_overrides_both_env_vars(monkeypatch):
    monkeypatch.setenv("MX_DHT_BOOTSTRAP_SLURM", "env-explicit")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "env-auto")
    client = MxDhtClient(bootstrap_slurm="kwarg-wins[01-02]")
    try:
        assert client._bootstrap_slurm == "kwarg-wins[01-02]"
    finally:
        client.close()


def test_construct_explicit_args_override_env(monkeypatch):
    monkeypatch.setenv("MX_DHT_LISTEN", "1.1.1.1:9999")
    monkeypatch.setenv("MX_DHT_BOOTSTRAP_PEERS", "/ip4/1.2.3.4/tcp/4001/p2p/Qm1")
    client = MxDhtClient(
        worker_rank=7,
        listen_addr="2.2.2.2:8888",
        bootstrap_peers=["/ip4/5.6.7.8/tcp/4001/p2p/Qm2"],
    )
    try:
        assert client._worker_rank == 7
        assert client._listen_addr == "2.2.2.2:8888"
        assert client._bootstrap_peers == ["/ip4/5.6.7.8/tcp/4001/p2p/Qm2"]
    finally:
        client.close()


# ---------------------------------------------------------------------------
# MxClientBase surface (paths that don't require a running node)
# ---------------------------------------------------------------------------


def test_publish_metadata_pre_start_returns_source_id():
    """First call carries empty endpoint; should return source_id without
    starting the DHT node."""
    client = MxDhtClient()
    try:
        identity = _base_identity()
        worker = p2p_pb2.WorkerMetadata(worker_rank=5)
        sid = client.publish_metadata(identity, worker, "worker-xyz")
        assert sid == compute_mx_source_id(identity)
        assert client._worker_rank == 5
        # Should NOT have started the asyncio loop / node
        assert not client._started.is_set()
        assert client._node is None
    finally:
        client.close()


def test_list_sources_requires_identity():
    client = MxDhtClient(worker_rank=0)
    try:
        with pytest.raises(ValueError, match="requires an identity"):
            client.list_sources(None)
    finally:
        client.close()


def test_list_sources_requires_worker_rank():
    client = MxDhtClient()
    try:
        with pytest.raises(RuntimeError, match="needs a worker_rank"):
            client.list_sources(_base_identity())
    finally:
        client.close()


def test_list_sources_returns_synthetic_ref():
    client = MxDhtClient(worker_rank=4)
    try:
        identity = _base_identity()
        resp = client.list_sources(identity)
        assert len(resp.instances) == 1
        ref = resp.instances[0]
        assert ref.mx_source_id == compute_mx_source_id(identity)
        assert ref.worker_id == ""
        assert ref.model_name == identity.model_name
        assert ref.worker_rank == 4
    finally:
        client.close()


def test_get_metadata_requires_worker_rank():
    client = MxDhtClient()
    try:
        with pytest.raises(RuntimeError, match="requires worker_rank"):
            client.get_metadata("abcdef0123456789", "worker-x")
    finally:
        client.close()


def test_update_status_is_noop():
    client = MxDhtClient()
    try:
        ok = client.update_status(
            "abcdef0123456789", "w0", 0, p2p_pb2.SOURCE_STATUS_READY,
        )
        assert ok is True
    finally:
        client.close()


def test_close_unstarted_is_noop():
    client = MxDhtClient()
    client.close()  # no exception, no thread to join
    assert client._loop is None
    assert client._loop_thread is None


def test_requires_p2p_metadata_class_attribute():
    assert MxDhtClient.REQUIRES_P2P_METADATA is True
    client = MxDhtClient()
    try:
        assert _is_p2p_metadata_enabled(client) is True
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Integration: real DhtNode round-trip (no GRPC)
# ---------------------------------------------------------------------------


def test_publish_writes_pointer_to_dht(monkeypatch):
    """Spin up the DHT node, publish a worker pointer, GET it back via the
    underlying kademlite node and verify the encoded payload matches.

    Skips the GetTensorManifest hop entirely - that's covered by the
    receiver-side tests (and integration tests with a real
    WorkerGrpcServer)."""
    monkeypatch.setenv("MX_DHT_LISTEN", "127.0.0.1:0")
    monkeypatch.delenv("MX_DHT_BOOTSTRAP_PEERS", raising=False)
    monkeypatch.delenv("MX_DHT_BOOTSTRAP_DNS", raising=False)

    client = MxDhtClient()
    try:
        identity = _base_identity()
        worker = p2p_pb2.WorkerMetadata(
            worker_rank=2,
            worker_grpc_endpoint="127.0.0.1:6557",
            metadata_endpoint="127.0.0.1:7000",
            agent_name="agent-test",
        )
        sid = client.publish_metadata(identity, worker, "worker-test")
        assert sid == compute_mx_source_id(identity)
        assert client._started.is_set()

        # Read back through the underlying node directly
        key = _key_for(sid, 2)
        raw = asyncio.run_coroutine_threadsafe(
            client._node.get(key), client._loop,
        ).result(timeout=10)
        assert raw is not None
        decoded = _decode_worker_pointer(raw)
        assert decoded["worker_grpc_endpoint"] == "127.0.0.1:6557"
        assert decoded["worker_rank"] == 2
        assert decoded["worker_id"] == "worker-test"
        assert decoded["metadata_endpoint"] == "127.0.0.1:7000"
        assert decoded["agent_name"] == "agent-test"
    finally:
        client.close()
        assert not client._started.is_set()


def test_get_metadata_missing_key_exhausts_retries(monkeypatch):
    """When no publisher exists, get_metadata exhausts retries and raises."""
    monkeypatch.setenv("MX_DHT_LISTEN", "127.0.0.1:0")
    monkeypatch.delenv("MX_DHT_BOOTSTRAP_PEERS", raising=False)
    monkeypatch.delenv("MX_DHT_BOOTSTRAP_DNS", raising=False)

    client = MxDhtClient(
        worker_rank=0, max_retries=1, backoff_seconds=0.01,
    )
    try:
        with pytest.raises(RuntimeError, match="exhausted"):
            client.get_metadata("nonexistent12345", "worker-x")
    finally:
        client.close()
