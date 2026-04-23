# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MxK8sServiceClient and the client factory."""

from __future__ import annotations

import threading
from concurrent import futures

import grpc
import pytest

from modelexpress import p2p_pb2
from modelexpress import p2p_pb2_grpc
from modelexpress.client import MxClient
from modelexpress.client_factory import create_metadata_client
from modelexpress.k8s_service_client import MxK8sServiceClient
from modelexpress.source_id import compute_mx_source_id
from modelexpress.metadata import _is_p2p_metadata_enabled


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
# Factory selection
# ---------------------------------------------------------------------------


def test_factory_default_returns_mx_client(monkeypatch):
    monkeypatch.delenv("MX_METADATA_BACKEND", raising=False)
    assert isinstance(create_metadata_client(), MxClient)


@pytest.mark.parametrize("value", ["", "server", "redis", "kubernetes", "k8s", "crd"])
def test_factory_central_aliases_return_mx_client(monkeypatch, value):
    monkeypatch.setenv("MX_METADATA_BACKEND", value)
    assert isinstance(create_metadata_client(), MxClient)


@pytest.mark.parametrize("value", ["k8s-service", "service", "K8S-SERVICE"])
def test_factory_k8s_service_aliases_return_k8s_client(monkeypatch, value):
    monkeypatch.setenv("MX_METADATA_BACKEND", value)
    client = create_metadata_client(worker_rank=3)
    assert isinstance(client, MxK8sServiceClient)
    assert client._worker_rank == 3


def test_factory_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("MX_METADATA_BACKEND", "bogus")
    with pytest.raises(ValueError, match="Unknown MX_METADATA_BACKEND"):
        create_metadata_client()


# ---------------------------------------------------------------------------
# publish_metadata / list_sources / update_status behavior
# ---------------------------------------------------------------------------


def test_publish_metadata_returns_local_source_id_and_records_rank():
    client = MxK8sServiceClient()
    identity = _base_identity()
    worker = p2p_pb2.WorkerMetadata(worker_rank=5)
    sid = client.publish_metadata(identity, worker, "worker-xyz")
    assert sid == compute_mx_source_id(identity)
    assert client._worker_rank == 5


def test_class_declares_requires_p2p_metadata():
    # metadata._is_p2p_metadata_enabled() consults this class attribute
    # to force the P2P path regardless of MX_P2P_METADATA env var.
    assert MxK8sServiceClient.REQUIRES_P2P_METADATA is True


# ---------------------------------------------------------------------------
# _is_p2p_metadata_enabled backend-driven forcing
# ---------------------------------------------------------------------------


def test_is_p2p_metadata_enabled_forced_by_k8s_service_client(monkeypatch):
    # Backend declares REQUIRES_P2P_METADATA=True, so env var is ignored.
    monkeypatch.delenv("MX_P2P_METADATA", raising=False)
    assert _is_p2p_metadata_enabled(MxK8sServiceClient()) is True

    monkeypatch.setenv("MX_P2P_METADATA", "0")
    assert _is_p2p_metadata_enabled(MxK8sServiceClient()) is True


def test_is_p2p_metadata_enabled_mx_client_honors_env_var(monkeypatch):
    # MxClient has no REQUIRES_P2P_METADATA, so env var is the source of truth.
    monkeypatch.delenv("MX_P2P_METADATA", raising=False)
    assert _is_p2p_metadata_enabled(MxClient()) is False

    monkeypatch.setenv("MX_P2P_METADATA", "0")
    assert _is_p2p_metadata_enabled(MxClient()) is False

    monkeypatch.setenv("MX_P2P_METADATA", "1")
    assert _is_p2p_metadata_enabled(MxClient()) is True


def test_is_p2p_metadata_enabled_warns_on_conflicting_env(monkeypatch, caplog):
    # If the user explicitly sets MX_P2P_METADATA=0 alongside a forcing
    # backend, their setting is ignored but a warning fires.
    import logging
    monkeypatch.setenv("MX_P2P_METADATA", "0")
    with caplog.at_level(logging.WARNING, logger="modelexpress.metadata"):
        result = _is_p2p_metadata_enabled(MxK8sServiceClient())
    assert result is True
    assert any("is ignored for backend" in rec.message for rec in caplog.records)


def test_list_sources_returns_single_synthetic_ref_at_caller_rank():
    client = MxK8sServiceClient(worker_rank=2)
    identity = _base_identity()
    resp = client.list_sources(identity=identity)
    assert len(resp.instances) == 1
    inst = resp.instances[0]
    assert inst.worker_rank == 2
    assert inst.mx_source_id == compute_mx_source_id(identity)
    assert inst.model_name == identity.model_name


def test_list_sources_requires_identity():
    client = MxK8sServiceClient(worker_rank=0)
    with pytest.raises(ValueError, match="identity"):
        client.list_sources()


def test_list_sources_requires_known_worker_rank():
    client = MxK8sServiceClient()  # no rank, no publish_metadata yet
    with pytest.raises(RuntimeError, match="worker_rank"):
        client.list_sources(identity=_base_identity())


def test_update_status_is_noop_returning_true():
    client = MxK8sServiceClient(worker_rank=0)
    assert client.update_status("sid", "wid", 0, p2p_pb2.SOURCE_STATUS_READY) is True


def test_resolve_endpoint_substitutes_rank():
    client = MxK8sServiceClient(
        worker_rank=7,
        service_pattern="my-svc-rank-{rank}.ns.svc.cluster.local:6555",
    )
    assert client._resolve_endpoint() == "my-svc-rank-7.ns.svc.cluster.local:6555"


def test_close_is_safe_noop():
    client = MxK8sServiceClient(worker_rank=0)
    client.close()
    client.close()  # idempotent


# ---------------------------------------------------------------------------
# get_metadata against a real in-process gRPC WorkerService
# ---------------------------------------------------------------------------


class _FakeWorkerServicer(p2p_pb2_grpc.WorkerServiceServicer):
    """In-process gRPC servicer that can be tuned to either succeed or
    fail with FAILED_PRECONDITION, to exercise the retry loop."""

    def __init__(
        self,
        mx_source_id: str,
        worker_rank: int,
        *,
        fail_first_n: int = 0,
    ):
        self._mx_source_id = mx_source_id
        self._worker_rank = worker_rank
        self._fail_first_n = fail_first_n
        self._calls = 0
        self._lock = threading.Lock()

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._calls

    def GetTensorManifest(self, request, context):
        with self._lock:
            self._calls += 1
            should_fail = self._calls <= self._fail_first_n
        if should_fail:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "simulated stale backend",
            )
        return p2p_pb2.GetTensorManifestResponse(
            mx_source_id=self._mx_source_id,
            tensors=[p2p_pb2.TensorDescriptor(name="t0", size=16, device_id=0)],
            metadata_endpoint="10.0.0.1:5555",
            agent_name="fake-agent",
            worker_rank=self._worker_rank,
        )


def _start_fake_server(servicer) -> tuple[grpc.Server, int]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    p2p_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, port


def test_get_metadata_success_builds_synthetic_response():
    identity = _base_identity()
    sid = compute_mx_source_id(identity)
    servicer = _FakeWorkerServicer(sid, worker_rank=4)
    server, port = _start_fake_server(servicer)
    try:
        client = MxK8sServiceClient(
            worker_rank=4,
            service_pattern=f"127.0.0.1:{port}",
        )
        resp = client.get_metadata(sid, "worker-xyz")
        assert resp.found is True
        assert resp.mx_source_id == sid
        assert resp.worker.worker_rank == 4
        assert resp.worker.metadata_endpoint == "10.0.0.1:5555"
        assert resp.worker.agent_name == "fake-agent"
        assert resp.worker.status == p2p_pb2.SOURCE_STATUS_READY
        assert len(resp.worker.tensors) == 1
        assert resp.worker.tensors[0].name == "t0"
        assert servicer.call_count == 1
    finally:
        server.stop(grace=None)


def test_get_metadata_retries_on_failed_precondition():
    identity = _base_identity()
    sid = compute_mx_source_id(identity)
    # Fail twice, succeed on the third attempt.
    servicer = _FakeWorkerServicer(sid, worker_rank=0, fail_first_n=2)
    server, port = _start_fake_server(servicer)
    try:
        client = MxK8sServiceClient(
            worker_rank=0,
            service_pattern=f"127.0.0.1:{port}",
            max_retries=5,
            backoff_seconds=0.0,
        )
        resp = client.get_metadata(sid, "worker-xyz")
        assert resp.found is True
        assert servicer.call_count == 3
    finally:
        server.stop(grace=None)


def test_get_metadata_gives_up_after_max_retries():
    identity = _base_identity()
    sid = compute_mx_source_id(identity)
    # Always fail.
    servicer = _FakeWorkerServicer(sid, worker_rank=0, fail_first_n=100)
    server, port = _start_fake_server(servicer)
    try:
        client = MxK8sServiceClient(
            worker_rank=0,
            service_pattern=f"127.0.0.1:{port}",
            max_retries=2,
            backoff_seconds=0.0,
        )
        with pytest.raises(grpc.RpcError):
            client.get_metadata(sid, "worker-xyz")
        # max_retries=2 means 3 attempts total (initial + 2 retries).
        assert servicer.call_count == 3
    finally:
        server.stop(grace=None)


def test_get_metadata_without_worker_rank_raises():
    client = MxK8sServiceClient()
    with pytest.raises(RuntimeError, match="worker_rank"):
        client.get_metadata("sid", "wid")


def test_get_metadata_rejects_source_id_mismatch():
    # Server advertises a different mx_source_id than the client asks
    # for. Client should retry up to max_retries times, then raise.
    identity = _base_identity()
    sid = compute_mx_source_id(identity)
    servicer = _FakeWorkerServicer("wrong-source-id", worker_rank=0)
    server, port = _start_fake_server(servicer)
    try:
        client = MxK8sServiceClient(
            worker_rank=0,
            service_pattern=f"127.0.0.1:{port}",
            max_retries=2,
            backoff_seconds=0.0,
        )
        with pytest.raises(RuntimeError, match="mx_source_id mismatch"):
            client.get_metadata(sid, "worker-xyz")
        # max_retries=2 -> 3 attempts total.
        assert servicer.call_count == 3
    finally:
        server.stop(grace=None)


def test_get_metadata_rejects_worker_rank_mismatch():
    # Server returns a different worker_rank than the caller's rank.
    # Typical "misconfigured Service selector" scenario.
    identity = _base_identity()
    sid = compute_mx_source_id(identity)
    servicer = _FakeWorkerServicer(sid, worker_rank=5)
    server, port = _start_fake_server(servicer)
    try:
        client = MxK8sServiceClient(
            worker_rank=3,
            service_pattern=f"127.0.0.1:{port}",
            max_retries=1,
            backoff_seconds=0.0,
        )
        with pytest.raises(RuntimeError, match="worker_rank mismatch"):
            client.get_metadata(sid, "worker-xyz")
        # max_retries=1 -> 2 attempts total.
        assert servicer.call_count == 2
    finally:
        server.stop(grace=None)
