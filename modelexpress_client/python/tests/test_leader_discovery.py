# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-gRPC integration tests for the single-Service discovery contract."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock

import grpc
import pytest
import torch

from modelexpress import p2p_pb2, p2p_pb2_grpc
from modelexpress.metadata.artifact_manifest import (
    artifact_manifest_id,
    build_artifact_manifest,
)
from modelexpress.metadata.artifact_transfer import transfer_artifact_from_worker
from modelexpress.metadata.artifact_transport import ArtifactTransportContext
from modelexpress.metadata.instance_metadata import grpc_endpoint
from modelexpress.metadata.k8s_service_client import MxK8sServiceClient
from modelexpress.metadata.nixl_artifact_transport import NixlArtifactTransport
from modelexpress.metadata.source_id import compute_mx_source_id
from modelexpress.metadata.worker_server import WorkerGrpcServer


def identity(source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS, world_size=2):
    return p2p_pb2.SourceIdentity(
        mx_version="0.8.0",
        model_name="model",
        revision="immutable-checkpoint",
        mx_source_type=source_type,
        tensor_parallel_size=world_size,
        pipeline_parallel_size=1,
        dtype="bfloat16",
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
    )


class Pool:
    def __init__(self, stack, monkeypatch, size, per_node):
        self.size = size
        self.per_node = per_node
        self.identity = identity(world_size=size)
        self.source_id = compute_mx_source_id(self.identity)
        self.servers = []
        self.sources = []
        self.healthy = True
        for rank in range(size):
            server = WorkerGrpcServer(
                tensor_protos=[
                    p2p_pb2.TensorDescriptor(
                        name=f"rank-{rank}", addr=100 + rank, size=16
                    )
                ],
                mx_source_id=None,
                worker_rank=rank,
                worker_id=f"generation-{rank}",
                accelerator="cuda",
                metadata_endpoint=f"127.0.0.1:{10000 + rank}",
                agent_name=f"nixl-{rank}",
            )
            server.start()
            stack.callback(server.stop, grace=0)
            self.servers.append(server)
        self.endpoint = f"127.0.0.1:{self.servers[0].port}"
        monkeypatch.setenv("MX_WORKER_GRPC_PORT", str(self.servers[0].port))
        for rank, server in enumerate(self.servers):
            client = self.client(rank)
            client._leader._ready_fn = lambda: self.healthy
            worker = p2p_pb2.WorkerMetadata(
                worker_rank=rank,
                worker_grpc_endpoint=f"127.0.0.1:{server.port}",
            )
            client.attach_worker_server(server, worker, f"generation-{rank}")
            assert client._leader._leader_endpoint == self.endpoint
            source_id = client.publish_metadata(
                self.identity, worker, f"generation-{rank}"
            )
            server.set_mx_source_id(source_id)
            self.sources.append(client)

    def client(self, rank=0, service=None, retries=0):
        client = MxK8sServiceClient(
            worker_rank=rank,
            leader_service=service or self.endpoint,
            max_retries=retries,
            backoff_seconds=0,
        )
        client.configure_load_context(
            SimpleNamespace(
                identity=self.identity,
                global_rank=rank,
                node_rank=rank // self.per_node,
                device_id=rank % self.per_node,
                head_addr="127.0.0.1",
            )
        )
        return client

    def publish_artifact(
        self, rank, path, source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE, manager=None
    ):
        artifact_identity = identity(source_type, self.size)
        manifest = build_artifact_manifest(
            path.parent, chunk_size=8, mx_source_type=source_type
        )
        self.servers[rank].register_artifact_source(
            compute_mx_source_id(artifact_identity),
            artifact_manifest_id(manifest),
            manifest,
            manager,
            node_rank=rank // self.per_node,
            worker_grpc_endpoint=f"127.0.0.1:{self.servers[rank].port}",
        )
        return artifact_identity, manifest


@pytest.fixture
def pool_factory(monkeypatch):
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    with ExitStack() as stack:
        yield lambda size=2, per_node=1: Pool(stack, monkeypatch, size, per_node)


@pytest.mark.parametrize("size,per_node", [(1, 1), (4, 2), (16, 8)])
def test_one_endpoint_routes_all_global_ranks_to_actual_worker_ports(
    pool_factory, size, per_node
):
    pool = pool_factory(size, per_node)
    for rank in range(size):
        client = pool.client(rank)
        refs = client.list_sources(pool.identity).instances
        assert len(refs) == 1
        assert refs[0].worker_id == f"generation-{rank}"
        result = client.get_metadata(refs[0].mx_source_id, refs[0].worker_id)
        assert result.found
        assert (
            result.worker.worker_grpc_endpoint == f"127.0.0.1:{pool.servers[rank].port}"
        )
        assert result.worker.tensor_source.tensors[0].name == f"rank-{rank}"
        assert result.worker.agent_name == f"nixl-{rank}"
        assert client.worker_rpc_retry_policy() == (0, 0.0)


def test_missing_types_use_one_inventory_and_never_probe_artifact_headers(
    pool_factory, monkeypatch
):
    pool = pool_factory()
    servicer = pool.servers[0]._servicer
    # The generated handler captures the method at server startup; count the
    # directory instead, which is resolved dynamically by that handler.
    directory = servicer._instance_directory
    original_metadata = directory.metadata
    directory.metadata = Mock(wraps=original_metadata)
    sleeps = Mock(side_effect=AssertionError("missing artifact must not back off"))
    monkeypatch.setattr("modelexpress.metadata.leader_discovery.time.sleep", sleeps)
    for server in pool.servers:
        server._servicer._select_artifact_source = Mock(
            side_effect=AssertionError("no per-type RPC")
        )
    target = pool.client(retries=5)
    for source_type in range(3, 9):
        with pytest.raises(LookupError, match="no published compatible artifact"):
            target.discover_artifact_source(
                identity(source_type), node_rank=1, accelerator="cuda"
            )
    target.list_sources(pool.identity)
    target.get_metadata(pool.source_id, "generation-0")
    assert directory.metadata.call_count == 1
    sleeps.assert_not_called()


@pytest.mark.parametrize("size,per_node", [(4, 2), (16, 8)])
def test_node_local_artifacts_transfer_via_selected_direct_endpoint(
    pool_factory, tmp_path, size, per_node
):
    # Exercise actual manifests, chunk leases, checksums and file writes; only
    # the NIXL byte copy is replaced by the existing CPU test transport.
    from .test_artifact_transfer import (
        _FakeArtifactChunkManager,
        _FakeTargetNixlManager,
    )

    pool = pool_factory(size, per_node)
    for rank in (0, per_node):
        node_rank = rank // per_node
        source_dir = tmp_path / f"node-{node_rank}"
        source_dir.mkdir()
        path = source_dir / "cache.bin"
        payload = f"compiled-cache-for-node-{node_rank}".encode()
        path.write_bytes(payload)
        manager = _FakeArtifactChunkManager({str(path): payload})
        artifact_identity, manifest = pool.publish_artifact(rank, path, manager=manager)
        target = pool.client(rank)
        selected = target.discover_artifact_source(
            artifact_identity, node_rank=node_rank, accelerator="cuda"
        )
        assert selected.worker_id == f"generation-{rank}"
        assert selected.worker_grpc_endpoint == f"127.0.0.1:{pool.servers[rank].port}"
        destination = tmp_path / f"installed-node-{node_rank}.bin"
        staged = NixlArtifactTransport(
            mx_client=target,
            nixl_manager=_FakeTargetNixlManager(manager),
        ).fetch(
            SimpleNamespace(name="triton", target_file_paths=lambda: [destination]),
            artifact_identity,
            ArtifactTransportContext(node_rank=node_rank, accelerator="cuda"),
        )
        header = staged.header
        assert header.node_rank == node_rank
        assert header.artifact_id == artifact_manifest_id(manifest)
        assert destination.read_bytes() == payload
        assert not manager.leases


def test_stale_artifact_directory_rejected_before_writing_files(pool_factory, tmp_path):
    pool = pool_factory()
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    path = source_dir / "cache.bin"
    path.write_bytes(b"compiled-cache")
    artifact_identity, _ = pool.publish_artifact(1, path)
    selected = pool.client().discover_artifact_source(artifact_identity, node_rank=1)
    pool.servers[1]._servicer._worker_id = "new-generation"
    destination = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match="selected worker generation"):
        transfer_artifact_from_worker(
            selected.worker_grpc_endpoint,
            selected.mx_source_id,
            selected.artifact_id,
            None,
            target_file_paths=[destination],
            expected_worker_id=selected.worker_id,
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "failure", ["unhealthy", "stale", "unavailable", "expired", "unpublished"]
)
def test_incomplete_or_unhealthy_instance_is_not_a_cache_miss(pool_factory, failure):
    pool = pool_factory()
    if failure == "unhealthy":
        pool.healthy = False
    elif failure == "stale":
        pool.sources[1].update_status(
            pool.source_id, "generation-1", 1, p2p_pb2.SOURCE_STATUS_STALE
        )
    elif failure == "unavailable":
        pool.servers[1].stop(grace=0)
    elif failure == "expired":
        pool.servers[0]._servicer._instance_directory._clock = lambda: float("inf")
    else:
        pool.servers[1].set_mx_source_id("another-checkpoint")
    with pytest.raises(grpc.RpcError) as exc:
        pool.client().discover_artifact_source(
            identity(p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE), node_rank=1
        )
    assert exc.value.code() == grpc.StatusCode.UNAVAILABLE


def test_restarted_worker_invalidates_cached_generation(pool_factory):
    pool = pool_factory()
    target = pool.client(1)
    ref = target.list_sources(pool.identity).instances[0]
    pool.servers[1]._servicer._worker_id = "replacement-generation"
    with pytest.raises(grpc.RpcError) as exc:
        target.get_metadata(ref.mx_source_id, ref.worker_id)
    assert exc.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    # A new discovery also fails until this generation registers with leader.
    with pytest.raises(grpc.RpcError) as exc:
        pool.client().list_sources(pool.identity)
    assert exc.value.code() == grpc.StatusCode.UNAVAILABLE


def test_stale_registration_cannot_overwrite_live_generation(pool_factory):
    pool = pool_factory()
    request = p2p_pb2.RegisterInstanceWorkerRequest(
        mx_source_id=pool.source_id,
        world_size=2,
        ready=True,
        worker=p2p_pb2.InstanceWorkerEndpoint(
            global_rank=1,
            worker_rank=1,
            node_rank=1,
            worker_id="replacement",
            worker_grpc_endpoint="127.0.0.1:12345",
        ),
    )
    with grpc.insecure_channel(pool.endpoint) as channel:
        stub = p2p_pb2_grpc.WorkerServiceStub(channel)
        with pytest.raises(grpc.RpcError) as exc:
            stub.RegisterInstanceWorker(request, timeout=1)
        assert exc.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        # Explicit withdrawal lets the replacement register immediately.
        pool.sources[1].update_status(
            pool.source_id, "generation-1", 1, p2p_pb2.SOURCE_STATUS_STALE
        )
        assert (
            stub.RegisterInstanceWorker(request, timeout=1).leader_worker_id
            == "generation-0"
        )


def test_directory_rereads_inventory_for_new_targets(pool_factory, tmp_path):
    pool = pool_factory()
    artifact_identity = identity(p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE)
    first = pool.client()
    with pytest.raises(LookupError):
        first.discover_artifact_source(artifact_identity, node_rank=1)
    source_dir = tmp_path / "cache"
    source_dir.mkdir()
    path = source_dir / "cache.bin"
    path.write_bytes(b"published-later")
    pool.publish_artifact(1, path)
    assert (
        pool.client().discover_artifact_source(artifact_identity, node_rank=1).worker_id
        == "generation-1"
    )
    # A target's startup snapshot is stable; a miss does not start polling.
    with pytest.raises(LookupError):
        first.discover_artifact_source(artifact_identity, node_rank=1)


def test_concurrent_targets_do_not_deadlock_leader_self_queries(pool_factory):
    pool = pool_factory(4, 2)
    with ThreadPoolExecutor(max_workers=8) as executor:
        refs = list(
            executor.map(lambda _: pool.client().list_sources(pool.identity), range(16))
        )
    assert all(ref.instances[0].worker_id == "generation-0" for ref in refs)


def test_identity_mismatch_is_rejected(pool_factory):
    pool = pool_factory()
    target = pool.client()
    target._leader._identity.revision = "wrong-revision"
    with pytest.raises(grpc.RpcError) as exc:
        target.list_sources(target._leader._identity)
    assert exc.value.code() == grpc.StatusCode.FAILED_PRECONDITION


def test_retry_selects_valid_metadata_and_reuses_snapshot(pool_factory):
    pool = pool_factory()
    directory = pool.servers[0]._servicer._instance_directory
    original_metadata = directory.metadata
    calls = []

    def first_wrong_then_valid(request, context, local_inventory):
        result = original_metadata(request, context, local_inventory)
        calls.append(1)
        if len(calls) == 1:
            result.mx_source_id = "another-model"
        return result

    directory.metadata = first_wrong_then_valid
    target = pool.client(retries=1)
    assert target.list_sources(pool.identity).instances[0].worker_id == "generation-0"
    for source_type in range(3, 9):
        with pytest.raises(LookupError):
            target.discover_artifact_source(identity(source_type), node_rank=1)
    assert len(calls) == 2


def test_discovery_failure_budget_is_shared_across_artifact_types(pool_factory):
    pool = pool_factory()
    pool.healthy = False
    directory = pool.servers[0]._servicer._instance_directory
    directory.metadata = Mock(wraps=directory.metadata)
    target = pool.client(retries=2)
    for source_type in range(3, 9):
        with pytest.raises(grpc.RpcError):
            target.discover_artifact_source(identity(source_type), node_rank=1)
    assert directory.metadata.call_count == 3


def test_registration_expires_before_new_generation_replaces_it(pool_factory):
    pool = pool_factory()
    directory = pool.servers[0]._servicer._instance_directory
    now = directory._clock()
    directory._clock = lambda: now + 100
    request = p2p_pb2.RegisterInstanceWorkerRequest(
        mx_source_id=pool.source_id,
        world_size=2,
        ready=True,
        worker=p2p_pb2.InstanceWorkerEndpoint(
            global_rank=1,
            worker_rank=1,
            node_rank=1,
            worker_id="replacement",
            worker_grpc_endpoint="127.0.0.1:12345",
        ),
    )
    with grpc.insecure_channel(pool.endpoint) as channel:
        response = p2p_pb2_grpc.WorkerServiceStub(channel).RegisterInstanceWorker(
            request, timeout=1
        )
    assert response.leader_worker_id == "generation-0"


def test_withdrawal_during_inventory_read_invalidates_snapshot(pool_factory):
    pool = pool_factory()
    servicer = pool.servers[0]._servicer
    original_inventory = servicer.GetWorkerInventory

    def withdraw_before_return(request, context):
        result = original_inventory(request, context)
        pool.sources[1].update_status(
            pool.source_id, "generation-1", 1, p2p_pb2.SOURCE_STATUS_STALE
        )
        return result

    servicer.GetWorkerInventory = withdraw_before_return
    with pytest.raises(grpc.RpcError) as exc:
        pool.client().list_sources(pool.identity)
    assert exc.value.code() == grpc.StatusCode.UNAVAILABLE


@pytest.mark.parametrize(
    "corruption", ["identity", "leader", "size", "rank", "endpoint"]
)
def test_client_rejects_invalid_directory_snapshots(pool_factory, corruption):
    pool = pool_factory()
    target = pool.client()
    snapshot = target._leader.metadata()
    if corruption == "identity":
        snapshot.mx_source_id = "wrong"
    elif corruption == "leader":
        snapshot.leader_worker_id = "old-leader"
    elif corruption == "size":
        snapshot.workers.pop()
    elif corruption == "rank":
        snapshot.workers[1].worker.global_rank = 0
    else:
        snapshot.workers[1].worker.worker_grpc_endpoint = snapshot.workers[
            0
        ].worker.worker_grpc_endpoint
    with pytest.raises(ValueError):
        target._leader._validate(snapshot, pool.source_id)


def test_env_enables_leader_mode_without_changing_legacy_default(monkeypatch):
    monkeypatch.delenv("MX_K8S_LEADER_SERVICE", raising=False)
    assert not MxK8sServiceClient().LEADER_DISCOVERY
    monkeypatch.setenv("MX_K8S_LEADER_SERVICE", "mx-sources:6555")
    client = MxK8sServiceClient(worker_rank=8)
    assert client.LEADER_DISCOVERY
    assert client._leader.service == "mx-sources:6555"


def test_load_context_publication_and_heartbeat_register_without_manual_wiring(
    monkeypatch,
):
    from modelexpress.load_strategy.context import LoadContext
    from modelexpress.metadata import publish
    from modelexpress.metadata.publisher import PublisherThread

    monkeypatch.setenv("MX_WORKER_GRPC_PORT", "0")
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    monkeypatch.setattr(publish, "_get_worker_host", lambda: "127.0.0.1")
    monkeypatch.setattr(PublisherThread, "start", lambda self: self._tick())
    monkeypatch.setattr(
        "modelexpress.metadata.artifact_lifecycle.artifact_health_ready",
        lambda *a, **kw: True,
    )
    monkeypatch.setattr(publish, "_worker_servers", {})
    monkeypatch.setattr(publish, "_heartbeat_threads", {})
    source = MxK8sServiceClient(worker_rank=0, leader_service="unused:6555")
    weights = identity(world_size=1)
    ctx = LoadContext(
        model_config=None,
        load_config=None,
        target_device=torch.device("cpu"),
        global_rank=0,
        worker_rank=0,
        local_rank=0,
        device_id=0,
        identity=weights,
        mx_client=source,
        worker_id="source-generation",
    )
    assert source._leader._identity == weights
    manager = SimpleNamespace(
        _listen_port=5555,
        agent_name="source-agent",
        is_healthy=lambda: True,
    )
    try:
        publish.publish_metadata_and_ready(
            source,
            manager,
            {"weight": torch.ones(4)},
            0,
            0,
            weights,
            ctx.worker_id,
        )
        server = publish._worker_servers[0]
        target = MxK8sServiceClient(
            worker_rank=0, leader_service=f"127.0.0.1:{server.port}", max_retries=0
        )
        target.configure_load_context(ctx)
        refs = target.list_sources(weights).instances
        result = target.get_metadata(refs[0].mx_source_id, refs[0].worker_id)
        assert result.worker.tensor_source.tensors[0].size == 16
        publisher = publish._heartbeat_threads[0]
        assert publisher.mx_source_id == compute_mx_source_id(weights)
        # Expiry disables discovery until the normal heartbeat renews it.
        directory = server._servicer._instance_directory
        now = directory._clock()
        directory._clock = lambda: now + 100
        with pytest.raises(grpc.RpcError):
            target._leader._snapshots.clear()
            target.list_sources(weights)
        publisher._tick()
        target._leader._snapshots.clear()
        assert target.list_sources(weights).instances[0].worker_id == ctx.worker_id
    finally:
        for publisher in publish._heartbeat_threads.values():
            publisher.stop()


@pytest.mark.parametrize(
    "framework,port,timeout",
    [
        (p2p_pb2.BACKEND_FRAMEWORK_VLLM, 8000, 1.0),
        (p2p_pb2.BACKEND_FRAMEWORK_SGLANG, 30000, 5.0),
    ],
)
def test_runtime_head_and_framework_health_probe(monkeypatch, framework, port, timeout):
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 16)
    monkeypatch.setenv("MX_WORKER_GRPC_PORT", "6555")
    monkeypatch.delenv("MX_ARTIFACT_READY_URL", raising=False)
    probe = Mock(return_value=True)
    monkeypatch.setattr(
        "modelexpress.metadata.artifact_lifecycle.artifact_health_ready", probe
    )
    weights = identity(world_size=16)
    weights.backend_framework = framework
    client = MxK8sServiceClient(worker_rank=8, leader_service="discovery:6555")
    client.configure_load_context(
        SimpleNamespace(
            identity=weights,
            global_rank=8,
            node_rank=1,
            device_id=0,
            head_addr="engine-leader",
        )
    )
    assert client._leader._world_size == 16
    assert client._leader._leader_endpoint == "engine-leader:6555"
    assert client._leader._ready_fn()
    probe.assert_called_once_with(
        f"http://engine-leader:{port}/health", timeout_secs=timeout
    )


@pytest.mark.parametrize(
    "host,expected",
    [("leader", "leader:6555"), ("::1", "[::1]:6555"), ("[::1]", "[::1]:6555")],
)
def test_direct_endpoints_support_ipv6(host, expected):
    assert grpc_endpoint(host, 6555) == expected
