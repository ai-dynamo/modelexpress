# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publication isolation for independent model sessions on the same worker."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import grpc
import pytest
import torch

from modelexpress import p2p_pb2, p2p_pb2_grpc
from modelexpress.load_strategy.base import drain_tensor_readers, unpublish_metadata
from modelexpress.load_strategy.context import LoadContext
from modelexpress.metadata import publish as publication
from modelexpress.metadata.source_id import compute_mx_source_id


@pytest.fixture(autouse=True)
def isolated_publications(monkeypatch):
    for name in ("_heartbeat_threads", "_worker_servers", "_central_manifests"):
        monkeypatch.setattr(publication, name, {})
    monkeypatch.setenv("MX_WORKER_GRPC_PORT", "6555")
    monkeypatch.setenv("MX_METADATA_BACKEND", "central")


def make_context(*, draft=True):
    namespace = "draft-0" if draft else ""
    identity = p2p_pb2.SourceIdentity(model_name="org/model")
    if draft:
        identity.extra_parameters["model_role"] = "draft"
    return LoadContext(
        model_config=SimpleNamespace(),
        load_config=SimpleNamespace(),
        target_device=torch.device("cpu"),
        global_rank=3,
        worker_rank=3,
        local_rank=0,
        device_id=0,
        identity=identity,
        mx_client=MagicMock(),
        worker_id=namespace or "main",
        p2p_role="draft" if draft else "main",
        publication_namespace=namespace,
        port_offset=10000 if draft else 0,
    )


@pytest.mark.parametrize("p2p_metadata", ["0", "1"])
def test_publication_and_cleanup_are_isolated_on_same_rank(p2p_metadata, monkeypatch):
    monkeypatch.setenv("MX_P2P_METADATA", p2p_metadata)
    workers = []
    publishers = []

    def publisher(**kwargs):
        heartbeat = MagicMock()
        heartbeat.options = kwargs
        publishers.append(heartbeat)
        return heartbeat

    def worker_server(**kwargs):
        worker = MagicMock()
        worker.options = kwargs
        worker.start.return_value = kwargs["port"]
        workers.append(worker)
        return worker

    main, draft = make_context(draft=False), make_context()
    with (
        patch.object(publication, "PublisherThread", side_effect=publisher),
        patch(
            "modelexpress.metadata.worker_server.WorkerGrpcServer",
            side_effect=worker_server,
        ),
    ):
        for ctx in (main, draft):
            ctx.mx_client = MagicMock()
            ctx.mx_client.publish_metadata.return_value = compute_mx_source_id(
                ctx.identity
            )
            manager = MagicMock(
                _listen_port=5555 + ctx.port_offset,
                nixl_metadata=b"metadata",
                agent_name=f"agent-{ctx.publication_namespace}",
            )
            publication.publish_metadata_and_ready(
                ctx.mx_client,
                manager,
                {"weight": torch.ones(2)},
                ctx.worker_rank,
                ctx.device_id,
                ctx.identity,
                ctx.worker_id,
                publication_namespace=ctx.publication_namespace,
                port_offset=ctx.port_offset,
            )
            publishers[-1].options["publish_fn"]()

    assert set(publication._heartbeat_threads) == {3, (3, "draft-0")}
    if p2p_metadata == "1":
        assert [worker.options["port"] for worker in workers] == [6555, 16555]
        assert set(publication._worker_servers) == {0, (0, "draft-0")}
        assert not publication._central_manifests
    else:
        assert not workers
        assert not publication._worker_servers
        assert set(publication._central_manifests) == {0, (0, "draft-0")}
    unpublish_metadata(draft)
    publishers[1].stop.assert_called_once()
    publishers[0].stop.assert_not_called()
    assert set(publication._heartbeat_threads) == {3}
    if p2p_metadata == "1":
        assert set(publication._worker_servers) == {0}
        workers[1].stop.assert_called_once()
        workers[0].stop.assert_not_called()
    else:
        assert set(publication._central_manifests) == {0}


@pytest.mark.parametrize("draft", [False, True])
def test_reader_drain_only_blocks_requested_session(draft):
    main_server, draft_server = MagicMock(), MagicMock()
    publication._worker_servers.update({0: main_server, (0, "draft-0"): draft_server})

    drain_tensor_readers(make_context(draft=draft), timeout=2.5)

    selected, other = (draft_server, main_server) if draft else (main_server, draft_server)
    selected.drain_tensor_reads.assert_called_once_with(2.5)
    other.drain_tensor_reads.assert_not_called()
    selected.stop.assert_not_called()
    other.stop.assert_not_called()
    assert set(publication._worker_servers) == {0, (0, "draft-0")}


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("published_draft", [False, True])
def test_reader_drain_guards_only_its_central_publication(draft, published_draft):
    heartbeat = MagicMock()
    key = (3, "draft-0") if published_draft else 3
    publication._heartbeat_threads[key] = heartbeat
    ctx = make_context(draft=draft)

    if draft == published_draft:
        with pytest.raises(RuntimeError, match="centrally published tensors"):
            drain_tensor_readers(ctx, timeout=2.5)
    else:
        drain_tensor_readers(ctx, timeout=2.5)

    heartbeat.stop.assert_not_called()
    assert publication._heartbeat_threads[key] is heartbeat


def test_main_and_draft_grpc_manifests_are_separate_and_survive_draft_cleanup(
    monkeypatch,
):
    monkeypatch.setenv("MX_P2P_METADATA", "1")
    monkeypatch.setenv("MX_WORKER_GRPC_PORT", "0")
    monkeypatch.setattr(publication, "_get_worker_host", lambda: "127.0.0.1")
    main, draft = make_context(draft=False), make_context()
    endpoints = []
    identities = []
    weights = [torch.ones(2), torch.ones(3)]
    try:
        for ctx, weight in zip((main, draft), weights):
            sid = compute_mx_source_id(ctx.identity)
            ctx.mx_client = MagicMock()
            ctx.mx_client.publish_metadata.return_value = sid
            manager = MagicMock(_listen_port=5555, agent_name=ctx.worker_id)
            with patch.object(publication, "PublisherThread") as heartbeat:
                # Ephemeral ports allow this test to run concurrently with servers.
                publication.publish_metadata_and_ready(
                    ctx.mx_client,
                    manager,
                    {"weight": weight},
                    ctx.worker_rank,
                    ctx.device_id,
                    ctx.identity,
                    ctx.worker_id,
                    publication_namespace=ctx.publication_namespace,
                )
                heartbeat.call_args.kwargs["publish_fn"]()
            endpoints.append(
                ctx.mx_client.publish_metadata.call_args.args[1].worker_grpc_endpoint
            )
            identities.append(sid)

        assert endpoints[0] != endpoints[1]
        for index, endpoint in enumerate(endpoints):
            with grpc.insecure_channel(endpoint) as channel:
                stub = p2p_pb2_grpc.WorkerServiceStub(channel)
                manifest = stub.GetTensorManifest(
                    p2p_pb2.GetTensorManifestRequest(mx_source_id=identities[index]),
                    timeout=5,
                )
                assert manifest.tensors[0].addr == weights[index].data_ptr()
                with pytest.raises(grpc.RpcError) as error:
                    stub.GetTensorManifest(
                        p2p_pb2.GetTensorManifestRequest(
                            mx_source_id=identities[1 - index]
                        ),
                        timeout=5,
                    )
                assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        unpublish_metadata(draft)
        with grpc.insecure_channel(endpoints[0]) as channel:
            manifest = p2p_pb2_grpc.WorkerServiceStub(channel).GetTensorManifest(
                p2p_pb2.GetTensorManifestRequest(mx_source_id=identities[0]),
                timeout=5,
            )
            assert manifest.tensors[0].addr == weights[0].data_ptr()
    finally:
        unpublish_metadata(draft)
        unpublish_metadata(main)
