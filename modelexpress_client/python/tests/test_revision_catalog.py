# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor

import grpc

from modelexpress import revision_pb2, revision_pb2_grpc
from modelexpress.refit.catalog import GrpcRevisionCatalog, RevisionCatalog
from modelexpress.refit.manifest import (
    RevisionManifest,
    RevisionRecord,
    RevisionState,
)


class FakeStub:
    def __init__(self, record: revision_pb2.RevisionRecord) -> None:
        self.record = record
        self.calls: list[tuple[str, object]] = []

    def PublishRevision(self, request, **_kwargs):
        self.calls.append(("publish", request))
        return self.record

    def GetRevision(self, request, **_kwargs):
        self.calls.append(("get", request))
        return self.record

    def CommitRevision(self, request, **_kwargs):
        self.calls.append(("commit", request))
        committed = revision_pb2.RevisionRecord()
        committed.CopyFrom(self.record)
        committed.state = revision_pb2.REVISION_STATE_COMMITTED
        return committed


def launch_manifest() -> RevisionManifest:
    return RevisionManifest(
        model_id="model",
        target_version="0",
        target_digest="sha256:target-0",
        format_digest="sha256:format",
    )


def catalog() -> tuple[GrpcRevisionCatalog, FakeStub]:
    record = RevisionRecord(
        manifest=launch_manifest(),
        state=RevisionState.READY,
    ).to_proto()
    stub = FakeStub(record)
    return GrpcRevisionCatalog(stub=stub), stub


def test_protocol_has_only_three_revision_operations():
    assert {
        name
        for name, value in vars(RevisionCatalog).items()
        if callable(value) and not name.startswith("_")
    } == {
        "publish_revision",
        "get_revision",
        "commit_revision",
    }


def test_publish_sends_only_the_manifest_and_returns_direct_record():
    client, stub = catalog()

    result = client.publish_revision(launch_manifest())

    assert result.state is RevisionState.READY
    operation, request = stub.calls[-1]
    assert operation == "publish"
    assert request == revision_pb2.PublishRevisionRequest(
        manifest=launch_manifest().to_proto()
    )


def test_get_and_commit_use_exact_model_and_target_version():
    client, stub = catalog()

    fetched = client.get_revision("model", "0")
    committed = client.commit_revision("model", "0")

    assert fetched.state is RevisionState.READY
    assert committed.state is RevisionState.COMMITTED
    assert stub.calls[-2] == (
        "get",
        revision_pb2.GetRevisionRequest(model_id="model", target_version="0"),
    )
    assert stub.calls[-1] == (
        "commit",
        revision_pb2.CommitRevisionRequest(model_id="model", target_version="0"),
    )


def test_client_has_no_deferred_catalog_operations():
    client, _ = catalog()

    assert not hasattr(client, "list_ready_revisions")
    assert not hasattr(client, "get_recovery_candidates")
    assert not hasattr(client, "update_receiver_state")
    assert not hasattr(client, "commit_version")


class LoopbackCatalog(revision_pb2_grpc.RevisionCatalogServiceServicer):
    def __init__(self) -> None:
        self.records: dict[tuple[str, str], revision_pb2.RevisionRecord] = {}

    def PublishRevision(self, request, context):
        manifest = request.manifest
        key = (manifest.model_id, manifest.target_version)
        existing = self.records.get(key)
        if existing is not None:
            if existing.manifest != manifest:
                context.abort(grpc.StatusCode.ALREADY_EXISTS, "manifest conflict")
            return existing
        record = revision_pb2.RevisionRecord(
            manifest=manifest,
            state=revision_pb2.REVISION_STATE_READY,
        )
        self.records[key] = record
        return record

    def GetRevision(self, request, context):
        record = self.records.get((request.model_id, request.target_version))
        if record is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "revision not found")
        return record

    def CommitRevision(self, request, context):
        key = (request.model_id, request.target_version)
        record = self.records.get(key)
        if record is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "revision not found")
        record.state = revision_pb2.REVISION_STATE_COMMITTED
        return record


def test_real_grpc_loopback_publish_get_and_commit():
    server = grpc.server(ThreadPoolExecutor(max_workers=1))
    revision_pb2_grpc.add_RevisionCatalogServiceServicer_to_server(
        LoopbackCatalog(), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with GrpcRevisionCatalog(f"127.0.0.1:{port}") as client:
            published = client.publish_revision(launch_manifest())
            fetched = client.get_revision("model", "0")
            committed = client.commit_revision("model", "0")
        assert published.state is RevisionState.READY
        assert fetched == published
        assert committed.state is RevisionState.COMMITTED
    finally:
        server.stop(grace=None).wait()
