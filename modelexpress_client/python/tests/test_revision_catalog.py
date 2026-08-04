# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed revision-catalog boundary and the concrete gRPC client."""

from __future__ import annotations

import dataclasses
import inspect
import re
from concurrent import futures

import grpc
import pytest

from modelexpress import revision_pb2, revision_pb2_grpc
from modelexpress.refit import catalog as catalog_module
from modelexpress.refit.catalog import (
    CatalogProtocolError,
    GrpcRevisionCatalog,
    PublishedRevision,
    RecoveryCandidatePage,
    RevisionCatalog,
    RevisionPage,
)
from modelexpress.refit.manifest import (
    ChangeState,
    DeltaLocation,
    DeltaTransferMethod,
    PublicationMode,
    RankDelta,
    ReceiverRevisionState,
    ReceiverStateRecord,
    RecoveryCandidateKind,
    RevisionLifecycleState,
    RevisionManifest,
    RevisionRank,
    RevisionRecord,
    S3Location,
)

MODEL_ID = "Qwen/Qwen3-30B-A3B"

CATALOG_RPCS = (
    "publish_revision",
    "get_revision",
    "list_ready_revisions",
    "get_recovery_candidates",
    "update_receiver_state",
    "commit_version",
)


def _manifest(version: str = "v2", base_version: str | None = "v1") -> RevisionManifest:
    return RevisionManifest(
        model_id=MODEL_ID,
        version=version,
        base_version=base_version,
        transfer_method=DeltaTransferMethod.CANONICAL,
        delta_method="tensor_byte_xor",
        compression_algorithm="zstd",
        format_digest="sha256:format",
        base_digest=f"sha256:target-{base_version}" if base_version else None,
        target_digest=f"sha256:target-{version}",
        ranks=(
            RevisionRank(
                trainer_rank=0,
                producer_id="trainer-0",
                source_layout_digest="sha256:layout",
                delta=RankDelta(
                    change_state=ChangeState.DIRTY,
                    checksum="8e1f2b3c",
                    location=DeltaLocation(
                        s3=S3Location(bucket="mx-delta", key=f"{version}/root.json")
                    ),
                ),
            ),
        ),
    )


class _RecordingStub:
    """Captures the exact protobuf request each typed call sends."""

    def __init__(self, **responses):
        self.requests: dict[str, object] = {}
        self._responses = responses

    def _call(self, name, request):
        self.requests[name] = request
        return self._responses[name]

    def PublishRevision(self, request):
        return self._call("PublishRevision", request)

    def GetRevision(self, request):
        return self._call("GetRevision", request)

    def ListReadyRevisions(self, request):
        return self._call("ListReadyRevisions", request)

    def GetRecoveryCandidates(self, request):
        return self._call("GetRecoveryCandidates", request)

    def UpdateReceiverState(self, request):
        return self._call("UpdateReceiverState", request)

    def CommitVersion(self, request):
        return self._call("CommitVersion", request)


def _ready_record(version: str = "v2") -> revision_pb2.RevisionRecord:
    return revision_pb2.RevisionRecord(
        manifest=_manifest(version).to_proto(),
        state=revision_pb2.REVISION_LIFECYCLE_STATE_READY,
        created_at_unix_ms=17,
        state_changed_at_unix_ms=17,
    )


def test_publish_revision_maps_request_and_response():
    stub = _RecordingStub(
        PublishRevision=revision_pb2.PublishRevisionResponse(
            revision=_ready_record(), created=True
        )
    )
    client = GrpcRevisionCatalog(stub=stub)

    result = client.publish_revision(
        _manifest(), publisher_id="trainer-0", publication_mode=PublicationMode.ASYNC
    )

    request = stub.requests["PublishRevision"]
    assert request.manifest == _manifest().to_proto()
    assert request.publisher_id == "trainer-0"
    assert request.HasField("publication_mode")
    assert request.publication_mode == revision_pb2.PUBLICATION_MODE_ASYNC
    assert result == PublishedRevision(
        revision=RevisionRecord(
            manifest=_manifest(),
            state=RevisionLifecycleState.READY,
            created_at_unix_ms=17,
            state_changed_at_unix_ms=17,
        ),
        created=True,
    )


def test_publish_revision_omits_absent_publication_mode():
    stub = _RecordingStub(
        PublishRevision=revision_pb2.PublishRevisionResponse(
            revision=_ready_record(), created=False
        )
    )
    client = GrpcRevisionCatalog(stub=stub)

    client.publish_revision(_manifest(), publisher_id="trainer-0")

    assert not stub.requests["PublishRevision"].HasField("publication_mode")


def test_get_revision_maps_identity_and_record():
    stub = _RecordingStub(GetRevision=revision_pb2.GetRevisionResponse(revision=_ready_record()))
    client = GrpcRevisionCatalog(stub=stub)

    record = client.get_revision(MODEL_ID, "v2")

    assert stub.requests["GetRevision"] == revision_pb2.GetRevisionRequest(
        model_id=MODEL_ID, version="v2"
    )
    assert record.manifest == _manifest()
    assert record.state is RevisionLifecycleState.READY


def test_list_ready_revisions_uses_opaque_page_tokens():
    stub = _RecordingStub(
        ListReadyRevisions=revision_pb2.ListReadyRevisionsResponse(
            revisions=[
                revision_pb2.RevisionSummary(
                    model_id=MODEL_ID,
                    version="v2",
                    state=revision_pb2.REVISION_LIFECYCLE_STATE_READY,
                    ready_at_unix_ms=17,
                )
            ],
            next_page_token="opaque::v2::7",
        )
    )
    client = GrpcRevisionCatalog(stub=stub)

    page = client.list_ready_revisions(MODEL_ID, page_token="opaque::v1::3", limit=25)

    request = stub.requests["ListReadyRevisions"]
    assert request.page_token == "opaque::v1::3"
    assert request.limit == 25
    assert isinstance(page, RevisionPage)
    assert page.next_page_token == "opaque::v2::7"
    assert [summary.version for summary in page.revisions] == ["v2"]


def test_list_ready_revisions_omits_absent_token_and_normalizes_end_of_pages():
    stub = _RecordingStub(
        ListReadyRevisions=revision_pb2.ListReadyRevisionsResponse(next_page_token="")
    )
    client = GrpcRevisionCatalog(stub=stub)

    page = client.list_ready_revisions(MODEL_ID)

    assert not stub.requests["ListReadyRevisions"].HasField("page_token")
    assert page.revisions == ()
    assert page.next_page_token is None


def test_get_recovery_candidates_keeps_zero_replay_length_meaningful():
    stub = _RecordingStub(
        GetRecoveryCandidates=revision_pb2.GetRecoveryCandidatesResponse(
            candidates=[
                revision_pb2.RecoveryCandidate(
                    kind=revision_pb2.RECOVERY_CANDIDATE_KIND_DIRECT_DELTA,
                    revisions=[_ready_record()],
                )
            ],
            next_page_token="",
        )
    )
    client = GrpcRevisionCatalog(stub=stub)

    page = client.get_recovery_candidates(
        MODEL_ID, target_version="v2", installed_version="v1", max_delta_replay_length=0
    )

    request = stub.requests["GetRecoveryCandidates"]
    assert request.target_version == "v2"
    assert request.HasField("installed_version")
    assert request.HasField("max_delta_replay_length")
    assert request.max_delta_replay_length == 0
    assert isinstance(page, RecoveryCandidatePage)
    assert page.next_page_token is None
    assert page.candidates[0].kind is RecoveryCandidateKind.DIRECT_DELTA


def test_get_recovery_candidates_omits_unknown_installed_version():
    stub = _RecordingStub(
        GetRecoveryCandidates=revision_pb2.GetRecoveryCandidatesResponse(next_page_token="")
    )
    client = GrpcRevisionCatalog(stub=stub)

    client.get_recovery_candidates(MODEL_ID, target_version="v2")

    request = stub.requests["GetRecoveryCandidates"]
    assert not request.HasField("installed_version")
    assert not request.HasField("max_delta_replay_length")


def test_update_receiver_state_maps_report_and_record():
    stub = _RecordingStub(
        UpdateReceiverState=revision_pb2.UpdateReceiverStateResponse(
            receiver=revision_pb2.ReceiverStateRecord(
                model_id=MODEL_ID,
                version="v2",
                receiver_id="rollout-tp0",
                state=revision_pb2.RECEIVER_REVISION_STATE_VERIFIED,
                installed_version="v2",
                detail="device verified",
                observed_at_unix_ms=21,
            )
        )
    )
    client = GrpcRevisionCatalog(stub=stub)

    record = client.update_receiver_state(
        MODEL_ID,
        "v2",
        "rollout-tp0",
        ReceiverRevisionState.VERIFIED,
        installed_version="v2",
        detail="device verified",
    )

    request = stub.requests["UpdateReceiverState"]
    assert request.state == revision_pb2.RECEIVER_REVISION_STATE_VERIFIED
    assert request.HasField("installed_version")
    assert record == ReceiverStateRecord(
        model_id=MODEL_ID,
        version="v2",
        receiver_id="rollout-tp0",
        state=ReceiverRevisionState.VERIFIED,
        installed_version="v2",
        detail="device verified",
        observed_at_unix_ms=21,
    )


def test_update_receiver_state_omits_unknown_installed_version():
    stub = _RecordingStub(
        UpdateReceiverState=revision_pb2.UpdateReceiverStateResponse(
            receiver=revision_pb2.ReceiverStateRecord(
                model_id=MODEL_ID,
                version="v2",
                receiver_id="rollout-tp0",
                state=revision_pb2.RECEIVER_REVISION_STATE_POISONED,
            )
        )
    )
    client = GrpcRevisionCatalog(stub=stub)

    record = client.update_receiver_state(
        MODEL_ID, "v2", "rollout-tp0", ReceiverRevisionState.POISONED
    )

    assert not stub.requests["UpdateReceiverState"].HasField("installed_version")
    assert record.installed_version is None


def test_commit_version_sends_only_model_and_version():
    committed = _ready_record()
    committed.state = revision_pb2.REVISION_LIFECYCLE_STATE_COMMITTED
    stub = _RecordingStub(CommitVersion=revision_pb2.CommitVersionResponse(revision=committed))
    client = GrpcRevisionCatalog(stub=stub)

    record = client.commit_version(MODEL_ID, "v2")

    request = stub.requests["CommitVersion"]
    assert [field.name for field, _ in request.ListFields()] == ["model_id", "version"]
    assert record.state is RevisionLifecycleState.COMMITTED


@pytest.mark.parametrize(
    ("rpc", "response", "call"),
    [
        (
            "PublishRevision",
            revision_pb2.PublishRevisionResponse(created=True),
            lambda client: client.publish_revision(_manifest(), publisher_id="trainer-0"),
        ),
        (
            "GetRevision",
            revision_pb2.GetRevisionResponse(),
            lambda client: client.get_revision(MODEL_ID, "v2"),
        ),
        (
            "UpdateReceiverState",
            revision_pb2.UpdateReceiverStateResponse(),
            lambda client: client.update_receiver_state(
                MODEL_ID, "v2", "rollout-tp0", ReceiverRevisionState.VERIFIED
            ),
        ),
        (
            "CommitVersion",
            revision_pb2.CommitVersionResponse(),
            lambda client: client.commit_version(MODEL_ID, "v2"),
        ),
    ],
)
def test_malformed_success_responses_are_rejected(rpc, response, call):
    client = GrpcRevisionCatalog(stub=_RecordingStub(**{rpc: response}))

    with pytest.raises(CatalogProtocolError):
        call(client)


def test_catalog_boundary_exposes_exactly_the_six_public_rpcs():
    protocol_methods = tuple(
        name
        for name, member in vars(RevisionCatalog).items()
        if not name.startswith("_") and inspect.isfunction(member)
    )

    assert protocol_methods == CATALOG_RPCS
    assert all(hasattr(GrpcRevisionCatalog, name) for name in CATALOG_RPCS)
    assert isinstance(GrpcRevisionCatalog(stub=_RecordingStub()), RevisionCatalog)


def test_client_carries_no_lifecycle_policy_or_backend_knowledge():
    source = inspect.getsource(catalog_module).lower()

    for forbidden in ("redis", "wait_for_commit", "continue_after_ready", "time.sleep"):
        assert forbidden not in source

    public_api = {
        name
        for name in vars(GrpcRevisionCatalog)
        if not name.startswith("_") and callable(getattr(GrpcRevisionCatalog, name))
    }
    assert public_api == {*CATALOG_RPCS, "close"}


class _InProcessCatalogServicer(revision_pb2_grpc.RevisionCatalogServiceServicer):
    """Deterministic in-process catalog; private to this test module."""

    def __init__(self) -> None:
        self.revisions: dict[tuple[str, str], revision_pb2.RevisionRecord] = {}
        self.receivers: list[revision_pb2.ReceiverStateRecord] = []
        self._clock = 0

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def PublishRevision(self, request, context):
        key = (request.manifest.model_id, request.manifest.version)
        existing = self.revisions.get(key)
        if existing is not None:
            if existing.manifest != request.manifest:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, "conflicting manifest")
            return revision_pb2.PublishRevisionResponse(revision=existing, created=False)
        now = self._tick()
        record = revision_pb2.RevisionRecord(
            manifest=request.manifest,
            state=revision_pb2.REVISION_LIFECYCLE_STATE_READY,
            created_at_unix_ms=now,
            state_changed_at_unix_ms=now,
        )
        self.revisions[key] = record
        return revision_pb2.PublishRevisionResponse(revision=record, created=True)

    def GetRevision(self, request, context):
        record = self.revisions.get((request.model_id, request.version))
        if record is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "unknown revision")
        return revision_pb2.GetRevisionResponse(revision=record)

    def ListReadyRevisions(self, request, context):
        summaries = [
            revision_pb2.RevisionSummary(
                model_id=record.manifest.model_id,
                version=record.manifest.version,
                state=record.state,
                ready_at_unix_ms=record.created_at_unix_ms,
            )
            for (model_id, _), record in sorted(self.revisions.items())
            if model_id == request.model_id
        ]
        return revision_pb2.ListReadyRevisionsResponse(revisions=summaries, next_page_token="")

    def GetRecoveryCandidates(self, request, context):
        record = self.revisions.get((request.model_id, request.target_version))
        candidates = []
        if record is not None and record.manifest.base_version == request.installed_version:
            candidates.append(
                revision_pb2.RecoveryCandidate(
                    kind=revision_pb2.RECOVERY_CANDIDATE_KIND_DIRECT_DELTA,
                    revisions=[record],
                )
            )
        return revision_pb2.GetRecoveryCandidatesResponse(
            candidates=candidates, next_page_token=""
        )

    def UpdateReceiverState(self, request, context):
        record = revision_pb2.ReceiverStateRecord(
            model_id=request.model_id,
            version=request.version,
            receiver_id=request.receiver_id,
            state=request.state,
            detail=request.detail,
            observed_at_unix_ms=self._tick(),
        )
        if request.HasField("installed_version"):
            record.installed_version = request.installed_version
        self.receivers.append(record)
        return revision_pb2.UpdateReceiverStateResponse(receiver=record)

    def CommitVersion(self, request, context):
        record = self.revisions.get((request.model_id, request.version))
        if record is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "unknown revision")
        if record.state != revision_pb2.REVISION_LIFECYCLE_STATE_COMMITTED:
            record.state = revision_pb2.REVISION_LIFECYCLE_STATE_COMMITTED
            record.state_changed_at_unix_ms = self._tick()
        return revision_pb2.CommitVersionResponse(revision=record)


@pytest.fixture
def in_process_catalog():
    servicer = _InProcessCatalogServicer()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    revision_pb2_grpc.add_RevisionCatalogServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with GrpcRevisionCatalog(f"http://127.0.0.1:{port}") as client:
            yield client, servicer
    finally:
        server.stop(None)


def test_publish_list_report_and_commit_against_an_in_process_server(in_process_catalog):
    client, servicer = in_process_catalog
    manifest = _manifest()

    # The in-process servicer is deliberately not a second copy of the shared
    # validator, so pin the one field format it cannot catch: a real catalog
    # rejects any dirty-delta checksum that is not eight lowercase hex digits.
    assert re.fullmatch(r"[0-9a-f]{8}", manifest.ranks[0].delta.checksum)

    published = client.publish_revision(manifest, publisher_id="trainer-0")
    retried = client.publish_revision(manifest, publisher_id="trainer-0")

    assert published.created is True
    assert published.revision.state is RevisionLifecycleState.READY
    assert retried.created is False
    assert retried.revision == published.revision

    assert client.get_revision(MODEL_ID, "v2").manifest == manifest
    page = client.list_ready_revisions(MODEL_ID)
    assert [summary.version for summary in page.revisions] == ["v2"]
    assert page.next_page_token is None

    report = client.update_receiver_state(
        MODEL_ID, "v2", "rollout-tp0", ReceiverRevisionState.VERIFIED, installed_version="v2"
    )
    assert report.state is ReceiverRevisionState.VERIFIED
    assert client.get_revision(MODEL_ID, "v2").state is RevisionLifecycleState.READY

    committed = client.commit_version(MODEL_ID, "v2")
    recommitted = client.commit_version(MODEL_ID, "v2")
    assert committed.state is RevisionLifecycleState.COMMITTED
    assert recommitted == committed
    assert len(servicer.receivers) == 1


def test_conflicting_manifest_retry_surfaces_the_server_error(in_process_catalog):
    client, _ = in_process_catalog
    client.publish_revision(_manifest(), publisher_id="trainer-0")

    conflicting = dataclasses.replace(_manifest(), target_digest="sha256:other")

    with pytest.raises(grpc.RpcError) as excinfo:
        client.publish_revision(conflicting, publisher_id="trainer-0")

    assert excinfo.value.code() is grpc.StatusCode.FAILED_PRECONDITION


def test_recovery_candidates_round_trip_over_the_wire(in_process_catalog):
    client, _ = in_process_catalog
    client.publish_revision(_manifest(), publisher_id="trainer-0")

    page = client.get_recovery_candidates(
        MODEL_ID, target_version="v2", installed_version="v1", max_delta_replay_length=1
    )

    assert page.candidates[0].kind is RecoveryCandidateKind.DIRECT_DELTA
    assert page.candidates[0].revisions[0].manifest == _manifest()
    assert page.next_page_token is None
