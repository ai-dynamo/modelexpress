# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections.abc import Iterator
from concurrent import futures
from dataclasses import replace
from types import SimpleNamespace

import grpc
import pytest

from modelexpress_rl.refit.timing import RefitTimingRecorder, use_refit_timing
import modelexpress_rl.inference.source.trainer as trainer_source_module

from modelexpress_rl import WeightPayloadFormat, refit_pb2, refit_pb2_grpc
from modelexpress_rl.control import WeightVersion, WeightVersionState
from modelexpress_rl.inference.source import TrainerSourceResolver
from tests.test_refit_generator_client import _RefitService, _STABLE_METADATA


class _Worker(refit_pb2_grpc.RefitWorkerServiceServicer):
    def __init__(self) -> None:
        self.blob = _STABLE_METADATA
        self.stable_requests = []
        self.version_requests = []
        self.metadata = refit_pb2.WeightVersionShardMetadata(
            version_id="version-a", worker_id="trainer-0", logical_shard_id="rank:0",
            stable_metadata_digest=hashlib.sha256(self.blob).hexdigest(),
            checksums=[refit_pb2.ShardChecksum(tensor_name="weight", digest="first")],
        )
        self.response_digest = ""
        self.blobs = {}
        self.clock = None
        self.stable_failures = 0
        self.stable_failure_code = grpc.StatusCode.UNAVAILABLE
        self.version_failures = 0

    def GetTrainerShardMetadata(
        self, request: refit_pb2.GetTrainerShardMetadataRequest, context: grpc.ServicerContext,
    ) -> refit_pb2.GetTrainerShardMetadataResponse:
        self.stable_requests.append(request)
        if self.clock is not None:
            self.clock.now += 2
        if self.stable_failures:
            self.stable_failures -= 1
            context.abort(self.stable_failure_code, "stable metadata unavailable")
        blob = self.blobs.get(request.metadata_digest, self.blob)
        return refit_pb2.GetTrainerShardMetadataResponse(
            metadata=blob, metadata_digest=hashlib.sha256(blob).hexdigest(),
        )

    def GetWeightVersionShardMetadata(
        self, request: refit_pb2.GetWeightVersionShardMetadataRequest,
        context: grpc.ServicerContext,
    ) -> refit_pb2.GetWeightVersionShardMetadataResponse:
        self.version_requests.append(request)
        if self.clock is not None:
            self.clock.now += 3
        if self.version_failures:
            self.version_failures -= 1
            context.abort(grpc.StatusCode.UNAVAILABLE, "version metadata unavailable")
        return refit_pb2.GetWeightVersionShardMetadataResponse(
            metadata=self.metadata, version_metadata_digest=self.response_digest,
        )


def _advertise(case: SimpleNamespace) -> None:
    case.control.shards[0].stable_metadata_digest = hashlib.sha256(case.worker.blob).hexdigest()
    digest = hashlib.sha256(
        case.worker.metadata.SerializeToString(deterministic=True)
    ).hexdigest()
    case.worker.response_digest = digest
    case.control.shards[0].version_metadata_digest = digest


@pytest.fixture
def protocol(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    port = server.add_insecure_port("127.0.0.1:0")
    control = _RefitService(endpoint=f"127.0.0.1:{port}")
    control.shards = control.shards[:1]
    worker = _Worker()
    refit_pb2_grpc.add_RefitServiceServicer_to_server(control, server)
    refit_pb2_grpc.add_RefitWorkerServiceServicer_to_server(worker, server)
    server.start()
    with grpc.insecure_channel(f"127.0.0.1:{port}") as channel:
        service = refit_pb2_grpc.RefitServiceStub(channel)
        case = SimpleNamespace(
            control=control, worker=worker, cached_snapshot=None,
            version=WeightVersion(
                version_id="version-a", model_name="test/model",
                payload_format=WeightPayloadFormat.FULL_TENSOR, layout_signature="layout-a",
                state=WeightVersionState.READY, created_at_unix_ms=0,
                trainer_mesh_generation=1, trainer_mesh_id="mesh-a",
            ),
        )
        case.resolver = TrainerSourceResolver(
            service=lambda: service, rpc_timeout_seconds=2,
            cached_source=lambda: case.cached_snapshot,
        )
        _advertise(case)
        try:
            yield case
        finally:
            server.stop(grace=None).wait()


@pytest.mark.parametrize("field", [
    "version_id", "worker_id", "logical_shard_id", "stable_metadata_digest",
])
def test_version_record_identity_mismatch_has_no_transfer_candidate(
    protocol: SimpleNamespace, field: str,
) -> None:
    setattr(protocol.worker.metadata, field, "different")
    _advertise(protocol)
    assert list(protocol.resolver.candidates(protocol.version)) == []
    assert len(protocol.worker.version_requests) == 1


@pytest.mark.parametrize("corrupt", ["advertised_hash", "response_hash", "record_bytes"])
def test_wrong_version_record_hash_has_no_transfer_candidate(
    protocol: SimpleNamespace, corrupt: str,
) -> None:
    if corrupt == "advertised_hash":
        protocol.control.shards[0].version_metadata_digest = "wrong"
    elif corrupt == "response_hash":
        protocol.worker.response_digest = "wrong"
    else:
        protocol.worker.metadata.checksums[0].digest = "changed-without-updating-hash"
    assert list(protocol.resolver.candidates(protocol.version)) == []


@pytest.mark.parametrize("corrupt", ["advertised_hash", "response_bytes"])
def test_wrong_stable_metadata_hash_is_rejected_before_version_fetch(
    protocol: SimpleNamespace, corrupt: str,
) -> None:
    if corrupt == "advertised_hash":
        protocol.control.shards[0].stable_metadata_digest = "wrong"
    else:
        payload = json.loads(protocol.worker.blob)
        payload["metadata_endpoint"] = "another-trainer:19000"
        protocol.worker.blob = json.dumps(payload).encode()
    assert list(protocol.resolver.candidates(protocol.version)) == []
    assert protocol.worker.version_requests == []


@pytest.mark.parametrize("rows", ["duplicate", "unknown", "missing", "empty_digest"])
def test_invalid_checksum_coverage_has_no_transfer_candidate(
    protocol: SimpleNamespace, rows: str,
) -> None:
    checksums = protocol.worker.metadata.checksums
    if rows == "duplicate":
        checksums.add().CopyFrom(checksums[0])
    elif rows == "unknown":
        checksums[0].tensor_name = "unknown"
    elif rows == "missing":
        del checksums[:]
    else:
        checksums[0].digest = ""
    _advertise(protocol)
    assert list(protocol.resolver.candidates(protocol.version)) == []


@pytest.mark.parametrize(("field", "value"), [
    ("tensor_count", True), ("tensor_count", 2), ("tensor_count", 0),
    ("total_bytes", True), ("total_bytes", 9), ("total_bytes", 0),
])
def test_invalid_stable_counts_are_rejected_before_version_fetch(
    protocol: SimpleNamespace, field: str, value: int | bool,
) -> None:
    payload = json.loads(protocol.worker.blob)
    payload[field] = value
    protocol.worker.blob = json.dumps(payload).encode()
    _advertise(protocol)
    assert list(protocol.resolver.candidates(protocol.version)) == []
    assert protocol.worker.version_requests == []


def test_duplicate_physical_checksum_keys_are_rejected_before_version_fetch(
    protocol: SimpleNamespace,
) -> None:
    payload = json.loads(protocol.worker.blob)
    payload["tensors"][0]["shards"].append(dict(payload["tensors"][0]["shards"][0]))
    payload["total_bytes"] = 16
    protocol.worker.blob = json.dumps(payload).encode()
    _advertise(protocol)
    assert list(protocol.resolver.candidates(protocol.version)) == []
    assert protocol.worker.version_requests == []


def test_coverage_only_metadata_cannot_resolve_physical_reads(protocol: SimpleNamespace) -> None:
    protocol.worker.blob = json.dumps({"tensors": [{
        "name": "weight", "dtype": "torch.bfloat16", "elsize": 2,
        "full_shape": [4], "shards": [{"shard_offset": [0], "shape": [4]}],
    }]}).encode()
    _advertise(protocol)
    assert list(protocol.resolver.candidates(protocol.version)) == []
    assert protocol.worker.version_requests == []


def _next_version(protocol: SimpleNamespace) -> WeightVersion:
    protocol.worker.metadata.version_id = "version-b"
    protocol.worker.metadata.checksums[0].digest = "second"
    protocol.control.shards[0].version_id = "version-b"
    protocol.control.version.uid = "version-b"
    _advertise(protocol)
    return replace(protocol.version, version_id="version-b")


@pytest.mark.parametrize("verify", [False, True])
def test_new_version_resolves_fresh_checksums_without_refetching_stable_metadata(
    protocol: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, verify: bool, discovery_timing,
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", str(int(verify)))
    if not verify:
        protocol.control.shards[0].version_metadata_digest = ""
    with use_refit_timing(discovery_timing.recorder):
        first, = protocol.resolver.candidates(protocol.version)
        protocol.cached_snapshot = first.snapshot
        second, = protocol.resolver.candidates(_next_version(protocol))
    stage = discovery_timing.recorder.as_dict()["stages"]["source_preparation"]
    assert stage["duration_ms"] == (8000 if verify else 2000)
    assert second.snapshot.shards[0].metadata == protocol.worker.blob
    assert list(first.checksums.values()) == (["first"] if verify else [])
    assert list(second.checksums.values()) == (["second"] if verify else [])
    assert len(protocol.worker.stable_requests) == 1

    assert [request.version_id for request in protocol.worker.version_requests] == (
        ["version-a", "version-b"] if verify else []
    )


@pytest.mark.parametrize("change", ["replica", "generation"])
def test_changed_source_resolves_current_storage(protocol, change) -> None:
    first, = protocol.resolver.candidates(protocol.version)
    protocol.cached_snapshot = first.snapshot
    generation = 2
    worker = "trainer-replacement" if change == "replica" else "trainer-0"
    protocol.control.shards[0].worker_id = worker
    protocol.worker.metadata.worker_id = worker
    protocol.control.mesh_generation_on_recheck = generation
    payload = json.loads(protocol.worker.blob)
    payload["tensors"][0]["shards"][0]["addr"] = 8192
    protocol.worker.blob = json.dumps(payload).encode()
    protocol.worker.metadata.stable_metadata_digest = hashlib.sha256(protocol.worker.blob).hexdigest()
    requested = replace(_next_version(protocol), trainer_mesh_generation=generation)
    second, = protocol.resolver.candidates(requested)
    assert second.snapshot.replica_groups is not first.snapshot.replica_groups
    assert second.snapshot.mesh_generation == generation
    assert second.snapshot.shards[0].worker_id == worker
    assert second.snapshot.shards[0].metadata == protocol.worker.blob
    assert list(second.checksums.values()) == ["second"]
    assert next(iter(second.checksums))[2] == 8192
    assert len(protocol.worker.stable_requests) == 2


def test_requested_version_generation_mismatch_rejects_cached_selection(
    protocol: SimpleNamespace,
) -> None:
    first, = protocol.resolver.candidates(protocol.version)
    protocol.cached_snapshot = first.snapshot
    protocol.control.mesh_generation_on_recheck = 2
    with pytest.raises(RuntimeError, match="generation"):
        next(protocol.resolver.candidates(_next_version(protocol)))
    assert len(protocol.worker.stable_requests) == 1
    assert len(protocol.worker.version_requests) == 1


@pytest.fixture
def discovery_timing(
    protocol: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    clock = SimpleNamespace(now=0.0)
    protocol.worker.clock = clock
    read_clock = lambda: clock.now
    monkeypatch.setattr(trainer_source_module, "perf_counter", read_clock)
    recorder = RefitTimingRecorder(backend="test", version="version-a", clock=read_clock)
    return SimpleNamespace(clock=clock, recorder=recorder)


@pytest.mark.parametrize("failure", ["stable", "version"])
@pytest.mark.parametrize("fallback", [False, True])
def test_discovery_reports_failed_rpc_time_and_can_use_another_replica(
    protocol: SimpleNamespace, discovery_timing: SimpleNamespace,
    failure: str, fallback: bool,
) -> None:
    if fallback:
        alternative = refit_pb2.WeightVersionShard()
        alternative.CopyFrom(protocol.control.shards[0])
        alternative.worker_id = "trainer-b"
        protocol.worker.metadata.worker_id = "trainer-b"
        _advertise(protocol)
        alternative.version_metadata_digest = protocol.control.shards[0].version_metadata_digest
        protocol.control.shards.append(alternative)
    setattr(protocol.worker, f"{failure}_failures", 1)
    with use_refit_timing(discovery_timing.recorder):
        candidates = list(protocol.resolver.candidates(protocol.version))
    if fallback:
        assert candidates[0].snapshot.shards[0].worker_id == "trainer-b"
        assert list(candidates[0].checksums.values()) == ["first"]
    else:
        assert candidates == []
    stage = discovery_timing.recorder.as_dict()["stages"]["source_preparation"]
    assert stage["duration_ms"] == (2000 if failure == "stable" else 5000) + 5000 * fallback


@pytest.mark.parametrize("resume", [False, True])
def test_lazy_discovery_timing_excludes_consumer_pause(
    protocol: SimpleNamespace, discovery_timing: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch, resume: bool,
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    alternative = refit_pb2.WeightVersionShard()
    alternative.CopyFrom(protocol.control.shards[0])
    alternative.worker_id = "trainer-b"
    protocol.control.shards.append(alternative)
    with use_refit_timing(discovery_timing.recorder):
        candidates = protocol.resolver.candidates(protocol.version)
        next(candidates)
        discovery_timing.clock.now += 100
        if resume:
            next(candidates)
        discovery_timing.clock.now += 100
        candidates.close()
    stage = discovery_timing.recorder.as_dict()["stages"]["source_preparation"]
    assert stage["duration_ms"] == (4000 if resume else 2000)


@pytest.fixture
def replicas(protocol, monkeypatch):
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    first = protocol.control.shards[0]
    first.worker_id = "trainer-a"
    second = refit_pb2.WeightVersionShard()
    second.CopyFrom(first)
    second.worker_id = "trainer-b"
    protocol.control.shards.append(second)
    return first, second


def test_replica_groups_include_unpublished_members_in_worker_order(
    protocol: SimpleNamespace, replicas,
) -> None:
    expected_members = ["trainer-a", "trainer-b", "trainer-unpublished"]
    expected_worker = "trainer-a"
    first, second = replicas
    mesh = refit_pb2.TrainerMesh(
        mesh_id="mesh-a", model_name="test/model", generation=1,
        workers={
            worker_id: refit_pb2.TrainerTensorsMetadata(
                logical_shard_id="rank:0", metadata_endpoint=first.metadata_endpoint,
            )
            for worker_id in ("trainer-a", "trainer-b", "trainer-unpublished")
        },
    )
    service = SimpleNamespace(
        GetTrainerMesh=lambda *_args, **_kwargs: refit_pb2.GetTrainerMeshResponse(mesh=mesh),
        ListWeightVersionShards=lambda *_args, **_kwargs: (
            refit_pb2.ListWeightVersionShardsResponse(shards=protocol.control.shards)
        ),
    )
    protocol.resolver = TrainerSourceResolver(
        service=lambda: service,
        rpc_timeout_seconds=2,
        cached_source=lambda: protocol.cached_snapshot,
    )

    candidates = protocol.resolver.candidates(protocol.version)
    resolved = next(candidates)
    retry = next(candidates)
    assert retry.snapshot.replica_groups is resolved.snapshot.replica_groups
    assert retry.snapshot.shards[0].worker_id == "trainer-b"
    candidates.close()

    assert resolved.snapshot.shards[0].worker_id == expected_worker
    group, = resolved.snapshot.replica_groups
    assert group.source_slot_id == "rank:0"
    assert [replica.worker_id for replica in group.replicas] == expected_members
    assert [replica.metadata is not None for replica in group.replicas] == [
        worker_id != "trainer-unpublished" for worker_id in expected_members
    ]

    protocol.cached_snapshot = resolved.snapshot
    first.version_id = "version-b"
    protocol.control.shards = [first]
    warm = next(protocol.resolver.candidates(replace(protocol.version, version_id="version-b")))
    assert warm.snapshot.shards[0].worker_id == "trainer-a"
    assert warm.snapshot.replica_groups is resolved.snapshot.replica_groups


def test_prepare_groups_fetches_metadata_only_for_requested_groups(
    protocol: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, replicas,
) -> None:
    monkeypatch.setenv("MX_REFIT_PREFETCH_REPLICA_METADATA", "1")
    first, second = replicas

    resolved = next(protocol.resolver.candidates(protocol.version))
    before = len(protocol.worker.stable_requests)
    assert resolved.prepare_groups is not None
    group, = resolved.snapshot.replica_groups
    assert group.replicas[1].metadata is None
    resolved.prepare_groups(frozenset({"rank:missing"}))
    assert len(protocol.worker.stable_requests) == before
    resolved.prepare_groups(frozenset({"rank:0"}))
    assert group.replicas[1].metadata == protocol.worker.blob
    assert len(protocol.worker.stable_requests) == before + 1
    assert protocol.worker.version_requests == []


@pytest.mark.parametrize("difference", ["coverage", "placement"])
@pytest.mark.parametrize("debug", [False, True])
def test_prepare_groups_checks_replica_coverage_only_in_debug_mode(
    protocol: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, replicas,
    debug: bool, difference: str,
) -> None:
    monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_REPLICAS", str(int(debug)))
    monkeypatch.setenv("MX_REFIT_PREFETCH_REPLICA_METADATA", "0")
    first, second = replicas
    alternate = json.loads(protocol.worker.blob)
    if difference == "coverage":
        alternate["tensors"][0]["name"] = "different-weight"
    else:
        alternate["tensors"][0]["shards"][0]["agent_name"] = "other-agent"
        alternate["tensors"][0]["shards"][0]["addr"] += 4096
    alternate_blob = json.dumps(alternate).encode()
    second.stable_metadata_digest = hashlib.sha256(alternate_blob).hexdigest()
    protocol.worker.blobs[second.stable_metadata_digest] = alternate_blob

    resolved = next(protocol.resolver.candidates(protocol.version))
    assert resolved.prepare_groups is not None
    before = len(protocol.worker.stable_requests)
    if debug:
        if difference == "coverage":
            with pytest.raises(RuntimeError, match="replicas disagree on tensor coverage"):
                resolved.prepare_groups(frozenset({"rank:0"}))
        else:
            resolved.prepare_groups(frozenset({"rank:0"}))
        assert len(protocol.worker.stable_requests) == before + 1
    else:
        resolved.prepare_groups(frozenset({"rank:0"}))
        assert len(protocol.worker.stable_requests) == before


@pytest.mark.parametrize("failure", ["unavailable", "debug_unavailable", "digest", "permission"])
def test_optional_prefetch_only_tolerates_unavailable_replica(
    protocol: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, replicas, failure: str,
) -> None:
    monkeypatch.setenv("MX_REFIT_PREFETCH_REPLICA_METADATA", "1")
    first, second = replicas
    alternate = json.loads(protocol.worker.blob)
    alternate["tensors"][0]["shards"][0]["addr"] += 4096
    alternate_blob = json.dumps(alternate).encode()
    second.stable_metadata_digest = hashlib.sha256(alternate_blob).hexdigest()
    protocol.worker.blobs[second.stable_metadata_digest] = alternate_blob
    if failure == "debug_unavailable":
        monkeypatch.setenv("MX_REFIT_DEBUG_VALIDATE_REPLICAS", "1")
    resolved = next(protocol.resolver.candidates(protocol.version))
    assert resolved.prepare_groups is not None
    group, = resolved.snapshot.replica_groups

    if failure in {"unavailable", "debug_unavailable"}:
        protocol.worker.stable_failures = 1
    elif failure == "digest":
        protocol.worker.blobs[second.stable_metadata_digest] = b"wrong metadata"
    else:
        protocol.worker.stable_failures = 1
        protocol.worker.stable_failure_code = grpc.StatusCode.PERMISSION_DENIED
    if failure == "unavailable":
        resolved.prepare_groups(frozenset({"rank:0"}))
        assert group.replicas[1].metadata is None
        resolved.prepare_groups(frozenset({"rank:0"}))
        assert group.replicas[1].metadata == alternate_blob
    else:
        with pytest.raises((RuntimeError, grpc.RpcError)):
            resolved.prepare_groups(frozenset({"rank:0"}))
        assert group.replicas[1].metadata is None


def test_warm_resolution_rechecks_mesh_after_listing_without_checksums(
    protocol: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    first = next(protocol.resolver.candidates(protocol.version))
    protocol.cached_snapshot = first.snapshot
    version = _next_version(protocol)
    service = protocol.resolver._service()

    def change_generation(request, **kwargs) -> refit_pb2.ListWeightVersionShardsResponse:
        response = service.ListWeightVersionShards(request, **kwargs)
        protocol.control.mesh_generation_on_recheck = 2
        return response

    monkeypatch.setattr(protocol.resolver, "_service", lambda: SimpleNamespace(
        GetTrainerMesh=service.GetTrainerMesh,
        ListWeightVersionShards=change_generation,
    ))
    with pytest.raises(RuntimeError, match="generation changed"):
        next(protocol.resolver.candidates(version))
