# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from concurrent import futures

import grpc
import modelexpress_rl.inference.client as generator_client_module
import pytest
from modelexpress_rl import (
    GeneratorInstallationMode,
    ModelExpressGeneratorClient,
    ModelExpressGeneratorConfig,
    VllmGeneratorContext,
    WeightPayloadFormat,
    WeightVersionRef,
    refit_pb2,
    refit_pb2_grpc,
)
from modelexpress_rl.inference import GeneratorEngineAdapter


class _RefitService(refit_pb2_grpc.RefitServiceServicer):
    def __init__(self, *, endpoint: str, state=None, manifest_digest=None):
        self.registrations = {}
        self.active_leases = set()
        self.lease_registrations = 0
        self.lease_deletions = 0
        self.fail_lease_deletion = False
        self.version = refit_pb2.WeightVersion(
            uid="version-a",
            model_name="test/model",
            payload_format=refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_TENSOR,
            expected_source_slots=["rank:0", "rank:1"],
            layout_signature="layout-a",
            state=state or refit_pb2.WEIGHT_VERSION_STATE_READY,
        )
        digest = manifest_digest or hashlib.sha256(b"manifest").hexdigest()
        self.shards = [
            refit_pb2.WeightVersionShard(
                version_id="version-a",
                source_slot_id=slot,
                worker_id=f"trainer-{rank}",
                tensor_count=2,
                total_bytes=128,
                manifest_digest=digest,
                manifest_endpoint=endpoint,
                transport="NIXL",
            )
            for rank, slot in enumerate(self.version.expected_source_slots)
        ]

    def RegisterWorker(self, request, _context):
        worker = request.worker
        worker.expires_at_unix_ms = 1234
        self.registrations[worker.worker_id] = worker
        return worker

    def GetWeightVersion(self, request, context):
        if request.uid != self.version.uid:
            context.abort(grpc.StatusCode.NOT_FOUND, "version not found")
        return self.version

    def ListWeightVersionShards(self, request, _context):
        return refit_pb2.ListWeightVersionShardsResponse(
            shards=self.shards if request.version_id == self.version.uid else []
        )

    def RegisterVersionLease(self, request, context):
        worker = self.registrations.get(request.worker_id)
        if worker is None or worker.role != refit_pb2.WORKER_ROLE_GENERATOR:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "generator not registered"
            )
        lease_id = f"lease-{request.worker_id}"
        self.active_leases.add(lease_id)
        self.lease_registrations += 1
        return refit_pb2.VersionLease(
            lease_id=lease_id,
            version_id=request.version_id,
            worker_id=request.worker_id,
            expires_at_unix_ms=1234,
        )

    def DeleteVersionLease(self, request, context):
        if self.fail_lease_deletion:
            context.abort(grpc.StatusCode.UNAVAILABLE, "lease backend unavailable")
        deleted = request.lease_id in self.active_leases
        self.active_leases.discard(request.lease_id)
        self.lease_deletions += 1
        return refit_pb2.DeleteVersionLeaseResponse(deleted=deleted)


class _WorkerService(refit_pb2_grpc.RefitWorkerServiceServicer):
    def GetWeightVersionShardManifest(self, _request, _context):
        return refit_pb2.GetWeightVersionShardManifestResponse(
            manifest=b"manifest",
            manifest_digest=hashlib.sha256(b"manifest").hexdigest(),
        )


class _Adapter(GeneratorEngineAdapter):
    supported_installation_modes = frozenset({GeneratorInstallationMode.STAGED})
    supported_payload_formats = frozenset({WeightPayloadFormat.FULL_TENSOR})

    def __init__(self, service):
        self.service = service
        self.create_calls = []
        self.validate_calls = []
        self.stage_calls = []
        self.apply_calls = []
        self.release_calls = []
        self.close_calls = 0
        self.stage_failures = 0

    def create_transfer_plan(self, inputs):
        self.create_calls.append(inputs)
        return {"sources": inputs.sources}

    def validate_transfer_plan(self, plan, inputs):
        self.validate_calls.append((plan, inputs))
        return True

    def stage_weight(self, plan):
        assert self.service.active_leases
        self.stage_calls.append(plan)
        if self.stage_failures:
            self.stage_failures -= 1
            raise RuntimeError("transfer failed")
        return {"plan": plan}

    def apply_weight(self, staged):
        assert not self.service.active_leases
        self.apply_calls.append(staged)
        return "installed"

    def release_staged_weight(self, staged):
        self.release_calls.append(staged)

    def close(self):
        self.close_calls += 1


def _start_server(*, state=None, manifest_digest=None):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    port = server.add_insecure_port("127.0.0.1:0")
    endpoint = f"127.0.0.1:{port}"
    service = _RefitService(
        endpoint=endpoint,
        state=state,
        manifest_digest=manifest_digest,
    )
    refit_pb2_grpc.add_RefitServiceServicer_to_server(service, server)
    refit_pb2_grpc.add_RefitWorkerServiceServicer_to_server(_WorkerService(), server)
    server.start()
    return server, endpoint, service


def _initialize(monkeypatch, endpoint, adapter):
    monkeypatch.setattr(
        generator_client_module,
        "_create_generator_adapter",
        lambda **_kwargs: adapter,
    )
    return ModelExpressGeneratorClient.initialize(
        ModelExpressGeneratorConfig(
            engine_context=VllmGeneratorContext(
                model=object(),
                vllm_config=object(),
                model_config=object(),
            ),
            model_name="test/model",
            installation_mode=GeneratorInstallationMode.STAGED,
            payload_format=WeightPayloadFormat.FULL_TENSOR,
            worker_endpoint=endpoint,
            worker_id="generator-0",
            server_url=endpoint,
            registration_ttl_seconds=60,
            lease_ttl_seconds=60,
        )
    )


def test_generator_stages_applies_releases_and_reuses_valid_plan(monkeypatch):
    server, endpoint, service = _start_server()
    adapter = _Adapter(service)
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        first = generator.stage_weight(version=WeightVersionRef("version-a"))
        duplicate = generator.stage_weight(version=WeightVersionRef("version-a"))
        first.wait()

        assert duplicate is first
        assert not service.active_leases
        assert generator.apply_weight(first) == "installed"
        assert generator.apply_weight(first) == "installed"
        first.release()
        first.release()

        second = generator.stage_weight(version=WeightVersionRef("version-a"))
        second.release()

        service.shards[0].worker_id = "replacement-trainer-0"
        replacement = generator.stage_weight(version=WeightVersionRef("version-a"))
        replacement.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert service.registrations["generator-0"].role == refit_pb2.WORKER_ROLE_GENERATOR
    assert service.lease_registrations == 3
    assert service.lease_deletions == 3
    assert len(adapter.create_calls) == 2
    assert len(adapter.validate_calls) == 1
    assert len(adapter.stage_calls) == 3
    assert len(adapter.apply_calls) == 1
    assert len(adapter.release_calls) == 3
    assert adapter.close_calls == 1
    assert [source.source_slot_id for source in adapter.create_calls[0].sources] == [
        "rank:0",
        "rank:1",
    ]


def test_generator_releases_lease_when_manifest_is_invalid(monkeypatch):
    server, endpoint, service = _start_server(manifest_digest="bad-digest")
    adapter = _Adapter(service)
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        with pytest.raises(RuntimeError, match=r"no usable source.*digest mismatch"):
            generator.stage_weight(version=WeightVersionRef("version-a"))
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert not service.active_leases
    assert service.lease_registrations == 1
    assert service.lease_deletions == 1
    assert adapter.create_calls == []
    assert adapter.stage_calls == []


def test_generator_retries_complete_staged_transfer_under_one_lease(monkeypatch):
    server, endpoint, service = _start_server()
    adapter = _Adapter(service)
    adapter.stage_failures = 1
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        staged = generator.stage_weight(version=WeightVersionRef("version-a"))
        staged.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert service.lease_registrations == 1
    assert service.lease_deletions == 1
    assert len(adapter.stage_calls) == 2
    assert len(adapter.create_calls) == 2


def test_generator_preserves_transfer_error_when_lease_cleanup_also_fails(
    monkeypatch,
):
    server, endpoint, service = _start_server()
    service.fail_lease_deletion = True
    adapter = _Adapter(service)
    adapter.stage_failures = 3
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        with pytest.raises(RuntimeError, match="transfer failed"):
            generator.stage_weight(version=WeightVersionRef("version-a"))
    finally:
        generator.close()
        server.stop(grace=None).wait()


def test_generator_reports_lease_cleanup_failure_after_success(monkeypatch):
    server, endpoint, service = _start_server()
    service.fail_lease_deletion = True
    adapter = _Adapter(service)
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        with pytest.raises(grpc.RpcError, match="lease backend unavailable"):
            generator.stage_weight(version=WeightVersionRef("version-a"))
    finally:
        generator.close()
        server.stop(grace=None).wait()


def test_generator_rejects_non_ready_version_before_leasing(monkeypatch):
    server, endpoint, service = _start_server(
        state=refit_pb2.WEIGHT_VERSION_STATE_STAGING
    )
    adapter = _Adapter(service)
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        with pytest.raises(RuntimeError, match="is not READY"):
            generator.stage_weight(version=WeightVersionRef("version-a"))
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert service.lease_registrations == 0
    assert service.lease_deletions == 0


def test_generator_rejects_direct_until_implemented(monkeypatch):
    server, endpoint, service = _start_server()
    adapter = _Adapter(service)
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        with pytest.raises(NotImplementedError, match="DIRECT installation"):
            generator.update_weight(version=WeightVersionRef("version-a"))
    finally:
        generator.close()
        server.stop(grace=None).wait()


def test_generator_closes_adapter_when_registration_fails(monkeypatch):
    service = _RefitService(endpoint="unused")
    adapter = _Adapter(service)
    monkeypatch.setattr(
        generator_client_module,
        "_create_generator_adapter",
        lambda **_kwargs: adapter,
    )
    monkeypatch.setattr(
        ModelExpressGeneratorClient,
        "_register_worker",
        lambda _self: (_ for _ in ()).throw(RuntimeError("registration failed")),
    )

    with pytest.raises(RuntimeError, match="registration failed"):
        ModelExpressGeneratorClient.initialize(
            ModelExpressGeneratorConfig(
                engine_context=VllmGeneratorContext(
                    model=object(),
                    vllm_config=object(),
                    model_config=object(),
                ),
                model_name="test/model",
                installation_mode=GeneratorInstallationMode.STAGED,
                payload_format=WeightPayloadFormat.FULL_TENSOR,
                worker_endpoint="generator:9000",
                worker_id="generator-0",
                server_url="mx-server:9000",
            )
        )

    assert adapter.close_calls == 1
