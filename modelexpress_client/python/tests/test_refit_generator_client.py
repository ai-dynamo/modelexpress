# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from concurrent import futures

import grpc
import modelexpress_rl.inference.client as generator_client_module
import pytest
from modelexpress_rl import (
    ModelExpressGeneratorClient,
    ModelExpressGeneratorConfig,
    S3GeneratorConfig,
    VllmGeneratorContext,
    WeightPayloadFormat,
    WeightVersionRef,
    refit_pb2,
    refit_pb2_grpc,
)
from modelexpress_rl.inference.adapter import GeneratorEngineAdapter, S3GeneratorSource


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
        self.base = refit_pb2.WeightVersion(
            uid="base-a",
            model_name="test/model",
            payload_format=refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_TENSOR,
            state=refit_pb2.WEIGHT_VERSION_STATE_READY,
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
                nixl=refit_pb2.NixlTransport(
                    manifest_endpoint=endpoint,
                ),
            )
            for rank, slot in enumerate(self.version.expected_source_slots)
        ]

    def RegisterWorker(self, request, _context):
        worker = request.worker
        worker.expires_at_unix_ms = 1234
        self.registrations[worker.worker_id] = worker
        return worker

    def GetWeightVersion(self, request, context):
        if request.uid == self.base.uid:
            return self.base
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
    supported_payload_formats = frozenset(
        {WeightPayloadFormat.FULL_TENSOR, WeightPayloadFormat.XOR_DELTA}
    )

    def __init__(self, service):
        self.service = service
        self.stage_calls = []
        self.apply_calls = []
        self.release_calls = []
        self.close_calls = 0
        self.stage_failures = 0
        self.apply_failure = False

    def stage_weight(self, inputs):
        assert self.service.active_leases
        self.stage_calls.append(inputs)
        if self.stage_failures:
            self.stage_failures -= 1
            raise RuntimeError("transfer failed")
        return {"inputs": inputs}

    def apply_weight(self, staged):
        assert self.service.active_leases
        self.apply_calls.append(staged)
        if self.apply_failure:
            raise RuntimeError("apply failed")
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


def _initialize(
    monkeypatch,
    endpoint,
    adapter,
    payload_format=WeightPayloadFormat.FULL_TENSOR,
):
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
            ),
            model_name="test/model",
            payload_format=payload_format,
            worker_id="generator-0",
            server_url=endpoint,
            registration_ttl_seconds=60,
            lease_ttl_seconds=60,
            s3=(
                S3GeneratorConfig(
                    initial_base_version_id="base-a",
                    launch_checkpoint="unused-launch",
                    preparation_cache_dir="unused-cache",
                )
                if payload_format is WeightPayloadFormat.XOR_DELTA
                else None
            ),
        )
    )


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("registration_ttl_seconds", 0, "registration_ttl_seconds must be positive"),
        ("lease_ttl_seconds", -1, "lease_ttl_seconds must be positive"),
        ("max_transfer_attempts", 0, "max_transfer_attempts must be positive"),
        (
            "rpc_timeout_seconds",
            float("inf"),
            "rpc_timeout_seconds must be finite and positive",
        ),
    ],
)
def test_generator_config_rejects_invalid_numeric_settings(setting, value, message):
    with pytest.raises(ValueError, match=message):
        ModelExpressGeneratorConfig(
            engine_context=VllmGeneratorContext(
                model=object(),
                vllm_config=object(),
            ),
            **{setting: value},
        )


def test_generator_config_rejects_unspecified_payload_format():
    with pytest.raises(ValueError, match="payload_format must be specified"):
        ModelExpressGeneratorConfig(
            engine_context=VllmGeneratorContext(
                model=object(),
                vllm_config=object(),
            ),
            payload_format=WeightPayloadFormat.UNSPECIFIED,
        )


def test_generator_stages_applies_and_releases(monkeypatch):
    server, endpoint, service = _start_server()
    adapter = _Adapter(service)
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        first = generator.stage_weight(version=WeightVersionRef("version-a"))
        duplicate = generator.stage_weight(version=WeightVersionRef("version-a"))
        assert duplicate is first
        assert service.active_leases
        assert generator.apply_weight(first) == "installed"
        assert generator.apply_weight(first) == "installed"
        assert not service.active_leases
        first.release()
        first.release()
        assert not service.active_leases

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
    assert len(adapter.stage_calls) == 3
    assert len(adapter.apply_calls) == 1
    assert len(adapter.release_calls) == 3
    assert adapter.close_calls == 1
    assert [source.source_slot_id for source in adapter.stage_calls[0].sources] == [
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
    assert adapter.stage_calls == []


def test_generator_dispatches_canonical_s3_without_fetching_a_worker_manifest(
    monkeypatch,
):
    server, endpoint, service = _start_server()
    service.version.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA
    service.version.base_version_id = "base-a"
    service.version.expected_source_slots[:] = ["canonical.delta.root"]
    service.shards[:] = [
        refit_pb2.WeightVersionShard(
            version_id="version-a",
            source_slot_id="canonical.delta.root",
            worker_id="trainer-0",
            manifest_digest="a" * 64,
            s3=refit_pb2.S3Transport(
                bucket="weights",
                key="model.safetensors.index.json",
                object_version="object-a",
                checksum="crc32c:12345678",
            ),
        )
    ]
    adapter = _Adapter(service)
    generator = _initialize(
        monkeypatch,
        endpoint,
        adapter,
        WeightPayloadFormat.XOR_DELTA,
    )

    try:
        staged = generator.stage_weight(version=WeightVersionRef("version-a"))
        source = adapter.stage_calls[0].sources[0]
        assert isinstance(source.transport, S3GeneratorSource)
        assert source.transport.location.bucket == "weights"
        assert source.transport.location.object_version == "object-a"
        assert generator.apply_weight(staged) == "installed"
        staged.release()
        repeated = generator.stage_weight(version=WeightVersionRef("version-a"))
        repeated.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()


def test_generator_rejects_missing_s3_transport_before_adapter_mutation(monkeypatch):
    server, endpoint, service = _start_server()
    service.version.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA
    service.version.base_version_id = "base-a"
    service.version.expected_source_slots[:] = ["canonical.delta.root"]
    service.shards[:] = [
        refit_pb2.WeightVersionShard(
            version_id="version-a",
            source_slot_id="canonical.delta.root",
            worker_id="trainer-0",
            manifest_digest="a" * 64,
        )
    ]
    adapter = _Adapter(service)
    generator = _initialize(
        monkeypatch,
        endpoint,
        adapter,
        WeightPayloadFormat.XOR_DELTA,
    )

    try:
        with pytest.raises(RuntimeError, match="unsupported shard transport"):
            generator.stage_weight(version=WeightVersionRef("version-a"))
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert adapter.stage_calls == []
    assert service.lease_deletions == 1


def test_generator_rejects_wrong_delta_base_before_leasing(monkeypatch):
    server, endpoint, service = _start_server()
    service.version.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA
    service.version.base_version_id = "other-base"
    service.version.expected_source_slots[:] = ["canonical.delta.root"]
    adapter = _Adapter(service)
    generator = _initialize(
        monkeypatch,
        endpoint,
        adapter,
        WeightPayloadFormat.XOR_DELTA,
    )

    try:
        with pytest.raises(RuntimeError, match="does not match serving version"):
            generator.stage_weight(version=WeightVersionRef("version-a"))
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert service.lease_registrations == 0
    assert adapter.stage_calls == []


def test_generator_validates_the_initial_s3_base_before_registration(monkeypatch):
    server, endpoint, service = _start_server()
    service.base.state = refit_pb2.WEIGHT_VERSION_STATE_STAGING
    adapter = _Adapter(service)

    try:
        with pytest.raises(RuntimeError, match="initial base.*not READY"):
            _initialize(
                monkeypatch,
                endpoint,
                adapter,
                WeightPayloadFormat.XOR_DELTA,
            )
    finally:
        server.stop(grace=None).wait()

    assert service.registrations == {}
    assert adapter.close_calls == 1


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


def test_generator_retries_with_redundant_worker_for_same_slot(monkeypatch):
    server, endpoint, service = _start_server()
    replica = refit_pb2.WeightVersionShard()
    replica.CopyFrom(service.shards[0])
    replica.worker_id = "trainer-replica"
    service.shards.append(replica)
    adapter = _Adapter(service)
    adapter.stage_failures = 1
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        staged = generator.stage_weight(version=WeightVersionRef("version-a"))
        staged.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert [call.sources[0].worker_id for call in adapter.stage_calls] == [
        "trainer-0",
        "trainer-replica",
    ]


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
        staged = generator.stage_weight(version=WeightVersionRef("version-a"))
        with pytest.raises(grpc.RpcError, match="lease backend unavailable"):
            staged.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()


def test_generator_preserves_apply_error_when_lease_cleanup_also_fails(monkeypatch):
    server, endpoint, service = _start_server()
    service.fail_lease_deletion = True
    adapter = _Adapter(service)
    adapter.apply_failure = True
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        staged = generator.stage_weight(version=WeightVersionRef("version-a"))
        with pytest.raises(RuntimeError, match="apply failed"):
            generator.apply_weight(staged)
    finally:
        service.fail_lease_deletion = False
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
                ),
                model_name="test/model",
                payload_format=WeightPayloadFormat.FULL_TENSOR,
                worker_id="generator-0",
                server_url="mx-server:9000",
            )
        )

    assert adapter.close_calls == 1
