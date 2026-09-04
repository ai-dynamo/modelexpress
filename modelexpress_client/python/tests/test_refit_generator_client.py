# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import logging
from concurrent import futures
from contextlib import contextmanager

import grpc
import pytest
from modelexpress import p2p_pb2, p2p_pb2_grpc
from modelexpress.client import MxClient
from modelexpress.types import ManifestMismatchError
from modelexpress_rl import (
    ModelExpressGeneratorClient,
    ModelExpressGeneratorConfig,
    ObjectStorageGeneratorConfig,
    ObjectStorageSource,
    ObjectStorageType,
    VllmGeneratorContext,
    WeightPayloadFormat,
    WeightSource,
    WeightVersionRef,
    refit_pb2,
    refit_pb2_grpc,
)
from modelexpress_rl.inference.adapter import GeneratorTransferInputs
from modelexpress_rl.inference.plan import (
    EngineCapabilities,
    EngineInstaller,
    GeneratorPeerUpdateSource,
    MethodCapabilities,
    ObjectStorageUpdateSource,
    PreparedEngineTensors,
    ResolvedSource,
    TrainerUpdateSource,
    UpdateMethod,
    WeightUpdatePlanner,
)
from modelexpress_rl.inference.runtime import EngineRuntime, GeneratorRuntime
from modelexpress_rl.inference.session import WeightUpdateSession
from modelexpress_rl.inference.source import (
    GeneratorSourceResolver,
    ObjectStorageSourceResolver,
    TrainerSourceResolver,
)


class _RefitService(refit_pb2_grpc.RefitServiceServicer):
    def __init__(self, *, endpoint: str, state=None, manifest_digest=None):
        self.registrations = {}
        self.active_leases = set()
        self.lease_registrations = 0
        self.lease_deletions = 0
        self.list_calls = 0
        self.fail_lease_deletion = False
        self.omit_base_version = False
        self.additional_versions = {}
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
                manifest_endpoint=endpoint,
            )
            for rank, slot in enumerate(self.version.expected_source_slots)
        ]

    def RegisterWorker(self, request, _context):
        worker = request.worker
        worker.expires_at_unix_ms = 1234
        self.registrations[worker.worker_id] = worker
        return refit_pb2.RegisterWorkerResponse(worker=worker)

    def GetWeightVersion(self, request, context):
        if request.uid in self.additional_versions:
            return refit_pb2.GetWeightVersionResponse(
                version=self.additional_versions[request.uid]
            )
        if request.uid == self.base.uid:
            if self.omit_base_version:
                return refit_pb2.GetWeightVersionResponse()
            return refit_pb2.GetWeightVersionResponse(version=self.base)
        if request.uid != self.version.uid:
            context.abort(grpc.StatusCode.NOT_FOUND, "version not found")
        return refit_pb2.GetWeightVersionResponse(version=self.version)

    def ListWeightVersionShards(self, request, _context):
        self.list_calls += 1
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
        return refit_pb2.RegisterVersionLeaseResponse(
            lease=refit_pb2.VersionLease(
                lease_id=lease_id,
                version_id=request.version_id,
                worker_id=request.worker_id,
                expires_at_unix_ms=1234,
            )
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


class _P2pService(p2p_pb2_grpc.P2pServiceServicer):
    def __init__(self):
        self.requests = []
        self.instances = []
        self.metadata = {}

    def ListSources(self, request, _context):
        self.requests.append(request)
        return p2p_pb2.ListSourcesResponse(instances=self.instances)

    def GetMetadata(self, request, _context):
        worker = self.metadata.get((request.mx_source_id, request.worker_id))
        if worker is None:
            return p2p_pb2.GetMetadataResponse(found=False)
        return p2p_pb2.GetMetadataResponse(found=True, worker=worker)


class _Adapter:
    def __init__(self, service):
        self.service = service
        self.create_calls = []
        self.validate_calls = []
        self.stage_calls = []
        self.peer_stage_calls = []
        self.apply_calls = []
        self.publish_calls = []
        self.publish_attempts = 0
        self.release_calls = []
        self.close_calls = 0
        self.identity_failure = False
        self.publish_failures = 0
        self.stage_failures = 0
        self.apply_failure = False
        self.installation_context_failure = False
        self.installation_failure_calls = []

    @property
    def worker_rank(self):
        return 0

    def build_p2p_identity(self, version_id):
        if self.identity_failure:
            raise RuntimeError("identity unavailable")
        return p2p_pb2.SourceIdentity(
            model_name="test/model",
            revision=version_id,
        )

    def stage_peer_weight(self, source):
        assert self.service.active_leases
        self.peer_stage_calls.append(source)
        return {"peer": source}

    def publish_weight_version(self, **kwargs):
        assert self.service.active_leases
        self.publish_attempts += 1
        if self.publish_failures:
            self.publish_failures -= 1
            raise RuntimeError("publication failed")
        self.publish_calls.append(kwargs)

    def create_transfer_plan(self, inputs):
        self.create_calls.append(inputs)
        return {"inputs": inputs}

    def validate_transfer_plan(self, plan, inputs):
        self.validate_calls.append((plan, inputs))
        return True

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


class _TestMethod(UpdateMethod):
    def __init__(self, adapter, *, publish_peer):
        self._adapter = adapter
        self._publish_peer = publish_peer
        self._cached_plan = None
        self._cached_fingerprint = None
        self._active_source = None

    @property
    def capabilities(self):
        return MethodCapabilities(
            payload_formats=frozenset(
                {
                    WeightPayloadFormat.FULL_TENSOR,
                    WeightPayloadFormat.XOR_DELTA,
                    WeightPayloadFormat.FULL_HF_CHECKPOINT,
                }
            ),
            sources=frozenset(WeightSource),
            artifact_type=PreparedEngineTensors,
        )

    def prepare(self, *, version, source: ResolvedSource):
        self._active_source = source.kind
        if isinstance(source, GeneratorPeerUpdateSource):
            staged = self._adapter.stage_peer_weight(source.worker)
        else:
            if isinstance(source, ObjectStorageUpdateSource):
                inputs = GeneratorTransferInputs(
                    version_id=version.version_id,
                    base_version_id=version.base_version_id,
                    layout_signature=version.layout_signature,
                    payload_format=version.payload_format,
                    sources=(),
                    object_storage=source.storage,
                )
            elif isinstance(source, TrainerUpdateSource):
                inputs = source.inputs
            else:
                raise TypeError("unsupported test source")
            reusable = (
                self._cached_plan is not None
                and self._cached_fingerprint == inputs.physical_fingerprint
                and self._adapter.validate_transfer_plan(self._cached_plan, inputs)
            )
            if not reusable:
                self._cached_plan = self._adapter.create_transfer_plan(inputs)
                self._cached_fingerprint = inputs.physical_fingerprint
            staged = self._adapter.stage_weight(inputs)
        return PreparedEngineTensors(staged=staged)

    def prepare_chain(self, chain):
        prepared = None
        for version, source in chain:
            prepared = self.prepare(version=version, source=source)
        assert prepared is not None
        return prepared

    def release(self, prepared):
        self._adapter.release_staged_weight(prepared.staged)

    @contextmanager
    def installation_context(self, prepared):
        del prepared
        if self._adapter.installation_context_failure:
            raise RuntimeError("installation context failed")
        yield

    def installation_failed(self, prepared):
        self._adapter.installation_failure_calls.append(prepared)

    def publish_applied(self, *, version_id, prepared):
        if not self._publish_peer or self._active_source is WeightSource.OBJECT_STORAGE:
            return
        self._adapter.publish_weight_version(
            version_id=version_id,
            staged=prepared.staged,
            p2p_client=None,
            worker_id="generator-0",
        )

    def close(self):
        self._adapter.close()


class _TestInstaller(EngineInstaller):
    def __init__(self, adapter):
        self._adapter = adapter

    @property
    def capabilities(self):
        return EngineCapabilities(
            artifact_types=frozenset({PreparedEngineTensors})
        )

    def install(self, prepared):
        return self._adapter.apply_weight(prepared.staged)


def _runtime(
    adapter,
    *,
    server_url,
    service,
    start_lease,
    worker_id,
    object_storage,
    source_order,
    max_transfer_attempts,
    rpc_timeout_seconds,
    **_kwargs,
):
    if source_order is None:
        source_order = (
            (WeightSource.OBJECT_STORAGE,)
            if object_storage is not None
            else (WeightSource.GENERATOR, WeightSource.TRAINER)
        )
    p2p_client = (
        MxClient(server_url=server_url)
        if WeightSource.GENERATOR in source_order
        else None
    )
    resolvers = []
    for source in source_order:
        if source is WeightSource.GENERATOR:
            assert p2p_client is not None
            resolvers.append(
                GeneratorSourceResolver(
                    p2p_client=p2p_client,
                    worker_id=worker_id,
                    worker_rank=adapter.worker_rank,
                    build_identity=adapter.build_p2p_identity,
                    rpc_timeout_seconds=rpc_timeout_seconds,
                )
            )
        elif source is WeightSource.TRAINER:
            resolvers.append(
                TrainerSourceResolver(
                    service=service,
                    rpc_timeout_seconds=rpc_timeout_seconds,
                )
            )
        else:
            resolvers.append(ObjectStorageSourceResolver())
    method = _TestMethod(
        adapter,
        publish_peer=WeightSource.GENERATOR in source_order,
    )
    installer = _TestInstaller(adapter)
    return GeneratorRuntime(
        engine=EngineRuntime(model_name="test/model", installer=installer),
        methods=(method,),
        session=WeightUpdateSession(
            planner=WeightUpdatePlanner(
                resolvers=tuple(resolvers),
                methods=(method,),
                installer=installer,
                max_transfer_attempts=max_transfer_attempts,
            ),
            start_lease=start_lease,
        ),
        p2p_client=p2p_client,
        initial_version_id=(
            object_storage.initial_base_version_id
            if object_storage is not None
            else None
        ),
    )


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
    p2p_service = _P2pService()
    p2p_pb2_grpc.add_P2pServiceServicer_to_server(p2p_service, server)
    service.p2p = p2p_service
    server.start()
    return server, endpoint, service


def _initialize(
    monkeypatch,
    endpoint,
    adapter,
    *,
    object_storage=False,
    max_transfer_attempts=3,
    max_replay_chain_length=64,
    source_order=None,
):
    monkeypatch.setattr(
        GeneratorRuntime,
        "initialize",
        classmethod(lambda _cls, **kwargs: _runtime(adapter, **kwargs)),
    )
    return ModelExpressGeneratorClient.initialize(
        ModelExpressGeneratorConfig(
            engine_context=VllmGeneratorContext(
                model=object(),
                vllm_config=object(),
            ),
            model_name="test/model",
            worker_id="generator-0",
            server_url=endpoint,
            registration_ttl_seconds=60,
            lease_ttl_seconds=60,
            object_storage=(
                ObjectStorageGeneratorConfig(
                    storage_type=ObjectStorageType.S3,
                    initial_base_version_id="base-a",
                    seed_checkpoint_path="unused-launch",
                    refit_checkpoint_dir="unused-cache",
                )
                if object_storage
                else None
            ),
            max_transfer_attempts=max_transfer_attempts,
            max_replay_chain_length=max_replay_chain_length,
            source_order=source_order,
        )
    )


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("registration_ttl_seconds", 0, "registration_ttl_seconds must be positive"),
        ("lease_ttl_seconds", -1, "lease_ttl_seconds must be positive"),
        ("max_transfer_attempts", 0, "max_transfer_attempts must be positive"),
        (
            "max_replay_chain_length",
            0,
            "max_replay_chain_length must be positive",
        ),
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


def test_generator_config_rejects_invalid_source_order():
    context = VllmGeneratorContext(model=object(), vllm_config=object())
    with pytest.raises(ValueError, match="non-empty tuple"):
        ModelExpressGeneratorConfig(engine_context=context, source_order=())
    with pytest.raises(ValueError, match="duplicates"):
        ModelExpressGeneratorConfig(
            engine_context=context,
            source_order=(WeightSource.GENERATOR, WeightSource.GENERATOR),
        )
    with pytest.raises(ValueError, match="requires object_storage settings"):
        ModelExpressGeneratorConfig(
            engine_context=context,
            source_order=(WeightSource.OBJECT_STORAGE,),
        )
    with pytest.raises(ValueError, match="require OBJECT_STORAGE in source_order"):
        ModelExpressGeneratorConfig(
            engine_context=context,
            object_storage=ObjectStorageGeneratorConfig(
                storage_type=ObjectStorageType.S3,
                initial_base_version_id="base-a",
                seed_checkpoint_path="unused-launch",
                refit_checkpoint_dir="unused-cache",
            ),
            source_order=(WeightSource.TRAINER,),
        )
    with pytest.raises(ValueError, match="currently requires source_order"):
        ModelExpressGeneratorConfig(
            engine_context=context,
            object_storage=ObjectStorageGeneratorConfig(
                storage_type=ObjectStorageType.S3,
                initial_base_version_id="base-a",
                seed_checkpoint_path="unused-launch",
                refit_checkpoint_dir="unused-cache",
            ),
            source_order=(
                WeightSource.GENERATOR,
                WeightSource.OBJECT_STORAGE,
            ),
        )


def test_generator_rejects_unsupported_object_storage_before_adapter_creation(
    monkeypatch,
):
    monkeypatch.setattr(
        GeneratorRuntime,
        "initialize",
        classmethod(
            lambda _cls, **_kwargs: pytest.fail("runtime must not be created")
        ),
    )

    with pytest.raises(ValueError, match="only S3 object storage"):
        ModelExpressGeneratorClient.initialize(
            ModelExpressGeneratorConfig(
                engine_context=VllmGeneratorContext(
                    model=object(),
                    vllm_config=object(),
                ),
                model_name="test/model",
                object_storage=ObjectStorageGeneratorConfig(
                    storage_type=ObjectStorageType.GCS,
                    initial_base_version_id="base-a",
                    seed_checkpoint_path="unused-launch",
                    refit_checkpoint_dir="unused-cache",
                ),
            )
        )


def test_generator_prefers_peer_source_before_trainer_memory(monkeypatch):
    server, endpoint, service = _start_server()
    adapter = _Adapter(service)
    generator = _initialize(monkeypatch, endpoint, adapter)
    try:
        resolvers = generator._runtime.session._planner._resolvers
        assert [resolver.kind for resolver in resolvers] == [
            WeightSource.GENERATOR,
            WeightSource.TRAINER,
        ]
    finally:
        generator.close()
        server.stop(grace=None).wait()


def test_generator_uses_configured_source_order(monkeypatch):
    server, endpoint, service = _start_server()
    adapter = _Adapter(service)
    generator = _initialize(
        monkeypatch,
        endpoint,
        adapter,
        source_order=(WeightSource.TRAINER,),
    )
    try:
        resolvers = generator._runtime.session._planner._resolvers
        assert [resolver.kind for resolver in resolvers] == [WeightSource.TRAINER]
    finally:
        generator.close()
        server.stop(grace=None).wait()


def test_generator_stages_applies_releases_and_reuses_valid_plan(monkeypatch):
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
    assert len(adapter.create_calls) == 2
    assert len(adapter.validate_calls) == 1
    assert len(adapter.stage_calls) == 3
    assert len(adapter.apply_calls) == 1
    assert len(adapter.publish_calls) == 1
    assert len(adapter.release_calls) == 3
    assert adapter.close_calls == 1
    assert [source.source_slot_id for source in adapter.create_calls[0].sources] == [
        "rank:0",
        "rank:1",
    ]
    assert (
        adapter.create_calls[0].payload_format is WeightPayloadFormat.FULL_TENSOR
    )


def test_generator_logs_weight_update_lifecycle(monkeypatch, caplog):
    server, endpoint, service = _start_server()
    adapter = _Adapter(service)
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        with caplog.at_level(
            logging.INFO,
            logger="modelexpress_rl.inference.session",
        ):
            staged = generator.stage_weight(version=WeightVersionRef("version-a"))
            generator.apply_weight(staged)
            staged.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "version=version-a trying source=TRAINER" in message for message in messages
    )
    assert any(
        "version=version-a prepared source=TRAINER" in message for message in messages
    )
    assert any("version=version-a installing" in message for message in messages)
    assert any("version=version-a installed" in message for message in messages)
    assert "ModelExpress weight update version=version-a released" in messages


def test_generator_does_not_retry_peer_publication(monkeypatch):
    server, endpoint, service = _start_server()
    adapter = _Adapter(service)
    adapter.publish_failures = 1
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        staged = generator.stage_weight(version=WeightVersionRef("version-a"))
        assert generator.apply_weight(staged) == "installed"
        staged.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert adapter.publish_attempts == 1
    assert adapter.publish_calls == []


def test_generator_releases_lease_when_manifest_is_invalid(monkeypatch):
    server, endpoint, service = _start_server(manifest_digest="bad-digest")
    adapter = _Adapter(service)
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        with pytest.raises(RuntimeError, match=r"no usable refit source"):
            generator.stage_weight(version=WeightVersionRef("version-a"))
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert not service.active_leases
    assert service.lease_registrations == 1
    assert service.lease_deletions == 1
    assert adapter.stage_calls == []


def test_generator_reports_missing_trainer_manifest_digest(monkeypatch, caplog):
    server, endpoint, service = _start_server()
    service.shards[0].manifest_digest = ""
    adapter = _Adapter(service)
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        with caplog.at_level(logging.WARNING):
            with pytest.raises(RuntimeError, match=r"no usable refit source"):
                generator.stage_weight(version=WeightVersionRef("version-a"))
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert "source is missing its manifest digest" in caplog.text
    assert adapter.stage_calls == []


def test_generator_dispatches_canonical_s3_without_fetching_a_worker_manifest(
    monkeypatch,
):
    server, endpoint, service = _start_server()
    service.version.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA
    service.version.base_version_id = "base-a"
    service.version.expected_source_slots[:] = []
    service.version.object_storage.CopyFrom(
        refit_pb2.ObjectStorageSource(
            storage_type=refit_pb2.OBJECT_STORAGE_TYPE_S3,
            uri="s3://weights/model.safetensors.index.json",
        )
    )
    adapter = _Adapter(service)
    generator = _initialize(
        monkeypatch,
        endpoint,
        adapter,
        object_storage=True,
    )
    assert generator._runtime.p2p_client is None
    assert [
        resolver.kind for resolver in generator._runtime.session._planner._resolvers
    ] == [WeightSource.OBJECT_STORAGE]

    try:
        staged = generator.stage_weight(version=WeightVersionRef("version-a"))
        assert adapter.stage_calls[0].sources == ()
        assert adapter.stage_calls[0].object_storage == ObjectStorageSource(
            storage_type=ObjectStorageType.S3,
            uri="s3://weights/model.safetensors.index.json",
        )
        assert service.list_calls == 0
        assert generator.apply_weight(staged) == "installed"
        assert adapter.publish_attempts == 0
        assert service.p2p.requests == []
        staged.release()
        repeated = generator.stage_weight(version=WeightVersionRef("version-a"))
        assert repeated.applied is True
        assert repeated.metrics == {}
        assert generator.apply_weight(repeated) is None
        repeated.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert len(adapter.stage_calls) == 1
    assert len(adapter.apply_calls) == 1
    assert service.lease_registrations == 1


def test_generator_retries_canonical_s3_under_one_lease(monkeypatch):
    server, endpoint, service = _start_server()
    service.version.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA
    service.version.base_version_id = "base-a"
    service.version.expected_source_slots[:] = []
    service.version.object_storage.CopyFrom(
        refit_pb2.ObjectStorageSource(
            storage_type=refit_pb2.OBJECT_STORAGE_TYPE_S3,
            uri="s3://weights/model.safetensors.index.json",
        )
    )
    adapter = _Adapter(service)
    adapter.stage_failures = 1
    generator = _initialize(
        monkeypatch,
        endpoint,
        adapter,
        object_storage=True,
    )

    try:
        staged = generator.stage_weight(version=WeightVersionRef("version-a"))
        staged.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert len(adapter.stage_calls) == 2
    assert service.lease_registrations == 1
    assert service.lease_deletions == 1


def test_generator_treats_installed_initial_base_as_successful_no_op(monkeypatch):
    server, endpoint, service = _start_server()
    adapter = _Adapter(service)
    generator = _initialize(monkeypatch, endpoint, adapter, object_storage=True)

    try:
        staged = generator.stage_weight(version=WeightVersionRef("base-a"))
        assert staged.applied is True
        assert generator.apply_weight(staged) is None
        staged.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert service.lease_registrations == 0
    assert adapter.stage_calls == []
    assert adapter.apply_calls == []


def _canonical_version(uid, base_version_id):
    return refit_pb2.WeightVersion(
        uid=uid,
        model_name="test/model",
        payload_format=refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA,
        base_version_id=base_version_id,
        layout_signature="layout-a",
        state=refit_pb2.WEIGHT_VERSION_STATE_READY,
        object_storage=refit_pb2.ObjectStorageSource(
            storage_type=refit_pb2.OBJECT_STORAGE_TYPE_S3,
            uri=f"s3://weights/{uid}/model.safetensors.index.json",
        ),
    )


def test_generator_resolves_and_stages_target_replay_chain(monkeypatch):
    server, endpoint, service = _start_server()
    service.version.CopyFrom(_canonical_version("version-c", "version-b"))
    service.additional_versions = {
        "version-a": _canonical_version("version-a", "base-a"),
        "version-b": _canonical_version("version-b", "version-a"),
    }
    adapter = _Adapter(service)
    generator = _initialize(monkeypatch, endpoint, adapter, object_storage=True)

    try:
        staged = generator.stage_weight(version=WeightVersionRef("version-c"))
        assert [call.version_id for call in adapter.stage_calls] == [
            "version-a",
            "version-b",
            "version-c",
        ]
        assert adapter.apply_calls == []
        assert generator.apply_weight(staged) == "installed"
        staged.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert len(adapter.apply_calls) == 1
    assert service.lease_registrations == 3
    assert service.lease_deletions == 3


def test_generator_rejects_replay_cycle_before_leasing(monkeypatch):
    server, endpoint, service = _start_server()
    service.version.CopyFrom(_canonical_version("version-b", "version-a"))
    service.additional_versions = {
        "version-a": _canonical_version("version-a", "version-b"),
    }
    adapter = _Adapter(service)
    generator = _initialize(monkeypatch, endpoint, adapter, object_storage=True)

    try:
        with pytest.raises(RuntimeError, match=r"cycle detected.*version-b"):
            generator.stage_weight(version=WeightVersionRef("version-b"))
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert service.lease_registrations == 0
    assert adapter.stage_calls == []


def test_generator_rejects_excessive_replay_chain_before_leasing(monkeypatch):
    server, endpoint, service = _start_server()
    service.version.CopyFrom(_canonical_version("version-b", "version-a"))
    service.additional_versions = {
        "version-a": _canonical_version("version-a", "base-a"),
    }
    adapter = _Adapter(service)
    generator = _initialize(
        monkeypatch,
        endpoint,
        adapter,
        object_storage=True,
        max_replay_chain_length=1,
    )

    try:
        with pytest.raises(RuntimeError, match=r"maximum chain length 1.*version-a"):
            generator.stage_weight(version=WeightVersionRef("version-b"))
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert service.lease_registrations == 0
    assert adapter.stage_calls == []


def test_generator_dispatches_full_hf_checkpoint_without_an_exact_base(
    monkeypatch,
):
    server, endpoint, service = _start_server()
    service.version.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_HF_CHECKPOINT
    service.version.ClearField("base_version_id")
    service.version.expected_source_slots[:] = []
    service.version.object_storage.CopyFrom(
        refit_pb2.ObjectStorageSource(
            storage_type=refit_pb2.OBJECT_STORAGE_TYPE_S3,
            uri="s3://weights/model.safetensors.index.json",
        )
    )
    adapter = _Adapter(service)
    generator = _initialize(
        monkeypatch,
        endpoint,
        adapter,
        object_storage=True,
    )

    try:
        staged = generator.stage_weight(version=WeightVersionRef("version-a"))
        inputs = adapter.stage_calls[0]
        assert inputs.payload_format is WeightPayloadFormat.FULL_HF_CHECKPOINT
        assert inputs.base_version_id is None
        assert service.list_calls == 0
        staged.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()


def test_generator_rejects_full_hf_checkpoint_with_a_base_before_leasing(
    monkeypatch,
):
    server, endpoint, service = _start_server()
    service.version.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_HF_CHECKPOINT
    service.version.base_version_id = "base-a"
    service.version.expected_source_slots[:] = []
    service.version.object_storage.CopyFrom(
        refit_pb2.ObjectStorageSource(
            storage_type=refit_pb2.OBJECT_STORAGE_TYPE_S3,
            uri="s3://weights/model.safetensors.index.json",
        )
    )
    adapter = _Adapter(service)
    generator = _initialize(
        monkeypatch,
        endpoint,
        adapter,
        object_storage=True,
    )

    try:
        with pytest.raises(RuntimeError, match="must not have base_version_id"):
            generator.stage_weight(version=WeightVersionRef("version-a"))
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert service.lease_registrations == 0
    assert adapter.stage_calls == []


def test_generator_rejects_missing_object_storage_before_adapter_mutation(
    monkeypatch,
):
    server, endpoint, service = _start_server()
    service.version.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA
    service.version.base_version_id = "base-a"
    service.version.expected_source_slots[:] = []
    adapter = _Adapter(service)
    generator = _initialize(
        monkeypatch,
        endpoint,
        adapter,
        object_storage=True,
    )

    try:
        with pytest.raises(RuntimeError, match="no legal source"):
            generator.stage_weight(version=WeightVersionRef("version-a"))
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert adapter.stage_calls == []
    assert service.lease_registrations == 0
    assert service.lease_deletions == 0


def test_generator_skips_non_s3_object_storage_before_adapter_mutation(
    monkeypatch,
):
    server, endpoint, service = _start_server()
    service.version.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA
    service.version.base_version_id = "base-a"
    service.version.expected_source_slots[:] = []
    service.version.object_storage.CopyFrom(
        refit_pb2.ObjectStorageSource(
            storage_type=refit_pb2.OBJECT_STORAGE_TYPE_GCS,
            uri="gs://weights/model.safetensors.index.json",
        )
    )
    adapter = _Adapter(service)
    generator = _initialize(
        monkeypatch,
        endpoint,
        adapter,
        object_storage=True,
    )

    try:
        with pytest.raises(RuntimeError, match="no usable refit source"):
            generator.stage_weight(version=WeightVersionRef("version-a"))
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert adapter.stage_calls == []
    assert service.lease_registrations == 1
    assert service.lease_deletions == 1


def test_generator_rejects_wrong_delta_base_before_lease(monkeypatch):
    server, endpoint, service = _start_server()
    service.version.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA
    service.version.base_version_id = "other-base"
    service.version.expected_source_slots[:] = []
    service.version.object_storage.CopyFrom(
        refit_pb2.ObjectStorageSource(
            storage_type=refit_pb2.OBJECT_STORAGE_TYPE_S3,
            uri="s3://weights/model.safetensors.index.json",
        )
    )
    adapter = _Adapter(service)
    generator = _initialize(
        monkeypatch,
        endpoint,
        adapter,
        object_storage=True,
    )

    try:
        with pytest.raises(RuntimeError, match="does not match serving version"):
            generator.stage_weight(version=WeightVersionRef("version-a"))
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert service.lease_registrations == 0
    assert service.lease_deletions == 0
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
                object_storage=True,
            )
    finally:
        server.stop(grace=None).wait()

    assert service.registrations == {}
    assert adapter.close_calls == 1


def test_generator_rejects_missing_initial_s3_base_before_registration(monkeypatch):
    server, endpoint, service = _start_server()
    service.omit_base_version = True
    adapter = _Adapter(service)

    try:
        with pytest.raises(RuntimeError, match="GetWeightVersion response is missing"):
            _initialize(
                monkeypatch,
                endpoint,
                adapter,
                object_storage=True,
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

    assert [call.sources[0].worker_id for call in adapter.create_calls] == [
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


@pytest.mark.parametrize("recovery_version", ["base-a", "version-b"])
def test_generator_recovers_uncertain_engine_with_active_or_new_version(
    monkeypatch, recovery_version
):
    server, endpoint, service = _start_server()
    service.version.CopyFrom(_canonical_version("version-a", "base-a"))
    service.base.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_HF_CHECKPOINT
    service.base.object_storage.CopyFrom(
        refit_pb2.ObjectStorageSource(
            storage_type=refit_pb2.OBJECT_STORAGE_TYPE_S3,
            uri="s3://weights/base-a/model.safetensors.index.json",
        )
    )
    service.additional_versions["version-b"] = _canonical_version(
        "version-b", "base-a"
    )
    adapter = _Adapter(service)
    adapter.apply_failure = True
    generator = _initialize(monkeypatch, endpoint, adapter, object_storage=True)

    try:
        failed = generator.stage_weight(version=WeightVersionRef("version-a"))
        with pytest.raises(RuntimeError, match="apply failed"):
            generator.apply_weight(failed)
        assert len(adapter.installation_failure_calls) == 1
        failed.release()

        adapter.apply_failure = False
        recovery = generator.stage_weight(version=WeightVersionRef(recovery_version))
        assert recovery.applied is False
        assert generator.apply_weight(recovery) == "installed"
        recovery.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert len(adapter.apply_calls) == 2


def test_generator_does_not_fence_when_installation_context_entry_fails(monkeypatch):
    server, endpoint, service = _start_server()
    adapter = _Adapter(service)
    adapter.installation_context_failure = True
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        staged = generator.stage_weight(version=WeightVersionRef("version-a"))
        with pytest.raises(RuntimeError, match="installation context failed"):
            generator.apply_weight(staged)
        staged.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert adapter.installation_failure_calls == []
    assert adapter.apply_calls == []


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


def test_generator_rejects_unsupported_payload_after_peer_miss(monkeypatch):
    server, endpoint, service = _start_server()
    service.version.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA
    service.version.base_version_id = "version-base"
    adapter = _Adapter(service)
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        with pytest.raises(RuntimeError, match=r"no usable refit source"):
            generator.stage_weight(version=WeightVersionRef("version-a"))
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert service.lease_registrations == 1
    assert service.lease_deletions == 1


def test_generator_falls_back_when_peer_identity_is_unavailable(monkeypatch):
    server, endpoint, service = _start_server()
    adapter = _Adapter(service)
    adapter.identity_failure = True
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        staged = generator.stage_weight(version=WeightVersionRef("version-a"))
        staged.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert len(adapter.stage_calls) == 1
    assert service.p2p.requests == []


def test_generator_discovers_rank_matched_p2p_peer(monkeypatch):
    server, endpoint, service = _start_server()
    service.p2p.instances.extend(
        [
            p2p_pb2.SourceInstanceRef(
                mx_source_id="wrong-rank",
                worker_id="generator-rank-1",
                worker_rank=1,
            ),
            p2p_pb2.SourceInstanceRef(
                mx_source_id="same-worker",
                worker_id="generator-0",
                worker_rank=0,
            ),
            p2p_pb2.SourceInstanceRef(
                mx_source_id="peer-source",
                worker_id="generator-peer",
                worker_rank=0,
            ),
        ]
    )
    service.p2p.metadata[("peer-source", "generator-peer")] = (
        p2p_pb2.WorkerMetadata(
            worker_rank=0,
            tensors=[
                p2p_pb2.TensorDescriptor(
                    name="weight",
                    addr=1234,
                    size=16,
                    device_id=0,
                    dtype="torch.float32",
                )
            ],
        )
    )
    adapter = _Adapter(service)
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        staged = generator.stage_weight(version=WeightVersionRef("version-a"))
        staged.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert service.p2p.requests[0].identity.revision == "version-a"
    assert service.lease_registrations == 1
    assert service.lease_deletions == 1
    assert len(adapter.peer_stage_calls) == 1
    assert adapter.create_calls == []


def test_generator_tries_next_peer_after_manifest_mismatch(monkeypatch):
    monkeypatch.setattr(
        "modelexpress_rl.inference.source.generator.random.Random.shuffle",
        lambda _random, _sources: None,
    )
    server, endpoint, service = _start_server()
    service.p2p.instances.extend(
        [
            p2p_pb2.SourceInstanceRef(
                mx_source_id="bad-source",
                worker_id="bad-peer",
                worker_rank=0,
            ),
            p2p_pb2.SourceInstanceRef(
                mx_source_id="good-source",
                worker_id="good-peer",
                worker_rank=0,
            ),
        ]
    )
    for source_id, worker_id, agent_name in (
        ("bad-source", "bad-peer", "bad-agent"),
        ("good-source", "good-peer", "good-agent"),
    ):
        service.p2p.metadata[(source_id, worker_id)] = p2p_pb2.WorkerMetadata(
            worker_rank=0,
            agent_name=agent_name,
            tensors=[
                p2p_pb2.TensorDescriptor(
                    name="weight",
                    addr=1234,
                    size=16,
                    device_id=0,
                    dtype="torch.float32",
                )
            ],
        )

    adapter = _Adapter(service)

    def stage_peer(source):
        adapter.peer_stage_calls.append(source)
        if source.agent_name == "bad-agent":
            raise ManifestMismatchError("incompatible peer manifest")
        return {"peer": source}

    adapter.stage_peer_weight = stage_peer
    generator = _initialize(monkeypatch, endpoint, adapter)

    try:
        staged = generator.stage_weight(version=WeightVersionRef("version-a"))
        staged.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert [source.agent_name for source in adapter.peer_stage_calls] == [
        "bad-agent",
        "good-agent",
    ]
    assert adapter.create_calls == []


def test_generator_randomizes_and_limits_peers_before_trainer_fallback(monkeypatch):
    monkeypatch.setattr(
        "modelexpress_rl.inference.source.generator.random.Random.shuffle",
        lambda _random, sources: sources.reverse(),
    )
    server, endpoint, service = _start_server()
    service.p2p.instances.extend(
        p2p_pb2.SourceInstanceRef(
            mx_source_id=f"peer-source-{index}",
            worker_id=f"peer-{index}",
            worker_rank=0,
        )
        for index in range(3)
    )
    for index in range(3):
        service.p2p.metadata[(f"peer-source-{index}", f"peer-{index}")] = (
            p2p_pb2.WorkerMetadata(worker_rank=0, agent_name=f"peer-agent-{index}")
        )

    adapter = _Adapter(service)

    def reject_peer(source):
        adapter.peer_stage_calls.append(source)
        raise ManifestMismatchError("incompatible peer manifest")

    adapter.stage_peer_weight = reject_peer
    generator = _initialize(
        monkeypatch,
        endpoint,
        adapter,
        max_transfer_attempts=2,
    )

    try:
        staged = generator.stage_weight(version=WeightVersionRef("version-a"))
        staged.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert [source.agent_name for source in adapter.peer_stage_calls] == [
        "peer-agent-2",
        "peer-agent-1",
    ]
    assert len(adapter.create_calls) == 1


def test_object_storage_generator_skips_full_peer_for_delta_version(monkeypatch):
    server, endpoint, service = _start_server()
    service.version.payload_format = refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA
    service.version.base_version_id = "base-a"
    service.version.object_storage.CopyFrom(
        refit_pb2.ObjectStorageSource(
            storage_type=refit_pb2.OBJECT_STORAGE_TYPE_S3,
            uri="s3://weights/model.safetensors.index.json",
        )
    )
    service.p2p.instances.append(
        p2p_pb2.SourceInstanceRef(
            mx_source_id="peer-source",
            worker_id="generator-peer",
            worker_rank=0,
        )
    )
    service.p2p.metadata[("peer-source", "generator-peer")] = (
        p2p_pb2.WorkerMetadata(
            worker_rank=0,
            tensors=[
                p2p_pb2.TensorDescriptor(
                    name="weight",
                    addr=1234,
                    size=16,
                    device_id=0,
                    dtype="torch.float32",
                )
            ],
        )
    )
    adapter = _Adapter(service)
    generator = _initialize(monkeypatch, endpoint, adapter, object_storage=True)

    try:
        staged = generator.stage_weight(version=WeightVersionRef("version-a"))
        staged.release()
    finally:
        generator.close()
        server.stop(grace=None).wait()

    assert adapter.peer_stage_calls == []
    assert len(adapter.create_calls) == 1


def test_generator_closes_adapter_when_registration_fails(monkeypatch):
    service = _RefitService(endpoint="unused")
    adapter = _Adapter(service)
    monkeypatch.setattr(
        GeneratorRuntime,
        "initialize",
        classmethod(lambda _cls, **kwargs: _runtime(adapter, **kwargs)),
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
                worker_id="generator-0",
                server_url="mx-server:9000",
            )
        )

    assert adapter.close_calls == 1
