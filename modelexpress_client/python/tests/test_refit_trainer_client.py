# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import json
import hashlib
from concurrent import futures
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock

import grpc
import modelexpress_rl.train.client as client_module
import modelexpress_rl.train.runtime as runtime_module
import pytest
from modelexpress_rl import (
    FSDPTrainerContext,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionRef,
    refit_pb2,
    refit_pb2_grpc,
)
from modelexpress_rl.train.adapter import (
    CompletionFence,
    StagedWeightVersionShardData,
    TrainerEngineAdapter,
    TrainerShardMetadata,
)
from modelexpress_rl.train.manifest import RefitWorkerService, bound_tensor_manifest


class _RefitService(refit_pb2_grpc.RefitServiceServicer):
    def __init__(self):
        self.registrations = {}
        self.registration_count = 0
        self.shards = []
        self.deleted_shards = []
        self.mesh_id = None
        self.mesh_generation = 1
        self.publication_error = None
        self.lease_active = False

    def GetWeightVersion(self, request, _context) -> refit_pb2.GetWeightVersionResponse:
        version = refit_pb2.WeightVersion(uid=request.uid)
        if self.mesh_id is not None:
            version.trainer_mesh_id = self.mesh_id
            version.trainer_mesh_generation = self.mesh_generation
        return refit_pb2.GetWeightVersionResponse(version=version)

    def RegisterWorker(self, request, _context):
        self.registration_count += 1
        worker = request.worker
        worker.expires_at_unix_ms = 1234
        self.registrations[worker.worker_id] = worker
        return refit_pb2.RegisterWorkerResponse(worker=worker)

    def CreateWeightVersionShard(self, request, context):
        shard = request.shard
        if shard.worker_id not in self.registrations:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "worker not registered")
        self.shards.append(shard)
        if self.publication_error is not None:
            context.abort(self.publication_error, "response lost")
        return refit_pb2.CreateWeightVersionShardResponse(
            shard=shard,
            version=refit_pb2.WeightVersion(
                uid=shard.version_id,
                state=refit_pb2.WEIGHT_VERSION_STATE_READY,
            ),
        )

    def DeleteWeightVersionShard(self, request, context) -> refit_pb2.DeleteWeightVersionShardResponse:
        if self.lease_active:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "weight version has an active lease")
        self.deleted_shards.append(request)
        return refit_pb2.DeleteWeightVersionShardResponse(deleted=True)


class _Manager:
    listen_port = 19000


class _Adapter(TrainerEngineAdapter):
    bound_manifest = bound_tensor_manifest([
        {
            "name": "weight",
            "dtype": "torch.bfloat16",
            "elsize": 2,
            "full_shape": [4],
            "shards": [{"shard_offset": [0], "shape": [4]}],
        }
    ])
    logical_shard_id = hashlib.sha256(bound_manifest).hexdigest()
    supported_staging_modes = frozenset({TrainerStagingMode.COPY_TO_DEVICE})
    supported_payload_formats = frozenset({WeightPayloadFormat.FULL_TENSOR})

    def __init__(self):
        self.calls = []
        self.metadata_revision = 0

    def bind_tensors(self, tensors):
        if tensors is None:
            raise ValueError("tensors must not be None")
        return self.logical_shard_id

    def stage_shard(self, *, tensors, staging_mode, payload_format):
        self.calls.append((tensors, staging_mode, payload_format))

        return StagedWeightVersionShardData(
            metadata=TrainerShardMetadata(
                data=json.dumps({"storage_revision": self.metadata_revision, "tensor_count": 2, "total_bytes": 128, "tensors": [{
                    "name": "weight", "dtype": "torch.bfloat16", "elsize": 2,
                    "full_shape": [4], "shards": [{"shard_offset": [0], "shape": [4]}],
                }]}).encode(),
                tensor_count=2,
                total_bytes=128,
                transport="NIXL",
            ),
            publish_ready=CompletionFence(lambda: None),
            buffer_owner=tensors,
            checksums=(("weight", 0, "weight-digest"),),
        )


def _patch_resources(monkeypatch, *, manager, manifest_service, worker_endpoint):
    resources = MagicMock(
        manager=manager,
        manifest_service=manifest_service,
        worker_endpoint=worker_endpoint,
    )
    initialize_resources = MagicMock(return_value=resources)
    monkeypatch.setattr(
        runtime_module._TrainerResources,
        "initialize",
        initialize_resources,
    )
    return resources


@pytest.mark.parametrize(
    ("setting", "value", "message"),
    [
        ("registration_ttl_seconds", 0, "registration_ttl_seconds must be positive"),
        (
            "rpc_timeout_seconds",
            float("nan"),
            "rpc_timeout_seconds must be finite and positive",
        ),
    ],
)
def test_trainer_config_rejects_invalid_numeric_settings(setting, value, message):
    with pytest.raises(ValueError, match=message):
        ModelExpressTrainerConfig(**{setting: value})


def test_trainer_config_rejects_unspecified_payload_format():
    with pytest.raises(ValueError, match="payload_format must be specified"):
        ModelExpressTrainerConfig(payload_format=WeightPayloadFormat.UNSPECIFIED)


def test_trainer_config_preserves_original_positional_field_order():
    config = ModelExpressTrainerConfig(2, "trainer-agent", "test/model")

    assert config.device_id == 2
    assert config.agent_name == "trainer-agent"
    assert config.model_name == "test/model"
    assert config.engine_context is None


def test_refit_shard_advertises_its_metadata_endpoint() -> None:
    shard = refit_pb2.WeightVersionShard(metadata_endpoint="trainer:9000")

    assert shard.metadata_endpoint == "trainer:9000"
    assert (
        refit_pb2.WeightVersion.DESCRIPTOR.fields_by_name["object_storage"].number == 10
    )
    assert list(refit_pb2.ObjectStorageSource.DESCRIPTOR.fields_by_name) == [
        "uri",
        "storage_type",
    ]
    assert refit_pb2.OBJECT_STORAGE_TYPE_S3 == 1


def test_refit_service_uses_named_response_messages():
    service = refit_pb2.DESCRIPTOR.services_by_name["RefitService"]

    assert len(service.methods) == 15
    assert all(
        method.output_type.name.endswith("Response") for method in service.methods
    )


def test_stable_and_version_metadata_have_independent_publication_lifetimes(lifecycle_trainer) -> None:
    service, worker = lifecycle_trainer.publisher, lifecycle_trainer.worker
    manifest = TrainerShardMetadata(
        data=b'{"tensor_count":1,"total_bytes":2}',
        tensor_count=1,
        total_bytes=2,
        transport="NIXL",
    )
    version_metadata = refit_pb2.WeightVersionShardMetadata(
        version_id="version-a",
        worker_id="trainer-0",
        logical_shard_id="rank:0",
        stable_metadata_digest=manifest.digest,
        checksums=[refit_pb2.ShardChecksum(tensor_name="weight", digest="digest")],
    )
    with pytest.raises(ValueError):
        service.publish_version_metadata(version_metadata)
    assert service.publish_metadata(metadata=manifest) == service.endpoint
    request = refit_pb2.GetWeightVersionShardMetadataRequest(
        version_id="version-a", logical_shard_id="rank:0",
    )
    stable_request = refit_pb2.GetTrainerShardMetadataRequest(
        metadata_digest=manifest.digest,
    )
    stable = worker.GetTrainerShardMetadata(stable_request)
    assert stable.metadata == manifest.data
    assert stable.metadata_digest == manifest.digest
    with pytest.raises(grpc.RpcError) as unpublished:
        worker.GetWeightVersionShardMetadata(request)
    assert unpublished.value.code() is grpc.StatusCode.NOT_FOUND
    service.publish_version_metadata(version_metadata)
    service.publish_version_metadata(metadata=version_metadata)
    assert worker.GetWeightVersionShardMetadata(request).metadata == version_metadata
    conflicting = refit_pb2.WeightVersionShardMetadata()
    conflicting.CopyFrom(version_metadata)
    conflicting.checksums[0].digest = "different-content"
    with pytest.raises(ValueError):
        service.publish_version_metadata(conflicting)
    assert worker.GetWeightVersionShardMetadata(request).metadata == version_metadata
    replacement_metadata = TrainerShardMetadata(
        data=b'{"tensor_count":1,"total_bytes":2,"address":8192}',
        tensor_count=1, total_bytes=2, transport="NIXL",
    )
    with pytest.raises(ValueError):
        service.publish_metadata(metadata=replacement_metadata)
    assert worker.GetTrainerShardMetadata(stable_request) == stable
    service.release_version_metadata(version_id="version-a", logical_shard_id="rank:0")
    with pytest.raises(grpc.RpcError) as released:
        worker.GetWeightVersionShardMetadata(request)
    assert released.value.code() is grpc.StatusCode.NOT_FOUND
    assert worker.GetTrainerShardMetadata(stable_request) == stable
    service.release_metadata()
    service.publish_metadata(metadata=replacement_metadata)
    with pytest.raises(grpc.RpcError) as replaced:
        worker.GetTrainerShardMetadata(stable_request)
    assert replaced.value.code() is grpc.StatusCode.NOT_FOUND
    assert worker.GetTrainerShardMetadata(refit_pb2.GetTrainerShardMetadataRequest(
        metadata_digest=replacement_metadata.digest,
    )).metadata == replacement_metadata.data
    old_coverage_request = refit_pb2.GetTrainerShardMetadataRequest(
        metadata_digest=hashlib.sha256(_Adapter.bound_manifest).hexdigest(),
    )
    assert worker.GetTrainerShardMetadata(old_coverage_request).metadata == _Adapter.bound_manifest
    changed_coverage = json.loads(_Adapter.bound_manifest)
    changed_coverage["tensors"][0]["full_shape"] = [8]
    changed_coverage["tensors"][0]["shards"][0]["shape"] = [8]
    new_coverage = json.dumps(changed_coverage).encode()
    service.publish_binding(new_coverage)
    with pytest.raises(grpc.RpcError) as old_binding:
        worker.GetTrainerShardMetadata(old_coverage_request)
    assert old_binding.value.code() is grpc.StatusCode.NOT_FOUND
    assert worker.GetTrainerShardMetadata(refit_pb2.GetTrainerShardMetadataRequest(
        metadata_digest=hashlib.sha256(new_coverage).hexdigest(),
    )).metadata == new_coverage


def test_trainer_stages_then_publishes_one_rank_local_shard(monkeypatch):
    service = _RefitService()
    service.mesh_id = "mesh-a"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    refit_pb2_grpc.add_RefitServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    manifest_service = RefitWorkerService(endpoint=f"127.0.0.1:{port}")
    refit_pb2_grpc.add_RefitWorkerServiceServicer_to_server(manifest_service, server)
    server.start()
    adapter = _Adapter()
    monkeypatch.setattr(
        runtime_module, "_create_trainer_adapter", lambda *_args, **_kwargs: adapter
    )
    monkeypatch.setenv("MX_MODEL_NAME_OVERRIDE", "test/model")
    monkeypatch.setenv("MX_TRAINER_STAGING_MODE", "COPY_TO_DEVICE")
    monkeypatch.setenv("MX_WEIGHT_PAYLOAD_FORMAT", "FULL_TENSOR")
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    monkeypatch.setenv("MX_WORKER_HOST", "127.0.0.1")
    monkeypatch.setenv("MX_WORKER_GRPC_PORT", str(port))
    _patch_resources(
        monkeypatch,
        manager=_Manager(),
        manifest_service=manifest_service,
        worker_endpoint=f"127.0.0.1:{port}",
    )

    try:
        trainer = ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                engine_context=FSDPTrainerContext(),
                device_id=0,
                worker_id="trainer-0",
                server_url=f"127.0.0.1:{port}",
                registration_ttl_seconds=1,
            )
        )
        deadline = time.monotonic() + 10.0
        while service.registration_count < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert service.registration_count >= 2
        with pytest.raises(ValueError, match="must not be None"):
            trainer.bind_tensors(None)
        metadata = trainer.bind_tensors("model")
        assert metadata.metadata_endpoint == f"127.0.0.1:{port}"
        assert len(metadata.logical_shard_id) == 64
        assert trainer.logical_shard_id == metadata.logical_shard_id
        assert adapter.calls == []
        with pytest.raises(RuntimeError, match="already bound"):
            trainer.bind_tensors("replacement")
        assert service.shards == []
        with grpc.insecure_channel(metadata.metadata_endpoint) as channel:
            binding = refit_pb2_grpc.RefitWorkerServiceStub(
                channel
            ).GetTrainerShardMetadata(
                refit_pb2.GetTrainerShardMetadataRequest(
                    metadata_digest=metadata.logical_shard_id
                )
            )
        assert binding.metadata == adapter.bound_manifest
        assert binding.metadata_digest == metadata.logical_shard_id
        trainer.publish_version(version=WeightVersionRef("version-a"))
        metrics = trainer.pop_metrics()
        assert metrics["trainer_refit_e2e_s"] >= 0
        assert metrics["publication_rpc_s"] >= 0

        worker_stub = refit_pb2_grpc.RefitWorkerServiceStub(
            grpc.insecure_channel(service.shards[0].metadata_endpoint)
        )
        fetched = worker_stub.GetTrainerShardMetadata(
            refit_pb2.GetTrainerShardMetadataRequest(
                metadata_digest=service.shards[0].stable_metadata_digest,
            )
        )
        version_metadata = worker_stub.GetWeightVersionShardMetadata(
            refit_pb2.GetWeightVersionShardMetadataRequest(
                version_id="version-a", logical_shard_id=metadata.logical_shard_id,
            )
        )
        assert version_metadata.metadata.version_id == "version-a"
        assert version_metadata.metadata.worker_id == "trainer-0"
        assert version_metadata.metadata.stable_metadata_digest == fetched.metadata_digest
        assert [(row.tensor_name, row.shard_index, row.digest)
                for row in version_metadata.metadata.checksums] == [("weight", 0, "weight-digest")]
        assert version_metadata.version_metadata_digest == service.shards[0].version_metadata_digest
        assert version_metadata.version_metadata_digest == hashlib.sha256(
            version_metadata.metadata.SerializeToString(deterministic=True)
        ).hexdigest()
        with pytest.raises(RuntimeError, match="canonical-delta"):
            trainer.stage_shard(
                version=WeightVersionRef("version-a"),
                tensors="model-2",
            )
        with pytest.raises(RuntimeError, match="canonical-delta"):
            trainer.stage_shard(
                version=WeightVersionRef("version-a"),
                tensor_iter=iter([]),
            )
        trainer.publish_version(version=WeightVersionRef("version-a"))
        retained_owners = [
            staged.buffer_owner for staged in trainer._runtime.method.published["version-a"]
        ]
        trainer.release_version(version=WeightVersionRef("version-a"))
        trainer.release_version(version=WeightVersionRef("version-a"))
        with pytest.raises(grpc.RpcError) as released:
            worker_stub.GetWeightVersionShardMetadata(
                refit_pb2.GetWeightVersionShardMetadataRequest(
                    version_id="version-a",
                    logical_shard_id=metadata.logical_shard_id,
                )
            )
        assert released.value.code() is grpc.StatusCode.NOT_FOUND
    finally:
        if "trainer" in locals():
            trainer.close()
        server.stop(grace=None).wait()

    assert adapter.calls == [
        (
            "model",
            TrainerStagingMode.COPY_TO_DEVICE,
            WeightPayloadFormat.FULL_TENSOR,
        ),
        (
            "model",
            TrainerStagingMode.COPY_TO_DEVICE,
            WeightPayloadFormat.FULL_TENSOR,
        ),
    ]
    assert retained_owners == ["model", "model"]
    assert len(service.shards) == 2
    assert len(service.deleted_shards) == 1
    assert service.deleted_shards[0].logical_shard_id == metadata.logical_shard_id
    assert service.registrations["trainer-0"].role == refit_pb2.WORKER_ROLE_TRAINER
    assert service.shards[0].version_id == "version-a"
    assert service.shards[0].logical_shard_id == metadata.logical_shard_id
    assert service.shards[0].worker_id == "trainer-0"
    assert json.loads(fetched.metadata)["tensor_count"] == 2
    assert json.loads(fetched.metadata)["total_bytes"] == 128
    assert service.shards[0].metadata_endpoint == f"127.0.0.1:{port}"
    assert json.loads(fetched.metadata)["tensors"][0]["name"] == "weight"
    assert metadata.logical_shard_id == hashlib.sha256(
        bound_tensor_manifest(json.loads(fetched.metadata)["tensors"])
    ).hexdigest()


def test_bound_manifest_excludes_process_addresses_and_weight_content():
    first = {"agent_name": "worker-a", "tensors": [{
        "name": "w", "dtype": "torch.bfloat16", "elsize": 2,
        "full_shape": [4], "shards": [{"shape": [2], "shard_offset": [0],
            "addr": 1234, "device_id": 0, "agent_name": "worker-a", "digest": "old"}],
    }]}
    second = json.loads(json.dumps(first))
    second["agent_name"] = "worker-b"
    second["tensors"][0]["shards"][0].update(addr=5678, device_id=1, agent_name="worker-b", digest="new")
    assert bound_tensor_manifest(first["tensors"]) == bound_tensor_manifest(second["tensors"])
    second["tensors"][0]["shards"][0]["shard_offset"] = [2]
    assert bound_tensor_manifest(first["tensors"]) != bound_tensor_manifest(second["tensors"])


def test_trainer_initialization_rejects_unspecified_fixed_settings(monkeypatch):
    monkeypatch.setenv("MX_WORKER_HOST", "trainer")
    with pytest.raises(ValueError, match="staging_mode must be specified"):
        ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                engine_context=FSDPTrainerContext(),
                model_name="test/model",
                staging_mode=TrainerStagingMode.UNSPECIFIED,
                payload_format=WeightPayloadFormat.FULL_TENSOR,
            )
        )

    with pytest.raises(ValueError, match="payload_format must be specified"):
        ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                engine_context=FSDPTrainerContext(),
                model_name="test/model",
                staging_mode=TrainerStagingMode.COPY_TO_DEVICE,
                payload_format=WeightPayloadFormat.UNSPECIFIED,
            )
        )


def test_trainer_initialization_rejects_adapter_unsupported_mode(monkeypatch):
    adapter = _Adapter()
    monkeypatch.setattr(
        runtime_module, "_create_trainer_adapter", lambda *_args, **_kwargs: adapter
    )
    monkeypatch.setenv("MX_WORKER_HOST", "trainer")
    resources = _patch_resources(
        monkeypatch,
        manager=_Manager(),
        manifest_service=object(),
        worker_endpoint="trainer:9000",
    )
    monkeypatch.setattr(
        ModelExpressTrainerClient, "_register_worker", lambda self: None
    )
    monkeypatch.setattr(client_module.threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(client_module.threading.Thread, "join", lambda self: None)
    trainer = ModelExpressTrainerClient.initialize(
        ModelExpressTrainerConfig(
            engine_context=FSDPTrainerContext(),
            device_id=0,
            model_name="test/model",
            staging_mode=TrainerStagingMode.IN_PLACE,
            payload_format=WeightPayloadFormat.FULL_TENSOR,
        )
    )
    with pytest.raises(ValueError, match="does not support staging mode IN_PLACE"):
        _ = trainer.logical_shard_id
    resources.close.assert_called_once_with()


def test_trainer_initialization_requires_explicit_engine_context(monkeypatch):
    monkeypatch.setenv("MX_WORKER_HOST", "trainer")
    resources = _patch_resources(
        monkeypatch,
        manager=_Manager(),
        manifest_service=object(),
        worker_endpoint="trainer:9000",
    )
    monkeypatch.setattr(
        ModelExpressTrainerClient, "_register_worker", lambda self: None
    )
    monkeypatch.setattr(client_module.threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(client_module.threading.Thread, "join", lambda self: None)
    with pytest.raises(ValueError, match="engine_context is required"):
        ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                device_id=0,
                model_name="test/model",
                staging_mode=TrainerStagingMode.IN_PLACE,
                payload_format=WeightPayloadFormat.FULL_TENSOR,
            )
        )
    resources.close.assert_not_called()


def test_trainer_client_owns_default_transport_resources(monkeypatch):
    manager = MagicMock(listen_port=19002)
    resources = MagicMock(
        manager=manager,
        manifest_service=MagicMock(),
        worker_endpoint="trainer:19002",
    )
    adapter = _Adapter()
    adapter_factory = MagicMock(return_value=adapter)
    monkeypatch.setenv("MX_WORKER_HOST", "trainer")
    initialize_resources = MagicMock(return_value=resources)
    monkeypatch.setattr(
        runtime_module._TrainerResources,
        "initialize",
        initialize_resources,
    )
    monkeypatch.setattr(runtime_module, "_create_trainer_adapter", adapter_factory)
    monkeypatch.setattr(
        ModelExpressTrainerClient, "_register_worker", lambda self: None
    )
    monkeypatch.setattr(client_module.threading.Thread, "start", lambda self: None)
    monkeypatch.setattr(client_module.threading.Thread, "join", lambda self: None)

    trainer = ModelExpressTrainerClient.initialize(
        ModelExpressTrainerConfig(
            engine_context=FSDPTrainerContext(),
            model_name="test/model",
            device_id=2,
            agent_name="trainer-agent",
            staging_mode=TrainerStagingMode.COPY_TO_DEVICE,
            server_url="mx:8000",
        )
    )
    adapter_factory.assert_not_called()
    with pytest.raises(RuntimeError, match="bind_tensors"):
        _ = trainer.logical_shard_id
    metadata = trainer.bind_tensors("model")
    assert trainer.logical_shard_id == metadata.logical_shard_id
    assert len(adapter_factory.call_args.args) == 1
    assert isinstance(adapter_factory.call_args.args[0], FSDPTrainerContext)
    assert adapter_factory.call_args.kwargs == {
        "manager": manager,
        "nixl_metadata_endpoint": "trainer:19002",
    }
    initialize_resources.assert_called_once_with(
        device_id=2,
        agent_name="trainer-agent",
    )
    trainer.close()
    resources.close.assert_called_once_with()


def test_trainer_initialization_cleans_up_when_renewal_thread_cannot_start(
    monkeypatch,
):
    manager = MagicMock(listen_port=19002)
    resources = MagicMock(
        manager=manager,
        manifest_service=MagicMock(),
        worker_endpoint="trainer:19002",
    )
    monkeypatch.setenv("MX_WORKER_HOST", "trainer")
    monkeypatch.setattr(
        runtime_module,
        "_create_trainer_adapter",
        lambda *_args, **_kwargs: _Adapter(),
    )
    monkeypatch.setattr(
        runtime_module._TrainerResources,
        "initialize",
        MagicMock(return_value=resources),
    )
    monkeypatch.setattr(
        ModelExpressTrainerClient, "_register_worker", lambda self: None
    )
    monkeypatch.setattr(
        client_module.threading.Thread,
        "start",
        lambda _self: (_ for _ in ()).throw(RuntimeError("thread start failed")),
    )

    with pytest.raises(RuntimeError, match="thread start failed"):
        ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                engine_context=FSDPTrainerContext(),
                model_name="test/model",
                device_id=2,
                staging_mode=TrainerStagingMode.COPY_TO_DEVICE,
                server_url="mx:8000",
            )
        )

    resources.close.assert_called_once_with()


def test_binding_manifest_is_available_before_version_publication():
    service = RefitWorkerService(endpoint="127.0.0.1:9000")
    manifest = _Adapter.bound_manifest
    service.publish_binding(manifest)
    binding_id = hashlib.sha256(manifest).hexdigest()
    response = service.GetTrainerShardMetadata(
        refit_pb2.GetTrainerShardMetadataRequest(metadata_digest=binding_id),
        None,
    )
    assert response.metadata == manifest
    assert response.metadata_digest == binding_id
    service.release_version_metadata(version_id="version-a", logical_shard_id=binding_id)
    assert service.GetTrainerShardMetadata(
        refit_pb2.GetTrainerShardMetadataRequest(metadata_digest=binding_id),
        None,
    ).metadata == manifest


@pytest.fixture
def lifecycle_trainer(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    service = _RefitService()
    service.mesh_id = "mesh-a"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    refit_pb2_grpc.add_RefitServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    publisher = RefitWorkerService(endpoint=f"127.0.0.1:{port}")
    refit_pb2_grpc.add_RefitWorkerServiceServicer_to_server(publisher, server)
    adapter = _Adapter()
    monkeypatch.setattr(runtime_module, "_create_trainer_adapter", lambda *_args, **_kwargs: adapter)
    monkeypatch.setenv("MX_WORKER_HOST", "127.0.0.1")
    _patch_resources(monkeypatch, manager=_Manager(), manifest_service=publisher,
                     worker_endpoint=publisher.endpoint)
    server.start()
    trainer = ModelExpressTrainerClient.initialize(ModelExpressTrainerConfig(
        engine_context=FSDPTrainerContext(), model_name="test/model", device_id=0,
        worker_id="trainer-lifecycle", server_url=publisher.endpoint,
        rpc_timeout_seconds=0.1, registration_ttl_seconds=60,
        staging_mode=TrainerStagingMode.COPY_TO_DEVICE,
        payload_format=WeightPayloadFormat.FULL_TENSOR,
    ))
    trainer.bind_tensors("model")
    with grpc.insecure_channel(publisher.endpoint) as channel:
        case = SimpleNamespace(
            trainer=trainer, service=service, adapter=adapter, publisher=publisher,
            worker=refit_pb2_grpc.RefitWorkerServiceStub(channel),
        )
        try:
            yield case
        finally:
            if not trainer._closed:
                for version_id in ("version-a", "version-b", "version-c"):
                    trainer.release_version(version=WeightVersionRef(version_id))
                trainer.close()
            server.stop(grace=None).wait()


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("change", ["mesh", "generation"])
def test_replacement_waits_for_every_published_version_to_release(
    lifecycle_trainer: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    verify: bool, change: str,
) -> None:
    case = lifecycle_trainer
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", str(int(verify)))
    for version_id in ("version-a", "version-b"):
        case.trainer.publish_version(version=WeightVersionRef(version_id))
    case.trainer.release_version(version=WeightVersionRef("version-a"))
    target = case.service
    field, value = {
        "mesh": ("mesh_id", "mesh-b"), "generation": ("mesh_generation", 2),
    }[change]
    previous = getattr(target, field)
    setattr(target, field, value)
    with pytest.raises(RuntimeError):
        case.trainer.publish_version(version=WeightVersionRef("version-c"))
    setattr(target, field, previous)
    case.trainer.publish_version(version=WeightVersionRef("version-b"))
    case.trainer.release_version(version=WeightVersionRef("version-b"))
    setattr(target, field, value)
    case.trainer.publish_version(version=WeightVersionRef("version-c"))
    assert [publication.version_id for publication in case.service.shards] == [
        "version-a", "version-b", "version-b", "version-c",
    ]


def test_same_mesh_generation_keeps_stable_metadata_digest(
    lifecycle_trainer: SimpleNamespace,
) -> None:
    case = lifecycle_trainer
    case.trainer.publish_version(version=WeightVersionRef("version-a"))
    case.adapter.metadata_revision = 1
    with pytest.raises(AssertionError):
        case.trainer.publish_version(version=WeightVersionRef("version-b"))


@pytest.mark.parametrize("failure", ["release", "publish"])
@pytest.mark.parametrize("retry_revision", [0, 1])
def test_failed_metadata_replacement_can_recover_old_or_new_storage(
    lifecycle_trainer: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    failure: str, retry_revision: int,
) -> None:
    case = lifecycle_trainer
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "0")
    case.trainer.publish_version(version=WeightVersionRef("version-a"))
    original = case.service.shards[-1]
    stable_request = refit_pb2.GetTrainerShardMetadataRequest(
        metadata_digest=original.stable_metadata_digest,
    )
    case.trainer.release_version(version=WeightVersionRef("version-a"))
    operation = "release_metadata" if failure == "release" else "publish_metadata"
    original_operation = getattr(case.publisher, operation)

    def fail_replacement(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("metadata store failed")

    monkeypatch.setattr(case.publisher, operation, fail_replacement)
    case.service.mesh_generation = 2
    case.adapter.metadata_revision = 1
    with pytest.raises(RuntimeError, match="metadata store failed"):
        case.trainer.publish_version(version=WeightVersionRef("version-b"))
    assert [shard.version_id for shard in case.service.shards] == ["version-a"]
    if failure == "release":
        assert case.worker.GetTrainerShardMetadata(stable_request).metadata
    else:
        with pytest.raises(grpc.RpcError) as retired:
            case.worker.GetTrainerShardMetadata(stable_request)
        assert retired.value.code() is grpc.StatusCode.NOT_FOUND
    monkeypatch.setattr(case.publisher, operation, original_operation)
    case.adapter.metadata_revision = retry_revision
    case.trainer.publish_version(version=WeightVersionRef("version-b"))
    recovered = case.service.shards[-1]
    response = case.worker.GetTrainerShardMetadata(
        refit_pb2.GetTrainerShardMetadataRequest(metadata_digest=recovered.stable_metadata_digest),
    )
    assert json.loads(response.metadata)["storage_revision"] == retry_revision
    assert recovered.version_id == "version-b"


@pytest.mark.parametrize("already_published", [False, True])
def test_registration_failure_retains_version_metadata_until_release(
    lifecycle_trainer: SimpleNamespace, monkeypatch: pytest.MonkeyPatch,
    already_published: bool,
) -> None:
    case = lifecycle_trainer
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", "1")
    version = WeightVersionRef("version-a")
    request = refit_pb2.GetWeightVersionShardMetadataRequest(
        version_id=version.version_id, logical_shard_id=case.adapter.logical_shard_id,
    )
    if already_published:
        case.trainer.publish_version(version=version)
        original = case.worker.GetWeightVersionShardMetadata(request)

    with monkeypatch.context() as rejected:
        rejected.setattr(case.service, "registrations", {})
        with pytest.raises(grpc.RpcError) as error:
            case.trainer.publish_version(version=version)
        assert error.value.code() is grpc.StatusCode.FAILED_PRECONDITION
        assert error.value.details() == "worker not registered"

    if already_published:
        assert case.worker.GetWeightVersionShardMetadata(request) == original
    assert case.worker.GetWeightVersionShardMetadata(request).metadata.version_id == version.version_id

    case.trainer.release_version(version=version)
    case.service.mesh_generation = 2
    case.adapter.metadata_revision = 1
    case.trainer.publish_version(version=WeightVersionRef("version-b"))
    recovered = case.service.shards[-1]
    assert recovered.version_id == "version-b"
    assert case.worker.GetWeightVersionShardMetadata(
        refit_pb2.GetWeightVersionShardMetadataRequest(
            version_id="version-b", logical_shard_id=case.adapter.logical_shard_id,
        )
    ).metadata.stable_metadata_digest == recovered.stable_metadata_digest


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("cleanup", ["release", "close"])
def test_lost_publication_response_retains_buffers_until_cleanup(
    lifecycle_trainer, monkeypatch, verify, cleanup,
) -> None:
    case = lifecycle_trainer
    monkeypatch.setenv("MX_RESHARD_PUBLISH_DIGEST", str(int(verify)))
    version = WeightVersionRef("version-a")
    with monkeypatch.context() as lost:
        lost.setattr(case.service, "publication_error", grpc.StatusCode.DEADLINE_EXCEEDED)
        with pytest.raises(grpc.RpcError):
            case.trainer.publish_version(version=version)
    method = case.trainer._runtime.method
    assert method.published[version.version_id][0].buffer_owner == "model"
    request = refit_pb2.GetWeightVersionShardMetadataRequest(
        version_id=version.version_id, logical_shard_id=case.adapter.logical_shard_id,
    )
    if verify:
        assert case.worker.GetWeightVersionShardMetadata(request).metadata.version_id == version.version_id
    if cleanup == "close":
        case.trainer.close()
        assert case.trainer._closed
        assert not method.published
        return
    with monkeypatch.context() as leased:
        leased.setattr(case.service, "lease_active", True)
        with pytest.raises(grpc.RpcError):
            case.trainer.release_version(version=version)
    assert method.published[version.version_id]
    case.trainer.release_version(version=version)
    assert version.version_id not in method.published
    if verify:
        with pytest.raises(grpc.RpcError) as retired:
            case.worker.GetWeightVersionShardMetadata(request)
        assert retired.value.code() is grpc.StatusCode.NOT_FOUND
    case.trainer.publish_version(version=WeightVersionRef("version-b"))
