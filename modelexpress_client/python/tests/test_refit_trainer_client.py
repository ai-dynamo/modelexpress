# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import json
import hashlib
from concurrent import futures
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
    WeightVersionShardManifest,
)
from modelexpress_rl.train.manifest import WeightVersionShardManifestService, bound_tensor_manifest


class _RefitService(refit_pb2_grpc.RefitServiceServicer):
    def __init__(self):
        self.registrations = {}
        self.registration_count = 0
        self.shards = []
        self.deleted_shards = []
        self.mesh_id = None

    def GetWeightVersion(self, request, _context) -> refit_pb2.GetWeightVersionResponse:
        version = refit_pb2.WeightVersion(uid=request.uid)
        if self.mesh_id is not None:
            version.trainer_mesh_id = self.mesh_id
            version.trainer_mesh_generation = 1
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
        return refit_pb2.CreateWeightVersionShardResponse(
            shard=shard,
            version=refit_pb2.WeightVersion(
                uid=shard.version_id,
                state=refit_pb2.WEIGHT_VERSION_STATE_READY,
            ),
        )

    def DeleteWeightVersionShard(self, request, _context):
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

    def bind_tensors(self, tensors):
        if tensors is None:
            raise ValueError("tensors must not be None")
        return self.logical_shard_id

    def stage_shard(self, *, tensors, staging_mode, payload_format):
        self.calls.append((tensors, staging_mode, payload_format))

        return StagedWeightVersionShardData(
            manifest=WeightVersionShardManifest(
                data=json.dumps({"tensors": [{
                    "name": "weight", "dtype": "torch.bfloat16", "elsize": 2,
                    "full_shape": [4], "shards": [{"shard_offset": [0], "shape": [4]}],
                }]}).encode(),
                tensor_count=2,
                total_bytes=128,
                transport="NIXL",
            ),
            publish_ready=CompletionFence(lambda: None),
            buffer_owner=tensors,
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


def test_refit_shard_keeps_only_its_nixl_manifest_endpoint():
    shard = refit_pb2.WeightVersionShard(manifest_endpoint="trainer:9000")

    assert shard.manifest_endpoint == "trainer:9000"
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


def test_released_manifests_do_not_accumulate():
    service = WeightVersionShardManifestService(endpoint="trainer:9000")
    manifest = WeightVersionShardManifest(
        data=b"manifest",
        tensor_count=1,
        total_bytes=2,
        transport="NIXL",
    )

    for index in range(1000):
        version_id = f"version-{index}"
        service.publish_manifest(
            version_id=version_id,
            logical_shard_id="rank:0",
            manifest=manifest,
        )
        service.release_manifest(
            version_id=version_id,
            logical_shard_id="rank:0",
        )

    assert service._manifests == {}


def test_trainer_stages_then_publishes_one_rank_local_shard(monkeypatch):
    service = _RefitService()
    service.mesh_id = "mesh-a"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    refit_pb2_grpc.add_RefitServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    manifest_service = WeightVersionShardManifestService(endpoint=f"127.0.0.1:{port}")
    refit_pb2_grpc.add_RefitWorkerServiceServicer_to_server(manifest_service, server)
    server.start()
    adapter = _Adapter()
    monkeypatch.setattr(
        runtime_module, "_create_trainer_adapter", lambda *_args, **_kwargs: adapter
    )
    monkeypatch.setenv("MX_MODEL_NAME_OVERRIDE", "test/model")
    monkeypatch.setenv("MX_TRAINER_STAGING_MODE", "COPY_TO_DEVICE")
    monkeypatch.setenv("MX_WEIGHT_PAYLOAD_FORMAT", "FULL_TENSOR")
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
            ).GetWeightVersionShardManifest(
                refit_pb2.GetWeightVersionShardManifestRequest(
                    logical_shard_id=metadata.logical_shard_id
                )
            )
        assert binding.manifest == adapter.bound_manifest
        assert binding.manifest_digest == metadata.logical_shard_id
        trainer.publish_version(version=WeightVersionRef("version-a"))
        metrics = trainer.pop_metrics()
        assert metrics["trainer_refit_e2e_s"] >= 0
        assert metrics["publication_rpc_s"] >= 0

        worker_stub = refit_pb2_grpc.RefitWorkerServiceStub(
            grpc.insecure_channel(service.shards[0].manifest_endpoint)
        )
        fetched = worker_stub.GetWeightVersionShardManifest(
            refit_pb2.GetWeightVersionShardManifestRequest(
                version_id="version-a",
                logical_shard_id=metadata.logical_shard_id,
            )
        )
        with pytest.raises(TypeError, match="tensors"):
            trainer.stage_shard(
                version=WeightVersionRef("version-a"),
                tensors="model-2",
                hf_tensor_iter=iter([]),
            )
        with pytest.raises(RuntimeError, match="canonical-delta"):
            trainer.stage_shard(
                version=WeightVersionRef("version-a"),
                hf_tensor_iter=iter([]),
            )
        trainer.publish_version(version=WeightVersionRef("version-a"))
        method = trainer._runtime.method
        retained_owners = [
            staged.buffer_owner for staged in method.published["version-a"]
        ]
        trainer.release_version(version=WeightVersionRef("version-a"))
        trainer.release_version(version=WeightVersionRef("version-a"))
        assert "version-a" not in method.published
        with pytest.raises(grpc.RpcError) as released:
            worker_stub.GetWeightVersionShardManifest(
                refit_pb2.GetWeightVersionShardManifestRequest(
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
    assert "endpoint" not in refit_pb2.WorkerRegistration.DESCRIPTOR.fields_by_name
    assert service.shards[0].version_id == "version-a"
    assert service.shards[0].logical_shard_id == metadata.logical_shard_id
    assert service.shards[0].worker_id == "trainer-0"
    assert service.shards[0].tensor_count == 2
    assert service.shards[0].total_bytes == 128
    assert service.shards[0].manifest_endpoint == f"127.0.0.1:{port}"
    assert json.loads(fetched.manifest)["tensors"][0]["name"] == "weight"
    assert metadata.logical_shard_id == hashlib.sha256(
        bound_tensor_manifest(json.loads(fetched.manifest)["tensors"])
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
    service = WeightVersionShardManifestService(endpoint="127.0.0.1:9000")
    manifest = _Adapter.bound_manifest
    service.publish_binding(manifest)
    binding_id = hashlib.sha256(manifest).hexdigest()
    response = service.GetWeightVersionShardManifest(
        refit_pb2.GetWeightVersionShardManifestRequest(
            version_id="", logical_shard_id=binding_id
        ),
        None,
    )
    assert response.manifest == manifest
    assert response.manifest_digest == binding_id
    service.release_manifest(version_id="version-a", logical_shard_id=binding_id)
    assert service.GetWeightVersionShardManifest(
        refit_pb2.GetWeightVersionShardManifestRequest(logical_shard_id=binding_id),
        None,
    ).manifest == manifest
