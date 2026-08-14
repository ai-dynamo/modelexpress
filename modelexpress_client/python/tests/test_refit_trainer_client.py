# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from concurrent import futures

import grpc
import pytest

import modelexpress_rl.client as client_module
from modelexpress_rl import (
    CompletionFence,
    ModelExpressTrainerClient,
    StagedWeightVersionShardData,
    TrainerEngineAdapter,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionRef,
    WeightVersionShardManifest,
    WeightVersionShardManifestService,
    refit_pb2,
    refit_pb2_grpc,
)


class _RefitService(refit_pb2_grpc.RefitServiceServicer):
    def __init__(self):
        self.registrations = {}
        self.registration_count = 0
        self.shards = []

    def RegisterWorker(self, request, _context):
        self.registration_count += 1
        worker = request.worker
        worker.expires_at_unix_ms = 1234
        self.registrations[worker.worker_id] = worker
        return worker

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


class _Manager:
    listen_port = 19000


class _Adapter(TrainerEngineAdapter):
    source_slot_id = "rank:0"
    supported_staging_modes = frozenset({TrainerStagingMode.COPY_TO_DEVICE})
    supported_payload_formats = frozenset({WeightPayloadFormat.FULL_TENSOR})

    def __init__(self):
        self.calls = []

    def stage_shard(self, *, tensors, staging_mode, payload_format):
        self.calls.append((tensors, staging_mode, payload_format))

        return StagedWeightVersionShardData(
            manifest=WeightVersionShardManifest(
                data=b"manifest",
                tensor_count=2,
                total_bytes=128,
                transport="NIXL",
            ),
            publish_ready=CompletionFence(lambda: None),
            source_reuse_ready=CompletionFence(lambda: None),
            buffer_owner=tensors,
        )


def test_trainer_stages_then_publishes_one_rank_local_shard(monkeypatch):
    service = _RefitService()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    refit_pb2_grpc.add_RefitServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    manifest_service = WeightVersionShardManifestService(endpoint=f"127.0.0.1:{port}")
    refit_pb2_grpc.add_RefitWorkerServiceServicer_to_server(manifest_service, server)
    server.start()
    adapter = _Adapter()
    monkeypatch.setattr(client_module, "_trainer_adapter", lambda **_kwargs: adapter)
    monkeypatch.setenv("MODEL_NAME", "test/model")
    monkeypatch.setenv("MX_TRAINER_STAGING_MODE", "COPY_TO_DEVICE")
    monkeypatch.setenv("MX_WEIGHT_PAYLOAD_FORMAT", "FULL_TENSOR")
    monkeypatch.setenv("MX_WORKER_HOST", "127.0.0.1")
    monkeypatch.setenv("MX_WORKER_GRPC_PORT", str(port))

    try:
        trainer = ModelExpressTrainerClient.initialize(
            manager=_Manager(),
            manifest_publisher=manifest_service,
            worker_id="trainer-0",
            server_url=f"127.0.0.1:{port}",
            registration_ttl_seconds=1,
        )
        deadline = time.monotonic() + 10.0
        while service.registration_count < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert service.registration_count >= 2
        shard = trainer.stage_shard(
            version=WeightVersionRef("version-a"),
            tensors="model",
        )

        assert service.shards == []
        shard.source_reuse_ready.wait()
        shard.publish()
        shard.publish()

        worker_stub = refit_pb2_grpc.RefitWorkerServiceStub(
            grpc.insecure_channel(service.shards[0].manifest_endpoint)
        )
        fetched = worker_stub.GetWeightVersionShardManifest(
            refit_pb2.GetWeightVersionShardManifestRequest(
                version_id="version-a",
                source_slot_id="rank:0",
            )
        )
        second = trainer.stage_shard(
            version=WeightVersionRef("version-a"),
            tensors="model-2",
        )
        second.publish()
        retained_owners = [
            staged.buffer_owner for staged in trainer._published_shards["version-a"]
        ]
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
            "model-2",
            TrainerStagingMode.COPY_TO_DEVICE,
            WeightPayloadFormat.FULL_TENSOR,
        ),
    ]
    assert retained_owners == ["model", "model-2"]
    assert len(service.shards) == 2
    assert service.registrations["trainer-0"].role == refit_pb2.WORKER_ROLE_TRAINER
    assert (
        service.registrations["trainer-0"].endpoint
        == service.shards[0].manifest_endpoint
    )
    assert service.shards[0].version_id == "version-a"
    assert service.shards[0].source_slot_id == "rank:0"
    assert service.shards[0].worker_id == "trainer-0"
    assert service.shards[0].tensor_count == 2
    assert service.shards[0].total_bytes == 128
    assert fetched.manifest == b"manifest"


def test_trainer_initialization_rejects_unspecified_fixed_settings(monkeypatch):
    monkeypatch.setenv("MX_WORKER_HOST", "trainer")
    with pytest.raises(ValueError, match="staging_mode must be specified"):
        ModelExpressTrainerClient.initialize(
            model_name="test/model",
            manager=_Manager(),
            staging_mode=TrainerStagingMode.UNSPECIFIED,
            payload_format=WeightPayloadFormat.FULL_TENSOR,
            manifest_publisher=object(),
            worker_endpoint="trainer:9000",
        )

    with pytest.raises(ValueError, match="payload_format must be specified"):
        ModelExpressTrainerClient.initialize(
            model_name="test/model",
            manager=_Manager(),
            staging_mode=TrainerStagingMode.COPY_TO_DEVICE,
            payload_format=WeightPayloadFormat.UNSPECIFIED,
            manifest_publisher=object(),
            worker_endpoint="trainer:9000",
        )


def test_trainer_initialization_rejects_adapter_unsupported_mode(monkeypatch):
    adapter = _Adapter()
    monkeypatch.setattr(client_module, "_trainer_adapter", lambda **_kwargs: adapter)
    monkeypatch.setenv("MX_WORKER_HOST", "trainer")
    with pytest.raises(ValueError, match="does not support staging mode IN_PLACE"):
        ModelExpressTrainerClient.initialize(
            model_name="test/model",
            manager=_Manager(),
            staging_mode=TrainerStagingMode.IN_PLACE,
            payload_format=WeightPayloadFormat.FULL_TENSOR,
            manifest_publisher=object(),
            worker_endpoint="trainer:9000",
        )


def test_trainer_initialization_rejects_unknown_configured_engine(monkeypatch):
    monkeypatch.setenv("MX_TRAINER_ENGINE", "unknown")
    monkeypatch.setenv("MX_WORKER_HOST", "trainer")
    with pytest.raises(ValueError, match="unsupported MX_TRAINER_ENGINE='UNKNOWN'"):
        ModelExpressTrainerClient.initialize(
            model_name="test/model",
            manager=_Manager(),
            staging_mode=TrainerStagingMode.IN_PLACE,
            payload_format=WeightPayloadFormat.FULL_TENSOR,
            manifest_publisher=object(),
            worker_endpoint="trainer:9000",
        )
