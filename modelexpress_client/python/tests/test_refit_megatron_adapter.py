# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from concurrent import futures
from types import SimpleNamespace

import grpc
import pytest

from modelexpress.refit.reshard.rendezvous import unwrap_rendezvous_blob
from modelexpress_rl import (
    ModelExpressTrainerClient,
    TrainerEngineAdapter,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionRef,
    WeightVersionShardManifestService,
    refit_pb2,
    refit_pb2_grpc,
)
from modelexpress_rl.train.engines.megatron import (
    MegatronTensorSpec,
    MegatronTrainerAdapter,
)


class _Tensor:
    shape = (8, 8)
    dtype = "torch.bfloat16"
    device = SimpleNamespace(index=0)

    def is_contiguous(self):
        return True

    def element_size(self):
        return 2

    def data_ptr(self):
        return 0x1234


class _Manager:
    agent_name = "trainer-r3"
    nixl_metadata = b"agent-metadata"
    listen_port = 19003


class _RefitService(refit_pb2_grpc.RefitServiceServicer):
    def __init__(self, events):
        self.events = events
        self.registration_ttl = None
        self.shard = None

    def RegisterWorker(self, request, _context):
        self.events.append("register-worker")
        self.registration_ttl = request.ttl_seconds
        return request.worker

    def CreateWeightVersionShard(self, request, _context):
        self.events.append("publish-version-shard")
        self.shard = request.shard
        return refit_pb2.CreateWeightVersionShardResponse(
            shard=request.shard,
            version=refit_pb2.WeightVersion(
                uid=request.shard.version_id,
                state=refit_pb2.WEIGHT_VERSION_STATE_READY,
            ),
        )


def test_megatron_adapter_requires_initialized_distributed_engine(monkeypatch):
    monkeypatch.setattr(
        "modelexpress_rl.train.engines.megatron.adapter.dist.is_initialized",
        lambda: False,
    )

    with pytest.raises(RuntimeError, match="distributed process group"):
        MegatronTrainerAdapter(
            manager=_Manager(),
            nixl_metadata_endpoint="10.0.0.3:19003",
        )


def test_megatron_adapter_uses_shared_trainer_publication_flow(monkeypatch):
    monkeypatch.setattr(
        "modelexpress_rl.train.engines.megatron.adapter.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "modelexpress_rl.train.engines.megatron.adapter.dist.get_rank",
        lambda: 3,
    )
    monkeypatch.setenv("MX_WORKER_HOST", "10.0.0.3")
    events = []
    refit_service = _RefitService(events)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    refit_pb2_grpc.add_RefitServiceServicer_to_server(refit_service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    manifest_service = WeightVersionShardManifestService(endpoint=f"127.0.0.1:{port}")
    refit_pb2_grpc.add_RefitWorkerServiceServicer_to_server(manifest_service, server)
    server.start()
    adapter = MegatronTrainerAdapter(
        manager=_Manager(),
        nixl_metadata_endpoint="10.0.0.3:19003",
    )
    tensors = [
        MegatronTensorSpec(
            name="column",
            tensor=_Tensor(),
            role="column",
            hf_names=("column",),
            global_shape=(16, 8),
            placement_kind="SHARD",
            shard_axis=0,
            local_shard_range=(8, 16),
        )
    ]

    with pytest.raises(NotImplementedError, match="COPY_TO_DEVICE staging"):
        adapter.stage_shard(
            tensors=tensors,
            staging_mode=TrainerStagingMode.COPY_TO_DEVICE,
            payload_format=WeightPayloadFormat.FULL_TENSOR,
        )
    with pytest.raises(NotImplementedError, match="XOR_DELTA payloads"):
        adapter.stage_shard(
            tensors=tensors,
            staging_mode=TrainerStagingMode.IN_PLACE,
            payload_format=WeightPayloadFormat.XOR_DELTA,
        )

    try:
        refit_client = ModelExpressTrainerClient.initialize(
            server_url=f"127.0.0.1:{port}",
            manager=_Manager(),
            model_name="model",
            staging_mode=TrainerStagingMode.IN_PLACE,
            payload_format=WeightPayloadFormat.FULL_TENSOR,
            manifest_publisher=manifest_service,
            worker_endpoint="trainer-3:9000",
            worker_id="worker-3",
            registration_ttl_seconds=60,
        )
        staged = refit_client.stage_shard(
            version=WeightVersionRef("version-a"),
            tensors=tensors,
        )
        with pytest.raises(NotImplementedError, match="version-retirement"):
            staged.source_reuse_ready.wait()
        staged.publish()
        worker_stub = refit_pb2_grpc.RefitWorkerServiceStub(
            grpc.insecure_channel(refit_service.shard.manifest_endpoint)
        )
        fetched = worker_stub.GetWeightVersionShardManifest(
            refit_pb2.GetWeightVersionShardManifestRequest(
                version_id="version-a",
                source_slot_id="publisher:global-rank:3",
            )
        )
    finally:
        if "refit_client" in locals():
            refit_client.close()
        server.stop(grace=None).wait()

    assert isinstance(adapter, TrainerEngineAdapter)
    assert isinstance(refit_client._adapter, MegatronTrainerAdapter)
    assert events == [
        "register-worker",
        "publish-version-shard",
    ]
    assert refit_service.registration_ttl == 60
    assert refit_service.shard.version_id == "version-a"
    assert refit_service.shard.source_slot_id == "publisher:global-rank:3"
    assert refit_service.shard.worker_id == "worker-3"
    assert refit_service.shard.tensor_count == 1
    assert refit_service.shard.total_bytes == 128
    assert (
        refit_service.shard.manifest_digest
        == hashlib.sha256(fetched.manifest).hexdigest()
    )
    assert refit_service.shard.manifest_endpoint == f"127.0.0.1:{port}"
    assert refit_service.shard.transport == "NIXL"
    assert fetched.manifest_digest == refit_service.shard.manifest_digest
    payload = unwrap_rendezvous_blob(fetched.manifest)
    assert payload.metadata_endpoint == "10.0.0.3:19003"
    assert payload.tensors[0].shards[0].addr == 0x1234
    assert payload.tensors[0].shards[0].shard_offset == (8, 0)
