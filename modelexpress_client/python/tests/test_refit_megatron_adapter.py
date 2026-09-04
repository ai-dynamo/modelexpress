# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
from concurrent import futures
from dataclasses import replace
from types import SimpleNamespace

import grpc
import modelexpress_rl.train.runtime as runtime_module
import pytest
from modelexpress.refit.reshard.rendezvous import unwrap_rendezvous_blob
from modelexpress_rl import (
    MegatronTrainerContext,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionRef,
    refit_pb2,
    refit_pb2_grpc,
)
from modelexpress_rl.train.adapter import TrainerEngineAdapter
from modelexpress_rl.train.manifest import WeightVersionShardManifestService
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

    def __init__(self):
        self.registered = []

    def register_tensors(self, tensors):
        self.registered.append(dict(tensors))
        return self.nixl_metadata


class _RefitService(refit_pb2_grpc.RefitServiceServicer):
    def __init__(self, events):
        self.events = events
        self.registration_ttl = None
        self.shard = None

    def RegisterWorker(self, request, _context):
        self.events.append("register-worker")
        self.registration_ttl = request.ttl_seconds
        return refit_pb2.RegisterWorkerResponse(worker=request.worker)

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


def test_megatron_source_slot_groups_replicas_by_logical_partition(monkeypatch):
    monkeypatch.setattr(
        "modelexpress_rl.train.engines.megatron.adapter.dist.is_initialized",
        lambda: True,
    )
    monkeypatch.setattr(
        "modelexpress_rl.train.engines.megatron.adapter.dist.get_rank",
        lambda: 3,
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
            local_shard_range=(0, 8),
        )
    ]

    first = MegatronTrainerAdapter(
        manager=_Manager(),
        nixl_metadata_endpoint="10.0.0.3:19003",
    )
    second = MegatronTrainerAdapter(
        manager=_Manager(),
        nixl_metadata_endpoint="10.0.0.4:19004",
    )
    other_partition = MegatronTrainerAdapter(
        manager=_Manager(),
        nixl_metadata_endpoint="10.0.0.5:19005",
    )

    assert first.bind_tensors(tensors) == second.bind_tensors(tensors)
    assert first.source_slot_id == second.source_slot_id
    assert other_partition.bind_tensors(
        [replace(tensors[0], local_shard_range=(8, 16))]
    ) != first.source_slot_id


def test_megatron_adapter_uses_shared_trainer_publication_flow(monkeypatch):
    monkeypatch.setattr(
        "modelexpress_rl.train.engines.megatron.aliases.tensor_digest",
        lambda _tensor: "tensor-digest",
    )
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
    resources = SimpleNamespace(
        manager=_Manager(),
        manifest_service=manifest_service,
        worker_endpoint="trainer-3:9000",
        close=lambda: None,
    )
    monkeypatch.setattr(
        runtime_module._TrainerResources,
        "initialize",
        lambda **_kwargs: resources,
    )
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
            ModelExpressTrainerConfig(
                engine_context=MegatronTrainerContext(),
                device_id=3,
                server_url=f"127.0.0.1:{port}",
                model_name="model",
                staging_mode=TrainerStagingMode.IN_PLACE,
                payload_format=WeightPayloadFormat.FULL_TENSOR,
                worker_id="worker-3",
                registration_ttl_seconds=60,
            )
        )
        source_slot_id = refit_client.bind_tensors(tensors)
        selected_adapter = refit_client._runtime.method._adapter
        refit_client.publish_version(version=WeightVersionRef("version-a"))
        worker_stub = refit_pb2_grpc.RefitWorkerServiceStub(
            grpc.insecure_channel(refit_service.shard.manifest_endpoint)
        )
        fetched = worker_stub.GetWeightVersionShardManifest(
            refit_pb2.GetWeightVersionShardManifestRequest(
                version_id="version-a",
                source_slot_id=source_slot_id,
            )
        )
    finally:
        if "refit_client" in locals():
            refit_client.close()
        server.stop(grace=None).wait()

    assert isinstance(adapter, TrainerEngineAdapter)
    assert isinstance(selected_adapter, MegatronTrainerAdapter)
    assert events == [
        "register-worker",
        "publish-version-shard",
    ]
    assert refit_service.registration_ttl == 60
    assert len(resources.manager.registered) == 1
    assert refit_service.shard.version_id == "version-a"
    assert source_slot_id.startswith("megatron:partition:")
    assert refit_service.shard.source_slot_id == source_slot_id
    assert refit_service.shard.worker_id == "worker-3"
    assert refit_service.shard.tensor_count == 1
    assert refit_service.shard.total_bytes == 128
    assert (
        refit_service.shard.manifest_digest
        == hashlib.sha256(fetched.manifest).hexdigest()
    )
    assert refit_service.shard.manifest_endpoint == f"127.0.0.1:{port}"
    assert fetched.manifest_digest == refit_service.shard.manifest_digest
    payload = unwrap_rendezvous_blob(fetched.manifest)
    assert payload.metadata_endpoint == "10.0.0.3:19003"
    assert payload.tensors[0].shards[0].addr == 0x1234
    assert payload.tensors[0].shards[0].shard_offset == (8, 0)
    assert payload.tensors[0].shards[0].digest == "tensor-digest"
