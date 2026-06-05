# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace

import torch

from modelexpress import p2p_pb2
from modelexpress.metadata.source_id import compute_mx_source_id
from modelexpress.refit_poc import (
    _live_mx_endpoint_context,
    _live_mx_plan_context,
    alternate_ownerships,
    inference_request,
    primary_ownerships,
    refit_source_identity,
)
from modelexpress.refit_poc_artifacts import artifact_base, build_planner_artifacts
from modelexpress.refit_trainer_step import (
    publish_trainer_loop_step,
    publish_trainer_step_source,
)
from modelexpress.types import TensorDescriptor
from modelexpress.resharding import SliceOwnership, SliceRequest
from modelexpress.resharding_control_plane import (
    build_refit_source_identity,
    list_refit_nixl_endpoints,
    list_slice_ownerships,
    plan_from_mx_refit_endpoints,
    plan_from_mx_metadata,
    publish_refit_nixl_endpoint,
    publish_slice_ownerships,
    segment_plan_from_proto,
    segment_plan_to_proto,
    slice_ownership_from_proto,
    slice_ownership_to_proto,
    slice_request_from_proto,
    slice_request_to_proto,
)

MODEL_NAME = "qwen3-moe-refit-poc"
MODEL_VERSION = "trainer-step-000001"
TENSOR_NAME = "model.layers.0.mlp.experts.w1.weight"


class InMemoryMxClient:
    def __init__(self):
        self._workers = {}

    def publish_metadata(self, identity, worker, worker_id):
        source_id = compute_mx_source_id(identity)
        worker_copy = p2p_pb2.WorkerMetadata()
        worker_copy.CopyFrom(worker)
        self._workers[(source_id, worker_id)] = {
            "identity": identity,
            "worker": worker_copy,
        }
        return source_id

    def list_sources(self, identity=None, status_filter=None):
        source_id_filter = (
            compute_mx_source_id(identity) if identity is not None else None
        )
        refs = []
        for (source_id, worker_id), record in sorted(self._workers.items()):
            worker = record["worker"]
            if source_id_filter is not None and source_id != source_id_filter:
                continue
            if status_filter is not None and worker.status != int(status_filter):
                continue
            refs.append(
                p2p_pb2.SourceInstanceRef(
                    mx_source_id=source_id,
                    worker_id=worker_id,
                    model_name=record["identity"].model_name,
                    worker_rank=worker.worker_rank,
                )
            )
        return p2p_pb2.ListSourcesResponse(instances=refs)

    def get_metadata(self, mx_source_id, worker_id):
        record = self._workers.get((mx_source_id, worker_id))
        if record is None:
            return p2p_pb2.GetMetadataResponse(
                found=False,
                mx_source_id=mx_source_id,
                worker_id=worker_id,
            )
        return p2p_pb2.GetMetadataResponse(
            found=True,
            worker=record["worker"],
            mx_source_id=mx_source_id,
            worker_id=worker_id,
        )

    def update_status(self, mx_source_id, worker_id, worker_rank, status):
        record = self._workers.get((mx_source_id, worker_id))
        if record is None:
            return False
        worker = record["worker"]
        if worker.worker_rank != worker_rank:
            return False
        worker.status = int(status)
        return True

    def close(self):
        pass


def _identity():
    return build_refit_source_identity(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        dtype="float32",
        trainer_framework="synthetic-fsdp",
        trainer_layout="fsdp",
    )


def _identity_for_model_version(model_version):
    return build_refit_source_identity(
        model_name=MODEL_NAME,
        model_version=model_version,
        dtype="float32",
        trainer_framework="synthetic-fsdp-trainer-loop-smoke",
        trainer_layout="fsdp",
    )


def _ownerships():
    return [
        SliceOwnership(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            tensor_name=TENSOR_NAME,
            global_shape=(8, 4),
            dtype="float32",
            source_range=((0, 3), (0, 4)),
            worker_id="rank0",
            worker_rank=0,
            source_id="trainer-rank0",
            source_lease="lease-rank0-primary",
            nixl_descriptor_id="nixl-rank0-primary",
            layout_tags={
                "trainer_layout": "fsdp",
                "storage_layout": "row-major",
                "moe_expert_axis": 0,
            },
        ),
        SliceOwnership(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            tensor_name=TENSOR_NAME,
            global_shape=(8, 4),
            dtype="float32",
            source_range=((3, 8), (0, 4)),
            worker_id="rank1",
            worker_rank=1,
            source_id="trainer-rank1",
            source_lease="lease-rank1-primary",
            nixl_descriptor_id="nixl-rank1-primary",
            layout_tags={
                "trainer_layout": "fsdp",
                "storage_layout": "row-major",
                "moe_expert_axis": 0,
            },
        ),
    ]


def _request():
    return SliceRequest(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        tensor_name=TENSOR_NAME,
        requested_range=((2, 6), (0, 4)),
        target_shape=(4, 4),
        dtype="float32",
        target_id="inference-rank3",
        runtime_framework="vllm",
        layout_tags={
            "target_layout": "tp2-ep2",
            "storage_layout": "row-major",
            "moe_expert_axis": 0,
        },
    )


def _request_for_model_version(model_version):
    request = _request()
    return replace(request, model_version=model_version)


def test_slice_ownership_proto_preserves_typed_layout_tags():
    ownership = _ownerships()[0]
    descriptor = slice_ownership_to_proto(ownership)
    restored = slice_ownership_from_proto(descriptor)

    assert restored == ownership
    assert restored.layout_tags["moe_expert_axis"] == 0


def test_slice_request_and_segment_plan_proto_roundtrip():
    request = _request()
    restored_request = slice_request_from_proto(slice_request_to_proto(request))
    assert restored_request == request

    plan = plan_from_mx_metadata(
        _published_ready_client(),
        identity=_identity(),
        requests=[request],
    )[0]
    restored_plan = segment_plan_from_proto(segment_plan_to_proto(plan))
    assert restored_plan == plan


def test_publish_list_get_status_and_plan_from_mx_metadata():
    client = InMemoryMxClient()
    identity = _identity()
    ownerships = _ownerships()

    source_id_0 = publish_slice_ownerships(
        client,
        identity=identity,
        ownerships=[ownerships[0]],
        worker_id="worker-rank0",
        worker_rank=0,
    )
    source_id_1 = publish_slice_ownerships(
        client,
        identity=identity,
        ownerships=[ownerships[1]],
        worker_id="worker-rank1",
        worker_rank=1,
    )

    assert source_id_0 == source_id_1
    assert list_slice_ownerships(client, identity=identity) == []

    assert client.update_status(
        source_id_0,
        "worker-rank0",
        0,
        p2p_pb2.SOURCE_STATUS_READY,
    )
    assert client.update_status(
        source_id_1,
        "worker-rank1",
        1,
        p2p_pb2.SOURCE_STATUS_READY,
    )

    discovered = list_slice_ownerships(client, identity=identity)
    assert discovered == ownerships

    plans = plan_from_mx_metadata(
        client,
        identity=identity,
        requests=[_request()],
    )
    assert [plan.source_id for plan in plans] == ["trainer-rank0", "trainer-rank1"]
    assert [plan.bytes for plan in plans] == [16, 48]
    assert plans[0].target_range == ((2, 3), (0, 4))
    assert plans[1].target_range == ((3, 6), (0, 4))


def test_multi_ownership_publish_sidecar_recovers_when_server_drops_proto_field():
    client = InMemoryMxClient()
    identity = _identity()
    ownerships = [
        replace(
            _ownerships()[0],
            source_id="trainer-rank0-a",
            source_range=((0, 2), (0, 4)),
            source_lease="lease-rank0-a",
            nixl_descriptor_id="nixl-rank0-a",
        ),
        replace(
            _ownerships()[0],
            source_id="trainer-rank0-b",
            source_range=((2, 3), (0, 4)),
            source_lease="lease-rank0-b",
            nixl_descriptor_id="nixl-rank0-b",
        ),
    ]

    source_id = publish_slice_ownerships(
        client,
        identity=identity,
        ownerships=ownerships,
        worker_id="rank0",
        worker_rank=0,
    )
    worker = client._workers[(source_id, "rank0")]["worker"]
    assert len(worker.slice_ownerships) == 2
    assert worker.metadata_endpoint.startswith("mx-refit-ownership-v1:")
    del worker.slice_ownerships[:]
    assert client.update_status(
        source_id,
        "rank0",
        0,
        p2p_pb2.SOURCE_STATUS_READY,
    )

    assert list_slice_ownerships(client, identity=identity) == ownerships


def test_single_ownership_publish_sidecar_recovers_when_server_drops_proto_field():
    client = InMemoryMxClient()
    identity = _identity()
    ownerships = _ownerships()

    source_id = None
    for ownership in ownerships:
        source_id = publish_slice_ownerships(
            client,
            identity=identity,
            ownerships=[ownership],
            worker_id=ownership.worker_id,
            worker_rank=ownership.worker_rank,
        )
        worker = client._workers[(source_id, ownership.worker_id)]["worker"]
        assert len(worker.slice_ownerships) == 1
        assert worker.metadata_endpoint.startswith("mx-refit-ownership-v1:")
        del worker.slice_ownerships[:]
        assert client.update_status(
            source_id,
            ownership.worker_id,
            ownership.worker_rank,
            p2p_pb2.SOURCE_STATUS_READY,
        )

    discovered = list_slice_ownerships(client, identity=identity)
    assert discovered == ownerships

    plans = plan_from_mx_metadata(
        client,
        identity=identity,
        requests=[_request()],
    )
    assert [plan.source_id for plan in plans] == ["trainer-rank0", "trainer-rank1"]
    assert [plan.bytes for plan in plans] == [16, 48]


def test_trainer_step_publications_roundtrip_through_mx_metadata():
    client = InMemoryMxClient()
    identity = _identity()
    source_publications = [
        publish_trainer_step_source(
            ownership,
            dtype=torch.float32,
            device=torch.device("cpu"),
            step_count=2,
            learning_rate=0.25,
        )
        for ownership in _ownerships()
    ]

    source_id = None
    for publication in source_publications:
        ownership = publication.ownership
        source_id = publish_slice_ownerships(
            client,
            identity=identity,
            ownerships=[ownership],
            worker_id=ownership.worker_id,
            worker_rank=ownership.worker_rank,
        )
        assert client.update_status(
            source_id,
            ownership.worker_id,
            ownership.worker_rank,
            p2p_pb2.SOURCE_STATUS_READY,
        )

    discovered = list_slice_ownerships(client, identity=identity)
    assert [owner.source_id for owner in discovered] == [
        "trainer-rank0",
        "trainer-rank1",
    ]
    assert all(
        owner.layout_tags["trainer_update_source"] == "torch.optim.SGD-step-publisher"
        for owner in discovered
    )
    assert all(owner.layout_tags["optimizer_step_count"] == 2 for owner in discovered)
    assert all(
        owner.layout_tags["optimizer_step_publisher"] is True for owner in discovered
    )
    assert all(owner.source_lease for owner in discovered)
    assert all(owner.nixl_descriptor_id for owner in discovered)

    plans = plan_from_mx_metadata(
        client,
        identity=identity,
        requests=[_request()],
    )
    assert [plan.source_id for plan in plans] == ["trainer-rank0", "trainer-rank1"]
    assert [plan.lease_version for plan in plans] == [
        discovered[0].source_lease,
        discovered[1].source_lease,
    ]
    assert (
        source_publications[0].to_artifact_metadata()["ownership"]["layout_tags"][
            "optimizer_step_publisher"
        ]
        is True
    )


def test_trainer_loop_step_publications_roundtrip_through_mx_metadata():
    client = InMemoryMxClient()
    loop_step = publish_trainer_loop_step(
        _ownerships(),
        dtype=torch.float32,
        device=torch.device("cpu"),
        step_index=3,
        learning_rate=0.25,
    )
    identity = _identity_for_model_version(loop_step.model_version)

    for publication in loop_step.source_publications:
        ownership = publication.ownership
        source_id = publish_slice_ownerships(
            client,
            identity=identity,
            ownerships=[ownership],
            worker_id=ownership.worker_id,
            worker_rank=ownership.worker_rank,
        )
        assert client.update_status(
            source_id,
            ownership.worker_id,
            ownership.worker_rank,
            p2p_pb2.SOURCE_STATUS_READY,
        )

    discovered = list_slice_ownerships(client, identity=identity)
    assert [owner.model_version for owner in discovered] == [
        loop_step.model_version,
        loop_step.model_version,
    ]
    assert all(
        owner.layout_tags["trainer_loop_publisher"] is True for owner in discovered
    )
    assert all(
        owner.layout_tags["trainer_loop_step_index"] == 3 for owner in discovered
    )
    assert all(loop_step.model_version in owner.source_lease for owner in discovered)

    plans = plan_from_mx_metadata(
        client,
        identity=identity,
        requests=[_request_for_model_version(loop_step.model_version)],
    )
    assert [plan.source_id for plan in plans] == ["trainer-rank0", "trainer-rank1"]
    assert [plan.lease_version for plan in plans] == [
        discovered[0].source_lease,
        discovered[1].source_lease,
    ]
    assert (
        loop_step.to_artifact_metadata()["trainer_loop_provenance"][
            "synthetic_trainer_loop_smoke_used"
        ]
        is True
    )


def test_publish_and_plan_refit_nixl_endpoints_from_mx_metadata():
    client = InMemoryMxClient()
    identity = _identity()
    ownerships = _ownerships()

    unused_ownership = replace(
        ownerships[1],
        source_id="trainer-rank-unused",
        worker_id="rank-unused",
        worker_rank=99,
        source_range=((6, 8), (0, 4)),
        source_lease="lease-unused",
        nixl_descriptor_id="nixl-unused",
    )

    for idx, ownership in enumerate([*ownerships, unused_ownership]):
        publish_refit_nixl_endpoint(
            client,
            identity=identity,
            ownership=ownership,
            tensor=TensorDescriptor(
                name=ownership.tensor_name,
                addr=0xABC000 + idx * 4096,
                size=128 + idx,
                device_id=idx,
                dtype=ownership.dtype,
            ),
            agent_name=f"mx-refit-rank{idx}",
            nixl_metadata=f"nixl-metadata-{idx}".encode(),
            worker_id=ownership.worker_id,
            worker_rank=idx,
        )

    endpoints = list_refit_nixl_endpoints(client, identity=identity)
    assert {endpoint.source_id for endpoint in endpoints} == {
        "trainer-rank0",
        "trainer-rank1",
        "trainer-rank-unused",
    }
    endpoint_by_source_id = {endpoint.source_id: endpoint for endpoint in endpoints}
    rank0_endpoint = endpoint_by_source_id["trainer-rank0"]
    assert rank0_endpoint.agent_name == "mx-refit-rank0"
    assert rank0_endpoint.nixl_metadata == b"nixl-metadata-0"
    assert rank0_endpoint.tensor.addr == 0xABC000
    assert rank0_endpoint.status == p2p_pb2.SOURCE_STATUS_READY
    assert rank0_endpoint.updated_at == 0
    rank0_source_info = rank0_endpoint.to_nixl_source_info()
    assert rank0_source_info["addr"] == 0xABC000
    assert rank0_source_info["status"] == p2p_pb2.SOURCE_STATUS_READY
    assert rank0_source_info["updated_at"] == 0
    assert rank0_endpoint.to_dict()["status"] == p2p_pb2.SOURCE_STATUS_READY

    plans, endpoints_by_source_id = plan_from_mx_refit_endpoints(
        client,
        identity=identity,
        requests=[_request()],
    )
    assert [plan.source_id for plan in plans] == ["trainer-rank0", "trainer-rank1"]
    assert sorted(endpoints_by_source_id) == ["trainer-rank0", "trainer-rank1"]
    assert "trainer-rank-unused" not in endpoints_by_source_id
    assert endpoints_by_source_id["trainer-rank1"].tensor.device_id == 1


def test_refit_nixl_endpoint_status_filter_can_discover_stale_endpoints():
    client = InMemoryMxClient()
    identity = _identity()
    ownership = _ownerships()[0]

    source_id = publish_refit_nixl_endpoint(
        client,
        identity=identity,
        ownership=ownership,
        tensor=TensorDescriptor(
            name=ownership.tensor_name,
            addr=0xABC000,
            size=128,
            device_id=0,
            dtype=ownership.dtype,
        ),
        agent_name="mx-refit-rank0",
        nixl_metadata=b"nixl-metadata-0",
        worker_id=ownership.worker_id,
        worker_rank=ownership.worker_rank,
    )
    assert client.update_status(
        source_id,
        ownership.worker_id,
        ownership.worker_rank,
        p2p_pb2.SOURCE_STATUS_STALE,
    )

    assert list_refit_nixl_endpoints(client, identity=identity) == []

    endpoints = list_refit_nixl_endpoints(
        client,
        identity=identity,
        status_filter=None,
    )

    assert len(endpoints) == 1
    endpoint = endpoints[0]
    assert endpoint.source_id == "trainer-rank0"
    assert endpoint.status == p2p_pb2.SOURCE_STATUS_STALE
    assert endpoint.to_dict()["status"] == p2p_pb2.SOURCE_STATUS_STALE


def test_refit_nixl_endpoint_recovers_ownership_from_legacy_sidecar():
    client = InMemoryMxClient()
    identity = _identity()
    ownership = _ownerships()[0]

    source_id = publish_refit_nixl_endpoint(
        client,
        identity=identity,
        ownership=ownership,
        tensor=TensorDescriptor(
            name=ownership.tensor_name,
            addr=0xD00D,
            size=128,
            device_id=0,
            dtype=ownership.dtype,
        ),
        agent_name="mx-refit-rank0",
        nixl_metadata=b"nixl-metadata-0",
        worker_id=ownership.worker_id,
        worker_rank=ownership.worker_rank,
        metadata_endpoint="10.0.0.1:5555",
    )

    worker = client._workers[(source_id, ownership.worker_id)]["worker"]
    assert worker.metadata_endpoint.startswith("mx-refit-ownership-v1:")
    del worker.slice_ownerships[:]

    endpoints = list_refit_nixl_endpoints(client, identity=identity)

    assert len(endpoints) == 1
    endpoint = endpoints[0]
    assert endpoint.ownership == ownership
    assert endpoint.source_id == "trainer-rank0"
    assert endpoint.metadata_endpoint == "10.0.0.1:5555"
    assert endpoint.nixl_metadata == b"nixl-metadata-0"


def test_refit_poc_live_mx_plan_context_uses_returned_metadata():
    client = InMemoryMxClient()
    identity = refit_source_identity()
    all_ownerships = [*primary_ownerships(), *alternate_ownerships()]
    source_id = None

    for ownership in all_ownerships:
        source_id = publish_slice_ownerships(
            client,
            identity=identity,
            ownerships=[ownership],
            worker_id=ownership.worker_id,
            worker_rank=ownership.worker_rank,
        )
        assert client.update_status(
            source_id,
            ownership.worker_id,
            ownership.worker_rank,
            p2p_pb2.SOURCE_STATUS_READY,
        )

    context = _live_mx_plan_context(
        inference_request(),
        timeout_seconds=0,
        mx_client=client,
    )

    assert context["plan_source"] == "live-mx-server"
    assert [plan.source_id for plan in context["primary_plans"]] == [
        "trainer-rank0",
        "trainer-rank1",
    ]
    assert [plan.source_id for plan in context["recovery_plans"]] == [
        "trainer-rank2-alt"
    ]
    assert {owner.source_id for owner in context["discovered_ownerships"]} == {
        "trainer-rank0",
        "trainer-rank1",
        "trainer-rank2-alt",
    }


def test_refit_poc_live_mx_endpoint_context_uses_returned_nixl_endpoints():
    client = InMemoryMxClient()
    identity = refit_source_identity()
    all_ownerships = [*primary_ownerships(), *alternate_ownerships()]

    for idx, ownership in enumerate(all_ownerships):
        publish_refit_nixl_endpoint(
            client,
            identity=identity,
            ownership=ownership,
            tensor=TensorDescriptor(
                name=ownership.tensor_name,
                addr=0xC0FFEE + idx * 4096,
                size=64,
                device_id=idx,
                dtype=ownership.dtype,
            ),
            agent_name=f"mx-refit-rank{idx}",
            nixl_metadata=f"nixl-endpoint-{idx}".encode(),
            worker_id=ownership.worker_id,
            worker_rank=ownership.worker_rank,
        )

    context = _live_mx_endpoint_context(
        inference_request(),
        timeout_seconds=0,
        mx_client=client,
    )

    assert context["plan_source"] == "live-mx-server"
    assert context["endpoint_source"] == "mx-worker-metadata"
    assert [plan.source_id for plan in context["primary_plans"]] == [
        "trainer-rank0",
        "trainer-rank1",
    ]
    assert [plan.source_id for plan in context["recovery_plans"]] == [
        "trainer-rank2-alt"
    ]
    endpoints_by_source_id = context["source_endpoints_by_id"]
    assert sorted(endpoints_by_source_id) == [
        "trainer-rank0",
        "trainer-rank1",
        "trainer-rank2-alt",
    ]
    rank0_endpoint = endpoints_by_source_id["trainer-rank0"]
    assert rank0_endpoint.nixl_metadata == b"nixl-endpoint-0"
    assert endpoints_by_source_id["trainer-rank2-alt"].tensor.device_id == 2


def test_refit_poc_artifacts_use_live_mx_returned_plan_context():
    client = InMemoryMxClient()
    identity = refit_source_identity()
    all_ownerships = [
        replace(primary_ownerships()[0], source_lease="live-lease-rank0"),
        primary_ownerships()[1],
        *alternate_ownerships(),
    ]
    source_id = None

    for ownership in all_ownerships:
        source_id = publish_slice_ownerships(
            client,
            identity=identity,
            ownerships=[ownership],
            worker_id=ownership.worker_id,
            worker_rank=ownership.worker_rank,
        )
        assert client.update_status(
            source_id,
            ownership.worker_id,
            ownership.worker_rank,
            p2p_pb2.SOURCE_STATUS_READY,
        )

    request = inference_request()
    context = _live_mx_plan_context(
        request,
        timeout_seconds=0,
        mx_client=client,
    )

    planner_artifacts = build_planner_artifacts(
        request=request,
        plan_context=context,
    )
    assert planner_artifacts["plan_source"] == "live-mx-server"
    assert planner_artifacts["control_plane_mode"] == "live-mx"
    assert planner_artifacts["primary_ownerships"][0]["source_lease"] == (
        "live-lease-rank0"
    )

    result = artifact_base(
        mode="nixl-distributed-4rank",
        gpu_count=0,
        copied_bytes=64,
        copy_duration_ms=0.0,
        validation={
            "allclose": True,
            "checksum": None,
            "expected_checksum": None,
            "max_abs_error": 0.0,
        },
        request=request,
        plan_context=context,
    )
    assert result["planner"]["plan_source"] == "live-mx-server"
    assert result["planner"]["primary_ownerships"][0]["source_lease"] == (
        "live-lease-rank0"
    )


def _published_ready_client():
    client = InMemoryMxClient()
    identity = _identity()
    source_id = None
    for idx, ownership in enumerate(_ownerships()):
        worker_id = f"worker-rank{idx}"
        source_id = publish_slice_ownerships(
            client,
            identity=identity,
            ownerships=[ownership],
            worker_id=worker_id,
            worker_rank=idx,
        )
        client.update_status(
            source_id,
            worker_id,
            idx,
            p2p_pb2.SOURCE_STATUS_READY,
        )
    return client
