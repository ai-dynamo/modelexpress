# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from modelexpress import p2p_pb2
from modelexpress.metadata.source_id import compute_mx_source_id
from modelexpress.refit_poc import (
    _live_mx_plan_context,
    alternate_ownerships,
    inference_request,
    primary_ownerships,
    refit_source_identity,
)
from modelexpress.resharding import SliceOwnership, SliceRequest
from modelexpress.resharding_control_plane import (
    build_refit_source_identity,
    list_slice_ownerships,
    plan_from_mx_metadata,
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
        source_id_filter = compute_mx_source_id(identity) if identity is not None else None
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
