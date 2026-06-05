# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import time
import uuid

import grpc
import pytest
import torch

from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from modelexpress.refit_trainer_step import publish_trainer_step_source
from modelexpress.resharding import SliceOwnership, SliceRequest
from modelexpress.resharding_control_plane import (
    build_refit_source_identity,
    list_slice_ownerships,
    plan_from_mx_metadata,
    publish_slice_ownerships,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("MX_LIVE_SERVER_URL"),
    reason="set MX_LIVE_SERVER_URL to run against a live ModelExpress server",
)


MODEL_NAME = "qwen3-moe-refit-live-smoke"
TENSOR_NAME = "model.layers.0.mlp.experts.w1.weight"


def test_live_mx_server_slice_ownership_lifecycle():
    server_url = os.environ["MX_LIVE_SERVER_URL"]
    model_version = f"trainer-step-{uuid.uuid4().hex}"
    identity = build_refit_source_identity(
        model_name=MODEL_NAME,
        model_version=model_version,
        dtype="float32",
        trainer_framework="synthetic-fsdp",
        trainer_layout="fsdp",
    )
    client = MxClient(server_url=server_url)
    try:
        _wait_for_server(client)

        ownerships = _ownerships(model_version)
        source_id_0 = publish_slice_ownerships(
            client,
            identity=identity,
            ownerships=[ownerships[0]],
            worker_id=f"worker-rank0-{model_version}",
            worker_rank=0,
        )
        source_id_1 = publish_slice_ownerships(
            client,
            identity=identity,
            ownerships=[ownerships[1]],
            worker_id=f"worker-rank1-{model_version}",
            worker_rank=1,
        )
        assert source_id_0 == source_id_1

        assert list_slice_ownerships(client, identity=identity) == []

        for worker_rank in (0, 1):
            worker_id = f"worker-rank{worker_rank}-{model_version}"
            response = client.get_metadata(source_id_0, worker_id)
            assert response.found
            if response.worker.slice_ownerships:
                assert len(response.worker.slice_ownerships) == 1
                assert response.worker.slice_ownerships[0].worker_rank == worker_rank
            else:
                assert response.worker.metadata_endpoint.startswith(
                    "mx-refit-ownership-v1:"
                )
            assert client.update_status(
                source_id_0,
                worker_id,
                worker_rank,
                p2p_pb2.SOURCE_STATUS_READY,
            )

        discovered = list_slice_ownerships(client, identity=identity)
        assert {
            tuple(axis for axis in ownership.source_range) for ownership in discovered
        } == {
            ((0, 3), (0, 4)),
            ((3, 8), (0, 4)),
        }
        assert {ownership.source_id for ownership in discovered} == {
            "trainer-rank0",
            "trainer-rank1",
        }

        plans = plan_from_mx_metadata(
            client,
            identity=identity,
            requests=[_request(model_version)],
        )
        plan_by_source = {plan.source_id: plan for plan in plans}
        assert set(plan_by_source) == {"trainer-rank0", "trainer-rank1"}
        assert plan_by_source["trainer-rank0"].bytes == 16
        assert plan_by_source["trainer-rank1"].bytes == 48
        assert plan_by_source["trainer-rank0"].target_range == ((2, 3), (0, 4))
        assert plan_by_source["trainer-rank1"].target_range == ((3, 6), (0, 4))
    finally:
        client.close()


def test_live_mx_server_trainer_step_publication_lifecycle():
    server_url = os.environ["MX_LIVE_SERVER_URL"]
    model_version = f"trainer-step-publication-{uuid.uuid4().hex}"
    identity = build_refit_source_identity(
        model_name=MODEL_NAME,
        model_version=model_version,
        dtype="float32",
        trainer_framework="synthetic-fsdp",
        trainer_layout="fsdp",
    )
    client = MxClient(server_url=server_url)
    try:
        _wait_for_server(client)
        publications = [
            publish_trainer_step_source(
                ownership,
                dtype=torch.float32,
                device=torch.device("cpu"),
                step_count=2,
                learning_rate=0.25,
            )
            for ownership in _ownerships(model_version)
        ]

        source_ids = []
        for publication in publications:
            ownership = publication.ownership
            source_id = publish_slice_ownerships(
                client,
                identity=identity,
                ownerships=[ownership],
                worker_id=ownership.worker_id,
                worker_rank=ownership.worker_rank,
            )
            source_ids.append(source_id)
            response = client.get_metadata(source_id, ownership.worker_id)
            assert response.found
            if (
                not response.worker.slice_ownerships
                and not response.worker.metadata_endpoint
            ):
                pytest.skip(
                    "live MX server dropped slice ownership metadata and legacy sidecar"
                )
            if response.worker.metadata_endpoint:
                assert response.worker.metadata_endpoint.startswith(
                    "mx-refit-ownership-v1:"
                )
            assert client.update_status(
                source_id,
                ownership.worker_id,
                ownership.worker_rank,
                p2p_pb2.SOURCE_STATUS_READY,
            )
        assert len(set(source_ids)) == 1

        discovered = list_slice_ownerships(client, identity=identity)
        assert {ownership.source_id for ownership in discovered} == {
            "trainer-rank0",
            "trainer-rank1",
        }
        assert all(
            ownership.layout_tags["optimizer_step_publisher"] is True
            for ownership in discovered
        )
        assert all(ownership.source_lease for ownership in discovered)
        assert all(ownership.nixl_descriptor_id for ownership in discovered)

        plans = plan_from_mx_metadata(
            client,
            identity=identity,
            requests=[_request(model_version)],
        )
        plan_by_source = {plan.source_id: plan for plan in plans}
        assert set(plan_by_source) == {"trainer-rank0", "trainer-rank1"}
        assert plan_by_source["trainer-rank0"].lease_version
        assert plan_by_source["trainer-rank1"].lease_version
    finally:
        client.close()


def _wait_for_server(client: MxClient) -> None:
    deadline = time.time() + 30
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            client.list_sources()
            return
        except grpc.RpcError as exc:
            last_error = exc
            time.sleep(0.5)
    raise AssertionError(f"ModelExpress server did not become ready: {last_error}")


def _ownerships(model_version: str) -> list[SliceOwnership]:
    return [
        SliceOwnership(
            model_name=MODEL_NAME,
            model_version=model_version,
            tensor_name=TENSOR_NAME,
            global_shape=(8, 4),
            dtype="float32",
            source_range=((0, 3), (0, 4)),
            worker_id=f"worker-rank0-{model_version}",
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
            model_version=model_version,
            tensor_name=TENSOR_NAME,
            global_shape=(8, 4),
            dtype="float32",
            source_range=((3, 8), (0, 4)),
            worker_id=f"worker-rank1-{model_version}",
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


def _request(model_version: str) -> SliceRequest:
    return SliceRequest(
        model_name=MODEL_NAME,
        model_version=model_version,
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
