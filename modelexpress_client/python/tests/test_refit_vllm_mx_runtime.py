# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from modelexpress import p2p_pb2
from modelexpress.refit_vllm_mx_runtime import (
    _effective_model_version,
    _filter_source_ownerships,
    build_vllm_mx_runtime_source_ownerships,
    materialize_vllm_mx_runtime_source_publications,
    scenario_with_mx_endpoint_plan,
)
from modelexpress.refit_vllm_nixl_runtime_smoke import build_vllm_runtime_nixl_plan
from modelexpress.resharding_control_plane import RefitNixlEndpoint
from modelexpress.types import TensorDescriptor


def test_vllm_mx_runtime_source_ownerships_are_filterable_per_pod():
    owners = build_vllm_mx_runtime_source_ownerships(
        tensor_name="lm_head.weight",
        shape=(8, 4),
        dtype_name="float32",
        model_name="mx-vllm-mx-unit",
        model_version="step-7",
    )

    assert _effective_model_version("base", "run-1") == "base-run-1"
    assert [owner.source_id for owner in owners] == ["trainer-rank0", "trainer-rank1"]
    assert [owner.worker_rank for owner in owners] == [0, 1]
    assert owners[0].source_range == ((0, 4), (0, 4))
    assert owners[1].source_range == ((4, 8), (0, 4))
    assert owners[0].layout_tags["optimizer_step_publisher"] is True

    rank1 = _filter_source_ownerships(owners, source_id="trainer-rank1")
    assert [owner.source_id for owner in rank1] == ["trainer-rank1"]


def test_vllm_mx_runtime_source_publications_keep_metadata():
    owners = build_vllm_mx_runtime_source_ownerships(
        tensor_name="lm_head.weight",
        shape=(6, 4),
        dtype_name="float32",
        model_name="mx-vllm-mx-unit",
        model_version="step-9",
    )

    publications = materialize_vllm_mx_runtime_source_publications(
        owners,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert [publication.ownership.source_id for publication in publications] == [
        "trainer-rank0",
        "trainer-rank1",
    ]
    assert [tuple(publication.tensor.shape) for publication in publications] == [
        (3, 4),
        (3, 4),
    ]
    assert all(
        publication.provenance["optimizer_step_publisher_used"]
        for publication in publications
    )


def test_vllm_mx_runtime_scenario_uses_mx_endpoint_ownerships_for_plan():
    target = torch.empty((8, 4), dtype=torch.float32)
    request, local_owners, local_plans = build_vllm_runtime_nixl_plan(
        target,
        tensor_name="lm_head.weight",
        model_name="mx-vllm-mx-unit",
        model_version="step-8",
    )
    scenario = {
        "target_tensor_name": "lm_head.weight",
        "target_shape": [8, 4],
        "target_dtype": "float32",
        "target_tensor_bytes": 128,
        "request": request.to_dict(),
        "source_ownerships": [owner.to_dict() for owner in local_owners],
        "segment_plans": [plan.to_dict() for plan in local_plans],
    }
    mx_owners = build_vllm_mx_runtime_source_ownerships(
        tensor_name="lm_head.weight",
        shape=(8, 4),
        dtype_name="float32",
        model_name="mx-vllm-mx-unit",
        model_version="step-8",
    )
    endpoints = [_endpoint(owner) for owner in mx_owners]

    planned, context = scenario_with_mx_endpoint_plan(
        scenario,
        endpoints,
        metadata_query_duration_ms=2.5,
        planner_duration_ms=0.25,
    )

    assert [owner["source_id"] for owner in planned["source_ownerships"]] == [
        "trainer-rank0",
        "trainer-rank1",
    ]
    assert [plan["source_id"] for plan in planned["segment_plans"]] == [
        "trainer-rank0",
        "trainer-rank1",
    ]
    assert [plan["target_byte_offset"] for plan in planned["segment_plans"]] == [
        0,
        64,
    ]
    assert [plan["bytes"] for plan in planned["segment_plans"]] == [64, 64]
    assert context["plan_source"] == "live-vllm-request+mx-endpoint-ownerships"
    assert context["endpoint_source"] == "mx-worker-metadata"
    assert sorted(context["source_endpoints_by_id"]) == [
        "trainer-rank0",
        "trainer-rank1",
    ]
    assert context["metadata_query_duration_ms"] == 2.5
    assert context["source_statuses_by_id"]["trainer-rank0"]["status_name"] == (
        "SOURCE_STATUS_READY"
    )


def _endpoint(owner) -> RefitNixlEndpoint:
    return RefitNixlEndpoint(
        mx_source_id=f"mx-{owner.source_id}",
        worker_id=owner.worker_id,
        worker_rank=int(owner.worker_rank or 0),
        ownership=owner,
        tensor=TensorDescriptor(
            name=owner.tensor_name,
            addr=0xBAD000 + int(owner.worker_rank or 0) * 4096,
            size=64,
            device_id=int(owner.worker_rank or 0),
            dtype=owner.dtype,
        ),
        agent_name=f"agent-{owner.source_id}",
        nixl_metadata=f"metadata-{owner.source_id}".encode(),
        status=p2p_pb2.SOURCE_STATUS_READY,
        updated_at=123,
    )
