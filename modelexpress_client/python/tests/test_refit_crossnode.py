# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from modelexpress import p2p_pb2
from modelexpress.refit_crossnode import (
    _plan_context_for_endpoints,
    _run_ownerships,
    _run_request,
)
from modelexpress.resharding_control_plane import RefitNixlEndpoint
from modelexpress.types import TensorDescriptor


def test_crossnode_plan_context_carries_stale_source_recovery_metadata():
    run_id = "unit-stale-source"
    endpoints = [
        _endpoint(
            owner,
            status=(
                p2p_pb2.SOURCE_STATUS_STALE
                if owner.source_id == "trainer-rank0"
                else p2p_pb2.SOURCE_STATUS_READY
            ),
        )
        for owner in _run_ownerships(run_id)
    ]

    context = _plan_context_for_endpoints(
        endpoints=endpoints,
        request=_run_request(run_id),
        metadata_query_duration_ms=1.0,
        planner_duration_ms=0.0,
        failed_source_ids={"trainer-rank0"},
        stale_source_ids={"trainer-rank0"},
    )

    assert context["failed_source_ids"] == ["trainer-rank0"]
    assert context["stale_source_ids"] == ["trainer-rank0"]
    assert context["unreadable_source_ids"] == ["trainer-rank0"]
    assert context["source_statuses_by_id"]["trainer-rank0"] == {
        "status": p2p_pb2.SOURCE_STATUS_STALE,
        "status_name": "SOURCE_STATUS_STALE",
        "updated_at": 123,
    }
    assert [plan.source_id for plan in context["primary_plans"]] == [
        "trainer-rank0",
        "trainer-rank1",
    ]
    assert [plan.source_id for plan in context["recovery_plans"]] == [
        "trainer-rank2-alt"
    ]
    assert context["recovery_plans"][0].target_range == ((2, 3), (0, 4))


def _endpoint(owner, *, status: int) -> RefitNixlEndpoint:
    return RefitNixlEndpoint(
        mx_source_id=f"mx-{owner.source_id}",
        worker_id=owner.worker_id,
        worker_rank=int(owner.worker_rank or 0),
        ownership=owner,
        tensor=TensorDescriptor(
            name=owner.tensor_name,
            addr=0xABC000 + int(owner.worker_rank or 0) * 4096,
            size=128,
            device_id=int(owner.worker_rank or 0),
            dtype=owner.dtype,
        ),
        agent_name=f"agent-{owner.source_id}",
        nixl_metadata=f"metadata-{owner.source_id}".encode(),
        status=status,
        updated_at=123,
    )
