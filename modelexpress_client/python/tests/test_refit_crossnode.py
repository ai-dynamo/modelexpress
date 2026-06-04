# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from modelexpress import p2p_pb2
from modelexpress.refit_crossnode import (
    _plan_context_for_endpoints,
    _read_plans_with_runtime_recovery,
    _run_ownerships,
    _run_ownerships_for_payload,
    _run_request,
    _run_request_for_payload,
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


def test_crossnode_runtime_read_failure_replans_only_failed_source_segments():
    run_id = "unit-runtime-read-failure"
    endpoints = [
        _endpoint(owner, status=p2p_pb2.SOURCE_STATUS_READY)
        for owner in _run_ownerships(run_id)
    ]
    request = _run_request(run_id)
    context = _plan_context_for_endpoints(
        endpoints=endpoints,
        request=request,
        metadata_query_duration_ms=1.0,
        planner_duration_ms=0.0,
        failed_source_ids=set(),
        stale_source_ids=set(),
    )
    sources_by_id = {
        endpoint.source_id: endpoint.to_nixl_source_info()
        for endpoint in context["source_endpoints_by_id"].values()
    }
    remote_agent_names = {
        source_id: f"remote-{source_id}" for source_id in sources_by_id
    }
    attempted_sources: list[str] = []

    def fake_read_groups(**kwargs):
        plans = kwargs["plans"]
        if not plans:
            return []
        source_ids = sorted({plan.source_id for plan in plans})
        attempted_sources.extend(source_ids)
        if source_ids == ["trainer-rank0"]:
            raise RuntimeError("source died during read")
        return [
            {
                "source_id": source_id,
                "source_rank": sources_by_id[source_id]["rank"],
                "segment_count": len(
                    [plan for plan in plans if plan.source_id == source_id]
                ),
                "bytes": sum(
                    plan.bytes for plan in plans if plan.source_id == source_id
                ),
                "prep_duration_ms": 0.0,
                "read_duration_ms": 0.0,
                "backend": "fake",
                "telemetry": None,
                "segments": [
                    plan.to_dict() for plan in plans if plan.source_id == source_id
                ],
            }
            for source_id in source_ids
        ]

    primary_reads, recovery_reads, runtime_recovery_plans, read_failures = (
        _read_plans_with_runtime_recovery(
            adapter=object(),
            target=object(),
            sources_by_id=sources_by_id,
            remote_agent_names=remote_agent_names,
            primary_plans=context["primary_plans"],
            planned_recovery_plans=context["recovery_plans"],
            alternate_ownerships=context["alternate_ownerships"],
            request=request,
            timeout_seconds=1.0,
            read_fn=fake_read_groups,
        )
    )

    assert attempted_sources == [
        "trainer-rank0",
        "trainer-rank1",
        "trainer-rank2-alt",
    ]
    assert [failure["source_id"] for failure in read_failures] == ["trainer-rank0"]
    assert [read["source_id"] for read in primary_reads] == ["trainer-rank1"]
    assert [read["source_id"] for read in recovery_reads] == ["trainer-rank2-alt"]
    assert [plan.source_id for plan in runtime_recovery_plans] == ["trainer-rank2-alt"]
    assert runtime_recovery_plans[0].target_range == ((2, 3), (0, 4))
    assert runtime_recovery_plans[0].bytes == 16


def test_crossnode_scaled_payload_preserves_exact_recovery_ranges():
    run_id = "unit-scaled-payload"
    request = _run_request_for_payload(run_id, payload_columns=16)
    endpoints = [
        _endpoint(owner, status=p2p_pb2.SOURCE_STATUS_READY)
        for owner in _run_ownerships_for_payload(run_id, payload_columns=16)
    ]

    context = _plan_context_for_endpoints(
        endpoints=endpoints,
        request=request,
        metadata_query_duration_ms=1.0,
        planner_duration_ms=0.0,
        failed_source_ids={"trainer-rank0"},
        stale_source_ids=set(),
    )

    assert request.requested_range == ((2, 6), (0, 16))
    assert request.target_shape == (4, 16)
    assert all(endpoint.ownership.global_shape == (8, 16) for endpoint in endpoints)
    assert [plan.source_id for plan in context["recovery_plans"]] == [
        "trainer-rank2-alt"
    ]
    assert context["recovery_plans"][0].source_range == ((2, 3), (0, 16))
    assert context["recovery_plans"][0].target_range == ((2, 3), (0, 16))
    assert context["recovery_plans"][0].bytes == 64


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
