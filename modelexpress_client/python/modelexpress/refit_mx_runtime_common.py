# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for MX-endpoint runtime refit harnesses."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from . import p2p_pb2
from .refit_trainer_step import (
    DistributedTrainerContext,
    TrainerLoopStepPublication,
    publish_distributed_trainer_loop_step,
    publish_trainer_loop_step,
    trainer_loop_model_version,
)
from .resharding import SliceOwnership, SliceRequest, plan_segments
from .resharding_control_plane import RefitNixlEndpoint

SOURCE_PUBLISHER_TRAINER_LOOP_SMOKE = "trainer-loop-smoke"
SOURCE_PUBLISHER_DISTRIBUTED_TRAINER_LOOP = "distributed-trainer-loop"
SOURCE_PUBLISHER_CHOICES = (
    SOURCE_PUBLISHER_TRAINER_LOOP_SMOKE,
    SOURCE_PUBLISHER_DISTRIBUTED_TRAINER_LOOP,
)


def effective_model_version(model_version: str, run_id: str) -> str:
    return f"{model_version}-{run_id}"


def trainer_loop_runtime_model_version(
    model_version: str,
    run_id: str,
    trainer_step_index: int,
) -> str:
    """Return the MX runtime model version for one trainer-loop source step."""

    return trainer_loop_model_version(
        effective_model_version(model_version, run_id),
        trainer_step_index,
    )


def materialize_trainer_loop_publication(
    ownerships: Sequence[SliceOwnership],
    *,
    dtype: torch.dtype,
    device: torch.device,
    trainer_step_index: int,
    model_version: str | None = None,
) -> TrainerLoopStepPublication:
    """Create one trainer-loop publication for runtime bridge source pods."""

    return publish_trainer_loop_step(
        ownerships,
        dtype=dtype,
        device=device,
        step_index=trainer_step_index,
        model_version=model_version,
    )


def materialize_runtime_source_publication(
    ownerships: Sequence[SliceOwnership],
    *,
    dtype: torch.dtype,
    device: torch.device,
    trainer_step_index: int,
    model_version: str | None = None,
    source_publisher: str = SOURCE_PUBLISHER_TRAINER_LOOP_SMOKE,
    distributed_context: DistributedTrainerContext | None = None,
    synchronize_distributed: bool = True,
) -> TrainerLoopStepPublication:
    """Create source publications for the selected runtime bridge publisher."""

    source_publisher = normalize_source_publisher(source_publisher)
    if source_publisher == SOURCE_PUBLISHER_DISTRIBUTED_TRAINER_LOOP:
        return publish_distributed_trainer_loop_step(
            ownerships,
            dtype=dtype,
            device=device,
            step_index=trainer_step_index,
            model_version=model_version,
            distributed_context=distributed_context,
            synchronize_distributed=synchronize_distributed,
        )
    return materialize_trainer_loop_publication(
        ownerships,
        dtype=dtype,
        device=device,
        trainer_step_index=trainer_step_index,
        model_version=model_version,
    )


def normalize_source_publisher(source_publisher: str) -> str:
    """Validate and normalize a runtime source-publisher name."""

    source_publisher = str(source_publisher or SOURCE_PUBLISHER_TRAINER_LOOP_SMOKE)
    if source_publisher not in SOURCE_PUBLISHER_CHOICES:
        raise ValueError(
            f"unsupported source publisher {source_publisher!r}; "
            f"expected one of {SOURCE_PUBLISHER_CHOICES}"
        )
    return source_publisher


def trainer_framework_for_source_publisher(source_publisher: str) -> str:
    """Return the MX endpoint identity framework for a source publisher."""

    source_publisher = normalize_source_publisher(source_publisher)
    if source_publisher == SOURCE_PUBLISHER_DISTRIBUTED_TRAINER_LOOP:
        return "torch.distributed+torch.optim.SGD-trainer-loop"
    return "torch.optim.SGD-trainer-loop-smoke"


def parse_shape(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        dims = tuple(int(part) for part in value.replace("x", ",").split(",") if part)
    else:
        dims = tuple(int(dim) for dim in value)
    if len(dims) < 2 or any(dim <= 0 for dim in dims):
        raise ValueError(f"target shape must be rank-2+ positive dims, got {value!r}")
    return dims


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def filter_source_ownerships(
    ownerships: Sequence[SliceOwnership],
    *,
    source_id: str = "",
    worker_rank: int | None = None,
) -> list[SliceOwnership]:
    filtered = list(ownerships)
    if source_id:
        filtered = [owner for owner in filtered if owner.source_id == source_id]
    if worker_rank is not None:
        filtered = [
            owner
            for owner in filtered
            if int(owner.worker_rank or 0) == int(worker_rank)
        ]
    if not filtered:
        available = [
            {
                "source_id": owner.source_id,
                "worker_rank": int(owner.worker_rank or 0),
            }
            for owner in ownerships
        ]
        raise ValueError(
            "source ownership filter matched no ownerships "
            f"(source_id={source_id!r}, worker_rank={worker_rank!r}, "
            f"available={available})"
        )
    return filtered


def source_status_name(status: int) -> str:
    try:
        return p2p_pb2.SourceStatus.Name(int(status))
    except ValueError:
        return f"SOURCE_STATUS_{int(status)}"


def scenario_with_mx_endpoint_plan(
    scenario: dict[str, Any],
    endpoints: Sequence[RefitNixlEndpoint],
    *,
    mode: str,
    plan_source: str,
    metadata_query_duration_ms: float,
    planner_duration_ms: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replace locally synthesized source metadata with MX-discovered endpoints."""

    request = SliceRequest.from_dict(scenario["request"])
    sorted_endpoints = sorted(
        endpoints, key=lambda item: (item.source_id, item.worker_id)
    )
    endpoints_by_source_id = {
        endpoint.source_id: endpoint for endpoint in sorted_endpoints
    }
    owners = [endpoint.ownership for endpoint in sorted_endpoints]
    plans = plan_segments(owners, [request])
    planned_source_ids = {plan.source_id for plan in plans}
    missing_endpoints = sorted(planned_source_ids - set(endpoints_by_source_id))
    if missing_endpoints:
        raise RuntimeError(
            f"planned source ids are missing endpoints: {missing_endpoints}"
        )

    planned_endpoint_by_source_id = {
        source_id: endpoints_by_source_id[source_id]
        for source_id in sorted(planned_source_ids)
    }
    updated = dict(scenario)
    updated["source_ownerships"] = [
        endpoint.ownership.to_dict()
        for endpoint in planned_endpoint_by_source_id.values()
    ]
    updated["segment_plans"] = [plan.to_dict() for plan in plans]

    plan_context = {
        "mode": mode,
        "plan_source": plan_source,
        "endpoint_source": "mx-worker-metadata",
        "request": request.to_dict(),
        "plans": plans,
        "source_endpoints_by_id": planned_endpoint_by_source_id,
        "discovered_source_endpoints_by_id": endpoints_by_source_id,
        "source_statuses_by_id": {
            source_id: {
                "status": int(endpoint.status),
                "status_name": source_status_name(endpoint.status),
                "updated_at": int(endpoint.updated_at),
            }
            for source_id, endpoint in sorted(endpoints_by_source_id.items())
        },
        "metadata_query_duration_ms": float(metadata_query_duration_ms),
        "planner_duration_ms": float(planner_duration_ms),
    }
    return updated, plan_context
