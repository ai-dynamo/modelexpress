# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live MX metadata helpers for the synthetic refit POC."""

from __future__ import annotations

import time
from typing import Any

from . import p2p_pb2
from .client import MxClient
from .refit_poc_scenario import (
    ALTERNATE_SOURCE_IDS,
    CONTROL_PLANE_LIVE_MX,
    FAILED_PRIMARY_SOURCE_IDS,
    MODEL_NAME,
    MODEL_VERSION,
    PRIMARY_SOURCE_IDS,
    inference_request,
    ownerships_by_rank,
)
from .resharding import SliceOwnership, SliceRequest, plan_segments
from .resharding_control_plane import (
    build_refit_source_identity,
    list_refit_nixl_endpoints,
    list_slice_ownerships,
    publish_refit_nixl_endpoint,
    publish_slice_ownerships,
)
from .types import TensorDescriptor


def refit_source_identity():
    return build_refit_source_identity(
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        dtype="float32",
        trainer_framework="synthetic-fsdp",
        trainer_layout="fsdp",
    )


def publish_live_mx_rank_ownership(rank: int) -> dict[str, Any]:
    owner = ownerships_by_rank().get(rank)
    if owner is None:
        return {
            "mode": CONTROL_PLANE_LIVE_MX,
            "published": False,
            "reason": "rank-has-no-source-ownership",
        }

    mx_client = MxClient()
    try:
        identity = refit_source_identity()
        publish_start = time.perf_counter()
        mx_source_id = publish_slice_ownerships(
            mx_client,
            identity=identity,
            ownerships=[owner],
            worker_id=owner.worker_id,
            worker_rank=rank,
        )
        publish_duration_ms = (time.perf_counter() - publish_start) * 1000

        status_start = time.perf_counter()
        status_updated = mx_client.update_status(
            mx_source_id,
            owner.worker_id,
            rank,
            p2p_pb2.SOURCE_STATUS_READY,
        )
        status_duration_ms = (time.perf_counter() - status_start) * 1000
        if not status_updated:
            raise RuntimeError(
                f"failed to mark live MX ownership READY for rank {rank}"
            )
        return {
            "mode": CONTROL_PLANE_LIVE_MX,
            "published": True,
            "mx_source_id": mx_source_id,
            "worker_id": owner.worker_id,
            "worker_rank": rank,
            "source_id": owner.source_id,
            "publish_duration_ms": publish_duration_ms,
            "status_update_duration_ms": status_duration_ms,
            "server_url": getattr(mx_client, "server_url", ""),
        }
    finally:
        mx_client.close()


def publish_live_mx_rank_endpoint(
    rank: int,
    *,
    agent_name: str,
    nixl_metadata: bytes,
    tensor_addr: int,
    tensor_bytes: int,
    device_id: int,
) -> dict[str, Any]:
    owner = ownerships_by_rank().get(rank)
    if owner is None:
        return {
            "mode": CONTROL_PLANE_LIVE_MX,
            "published": False,
            "reason": "rank-has-no-source-ownership",
        }

    mx_client = MxClient()
    try:
        identity = refit_source_identity()
        tensor = TensorDescriptor(
            name=owner.tensor_name,
            addr=tensor_addr,
            size=tensor_bytes,
            device_id=device_id,
            dtype=owner.dtype,
        )
        publish_start = time.perf_counter()
        mx_source_id = publish_refit_nixl_endpoint(
            mx_client,
            identity=identity,
            ownership=owner,
            tensor=tensor,
            agent_name=agent_name,
            nixl_metadata=nixl_metadata,
            worker_id=owner.worker_id,
            worker_rank=rank,
            status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
        )
        publish_duration_ms = (time.perf_counter() - publish_start) * 1000

        status_start = time.perf_counter()
        status_updated = mx_client.update_status(
            mx_source_id,
            owner.worker_id,
            rank,
            p2p_pb2.SOURCE_STATUS_READY,
        )
        status_duration_ms = (time.perf_counter() - status_start) * 1000
        if not status_updated:
            raise RuntimeError(
                f"failed to mark live MX NIXL endpoint READY for rank {rank}"
            )
        return {
            "mode": CONTROL_PLANE_LIVE_MX,
            "published": True,
            "mx_source_id": mx_source_id,
            "worker_id": owner.worker_id,
            "worker_rank": rank,
            "source_id": owner.source_id,
            "publish_duration_ms": publish_duration_ms,
            "status_update_duration_ms": status_duration_ms,
            "server_url": getattr(mx_client, "server_url", ""),
            "agent_name": agent_name,
            "metadata_bytes": len(nixl_metadata),
            "tensor_bytes": tensor_bytes,
            "tensor_addr": tensor_addr,
            "tensor_device_id": device_id,
            "endpoint_published": True,
        }
    finally:
        mx_client.close()


def live_mx_plan_context(
    request: SliceRequest,
    *,
    timeout_seconds: float = 30.0,
    mx_client=None,
) -> dict[str, Any]:
    owns_client = mx_client is None
    if mx_client is None:
        mx_client = MxClient()

    try:
        identity = refit_source_identity()
        expected_source_ids = set(PRIMARY_SOURCE_IDS) | set(ALTERNATE_SOURCE_IDS)
        deadline = time.time() + timeout_seconds
        metadata_query_start = time.perf_counter()
        discovered: list[SliceOwnership] = []
        while True:
            discovered = list_slice_ownerships(
                mx_client,
                identity=identity,
            )
            present_source_ids = {owner.source_id for owner in discovered}
            if expected_source_ids.issubset(present_source_ids):
                break
            if time.time() >= deadline:
                missing = sorted(expected_source_ids - present_source_ids)
                raise RuntimeError(
                    "timed out waiting for live MX slice ownership metadata "
                    f"(missing={missing}, present={sorted(present_source_ids)})"
                )
            time.sleep(0.1)
        metadata_query_duration_ms = (
            time.perf_counter() - metadata_query_start
        ) * 1000

        primary_owners = _select_ownerships_by_source_id(
            discovered,
            PRIMARY_SOURCE_IDS,
        )
        alternate_owners = _select_ownerships_by_source_id(
            discovered,
            ALTERNATE_SOURCE_IDS,
        )
        plan_context = _plan_primary_and_recovery(
            request,
            primary_owners=primary_owners,
            alternate_owners=alternate_owners,
        )
        plan_context.update(
            {
                "mode": CONTROL_PLANE_LIVE_MX,
                "plan_source": "live-mx-server",
                "discovered_ownerships": discovered,
                "metadata_query_duration_ms": metadata_query_duration_ms,
                "server_url": getattr(mx_client, "server_url", ""),
            }
        )
        return plan_context
    finally:
        if owns_client:
            mx_client.close()


def live_mx_endpoint_context(
    request: SliceRequest,
    *,
    timeout_seconds: float = 30.0,
    mx_client=None,
) -> dict[str, Any]:
    owns_client = mx_client is None
    if mx_client is None:
        mx_client = MxClient()

    try:
        identity = refit_source_identity()
        expected_source_ids = set(PRIMARY_SOURCE_IDS) | set(ALTERNATE_SOURCE_IDS)
        deadline = time.time() + timeout_seconds
        metadata_query_start = time.perf_counter()
        endpoints_by_source_id = {}
        while True:
            endpoints = list_refit_nixl_endpoints(
                mx_client,
                identity=identity,
            )
            endpoints_by_source_id = {
                endpoint.source_id: endpoint
                for endpoint in sorted(
                    endpoints,
                    key=lambda item: (item.source_id, item.worker_id),
                )
            }
            present_source_ids = set(endpoints_by_source_id)
            if expected_source_ids.issubset(present_source_ids):
                break
            if time.time() >= deadline:
                missing = sorted(expected_source_ids - present_source_ids)
                raise RuntimeError(
                    "timed out waiting for live MX NIXL endpoint metadata "
                    f"(missing={missing}, present={sorted(present_source_ids)})"
                )
            time.sleep(0.1)
        metadata_query_duration_ms = (
            time.perf_counter() - metadata_query_start
        ) * 1000

        primary_endpoints = _select_endpoints_by_source_id(
            endpoints_by_source_id,
            PRIMARY_SOURCE_IDS,
        )
        alternate_endpoints = _select_endpoints_by_source_id(
            endpoints_by_source_id,
            ALTERNATE_SOURCE_IDS,
        )
        plan_context = _plan_primary_and_recovery(
            request,
            primary_owners=[endpoint.ownership for endpoint in primary_endpoints],
            alternate_owners=[endpoint.ownership for endpoint in alternate_endpoints],
        )
        plan_context.update(
            {
                "mode": CONTROL_PLANE_LIVE_MX,
                "plan_source": "live-mx-server",
                "endpoint_source": "mx-worker-metadata",
                "discovered_ownerships": [
                    endpoint.ownership
                    for endpoint in endpoints_by_source_id.values()
                ],
                "source_endpoints_by_id": endpoints_by_source_id,
                "metadata_query_duration_ms": metadata_query_duration_ms,
                "server_url": getattr(mx_client, "server_url", ""),
            }
        )
        return plan_context
    finally:
        if owns_client:
            mx_client.close()


def _plan_primary_and_recovery(
    request: SliceRequest,
    *,
    primary_owners: list[SliceOwnership],
    alternate_owners: list[SliceOwnership],
) -> dict[str, Any]:
    planner_start = time.perf_counter()
    primary_plans = plan_segments(primary_owners, [request])
    failed_primary_plans = [
        plan for plan in primary_plans if plan.source_id in FAILED_PRIMARY_SOURCE_IDS
    ]
    recovery_requests = [
        inference_request(requested_range=plan.target_range)
        for plan in failed_primary_plans
    ]
    recovery_plans = plan_segments(alternate_owners, recovery_requests)
    planner_duration_ms = (time.perf_counter() - planner_start) * 1000

    return {
        "primary_plans": primary_plans,
        "recovery_plans": recovery_plans,
        "primary_ownerships": primary_owners,
        "alternate_ownerships": alternate_owners,
        "planner_duration_ms": planner_duration_ms,
    }


def _select_ownerships_by_source_id(
    ownerships: list[SliceOwnership],
    source_ids: tuple[str, ...],
) -> list[SliceOwnership]:
    selected: dict[str, SliceOwnership] = {}
    for owner in sorted(ownerships, key=lambda item: (item.source_id, item.worker_id)):
        if owner.source_id in source_ids:
            selected[owner.source_id] = owner
    missing = [source_id for source_id in source_ids if source_id not in selected]
    if missing:
        raise RuntimeError(f"missing expected source ownerships: {missing}")
    return [selected[source_id] for source_id in source_ids]


def _select_endpoints_by_source_id(
    endpoints_by_source_id: dict[str, Any],
    source_ids: tuple[str, ...],
) -> list[Any]:
    missing = [
        source_id
        for source_id in source_ids
        if source_id not in endpoints_by_source_id
    ]
    if missing:
        raise RuntimeError(f"missing expected source endpoints: {missing}")
    return [endpoints_by_source_id[source_id] for source_id in source_ids]
