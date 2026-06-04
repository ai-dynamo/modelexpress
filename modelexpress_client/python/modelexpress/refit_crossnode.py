# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-pod cross-node NIXL refit proof runner.

This module is intentionally separate from ``refit_poc.py``. The same-node POC
uses torchrun for process synchronization; this runner uses MX worker metadata
as the only source endpoint exchange path so the source and target can live in
different Kubernetes pods.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from . import p2p_pb2
from .client import MxClient
from .metadata.heartbeat import HeartbeatThread
from .refit_nixl import (
    NixlAdapter,
    apply_nixl_ucx_pin,
    read_segment_groups,
    select_cuda_device,
)
from .refit_poc_artifacts import artifact_base, write_artifact
from .refit_poc_scenario import (
    FAILED_PRIMARY_SOURCE_IDS,
    GLOBAL_SHAPE,
    MODEL_NAME,
    MODEL_VERSION,
    PRIMARY_SOURCE_IDS,
    alternate_ownerships,
    inference_request,
    primary_ownerships,
)
from .resharding import SliceOwnership, SliceRequest, plan_segments
from .resharding_control_plane import (
    build_refit_source_identity,
    list_refit_nixl_endpoints,
    publish_refit_nixl_endpoint,
)
from .types import TensorDescriptor


def _import_torch():
    import torch

    return torch


def _run_model_version(run_id: str) -> str:
    return f"{MODEL_VERSION}-{run_id}"


def _run_identity(run_id: str):
    return build_refit_source_identity(
        model_name=MODEL_NAME,
        model_version=_run_model_version(run_id),
        dtype="float32",
        trainer_framework="synthetic-fsdp",
        trainer_layout="fsdp-cross-node",
    )


def _trace(phase: str, **fields: Any) -> None:
    payload = {"phase": phase, **fields}
    print(
        f"[mx-refit-crossnode] {json.dumps(payload, default=_json_default, sort_keys=True)}",
        flush=True,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, bytes):
        return {
            "bytes": len(value),
            "hex_prefix": value[:16].hex(),
        }
    return repr(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return _json_default(value)
    return value


def _source_status_name(status: int) -> str:
    try:
        return p2p_pb2.SourceStatus.Name(int(status))
    except ValueError:
        return f"SOURCE_STATUS_{int(status)}"


def _source_statuses_by_source_id(
    endpoints_by_source_id,
) -> dict[str, dict[str, int | str]]:
    return {
        source_id: {
            "status": int(endpoint.status),
            "status_name": _source_status_name(endpoint.status),
            "updated_at": int(endpoint.updated_at),
        }
        for source_id, endpoint in sorted(endpoints_by_source_id.items())
    }


def _run_ownerships(run_id: str) -> list[SliceOwnership]:
    return _run_ownerships_for_payload(run_id, payload_columns=GLOBAL_SHAPE[1])


def _run_ownerships_for_payload(
    run_id: str, *, payload_columns: int
) -> list[SliceOwnership]:
    model_version = _run_model_version(run_id)
    return [
        _ownership_for_payload(
            replace(owner, model_version=model_version),
            payload_columns=payload_columns,
        )
        for owner in [*primary_ownerships(), *alternate_ownerships()]
    ]


def _run_request(run_id: str) -> SliceRequest:
    return _run_request_for_payload(run_id, payload_columns=GLOBAL_SHAPE[1])


def _run_request_for_payload(run_id: str, *, payload_columns: int) -> SliceRequest:
    return _request_for_payload(
        replace(
            inference_request(target_id="inference-crossnode-target"),
            model_version=_run_model_version(run_id),
        ),
        payload_columns=payload_columns,
    )


def _normalize_payload_columns(payload_columns: int) -> int:
    payload_columns = int(payload_columns)
    if payload_columns < GLOBAL_SHAPE[1]:
        raise ValueError(
            f"payload_columns must be >= {GLOBAL_SHAPE[1]}, got {payload_columns}"
        )
    return payload_columns


def _ownership_for_payload(
    owner: SliceOwnership, *, payload_columns: int
) -> SliceOwnership:
    payload_columns = _normalize_payload_columns(payload_columns)
    row_range = owner.source_range[0]
    return replace(
        owner,
        global_shape=(owner.global_shape[0], payload_columns),
        source_range=(row_range, (0, payload_columns)),
    )


def _request_for_payload(
    request: SliceRequest, *, payload_columns: int
) -> SliceRequest:
    payload_columns = _normalize_payload_columns(payload_columns)
    row_range = request.requested_range[0]
    requested_range = (row_range, (0, payload_columns))
    return replace(
        request,
        requested_range=requested_range,
        target_shape=_range_extents(requested_range),
    )


def _materialize_range(tensor_range, device):
    torch = _import_torch()
    rows = torch.arange(
        tensor_range[0][0],
        tensor_range[0][1],
        device=device,
        dtype=torch.float32,
    ).view(-1, 1)
    cols = torch.arange(
        tensor_range[1][0],
        tensor_range[1][1],
        device=device,
        dtype=torch.float32,
    ).view(1, -1)
    return rows * 1000.0 + cols


def _target_expected(request: SliceRequest, device):
    return _materialize_range(request.requested_range, device)


def _checksum(tensor) -> float:
    torch = _import_torch()
    flat = tensor.float().reshape(-1)
    weights = torch.arange(
        1, flat.numel() + 1, device=tensor.device, dtype=torch.float32
    )
    return float((flat * weights).sum().item())


def _validate_target(target, request: SliceRequest) -> dict[str, Any]:
    torch = _import_torch()
    expected = _target_expected(request, target.device)
    torch.cuda.synchronize(target.device)
    return {
        "allclose": bool(torch.allclose(target, expected)),
        "checksum": _checksum(target),
        "expected_checksum": _checksum(expected),
        "max_abs_error": float((target - expected).abs().max().item()),
    }


def _plan_context_for_endpoints(
    *,
    endpoints,
    request: SliceRequest,
    metadata_query_duration_ms: float,
    planner_duration_ms: float,
    failed_source_ids: set[str] | None = None,
    stale_source_ids: set[str] | None = None,
) -> dict[str, Any]:
    failed_source_ids = set(
        FAILED_PRIMARY_SOURCE_IDS if failed_source_ids is None else failed_source_ids
    )
    stale_source_ids = set(stale_source_ids or ())
    unreadable_source_ids = failed_source_ids | stale_source_ids
    endpoints_by_source_id = {
        endpoint.source_id: endpoint
        for endpoint in sorted(
            endpoints, key=lambda item: (item.source_id, item.worker_id)
        )
    }
    primary_owners = [
        endpoints_by_source_id[source_id].ownership for source_id in PRIMARY_SOURCE_IDS
    ]
    alternate_owners = [
        endpoint.ownership
        for endpoint in endpoints_by_source_id.values()
        if endpoint.source_id not in set(PRIMARY_SOURCE_IDS)
    ]
    primary_plans = plan_segments(primary_owners, [request])
    failed_primary_plans = [
        plan for plan in primary_plans if plan.source_id in unreadable_source_ids
    ]
    recovery_requests = [
        replace(
            request,
            requested_range=plan.target_range,
            target_shape=_range_extents(plan.target_range),
        )
        for plan in failed_primary_plans
    ]
    recovery_plans = plan_segments(alternate_owners, recovery_requests)
    return {
        "mode": "live-mx-cross-node",
        "plan_source": "live-mx-server",
        "endpoint_source": "mx-worker-metadata",
        "primary_plans": primary_plans,
        "recovery_plans": recovery_plans,
        "primary_ownerships": primary_owners,
        "alternate_ownerships": alternate_owners,
        "discovered_ownerships": [
            endpoint.ownership for endpoint in endpoints_by_source_id.values()
        ],
        "source_endpoints_by_id": endpoints_by_source_id,
        "source_statuses_by_id": _source_statuses_by_source_id(endpoints_by_source_id),
        "failed_source_ids": sorted(failed_source_ids),
        "stale_source_ids": sorted(stale_source_ids),
        "unreadable_source_ids": sorted(unreadable_source_ids),
        "metadata_query_duration_ms": metadata_query_duration_ms,
        "planner_duration_ms": planner_duration_ms,
    }


def _range_extents(tensor_range):
    return tuple(int(end) - int(start) for start, end in tensor_range)


ReadSegmentGroupsFn = Callable[..., list[dict[str, Any]]]


def _group_plans_by_source(plans) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for plan in plans:
        grouped.setdefault(plan.source_id, []).append(plan)
    return grouped


def _recovery_requests_for_plans(request: SliceRequest, plans) -> list[SliceRequest]:
    return [
        replace(
            request,
            requested_range=plan.target_range,
            target_shape=_range_extents(plan.target_range),
        )
        for plan in plans
    ]


def _read_plans_with_runtime_recovery(
    *,
    adapter,
    target,
    sources_by_id: dict[str, dict[str, Any]],
    remote_agent_names: dict[str, str],
    primary_plans,
    planned_recovery_plans,
    alternate_ownerships,
    request: SliceRequest,
    timeout_seconds: float,
    read_fn: ReadSegmentGroupsFn = read_segment_groups,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Any], list[dict[str, Any]]]:
    """Read primary segments and replan failed source groups from alternates.

    Planned failed/stale recovery still happens before this helper is called.
    This helper covers a different case: a source that was READY and selected for
    a primary read fails while the target is issuing that source's read group.
    """

    primary_reads: list[dict[str, Any]] = []
    read_failures: list[dict[str, Any]] = []
    failed_primary_plans = []
    for source_id, source_plans in sorted(
        _group_plans_by_source(primary_plans).items()
    ):
        try:
            primary_reads.extend(
                read_fn(
                    adapter=adapter,
                    target=target,
                    sources_by_id=sources_by_id,
                    remote_agent_names=remote_agent_names,
                    plans=source_plans,
                    timeout_seconds=timeout_seconds,
                )
            )
        except Exception as exc:
            _trace(
                "target.read_segments.primary_source_failed",
                source_id=source_id,
                segment_count=len(source_plans),
                error=str(exc),
            )
            read_failures.append(
                {
                    "source_id": source_id,
                    "segment_count": len(source_plans),
                    "target_ranges": [plan.target_range for plan in source_plans],
                    "error": str(exc),
                }
            )
            failed_primary_plans.extend(source_plans)

    runtime_recovery_plans = []
    if failed_primary_plans:
        runtime_recovery_plans = plan_segments(
            alternate_ownerships,
            _recovery_requests_for_plans(request, failed_primary_plans),
        )

    recovery_plans = [*planned_recovery_plans, *runtime_recovery_plans]
    recovery_reads = read_fn(
        adapter=adapter,
        target=target,
        sources_by_id=sources_by_id,
        remote_agent_names=remote_agent_names,
        plans=recovery_plans,
        timeout_seconds=timeout_seconds,
    )
    return primary_reads, recovery_reads, runtime_recovery_plans, read_failures


def _wait_for_endpoints(
    *,
    run_id: str,
    request: SliceRequest,
    expected_source_ids: set[str],
    timeout_seconds: float,
    failed_source_ids: set[str] | None = None,
    stale_source_ids: set[str] | None = None,
) -> dict[str, Any]:
    failed_source_ids = set(
        FAILED_PRIMARY_SOURCE_IDS if failed_source_ids is None else failed_source_ids
    )
    stale_source_ids = set(stale_source_ids or ())
    expected_source_ids = set(expected_source_ids) | stale_source_ids
    expected_ready_source_ids = expected_source_ids - stale_source_ids
    status_filter = None if stale_source_ids else p2p_pb2.SOURCE_STATUS_READY

    mx_client = MxClient()
    try:
        identity = _run_identity(run_id)
        deadline = time.time() + timeout_seconds
        metadata_query_start = time.perf_counter()
        endpoints = []
        while True:
            endpoints = list_refit_nixl_endpoints(
                mx_client,
                identity=identity,
                status_filter=status_filter,
            )
            present = {endpoint.source_id for endpoint in endpoints}
            ready = {
                endpoint.source_id
                for endpoint in endpoints
                if int(endpoint.status) == p2p_pb2.SOURCE_STATUS_READY
            }
            stale = {
                endpoint.source_id
                for endpoint in endpoints
                if int(endpoint.status) == p2p_pb2.SOURCE_STATUS_STALE
            }
            if (
                expected_source_ids.issubset(present)
                and expected_ready_source_ids.issubset(ready)
                and stale_source_ids.issubset(stale)
            ):
                break
            if time.time() >= deadline:
                raise RuntimeError(
                    "timed out waiting for cross-node MX NIXL endpoints "
                    f"(run_id={run_id}, "
                    f"missing={sorted(expected_source_ids - present)}, "
                    f"missing_ready={sorted(expected_ready_source_ids - ready)}, "
                    f"missing_stale={sorted(stale_source_ids - stale)}, "
                    f"present={sorted(present)}, ready={sorted(ready)}, stale={sorted(stale)})"
                )
            time.sleep(0.25)
        metadata_query_duration_ms = (time.perf_counter() - metadata_query_start) * 1000

        planner_start = time.perf_counter()
        plan_context = _plan_context_for_endpoints(
            endpoints=endpoints,
            request=request,
            metadata_query_duration_ms=metadata_query_duration_ms,
            planner_duration_ms=0.0,
            failed_source_ids=failed_source_ids,
            stale_source_ids=stale_source_ids,
        )
        plan_context["planner_duration_ms"] = (
            time.perf_counter() - planner_start
        ) * 1000
        plan_context["server_url"] = getattr(mx_client, "server_url", "")
        return plan_context
    finally:
        mx_client.close()


def _source_agent_name(run_id: str, ownerships: list[SliceOwnership]) -> str:
    if len(ownerships) == 1:
        return f"mx-crossnode-source-{run_id}-{ownerships[0].source_id}"
    return f"mx-crossnode-source-{run_id}"


def _filter_ownerships(
    ownerships: list[SliceOwnership],
    *,
    source_id: str,
    worker_rank: int | None,
) -> list[SliceOwnership]:
    filtered = ownerships
    if source_id:
        filtered = [owner for owner in filtered if owner.source_id == source_id]
    if worker_rank is not None:
        filtered = [
            owner for owner in filtered if int(owner.worker_rank or 0) == worker_rank
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
            f"(source_id={source_id!r}, worker_rank={worker_rank!r}, available={available})"
        )
    return filtered


def run_source(
    *,
    run_id: str,
    hold_seconds: float,
    local_rank: int,
    artifact_path: Path | None,
    source_id: str = "",
    source_worker_rank: int | None = None,
    payload_columns: int = GLOBAL_SHAPE[1],
) -> dict[str, Any]:
    torch = _import_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cross-node source publishing")

    device_index, gpu_reuse_used = select_cuda_device(local_rank)
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    apply_nixl_ucx_pin(device_index)

    payload_columns = _normalize_payload_columns(payload_columns)
    ownerships = _filter_ownerships(
        _run_ownerships_for_payload(run_id, payload_columns=payload_columns),
        source_id=source_id,
        worker_rank=source_worker_rank,
    )
    agent_name = _source_agent_name(run_id, ownerships)
    adapter = NixlAdapter(agent_name)
    source_tensors = {
        owner.source_id: _materialize_range(owner.source_range, device).contiguous()
        for owner in ownerships
    }

    register_start = time.perf_counter()
    for tensor in source_tensors.values():
        adapter.register_tensor(tensor)
    torch.cuda.synchronize(device)
    registration_duration_ms = (time.perf_counter() - register_start) * 1000

    metadata_start = time.perf_counter()
    metadata = adapter.metadata_bytes()
    metadata_duration_ms = (time.perf_counter() - metadata_start) * 1000

    mx_client = MxClient()
    heartbeats: list[HeartbeatThread] = []
    publications = []
    try:
        identity = _run_identity(run_id)
        for owner in ownerships:
            tensor = source_tensors[owner.source_id]
            descriptor = TensorDescriptor(
                name=owner.tensor_name,
                addr=int(tensor.data_ptr()),
                size=int(tensor.numel() * tensor.element_size()),
                device_id=int(tensor.get_device()),
                dtype=owner.dtype,
            )
            publish_start = time.perf_counter()
            mx_source_id = publish_refit_nixl_endpoint(
                mx_client,
                identity=identity,
                ownership=owner,
                tensor=descriptor,
                agent_name=agent_name,
                nixl_metadata=metadata,
                worker_id=owner.worker_id,
                worker_rank=int(owner.worker_rank or 0),
                status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
            )
            publish_duration_ms = (time.perf_counter() - publish_start) * 1000
            heartbeat = HeartbeatThread(
                mx_client=mx_client,
                mx_source_id=mx_source_id,
                worker_id=owner.worker_id,
                worker_rank=int(owner.worker_rank or 0),
                nixl_manager=None,
            )
            heartbeat.start()
            heartbeats.append(heartbeat)
            publications.append(
                {
                    "mx_source_id": mx_source_id,
                    "worker_id": owner.worker_id,
                    "worker_rank": owner.worker_rank,
                    "source_id": owner.source_id,
                    "agent_name": agent_name,
                    "metadata_bytes": len(metadata),
                    "tensor_addr": descriptor.addr,
                    "tensor_bytes": descriptor.size,
                    "tensor_device_id": descriptor.device_id,
                    "publish_duration_ms": publish_duration_ms,
                }
            )

        result = {
            "schema_version": 1,
            "result": "source-ready",
            "run_id": run_id,
            "role": "source",
            "mode": "nixl-crossnode-source",
            "pod_name": os.environ.get("POD_NAME", ""),
            "node_name": os.environ.get("NODE_NAME", ""),
            "agent_name": agent_name,
            "single_source_ownership_mode": len(ownerships) == 1,
            "source_ownership_count": len(ownerships),
            "source_filter": {
                "source_id": source_id,
                "source_worker_rank": source_worker_rank,
            },
            "payload_columns": payload_columns,
            "gpu_reuse_used": gpu_reuse_used,
            "cuda_device": device_index,
            "nixl_backends": adapter.backends,
            "ucx": _ucx_env_snapshot(),
            "metrics": {
                "nixl_registration_duration_ms": registration_duration_ms,
                "nixl_metadata_duration_ms": metadata_duration_ms,
                "publish_duration_ms": sum(
                    item["publish_duration_ms"] for item in publications
                ),
            },
            "source_publications": publications,
        }
        if artifact_path is not None:
            write_artifact(result, artifact_path)
        else:
            print(json.dumps(result, sort_keys=True), flush=True)

        time.sleep(max(0.0, hold_seconds))
        return result
    finally:
        for heartbeat in heartbeats:
            heartbeat.stop()
        mx_client.close()


def run_target(
    *,
    run_id: str,
    local_rank: int,
    artifact_path: Path,
    timeout_seconds: float,
    source_pod: str,
    source_node: str,
    target_pod: str,
    target_node: str,
    gpu_count: int,
    failed_source_ids: set[str] | None = None,
    stale_source_ids: set[str] | None = None,
    payload_columns: int = GLOBAL_SHAPE[1],
) -> dict[str, Any]:
    torch = _import_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cross-node target reads")

    failed_source_ids = set(
        FAILED_PRIMARY_SOURCE_IDS if failed_source_ids is None else failed_source_ids
    )
    stale_source_ids = set(stale_source_ids or ())
    unreadable_source_ids = failed_source_ids | stale_source_ids

    device_index, gpu_reuse_used = select_cuda_device(local_rank)
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    apply_nixl_ucx_pin(device_index)

    payload_columns = _normalize_payload_columns(payload_columns)
    request = _run_request_for_payload(run_id, payload_columns=payload_columns)
    expected_source_ids = {
        *PRIMARY_SOURCE_IDS,
        *{owner.source_id for owner in alternate_ownerships()},
    }
    _trace(
        "target.wait_for_mx_endpoints.start",
        run_id=run_id,
        expected_source_ids=sorted(expected_source_ids),
        failed_source_ids=sorted(failed_source_ids),
        stale_source_ids=sorted(stale_source_ids),
    )
    plan_context = _wait_for_endpoints(
        run_id=run_id,
        request=request,
        expected_source_ids=expected_source_ids,
        timeout_seconds=timeout_seconds,
        failed_source_ids=failed_source_ids,
        stale_source_ids=stale_source_ids,
    )
    _trace(
        "target.wait_for_mx_endpoints.done",
        run_id=run_id,
        source_ids=sorted(plan_context["source_endpoints_by_id"]),
        source_statuses=plan_context["source_statuses_by_id"],
        metadata_query_duration_ms=plan_context["metadata_query_duration_ms"],
        planner_duration_ms=plan_context["planner_duration_ms"],
    )

    _trace("target.allocate_buffer.start", shape=list(request.target_shape))
    target = torch.full(request.target_shape, float("nan"), device=device)
    torch.cuda.synchronize(device)
    _trace("target.allocate_buffer.done", device=str(target.device))

    agent_name = f"mx-crossnode-target-{run_id}"
    _trace(
        "target.nixl_agent.construct.start",
        agent_name=agent_name,
        ucx=_ucx_env_snapshot(),
    )
    adapter = NixlAdapter(agent_name)
    _trace(
        "target.nixl_agent.construct.done",
        agent_name=agent_name,
        nixl_backends=adapter.backends,
    )
    _trace("target.register_target.start", bytes=target.numel() * target.element_size())
    register_start = time.perf_counter()
    adapter.register_tensor(target)
    torch.cuda.synchronize(device)
    registration_duration_ms = (time.perf_counter() - register_start) * 1000
    _trace(
        "target.register_target.done",
        registration_duration_ms=registration_duration_ms,
    )

    endpoint_by_source_id = plan_context["source_endpoints_by_id"]
    readable_endpoint_by_source_id = {
        source_id: endpoint
        for source_id, endpoint in endpoint_by_source_id.items()
        if source_id not in unreadable_source_ids
        and int(endpoint.status) == p2p_pb2.SOURCE_STATUS_READY
    }
    sources_by_id = {
        source_id: endpoint.to_nixl_source_info()
        for source_id, endpoint in readable_endpoint_by_source_id.items()
    }

    remote_agent_cache: dict[str, str] = {}
    add_remote_timings: dict[str, float] = {}
    remote_agent_names: dict[str, str] = {}
    for source_id, endpoint in sorted(readable_endpoint_by_source_id.items()):
        cache_key = endpoint.agent_name
        if cache_key not in remote_agent_cache:
            _trace(
                "target.add_remote_agent.start",
                source_id=source_id,
                agent_name=endpoint.agent_name,
                metadata_bytes=len(endpoint.nixl_metadata),
            )
            add_start = time.perf_counter()
            remote_agent_cache[cache_key] = adapter.add_remote_agent(
                endpoint.nixl_metadata
            )
            add_remote_timings[cache_key] = (time.perf_counter() - add_start) * 1000
            _trace(
                "target.add_remote_agent.done",
                source_id=source_id,
                remote_agent_name=remote_agent_cache[cache_key],
                add_remote_agent_duration_ms=add_remote_timings[cache_key],
            )
        remote_agent_names[source_id] = remote_agent_cache[cache_key]

    primary_read_plans = [
        plan
        for plan in plan_context["primary_plans"]
        if plan.source_id not in unreadable_source_ids
    ]
    planned_read_source_ids = {
        plan.source_id
        for plan in [*primary_read_plans, *plan_context["recovery_plans"]]
    }
    missing_readable_sources = sorted(planned_read_source_ids - set(sources_by_id))
    if missing_readable_sources:
        raise RuntimeError(
            "planned NIXL reads reference non-ready or failed MX sources "
            f"(missing_readable_sources={missing_readable_sources}, "
            f"source_statuses={plan_context['source_statuses_by_id']})"
        )

    _trace(
        "target.read_segments.start",
        primary_segment_count=len(primary_read_plans),
        recovery_segment_count=len(plan_context["recovery_plans"]),
        readable_source_ids=sorted(sources_by_id),
    )
    transfer_start = time.perf_counter()
    (
        primary_reads,
        recovery_reads,
        runtime_recovery_plans,
        read_failures,
    ) = _read_plans_with_runtime_recovery(
        adapter=adapter,
        target=target,
        sources_by_id=sources_by_id,
        remote_agent_names=remote_agent_names,
        primary_plans=primary_read_plans,
        planned_recovery_plans=plan_context["recovery_plans"],
        alternate_ownerships=plan_context["alternate_ownerships"],
        request=request,
        timeout_seconds=timeout_seconds,
    )
    torch.cuda.synchronize(device)
    raw_nixl_read_duration_ms = (time.perf_counter() - transfer_start) * 1000
    _trace(
        "target.read_segments.done",
        raw_nixl_read_duration_ms=raw_nixl_read_duration_ms,
    )
    validation = _validate_target(target, request)
    _trace("target.validation.done", **validation)
    all_reads = [*primary_reads, *recovery_reads]
    copied_bytes = sum(read["bytes"] for read in all_reads)
    read_source_ids = {read["source_id"] for read in all_reads}
    discovered_source_agent_names = {
        source_id: endpoint.agent_name
        for source_id, endpoint in sorted(endpoint_by_source_id.items())
    }
    readable_source_agent_names = {
        source_id: endpoint.agent_name
        for source_id, endpoint in sorted(readable_endpoint_by_source_id.items())
    }
    distinct_source_agent_count = len(set(readable_source_agent_names.values()))
    source_endpoint_count = len(sources_by_id)
    one_agent_per_source_rank = distinct_source_agent_count == source_endpoint_count
    multipod_source_mode = one_agent_per_source_rank and source_endpoint_count > 1
    stale_primary_planned = stale_source_ids.intersection(
        plan.source_id for plan in plan_context["primary_plans"]
    )
    stale_source_recovery_used = (
        bool(stale_primary_planned)
        and bool(recovery_reads)
        and validation["allclose"]
        and stale_source_ids.isdisjoint(read_source_ids)
    )
    read_failure_recovery_used = (
        bool(runtime_recovery_plans)
        and validation["allclose"]
        and {failure["source_id"] for failure in read_failures}.isdisjoint(
            read_source_ids
        )
    )

    result = artifact_base(
        mode=(
            "nixl-crossnode-one-pod-per-source-rank"
            if multipod_source_mode
            else "nixl-crossnode-2pod"
        ),
        gpu_count=gpu_count,
        copied_bytes=copied_bytes,
        copy_duration_ms=raw_nixl_read_duration_ms,
        validation=validation,
        request=request,
        plan_context=plan_context,
    )
    result["proof"].update(
        {
            "actual_nixl_reads_used": True,
            "cross_node_pods": bool(
                source_node and target_node and source_node != target_node
            ),
            "source_pod_separate_from_target_pod": bool(
                source_pod and target_pod and source_pod != target_pod
            ),
            "nixl_source_endpoints_from_mx": True,
            "torch_distributed_control_plane_used": False,
            "torch_distributed_data_transfer_used": False,
            "torch_distributed_nixl_metadata_exchange_used": False,
            "live_mx_returned_metadata_used_for_nixl_plan": True,
            "target_buffer_preallocated": True,
            "nixl_reads_land_at_segment_offsets": validation["allclose"],
            "trainer_full_all_gather_used": False,
            "trainer_side_inference_layout_conversion_used": False,
            "host_side_torch_cat_used": False,
            "ucx_ib_rc_requested": _ucx_ib_rc_requested(),
            "mlx5_bond_excluded_from_ucx_devices": (
                "bond" not in os.environ.get("UCX_NET_DEVICES", "")
                and "bond" not in os.environ.get("UCX_IB_DEVICES", "")
            ),
            "one_nixl_agent_per_source_rank": one_agent_per_source_rank,
            "source_endpoint_count": source_endpoint_count,
            "readable_source_endpoint_count": source_endpoint_count,
            "discovered_source_endpoint_count": len(endpoint_by_source_id),
            "distinct_source_agent_count": distinct_source_agent_count,
            "failed_source_ids": sorted(failed_source_ids),
            "stale_source_ids": sorted(stale_source_ids),
            "unreadable_source_ids": sorted(unreadable_source_ids),
            "readable_source_ids": sorted(sources_by_id),
            "read_source_ids": sorted(read_source_ids),
            "stale_source_endpoint_statuses": {
                source_id: plan_context["source_statuses_by_id"].get(source_id)
                for source_id in sorted(stale_source_ids)
            },
            "stale_source_ids_excluded_from_nixl_reads": stale_source_ids.isdisjoint(
                read_source_ids
            ),
            "stale_source_recovery_used": stale_source_recovery_used,
            "read_failure_recovery_used": read_failure_recovery_used,
            "read_failure_count": len(read_failures),
            "read_failure_source_ids": sorted(
                {failure["source_id"] for failure in read_failures}
            ),
            "read_failure_sources_excluded_from_successful_reads": (
                {failure["source_id"] for failure in read_failures}.isdisjoint(
                    read_source_ids
                )
            ),
            "payload_columns": payload_columns,
        }
    )
    result["distributed"] = {
        "backend": "nixl-read+mx-endpoint-control+no-torch-distributed",
        "world_size": gpu_count,
        "source_pod": source_pod,
        "target_pod": target_pod,
        "source_node": source_node,
        "target_node": target_node,
        "cross_node": bool(source_node and target_node and source_node != target_node),
        "source_ids": sorted(sources_by_id),
        "readable_source_ids": sorted(sources_by_id),
        "discovered_source_ids": sorted(endpoint_by_source_id),
        "source_roles_in_source_pod": sorted(sources_by_id),
        "source_agent_names_by_source_id": readable_source_agent_names,
        "discovered_source_agent_names_by_source_id": discovered_source_agent_names,
        "target_agent_name": agent_name,
        "gpu_reuse_used": gpu_reuse_used,
        "payload_columns": payload_columns,
    }
    result["control_plane"] = {
        "mode": "live-mx-cross-node",
        "server_url": plan_context.get("server_url", ""),
        "source_endpoints_from_control_plane": [
            endpoint.to_dict() for endpoint in endpoint_by_source_id.values()
        ],
        "source_statuses_by_id": plan_context["source_statuses_by_id"],
        "metadata_query_duration_ms": plan_context["metadata_query_duration_ms"],
        "planner_duration_ms": plan_context["planner_duration_ms"],
    }
    result["nixl"] = {
        "target_agent_name": agent_name,
        "nixl_backends": adapter.backends,
        "ucx": _ucx_env_snapshot(),
        "source_metadata": {
            source_id: {
                "agent_name": readable_endpoint_by_source_id[source_id].agent_name,
                "metadata_bytes": len(
                    readable_endpoint_by_source_id[source_id].nixl_metadata
                ),
                "source_range": sources_by_id[source_id]["source_range"],
                "registered_bytes": sources_by_id[source_id]["registered_bytes"],
                "device_id": sources_by_id[source_id]["device_id"],
                "worker_rank": sources_by_id[source_id]["rank"],
            }
            for source_id in sorted(sources_by_id)
        },
        "remote_agent_names": _json_safe(remote_agent_names),
        "add_remote_agent_duration_ms": add_remote_timings,
        "primary_reads": primary_reads,
        "recovery_reads": recovery_reads,
        "runtime_recovery_segment_plans": [
            plan.to_dict() for plan in runtime_recovery_plans
        ],
        "read_failures": _json_safe(read_failures),
    }
    result["metrics"].update(
        {
            "gpu_copy_duration_ms": 0.0,
            "raw_nixl_read_duration_ms": raw_nixl_read_duration_ms,
            "nixl_registration_duration_ms": registration_duration_ms,
            "nixl_add_remote_agent_duration_ms": sum(add_remote_timings.values()),
            "nixl_prep_duration_ms": sum(
                read["prep_duration_ms"] for read in all_reads
            ),
            "nixl_read_group_count": len(all_reads),
            "successful_nixl_source_count": len(read_source_ids),
            "metadata_query_duration_ms": plan_context["metadata_query_duration_ms"],
            "planner_duration_ms": plan_context["planner_duration_ms"],
            "discovered_source_endpoint_count": len(endpoint_by_source_id),
            "readable_source_endpoint_count": source_endpoint_count,
            "failed_source_count": len(failed_source_ids),
            "stale_source_count": len(stale_source_ids),
            "read_failure_count": len(read_failures),
            "runtime_recovery_segment_count": len(runtime_recovery_plans),
        }
    )
    write_artifact(result, artifact_path)
    if not validation["allclose"]:
        raise RuntimeError(f"cross-node NIXL validation failed: {validation}")
    return result


def _ucx_ib_rc_requested() -> bool:
    ucx_tls = os.environ.get("UCX_TLS", "").lower()
    return any(token in ucx_tls.split(",") for token in ("rc", "rc_x", "dc", "dc_x"))


def _ucx_env_snapshot() -> dict[str, str]:
    keys = [
        "MX_NIXL_BACKEND",
        "MX_RDMA_NIC_PIN",
        "MX_RDMA_NIC_PIN_MIN_RATE_GBPS",
        "UCX_TLS",
        "UCX_NET_DEVICES",
        "UCX_IB_DEVICES",
        "UCX_RNDV_SCHEME",
        "UCX_RNDV_THRESH",
        "UCX_MAX_RNDV_RAILS",
    ]
    return {key: os.environ.get(key, "") for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["source", "target"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--hold-seconds", type=float, default=300.0)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--source-id", default=os.environ.get("SOURCE_ID", ""))
    parser.add_argument("--source-worker-rank", type=int)
    parser.add_argument("--source-pod", default=os.environ.get("SOURCE_POD", ""))
    parser.add_argument("--source-node", default=os.environ.get("SOURCE_NODE", ""))
    parser.add_argument("--target-pod", default=os.environ.get("POD_NAME", ""))
    parser.add_argument("--target-node", default=os.environ.get("NODE_NAME", ""))
    parser.add_argument(
        "--gpu-count", type=int, default=int(os.environ.get("GPU_COUNT", "2"))
    )
    parser.add_argument(
        "--payload-columns",
        type=int,
        default=int(os.environ.get("MX_REFIT_PAYLOAD_COLUMNS", str(GLOBAL_SHAPE[1]))),
        help=(
            "Column count for the synthetic tensor payload. Values larger than "
            "the default widen each source and target slice so hard-kill NIXL "
            "proofs can keep reads in flight long enough for pod deletion."
        ),
    )
    parser.add_argument(
        "--failed-source-id",
        action="append",
        default=[],
        help="Additional source id to plan around without reading.",
    )
    parser.add_argument(
        "--stale-source-id",
        action="append",
        default=[],
        help="Source id expected to be STALE in MX before target reads.",
    )
    parser.add_argument(
        "--disable-default-failed-source",
        action="store_true",
        help=(
            "Attempt the default primary source instead of preplanning around it; "
            "useful for read-failure recovery proofs."
        ),
    )
    args = parser.parse_args()

    if args.role == "source":
        run_source(
            run_id=args.run_id,
            hold_seconds=args.hold_seconds,
            local_rank=args.local_rank,
            artifact_path=args.artifact,
            source_id=args.source_id,
            source_worker_rank=args.source_worker_rank,
            payload_columns=args.payload_columns,
        )
    else:
        if args.artifact is None:
            raise ValueError("--artifact is required for target role")
        run_target(
            run_id=args.run_id,
            local_rank=args.local_rank,
            artifact_path=args.artifact,
            timeout_seconds=args.timeout_seconds,
            source_pod=args.source_pod,
            source_node=args.source_node,
            target_pod=args.target_pod,
            target_node=args.target_node,
            gpu_count=args.gpu_count,
            failed_source_ids={
                *(
                    ()
                    if args.disable_default_failed_source
                    else FAILED_PRIMARY_SOURCE_IDS
                ),
                *{source_id for source_id in args.failed_source_id if source_id},
            },
            stale_source_ids={
                source_id for source_id in args.stale_source_id if source_id
            },
            payload_columns=args.payload_columns,
        )


if __name__ == "__main__":
    main()
