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
from typing import Any

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


def _run_ownerships(run_id: str) -> list[SliceOwnership]:
    model_version = _run_model_version(run_id)
    return [
        replace(owner, model_version=model_version)
        for owner in [*primary_ownerships(), *alternate_ownerships()]
    ]


def _run_request(run_id: str) -> SliceRequest:
    return replace(
        inference_request(target_id="inference-crossnode-target"),
        model_version=_run_model_version(run_id),
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
    weights = torch.arange(1, flat.numel() + 1, device=tensor.device, dtype=torch.float32)
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
) -> dict[str, Any]:
    endpoints_by_source_id = {
        endpoint.source_id: endpoint
        for endpoint in sorted(endpoints, key=lambda item: (item.source_id, item.worker_id))
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
        plan for plan in primary_plans if plan.source_id in FAILED_PRIMARY_SOURCE_IDS
    ]
    recovery_requests = [
        replace(request, requested_range=plan.target_range, target_shape=_range_extents(plan.target_range))
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
        "metadata_query_duration_ms": metadata_query_duration_ms,
        "planner_duration_ms": planner_duration_ms,
    }


def _range_extents(tensor_range):
    return tuple(int(end) - int(start) for start, end in tensor_range)


def _wait_for_endpoints(
    *,
    run_id: str,
    request: SliceRequest,
    expected_source_ids: set[str],
    timeout_seconds: float,
) -> dict[str, Any]:
    mx_client = MxClient()
    try:
        identity = _run_identity(run_id)
        deadline = time.time() + timeout_seconds
        metadata_query_start = time.perf_counter()
        endpoints = []
        while True:
            endpoints = list_refit_nixl_endpoints(mx_client, identity=identity)
            present = {endpoint.source_id for endpoint in endpoints}
            if expected_source_ids.issubset(present):
                break
            if time.time() >= deadline:
                missing = sorted(expected_source_ids - present)
                raise RuntimeError(
                    "timed out waiting for cross-node MX NIXL endpoints "
                    f"(run_id={run_id}, missing={missing}, present={sorted(present)})"
                )
            time.sleep(0.25)
        metadata_query_duration_ms = (
            time.perf_counter() - metadata_query_start
        ) * 1000

        planner_start = time.perf_counter()
        plan_context = _plan_context_for_endpoints(
            endpoints=endpoints,
            request=request,
            metadata_query_duration_ms=metadata_query_duration_ms,
            planner_duration_ms=0.0,
        )
        plan_context["planner_duration_ms"] = (time.perf_counter() - planner_start) * 1000
        plan_context["server_url"] = getattr(mx_client, "server_url", "")
        return plan_context
    finally:
        mx_client.close()


def run_source(
    *,
    run_id: str,
    hold_seconds: float,
    local_rank: int,
    artifact_path: Path | None,
) -> dict[str, Any]:
    torch = _import_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cross-node source publishing")

    device_index, gpu_reuse_used = select_cuda_device(local_rank)
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    apply_nixl_ucx_pin(device_index)

    agent_name = f"mx-crossnode-source-{run_id}"
    adapter = NixlAdapter(agent_name)
    ownerships = _run_ownerships(run_id)
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
) -> dict[str, Any]:
    torch = _import_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cross-node target reads")

    device_index, gpu_reuse_used = select_cuda_device(local_rank)
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    apply_nixl_ucx_pin(device_index)

    request = _run_request(run_id)
    expected_source_ids = {
        *PRIMARY_SOURCE_IDS,
        *{owner.source_id for owner in alternate_ownerships()},
    }
    _trace(
        "target.wait_for_mx_endpoints.start",
        run_id=run_id,
        expected_source_ids=sorted(expected_source_ids),
    )
    plan_context = _wait_for_endpoints(
        run_id=run_id,
        request=request,
        expected_source_ids=expected_source_ids,
        timeout_seconds=timeout_seconds,
    )
    _trace(
        "target.wait_for_mx_endpoints.done",
        run_id=run_id,
        source_ids=sorted(plan_context["source_endpoints_by_id"]),
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
    sources_by_id = {
        source_id: endpoint.to_nixl_source_info()
        for source_id, endpoint in endpoint_by_source_id.items()
    }

    remote_agent_cache: dict[str, str] = {}
    add_remote_timings: dict[str, float] = {}
    remote_agent_names: dict[str, str] = {}
    for source_id, endpoint in sorted(endpoint_by_source_id.items()):
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
        if plan.source_id not in FAILED_PRIMARY_SOURCE_IDS
    ]
    _trace(
        "target.read_segments.start",
        primary_segment_count=len(primary_read_plans),
        recovery_segment_count=len(plan_context["recovery_plans"]),
    )
    transfer_start = time.perf_counter()
    primary_reads = read_segment_groups(
        adapter=adapter,
        target=target,
        sources_by_id=sources_by_id,
        remote_agent_names=remote_agent_names,
        plans=primary_read_plans,
        timeout_seconds=timeout_seconds,
    )
    recovery_reads = read_segment_groups(
        adapter=adapter,
        target=target,
        sources_by_id=sources_by_id,
        remote_agent_names=remote_agent_names,
        plans=plan_context["recovery_plans"],
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
    copied_bytes = sum(read["bytes"] for read in [*primary_reads, *recovery_reads])

    result = artifact_base(
        mode="nixl-crossnode-2pod",
        gpu_count=2,
        copied_bytes=copied_bytes,
        copy_duration_ms=raw_nixl_read_duration_ms,
        validation=validation,
        request=request,
        plan_context=plan_context,
    )
    result["proof"].update(
        {
            "actual_nixl_reads_used": True,
            "cross_node_pods": bool(source_node and target_node and source_node != target_node),
            "source_pod_separate_from_target_pod": bool(source_pod and target_pod and source_pod != target_pod),
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
        }
    )
    result["distributed"] = {
        "backend": "nixl-read+mx-endpoint-control+no-torch-distributed",
        "world_size": 2,
        "source_pod": source_pod,
        "target_pod": target_pod,
        "source_node": source_node,
        "target_node": target_node,
        "cross_node": bool(source_node and target_node and source_node != target_node),
        "source_roles_in_source_pod": list(sources_by_id),
        "target_agent_name": agent_name,
        "gpu_reuse_used": gpu_reuse_used,
    }
    result["control_plane"] = {
        "mode": "live-mx-cross-node",
        "server_url": plan_context.get("server_url", ""),
        "source_endpoints_from_control_plane": [
            endpoint.to_dict() for endpoint in endpoint_by_source_id.values()
        ],
        "metadata_query_duration_ms": plan_context["metadata_query_duration_ms"],
        "planner_duration_ms": plan_context["planner_duration_ms"],
    }
    result["nixl"] = {
        "target_agent_name": agent_name,
        "nixl_backends": adapter.backends,
        "ucx": _ucx_env_snapshot(),
        "source_metadata": {
            source_id: {
                "agent_name": endpoint_by_source_id[source_id].agent_name,
                "metadata_bytes": len(endpoint_by_source_id[source_id].nixl_metadata),
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
    }
    result["metrics"].update(
        {
            "gpu_copy_duration_ms": 0.0,
            "raw_nixl_read_duration_ms": raw_nixl_read_duration_ms,
            "nixl_registration_duration_ms": registration_duration_ms,
            "nixl_add_remote_agent_duration_ms": sum(add_remote_timings.values()),
            "nixl_prep_duration_ms": sum(
                read["prep_duration_ms"] for read in [*primary_reads, *recovery_reads]
            ),
            "nixl_read_group_count": len(primary_reads) + len(recovery_reads),
            "successful_nixl_source_count": len(
                {read["source_id"] for read in [*primary_reads, *recovery_reads]}
            ),
            "metadata_query_duration_ms": plan_context["metadata_query_duration_ms"],
            "planner_duration_ms": plan_context["planner_duration_ms"],
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
    parser.add_argument("--source-pod", default=os.environ.get("SOURCE_POD", ""))
    parser.add_argument("--source-node", default=os.environ.get("SOURCE_NODE", ""))
    parser.add_argument("--target-pod", default=os.environ.get("POD_NAME", ""))
    parser.add_argument("--target-node", default=os.environ.get("NODE_NAME", ""))
    args = parser.parse_args()

    if args.role == "source":
        run_source(
            run_id=args.run_id,
            hold_seconds=args.hold_seconds,
            local_rank=args.local_rank,
            artifact_path=args.artifact,
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
        )


if __name__ == "__main__":
    main()
