# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU POC for multi-source trainer-to-inference refit.

Run directly so it can be mounted into an existing GPU image without rebuilding:

    torchrun --standalone --nproc_per_node=4 refit_poc.py --mode distributed
    torchrun --standalone --nproc_per_node=4 refit_poc.py --mode nixl-distributed

The transfer mechanism is GPU tensor send/recv, device-to-device copy, or
one-sided NIXL READ driven by SegmentPlan offsets. It intentionally does not use
trainer all-gather or host-side torch.cat.
"""

from __future__ import annotations

import sys

_SCRIPT_DIR = __file__.rsplit("/", 1)[0]
if _SCRIPT_DIR.endswith("/modelexpress") and _SCRIPT_DIR in sys.path:
    sys.path.remove(_SCRIPT_DIR)
_PACKAGE_PARENT = _SCRIPT_DIR.rsplit("/", 1)[0] if "/" in _SCRIPT_DIR else "."
if _SCRIPT_DIR.endswith("/modelexpress") and _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

import argparse
import importlib.util
import os
from pathlib import Path
import time
from typing import Any


try:
    from .refit_nixl import (
        NixlAdapter,
        apply_nixl_ucx_pin,
        read_segment_groups,
        select_cuda_device,
    )
    from .refit_live_mx import (
        live_mx_endpoint_context as _live_mx_endpoint_context,
        live_mx_plan_context as _live_mx_plan_context,
        publish_live_mx_rank_endpoint as _publish_live_mx_rank_endpoint,
        publish_live_mx_rank_ownership as _publish_live_mx_rank_ownership,
        refit_source_identity,
    )
    from .refit_poc_artifacts import (
        artifact_base as _artifact_base,
        build_planner_artifacts,
        metadata_smokes,
        synthetic_plan_context as _synthetic_plan_context,
        write_artifact as _write_artifact,
    )
    from .refit_poc_scenario import (
        ALTERNATE_SOURCE_IDS,
        CONTROL_PLANE_LIVE_MX,
        CONTROL_PLANE_SYNTHETIC,
        FAILED_PRIMARY_SOURCE_IDS,
        MODEL_NAME,
        MODEL_VERSION,
        PRIMARY_SOURCE_IDS,
        REQUEST_RANGE,
        TENSOR_NAME,
        alternate_ownerships,
        inference_request,
        ownerships_by_rank,
        primary_ownerships,
    )
    from .resharding import (
        SegmentPlan,
        SliceRequest,
        range_extents,
    )
except ImportError:
    from modelexpress.refit_nixl import (
        NixlAdapter,
        apply_nixl_ucx_pin,
        read_segment_groups,
        select_cuda_device,
    )
    from modelexpress.refit_live_mx import (
        live_mx_endpoint_context as _live_mx_endpoint_context,
        live_mx_plan_context as _live_mx_plan_context,
        publish_live_mx_rank_endpoint as _publish_live_mx_rank_endpoint,
        publish_live_mx_rank_ownership as _publish_live_mx_rank_ownership,
        refit_source_identity,
    )
    from modelexpress.refit_poc_artifacts import (
        artifact_base as _artifact_base,
        build_planner_artifacts,
        metadata_smokes,
        synthetic_plan_context as _synthetic_plan_context,
        write_artifact as _write_artifact,
    )
    from modelexpress.refit_poc_scenario import (
        ALTERNATE_SOURCE_IDS,
        CONTROL_PLANE_LIVE_MX,
        CONTROL_PLANE_SYNTHETIC,
        FAILED_PRIMARY_SOURCE_IDS,
        MODEL_NAME,
        MODEL_VERSION,
        PRIMARY_SOURCE_IDS,
        REQUEST_RANGE,
        TENSOR_NAME,
        alternate_ownerships,
        inference_request,
        ownerships_by_rank,
        primary_ownerships,
    )
    from modelexpress.resharding import (
        SegmentPlan,
        SliceRequest,
        range_extents,
    )


def _import_torch():
    import torch

    return torch


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


def _target_expected(device):
    return _materialize_range(REQUEST_RANGE, device)


def _slice_for_plan(source_tensor, owner_range, plan: SegmentPlan):
    row_start = plan.source_range[0][0] - owner_range[0][0]
    row_end = plan.source_range[0][1] - owner_range[0][0]
    col_start = plan.source_range[1][0] - owner_range[1][0]
    col_end = plan.source_range[1][1] - owner_range[1][0]
    return source_tensor[row_start:row_end, col_start:col_end]


def _copy_to_target(target, request_range, plan: SegmentPlan, segment):
    row_start = plan.target_range[0][0] - request_range[0][0]
    row_end = plan.target_range[0][1] - request_range[0][0]
    col_start = plan.target_range[1][0] - request_range[1][0]
    col_end = plan.target_range[1][1] - request_range[1][0]
    target[row_start:row_end, col_start:col_end].copy_(segment)


def _checksum(tensor) -> float:
    torch = _import_torch()
    flat = tensor.float().reshape(-1)
    weights = torch.arange(1, flat.numel() + 1, device=tensor.device, dtype=torch.float32)
    return float((flat * weights).sum().item())


def _validate_target(target) -> dict[str, Any]:
    torch = _import_torch()
    expected = _target_expected(target.device)
    torch.cuda.synchronize(target.device)
    return {
        "allclose": bool(torch.allclose(target, expected)),
        "checksum": _checksum(target),
        "expected_checksum": _checksum(expected),
        "max_abs_error": float((target - expected).abs().max().item()),
    }


def run_single_gpu(artifact_path: Path) -> dict[str, Any]:
    torch = _import_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU refit POC")
    device = torch.device("cuda:0")
    request = inference_request()
    plan_context = _synthetic_plan_context(request)
    primary_plans = plan_context["primary_plans"]
    recovery_plans = plan_context["recovery_plans"]

    source_by_id = {
        owner.source_id: _materialize_range(owner.source_range, device)
        for owner in [
            *plan_context["primary_ownerships"],
            *plan_context["alternate_ownerships"],
        ]
    }
    owner_by_id = {
        owner.source_id: owner
        for owner in [
            *plan_context["primary_ownerships"],
            *plan_context["alternate_ownerships"],
        ]
    }
    target = torch.full(request.target_shape, float("nan"), device=device)

    torch.cuda.synchronize(device)
    start = time.perf_counter()
    failed_source_ids = {"trainer-rank0"}
    copied_bytes = 0
    for plan in primary_plans:
        if plan.source_id in failed_source_ids:
            continue
        owner = owner_by_id[plan.source_id]
        segment = _slice_for_plan(
            source_by_id[plan.source_id],
            owner.source_range,
            plan,
        ).contiguous()
        _copy_to_target(target, request.requested_range, plan, segment)
        copied_bytes += plan.bytes

    for plan in recovery_plans:
        owner = owner_by_id[plan.source_id]
        segment = _slice_for_plan(
            source_by_id[plan.source_id],
            owner.source_range,
            plan,
        ).contiguous()
        _copy_to_target(target, request.requested_range, plan, segment)
        copied_bytes += plan.bytes
    torch.cuda.synchronize(device)
    duration_ms = (time.perf_counter() - start) * 1000

    validation = _validate_target(target)
    result = _artifact_base(
        mode="single-gpu",
        gpu_count=1,
        copied_bytes=copied_bytes,
        copy_duration_ms=duration_ms,
        validation=validation,
        request=request,
        plan_context=plan_context,
    )
    _write_artifact(result, artifact_path)
    if not validation["allclose"]:
        raise RuntimeError(f"single-GPU validation failed: {validation}")
    return result


def run_distributed(artifact_path: Path) -> dict[str, Any] | None:
    torch = _import_torch()
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the distributed refit POC")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 4:
        raise RuntimeError("distributed refit POC requires at least 4 ranks")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")

    request = inference_request()
    plan_context = _synthetic_plan_context(request)
    primary_plans = plan_context["primary_plans"]
    recovery_plans = plan_context["recovery_plans"]
    owner_by_rank = {
        owner.worker_rank: owner
        for owner in [
            *plan_context["primary_ownerships"],
            *plan_context["alternate_ownerships"],
        ]
    }

    dist.barrier()
    if rank in owner_by_rank:
        owner = owner_by_rank[rank]
        source_tensor = _materialize_range(owner.source_range, device)
    else:
        source_tensor = None

    target = None
    copied_bytes = 0
    recv_duration_ms = 0.0
    validation: dict[str, Any] | None = None

    if rank == 3:
        target = torch.full(request.target_shape, float("nan"), device=device)
        start = time.perf_counter()

    # Primary phase intentionally treats rank 0 as failed, so only rank 1 sends.
    if rank == 1 and source_tensor is not None:
        owner = owner_by_rank[rank]
        for plan in primary_plans:
            if plan.source_id != owner.source_id:
                continue
            dist.send(
                _slice_for_plan(source_tensor, owner.source_range, plan).contiguous(),
                dst=3,
            )
    elif rank == 3 and target is not None:
        for plan in primary_plans:
            if plan.source_id == "trainer-rank0":
                continue
            shape = range_extents(plan.target_range)
            recv_buf = torch.empty(shape, device=device, dtype=torch.float32)
            dist.recv(recv_buf, src=1)
            _copy_to_target(target, request.requested_range, plan, recv_buf)
            copied_bytes += plan.bytes

    dist.barrier()

    # Recovery phase replans only the failed segment from alternate rank 2.
    if rank == 2 and source_tensor is not None:
        owner = owner_by_rank[rank]
        for plan in recovery_plans:
            dist.send(
                _slice_for_plan(source_tensor, owner.source_range, plan).contiguous(),
                dst=3,
            )
    elif rank == 3 and target is not None:
        for plan in recovery_plans:
            shape = range_extents(plan.target_range)
            recv_buf = torch.empty(shape, device=device, dtype=torch.float32)
            dist.recv(recv_buf, src=2)
            _copy_to_target(target, request.requested_range, plan, recv_buf)
            copied_bytes += plan.bytes
        torch.cuda.synchronize(device)
        recv_duration_ms = (time.perf_counter() - start) * 1000
        validation = _validate_target(target)

    dist.barrier()

    result = None
    if rank == 3:
        assert validation is not None
        result = _artifact_base(
            mode="distributed-4rank",
            gpu_count=world_size,
            copied_bytes=copied_bytes,
            copy_duration_ms=recv_duration_ms,
            validation=validation,
            request=request,
            plan_context=plan_context,
        )
        result["distributed"] = {
            "backend": "nccl",
            "world_size": world_size,
            "target_rank": 3,
            "primary_source_ranks": [0, 1],
            "alternate_source_ranks": [2],
        }
        _write_artifact(result, artifact_path)
        if not validation["allclose"]:
            raise RuntimeError(f"distributed validation failed: {validation}")

    dist.destroy_process_group()
    return result


def run_nixl_distributed(
    artifact_path: Path,
    *,
    control_plane: str = CONTROL_PLANE_SYNTHETIC,
) -> dict[str, Any] | None:
    torch = _import_torch()
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the NIXL refit POC")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 4:
        raise RuntimeError("NIXL distributed refit POC requires exactly 4 ranks")

    device_index, gpu_reuse_used = select_cuda_device(local_rank)
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    dist.init_process_group(backend="gloo")
    apply_nixl_ucx_pin(device_index)

    result = None
    try:
        request = inference_request()
        owner_by_rank = ownerships_by_rank()
        publish_context: dict[str, Any] = {
            "mode": control_plane,
            "published": False,
        }
        if control_plane == CONTROL_PLANE_LIVE_MX:
            plan_context = {
                "mode": CONTROL_PLANE_LIVE_MX,
                "plan_source": "live-mx-server",
                "primary_plans": [],
                "recovery_plans": [],
                "primary_ownerships": [],
                "alternate_ownerships": [],
                "discovered_ownerships": [],
                "source_endpoints_by_id": {},
                "metadata_query_duration_ms": 0.0,
                "planner_duration_ms": 0.0,
            }
        else:
            plan_context = _synthetic_plan_context(request)

        agent_name = f"mx-refit-rank{rank}"
        adapter = NixlAdapter(agent_name)
        backends = adapter.backends

        source_tensor = None
        target = None
        registered_bytes = 0
        register_start = time.perf_counter()
        if rank in owner_by_rank:
            owner = owner_by_rank[rank]
            source_tensor = _materialize_range(owner.source_range, device).contiguous()
            adapter.register_tensor(source_tensor)
            registered_bytes = source_tensor.numel() * source_tensor.element_size()
        elif rank == 3:
            target = torch.full(request.target_shape, float("nan"), device=device)
            adapter.register_tensor(target)
            registered_bytes = target.numel() * target.element_size()
        torch.cuda.synchronize(device)
        registration_duration_ms = (time.perf_counter() - register_start) * 1000

        metadata_start = time.perf_counter()
        metadata = adapter.metadata_bytes()
        metadata_duration_ms = (time.perf_counter() - metadata_start) * 1000

        local_info: dict[str, Any] = {
            "rank": rank,
            "agent_name": agent_name,
            "cuda_device": device_index,
            "gpu_reuse_used": gpu_reuse_used,
            "metadata": metadata,
            "metadata_bytes": len(metadata),
            "registered_bytes": registered_bytes,
            "registration_duration_ms": registration_duration_ms,
            "metadata_duration_ms": metadata_duration_ms,
            "config_errors": adapter.config_errors,
            "role": (
                "source"
                if rank in owner_by_rank
                else "target"
                if rank == 3
                else "unused"
            ),
            "control_plane": publish_context,
        }
        if control_plane != CONTROL_PLANE_LIVE_MX:
            local_info["metadata"] = metadata
        if rank in owner_by_rank:
            assert source_tensor is not None
            owner = owner_by_rank[rank]
            local_info.update(
                {
                    "source_id": owner.source_id,
                    "worker_id": owner.worker_id,
                    "source_range": owner.source_range,
                    "addr": int(source_tensor.data_ptr()),
                    "device_id": int(source_tensor.get_device()),
                    "tensor_bytes": int(
                        source_tensor.numel() * source_tensor.element_size()
                    ),
                }
            )
        elif rank == 3:
            assert target is not None
            local_info.update(
                {
                    "target_id": request.target_id,
                    "addr": int(target.data_ptr()),
                    "device_id": int(target.get_device()),
                    "tensor_bytes": int(target.numel() * target.element_size()),
                }
            )

        if control_plane == CONTROL_PLANE_LIVE_MX:
            if rank in owner_by_rank:
                publish_context = _publish_live_mx_rank_endpoint(
                    rank,
                    agent_name=agent_name,
                    nixl_metadata=metadata,
                    tensor_addr=int(local_info["addr"]),
                    tensor_bytes=int(local_info["tensor_bytes"]),
                    device_id=int(local_info["device_id"]),
                )
                local_info["control_plane"] = publish_context
            dist.barrier()
            if rank == 3:
                plan_context = _live_mx_endpoint_context(request)
            else:
                plan_context["endpoint_source"] = "mx-worker-metadata"

        primary_plans = plan_context["primary_plans"]
        recovery_plans = plan_context["recovery_plans"]
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_info)

        if rank != 3:
            dist.barrier()

        if rank == 3:
            assert target is not None
            if not primary_plans or not recovery_plans:
                raise RuntimeError("target rank did not receive segment plans")
            endpoint_by_source_id = plan_context.get("source_endpoints_by_id", {})
            if endpoint_by_source_id:
                sources_by_id = {
                    source_id: endpoint.to_nixl_source_info()
                    for source_id, endpoint in endpoint_by_source_id.items()
                }
            else:
                sources_by_id = {
                    info["source_id"]: info
                    for info in gathered
                    if info is not None and info.get("role") == "source"
                }

            add_remote_timings: dict[str, float] = {}
            remote_agent_names: dict[str, str] = {}
            for source_id, info in sorted(sources_by_id.items()):
                add_start = time.perf_counter()
                if endpoint_by_source_id:
                    remote_metadata = endpoint_by_source_id[source_id].nixl_metadata
                else:
                    remote_metadata = info["metadata"]
                remote_agent_names[source_id] = adapter.add_remote_agent(remote_metadata)
                add_remote_timings[source_id] = (time.perf_counter() - add_start) * 1000

            transfer_start = time.perf_counter()
            primary_read_plans = [
                plan
                for plan in primary_plans
                if plan.source_id not in FAILED_PRIMARY_SOURCE_IDS
            ]
            primary_reads = read_segment_groups(
                adapter=adapter,
                target=target,
                sources_by_id=sources_by_id,
                remote_agent_names=remote_agent_names,
                plans=primary_read_plans,
                timeout_seconds=120,
            )
            recovery_reads = read_segment_groups(
                adapter=adapter,
                target=target,
                sources_by_id=sources_by_id,
                remote_agent_names=remote_agent_names,
                plans=recovery_plans,
                timeout_seconds=120,
            )
            torch.cuda.synchronize(device)
            total_read_duration_ms = (time.perf_counter() - transfer_start) * 1000
            validation = _validate_target(target)
            copied_bytes = sum(read["bytes"] for read in [*primary_reads, *recovery_reads])

            result = _artifact_base(
                mode="nixl-distributed-4rank",
                gpu_count=torch.cuda.device_count(),
                copied_bytes=copied_bytes,
                copy_duration_ms=total_read_duration_ms,
                validation=validation,
                request=request,
                plan_context=plan_context,
            )
            result["proof"].update(
                {
                    "actual_nixl_reads_used": True,
                    "torch_distributed_data_transfer_used": False,
                    "torch_distributed_control_plane_used": True,
                    "torch_distributed_nixl_metadata_exchange_used": not bool(
                        endpoint_by_source_id
                    ),
                    "nixl_source_endpoints_from_mx": bool(endpoint_by_source_id),
                    "live_mx_returned_metadata_used_for_nixl_plan": (
                        control_plane == CONTROL_PLANE_LIVE_MX
                    ),
                    "target_buffer_preallocated": True,
                    "nixl_reads_land_at_segment_offsets": validation["allclose"],
                }
            )
            result["distributed"] = {
                "backend": (
                    "nixl-read+mx-endpoint-control+gloo-sync"
                    if endpoint_by_source_id
                    else "nixl-read+gloo-control"
                ),
                "nixl_backends": backends,
                "world_size": world_size,
                "target_rank": 3,
                "primary_source_ranks": [0, 1],
                "alternate_source_ranks": [2],
                "failed_primary_source_ranks": [0],
                "rank_to_cuda_device": {
                    info["rank"]: info.get("cuda_device")
                    for info in gathered
                    if info is not None
                },
                "gpu_reuse_used": any(
                    bool(info.get("gpu_reuse_used"))
                    for info in gathered
                    if info is not None
                ),
            }
            result["control_plane"] = _control_plane_artifact(
                control_plane=control_plane,
                plan_context=plan_context,
                gathered=gathered,
            )
            result["nixl"] = {
                "target_agent_name": agent_name,
                "source_metadata": {
                    source_id: {
                        "agent_name": info["agent_name"],
                        "rank": info["rank"],
                        "metadata_bytes": info["metadata_bytes"],
                        "nixl_descriptor_identity": source_id,
                        "source_range": info["source_range"],
                        "registered_bytes": info["registered_bytes"],
                        "registration_duration_ms": info["registration_duration_ms"],
                        "metadata_duration_ms": info["metadata_duration_ms"],
                    }
                    for source_id, info in sorted(sources_by_id.items())
                },
                "add_remote_agent_duration_ms": add_remote_timings,
                "primary_reads": primary_reads,
                "recovery_reads": recovery_reads,
            }
            result["metrics"].update(
                {
                    "gpu_copy_duration_ms": 0.0,
                    "raw_nixl_read_duration_ms": total_read_duration_ms,
                    "nixl_registration_duration_ms": sum(
                        info["registration_duration_ms"]
                        for info in gathered
                        if info is not None
                    ),
                    "nixl_metadata_duration_ms": sum(
                        info["metadata_duration_ms"]
                        for info in gathered
                        if info is not None
                    ),
                    "nixl_add_remote_agent_duration_ms": sum(
                        add_remote_timings.values()
                    ),
                    "nixl_prep_duration_ms": sum(
                        read["prep_duration_ms"]
                        for read in [*primary_reads, *recovery_reads]
                    ),
                    "nixl_read_group_count": len(primary_reads) + len(recovery_reads),
                    "successful_nixl_source_count": len(
                        {read["source_id"] for read in [*primary_reads, *recovery_reads]}
                    ),
                    "publish_duration_ms": sum(
                        _nested_float(info, "control_plane", "publish_duration_ms")
                        for info in gathered
                        if info is not None
                    ),
                    "metadata_query_duration_ms": plan_context.get(
                        "metadata_query_duration_ms",
                        0.0,
                    ),
                    "planner_duration_ms": plan_context.get(
                        "planner_duration_ms",
                        0.0,
                    ),
                }
            )
            _write_artifact(result, artifact_path)
            if not validation["allclose"]:
                raise RuntimeError(f"NIXL distributed validation failed: {validation}")
            dist.barrier()

    finally:
        dist.destroy_process_group()

    return result


def _control_plane_artifact(
    *,
    control_plane: str,
    plan_context: dict[str, Any],
    gathered: list[dict[str, Any] | None],
) -> dict[str, Any]:
    source_publications = [
        info["control_plane"]
        for info in gathered
        if info is not None and info.get("control_plane", {}).get("published")
    ]
    return {
        "mode": control_plane,
        "plan_source": plan_context.get("plan_source", "unknown"),
        "server_url": plan_context.get("server_url", ""),
        "source_publications": source_publications,
        "discovered_ownerships": [
            owner.to_dict() for owner in plan_context.get("discovered_ownerships", [])
        ],
        "primary_ownerships_from_control_plane": [
            owner.to_dict() for owner in plan_context.get("primary_ownerships", [])
        ],
        "alternate_ownerships_from_control_plane": [
            owner.to_dict() for owner in plan_context.get("alternate_ownerships", [])
        ],
        "source_endpoints_from_control_plane": [
            endpoint.to_dict()
            for endpoint in plan_context.get("source_endpoints_by_id", {}).values()
        ],
        "primary_segment_plans": [
            plan.to_dict() for plan in plan_context.get("primary_plans", [])
        ],
        "recovery_segment_plans": [
            plan.to_dict() for plan in plan_context.get("recovery_plans", [])
        ],
        "metadata_query_duration_ms": plan_context.get(
            "metadata_query_duration_ms",
            0.0,
        ),
        "planner_duration_ms": plan_context.get("planner_duration_ms", 0.0),
    }


def _nested_float(data: dict[str, Any], *keys: str) -> float:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return 0.0
        current = current.get(key)
    if isinstance(current, (int, float)):
        return float(current)
    return 0.0


def run_planner_only(artifact_path: Path) -> dict[str, Any]:
    request = inference_request()
    plan_context = _synthetic_plan_context(request)
    planner_artifacts = build_planner_artifacts(
        request=request,
        plan_context=plan_context,
    )
    validation = {
        "allclose": True,
        "checksum": None,
        "expected_checksum": None,
        "max_abs_error": 0.0,
    }
    result = _artifact_base(
        mode="planner-only",
        gpu_count=0,
        copied_bytes=sum(
            plan["bytes"] for plan in planner_artifacts["primary_segment_plans"]
        ),
        copy_duration_ms=0.0,
        validation=validation,
        request=request,
        plan_context=plan_context,
    )
    _write_artifact(result, artifact_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["planner-only", "single-gpu", "distributed", "nixl-distributed"],
        default="planner-only",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("/tmp/mx_refit_artifact.json"),
    )
    parser.add_argument(
        "--control-plane",
        choices=[CONTROL_PLANE_SYNTHETIC, CONTROL_PLANE_LIVE_MX],
        default=os.environ.get("MX_REFIT_CONTROL_PLANE", CONTROL_PLANE_SYNTHETIC),
        help=(
            "Planner metadata source for nixl-distributed. "
            "'live-mx' publishes slice ownerships through the MX server and "
            "plans from returned READY metadata."
        ),
    )
    args = parser.parse_args()

    if args.mode == "planner-only":
        run_planner_only(args.artifact)
    elif args.mode == "single-gpu":
        run_single_gpu(args.artifact)
    elif args.mode == "nixl-distributed":
        run_nixl_distributed(args.artifact, control_plane=args.control_plane)
    else:
        run_distributed(args.artifact)


if __name__ == "__main__":
    main()
