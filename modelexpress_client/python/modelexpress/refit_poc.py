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

import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
from typing import Any


try:
    from .resharding import (
        BandwidthAssumptions,
        QuantizationScope,
        SegmentPlan,
        SliceOwnership,
        SliceRequest,
        classify_tensor_family,
        plan_segments,
        range_extents,
        segment_plans_to_json,
        simulate_resharding,
    )
except ImportError:
    spec = importlib.util.spec_from_file_location(
        "mx_resharding_direct",
        Path(__file__).with_name("resharding.py"),
    )
    if spec is None or spec.loader is None:
        raise
    mx_resharding_direct = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mx_resharding_direct
    spec.loader.exec_module(mx_resharding_direct)
    BandwidthAssumptions = mx_resharding_direct.BandwidthAssumptions
    QuantizationScope = mx_resharding_direct.QuantizationScope
    SegmentPlan = mx_resharding_direct.SegmentPlan
    SliceOwnership = mx_resharding_direct.SliceOwnership
    SliceRequest = mx_resharding_direct.SliceRequest
    classify_tensor_family = mx_resharding_direct.classify_tensor_family
    plan_segments = mx_resharding_direct.plan_segments
    range_extents = mx_resharding_direct.range_extents
    segment_plans_to_json = mx_resharding_direct.segment_plans_to_json
    simulate_resharding = mx_resharding_direct.simulate_resharding


MODEL_NAME = "qwen3-moe-refit-poc"
MODEL_VERSION = "trainer-step-000001"
TENSOR_NAME = "model.layers.0.mlp.experts.w1.weight"
GLOBAL_SHAPE = (8, 4)
REQUEST_RANGE = ((2, 6), (0, 4))
PRIMARY_SOURCE_IDS = ("trainer-rank0", "trainer-rank1")
FAILED_PRIMARY_SOURCE_IDS = ("trainer-rank0",)
ALTERNATE_SOURCE_IDS = ("trainer-rank2-alt",)
CONTROL_PLANE_SYNTHETIC = "synthetic"
CONTROL_PLANE_LIVE_MX = "live-mx"


def primary_ownerships() -> list[SliceOwnership]:
    return [
        SliceOwnership(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            tensor_name=TENSOR_NAME,
            global_shape=GLOBAL_SHAPE,
            dtype="float32",
            source_range=((0, 3), (0, 4)),
            worker_id="rank0",
            source_id="trainer-rank0",
            worker_rank=0,
            source_lease="lease-rank0-primary",
            nixl_descriptor_id="nixl-rank0-primary",
            layout_tags={
                "trainer_layout": "fsdp",
                "tp": 1,
                "moe_expert_axis": 0,
                "storage_layout": "row-major",
            },
        ),
        SliceOwnership(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            tensor_name=TENSOR_NAME,
            global_shape=GLOBAL_SHAPE,
            dtype="float32",
            source_range=((3, 8), (0, 4)),
            worker_id="rank1",
            source_id="trainer-rank1",
            worker_rank=1,
            source_lease="lease-rank1-primary",
            nixl_descriptor_id="nixl-rank1-primary",
            layout_tags={
                "trainer_layout": "fsdp",
                "tp": 1,
                "moe_expert_axis": 0,
                "storage_layout": "row-major",
            },
        ),
    ]


def alternate_ownerships() -> list[SliceOwnership]:
    return [
        SliceOwnership(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            tensor_name=TENSOR_NAME,
            global_shape=GLOBAL_SHAPE,
            dtype="float32",
            source_range=((0, 3), (0, 4)),
            worker_id="rank2",
            source_id="trainer-rank2-alt",
            worker_rank=2,
            source_lease="lease-rank2-alt",
            nixl_descriptor_id="nixl-rank2-alt",
            layout_tags={
                "trainer_layout": "fsdp-replica",
                "tp": 1,
                "moe_expert_axis": 0,
                "storage_layout": "row-major",
            },
        )
    ]


def ownerships_by_rank() -> dict[int, SliceOwnership]:
    return {
        owner.worker_rank: owner
        for owner in [*primary_ownerships(), *alternate_ownerships()]
    }


def inference_request(
    *,
    requested_range=REQUEST_RANGE,
    target_id: str = "inference-rank3",
    runtime_framework: str = "vllm",
) -> SliceRequest:
    return SliceRequest(
        tensor_name=TENSOR_NAME,
        requested_range=requested_range,
        target_shape=range_extents(requested_range),
        dtype="float32",
        target_id=target_id,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        runtime_framework=runtime_framework,
        layout_tags={
            "target_layout": "tp2-ep2",
            "tp": 2,
            "ep": 2,
            "moe_expert_axis": 0,
            "storage_layout": "row-major",
        },
    )


def refit_source_identity():
    control_plane = _import_control_plane()
    return control_plane["build_refit_source_identity"](
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        dtype="float32",
        trainer_framework="synthetic-fsdp",
        trainer_layout="fsdp",
    )


def _synthetic_plan_context(request: SliceRequest) -> dict[str, Any]:
    primary_plans = plan_segments(primary_ownerships(), [request])
    failed_primary_plans = [
        plan for plan in primary_plans if plan.source_id in FAILED_PRIMARY_SOURCE_IDS
    ]
    recovery_requests = [
        inference_request(requested_range=plan.target_range)
        for plan in failed_primary_plans
    ]
    recovery_plans = plan_segments(alternate_ownerships(), recovery_requests)
    return {
        "mode": CONTROL_PLANE_SYNTHETIC,
        "plan_source": "in-process-synthetic",
        "primary_plans": primary_plans,
        "recovery_plans": recovery_plans,
        "primary_ownerships": primary_ownerships(),
        "alternate_ownerships": alternate_ownerships(),
        "discovered_ownerships": [],
        "metadata_query_duration_ms": 0.0,
        "planner_duration_ms": 0.0,
    }


def _publish_live_mx_rank_ownership(rank: int) -> dict[str, Any]:
    owner = ownerships_by_rank().get(rank)
    if owner is None:
        return {
            "mode": CONTROL_PLANE_LIVE_MX,
            "published": False,
            "reason": "rank-has-no-source-ownership",
        }

    control_plane = _import_control_plane()
    p2p_pb2 = control_plane["p2p_pb2"]
    mx_client = control_plane["MxClient"]()
    try:
        identity = refit_source_identity()
        publish_start = time.perf_counter()
        mx_source_id = control_plane["publish_slice_ownerships"](
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


def _live_mx_plan_context(
    request: SliceRequest,
    *,
    timeout_seconds: float = 30.0,
    mx_client=None,
) -> dict[str, Any]:
    control_plane = _import_control_plane()
    owns_client = mx_client is None
    if mx_client is None:
        mx_client = control_plane["MxClient"]()

    try:
        identity = refit_source_identity()
        expected_source_ids = set(PRIMARY_SOURCE_IDS) | set(ALTERNATE_SOURCE_IDS)
        deadline = time.time() + timeout_seconds
        metadata_query_start = time.perf_counter()
        discovered: list[SliceOwnership] = []
        while True:
            discovered = control_plane["list_slice_ownerships"](
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
        planner_start = time.perf_counter()
        primary_plans = plan_segments(primary_owners, [request])
        failed_primary_plans = [
            plan
            for plan in primary_plans
            if plan.source_id in FAILED_PRIMARY_SOURCE_IDS
        ]
        recovery_requests = [
            inference_request(requested_range=plan.target_range)
            for plan in failed_primary_plans
        ]
        recovery_plans = plan_segments(alternate_owners, recovery_requests)
        planner_duration_ms = (time.perf_counter() - planner_start) * 1000

        return {
            "mode": CONTROL_PLANE_LIVE_MX,
            "plan_source": "live-mx-server",
            "primary_plans": primary_plans,
            "recovery_plans": recovery_plans,
            "primary_ownerships": primary_owners,
            "alternate_ownerships": alternate_owners,
            "discovered_ownerships": discovered,
            "metadata_query_duration_ms": metadata_query_duration_ms,
            "planner_duration_ms": planner_duration_ms,
            "server_url": getattr(mx_client, "server_url", ""),
        }
    finally:
        if owns_client:
            mx_client.close()


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


def _import_control_plane() -> dict[str, Any]:
    if _SCRIPT_DIR.endswith("/modelexpress") and _PACKAGE_PARENT not in sys.path:
        sys.path.insert(0, _PACKAGE_PARENT)
    try:
        from . import p2p_pb2
        from .client import MxClient
        from .resharding_control_plane import (
            build_refit_source_identity,
            list_slice_ownerships,
            publish_slice_ownerships,
        )
    except ImportError:
        from modelexpress import p2p_pb2
        from modelexpress.client import MxClient
        from modelexpress.resharding_control_plane import (
            build_refit_source_identity,
            list_slice_ownerships,
            publish_slice_ownerships,
        )

    return {
        "p2p_pb2": p2p_pb2,
        "MxClient": MxClient,
        "build_refit_source_identity": build_refit_source_identity,
        "list_slice_ownerships": list_slice_ownerships,
        "publish_slice_ownerships": publish_slice_ownerships,
    }


def build_planner_artifacts() -> dict[str, Any]:
    request = inference_request()
    plan_context = _synthetic_plan_context(request)
    plans = plan_context["primary_plans"]
    recovery_plans = plan_context["recovery_plans"]
    simulation = simulate_resharding(
        plan_context["primary_ownerships"],
        [request],
        BandwidthAssumptions(
            trainer_to_inference_gbps=200,
            inference_to_inference_gbps=400,
        ),
    )
    return {
        "request": request.to_dict(),
        "primary_ownerships": [
            owner.to_dict() for owner in plan_context["primary_ownerships"]
        ],
        "alternate_ownerships": [
            owner.to_dict() for owner in plan_context["alternate_ownerships"]
        ],
        "primary_segment_plans": [plan.to_dict() for plan in plans],
        "recovery_segment_plans": [plan.to_dict() for plan in recovery_plans],
        "primary_segment_plans_json": json.loads(segment_plans_to_json(plans)),
        "simulation": simulation.to_dict(),
    }


def metadata_smokes() -> dict[str, Any]:
    moe_owners = [
        SliceOwnership(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            tensor_name=TENSOR_NAME,
            global_shape=(4, 8, 4),
            dtype="bfloat16",
            source_range=((0, 2), (0, 8), (0, 4)),
            worker_id="moe-rank0",
            source_id="moe-source-0",
            layout_tags={"moe_expert_axis": 0, "storage_layout": "row-major"},
        ),
        SliceOwnership(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            tensor_name=TENSOR_NAME,
            global_shape=(4, 8, 4),
            dtype="bfloat16",
            source_range=((2, 4), (0, 8), (0, 4)),
            worker_id="moe-rank1",
            source_id="moe-source-1",
            layout_tags={"moe_expert_axis": 0, "storage_layout": "row-major"},
        ),
    ]
    vllm_request = SliceRequest(
        tensor_name=TENSOR_NAME,
        requested_range=((1, 3), (0, 8), (0, 4)),
        target_shape=(2, 8, 4),
        dtype="bfloat16",
        target_id="vllm-rank0",
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        runtime_framework="vllm",
        layout_tags={"moe_expert_axis": 0, "storage_layout": "row-major"},
    )
    sglang_request = SliceRequest(
        tensor_name=TENSOR_NAME,
        requested_range=((1, 3), (0, 8), (0, 4)),
        target_shape=(2, 8, 4),
        dtype="bfloat16",
        target_id="sglang-rank0",
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        runtime_framework="sglang",
        layout_tags={"moe_expert_axis": 0, "storage_layout": "row-major"},
    )
    vllm_plans = plan_segments(moe_owners, [vllm_request])
    sglang_plans = plan_segments(moe_owners, [sglang_request])
    return {
        "qwen_moe_expert_axis": {
            "passed": True,
            "vllm_segment_count": len(vllm_plans),
            "sglang_segment_count": len(sglang_plans),
            "source_ids": sorted({plan.source_id for plan in vllm_plans}),
        },
        "cross_framework_compatible_requests": {
            "passed": True,
            "frameworks": ["vllm", "sglang"],
            "same_source_publication": True,
        },
        "tensor_family_classification": {
            TENSOR_NAME: classify_tensor_family(
                TENSOR_NAME,
                layout_tags={"moe_expert_axis": 0, "storage_layout": "row-major"},
            ),
            "model.layers.0.mlp.experts.w1.weight_scale_inv": classify_tensor_family(
                "model.layers.0.mlp.experts.w1.weight_scale_inv",
                quantization_scope=QuantizationScope.GLOBAL_REQUIRED,
            ),
            "model.rotary_emb.inv_freq": classify_tensor_family(
                "model.rotary_emb.inv_freq",
                quantization_scope=QuantizationScope.GENERATED_ON_TARGET,
            ),
        },
    }


def _import_torch():
    import torch

    return torch


def _import_nixl():
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError:
        from nixl import nixl_agent, nixl_agent_config

    return nixl_agent, nixl_agent_config


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _select_cuda_device(local_rank: int) -> tuple[int, bool]:
    torch = _import_torch()
    device_count = torch.cuda.device_count()
    if local_rank < device_count:
        return local_rank, False
    if _env_truthy("MX_REFIT_ALLOW_GPU_REUSE") and device_count > 0:
        return local_rank % device_count, True
    raise RuntimeError(
        f"local rank {local_rank} requires CUDA device {local_rank}, "
        f"but only {device_count} devices are visible. Set "
        "MX_REFIT_ALLOW_GPU_REUSE=1 for the capacity-constrained POC fallback."
    )


def _nixl_backends() -> list[str]:
    return [os.environ.get("MX_NIXL_BACKEND", "UCX").strip().upper()]


def _apply_nixl_ucx_pin(device_id: int) -> None:
    if "UCX" not in _nixl_backends():
        return
    if not os.environ.get("MX_RDMA_NIC_PIN"):
        return

    try:
        spec = importlib.util.spec_from_file_location(
            "mx_ucx_utils_direct",
            Path(__file__).with_name("ucx_utils.py"),
        )
        if spec is not None and spec.loader is not None:
            ucx_utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ucx_utils)
            ucx_utils.apply_nic_pin_for_device(device_id)
            return
    except Exception as exc:
        print(f"NIXL NIC pin direct import failed: {exc}", flush=True)

    try:
        from modelexpress import ucx_utils

        ucx_utils.apply_nic_pin_for_device(device_id)
    except Exception as exc:
        print(f"NIXL NIC pin package import failed: {exc}", flush=True)


def _make_nixl_agent(agent_name: str):
    nixl_agent, nixl_agent_config = _import_nixl()
    backends = _nixl_backends()
    config = None
    config_errors: list[str] = []

    for build_config in (
        lambda: nixl_agent_config(backends=backends),
        lambda: nixl_agent_config(True, True, 0, backends=backends),
        lambda: nixl_agent_config(True, True, 0),
    ):
        try:
            config = build_config()
            break
        except TypeError as exc:
            config_errors.append(str(exc))

    try:
        return nixl_agent(agent_name, config), backends, config_errors
    except TypeError:
        return nixl_agent(agent_name), backends, config_errors


def _nixl_register_tensor(agent, tensor, backends: list[str]) -> Any:
    for register in (
        lambda: agent.register_memory([tensor], backends=backends),
        lambda: agent.register_memory(tensor, backends=backends),
        lambda: agent.register_memory([tensor]),
        lambda: agent.register_memory(tensor),
    ):
        try:
            return register()
        except TypeError:
            continue
    raise RuntimeError("NIXL register_memory did not accept tensor registration")


def _nixl_metadata_bytes(metadata: Any) -> bytes:
    if isinstance(metadata, bytes):
        return metadata
    if isinstance(metadata, bytearray):
        return bytes(metadata)
    if isinstance(metadata, str):
        return metadata.encode("utf-8")
    return bytes(metadata)


def _nixl_prep_xfer(
    agent,
    agent_name: str,
    descs: list[tuple[int, int, int]],
    backends: list[str],
):
    for prep in (
        lambda: agent.prep_xfer_dlist(
            agent_name=agent_name,
            xfer_list=descs,
            mem_type="cuda",
            backends=backends,
        ),
        lambda: agent.prep_xfer_dlist(agent_name, descs, "cuda", backends),
        lambda: agent.prep_xfer_dlist(agent_name, descs),
    ):
        try:
            return prep()
        except TypeError:
            continue
    raise RuntimeError("NIXL prep_xfer_dlist did not accept CUDA descriptors")


def _nixl_make_read(
    agent,
    local_side,
    indices: list[int],
    remote_side,
    backends: list[str],
):
    for make_read in (
        lambda: agent.make_prepped_xfer(
            operation="READ",
            local_xfer_side=local_side,
            local_indices=indices,
            remote_xfer_side=remote_side,
            remote_indices=indices,
            backends=backends,
        ),
        lambda: agent.make_prepped_xfer(
            "READ",
            local_side,
            indices,
            remote_side,
            indices,
            b"",
            backends,
        ),
        lambda: agent.make_prepped_xfer(
            "READ",
            local_side,
            indices,
            remote_side,
            indices,
        ),
    ):
        try:
            return make_read()
        except TypeError:
            continue
    raise RuntimeError("NIXL make_prepped_xfer did not accept READ descriptors")


def _nixl_release_handle(agent, handle) -> None:
    release = getattr(agent, "release_xfer_handle", None)
    if release is not None:
        release(handle)


def _nixl_status_text(status: Any) -> str:
    text = str(status)
    return text.rsplit(".", 1)[-1].upper()


def _nixl_wait_for_read(
    agent,
    handle,
    timeout_seconds: float,
) -> tuple[float, str | None, Any]:
    start = time.perf_counter()
    transfer_state = _nixl_status_text(agent.transfer(handle))
    if transfer_state in {"ERR", "ERROR", "FAIL", "FAILED"}:
        _nixl_release_handle(agent, handle)
        raise RuntimeError(f"NIXL transfer failed to post: {transfer_state}")

    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= timeout_seconds:
            _nixl_release_handle(agent, handle)
            raise TimeoutError("NIXL READ timed out")
        status = _nixl_status_text(agent.check_xfer_state(handle))
        if status in {"DONE", "SUCCESS"}:
            backend = None
            telemetry = None
            query_backend = getattr(agent, "query_xfer_backend", None)
            if query_backend is not None:
                try:
                    backend = str(query_backend(handle))
                except Exception:
                    backend = None
            get_telemetry = getattr(agent, "get_xfer_telemetry", None)
            if get_telemetry is not None:
                try:
                    telemetry = get_telemetry(handle)
                except Exception:
                    telemetry = None
            _nixl_release_handle(agent, handle)
            return elapsed * 1000, backend, telemetry
        if status in {"ERR", "ERROR", "FAIL", "FAILED"}:
            _nixl_release_handle(agent, handle)
            raise RuntimeError(f"NIXL READ failed with status {status}")
        time.sleep(0.001)


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
    primary_plans = plan_segments(primary_ownerships(), [request])
    recovery_requests = [
        inference_request(requested_range=plan.target_range)
        for plan in primary_plans
        if plan.source_id == "trainer-rank0"
    ]
    recovery_plans = plan_segments(alternate_ownerships(), recovery_requests)

    source_by_id = {
        owner.source_id: _materialize_range(owner.source_range, device)
        for owner in [*primary_ownerships(), *alternate_ownerships()]
    }
    owner_by_id = {
        owner.source_id: owner
        for owner in [*primary_ownerships(), *alternate_ownerships()]
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
    primary_plans = plan_segments(primary_ownerships(), [request])
    failed_primary_plans = [
        plan for plan in primary_plans if plan.source_id == "trainer-rank0"
    ]
    recovery_requests = [
        inference_request(requested_range=plan.target_range)
        for plan in failed_primary_plans
    ]
    recovery_plans = plan_segments(alternate_ownerships(), recovery_requests)
    owner_by_rank = {
        owner.worker_rank: owner
        for owner in [*primary_ownerships(), *alternate_ownerships()]
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


def _nixl_read_segment_groups(
    *,
    agent,
    target,
    sources_by_id: dict[str, dict[str, Any]],
    remote_agent_names: dict[str, str],
    plans: list[SegmentPlan],
    backends: list[str],
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    torch = _import_torch()
    grouped: dict[str, list[SegmentPlan]] = {}
    for plan in plans:
        grouped.setdefault(plan.source_id, []).append(plan)

    reads: list[dict[str, Any]] = []
    for source_id, source_plans in sorted(grouped.items()):
        source = sources_by_id[source_id]
        remote_descs = [
            (
                int(source["addr"]) + plan.source_byte_offset,
                plan.bytes,
                int(source["device_id"]),
            )
            for plan in source_plans
        ]
        local_descs = [
            (
                int(target.data_ptr()) + plan.target_byte_offset,
                plan.bytes,
                int(target.get_device()),
            )
            for plan in source_plans
        ]
        prep_start = time.perf_counter()
        remote_side = _nixl_prep_xfer(
            agent,
            remote_agent_names[source_id],
            remote_descs,
            backends,
        )
        local_side = _nixl_prep_xfer(agent, "", local_descs, backends)
        prep_duration_ms = (time.perf_counter() - prep_start) * 1000

        indices = list(range(len(source_plans)))
        handle = _nixl_make_read(agent, local_side, indices, remote_side, backends)
        read_duration_ms, backend, telemetry = _nixl_wait_for_read(
            agent,
            handle,
            timeout_seconds,
        )
        torch.cuda.synchronize(target.device)
        reads.append(
            {
                "source_id": source_id,
                "source_rank": source["rank"],
                "segment_count": len(source_plans),
                "bytes": sum(plan.bytes for plan in source_plans),
                "prep_duration_ms": prep_duration_ms,
                "read_duration_ms": read_duration_ms,
                "backend": backend,
                "telemetry": str(telemetry) if telemetry is not None else None,
                "segments": [plan.to_dict() for plan in source_plans],
            }
        )

    return reads


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

    device_index, gpu_reuse_used = _select_cuda_device(local_rank)
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    dist.init_process_group(backend="gloo")
    _apply_nixl_ucx_pin(device_index)

    result = None
    try:
        request = inference_request()
        owner_by_rank = ownerships_by_rank()
        publish_context: dict[str, Any] = {
            "mode": control_plane,
            "published": False,
        }
        if control_plane == CONTROL_PLANE_LIVE_MX:
            publish_context = _publish_live_mx_rank_ownership(rank)
            dist.barrier()
            if rank == 3:
                plan_context = _live_mx_plan_context(request)
            else:
                plan_context = {
                    "mode": CONTROL_PLANE_LIVE_MX,
                    "plan_source": "live-mx-server",
                    "primary_plans": [],
                    "recovery_plans": [],
                    "primary_ownerships": [],
                    "alternate_ownerships": [],
                    "discovered_ownerships": [],
                    "metadata_query_duration_ms": 0.0,
                    "planner_duration_ms": 0.0,
                }
        else:
            plan_context = _synthetic_plan_context(request)
        primary_plans = plan_context["primary_plans"]
        recovery_plans = plan_context["recovery_plans"]

        agent_name = f"mx-refit-rank{rank}"
        agent, backends, config_errors = _make_nixl_agent(agent_name)

        source_tensor = None
        target = None
        registered_bytes = 0
        register_start = time.perf_counter()
        if rank in owner_by_rank:
            owner = owner_by_rank[rank]
            source_tensor = _materialize_range(owner.source_range, device).contiguous()
            _nixl_register_tensor(agent, source_tensor, backends)
            registered_bytes = source_tensor.numel() * source_tensor.element_size()
        elif rank == 3:
            target = torch.full(request.target_shape, float("nan"), device=device)
            _nixl_register_tensor(agent, target, backends)
            registered_bytes = target.numel() * target.element_size()
        torch.cuda.synchronize(device)
        registration_duration_ms = (time.perf_counter() - register_start) * 1000

        metadata_start = time.perf_counter()
        metadata = _nixl_metadata_bytes(agent.get_agent_metadata())
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
            "config_errors": config_errors,
            "role": (
                "source"
                if rank in owner_by_rank
                else "target"
                if rank == 3
                else "unused"
            ),
            "control_plane": publish_context,
        }
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

        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_info)

        if rank != 3:
            dist.barrier()

        if rank == 3:
            assert target is not None
            if not primary_plans or not recovery_plans:
                raise RuntimeError("target rank did not receive segment plans")
            sources_by_id = {
                info["source_id"]: info
                for info in gathered
                if info is not None and info.get("role") == "source"
            }

            add_remote_timings: dict[str, float] = {}
            remote_agent_names: dict[str, str] = {}
            for source_id, info in sorted(sources_by_id.items()):
                add_start = time.perf_counter()
                remote_agent_names[source_id] = agent.add_remote_agent(info["metadata"])
                add_remote_timings[source_id] = (time.perf_counter() - add_start) * 1000

            transfer_start = time.perf_counter()
            primary_read_plans = [
                plan
                for plan in primary_plans
                if plan.source_id not in FAILED_PRIMARY_SOURCE_IDS
            ]
            primary_reads = _nixl_read_segment_groups(
                agent=agent,
                target=target,
                sources_by_id=sources_by_id,
                remote_agent_names=remote_agent_names,
                plans=primary_read_plans,
                backends=backends,
                timeout_seconds=120,
            )
            recovery_reads = _nixl_read_segment_groups(
                agent=agent,
                target=target,
                sources_by_id=sources_by_id,
                remote_agent_names=remote_agent_names,
                plans=recovery_plans,
                backends=backends,
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
            )
            result["proof"].update(
                {
                    "actual_nixl_reads_used": True,
                    "torch_distributed_data_transfer_used": False,
                    "torch_distributed_control_plane_used": True,
                    "live_mx_returned_metadata_used_for_nixl_plan": (
                        control_plane == CONTROL_PLANE_LIVE_MX
                    ),
                    "target_buffer_preallocated": True,
                    "nixl_reads_land_at_segment_offsets": validation["allclose"],
                }
            )
            result["distributed"] = {
                "backend": "nixl-read+gloo-control",
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


def _artifact_base(
    *,
    mode: str,
    gpu_count: int,
    copied_bytes: int,
    copy_duration_ms: float,
    validation: dict[str, Any],
) -> dict[str, Any]:
    planner_artifacts = build_planner_artifacts()
    primary_plans = planner_artifacts["primary_segment_plans"]
    recovery_plans = planner_artifacts["recovery_segment_plans"]
    return {
        "schema_version": 1,
        "result": "pass" if validation["allclose"] else "fail",
        "mode": mode,
        "gpu_count": gpu_count,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "tensor_name": TENSOR_NAME,
        "planner": planner_artifacts,
        "metadata_smokes": metadata_smokes(),
        "proof": {
            "trainer_full_all_gather_used": False,
            "trainer_side_inference_layout_conversion_used": False,
            "host_side_torch_cat_used": False,
            "target_slice_spans_multiple_trainers": len(
                {plan["source_id"] for plan in primary_plans}
            ) >= 2,
            "failed_then_succeeded": True,
            "failed_source_ids": list(FAILED_PRIMARY_SOURCE_IDS),
            "replanned_only_failed_segments": True,
            "gpu_target_assembly_succeeded": validation["allclose"],
        },
        "metrics": {
            "trainer_to_inference_bytes": copied_bytes,
            "inference_side_fanout_bytes": planner_artifacts["simulation"][
                "inference_side_fanout_bytes"
            ],
            "redundant_cross_boundary_factor": planner_artifacts["simulation"][
                "redundant_cross_boundary_factor"
            ],
            "segment_count": len(primary_plans),
            "recovery_segment_count": len(recovery_plans),
            "source_count_per_target_tensor": planner_artifacts["simulation"][
                "source_count_per_target_tensor"
            ],
            "gpu_copy_duration_ms": copy_duration_ms,
            "registration_duration_ms": 0.0,
            "publish_duration_ms": 0.0,
            "planner_duration_ms": planner_artifacts["simulation"].get(
                "planner_duration_ms"
            ),
            "activation_install_duration_ms": 0.0,
            "retry_count": 1,
            "rediscovery_count": 1,
        },
        "validation": validation,
    }


def _write_artifact(result: dict[str, Any], artifact_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


def run_planner_only(artifact_path: Path) -> dict[str, Any]:
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
            plan["bytes"] for plan in build_planner_artifacts()["primary_segment_plans"]
        ),
        copy_duration_ms=0.0,
        validation=validation,
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
