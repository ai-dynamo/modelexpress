# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-node SGLang runtime refit through MX-published NIXL endpoints.

This is the role-split SGLang counterpart to ``refit_vllm_mx_runtime``. Source
pods publish trainer-step shard tensors and NIXL descriptors through the MX
endpoint control plane. A separate target pod starts a live ``sglang.Engine``,
builds its receiver request from the engine-owned weight, discovers source
endpoints from MX, issues one-sided NIXL READs into a CUDA staging tensor, and
installs the assembled payload with ``Engine.update_weights_from_tensor``.

The READs still land in staging before the SGLang install callback. This proves
cross-node trainer-to-inference runtime refit for SGLang placement, without
trainer full all-gather or trainer-side inference-layout conversion; it does
not claim direct NIXL landing into SGLang-owned storage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

import torch

from . import p2p_pb2
from .client import MxClient
from .metadata.heartbeat import HeartbeatThread
from .refit_nixl import (
    NixlAdapter,
    apply_nixl_ucx_pin,
    read_segment_groups,
    select_cuda_device,
)
from .refit_sglang_nixl_runtime_smoke import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TENSOR_NAME,
    _clear_torchrun_env_for_runtime_engine,
    _torch_dtype_from_name,
    build_sglang_nixl_source_ownerships,
    build_sglang_runtime_nixl_plan,
    run_sglang_receiver_refit_from_nixl_staging_tensor,
)
from .refit_sglang_receiver_smoke import (
    _select_sglang_engine_weight,
    create_tiny_llama_checkpoint,
    detect_sglang_version,
    load_sglang_engine,
)
from .refit_trainer_step import (
    publish_trainer_step_source,
    trainer_step_source_provenance,
)
from .refit_mx_runtime_common import (
    effective_model_version as _effective_model_version,
    filter_source_ownerships as _filter_source_ownerships,
    parse_csv as _parse_csv,
    parse_shape as _parse_shape,
    scenario_with_mx_endpoint_plan as _common_scenario_with_mx_endpoint_plan,
    source_status_name as _source_status_name,
)
from .refit_vllm_receiver_smoke import _json_default, _torch_dtype_name
from .resharding import SliceOwnership, SliceRequest
from .resharding_control_plane import (
    RefitNixlEndpoint,
    build_refit_source_identity,
    list_refit_nixl_endpoints,
    publish_refit_nixl_endpoint,
)
from .types import TensorDescriptor

DEFAULT_SOURCE_SHAPE = (64, 64)
DEFAULT_MX_RUNTIME_MODEL_VERSION = "mx-endpoint-crossnode-sglang-nixl"
SOURCE_IDS = ("trainer-rank0", "trainer-rank1")


def _trace(phase: str, **fields: Any) -> None:
    payload = {"phase": phase, **fields}
    print(
        "[mx-sglang-mx-runtime] "
        f"{json.dumps(payload, default=_json_default, sort_keys=True)}",
        flush=True,
    )


def _refit_identity(*, model_name: str, model_version: str, dtype: str):
    return build_refit_source_identity(
        model_name=model_name,
        model_version=model_version,
        dtype=dtype,
        trainer_framework="torch.optim.SGD-step-publisher",
        trainer_layout="fsdp-row-shard-sglang-runtime",
    )


def scenario_with_mx_endpoint_plan(
    scenario: dict[str, Any],
    endpoints: Sequence[RefitNixlEndpoint],
    *,
    metadata_query_duration_ms: float,
    planner_duration_ms: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _common_scenario_with_mx_endpoint_plan(
        scenario,
        endpoints,
        mode="live-sglang-mx-endpoint-runtime",
        plan_source="live-sglang-request+mx-endpoint-ownerships",
        metadata_query_duration_ms=metadata_query_duration_ms,
        planner_duration_ms=planner_duration_ms,
    )


def build_sglang_mx_runtime_source_ownerships(
    *,
    tensor_name: str,
    shape: tuple[int, ...] = DEFAULT_SOURCE_SHAPE,
    dtype_name: str = "float32",
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MX_RUNTIME_MODEL_VERSION,
) -> list[SliceOwnership]:
    dtype_obj = _torch_dtype_from_name(dtype_name)
    return build_sglang_nixl_source_ownerships(
        tensor_name=tensor_name,
        shape=tuple(int(dim) for dim in shape),
        dtype_name=_torch_dtype_name(dtype_obj),
        element_size_bytes=int(torch.empty((), dtype=dtype_obj).element_size()),
        model_name=model_name,
        model_version=model_version,
    )


def _source_agent_name(run_id: str, ownerships: Sequence[SliceOwnership]) -> str:
    if len(ownerships) == 1:
        return f"mx-sglang-mx-source-{run_id}-{ownerships[0].source_id}"
    return f"mx-sglang-mx-source-{run_id}"


def materialize_sglang_mx_runtime_source_publications(
    ownerships: Sequence[SliceOwnership],
    *,
    dtype: torch.dtype,
    device: torch.device,
):
    """Create trainer-step source publications for MX endpoint publishing."""

    return [
        publish_trainer_step_source(owner, dtype=dtype, device=device)
        for owner in ownerships
    ]


def _wait_for_mx_endpoints(
    *,
    model_name: str,
    model_version: str,
    dtype: str,
    scenario: dict[str, Any],
    expected_source_ids: set[str],
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _refit_identity(
        model_name=model_name, model_version=model_version, dtype=dtype
    )
    mx_client = MxClient()
    try:
        deadline = time.time() + float(timeout_seconds)
        metadata_query_start = time.perf_counter()
        endpoints: list[RefitNixlEndpoint] = []
        while True:
            endpoints = list_refit_nixl_endpoints(
                mx_client,
                identity=identity,
                status_filter=p2p_pb2.SOURCE_STATUS_READY,
            )
            present = {endpoint.source_id for endpoint in endpoints}
            if expected_source_ids.issubset(present):
                break
            if time.time() >= deadline:
                raise RuntimeError(
                    "timed out waiting for SGLang MX runtime NIXL endpoints "
                    f"(model_name={model_name}, model_version={model_version}, "
                    f"missing={sorted(expected_source_ids - present)}, "
                    f"present={sorted(present)})"
                )
            time.sleep(0.25)

        metadata_query_duration_ms = (time.perf_counter() - metadata_query_start) * 1000
        planner_start = time.perf_counter()
        planned_scenario, plan_context = scenario_with_mx_endpoint_plan(
            scenario,
            endpoints,
            metadata_query_duration_ms=metadata_query_duration_ms,
            planner_duration_ms=0.0,
        )
        plan_context["planner_duration_ms"] = (
            time.perf_counter() - planner_start
        ) * 1000
        plan_context["server_url"] = getattr(mx_client, "server_url", "")
        return planned_scenario, plan_context
    finally:
        mx_client.close()


def run_source(
    *,
    run_id: str,
    hold_seconds: float,
    artifact_path: Path | None,
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MX_RUNTIME_MODEL_VERSION,
    tensor_name: str = DEFAULT_TENSOR_NAME,
    target_shape: tuple[int, ...] = DEFAULT_SOURCE_SHAPE,
    source_dtype: str = "float32",
    source_id: str = "",
    source_worker_rank: int | None = None,
    local_rank: int = 0,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SGLang MX runtime source publishing")

    effective_model_version = _effective_model_version(model_version, run_id)
    device_index, gpu_reuse_used = select_cuda_device(local_rank)
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    apply_nixl_ucx_pin(device_index)

    dtype_obj = _torch_dtype_from_name(source_dtype)
    ownerships = _filter_source_ownerships(
        build_sglang_mx_runtime_source_ownerships(
            tensor_name=tensor_name,
            shape=target_shape,
            dtype_name=_torch_dtype_name(dtype_obj),
            model_name=model_name,
            model_version=effective_model_version,
        ),
        source_id=source_id,
        worker_rank=source_worker_rank,
    )
    adapter = NixlAdapter(_source_agent_name(run_id, ownerships))
    source_publications = materialize_sglang_mx_runtime_source_publications(
        ownerships,
        dtype=dtype_obj,
        device=device,
    )
    source_tensors = {
        publication.ownership.source_id: publication.tensor
        for publication in source_publications
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
    publications: list[dict[str, Any]] = []
    try:
        identity = _refit_identity(
            model_name=model_name,
            model_version=effective_model_version,
            dtype=_torch_dtype_name(dtype_obj),
        )
        for publication in source_publications:
            owner = publication.ownership
            tensor = publication.tensor
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
                agent_name=adapter.agent_name,
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
                    "worker_rank": int(owner.worker_rank or 0),
                    "source_id": owner.source_id,
                    "agent_name": adapter.agent_name,
                    "metadata_bytes": len(metadata),
                    "tensor_addr": descriptor.addr,
                    "tensor_bytes": descriptor.size,
                    "tensor_device_id": descriptor.device_id,
                    "publish_duration_ms": publish_duration_ms,
                    "source_payload_provenance": publication.provenance,
                    "source_publication": publication.to_artifact_metadata(),
                }
            )

        result = {
            "schema_version": 1,
            "result": "source-ready",
            "role": "source",
            "mode": "live-sglang-mx-runtime-source",
            "run_id": run_id,
            "model_name": model_name,
            "model_version": effective_model_version,
            "model_version_base": model_version,
            "target_tensor_name": tensor_name,
            "target_shape": list(target_shape),
            "target_dtype": _torch_dtype_name(dtype_obj),
            "pod_name": os.environ.get("POD_NAME", ""),
            "node_name": os.environ.get("NODE_NAME", ""),
            "agent_name": adapter.agent_name,
            "single_source_ownership_mode": len(ownerships) == 1,
            "source_ownership_count": len(ownerships),
            "source_filter": {
                "source_id": source_id,
                "source_worker_rank": source_worker_rank,
            },
            "gpu_reuse_used": gpu_reuse_used,
            "cuda_device": device_index,
            "nixl_backends": adapter.backends,
            "metrics": {
                "nixl_registration_duration_ms": registration_duration_ms,
                "nixl_metadata_duration_ms": metadata_duration_ms,
                "publish_duration_ms": sum(
                    item["publish_duration_ms"] for item in publications
                ),
            },
            "source_publications": publications,
            "trainer_source_update": trainer_step_source_provenance(),
        }
        if artifact_path is not None:
            _write_runtime_artifact(result, artifact_path)
        else:
            print(json.dumps(result, default=_json_default, sort_keys=True), flush=True)

        time.sleep(max(0.0, float(hold_seconds)))
        return result
    finally:
        for heartbeat in heartbeats:
            heartbeat.stop()
        mx_client.close()


def run_target(
    *,
    run_id: str,
    artifact_path: Path,
    model_path: str = "",
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MX_RUNTIME_MODEL_VERSION,
    preferred_tensor_name: str = DEFAULT_TENSOR_NAME,
    engine_dtype: str = "bfloat16",
    context_length: int = 64,
    mem_fraction_static: float = 0.2,
    disable_cuda_graph: bool = True,
    disable_piecewise_cuda_graph: bool = True,
    log_level: str = "info",
    timeout_seconds: float = 180.0,
    local_rank: int = 0,
    expected_source_ids: set[str] | None = None,
    source_pods: Sequence[str] = (),
    source_nodes: Sequence[str] = (),
    target_pod: str = "",
    target_node: str = "",
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SGLang MX runtime target reads")

    effective_model_version = _effective_model_version(model_version, run_id)
    expected_source_ids = set(
        SOURCE_IDS if expected_source_ids is None else expected_source_ids
    )

    device_index, gpu_reuse_used = select_cuda_device(local_rank)
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")
    apply_nixl_ucx_pin(device_index)

    resolved_model_path = model_path
    if not resolved_model_path:
        resolved_model_path = str(
            create_tiny_llama_checkpoint(
                Path(tempfile.mkdtemp(prefix="mx-sglang-mx-runtime-model-"))
            )
        )

    engine = None
    try:
        sglang_version, _ = detect_sglang_version()
        _trace(
            "target.sglang_start.start",
            run_id=run_id,
            model_version=effective_model_version,
            model_path=resolved_model_path,
        )
        engine_start = time.perf_counter()
        saved_engine_env = _clear_torchrun_env_for_runtime_engine()
        try:
            engine = load_sglang_engine(
                model_path=resolved_model_path,
                dtype=engine_dtype,
                context_length=context_length,
                mem_fraction_static=mem_fraction_static,
                disable_cuda_graph=disable_cuda_graph,
                disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
                log_level=log_level,
            )
        finally:
            os.environ.update(saved_engine_env)
        engine_start_duration_ms = (time.perf_counter() - engine_start) * 1000
        _trace("target.sglang_start.done", duration_ms=engine_start_duration_ms)

        fetch_start = time.perf_counter()
        tensor_name, original = _select_sglang_engine_weight(
            engine,
            preferred_tensor_name=preferred_tensor_name,
        )
        initial_weight_fetch_duration_ms = (time.perf_counter() - fetch_start) * 1000
        request, local_owners, local_plans = build_sglang_runtime_nixl_plan(
            original,
            tensor_name=tensor_name,
            model_name=model_name,
            model_version=effective_model_version,
        )
        scenario = {
            "tensor_name": tensor_name,
            "target_tensor_name": tensor_name,
            "target_shape": list(original.shape),
            "target_dtype": _torch_dtype_name(original.dtype),
            "request": request.to_dict(),
            "source_ownerships": [owner.to_dict() for owner in local_owners],
            "segment_plans": [plan.to_dict() for plan in local_plans],
            "model_path": resolved_model_path,
            "sglang_version": sglang_version,
            "engine_start_duration_ms": engine_start_duration_ms,
            "initial_weight_fetch_duration_ms": initial_weight_fetch_duration_ms,
        }
        runtime_dtype = str(scenario["target_dtype"])

        _trace(
            "target.wait_for_mx_endpoints.start",
            expected_source_ids=sorted(expected_source_ids),
            runtime_dtype=runtime_dtype,
        )
        scenario, plan_context = _wait_for_mx_endpoints(
            model_name=model_name,
            model_version=effective_model_version,
            dtype=runtime_dtype,
            scenario=scenario,
            expected_source_ids=expected_source_ids,
            timeout_seconds=timeout_seconds,
        )
        _trace(
            "target.wait_for_mx_endpoints.done",
            source_ids=sorted(plan_context["source_endpoints_by_id"]),
            metadata_query_duration_ms=plan_context["metadata_query_duration_ms"],
            planner_duration_ms=plan_context["planner_duration_ms"],
        )

        target_shape = tuple(int(dim) for dim in scenario["target_shape"])
        dtype_obj = _torch_dtype_from_name(runtime_dtype)
        target_staging = torch.full(
            target_shape,
            float("nan"),
            device=device,
            dtype=dtype_obj,
        )
        adapter = NixlAdapter(f"mx-sglang-mx-target-{run_id}")
        register_start = time.perf_counter()
        adapter.register_tensor(target_staging)
        torch.cuda.synchronize(device)
        registration_duration_ms = (time.perf_counter() - register_start) * 1000

        endpoint_by_source_id: dict[str, RefitNixlEndpoint] = plan_context[
            "source_endpoints_by_id"
        ]
        sources_by_id = {
            source_id: endpoint.to_nixl_source_info()
            for source_id, endpoint in endpoint_by_source_id.items()
        }
        remote_agent_cache: dict[str, str] = {}
        add_remote_timings: dict[str, float] = {}
        remote_agent_names: dict[str, str] = {}
        for source_id, endpoint in sorted(endpoint_by_source_id.items()):
            if endpoint.agent_name not in remote_agent_cache:
                add_start = time.perf_counter()
                remote_agent_cache[endpoint.agent_name] = adapter.add_remote_agent(
                    endpoint.nixl_metadata
                )
                add_remote_timings[endpoint.agent_name] = (
                    time.perf_counter() - add_start
                ) * 1000
            remote_agent_names[source_id] = remote_agent_cache[endpoint.agent_name]

        plans = plan_context["plans"]
        _trace("target.read_segments.start", segment_count=len(plans))
        transfer_start = time.perf_counter()
        reads = read_segment_groups(
            adapter=adapter,
            target=target_staging,
            sources_by_id=sources_by_id,
            remote_agent_names=remote_agent_names,
            plans=plans,
            timeout_seconds=timeout_seconds,
        )
        torch.cuda.synchronize(device)
        raw_nixl_read_duration_ms = (time.perf_counter() - transfer_start) * 1000
        _trace(
            "target.read_segments.done",
            raw_nixl_read_duration_ms=raw_nixl_read_duration_ms,
        )

        copied_bytes = sum(read["bytes"] for read in reads)
        readable_source_agent_names = {
            source_id: endpoint.agent_name
            for source_id, endpoint in sorted(endpoint_by_source_id.items())
        }
        distinct_source_agent_count = len(set(readable_source_agent_names.values()))
        source_endpoint_count = len(endpoint_by_source_id)
        one_agent_per_source_rank = distinct_source_agent_count == source_endpoint_count
        source_nodes = [node for node in source_nodes if node]
        source_pods = [pod for pod in source_pods if pod]
        target_pod = target_pod or os.environ.get("POD_NAME", "")
        target_node = target_node or os.environ.get("NODE_NAME", "")
        cross_node = bool(
            target_node
            and source_nodes
            and all(node != target_node for node in source_nodes)
        )

        nixl_metrics = {
            "raw_nixl_read_duration_ms": raw_nixl_read_duration_ms,
            "nixl_registration_duration_ms": registration_duration_ms,
            "nixl_add_remote_agent_duration_ms": sum(add_remote_timings.values()),
            "nixl_prep_duration_ms": sum(read["prep_duration_ms"] for read in reads),
            "nixl_read_group_count": len(reads),
            "successful_nixl_source_count": len({read["source_id"] for read in reads}),
            "trainer_to_inference_bytes": copied_bytes,
            "metadata_query_duration_ms": plan_context["metadata_query_duration_ms"],
            "planner_duration_ms": plan_context["planner_duration_ms"],
            "source_endpoint_count": source_endpoint_count,
            "distinct_source_agent_count": distinct_source_agent_count,
        }
        distributed = {
            "backend": "nixl-read+mx-endpoint-control+sglang-engine-update",
            "run_id": run_id,
            "target_pod": target_pod,
            "target_node": target_node,
            "source_pods": list(source_pods),
            "source_nodes": list(source_nodes),
            "cross_node": cross_node,
            "source_ids": sorted(endpoint_by_source_id),
            "source_agent_names_by_source_id": readable_source_agent_names,
            "target_agent_name": adapter.agent_name,
            "gpu_reuse_used": gpu_reuse_used,
            "torch_distributed_control_plane_used": False,
            "torch_distributed_nixl_metadata_exchange_used": False,
        }

        request = SliceRequest.from_dict(plan_context["request"])
        result = run_sglang_receiver_refit_from_nixl_staging_tensor(
            engine,
            tensor_name=str(scenario["tensor_name"]),
            original=original,
            assembled=target_staging,
            request=request,
            source_ownerships=[
                endpoint.ownership for endpoint in endpoint_by_source_id.values()
            ],
            segment_plans=plans,
            model_name=model_name,
            model_version=effective_model_version,
            sglang_version=str(scenario["sglang_version"]),
            artifact_path=None,
            mode="live-sglang-nixl-runtime-mx-crossnode-refit",
            model_path=str(scenario["model_path"]),
            engine_start_duration_ms=float(scenario["engine_start_duration_ms"]),
            initial_weight_fetch_duration_ms=float(
                scenario["initial_weight_fetch_duration_ms"]
            ),
            nixl_reads=reads,
            nixl_metrics=nixl_metrics,
            distributed=distributed,
            actual_nixl_reads_used=True,
            runtime_storage_dtype=_torch_dtype_from_name(engine_dtype),
        )
        result["run_id"] = run_id
        result["model_version_base"] = model_version
        result["proof"].update(
            {
                "cross_node_pods": cross_node,
                "source_pod_separate_from_target_pod": bool(
                    target_pod and source_pods and target_pod not in set(source_pods)
                ),
                "nixl_source_endpoints_from_mx": True,
                "torch_distributed_control_plane_used": False,
                "torch_distributed_nixl_metadata_exchange_used": False,
                "live_mx_returned_metadata_used_for_nixl_plan": True,
                "receiver_request_from_live_sglang_engine_weight": True,
                "target_buffer_preallocated": True,
                "one_nixl_agent_per_source_rank": one_agent_per_source_rank,
                "one_pod_per_source_rank": one_agent_per_source_rank
                and source_endpoint_count > 1,
                "source_endpoint_count": source_endpoint_count,
                "distinct_source_agent_count": distinct_source_agent_count,
                "cross_node_runtime_target_pod": cross_node,
            }
        )
        result["control_plane"] = {
            "mode": "live-sglang-mx-endpoint-runtime",
            "server_url": plan_context.get("server_url", ""),
            "source_endpoints_from_control_plane": [
                endpoint.to_dict()
                for endpoint in plan_context[
                    "discovered_source_endpoints_by_id"
                ].values()
            ],
            "source_statuses_by_id": {
                source_id: {
                    **status,
                    "status_name": status.get("status_name")
                    or _source_status_name(status["status"]),
                }
                for source_id, status in plan_context["source_statuses_by_id"].items()
            },
            "metadata_query_duration_ms": plan_context["metadata_query_duration_ms"],
            "planner_duration_ms": plan_context["planner_duration_ms"],
        }
        result["nixl"].update(
            {
                "target_agent_name": adapter.agent_name,
                "nixl_backends": adapter.backends,
                "source_metadata": {
                    source_id: {
                        "agent_name": endpoint.agent_name,
                        "metadata_bytes": len(endpoint.nixl_metadata),
                        "source_range": endpoint.ownership.source_range,
                        "registered_bytes": endpoint.tensor.size,
                        "device_id": endpoint.tensor.device_id,
                        "worker_rank": endpoint.worker_rank,
                    }
                    for source_id, endpoint in sorted(endpoint_by_source_id.items())
                },
                "remote_agent_names": remote_agent_names,
                "add_remote_agent_duration_ms": add_remote_timings,
            }
        )
        _write_runtime_artifact(result, artifact_path)
        return result
    finally:
        if engine is not None:
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                shutdown()


def _write_runtime_artifact(payload: dict[str, Any], artifact_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["source", "target"], required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-version", default=DEFAULT_MX_RUNTIME_MODEL_VERSION)
    parser.add_argument("--preferred-tensor-name", default=DEFAULT_TENSOR_NAME)
    parser.add_argument("--tensor-name", default=DEFAULT_TENSOR_NAME)
    parser.add_argument(
        "--target-shape", default=",".join(str(dim) for dim in DEFAULT_SOURCE_SHAPE)
    )
    parser.add_argument("--source-dtype", default="float32")
    parser.add_argument("--engine-dtype", default="bfloat16")
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--mem-fraction-static", type=float, default=0.2)
    parser.add_argument("--enable-cuda-graph", action="store_true")
    parser.add_argument("--enable-piecewise-cuda-graph", action="store_true")
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--hold-seconds", type=float, default=300.0)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--source-id", default=os.environ.get("SOURCE_ID", ""))
    parser.add_argument("--source-worker-rank", type=int)
    parser.add_argument("--expected-source-id", action="append", default=[])
    parser.add_argument("--source-pod", action="append", default=[])
    parser.add_argument("--source-node", action="append", default=[])
    parser.add_argument("--source-pods", default=os.environ.get("SOURCE_PODS", ""))
    parser.add_argument("--source-nodes", default=os.environ.get("SOURCE_NODES", ""))
    parser.add_argument("--target-pod", default=os.environ.get("POD_NAME", ""))
    parser.add_argument("--target-node", default=os.environ.get("NODE_NAME", ""))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.role == "source":
        result = run_source(
            run_id=args.run_id,
            hold_seconds=args.hold_seconds,
            artifact_path=args.artifact_path,
            model_name=args.model_name,
            model_version=args.model_version,
            tensor_name=args.tensor_name,
            target_shape=_parse_shape(args.target_shape),
            source_dtype=args.source_dtype,
            source_id=args.source_id,
            source_worker_rank=args.source_worker_rank,
            local_rank=args.local_rank,
        )
    else:
        source_pods = [*args.source_pod, *_parse_csv(args.source_pods)]
        source_nodes = [*args.source_node, *_parse_csv(args.source_nodes)]
        result = run_target(
            run_id=args.run_id,
            artifact_path=args.artifact_path,
            model_path=args.model_path,
            model_name=args.model_name,
            model_version=args.model_version,
            preferred_tensor_name=args.preferred_tensor_name,
            engine_dtype=args.engine_dtype,
            context_length=args.context_length,
            mem_fraction_static=args.mem_fraction_static,
            disable_cuda_graph=not args.enable_cuda_graph,
            disable_piecewise_cuda_graph=not args.enable_piecewise_cuda_graph,
            log_level=args.log_level,
            timeout_seconds=args.timeout_seconds,
            local_rank=args.local_rank,
            expected_source_ids=set(args.expected_source_id or SOURCE_IDS),
            source_pods=source_pods,
            source_nodes=source_nodes,
            target_pod=args.target_pod,
            target_node=args.target_node,
        )
    print(
        "MX_SGLANG_MX_RUNTIME_REFIT "
        f"role={args.role} result={result['result']} "
        f"model_version={result['model_version']}",
        flush=True,
    )
    print(json.dumps(result, default=_json_default, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
