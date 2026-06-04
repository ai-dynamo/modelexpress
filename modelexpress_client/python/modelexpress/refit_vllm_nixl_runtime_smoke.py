# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live vLLM receiver refit driven by NIXL-read trainer source tensors.

The proof bridges source-rank-owned NIXL payloads into vLLM's worker-owned
model tensor lifecycle. Source ranks own CUDA shard tensors and the target rank
issues one-sided NIXL READs into a preallocated CUDA staging tensor. The target
then uses ``LLM.apply_model`` to install that assembled payload into the vLLM
worker-owned tensor, validate checksum/allclose, and restore the original
weight.

The NIXL reads do not land directly in vLLM-owned storage; the target copies the
assembled staging tensor through the apply_model callback boundary. Source
values are deterministic POC values, not a live optimizer step.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from functools import partial
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import torch

from .refit_nixl import (
    NixlAdapter,
    apply_nixl_ucx_pin,
    read_segment_groups,
    select_cuda_device,
)
from .refit_vllm_receiver_smoke import (
    _json_default,
    _nbytes,
    _select_refit_tensor,
    _shape,
    _tensor_checksum,
    _tensor_close,
    _tensor_max_abs_error,
    _torch_dtype_name,
    _write_artifact,
    create_tiny_qwen2_checkpoint,
    load_vllm_llm,
)
from .resharding import (
    SegmentPlan,
    SliceOwnership,
    SliceRequest,
    TensorRange,
    plan_segments,
)
from .resharding_receiver import build_receiver_requests_from_runtime_tensors

DEFAULT_MODEL_NAME = "mx-live-vllm-nixl-runtime-smoke"
DEFAULT_MODEL_VERSION = "trainer-step-live-vllm-nixl"
DEFAULT_TENSOR_NAME = "lm_head.weight"
DEFAULT_TARGET_RANK = 2
SOURCE_RANKS = (0, 1)
_TORCHRUN_ENV_KEYS = (
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_USE_AGENT_STORE",
)


def _clear_torchrun_env_for_runtime_engine() -> dict[str, str]:
    saved: dict[str, str] = {}
    for key in _TORCHRUN_ENV_KEYS:
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    return saved


def _restore_torchrun_env(saved: dict[str, str]) -> None:
    os.environ.update(saved)


def _torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.removeprefix("torch.").lower()
    dtype = getattr(torch, normalized, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    aliases = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float": torch.float32,
    }
    if normalized in aliases:
        return aliases[normalized]
    raise ValueError(f"unsupported torch dtype for runtime NIXL smoke: {dtype_name!r}")


def _replacement_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str,
) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= int(dim)
    values = torch.arange(numel, device=device, dtype=torch.float32)
    values = ((values % 257) - 128) / 8
    return values.reshape(shape).to(dtype=dtype)


def _range_slices(tensor_range: TensorRange) -> tuple[slice, ...]:
    return tuple(slice(start, end) for start, end in tensor_range)


def build_vllm_nixl_source_ownerships(
    *,
    tensor_name: str,
    shape: tuple[int, ...],
    dtype_name: str,
    element_size_bytes: int,
    model_name: str,
    model_version: str,
) -> list[SliceOwnership]:
    """Return two trainer-rank ownership records for a vLLM tensor shape."""

    if len(shape) < 2:
        raise ValueError("vLLM NIXL runtime smoke requires a rank-2 tensor")
    rows = int(shape[0])
    if rows < 2:
        raise ValueError("vLLM NIXL runtime smoke requires at least two rows")

    split = max(1, rows // 2)
    trailing_ranges = tuple((0, int(dim)) for dim in shape[1:])
    common_tags: dict[str, str | int | bool] = {
        "trainer_layout": "fsdp-row-shard-poc",
        "storage_layout": "row-major",
        "source_tensor_owner": "torchrun-trainer-rank",
        "runtime_refit_target": "vllm.LLM.apply_model",
    }
    return [
        SliceOwnership(
            model_name=model_name,
            model_version=model_version,
            tensor_name=tensor_name,
            global_shape=shape,
            dtype=dtype_name,
            source_range=((0, split), *trailing_ranges),
            worker_id="trainer-rank0-worker",
            source_id="trainer-rank0",
            worker_rank=0,
            source_lease="trainer-rank0-live-vllm-nixl-refit",
            nixl_descriptor_id="trainer-rank0-live-vllm-nixl-refit",
            layout_tags=dict(common_tags),
            element_size_bytes=element_size_bytes,
        ),
        SliceOwnership(
            model_name=model_name,
            model_version=model_version,
            tensor_name=tensor_name,
            global_shape=shape,
            dtype=dtype_name,
            source_range=((split, rows), *trailing_ranges),
            worker_id="trainer-rank1-worker",
            source_id="trainer-rank1",
            worker_rank=1,
            source_lease="trainer-rank1-live-vllm-nixl-refit",
            nixl_descriptor_id="trainer-rank1-live-vllm-nixl-refit",
            layout_tags=dict(common_tags),
            element_size_bytes=element_size_bytes,
        ),
    ]


def build_vllm_runtime_nixl_plan(
    target_tensor: torch.Tensor,
    *,
    tensor_name: str,
    model_name: str,
    model_version: str,
    module_path: str = "llm.apply_model.worker_model",
) -> tuple[SliceRequest, list[SliceOwnership], list[SegmentPlan]]:
    """Build receiver request and source plans from a vLLM worker tensor."""

    request = build_receiver_requests_from_runtime_tensors(
        {tensor_name: target_tensor},
        model_name=model_name,
        model_version=model_version,
        runtime_framework="vllm",
        target_id_prefix="vllm",
        layout_tags_by_tensor={
            tensor_name: {
                "runtime_module_path": module_path,
                "runtime_tensor_name": tensor_name,
                "vllm_module_path": module_path,
                "vllm_tensor_name": tensor_name,
                "runtime_lifecycle": "apply-model-update-from-nixl-staging",
            }
        },
    )[0]
    owners = build_vllm_nixl_source_ownerships(
        tensor_name=tensor_name,
        shape=_shape(target_tensor),
        dtype_name=_torch_dtype_name(target_tensor.dtype),
        element_size_bytes=int(target_tensor.element_size()),
        model_name=model_name,
        model_version=model_version,
    )
    return request, owners, plan_segments(owners, [request])


def materialize_vllm_nixl_source_tensor(
    owner: SliceOwnership,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create the rank-owned trainer payload for one source range."""

    full_replacement = _replacement_tensor(
        owner.global_shape,
        dtype=dtype,
        device=device,
    )
    return full_replacement[_range_slices(owner.source_range)].contiguous()


def _inspect_vllm_worker_model(
    model: torch.nn.Module,
    *,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    tensor_name, target_tensor = _select_refit_tensor(
        model.named_parameters(),
        preferred_name=kwargs.get("preferred_tensor_name", ""),
    )
    module_path = str(kwargs.get("module_path", "llm.apply_model.worker_model"))
    request, owners, plans = build_vllm_runtime_nixl_plan(
        target_tensor,
        tensor_name=tensor_name,
        model_name=str(kwargs["model_name"]),
        model_version=str(kwargs["model_version"]),
        module_path=module_path,
    )
    return {
        "module_class": type(model).__name__,
        "module_path": module_path,
        "target_tensor_name": tensor_name,
        "target_shape": list(_shape(target_tensor)),
        "target_dtype": _torch_dtype_name(target_tensor.dtype),
        "target_device": str(target_tensor.device),
        "target_tensor_bytes": _nbytes(target_tensor),
        "original_checksum": _tensor_checksum(target_tensor),
        "request": request.to_dict(),
        "source_ownerships": [owner.to_dict() for owner in owners],
        "segment_plans": [plan.to_dict() for plan in plans],
    }


def _apply_vllm_nixl_payload_on_worker_model(
    model: torch.nn.Module,
    *,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    tensor_name, target_tensor = _select_refit_tensor(
        model.named_parameters(),
        preferred_name=str(kwargs["tensor_name"]),
    )
    assembled_payload = kwargs["assembled_payload_cpu"]
    if not isinstance(assembled_payload, torch.Tensor):
        raise TypeError("assembled_payload_cpu must be a torch.Tensor")

    original = target_tensor.detach().clone()
    expected = _replacement_tensor(
        _shape(target_tensor),
        dtype=target_tensor.dtype,
        device=target_tensor.device,
    )
    prepared = assembled_payload.to(
        device=target_tensor.device,
        dtype=target_tensor.dtype,
    ).contiguous()
    payload_allclose = _tensor_close(prepared, expected)
    payload_checksum = _tensor_checksum(prepared)
    expected_checksum = _tensor_checksum(expected)
    payload_checksum_matches = payload_checksum == expected_checksum

    install_start = time.perf_counter()
    with torch.no_grad():
        target_tensor.copy_(prepared)
    if target_tensor.is_cuda:
        torch.cuda.synchronize(target_tensor.device)
    install_duration_ms = (time.perf_counter() - install_start) * 1000

    allclose = _tensor_close(target_tensor, expected)
    checksum = _tensor_checksum(target_tensor)
    checksum_matches = checksum == expected_checksum
    max_abs_error = _tensor_max_abs_error(target_tensor, expected)

    restore_start = time.perf_counter()
    with torch.no_grad():
        target_tensor.copy_(original)
    if target_tensor.is_cuda:
        torch.cuda.synchronize(target_tensor.device)
    restore_duration_ms = (time.perf_counter() - restore_start) * 1000
    restored_original = _tensor_close(target_tensor, original)

    return {
        "module_class": type(model).__name__,
        "module_path": str(kwargs.get("module_path", "llm.apply_model.worker_model")),
        "target_tensor_name": tensor_name,
        "target_shape": list(_shape(target_tensor)),
        "target_dtype": _torch_dtype_name(target_tensor.dtype),
        "target_device": str(target_tensor.device),
        "proof": {
            "vllm_apply_model_used": True,
            "vllm_worker_owned_target_tensor": True,
            "vllm_owned_target_tensor": True,
            "receiver_installed_into_vllm_owned_tensor": allclose,
            "receiver_installed_into_runtime_owned_tensor": allclose,
            "runtime_owned_target_tensor": True,
            "runtime_update_from_nixl_staging_tensor": allclose,
            "runtime_update_payload_copied_through_apply_model": True,
            "restored_original_tensor": restored_original,
            "checksum_gate": checksum_matches,
            "allclose": allclose,
        },
        "validation": {
            "payload_allclose": payload_allclose,
            "payload_checksum": payload_checksum,
            "payload_checksum_matches": payload_checksum_matches,
            "allclose": allclose,
            "checksum": checksum,
            "expected_checksum": expected_checksum,
            "checksum_matches": checksum_matches,
            "max_abs_error": max_abs_error,
            "original_checksum": _tensor_checksum(original),
            "restored_checksum": _tensor_checksum(target_tensor),
            "restored_original": restored_original,
        },
        "metrics": {
            "activation_install_duration_ms": install_duration_ms,
            "restore_duration_ms": restore_duration_ms,
            "target_tensor_bytes": _nbytes(target_tensor),
        },
    }


def inspect_vllm_runtime_nixl_scenario(
    llm: Any,
    *,
    model_name: str,
    model_version: str,
    preferred_tensor_name: str = "",
) -> dict[str, Any]:
    apply_model = getattr(llm, "apply_model", None)
    if not callable(apply_model):
        raise RuntimeError("vLLM LLM.apply_model is required for runtime NIXL smoke")

    worker_kwargs = {
        "model_name": model_name,
        "model_version": model_version,
        "preferred_tensor_name": preferred_tensor_name,
        "module_path": "llm.apply_model.worker_model",
    }
    results = apply_model(partial(_inspect_vllm_worker_model, kwargs=worker_kwargs))
    if not results:
        raise RuntimeError("vLLM apply_model returned no tensor-inspection results")
    result = results[0]
    result["worker_result_count"] = len(results)
    return result


def run_vllm_receiver_refit_from_nixl_staging_tensor(
    llm: Any,
    *,
    assembled: torch.Tensor,
    scenario: dict[str, Any],
    model_name: str,
    model_version: str,
    vllm_version: str,
    artifact_path: str | Path | None = None,
    mode: str = "live-vllm-nixl-runtime-refit-smoke",
    model_path: str = "",
    engine_start_duration_ms: float | None = None,
    initial_tensor_inspection_duration_ms: float | None = None,
    staging_to_apply_model_payload_duration_ms: float | None = None,
    nixl_reads: list[dict[str, Any]] | None = None,
    nixl_metrics: dict[str, Any] | None = None,
    distributed: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Install a NIXL-assembled staging tensor into vLLM via apply_model."""

    apply_model = getattr(llm, "apply_model", None)
    if not callable(apply_model):
        raise RuntimeError("vLLM LLM.apply_model is required for runtime NIXL smoke")

    nixl_reads = list(nixl_reads or [])
    nixl_metrics = dict(nixl_metrics or {})
    distributed = dict(distributed or {})

    tensor_name = str(scenario["target_tensor_name"])
    target_shape = tuple(int(dim) for dim in scenario["target_shape"])
    dtype_obj = _torch_dtype_from_name(str(scenario["target_dtype"]))
    expected = _replacement_tensor(
        target_shape, dtype=dtype_obj, device=assembled.device
    )
    staging_allclose = _tensor_close(assembled, expected)
    staging_checksum = _tensor_checksum(assembled)
    expected_checksum = _tensor_checksum(expected)
    staging_checksum_matches = staging_checksum == expected_checksum

    payload_copy_start = time.perf_counter()
    assembled_payload_cpu = assembled.detach().to("cpu").contiguous()
    payload_copy_duration_ms = (time.perf_counter() - payload_copy_start) * 1000
    if staging_to_apply_model_payload_duration_ms is not None:
        payload_copy_duration_ms = staging_to_apply_model_payload_duration_ms

    worker_kwargs = {
        "tensor_name": tensor_name,
        "module_path": str(scenario.get("module_path", "llm.apply_model.worker_model")),
        "assembled_payload_cpu": assembled_payload_cpu,
    }
    apply_start = time.perf_counter()
    worker_results = apply_model(
        partial(_apply_vllm_nixl_payload_on_worker_model, kwargs=worker_kwargs)
    )
    apply_model_duration_ms = (time.perf_counter() - apply_start) * 1000
    if not worker_results:
        raise RuntimeError("vLLM apply_model returned no install results")
    worker_result = worker_results[0]

    request = SliceRequest.from_dict(scenario["request"])
    owners = [SliceOwnership.from_dict(item) for item in scenario["source_ownerships"]]
    plans = [SegmentPlan.from_dict(item) for item in scenario["segment_plans"]]
    copied_bytes = sum(plan.bytes for plan in plans)
    source_count = len({plan.source_id for plan in plans})
    target_bytes = int(scenario.get("target_tensor_bytes", copied_bytes))

    validation = dict(worker_result["validation"])
    validation.update(
        {
            "nixl_staging_allclose": staging_allclose,
            "nixl_staging_checksum": staging_checksum,
            "nixl_staging_checksum_matches": staging_checksum_matches,
            "expected_checksum": expected_checksum,
        }
    )
    proof = {
        "vllm_imported": True,
        "vllm_apply_model_used": True,
        "vllm_worker_owned_target_tensor": True,
        "vllm_owned_target_tensor": True,
        "receiver_request_from_vllm_worker_tensor": True,
        "receiver_request_from_runtime_owned_tensor": True,
        "receiver_installed_into_vllm_worker_owned_tensor": bool(
            worker_result["proof"]["receiver_installed_into_vllm_owned_tensor"]
        ),
        "receiver_installed_into_runtime_owned_tensor": bool(
            worker_result["proof"]["receiver_installed_into_runtime_owned_tensor"]
        ),
        "runtime_imported": True,
        "runtime_owned_target_tensor": True,
        "real_runtime_engine_used": True,
        "actual_nixl_reads_used": True,
        "nixl_reads_land_at_segment_offsets": staging_allclose,
        "nixl_reads_land_into_staging_tensor": staging_allclose,
        "nixl_reads_land_directly_in_runtime_tensor": False,
        "runtime_update_from_nixl_staging_tensor": bool(
            worker_result["proof"]["runtime_update_from_nixl_staging_tensor"]
        ),
        "runtime_update_payload_copied_through_apply_model": True,
        "source_rank_owned_trainer_tensors_used": True,
        "trainer_like_source_processes_used": True,
        "real_trainer_process_used": False,
        "real_training_loop_used": False,
        "synthetic_trainer_payloads_used": False,
        "synthetic_source_values_used": True,
        "target_slice_spans_multiple_trainers": source_count >= 2,
        "checksum_gate": bool(validation["checksum_matches"]),
        "staging_checksum_gate": staging_checksum_matches,
        "allclose": bool(validation["allclose"]),
        "restored_original_tensor": bool(validation["restored_original"]),
        "trainer_full_all_gather_used": False,
        "trainer_side_inference_layout_conversion_used": False,
        "host_side_torch_cat_used": False,
        "torch_distributed_data_transfer_used": False,
    }

    result = {
        "schema_version": 1,
        "result": (
            "pass"
            if staging_allclose
            and staging_checksum_matches
            and validation["allclose"]
            and validation["checksum_matches"]
            and validation["payload_allclose"]
            and validation["payload_checksum_matches"]
            and validation["restored_original"]
            else "fail"
        ),
        "mode": mode,
        "runtime_framework": "vllm",
        "framework_version": vllm_version,
        "vllm_version": vllm_version,
        "model_name": model_name,
        "model_version": model_version,
        "model_path": model_path,
        "module_class": worker_result["module_class"],
        "module_path": worker_result["module_path"],
        "target_tensor_name": tensor_name,
        "target_key": request.target_id or tensor_name,
        "target_shape": list(target_shape),
        "target_dtype": str(scenario["target_dtype"]),
        "target_device": worker_result["target_device"],
        "nixl_staging_shape": list(_shape(assembled)),
        "nixl_staging_dtype": _torch_dtype_name(assembled.dtype),
        "nixl_staging_device": str(assembled.device),
        "request": request.to_dict(),
        "source_ownerships": [owner.to_dict() for owner in owners],
        "segment_plans": [plan.to_dict() for plan in plans],
        "proof": proof,
        "validation": validation,
        "metrics": {
            "engine_start_duration_ms": engine_start_duration_ms,
            "initial_tensor_inspection_duration_ms": initial_tensor_inspection_duration_ms,
            "staging_to_apply_model_payload_duration_ms": payload_copy_duration_ms,
            "apply_model_install_duration_ms": apply_model_duration_ms,
            "activation_install_duration_ms": worker_result["metrics"][
                "activation_install_duration_ms"
            ],
            "restore_duration_ms": worker_result["metrics"]["restore_duration_ms"],
            "trainer_to_inference_bytes": copied_bytes,
            "inference_side_fanout_bytes": 0,
            "redundant_cross_boundary_factor": (
                copied_bytes / target_bytes if target_bytes else 0.0
            ),
            "segment_count": len(plans),
            "source_count_per_target_tensor": {tensor_name: source_count},
            "target_tensor_bytes": target_bytes,
            **nixl_metrics,
        },
        "distributed": distributed,
        "nixl": {"reads": nixl_reads},
        "worker_result_count": len(worker_results),
        "runtime_module_discovery_path": "vllm.apply_model",
    }
    if artifact_path is not None:
        _write_artifact(result, artifact_path)
    if result["result"] != "pass":
        raise RuntimeError("vLLM NIXL runtime receiver smoke failed: " f"{result!r}")
    return result


def run_vllm_nixl_runtime_refit_distributed(
    artifact_path: str | Path,
    *,
    model_path: str = "",
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
    preferred_tensor_name: str = DEFAULT_TENSOR_NAME,
    dtype: str = "bfloat16",
    max_model_len: int = 64,
    gpu_memory_utilization: float = 0.1,
    distributed_executor_backend: str = "uni",
    target_rank: int = DEFAULT_TARGET_RANK,
) -> dict[str, Any] | None:
    """Run the 2-source-rank plus live-vLLM-target NIXL proof."""

    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the vLLM NIXL runtime smoke")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 3:
        raise RuntimeError("vLLM NIXL runtime smoke requires exactly 3 ranks")
    if target_rank != DEFAULT_TARGET_RANK:
        raise RuntimeError("vLLM NIXL runtime smoke currently expects target rank 2")

    device_index, gpu_reuse_used = select_cuda_device(local_rank)
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")

    llm = None
    scenario: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    dist_initialized = False
    try:
        if rank == target_rank:
            print(
                "[mx-vllm-nixl-runtime] target rank starting vLLM before "
                "outer gloo init",
                flush=True,
            )
            resolved_model_path = model_path
            if not resolved_model_path:
                resolved_model_path = str(
                    create_tiny_qwen2_checkpoint(
                        Path(tempfile.mkdtemp(prefix="mx-vllm-nixl-runtime-model-"))
                    )
                )
            import vllm

            vllm_version = getattr(vllm, "__version__", "unknown")
            engine_start = time.perf_counter()
            saved_engine_env = _clear_torchrun_env_for_runtime_engine()
            try:
                llm = load_vllm_llm(
                    model_path=resolved_model_path,
                    dtype=dtype,
                    max_model_len=max_model_len,
                    gpu_memory_utilization=gpu_memory_utilization,
                    distributed_executor_backend=distributed_executor_backend,
                )
            finally:
                _restore_torchrun_env(saved_engine_env)
            engine_start_duration_ms = (time.perf_counter() - engine_start) * 1000
            inspect_start = time.perf_counter()
            scenario = inspect_vllm_runtime_nixl_scenario(
                llm,
                model_name=model_name,
                model_version=model_version,
                preferred_tensor_name=preferred_tensor_name,
            )
            scenario.update(
                {
                    "model_path": resolved_model_path,
                    "vllm_version": vllm_version,
                    "engine_start_duration_ms": engine_start_duration_ms,
                    "initial_tensor_inspection_duration_ms": (
                        time.perf_counter() - inspect_start
                    )
                    * 1000,
                }
            )
            print(
                "[mx-vllm-nixl-runtime] target rank finished vLLM startup; "
                "joining outer gloo init",
                flush=True,
            )
        else:
            print(
                f"[mx-vllm-nixl-runtime] source rank {rank} waiting in "
                "outer gloo init",
                flush=True,
            )

        timeout_minutes = int(os.environ.get("MX_REFIT_DIST_TIMEOUT_MINUTES", "10"))
        dist.init_process_group(
            backend="gloo",
            timeout=timedelta(minutes=timeout_minutes),
        )
        dist_initialized = True
        apply_nixl_ucx_pin(device_index)

        holder = [scenario]
        dist.broadcast_object_list(holder, src=target_rank)
        scenario = holder[0]
        if scenario is None:
            raise RuntimeError("target rank did not publish vLLM NIXL scenario")

        request = SliceRequest.from_dict(scenario["request"])
        owners = [
            SliceOwnership.from_dict(item) for item in scenario["source_ownerships"]
        ]
        plans = [SegmentPlan.from_dict(item) for item in scenario["segment_plans"]]
        owner_by_rank = {
            owner.worker_rank: owner
            for owner in owners
            if owner.worker_rank is not None
        }
        dtype_obj = _torch_dtype_from_name(str(scenario["target_dtype"]))
        shape = tuple(int(dim) for dim in scenario["target_shape"])

        adapter = NixlAdapter(f"mx-vllm-nixl-runtime-rank{rank}")
        source_tensor = None
        target_staging = None
        registered_bytes = 0
        register_start = time.perf_counter()
        if rank in owner_by_rank:
            source_tensor = materialize_vllm_nixl_source_tensor(
                owner_by_rank[rank],
                dtype=dtype_obj,
                device=device,
            )
            adapter.register_tensor(source_tensor)
            registered_bytes = int(source_tensor.numel() * source_tensor.element_size())
        elif rank == target_rank:
            target_staging = torch.full(
                shape,
                float("nan"),
                device=device,
                dtype=dtype_obj,
            )
            adapter.register_tensor(target_staging)
            registered_bytes = int(
                target_staging.numel() * target_staging.element_size()
            )
        torch.cuda.synchronize(device)
        registration_duration_ms = (time.perf_counter() - register_start) * 1000

        metadata_start = time.perf_counter()
        metadata = adapter.metadata_bytes()
        metadata_duration_ms = (time.perf_counter() - metadata_start) * 1000

        local_info: dict[str, Any] = {
            "rank": rank,
            "agent_name": adapter.agent_name,
            "cuda_device": device_index,
            "gpu_reuse_used": gpu_reuse_used,
            "metadata": metadata,
            "metadata_bytes": len(metadata),
            "registered_bytes": registered_bytes,
            "registration_duration_ms": registration_duration_ms,
            "metadata_duration_ms": metadata_duration_ms,
            "config_errors": adapter.config_errors,
            "role": "target" if rank == target_rank else "source",
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
        elif rank == target_rank:
            assert target_staging is not None
            local_info.update(
                {
                    "target_id": request.target_id,
                    "addr": int(target_staging.data_ptr()),
                    "device_id": int(target_staging.get_device()),
                    "tensor_bytes": int(
                        target_staging.numel() * target_staging.element_size()
                    ),
                }
            )

        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_info)

        if rank != target_rank:
            dist.barrier()
            return None

        assert llm is not None
        assert target_staging is not None
        sources_by_id = {
            info["source_id"]: info
            for info in gathered
            if info is not None and info.get("role") == "source"
        }
        add_remote_timings: dict[str, float] = {}
        remote_agent_names: dict[str, str] = {}
        for source_id, info in sorted(sources_by_id.items()):
            add_start = time.perf_counter()
            remote_agent_names[source_id] = adapter.add_remote_agent(info["metadata"])
            add_remote_timings[source_id] = (time.perf_counter() - add_start) * 1000

        transfer_start = time.perf_counter()
        reads = read_segment_groups(
            adapter=adapter,
            target=target_staging,
            sources_by_id=sources_by_id,
            remote_agent_names=remote_agent_names,
            plans=plans,
            timeout_seconds=120,
        )
        torch.cuda.synchronize(device)
        raw_nixl_read_duration_ms = (time.perf_counter() - transfer_start) * 1000

        copied_bytes = sum(read["bytes"] for read in reads)
        nixl_metrics = {
            "raw_nixl_read_duration_ms": raw_nixl_read_duration_ms,
            "nixl_registration_duration_ms": sum(
                info["registration_duration_ms"]
                for info in gathered
                if info is not None
            ),
            "nixl_metadata_duration_ms": sum(
                info["metadata_duration_ms"] for info in gathered if info is not None
            ),
            "nixl_add_remote_agent_duration_ms": sum(add_remote_timings.values()),
            "nixl_prep_duration_ms": sum(read["prep_duration_ms"] for read in reads),
            "nixl_read_group_count": len(reads),
            "successful_nixl_source_count": len({read["source_id"] for read in reads}),
            "trainer_to_inference_bytes": copied_bytes,
        }
        distributed = {
            "backend": "nixl-read+gloo-control+vllm-apply-model-update",
            "world_size": world_size,
            "target_rank": target_rank,
            "source_ranks": list(SOURCE_RANKS),
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
        result = run_vllm_receiver_refit_from_nixl_staging_tensor(
            llm,
            assembled=target_staging,
            scenario=scenario,
            model_name=model_name,
            model_version=model_version,
            vllm_version=str(scenario["vllm_version"]),
            artifact_path=artifact_path,
            model_path=str(scenario["model_path"]),
            engine_start_duration_ms=float(scenario["engine_start_duration_ms"]),
            initial_tensor_inspection_duration_ms=float(
                scenario["initial_tensor_inspection_duration_ms"]
            ),
            nixl_reads=reads,
            nixl_metrics=nixl_metrics,
            distributed=distributed,
        )
        result["nixl"].update(
            {
                "target_agent_name": adapter.agent_name,
                "nixl_backends": adapter.backends,
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
            }
        )
        _write_artifact(result, artifact_path)
        dist.barrier()
        return result

    finally:
        if llm is not None:
            shutdown = getattr(llm, "shutdown", None)
            if callable(shutdown):
                shutdown()
        if dist_initialized:
            dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--preferred-tensor-name", default=DEFAULT_TENSOR_NAME)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.1)
    parser.add_argument("--distributed-executor-backend", default="uni")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_vllm_nixl_runtime_refit_distributed(
        args.artifact_path,
        model_path=args.model_path,
        model_name=args.model_name,
        model_version=args.model_version,
        preferred_tensor_name=args.preferred_tensor_name,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        distributed_executor_backend=args.distributed_executor_backend,
    )
    if result is not None:
        print(
            "MX_VLLM_NIXL_RUNTIME_REFIT_SMOKE "
            f"result={result['result']} tensor={result['target_tensor_name']} "
            f"shape={result['target_shape']} "
            f"nixl_allclose={result['validation']['nixl_staging_allclose']} "
            f"runtime_allclose={result['validation']['allclose']} "
            f"checksum_matches={result['validation']['checksum_matches']}",
            flush=True,
        )
        print(json.dumps(result, default=_json_default, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
