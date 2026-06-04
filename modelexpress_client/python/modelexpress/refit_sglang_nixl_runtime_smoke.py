# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live SGLang receiver refit driven by NIXL-read trainer source tensors.

This proof bridges the previous Level-2 NIXL segment-read path and the Level-4
SGLang receiver-owned weight path. Source ranks own GPU tensors, publish their
NIXL descriptors through torchrun control metadata, and the target rank issues
one-sided NIXL READs into a preallocated staging tensor. The target then installs
that assembled tensor through ``sglang.Engine.update_weights_from_tensor`` and
validates the engine-owned weight by checksum/allclose.

The source values come from a source-rank optimizer-step publisher over a
small synthetic objective, not from the older static replacement formula. The
artifact therefore distinguishes source-rank-owned NIXL payloads from a full RL
training loop and from direct zero-copy into framework-owned storage.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
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
from .refit_trainer_step import (
    materialize_trainer_step_source_tensor,
    trainer_step_replacement_tensor,
    trainer_step_source_provenance,
)
from .refit_sglang_receiver_smoke import (
    _coerce_sglang_weight_to_tensor,
    _select_sglang_engine_weight,
    _sglang_update_succeeded,
    create_tiny_llama_checkpoint,
    detect_sglang_version,
    load_sglang_engine,
)
from .refit_vllm_receiver_smoke import (
    _nbytes,
    _shape,
    _tensor_checksum,
    _tensor_close,
    _tensor_max_abs_error,
    _torch_dtype_name,
    _write_artifact,
)
from .resharding import (
    SegmentPlan,
    SliceOwnership,
    SliceRequest,
    TensorRange,
    plan_segments,
)
from .resharding_receiver import build_receiver_requests_from_runtime_tensors

DEFAULT_MODEL_NAME = "mx-live-sglang-nixl-runtime-smoke"
DEFAULT_MODEL_VERSION = "trainer-step-live-sglang-nixl"
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
    return trainer_step_replacement_tensor(shape, dtype=dtype, device=device)


def _range_slices(tensor_range: TensorRange) -> tuple[slice, ...]:
    return tuple(slice(start, end) for start, end in tensor_range)


def build_sglang_nixl_source_ownerships(
    *,
    tensor_name: str,
    shape: tuple[int, ...],
    dtype_name: str,
    element_size_bytes: int,
    model_name: str,
    model_version: str,
) -> list[SliceOwnership]:
    """Return two trainer-rank ownership records for a runtime tensor shape."""

    if len(shape) < 2:
        raise ValueError("SGLang NIXL runtime smoke requires a rank-2 tensor")
    rows = int(shape[0])
    if rows < 2:
        raise ValueError("SGLang NIXL runtime smoke requires at least two rows")

    split = max(1, rows // 2)
    trailing_ranges = tuple((0, int(dim)) for dim in shape[1:])
    common_tags: dict[str, str | int | bool] = {
        "trainer_layout": "fsdp-row-shard-poc",
        "storage_layout": "row-major",
        "source_tensor_owner": "torchrun-trainer-rank",
        "runtime_refit_target": "sglang.Engine",
        "trainer_update_source": "torch.optim.SGD-step-publisher",
        "optimizer_step_publisher": True,
        "synthetic_training_objective": True,
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
            source_lease="trainer-rank0-live-sglang-nixl-refit",
            nixl_descriptor_id="trainer-rank0-live-sglang-nixl-refit",
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
            source_lease="trainer-rank1-live-sglang-nixl-refit",
            nixl_descriptor_id="trainer-rank1-live-sglang-nixl-refit",
            layout_tags=dict(common_tags),
            element_size_bytes=element_size_bytes,
        ),
    ]


def build_sglang_runtime_nixl_plan(
    original: torch.Tensor,
    *,
    tensor_name: str,
    model_name: str,
    model_version: str,
) -> tuple[SliceRequest, list[SliceOwnership], list[SegmentPlan]]:
    """Build receiver request and source plans from a live SGLang weight."""

    request = build_receiver_requests_from_runtime_tensors(
        {tensor_name: original},
        model_name=model_name,
        model_version=model_version,
        runtime_framework="sglang",
        target_id_prefix="sglang",
        layout_tags_by_tensor={
            tensor_name: {
                "runtime_module_path": "sglang.Engine",
                "runtime_tensor_name": tensor_name,
                "sglang_module_path": "sglang.Engine",
                "sglang_tensor_name": tensor_name,
                "runtime_lifecycle": "engine-update-weights-from-nixl-staging",
            }
        },
    )[0]
    owners = build_sglang_nixl_source_ownerships(
        tensor_name=tensor_name,
        shape=_shape(original),
        dtype_name=_torch_dtype_name(original.dtype),
        element_size_bytes=int(original.element_size()),
        model_name=model_name,
        model_version=model_version,
    )
    return request, owners, plan_segments(owners, [request])


def materialize_sglang_nixl_source_tensor(
    owner: SliceOwnership,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create the rank-owned trainer payload for one source range."""

    return materialize_trainer_step_source_tensor(
        owner,
        dtype=dtype,
        device=device,
    )


def run_sglang_receiver_refit_from_nixl_staging_tensor(
    engine: Any,
    *,
    tensor_name: str,
    original: torch.Tensor,
    assembled: torch.Tensor,
    request: SliceRequest,
    source_ownerships: list[SliceOwnership],
    segment_plans: list[SegmentPlan],
    model_name: str,
    model_version: str,
    sglang_version: str = "",
    artifact_path: str | Path | None = None,
    mode: str = "live-sglang-nixl-runtime-refit-smoke",
    model_path: str = "",
    engine_start_duration_ms: float | None = None,
    initial_weight_fetch_duration_ms: float | None = None,
    nixl_reads: list[dict[str, Any]] | None = None,
    nixl_metrics: dict[str, Any] | None = None,
    distributed: dict[str, Any] | None = None,
    actual_nixl_reads_used: bool = True,
) -> dict[str, Any]:
    """Install a NIXL-assembled staging tensor into a live SGLang Engine."""

    if not sglang_version:
        sglang_version, _ = detect_sglang_version()

    nixl_reads = list(nixl_reads or [])
    nixl_metrics = dict(nixl_metrics or {})
    distributed = dict(distributed or {})

    expected = _replacement_tensor(
        _shape(original),
        dtype=original.dtype,
        device=original.device,
    )
    assembled_for_engine = (
        assembled.detach()
        .to(
            device=original.device,
            dtype=original.dtype,
        )
        .contiguous()
    )
    staging_allclose = _tensor_close(assembled_for_engine, expected)
    staging_checksum = _tensor_checksum(assembled_for_engine)
    expected_checksum = _tensor_checksum(expected)
    staging_checksum_matches = staging_checksum == expected_checksum

    update_start = time.perf_counter()
    update_result = engine.update_weights_from_tensor(
        [(tensor_name, assembled_for_engine)],
        load_format="direct",
        flush_cache=True,
    )
    update_duration_ms = (time.perf_counter() - update_start) * 1000
    update_success, update_message = _sglang_update_succeeded(update_result)

    validate_fetch_start = time.perf_counter()
    updated = _coerce_sglang_weight_to_tensor(
        engine.get_weights_by_name(tensor_name, truncate_size=10_000_000),
        name=tensor_name,
    )
    validate_fetch_duration_ms = (time.perf_counter() - validate_fetch_start) * 1000
    allclose = _tensor_close(updated, expected)
    checksum = _tensor_checksum(updated)
    checksum_matches = checksum == expected_checksum
    max_abs_error = _tensor_max_abs_error(updated, expected)

    restore_start = time.perf_counter()
    restore_result = engine.update_weights_from_tensor(
        [(tensor_name, original)],
        load_format="direct",
        flush_cache=True,
    )
    restore_duration_ms = (time.perf_counter() - restore_start) * 1000
    restore_success, restore_message = _sglang_update_succeeded(restore_result)
    restored = _coerce_sglang_weight_to_tensor(
        engine.get_weights_by_name(tensor_name, truncate_size=10_000_000),
        name=tensor_name,
    )
    restored_original = _tensor_close(restored, original)

    source_count = len({plan.source_id for plan in segment_plans})
    copied_bytes = sum(plan.bytes for plan in segment_plans)
    target_bytes = _nbytes(original)
    proof = {
        "sglang_imported": True,
        "sglang_engine_started": True,
        "sglang_engine_get_weights_by_name_used": True,
        "sglang_engine_update_weights_from_tensor_used": True,
        "sglang_engine_owned_target_tensor": True,
        "sglang_owned_target_tensor": True,
        "receiver_request_from_sglang_engine_weight": True,
        "receiver_request_from_runtime_owned_tensor": True,
        "receiver_installed_into_sglang_engine_owned_tensor": allclose,
        "receiver_installed_into_sglang_owned_tensor": allclose,
        "receiver_installed_into_runtime_owned_tensor": allclose,
        "runtime_imported": True,
        "runtime_owned_target_tensor": True,
        "real_runtime_engine_used": True,
        "actual_nixl_reads_used": bool(actual_nixl_reads_used),
        "nixl_reads_land_at_segment_offsets": staging_allclose,
        "nixl_reads_land_into_staging_tensor": staging_allclose,
        "nixl_reads_land_directly_in_runtime_tensor": False,
        "runtime_update_from_nixl_staging_tensor": allclose,
        "source_rank_owned_trainer_tensors_used": True,
        "trainer_like_source_processes_used": True,
        "real_trainer_process_used": False,
        "trainer_optimizer_step_publisher_used": True,
        "trainer_owned_parameter_tensor_used": True,
        "real_training_loop_used": False,
        "real_rl_training_loop_used": False,
        "synthetic_training_objective_used": True,
        "synthetic_trainer_payloads_used": False,
        "synthetic_source_values_used": False,
        "static_replacement_formula_source_values_used": False,
        "target_slice_spans_multiple_trainers": source_count >= 2,
        "checksum_gate": checksum_matches,
        "staging_checksum_gate": staging_checksum_matches,
        "allclose": allclose,
        "restored_original_tensor": restored_original,
        "trainer_full_all_gather_used": False,
        "trainer_side_inference_layout_conversion_used": False,
        "host_side_torch_cat_used": False,
        "torch_distributed_data_transfer_used": False,
    }

    result = {
        "schema_version": 1,
        "result": (
            "pass"
            if update_success
            and staging_allclose
            and staging_checksum_matches
            and allclose
            and checksum_matches
            and restore_success
            and restored_original
            else "fail"
        ),
        "mode": mode,
        "runtime_framework": "sglang",
        "framework_version": sglang_version,
        "sglang_version": sglang_version,
        "model_name": model_name,
        "model_version": model_version,
        "model_path": model_path,
        "engine_class": type(engine).__name__,
        "module_class": "sglang.Engine",
        "module_path": "sglang.Engine",
        "target_tensor_name": tensor_name,
        "target_key": request.target_id or tensor_name,
        "target_shape": list(_shape(original)),
        "target_dtype": _torch_dtype_name(original.dtype),
        "target_device": str(original.device),
        "nixl_staging_shape": list(_shape(assembled)),
        "nixl_staging_dtype": _torch_dtype_name(assembled.dtype),
        "nixl_staging_device": str(assembled.device),
        "request": request.to_dict(),
        "source_ownerships": [owner.to_dict() for owner in source_ownerships],
        "segment_plans": [plan.to_dict() for plan in segment_plans],
        "proof": proof,
        "validation": {
            "nixl_staging_allclose": staging_allclose,
            "nixl_staging_checksum": staging_checksum,
            "nixl_staging_checksum_matches": staging_checksum_matches,
            "allclose": allclose,
            "checksum": checksum,
            "expected_checksum": expected_checksum,
            "checksum_matches": checksum_matches,
            "max_abs_error": max_abs_error,
            "original_checksum": _tensor_checksum(original),
            "restored_checksum": _tensor_checksum(restored),
            "restored_original": restored_original,
            "update_success": update_success,
            "update_message": update_message,
            "restore_success": restore_success,
            "restore_message": restore_message,
        },
        "metrics": {
            "engine_start_duration_ms": engine_start_duration_ms,
            "initial_weight_fetch_duration_ms": initial_weight_fetch_duration_ms,
            "activation_install_duration_ms": update_duration_ms,
            "validate_weight_fetch_duration_ms": validate_fetch_duration_ms,
            "restore_duration_ms": restore_duration_ms,
            "trainer_to_inference_bytes": copied_bytes,
            "inference_side_fanout_bytes": 0,
            "redundant_cross_boundary_factor": (
                copied_bytes / target_bytes if target_bytes else 0.0
            ),
            "segment_count": len(segment_plans),
            "source_count_per_target_tensor": {tensor_name: source_count},
            "target_tensor_bytes": target_bytes,
            **nixl_metrics,
        },
        "distributed": distributed,
        "nixl": {
            "reads": nixl_reads,
        },
        "trainer_source_update": trainer_step_source_provenance(),
    }
    if artifact_path is not None:
        _write_artifact(result, artifact_path)
    if result["result"] != "pass":
        raise RuntimeError(f"SGLang NIXL runtime receiver smoke failed: {result!r}")
    return result


def _scenario_payload(
    *,
    tensor_name: str,
    original: torch.Tensor,
    request: SliceRequest,
    owners: list[SliceOwnership],
    plans: list[SegmentPlan],
    model_path: str,
    sglang_version: str,
    engine_start_duration_ms: float,
    initial_weight_fetch_duration_ms: float,
) -> dict[str, Any]:
    return {
        "tensor_name": tensor_name,
        "target_shape": list(_shape(original)),
        "target_dtype": _torch_dtype_name(original.dtype),
        "request": request.to_dict(),
        "source_ownerships": [owner.to_dict() for owner in owners],
        "segment_plans": [plan.to_dict() for plan in plans],
        "model_path": model_path,
        "sglang_version": sglang_version,
        "engine_start_duration_ms": engine_start_duration_ms,
        "initial_weight_fetch_duration_ms": initial_weight_fetch_duration_ms,
    }


def run_sglang_nixl_runtime_refit_distributed(
    artifact_path: str | Path,
    *,
    model_path: str = "",
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
    preferred_tensor_name: str = DEFAULT_TENSOR_NAME,
    dtype: str = "bfloat16",
    context_length: int = 64,
    mem_fraction_static: float = 0.2,
    disable_cuda_graph: bool = True,
    disable_piecewise_cuda_graph: bool = True,
    log_level: str = "info",
    target_rank: int = DEFAULT_TARGET_RANK,
) -> dict[str, Any] | None:
    """Run the 2-source-rank plus live-SGLang-target NIXL proof."""

    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the SGLang NIXL runtime smoke")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 3:
        raise RuntimeError("SGLang NIXL runtime smoke requires exactly 3 ranks")
    if target_rank != DEFAULT_TARGET_RANK:
        raise RuntimeError("SGLang NIXL runtime smoke currently expects target rank 2")

    device_index, gpu_reuse_used = select_cuda_device(local_rank)
    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")

    engine = None
    original: torch.Tensor | None = None
    scenario: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    dist_initialized = False
    try:
        if rank == target_rank:
            print(
                "[mx-sglang-nixl-runtime] target rank starting SGLang before "
                "outer gloo init",
                flush=True,
            )
            resolved_model_path = model_path
            if not resolved_model_path:
                resolved_model_path = str(
                    create_tiny_llama_checkpoint(
                        Path(tempfile.mkdtemp(prefix="mx-sglang-nixl-runtime-model-"))
                    )
                )
            sglang_version, _ = detect_sglang_version()
            engine_start = time.perf_counter()
            saved_engine_env = _clear_torchrun_env_for_runtime_engine()
            try:
                engine = load_sglang_engine(
                    model_path=resolved_model_path,
                    dtype=dtype,
                    context_length=context_length,
                    mem_fraction_static=mem_fraction_static,
                    disable_cuda_graph=disable_cuda_graph,
                    disable_piecewise_cuda_graph=disable_piecewise_cuda_graph,
                    log_level=log_level,
                )
            finally:
                _restore_torchrun_env(saved_engine_env)
            engine_start_duration_ms = (time.perf_counter() - engine_start) * 1000
            fetch_start = time.perf_counter()
            tensor_name, original = _select_sglang_engine_weight(
                engine,
                preferred_tensor_name=preferred_tensor_name,
            )
            initial_weight_fetch_duration_ms = (
                time.perf_counter() - fetch_start
            ) * 1000
            request, owners, plans = build_sglang_runtime_nixl_plan(
                original,
                tensor_name=tensor_name,
                model_name=model_name,
                model_version=model_version,
            )
            scenario = _scenario_payload(
                tensor_name=tensor_name,
                original=original,
                request=request,
                owners=owners,
                plans=plans,
                model_path=resolved_model_path,
                sglang_version=sglang_version,
                engine_start_duration_ms=engine_start_duration_ms,
                initial_weight_fetch_duration_ms=initial_weight_fetch_duration_ms,
            )
            print(
                "[mx-sglang-nixl-runtime] target rank finished SGLang startup; "
                "joining outer gloo init",
                flush=True,
            )
        else:
            print(
                f"[mx-sglang-nixl-runtime] source rank {rank} waiting in "
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
            raise RuntimeError("target rank did not publish SGLang NIXL scenario")

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

        adapter = NixlAdapter(f"mx-sglang-nixl-runtime-rank{rank}")
        source_tensor = None
        target_staging = None
        registered_bytes = 0
        register_start = time.perf_counter()
        if rank in owner_by_rank:
            source_tensor = materialize_sglang_nixl_source_tensor(
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
                    "source_payload_provenance": trainer_step_source_provenance(),
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

        assert engine is not None
        assert original is not None
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
            "backend": "nixl-read+gloo-control+sglang-engine-update",
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
        result = run_sglang_receiver_refit_from_nixl_staging_tensor(
            engine,
            tensor_name=str(scenario["tensor_name"]),
            original=original,
            assembled=target_staging,
            request=request,
            source_ownerships=owners,
            segment_plans=plans,
            model_name=model_name,
            model_version=model_version,
            sglang_version=str(scenario["sglang_version"]),
            artifact_path=artifact_path,
            model_path=str(scenario["model_path"]),
            engine_start_duration_ms=float(scenario["engine_start_duration_ms"]),
            initial_weight_fetch_duration_ms=float(
                scenario["initial_weight_fetch_duration_ms"]
            ),
            nixl_reads=reads,
            nixl_metrics=nixl_metrics,
            distributed=distributed,
            actual_nixl_reads_used=True,
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
                        "source_payload_provenance": info.get(
                            "source_payload_provenance", {}
                        ),
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
        if engine is not None:
            engine.shutdown()
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
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--mem-fraction-static", type=float, default=0.2)
    parser.add_argument("--enable-cuda-graph", action="store_true")
    parser.add_argument("--enable-piecewise-cuda-graph", action="store_true")
    parser.add_argument("--log-level", default="info")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_sglang_nixl_runtime_refit_distributed(
        args.artifact_path,
        model_path=args.model_path,
        model_name=args.model_name,
        model_version=args.model_version,
        preferred_tensor_name=args.preferred_tensor_name,
        dtype=args.dtype,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
        disable_cuda_graph=not args.enable_cuda_graph,
        disable_piecewise_cuda_graph=not args.enable_piecewise_cuda_graph,
        log_level=args.log_level,
    )
    if result is not None:
        print(
            "MX_SGLANG_NIXL_RUNTIME_REFIT_SMOKE "
            f"result={result['result']} tensor={result['target_tensor_name']} "
            f"shape={result['target_shape']} "
            f"nixl_allclose={result['validation']['nixl_staging_allclose']} "
            f"runtime_allclose={result['validation']['allclose']} "
            f"checksum_matches={result['validation']['checksum_matches']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
