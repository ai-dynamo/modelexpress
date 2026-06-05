# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-tensor runtime refit smoke for MX resharding contracts.

These smokes are intentionally narrower than the live vLLM/SGLang NIXL bridges:
they do not start a runtime engine and they do not issue NIXL reads. They prove
the next contract boundaries those bridges need before full-model refit:

* one runtime refit transaction can request, plan, install, validate, and roll
  back multiple runtime-owned tensors using source-published slice ownership
  metadata.
* one NIXL-style staging bundle can accept planned segment payloads for several
  target tensors, validate those staging tensors, install from staging into the
  runtime transaction, and roll back.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from .refit_trainer_step import (
    DEFAULT_TRAINER_LR,
    DEFAULT_TRAINER_STEP_COUNT,
    publish_trainer_step_source,
    trainer_step_replacement_tensor,
    trainer_step_source_provenance,
    trainer_update_parameters_from_ownerships,
)
from .refit_vllm_receiver_smoke import (
    _json_default,
    _nbytes,
    _shape,
    _tensor_checksum,
    _tensor_close,
    _tensor_max_abs_error,
    _torch_dtype_name,
    _write_artifact,
)
from .resharding import SegmentPlan, SliceOwnership, SliceRequest, plan_segments
from .resharding_receiver import (
    begin_runtime_refit_transaction,
    install_segment_payloads_into_runtime_tensors,
)

DEFAULT_MODEL_NAME = "mx-runtime-multitensor-refit-smoke"
DEFAULT_MODEL_VERSION = "trainer-step-runtime-multitensor"
DEFAULT_TENSOR_SHAPES = {
    "model.embed_tokens.weight": (8, 4),
    "lm_head.weight": (6, 4),
}


def build_runtime_multitensor_source_ownerships(
    tensors: Mapping[str, torch.Tensor],
    *,
    runtime_framework: str,
    model_name: str,
    model_version: str,
) -> list[SliceOwnership]:
    """Return two row-shard trainer ownerships for every runtime tensor."""

    owners: list[SliceOwnership] = []
    for tensor_name, tensor in tensors.items():
        shape = _shape(tensor)
        if len(shape) < 2:
            raise ValueError(
                f"multi-tensor runtime smoke requires rank-2 tensors; "
                f"{tensor_name!r} has shape {shape}"
            )
        rows = int(shape[0])
        if rows < 2:
            raise ValueError(
                f"multi-tensor runtime smoke requires at least two rows; "
                f"{tensor_name!r} has shape {shape}"
            )
        split = max(1, rows // 2)
        trailing_ranges = tuple((0, int(dim)) for dim in shape[1:])
        common_tags: dict[str, str | int | bool] = {
            "trainer_layout": "fsdp-row-shard-poc",
            "storage_layout": "row-major",
            "source_tensor_owner": "torchrun-trainer-rank",
            "runtime_framework": runtime_framework,
            "runtime_refit_scope": "multi-tensor-transaction",
            "trainer_update_source": "torch.optim.SGD-step-publisher",
            "optimizer_step_publisher": True,
            "synthetic_training_objective": True,
        }
        owners.extend(
            [
                SliceOwnership(
                    model_name=model_name,
                    model_version=model_version,
                    tensor_name=tensor_name,
                    global_shape=shape,
                    dtype=_torch_dtype_name(tensor.dtype),
                    source_range=((0, split), *trailing_ranges),
                    worker_id="trainer-rank0-worker",
                    source_id="trainer-rank0",
                    worker_rank=0,
                    source_lease=f"trainer-rank0-{tensor_name}-multitensor-refit",
                    nixl_descriptor_id=(
                        f"trainer-rank0-{tensor_name}-multitensor-refit"
                    ),
                    layout_tags=dict(common_tags),
                    element_size_bytes=int(tensor.element_size()),
                ),
                SliceOwnership(
                    model_name=model_name,
                    model_version=model_version,
                    tensor_name=tensor_name,
                    global_shape=shape,
                    dtype=_torch_dtype_name(tensor.dtype),
                    source_range=((split, rows), *trailing_ranges),
                    worker_id="trainer-rank1-worker",
                    source_id="trainer-rank1",
                    worker_rank=1,
                    source_lease=f"trainer-rank1-{tensor_name}-multitensor-refit",
                    nixl_descriptor_id=(
                        f"trainer-rank1-{tensor_name}-multitensor-refit"
                    ),
                    layout_tags=dict(common_tags),
                    element_size_bytes=int(tensor.element_size()),
                ),
            ]
        )
    return owners


def build_runtime_multitensor_requests(
    tensors: Mapping[str, torch.Tensor],
    *,
    runtime_framework: str,
    model_name: str,
    model_version: str,
) -> list[SliceRequest]:
    """Build receiver requests for a runtime-owned tensor bundle."""

    from .resharding_receiver import build_receiver_requests_from_runtime_tensors

    layout_tags_by_tensor = {
        tensor_name: {
            "runtime_tensor_name": tensor_name,
            "runtime_refit_scope": "multi-tensor-transaction",
            "runtime_lifecycle": "receiver-owned-multitensor-install",
            f"{runtime_framework}_tensor_name": tensor_name,
        }
        for tensor_name in tensors
    }
    return build_receiver_requests_from_runtime_tensors(
        tensors,
        model_name=model_name,
        model_version=model_version,
        runtime_framework=runtime_framework,
        target_id_prefix=runtime_framework,
        layout_tags_by_tensor=layout_tags_by_tensor,
    )


def build_runtime_multitensor_plan(
    tensors: Mapping[str, torch.Tensor],
    *,
    runtime_framework: str,
    model_name: str,
    model_version: str,
) -> tuple[list[SliceRequest], list[SliceOwnership], list[SegmentPlan]]:
    """Build multi-tensor receiver requests, source ownerships, and segments."""

    requests = build_runtime_multitensor_requests(
        tensors,
        runtime_framework=runtime_framework,
        model_name=model_name,
        model_version=model_version,
    )
    owners = build_runtime_multitensor_source_ownerships(
        tensors,
        runtime_framework=runtime_framework,
        model_name=model_name,
        model_version=model_version,
    )
    return requests, owners, plan_segments(owners, requests)


def _target_key_by_tensor(requests: list[SliceRequest]) -> dict[str, str]:
    return {
        request.tensor_name: request.target_id or request.tensor_name
        for request in requests
    }


def _runtime_targets_by_request(
    tensors: Mapping[str, torch.Tensor],
    requests: list[SliceRequest],
) -> dict[str, torch.Tensor]:
    by_tensor = dict(tensors)
    return {
        request.target_id or request.tensor_name: by_tensor[request.tensor_name]
        for request in requests
    }


def _materialize_segment_payloads(
    plans: list[SegmentPlan],
    owners: list[SliceOwnership],
    *,
    tensors: Mapping[str, torch.Tensor],
) -> list[tuple[SegmentPlan, torch.Tensor]]:
    owners_by_key = {
        (owner.tensor_name, owner.source_id, owner.source_range): owner
        for owner in owners
    }
    payloads: list[tuple[SegmentPlan, torch.Tensor]] = []
    for plan in plans:
        owner = owners_by_key[(plan.tensor_name, plan.source_id, plan.source_range)]
        target_tensor = tensors[plan.tensor_name]
        publication = publish_trainer_step_source(
            owner,
            dtype=target_tensor.dtype,
            device=target_tensor.device,
        )
        payloads.append((plan, publication.tensor))
    return payloads


def _expected_by_tensor(
    tensors: Mapping[str, torch.Tensor],
    *,
    step_count: int,
    learning_rate: float,
) -> dict[str, torch.Tensor]:
    return {
        tensor_name: trainer_step_replacement_tensor(
            _shape(tensor),
            dtype=tensor.dtype,
            device=tensor.device,
            step_count=step_count,
            learning_rate=learning_rate,
        )
        for tensor_name, tensor in tensors.items()
    }


def _validation_by_tensor(
    tensors: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    original: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    validation: dict[str, dict[str, Any]] = {}
    for tensor_name, tensor in tensors.items():
        expected_tensor = expected[tensor_name]
        original_tensor = original[tensor_name]
        checksum = _tensor_checksum(tensor)
        expected_checksum = _tensor_checksum(expected_tensor)
        validation[tensor_name] = {
            "allclose": _tensor_close(tensor, expected_tensor),
            "checksum": checksum,
            "expected_checksum": expected_checksum,
            "checksum_matches": checksum == expected_checksum,
            "max_abs_error": _tensor_max_abs_error(tensor, expected_tensor),
            "original_checksum": _tensor_checksum(original_tensor),
        }
    return validation


def _restored_validation_by_tensor(
    tensors: Mapping[str, torch.Tensor],
    original: Mapping[str, torch.Tensor],
) -> dict[str, bool]:
    return {
        tensor_name: _tensor_close(tensor, original[tensor_name])
        for tensor_name, tensor in tensors.items()
    }


def _staging_targets_by_request(
    tensors: Mapping[str, torch.Tensor],
    requests: list[SliceRequest],
) -> dict[str, torch.Tensor]:
    return {
        request.target_id
        or request.tensor_name: torch.full_like(
            tensors[request.tensor_name],
            float("nan"),
        )
        for request in requests
    }


def _staging_by_tensor(
    staging_targets: Mapping[str, torch.Tensor],
    requests: list[SliceRequest],
) -> dict[str, torch.Tensor]:
    return {
        request.tensor_name: staging_targets[request.target_id or request.tensor_name]
        for request in requests
    }


def _planned_read_groups(plans: list[SegmentPlan]) -> list[dict[str, Any]]:
    grouped: dict[str, list[SegmentPlan]] = {}
    for plan in plans:
        grouped.setdefault(plan.source_id, []).append(plan)
    return [
        {
            "source_id": source_id,
            "tensor_names": sorted({plan.tensor_name for plan in source_plans}),
            "segment_count": len(source_plans),
            "bytes": sum(plan.bytes for plan in source_plans),
            "segments": [plan.to_dict() for plan in source_plans],
        }
        for source_id, source_plans in sorted(grouped.items())
    ]


def _copy_staging_into_runtime_targets(
    staging_by_tensor: Mapping[str, torch.Tensor],
    target_tensors: Mapping[str, torch.Tensor],
) -> None:
    for tensor_name, staging in staging_by_tensor.items():
        target = target_tensors[tensor_name]
        prepared = staging.to(device=target.device, dtype=target.dtype).contiguous()
        with torch.no_grad():
            target.copy_(prepared)
        if target.is_cuda:
            torch.cuda.synchronize(target.device)


def run_runtime_multitensor_refit_smoke(
    tensors: Mapping[str, torch.Tensor],
    *,
    runtime_framework: str,
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
    previous_model_version: str = "previous-runtime-version",
    artifact_path: str | Path | None = None,
    mode: str = "runtime-multitensor-refit-smoke",
    runtime_imported: bool = False,
    real_runtime_engine_used: bool = False,
    actual_nixl_reads_used: bool = False,
    gpu_nixl_reads_used: bool = False,
) -> dict[str, Any]:
    """Install and roll back a multi-tensor refit bundle."""

    if runtime_framework not in {"vllm", "sglang"}:
        raise ValueError("runtime_framework must be 'vllm' or 'sglang'")
    if len(tensors) < 2:
        raise ValueError("multi-tensor runtime smoke requires at least two tensors")

    owned_tensors = dict(tensors)
    original = {
        tensor_name: tensor.detach().clone()
        for tensor_name, tensor in owned_tensors.items()
    }

    planner_start = time.perf_counter()
    requests, owners, plans = build_runtime_multitensor_plan(
        owned_tensors,
        runtime_framework=runtime_framework,
        model_name=model_name,
        model_version=model_version,
    )
    planner_duration_ms = (time.perf_counter() - planner_start) * 1000
    trainer_update = trainer_update_parameters_from_ownerships(owners)
    expected = _expected_by_tensor(
        owned_tensors,
        step_count=trainer_update.step_count,
        learning_rate=trainer_update.learning_rate,
    )

    payload_start = time.perf_counter()
    segment_payloads = _materialize_segment_payloads(
        plans,
        owners,
        tensors=owned_tensors,
    )
    payload_materialization_duration_ms = (time.perf_counter() - payload_start) * 1000

    runtime_targets = _runtime_targets_by_request(owned_tensors, requests)
    transaction = begin_runtime_refit_transaction(
        runtime_targets,
        previous_model_version=previous_model_version,
        target_model_version=model_version,
    )
    install_start = time.perf_counter()
    installed = install_segment_payloads_into_runtime_tensors(
        segment_payloads,
        runtime_targets,
    )
    activation_install_duration_ms = (time.perf_counter() - install_start) * 1000
    validation = _validation_by_tensor(owned_tensors, expected, original)
    rollback_start = time.perf_counter()
    transaction.rollback()
    rollback_duration_ms = (time.perf_counter() - rollback_start) * 1000
    restored = _restored_validation_by_tensor(owned_tensors, original)

    target_key_by_tensor = _target_key_by_tensor(requests)
    source_count_per_target_tensor = {
        tensor_name: len(
            {plan.source_id for plan in plans if plan.tensor_name == tensor_name}
        )
        for tensor_name in owned_tensors
    }
    segment_count_per_target_tensor = {
        tensor_name: sum(1 for plan in plans if plan.tensor_name == tensor_name)
        for tensor_name in owned_tensors
    }
    trainer_to_inference_bytes = sum(plan.bytes for plan in plans)
    target_tensor_bytes = sum(_nbytes(tensor) for tensor in owned_tensors.values())
    allclose = all(item["allclose"] for item in validation.values())
    checksum_matches = all(item["checksum_matches"] for item in validation.values())
    restored_original = all(restored.values())

    proof = {
        "runtime_framework": runtime_framework,
        "receiver_requests_from_runtime_owned_tensors": True,
        "receiver_installed_into_runtime_owned_tensors": allclose,
        f"{runtime_framework}_owned_target_tensors": True,
        f"receiver_installed_into_{runtime_framework}_owned_tensors": allclose,
        "runtime_imported": bool(runtime_imported),
        "runtime_owned_target_tensors": True,
        "real_runtime_engine_used": bool(real_runtime_engine_used),
        "live_runtime_engine_used": bool(real_runtime_engine_used),
        "multi_tensor_refit_transaction_used": True,
        "target_tensor_count_gt1": len(owned_tensors) > 1,
        "target_slice_spans_multiple_trainers": all(
            count >= 2 for count in source_count_per_target_tensor.values()
        ),
        "trainer_optimizer_step_publisher_used": True,
        "receiver_expected_update_from_source_metadata": True,
        "trainer_owned_parameter_tensor_used": True,
        "synthetic_training_objective_used": True,
        "synthetic_source_values_used": False,
        "static_replacement_formula_source_values_used": False,
        "actual_nixl_reads_used": bool(actual_nixl_reads_used),
        "gpu_nixl_reads_used": bool(gpu_nixl_reads_used),
        "nixl_reads_land_directly_in_runtime_tensor": False,
        "runtime_update_from_segment_payloads": allclose,
        "restored_original_tensors": restored_original,
        "checksum_gate": checksum_matches,
        "allclose": allclose,
        "trainer_full_all_gather_used": False,
        "trainer_side_inference_layout_conversion_used": False,
        "host_side_torch_cat_used": False,
        "torch_distributed_data_transfer_used": False,
        "real_rl_training_loop_used": False,
    }

    result = {
        "schema_version": 1,
        "result": (
            "pass" if allclose and checksum_matches and restored_original else "fail"
        ),
        "mode": mode,
        "runtime_framework": runtime_framework,
        "model_name": model_name,
        "model_version": model_version,
        "previous_model_version": previous_model_version,
        "target_tensor_count": len(owned_tensors),
        "target_tensor_names": list(owned_tensors),
        "target_key_by_tensor": target_key_by_tensor,
        "target_tensors": {
            tensor_name: {
                "shape": list(_shape(tensor)),
                "dtype": _torch_dtype_name(tensor.dtype),
                "device": str(tensor.device),
                "bytes": _nbytes(tensor),
            }
            for tensor_name, tensor in owned_tensors.items()
        },
        "requests": [request.to_dict() for request in requests],
        "source_ownerships": [owner.to_dict() for owner in owners],
        "segment_plans": [plan.to_dict() for plan in plans],
        "installed_segments": [
            {
                "tensor_name": segment.tensor_name,
                "target_key": segment.target_key,
                "target_range": [list(bounds) for bounds in segment.target_range],
                "bytes": segment.bytes,
                "source_id": segment.source_id,
            }
            for segment in installed
        ],
        "transaction": transaction.to_dict(),
        "proof": proof,
        "validation": {
            "allclose": allclose,
            "checksum_matches": checksum_matches,
            "restored_original": restored_original,
            "by_tensor": validation,
            "restored_by_tensor": restored,
            "expected_optimizer_step_count": trainer_update.step_count,
            "expected_learning_rate": trainer_update.learning_rate,
        },
        "metrics": {
            "planner_duration_ms": planner_duration_ms,
            "source_payload_materialization_duration_ms": (
                payload_materialization_duration_ms
            ),
            "activation_install_duration_ms": activation_install_duration_ms,
            "rollback_duration_ms": rollback_duration_ms,
            "trainer_to_inference_bytes": trainer_to_inference_bytes,
            "inference_side_fanout_bytes": 0,
            "redundant_cross_boundary_factor": (
                trainer_to_inference_bytes / target_tensor_bytes
                if target_tensor_bytes
                else 0.0
            ),
            "target_tensor_bytes": target_tensor_bytes,
            "target_tensor_count": len(owned_tensors),
            "segment_count": len(plans),
            "segment_count_per_target_tensor": segment_count_per_target_tensor,
            "source_count_per_target_tensor": source_count_per_target_tensor,
        },
        "trainer_source_update": trainer_step_source_provenance(
            step_count=trainer_update.step_count,
            learning_rate=trainer_update.learning_rate,
        ),
    }
    if artifact_path is not None:
        _write_artifact(result, artifact_path)
    if result["result"] != "pass":
        raise RuntimeError(f"runtime multi-tensor refit smoke failed: {result!r}")
    return result


def run_runtime_multitensor_nixl_staging_smoke(
    tensors: Mapping[str, torch.Tensor],
    *,
    runtime_framework: str,
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
    previous_model_version: str = "previous-runtime-version",
    artifact_path: str | Path | None = None,
    mode: str = "runtime-multitensor-nixl-staging-smoke",
    runtime_imported: bool = False,
    real_runtime_engine_used: bool = False,
) -> dict[str, Any]:
    """Validate multi-tensor NIXL-style staging before runtime install.

    Segment payloads are copied into per-target staging tensors to model where
    future NIXL reads will land. The runtime-owned tensors are then updated from
    those staging tensors and rolled back. This smoke does not issue NIXL reads.
    """

    if runtime_framework not in {"vllm", "sglang"}:
        raise ValueError("runtime_framework must be 'vllm' or 'sglang'")
    if len(tensors) < 2:
        raise ValueError(
            "multi-tensor NIXL staging smoke requires at least two tensors"
        )

    owned_tensors = dict(tensors)
    original = {
        tensor_name: tensor.detach().clone()
        for tensor_name, tensor in owned_tensors.items()
    }

    planner_start = time.perf_counter()
    requests, owners, plans = build_runtime_multitensor_plan(
        owned_tensors,
        runtime_framework=runtime_framework,
        model_name=model_name,
        model_version=model_version,
    )
    planner_duration_ms = (time.perf_counter() - planner_start) * 1000
    trainer_update = trainer_update_parameters_from_ownerships(owners)
    expected = _expected_by_tensor(
        owned_tensors,
        step_count=trainer_update.step_count,
        learning_rate=trainer_update.learning_rate,
    )

    payload_start = time.perf_counter()
    segment_payloads = _materialize_segment_payloads(
        plans,
        owners,
        tensors=owned_tensors,
    )
    payload_materialization_duration_ms = (time.perf_counter() - payload_start) * 1000

    staging_targets = _staging_targets_by_request(owned_tensors, requests)
    staging_start = time.perf_counter()
    installed_staging = install_segment_payloads_into_runtime_tensors(
        segment_payloads,
        staging_targets,
    )
    staging_assembly_duration_ms = (time.perf_counter() - staging_start) * 1000
    staging_by_tensor = _staging_by_tensor(staging_targets, requests)
    staging_validation = _validation_by_tensor(
        staging_by_tensor,
        expected,
        original,
    )
    staging_allclose = all(item["allclose"] for item in staging_validation.values())
    staging_checksum_matches = all(
        item["checksum_matches"] for item in staging_validation.values()
    )

    runtime_targets = _runtime_targets_by_request(owned_tensors, requests)
    transaction = begin_runtime_refit_transaction(
        runtime_targets,
        previous_model_version=previous_model_version,
        target_model_version=model_version,
    )
    install_start = time.perf_counter()
    _copy_staging_into_runtime_targets(staging_by_tensor, owned_tensors)
    activation_install_duration_ms = (time.perf_counter() - install_start) * 1000
    runtime_validation = _validation_by_tensor(owned_tensors, expected, original)
    runtime_allclose = all(item["allclose"] for item in runtime_validation.values())
    runtime_checksum_matches = all(
        item["checksum_matches"] for item in runtime_validation.values()
    )

    rollback_start = time.perf_counter()
    transaction.rollback()
    rollback_duration_ms = (time.perf_counter() - rollback_start) * 1000
    restored = _restored_validation_by_tensor(owned_tensors, original)
    restored_original = all(restored.values())

    target_key_by_tensor = _target_key_by_tensor(requests)
    source_count_per_target_tensor = {
        tensor_name: len(
            {plan.source_id for plan in plans if plan.tensor_name == tensor_name}
        )
        for tensor_name in owned_tensors
    }
    segment_count_per_target_tensor = {
        tensor_name: sum(1 for plan in plans if plan.tensor_name == tensor_name)
        for tensor_name in owned_tensors
    }
    trainer_to_inference_bytes = sum(plan.bytes for plan in plans)
    target_tensor_bytes = sum(_nbytes(tensor) for tensor in owned_tensors.values())

    proof = {
        "runtime_framework": runtime_framework,
        "receiver_requests_from_runtime_owned_tensors": True,
        "receiver_installed_into_runtime_owned_tensors": runtime_allclose,
        f"{runtime_framework}_owned_target_tensors": True,
        f"receiver_installed_into_{runtime_framework}_owned_tensors": runtime_allclose,
        "runtime_imported": bool(runtime_imported),
        "runtime_owned_target_tensors": True,
        "real_runtime_engine_used": bool(real_runtime_engine_used),
        "live_runtime_engine_used": bool(real_runtime_engine_used),
        "multi_tensor_refit_transaction_used": True,
        "multi_tensor_nixl_staging_contract_used": True,
        "target_tensor_count_gt1": len(owned_tensors) > 1,
        "target_slice_spans_multiple_trainers": all(
            count >= 2 for count in source_count_per_target_tensor.values()
        ),
        "trainer_optimizer_step_publisher_used": True,
        "receiver_expected_update_from_source_metadata": True,
        "trainer_owned_parameter_tensor_used": True,
        "synthetic_training_objective_used": True,
        "synthetic_source_values_used": False,
        "static_replacement_formula_source_values_used": False,
        "actual_nixl_reads_used": False,
        "gpu_nixl_reads_used": False,
        "nixl_read_descriptor_groups_planned": True,
        "nixl_reads_land_into_staging_tensors": staging_allclose,
        "nixl_reads_land_directly_in_runtime_tensor": False,
        "runtime_update_from_nixl_staging_tensors": runtime_allclose,
        "runtime_update_from_segment_payloads": runtime_allclose,
        "restored_original_tensors": restored_original,
        "checksum_gate": runtime_checksum_matches,
        "staging_checksum_gate": staging_checksum_matches,
        "allclose": runtime_allclose,
        "trainer_full_all_gather_used": False,
        "trainer_side_inference_layout_conversion_used": False,
        "host_side_torch_cat_used": False,
        "torch_distributed_data_transfer_used": False,
        "real_rl_training_loop_used": False,
    }

    result = {
        "schema_version": 1,
        "result": (
            "pass"
            if staging_allclose
            and staging_checksum_matches
            and runtime_allclose
            and runtime_checksum_matches
            and restored_original
            else "fail"
        ),
        "mode": mode,
        "runtime_framework": runtime_framework,
        "model_name": model_name,
        "model_version": model_version,
        "previous_model_version": previous_model_version,
        "target_tensor_count": len(owned_tensors),
        "target_tensor_names": list(owned_tensors),
        "target_key_by_tensor": target_key_by_tensor,
        "target_tensors": {
            tensor_name: {
                "shape": list(_shape(tensor)),
                "dtype": _torch_dtype_name(tensor.dtype),
                "device": str(tensor.device),
                "bytes": _nbytes(tensor),
            }
            for tensor_name, tensor in owned_tensors.items()
        },
        "nixl_staging_targets": {
            tensor_name: {
                "target_key": target_key_by_tensor[tensor_name],
                "shape": list(_shape(staging_by_tensor[tensor_name])),
                "dtype": _torch_dtype_name(staging_by_tensor[tensor_name].dtype),
                "device": str(staging_by_tensor[tensor_name].device),
                "bytes": _nbytes(staging_by_tensor[tensor_name]),
            }
            for tensor_name in owned_tensors
        },
        "requests": [request.to_dict() for request in requests],
        "source_ownerships": [owner.to_dict() for owner in owners],
        "segment_plans": [plan.to_dict() for plan in plans],
        "installed_staging_segments": [
            {
                "tensor_name": segment.tensor_name,
                "target_key": segment.target_key,
                "target_range": [list(bounds) for bounds in segment.target_range],
                "bytes": segment.bytes,
                "source_id": segment.source_id,
            }
            for segment in installed_staging
        ],
        "transaction": transaction.to_dict(),
        "proof": proof,
        "validation": {
            "nixl_staging_allclose": staging_allclose,
            "nixl_staging_checksum_matches": staging_checksum_matches,
            "allclose": runtime_allclose,
            "checksum_matches": runtime_checksum_matches,
            "restored_original": restored_original,
            "staging_by_tensor": staging_validation,
            "by_tensor": runtime_validation,
            "restored_by_tensor": restored,
            "expected_optimizer_step_count": trainer_update.step_count,
            "expected_learning_rate": trainer_update.learning_rate,
        },
        "metrics": {
            "planner_duration_ms": planner_duration_ms,
            "source_payload_materialization_duration_ms": (
                payload_materialization_duration_ms
            ),
            "nixl_staging_assembly_duration_ms": staging_assembly_duration_ms,
            "activation_install_duration_ms": activation_install_duration_ms,
            "rollback_duration_ms": rollback_duration_ms,
            "trainer_to_inference_bytes": trainer_to_inference_bytes,
            "inference_side_fanout_bytes": 0,
            "redundant_cross_boundary_factor": (
                trainer_to_inference_bytes / target_tensor_bytes
                if target_tensor_bytes
                else 0.0
            ),
            "target_tensor_bytes": target_tensor_bytes,
            "target_tensor_count": len(owned_tensors),
            "staging_tensor_count": len(staging_targets),
            "segment_count": len(plans),
            "segment_count_per_target_tensor": segment_count_per_target_tensor,
            "source_count_per_target_tensor": source_count_per_target_tensor,
        },
        "nixl": {
            "planned_read_groups": _planned_read_groups(plans),
            "actual_read_groups": [],
            "actual_nixl_reads_used": False,
        },
        "trainer_source_update": trainer_step_source_provenance(
            step_count=trainer_update.step_count,
            learning_rate=trainer_update.learning_rate,
        ),
    }
    if artifact_path is not None:
        _write_artifact(result, artifact_path)
    if result["result"] != "pass":
        raise RuntimeError(
            f"runtime multi-tensor NIXL staging smoke failed: {result!r}"
        )
    return result


def _parse_shape(value: str) -> tuple[int, ...]:
    dims = tuple(int(part) for part in value.split(",") if part)
    if not dims:
        raise ValueError("shape must contain at least one dimension")
    return dims


def _torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.removeprefix("torch.").lower()
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported dtype for multi-tensor smoke: {dtype_name!r}")
    return aliases[normalized]


def _build_cli_tensors(
    *,
    tensor_specs: list[str],
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    specs = tensor_specs or [
        f"{name}:{','.join(str(dim) for dim in shape)}"
        for name, shape in DEFAULT_TENSOR_SHAPES.items()
    ]
    tensors: dict[str, torch.Tensor] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"tensor spec must be name:dim,dim: {spec!r}")
        name, shape_text = spec.split(":", 1)
        shape = _parse_shape(shape_text)
        tensor = torch.arange(
            int(torch.tensor(shape).prod().item()),
            dtype=torch.float32,
            device=device,
        ).reshape(shape)
        tensors[name] = tensor.to(dtype=dtype).contiguous()
    return tensors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-framework",
        choices=("vllm", "sglang"),
        required=True,
    )
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--previous-model-version", default="previous-runtime-version")
    parser.add_argument(
        "--nixl-staging",
        action="store_true",
        help="validate the multi-tensor NIXL staging contract instead of direct install",
    )
    parser.add_argument("--dtype", default="float32")
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device for runtime-owned tensors; use nscale for execution",
    )
    parser.add_argument(
        "--tensor",
        action="append",
        default=[],
        help="runtime tensor spec as name:dim,dim; may be repeated",
    )
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    dtype = _torch_dtype_from_name(args.dtype)
    tensors = _build_cli_tensors(
        tensor_specs=args.tensor,
        dtype=dtype,
        device=device,
    )
    if args.nixl_staging:
        result = run_runtime_multitensor_nixl_staging_smoke(
            tensors,
            runtime_framework=args.runtime_framework,
            model_name=args.model_name,
            model_version=args.model_version,
            previous_model_version=args.previous_model_version,
            artifact_path=args.artifact_path,
        )
    else:
        result = run_runtime_multitensor_refit_smoke(
            tensors,
            runtime_framework=args.runtime_framework,
            model_name=args.model_name,
            model_version=args.model_version,
            previous_model_version=args.previous_model_version,
            artifact_path=args.artifact_path,
        )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
