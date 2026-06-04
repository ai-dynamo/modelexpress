# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checksum-gated Level-5 timing smoke helpers for refit comparisons.

The production Level-5 goal is a full-model, comparable MX/NIXL vs NCCL Reshard
vs CheckpointEngine benchmark. This module intentionally supports a smaller
synthetic smoke first: each row must be real measured code, checksum/allclose
validated, and normalized into one timing schema. The table keeps placement and
claim scope explicit so same-node, cross-node, synthetic, and full-model results
cannot be mixed silently.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

from .refit_poc_scenario import (
    GLOBAL_SHAPE,
    MODEL_NAME,
    MODEL_VERSION,
    REQUEST_RANGE,
    TENSOR_NAME,
    inference_request,
    primary_ownerships,
)
from .resharding import dtype_itemsize, range_volume

MX_NIXL_STRATEGY = "mx-nixl-segment-read"
NCCL_RESHARD_STRATEGY = "nccl-reshard-fixed-membership"
CHECKPOINT_ENGINE_STRATEGY = "checkpoint-engine-full-gather-apply"
REQUIRED_STRATEGIES = (
    MX_NIXL_STRATEGY,
    NCCL_RESHARD_STRATEGY,
    CHECKPOINT_ENGINE_STRATEGY,
)
REQUIRED_TIMING_FIELDS = (
    "registration_duration_ms",
    "publish_duration_ms",
    "planner_duration_ms",
    "read_duration_ms",
    "activation_install_duration_ms",
)


def _import_torch():
    import torch

    return torch


def scenario_metadata() -> dict[str, Any]:
    request = inference_request()
    requested_bytes = range_volume(request.requested_range) * dtype_itemsize(
        request.dtype,
        request.element_size_bytes,
    )
    full_tensor_bytes = range_volume(tuple((0, dim) for dim in GLOBAL_SHAPE))
    full_tensor_bytes *= dtype_itemsize(request.dtype, request.element_size_bytes)
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "tensor_name": TENSOR_NAME,
        "global_shape": list(GLOBAL_SHAPE),
        "requested_range": [list(axis) for axis in REQUEST_RANGE],
        "requested_bytes": requested_bytes,
        "full_tensor_bytes": full_tensor_bytes,
        "dtype": request.dtype,
        "trainer_source_count": len(primary_ownerships()),
        "target_request_count": 1,
        "synthetic_contract": True,
    }


def run_nccl_reshard_baseline(artifact_path: Path) -> dict[str, Any] | None:
    """Measure a same-node synthetic NCCL Reshard-style full-tensor row.

    Ranks 0 and 1 are trainer holders. They all-gather the full tensor in a
    trainer subgroup, rank 0 materializes the full tensor, and rank 3 receives
    the full tensor before narrowing/installing the target request.
    """

    torch = _import_torch()
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Level-5 NCCL baseline")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 4:
        raise RuntimeError("Level-5 NCCL baseline requires at least 4 ranks")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            "Level-5 NCCL baseline requires one visible CUDA device per rank "
            f"(visible={torch.cuda.device_count()}, world_size={world_size})"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    trainer_ranks = [0, 1]
    target_rank = 3
    trainer_group = dist.new_group(ranks=trainer_ranks, backend="nccl")

    try:
        request = inference_request()
        owners = primary_ownerships()
        owner_by_rank = {int(owner.worker_rank or 0): owner for owner in owners}
        full_tensor = None
        collective_duration_ms = 0.0
        materialize_duration_ms = 0.0
        send_duration_ms = 0.0
        receive_duration_ms = 0.0
        install_duration_ms = 0.0
        validation: dict[str, Any] | None = None

        dist.barrier()
        if rank in owner_by_rank:
            owner = owner_by_rank[rank]
            source_tensor = _materialize_range(owner.source_range, device).contiguous()
            padded_source = _pad_trainer_source(source_tensor, owners)
            gathered = [torch.empty_like(padded_source) for _ in trainer_ranks]
            collective_start = time.perf_counter()
            dist.all_gather(gathered, padded_source, group=trainer_group)
            torch.cuda.synchronize(device)
            collective_duration_ms = (time.perf_counter() - collective_start) * 1000

            if rank == 0:
                materialize_start = time.perf_counter()
                full_tensor = _materialize_full_tensor_from_padded_gather(
                    gathered,
                    owners,
                    device,
                )
                torch.cuda.synchronize(device)
                materialize_duration_ms = (
                    time.perf_counter() - materialize_start
                ) * 1000

        dist.barrier()
        if rank == 0:
            assert full_tensor is not None
            send_start = time.perf_counter()
            dist.send(full_tensor.contiguous(), dst=target_rank)
            torch.cuda.synchronize(device)
            send_duration_ms = (time.perf_counter() - send_start) * 1000
            metric_tensor = torch.tensor(
                [collective_duration_ms, materialize_duration_ms, send_duration_ms],
                device=device,
                dtype=torch.float64,
            )
            dist.send(metric_tensor, dst=target_rank)
        elif rank == target_rank:
            received_full = torch.empty(
                GLOBAL_SHAPE, device=device, dtype=torch.float32
            )
            receive_start = time.perf_counter()
            dist.recv(received_full, src=0)
            torch.cuda.synchronize(device)
            receive_duration_ms = (time.perf_counter() - receive_start) * 1000
            metric_tensor = torch.empty(3, device=device, dtype=torch.float64)
            dist.recv(metric_tensor, src=0)
            collective_duration_ms = float(metric_tensor[0].item())
            materialize_duration_ms = float(metric_tensor[1].item())
            send_duration_ms = float(metric_tensor[2].item())

            target = torch.full(request.target_shape, float("nan"), device=device)
            install_start = time.perf_counter()
            _install_request_from_full_tensor(
                received_full, target, request.requested_range
            )
            torch.cuda.synchronize(device)
            install_duration_ms = (time.perf_counter() - install_start) * 1000
            validation = _validate_target(target, request.requested_range)

        dist.barrier()
        result = None
        if rank == target_rank:
            assert validation is not None
            result = _baseline_artifact(
                strategy=NCCL_RESHARD_STRATEGY,
                mode="level5-nccl-reshard-synthetic-same-node",
                placement_scope="same-node-single-pod",
                validation=validation,
                metrics={
                    "trainer_to_inference_bytes": _full_tensor_bytes(),
                    "inference_side_fanout_bytes": 0,
                    "trainer_collective_bytes": _padded_trainer_collective_bytes(
                        owners,
                        len(trainer_ranks),
                    ),
                    "trainer_collective_padding_bytes": (
                        _padded_trainer_collective_bytes(owners, len(trainer_ranks))
                        - _full_tensor_bytes()
                    ),
                    "checkpoint_storage_bytes": 0,
                    "redundant_cross_boundary_factor": _ratio(
                        _full_tensor_bytes(),
                        _requested_bytes(),
                    ),
                    "segment_count": 1,
                    "source_count_per_target_tensor": {TENSOR_NAME: len(trainer_ranks)},
                    "registration_duration_ms": 0.0,
                    "publish_duration_ms": 0.0,
                    "metadata_query_duration_ms": 0.0,
                    "planner_duration_ms": 0.0,
                    "read_duration_ms": receive_duration_ms,
                    "raw_read_duration_ms": receive_duration_ms,
                    "trainer_collective_duration_ms": collective_duration_ms,
                    "full_tensor_materialize_duration_ms": materialize_duration_ms,
                    "trainer_to_inference_send_duration_ms": send_duration_ms,
                    "activation_install_duration_ms": install_duration_ms,
                },
                proof={
                    "real_measured": True,
                    "checksum_gate": True,
                    "trainer_full_all_gather_used": True,
                    "full_tensor_materialized_before_target_apply": True,
                    "trainer_side_inference_layout_conversion_used": False,
                    "host_side_torch_cat_used": False,
                    "actual_nixl_reads_used": False,
                    "torch_distributed_data_transfer_used": True,
                },
            )
            _write_artifact(result, artifact_path)
            if not validation["allclose"]:
                raise RuntimeError(f"NCCL baseline validation failed: {validation}")
        return result
    finally:
        dist.destroy_process_group()


def run_checkpoint_engine_baseline(artifact_path: Path) -> dict[str, Any] | None:
    """Measure a same-node synthetic CheckpointEngine-style row.

    Ranks 0 and 1 full-gather the trainer tensor, rank 0 writes a checkpoint
    file, and rank 3 reads that file before installing the requested target
    slice. This is a real measured storage/apply smoke, not a production CE
    implementation.
    """

    torch = _import_torch()
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Level-5 checkpoint baseline")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 4:
        raise RuntimeError("Level-5 checkpoint baseline requires at least 4 ranks")
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            "Level-5 checkpoint baseline requires one visible CUDA device per rank "
            f"(visible={torch.cuda.device_count()}, world_size={world_size})"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl")
    trainer_ranks = [0, 1]
    target_rank = 3
    trainer_group = dist.new_group(ranks=trainer_ranks, backend="nccl")
    checkpoint_path = _checkpoint_path()

    try:
        request = inference_request()
        owners = primary_ownerships()
        owner_by_rank = {int(owner.worker_rank or 0): owner for owner in owners}
        collective_duration_ms = 0.0
        materialize_duration_ms = 0.0
        write_duration_ms = 0.0
        read_duration_ms = 0.0
        install_duration_ms = 0.0
        validation: dict[str, Any] | None = None

        dist.barrier()
        if rank in owner_by_rank:
            owner = owner_by_rank[rank]
            source_tensor = _materialize_range(owner.source_range, device).contiguous()
            padded_source = _pad_trainer_source(source_tensor, owners)
            gathered = [torch.empty_like(padded_source) for _ in trainer_ranks]
            collective_start = time.perf_counter()
            dist.all_gather(gathered, padded_source, group=trainer_group)
            torch.cuda.synchronize(device)
            collective_duration_ms = (time.perf_counter() - collective_start) * 1000

            if rank == 0:
                materialize_start = time.perf_counter()
                full_tensor = _materialize_full_tensor_from_padded_gather(
                    gathered,
                    owners,
                    device,
                )
                torch.cuda.synchronize(device)
                materialize_duration_ms = (
                    time.perf_counter() - materialize_start
                ) * 1000

                write_start = time.perf_counter()
                torch.save(full_tensor.cpu(), checkpoint_path)
                write_duration_ms = (time.perf_counter() - write_start) * 1000

        dist.barrier()
        if rank == 0:
            metric_tensor = torch.tensor(
                [collective_duration_ms, materialize_duration_ms, write_duration_ms],
                device=device,
                dtype=torch.float64,
            )
            dist.send(metric_tensor, dst=target_rank)
        elif rank == target_rank:
            metric_tensor = torch.empty(3, device=device, dtype=torch.float64)
            dist.recv(metric_tensor, src=0)
            collective_duration_ms = float(metric_tensor[0].item())
            materialize_duration_ms = float(metric_tensor[1].item())
            write_duration_ms = float(metric_tensor[2].item())

            read_start = time.perf_counter()
            full_tensor = torch.load(checkpoint_path, map_location=device)
            torch.cuda.synchronize(device)
            read_duration_ms = (time.perf_counter() - read_start) * 1000

            target = torch.full(request.target_shape, float("nan"), device=device)
            install_start = time.perf_counter()
            _install_request_from_full_tensor(
                full_tensor, target, request.requested_range
            )
            torch.cuda.synchronize(device)
            install_duration_ms = (time.perf_counter() - install_start) * 1000
            validation = _validate_target(target, request.requested_range)

        dist.barrier()
        if rank == 0:
            checkpoint_path.unlink(missing_ok=True)
        result = None
        if rank == target_rank:
            assert validation is not None
            result = _baseline_artifact(
                strategy=CHECKPOINT_ENGINE_STRATEGY,
                mode="level5-checkpoint-engine-synthetic-same-node",
                placement_scope="same-node-single-pod",
                validation=validation,
                metrics={
                    "trainer_to_inference_bytes": 0,
                    "inference_side_fanout_bytes": 0,
                    "trainer_collective_bytes": _padded_trainer_collective_bytes(
                        owners,
                        len(trainer_ranks),
                    ),
                    "trainer_collective_padding_bytes": (
                        _padded_trainer_collective_bytes(owners, len(trainer_ranks))
                        - _full_tensor_bytes()
                    ),
                    "checkpoint_storage_bytes": _full_tensor_bytes() * 2,
                    "redundant_cross_boundary_factor": _ratio(
                        _full_tensor_bytes() * 2,
                        _requested_bytes(),
                    ),
                    "segment_count": 1,
                    "source_count_per_target_tensor": {TENSOR_NAME: len(trainer_ranks)},
                    "registration_duration_ms": 0.0,
                    "publish_duration_ms": 0.0,
                    "metadata_query_duration_ms": 0.0,
                    "planner_duration_ms": 0.0,
                    "read_duration_ms": read_duration_ms,
                    "raw_read_duration_ms": read_duration_ms,
                    "trainer_collective_duration_ms": collective_duration_ms,
                    "full_tensor_materialize_duration_ms": materialize_duration_ms,
                    "checkpoint_write_duration_ms": write_duration_ms,
                    "checkpoint_read_duration_ms": read_duration_ms,
                    "activation_install_duration_ms": install_duration_ms,
                },
                proof={
                    "real_measured": True,
                    "checksum_gate": True,
                    "trainer_full_all_gather_used": True,
                    "full_tensor_materialized_before_target_apply": True,
                    "checkpoint_storage_used": True,
                    "trainer_side_inference_layout_conversion_used": False,
                    "actual_nixl_reads_used": False,
                    "torch_distributed_data_transfer_used": False,
                },
            )
            _write_artifact(result, artifact_path)
            if not validation["allclose"]:
                raise RuntimeError(
                    f"CheckpointEngine baseline validation failed: {validation}"
                )
        return result
    finally:
        dist.destroy_process_group()


def normalize_level5_artifact(
    artifact: Mapping[str, Any],
    *,
    source_artifact: str = "",
    strategy: str | None = None,
) -> dict[str, Any]:
    """Normalize a measured refit artifact into one Level-5 timing row."""

    if "level5_row" in artifact:
        row = dict(artifact["level5_row"])
        if source_artifact:
            row["source_artifact"] = source_artifact
        return _finalize_row(row)

    metrics = dict(artifact.get("metrics", {}))
    validation = dict(artifact.get("validation", {}))
    proof = dict(artifact.get("proof", {}))
    selected_strategy = strategy or _strategy_from_artifact(artifact)
    placement_scope = _placement_scope_from_artifact(artifact)
    row = {
        "strategy": selected_strategy,
        "source_artifact": source_artifact,
        "mode": artifact.get("mode", ""),
        "placement_scope": placement_scope,
        "measured": True,
        "result": artifact.get("result", "unknown"),
        "allclose": bool(validation.get("allclose", False)),
        "checksum": validation.get("checksum"),
        "expected_checksum": validation.get("expected_checksum"),
        "checksum_matches": _checksum_matches(validation),
        "max_abs_error": validation.get("max_abs_error"),
        "trainer_to_inference_bytes": _int_metric(
            metrics,
            "trainer_to_inference_bytes",
            default=0,
        ),
        "inference_side_fanout_bytes": _int_metric(
            metrics,
            "inference_side_fanout_bytes",
            default=0,
        ),
        "trainer_collective_bytes": _int_metric(
            metrics,
            "trainer_collective_bytes",
            default=0,
        ),
        "checkpoint_storage_bytes": _int_metric(
            metrics,
            "checkpoint_storage_bytes",
            default=0,
        ),
        "redundant_cross_boundary_factor": _float_metric(
            metrics,
            "redundant_cross_boundary_factor",
            default=1.0,
        ),
        "segment_count": _int_metric(metrics, "segment_count", default=0),
        "source_count_per_target_tensor": dict(
            metrics.get("source_count_per_target_tensor", {})
        ),
        "registration_duration_ms": _first_float_metric(
            metrics,
            "nixl_registration_duration_ms",
            "registration_duration_ms",
            default=0.0,
        ),
        "publish_duration_ms": _float_metric(
            metrics,
            "publish_duration_ms",
            default=0.0,
        ),
        "metadata_query_duration_ms": _float_metric(
            metrics,
            "metadata_query_duration_ms",
            default=0.0,
        ),
        "planner_duration_ms": _float_metric(
            metrics,
            "planner_duration_ms",
            default=0.0,
        ),
        "read_duration_ms": _first_float_metric(
            metrics,
            "raw_nixl_read_duration_ms",
            "raw_read_duration_ms",
            "read_duration_ms",
            "gpu_copy_duration_ms",
            default=0.0,
        ),
        "activation_install_duration_ms": _float_metric(
            metrics,
            "activation_install_duration_ms",
            default=0.0,
        ),
        "retry_count": _int_metric(metrics, "retry_count", default=0),
        "rediscovery_count": _int_metric(metrics, "rediscovery_count", default=0),
        "proof_flags": proof,
    }
    return _finalize_row(row)


def unmeasured_level5_row(
    strategy: str,
    *,
    reason: str,
    source_artifact: str = "",
) -> dict[str, Any]:
    row = {
        "strategy": strategy,
        "source_artifact": source_artifact,
        "mode": "unmeasured",
        "placement_scope": "unknown",
        "measured": False,
        "result": "blocked",
        "allclose": False,
        "checksum": None,
        "expected_checksum": None,
        "checksum_matches": False,
        "max_abs_error": None,
        "trainer_to_inference_bytes": 0,
        "inference_side_fanout_bytes": 0,
        "trainer_collective_bytes": 0,
        "checkpoint_storage_bytes": 0,
        "redundant_cross_boundary_factor": 0.0,
        "segment_count": 0,
        "source_count_per_target_tensor": {},
        "registration_duration_ms": None,
        "publish_duration_ms": None,
        "metadata_query_duration_ms": None,
        "planner_duration_ms": None,
        "read_duration_ms": None,
        "activation_install_duration_ms": None,
        "retry_count": 0,
        "rediscovery_count": 0,
        "proof_flags": {},
        "block_reason": reason,
    }
    return _finalize_row(row)


def build_level5_timing_table(
    rows: list[Mapping[str, Any]],
    *,
    claim_scope: str = "synthetic-same-node-smoke",
) -> dict[str, Any]:
    normalized_rows = [_finalize_row(dict(row)) for row in rows]
    by_strategy = {row["strategy"]: row for row in normalized_rows}
    missing_strategies = [
        strategy for strategy in REQUIRED_STRATEGIES if strategy not in by_strategy
    ]
    failed_rows = [
        row["strategy"]
        for row in normalized_rows
        if row["strategy"] in REQUIRED_STRATEGIES and not row["pass"]
    ]
    measured_scopes = {
        row["placement_scope"]
        for row in normalized_rows
        if row.get("measured") and row.get("placement_scope") != "unknown"
    }
    comparable_placement = len(measured_scopes) <= 1
    result_pass = not missing_strategies and not failed_rows and comparable_placement
    return {
        "schema_version": 1,
        "result": "pass" if result_pass else "fail",
        "claim_scope": claim_scope,
        "scenario": scenario_metadata(),
        "required_strategies": list(REQUIRED_STRATEGIES),
        "rows": normalized_rows,
        "missing_strategies": missing_strategies,
        "failed_rows": failed_rows,
        "placement_scopes": sorted(measured_scopes),
        "comparable_placement_scope": comparable_placement,
        "level5_synthetic_smoke_pass": result_pass,
        "level5_full_model_claim_safe": False,
        "production_competitive_claim_safe": False,
        "notes": [
            "Rows are checksum/allclose gated and real measured for the declared placement scope.",
            "This synthetic table does not prove full-model production competitiveness.",
        ],
    }


def build_level5_table_from_artifacts(
    *,
    mx_artifact: Mapping[str, Any] | None = None,
    nccl_artifact: Mapping[str, Any] | None = None,
    checkpoint_artifact: Mapping[str, Any] | None = None,
    mx_artifact_name: str = "",
    nccl_artifact_name: str = "",
    checkpoint_artifact_name: str = "",
    claim_scope: str = "synthetic-same-node-smoke",
) -> dict[str, Any]:
    rows = []
    if mx_artifact is None:
        rows.append(
            unmeasured_level5_row(
                MX_NIXL_STRATEGY,
                reason="MX/NIXL checksum-backed artifact was not provided.",
                source_artifact=mx_artifact_name,
            )
        )
    else:
        rows.append(
            normalize_level5_artifact(
                mx_artifact,
                source_artifact=mx_artifact_name,
                strategy=MX_NIXL_STRATEGY,
            )
        )
    if nccl_artifact is None:
        rows.append(
            unmeasured_level5_row(
                NCCL_RESHARD_STRATEGY,
                reason="NCCL Reshard checksum-backed artifact was not provided.",
                source_artifact=nccl_artifact_name,
            )
        )
    else:
        rows.append(
            normalize_level5_artifact(
                nccl_artifact,
                source_artifact=nccl_artifact_name,
                strategy=NCCL_RESHARD_STRATEGY,
            )
        )
    if checkpoint_artifact is None:
        rows.append(
            unmeasured_level5_row(
                CHECKPOINT_ENGINE_STRATEGY,
                reason="CheckpointEngine checksum-backed artifact was not provided.",
                source_artifact=checkpoint_artifact_name,
            )
        )
    else:
        rows.append(
            normalize_level5_artifact(
                checkpoint_artifact,
                source_artifact=checkpoint_artifact_name,
                strategy=CHECKPOINT_ENGINE_STRATEGY,
            )
        )
    return build_level5_timing_table(rows, claim_scope=claim_scope)


def _baseline_artifact(
    *,
    strategy: str,
    mode: str,
    placement_scope: str,
    validation: Mapping[str, Any],
    metrics: Mapping[str, Any],
    proof: Mapping[str, Any],
) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "result": "pass" if validation.get("allclose") else "fail",
        "mode": mode,
        "strategy": strategy,
        "placement_scope": placement_scope,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "tensor_name": TENSOR_NAME,
        "scenario": scenario_metadata(),
        "metrics": dict(metrics),
        "validation": dict(validation),
        "proof": dict(proof),
    }
    result["level5_row"] = normalize_level5_artifact(result)
    return result


def _finalize_row(row: dict[str, Any]) -> dict[str, Any]:
    row.setdefault("measured", False)
    row.setdefault("result", "unknown")
    row.setdefault("proof_flags", {})
    row.setdefault("placement_scope", "unknown")
    row.setdefault("source_count_per_target_tensor", {})
    row.setdefault("checksum_matches", _checksum_pair_matches(row))
    row.setdefault("allclose", False)
    row.setdefault("block_reason", "")
    missing_metrics = [
        field
        for field in REQUIRED_TIMING_FIELDS
        if row.get(field) is None and row.get("measured")
    ]
    row["missing_timing_fields"] = missing_metrics
    row["pass"] = bool(
        row.get("measured")
        and row.get("result") == "pass"
        and row.get("allclose")
        and row.get("checksum_matches")
        and not missing_metrics
    )
    return row


def _strategy_from_artifact(artifact: Mapping[str, Any]) -> str:
    strategy = artifact.get("strategy")
    if isinstance(strategy, str) and strategy:
        return strategy
    mode = str(artifact.get("mode", ""))
    proof = artifact.get("proof", {})
    if proof.get("actual_nixl_reads_used") or "nixl" in mode:
        return MX_NIXL_STRATEGY
    return str(strategy or "unknown")


def _placement_scope_from_artifact(artifact: Mapping[str, Any]) -> str:
    placement = artifact.get("placement_scope")
    if isinstance(placement, str) and placement:
        return placement
    distributed = artifact.get("distributed", {})
    proof = artifact.get("proof", {})
    mode = str(artifact.get("mode", ""))
    if distributed.get("cross_node") or proof.get("cross_node_pods"):
        return "cross-node"
    if "same-node" in mode or "distributed-4rank" in mode:
        return "same-node-single-pod"
    return "unknown"


def _install_request_from_full_tensor(full_tensor, target, requested_range) -> None:
    slices = tuple(slice(start, end) for start, end in requested_range)
    target.copy_(full_tensor[slices])


def _pad_trainer_source(source_tensor, owners):
    torch = _import_torch()
    max_rows = max(
        owner.source_range[0][1] - owner.source_range[0][0] for owner in owners
    )
    padded = torch.zeros(
        (max_rows, source_tensor.shape[1]),
        device=source_tensor.device,
        dtype=source_tensor.dtype,
    )
    padded[: source_tensor.shape[0], :].copy_(source_tensor)
    return padded


def _materialize_full_tensor_from_padded_gather(gathered, owners, device):
    torch = _import_torch()
    full_tensor = torch.empty(GLOBAL_SHAPE, device=device, dtype=torch.float32)
    for gathered_tensor, gathered_owner in zip(gathered, owners):
        row_start, row_end = gathered_owner.source_range[0]
        row_count = row_end - row_start
        full_tensor[row_start:row_end, :].copy_(gathered_tensor[:row_count, :])
    return full_tensor


def _padded_trainer_collective_bytes(owners, trainer_rank_count: int) -> int:
    request = inference_request()
    max_rows = max(
        owner.source_range[0][1] - owner.source_range[0][0] for owner in owners
    )
    padded_elements_per_rank = max_rows * GLOBAL_SHAPE[1]
    padded_bytes_per_rank = padded_elements_per_rank * dtype_itemsize(
        request.dtype,
        request.element_size_bytes,
    )
    return padded_bytes_per_rank * trainer_rank_count * max(0, trainer_rank_count - 1)


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


def _checksum(tensor) -> float:
    torch = _import_torch()
    flat = tensor.float().reshape(-1)
    weights = torch.arange(
        1,
        flat.numel() + 1,
        device=tensor.device,
        dtype=torch.float32,
    )
    return float((flat * weights).sum().item())


def _validate_target(target, requested_range) -> dict[str, Any]:
    torch = _import_torch()
    expected = _materialize_range(requested_range, target.device)
    torch.cuda.synchronize(target.device)
    return {
        "allclose": bool(torch.allclose(target, expected)),
        "checksum": _checksum(target),
        "expected_checksum": _checksum(expected),
        "max_abs_error": float((target - expected).abs().max().item()),
    }


def _requested_bytes() -> int:
    request = inference_request()
    return range_volume(request.requested_range) * dtype_itemsize(
        request.dtype,
        request.element_size_bytes,
    )


def _full_tensor_bytes() -> int:
    request = inference_request()
    return range_volume(tuple((0, dim) for dim in GLOBAL_SHAPE)) * dtype_itemsize(
        request.dtype,
        request.element_size_bytes,
    )


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return round(float(numerator) / float(denominator), 6)


def _checkpoint_path() -> Path:
    tmp_dir = Path(os.environ.get("MX_LEVEL5_TMP_DIR", "/tmp"))
    run_id = os.environ.get("MX_LEVEL5_RUN_ID") or os.environ.get("TORCHELASTIC_RUN_ID")
    if not run_id:
        run_id = f"{os.getpid()}-{int(time.time() * 1000)}"
    return tmp_dir / f"mx_level5_checkpoint_{run_id}.pt"


def _checksum_matches(validation: Mapping[str, Any]) -> bool:
    checksum = validation.get("checksum")
    expected = validation.get("expected_checksum")
    if checksum is None or expected is None:
        return False
    try:
        checksum_f = float(checksum)
        expected_f = float(expected)
    except (TypeError, ValueError):
        return False
    tolerance = max(1e-5, abs(expected_f) * 1e-6)
    return abs(checksum_f - expected_f) <= tolerance


def _checksum_pair_matches(row: Mapping[str, Any]) -> bool:
    return _checksum_matches(
        {
            "checksum": row.get("checksum"),
            "expected_checksum": row.get("expected_checksum"),
        }
    )


def _int_metric(metrics: Mapping[str, Any], key: str, *, default: int) -> int:
    value = metrics.get(key, default)
    if value is None:
        return default
    return int(value)


def _float_metric(metrics: Mapping[str, Any], key: str, *, default: float) -> float:
    value = metrics.get(key, default)
    if value is None:
        return default
    return float(value)


def _first_float_metric(
    metrics: Mapping[str, Any],
    *keys: str,
    default: float,
) -> float:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return float(metrics[key])
    return default


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_artifact(result: Mapping[str, Any], artifact_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline_parser = subparsers.add_parser("baseline")
    baseline_parser.add_argument(
        "--strategy",
        choices=[NCCL_RESHARD_STRATEGY, CHECKPOINT_ENGINE_STRATEGY],
        required=True,
    )
    baseline_parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("/tmp/mx_level5_baseline.json"),
    )

    table_parser = subparsers.add_parser("table")
    table_parser.add_argument("--mx-artifact", type=Path)
    table_parser.add_argument("--nccl-artifact", type=Path)
    table_parser.add_argument("--checkpoint-artifact", type=Path)
    table_parser.add_argument(
        "--claim-scope",
        default="synthetic-same-node-smoke",
    )
    table_parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("/tmp/mx_level5_timing_table.json"),
    )

    args = parser.parse_args()
    if args.command == "baseline":
        if args.strategy == NCCL_RESHARD_STRATEGY:
            run_nccl_reshard_baseline(args.artifact)
        else:
            run_checkpoint_engine_baseline(args.artifact)
        return

    table = build_level5_table_from_artifacts(
        mx_artifact=_load_json(args.mx_artifact),
        nccl_artifact=_load_json(args.nccl_artifact),
        checkpoint_artifact=_load_json(args.checkpoint_artifact),
        mx_artifact_name=str(args.mx_artifact or ""),
        nccl_artifact_name=str(args.nccl_artifact or ""),
        checkpoint_artifact_name=str(args.checkpoint_artifact or ""),
        claim_scope=args.claim_scope,
    )
    _write_artifact(table, args.artifact)


if __name__ == "__main__":
    main()
