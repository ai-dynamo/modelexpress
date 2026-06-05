# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real torch.distributed trainer source-publication smoke for MX refit.

This smoke proves the source side without runtime/NIXL transfer: each
torch.distributed rank owns one row shard, runs a local optimizer step under an
initialized process group, and emits MX-compatible slice ownership metadata for
that shard. Rank 0 writes an artifact assembled from scalar/metadata results;
tensor payloads are not gathered through torch.distributed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from .refit_trainer_step import (
    destroy_distributed_trainer_process_group,
    ensure_distributed_trainer_process_group,
    publish_distributed_trainer_loop_step,
    trainer_loop_model_version,
    trainer_step_tensor_for_range,
)
from .resharding import SliceOwnership

DEFAULT_MODEL_NAME = "mx-distributed-trainer-publication-smoke"
DEFAULT_MODEL_VERSION = "distributed-trainer-base"
DEFAULT_TENSOR_NAME = "lm_head.weight"
DEFAULT_SHAPE = (8, 4)
DEFAULT_STEP_INDEX = 2
DEFAULT_LR = 0.125


def run_distributed_trainer_publication_smoke(
    *,
    artifact_path: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    model_version: str = DEFAULT_MODEL_VERSION,
    tensor_name: str = DEFAULT_TENSOR_NAME,
    shape: tuple[int, int] = DEFAULT_SHAPE,
    dtype: torch.dtype = torch.float32,
    backend: str = "gloo",
    device: str = "auto",
    step_index: int = DEFAULT_STEP_INDEX,
    learning_rate: float = DEFAULT_LR,
) -> dict[str, Any] | None:
    """Run a distributed trainer source publication and write rank-0 artifact."""

    context, initialized_by_mx = ensure_distributed_trainer_process_group(
        backend=backend
    )
    try:
        selected_device = _select_device(device)
        owner = _ownership_for_rank(
            model_name=model_name,
            model_version=model_version,
            tensor_name=tensor_name,
            shape=shape,
            dtype=dtype,
            rank=context.rank,
            world_size=context.world_size,
        )
        loop_step = publish_distributed_trainer_loop_step(
            [owner],
            dtype=dtype,
            device=selected_device,
            step_index=step_index,
            learning_rate=learning_rate,
            distributed_context=context,
        )
        publication = loop_step.source_publications[0]
        expected = trainer_step_tensor_for_range(
            owner.global_shape,
            owner.source_range,
            dtype=dtype,
            device=selected_device,
            step_count=step_index,
            learning_rate=learning_rate,
        )
        torch.testing.assert_close(publication.tensor, expected)
        checksum = float(publication.tensor.float().sum().item())
        rank_payload = {
            "rank": context.rank,
            "world_size": context.world_size,
            "backend": context.backend,
            "local_rank": context.local_rank,
            "device": str(selected_device),
            "source_id": publication.ownership.source_id,
            "source_range": [
                [int(start), int(end)]
                for start, end in publication.ownership.source_range
            ],
            "tensor_shape": [int(dim) for dim in publication.tensor.shape],
            "tensor_dtype": str(publication.tensor.dtype).removeprefix("torch."),
            "tensor_checksum": checksum,
            "source_publication": publication.to_artifact_metadata(),
        }
        rank_payloads = _all_gather_rank_payloads(rank_payload)
        if context.rank != 0:
            return None

        result = {
            "schema_version": 1,
            "result": "pass",
            "mode": "distributed-trainer-source-publication-smoke",
            "model_name": model_name,
            "model_version": trainer_loop_model_version(model_version, step_index),
            "model_version_base": model_version,
            "tensor_name": tensor_name,
            "global_shape": [int(dim) for dim in shape],
            "dtype": str(dtype).removeprefix("torch."),
            "trainer_step_index": int(step_index),
            "learning_rate": float(learning_rate),
            "rank_count": len(rank_payloads),
            "source_ids": [payload["source_id"] for payload in rank_payloads],
            "source_ranges": [payload["source_range"] for payload in rank_payloads],
            "rank_payloads": rank_payloads,
            "proof": {
                "real_distributed_trainer_loop_used": True,
                "torch_distributed_process_group_used": True,
                "torch_distributed_backend": context.backend,
                "torch_distributed_world_size": context.world_size,
                "torch_distributed_scalar_sync_used": True,
                "torch_distributed_artifact_metadata_gather_used": True,
                "torch_distributed_tensor_payload_gather_used": False,
                "torch_distributed_data_transfer_used": False,
                "trainer_owned_parameter_tensor_used": True,
                "optimizer_step_publisher_used": True,
                "synthetic_trainer_loop_smoke_used": False,
                "synthetic_training_objective_used": True,
                "static_replacement_formula_used": False,
                "real_rl_training_loop_used": False,
                "source_publications_cover_full_tensor": _covers_full_rows(
                    rank_payloads, shape
                ),
                "source_payloads_validated_against_expected": True,
                "runtime_refit_used": False,
                "nixl_transfer_used": False,
            },
        }
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result
    finally:
        if initialized_by_mx:
            destroy_distributed_trainer_process_group()


def _ownership_for_rank(
    *,
    model_name: str,
    model_version: str,
    tensor_name: str,
    shape: tuple[int, int],
    dtype: torch.dtype,
    rank: int,
    world_size: int,
) -> SliceOwnership:
    start, end = _row_shard(rank, world_size, shape[0])
    dtype_name = str(dtype).removeprefix("torch.")
    element_size = int(torch.empty((), dtype=dtype).element_size())
    return SliceOwnership(
        model_name=model_name,
        model_version=model_version,
        tensor_name=tensor_name,
        global_shape=shape,
        dtype=dtype_name,
        source_range=((start, end), (0, shape[1])),
        worker_id=f"trainer-rank{rank}-worker",
        source_id=f"trainer-rank{rank}",
        worker_rank=rank,
        layout_tags={
            "trainer_layout": "torch-distributed-row-shard-poc",
            "source_tensor_owner": "torch.distributed-trainer-rank",
            "storage_layout": "row-major",
        },
        element_size_bytes=element_size,
    )


def _row_shard(rank: int, world_size: int, rows: int) -> tuple[int, int]:
    base = rows // world_size
    remainder = rows % world_size
    start = rank * base + min(rank, remainder)
    end = start + base + (1 if rank < remainder else 0)
    if end <= start:
        raise ValueError(
            f"world_size={world_size} creates empty row shard for shape rows={rows}"
        )
    return start, end


def _select_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
            torch.cuda.set_device(local_rank)
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")
    selected = torch.device(device)
    if selected.type == "cuda":
        torch.cuda.set_device(selected)
    return selected


def _all_gather_rank_payloads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    import torch.distributed as dist

    gathered: list[dict[str, Any] | None] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, payload)
    return [item for item in gathered if item is not None]


def _covers_full_rows(rank_payloads: list[dict[str, Any]], shape: tuple[int, int]) -> bool:
    ranges = sorted(
        (
            int(payload["source_range"][0][0]),
            int(payload["source_range"][0][1]),
        )
        for payload in rank_payloads
    )
    cursor = 0
    for start, end in ranges:
        if start != cursor:
            return False
        cursor = end
    return cursor == int(shape[0])


def _torch_dtype_from_name(name: str) -> torch.dtype:
    normalized = name.removeprefix("torch.").lower()
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    if normalized not in aliases:
        raise ValueError(f"unsupported dtype for smoke: {name!r}")
    return aliases[normalized]


def _parse_shape(value: str) -> tuple[int, int]:
    dims = tuple(int(part) for part in value.replace("x", ",").split(",") if part)
    if len(dims) != 2 or any(dim <= 0 for dim in dims):
        raise ValueError(f"shape must be two positive dims, got {value!r}")
    return dims


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-path", type=Path, required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-version", default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--tensor-name", default=DEFAULT_TENSOR_NAME)
    parser.add_argument("--shape", default=",".join(str(dim) for dim in DEFAULT_SHAPE))
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--step-index", type=int, default=DEFAULT_STEP_INDEX)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_distributed_trainer_publication_smoke(
        artifact_path=args.artifact_path,
        model_name=args.model_name,
        model_version=args.model_version,
        tensor_name=args.tensor_name,
        shape=_parse_shape(args.shape),
        dtype=_torch_dtype_from_name(args.dtype),
        backend=args.backend,
        device=args.device,
        step_index=args.step_index,
        learning_rate=args.learning_rate,
    )
    if result is not None:
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
