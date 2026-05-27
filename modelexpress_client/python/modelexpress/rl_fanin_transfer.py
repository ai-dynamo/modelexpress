# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-source RL dense fan-in transfer helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from modelexpress import p2p_pb2
from modelexpress.client import MxClientBase
from modelexpress.rl_metadata import RlSourceCandidate, RlSourceRole
from modelexpress.rl_reshard import (
    TensorReceiveSpec,
    plan_dense_reshard_transfers,
    receive_specs_from_tensors,
    source_specs_from_shape_registry,
)
from modelexpress.rl_slice_transfer import (
    GroupedSliceTransferManifest,
    build_grouped_slice_transfer_manifest,
)
from modelexpress.types import TensorDescriptor


@dataclass(frozen=True)
class DenseFanInSourceTransfer:
    """One source worker participating in a dense fan-in receive."""

    candidate: RlSourceCandidate
    source_metadata: bytes
    source_descriptors: list[TensorDescriptor]


@dataclass(frozen=True)
class DenseFanInPlan:
    """Prepared metadata and transfer views for one dense fan-in group."""

    model_version: int
    manifest: GroupedSliceTransferManifest
    source_transfers: tuple[DenseFanInSourceTransfer, ...]


@dataclass(frozen=True)
class DenseFanInSourceResult:
    """NIXL transfer stats for one source worker."""

    candidate: RlSourceCandidate
    bytes_transferred: int
    tensor_count: int
    duration_seconds: float


@dataclass(frozen=True)
class DenseFanInReceiveResult:
    """Completed dense fan-in receive result."""

    tensors: list[tuple[str, torch.Tensor]]
    source_results: tuple[DenseFanInSourceResult, ...]

    @property
    def bytes_transferred(self) -> int:
        return sum(result.bytes_transferred for result in self.source_results)

    @property
    def tensor_count(self) -> int:
        return sum(result.tensor_count for result in self.source_results)

    @property
    def duration_seconds(self) -> float:
        return sum(result.duration_seconds for result in self.source_results)


def prepare_dense_fanin_receive(
    *,
    mx_client: MxClientBase,
    candidates: Sequence[RlSourceCandidate],
    target_tensors: dict[str, torch.Tensor],
    target_specs: Sequence[TensorReceiveSpec] | None,
    receiver_rank: int,
    same_rank_only: bool,
) -> DenseFanInPlan:
    """Resolve source metadata and build a complete dense fan-in plan."""
    unique_candidates = _require_unique_rank_candidates(candidates)
    if len(unique_candidates) < 2:
        raise RuntimeError("ModelExpress dense fan-in requires at least two source ranks")
    model_version = _validate_candidate_group(unique_candidates)
    if not target_tensors:
        raise RuntimeError("ModelExpress dense fan-in requires caller-owned target tensors")

    source_specs = []
    source_descriptors_by_rank = {}
    source_metadata_by_rank = {}
    candidate_by_rank = {}
    for candidate in unique_candidates:
        if not candidate.metadata.shape_registry:
            raise RuntimeError(
                f"ModelExpress source {candidate.mx_source_id} has no RL shape registry"
            )
        metadata_resp = mx_client.get_metadata(candidate.mx_source_id, candidate.worker_id)
        if not metadata_resp.found:
            raise RuntimeError(
                f"ModelExpress source metadata not found for {candidate.mx_source_id}/"
                f"{candidate.worker_id}"
            )
        if metadata_resp.worker.worker_rank != candidate.worker_rank:
            raise RuntimeError(
                f"ModelExpress source metadata rank mismatch for {candidate.mx_source_id}/"
                f"{candidate.worker_id}: expected {candidate.worker_rank}, "
                f"got {metadata_resp.worker.worker_rank}"
            )

        candidate_by_rank[candidate.worker_rank] = candidate
        source_metadata_by_rank[candidate.worker_rank] = metadata_resp.worker.nixl_metadata
        source_descriptors_by_rank[candidate.worker_rank] = source_descriptors_from_worker(
            metadata_resp.worker
        )
        source_specs.extend(
            source_specs_from_shape_registry(
                candidate.metadata.shape_registry,
                worker_rank=candidate.worker_rank,
            )
        )

    effective_target_specs = tuple(
        target_specs
        if target_specs is not None
        else receive_specs_from_tensors(target_tensors, receiver_rank=receiver_rank)
    )
    transfer_plan = plan_dense_reshard_transfers(
        source_specs,
        effective_target_specs,
        same_rank_only=same_rank_only,
    )
    transfer_plan.raise_if_incomplete()
    manifest = build_grouped_slice_transfer_manifest(
        transfer_plan,
        source_descriptors_by_rank=source_descriptors_by_rank,
        target_tensors=target_tensors,
    )
    source_transfers = tuple(
        DenseFanInSourceTransfer(
            candidate=candidate_by_rank[transfer.source_worker_rank],
            source_metadata=source_metadata_by_rank[transfer.source_worker_rank],
            source_descriptors=transfer.source_descriptors,
        )
        for transfer in manifest.source_transfers
    )
    return DenseFanInPlan(
        model_version=model_version,
        manifest=manifest,
        source_transfers=source_transfers,
    )


def execute_dense_fanin_receive(
    plan: DenseFanInPlan,
    *,
    manager: Any,
    timeout_seconds: float,
) -> DenseFanInReceiveResult:
    """Register fan-in target slices and receive them from each source worker."""
    manager.register_tensors(plan.manifest.target_tensors)
    source_results = []
    for source in plan.source_transfers:
        bytes_transferred, tensor_count, duration_seconds = manager.receive_from_source(
            source_metadata=source.source_metadata,
            source_tensors=source.source_descriptors,
            timeout_seconds=timeout_seconds,
        )
        source_results.append(
            DenseFanInSourceResult(
                candidate=source.candidate,
                bytes_transferred=bytes_transferred,
                tensor_count=tensor_count,
                duration_seconds=duration_seconds,
            )
        )
    return DenseFanInReceiveResult(
        tensors=plan.manifest.output_tensors,
        source_results=tuple(source_results),
    )


def candidate_fanin_groups(
    candidates: Sequence[RlSourceCandidate],
) -> tuple[tuple[RlSourceCandidate, ...], ...]:
    """Group compatible multi-rank candidates for dense fan-in attempts."""
    groups: dict[tuple[int, RlSourceRole], list[RlSourceCandidate]] = {}
    for candidate in candidates:
        key = (candidate.metadata.model_version, candidate.metadata.role)
        groups.setdefault(key, []).append(candidate)
    return tuple(
        tuple(group)
        for group in groups.values()
        if len({candidate.worker_rank for candidate in group}) > 1
    )


def candidate_group_label(candidates: Sequence[RlSourceCandidate]) -> str:
    first = candidates[0]
    ranks = ",".join(str(candidate.worker_rank) for candidate in candidates)
    return (
        f"version={first.metadata.model_version} role={first.metadata.role.value} "
        f"ranks={ranks}"
    )


def source_descriptors_from_worker(
    worker: "p2p_pb2.WorkerMetadata",
) -> list[TensorDescriptor]:
    return [
        TensorDescriptor(
            name=tensor.name,
            addr=tensor.addr,
            size=tensor.size,
            device_id=tensor.device_id,
            dtype=tensor.dtype,
        )
        for tensor in worker.tensors
    ]


def _require_unique_rank_candidates(
    candidates: Sequence[RlSourceCandidate],
) -> tuple[RlSourceCandidate, ...]:
    by_rank: dict[int, RlSourceCandidate] = {}
    for candidate in candidates:
        existing = by_rank.get(candidate.worker_rank)
        if existing is not None:
            raise RuntimeError(
                "ModelExpress dense fan-in group has duplicate source rank "
                f"{candidate.worker_rank}: {existing.worker_id}, {candidate.worker_id}"
            )
        by_rank[candidate.worker_rank] = candidate
    return tuple(by_rank.values())


def _validate_candidate_group(candidates: Sequence[RlSourceCandidate]) -> int:
    first = candidates[0]
    model_version = first.metadata.model_version
    role = first.metadata.role
    for candidate in candidates[1:]:
        if candidate.metadata.model_version != model_version or candidate.metadata.role != role:
            raise RuntimeError("ModelExpress dense fan-in group mixes source versions or roles")
    return model_version
