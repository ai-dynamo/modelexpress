# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer result models for RL weight pulls."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from modelexpress import p2p_pb2
from modelexpress.rl_metadata import RlSourceCandidate, RlSourceRole


@dataclass(frozen=True)
class RlTransferAttempt:
    """One attempted RL receive source."""

    mx_source_id: str
    worker_id: str
    worker_rank: int
    role: RlSourceRole
    model_version: int
    success: bool
    error: str | None = None
    lease_id: str = ""
    source_status: int = p2p_pb2.SOURCE_STATUS_UNKNOWN
    source_updated_at: int = 0
    bytes_transferred: int = 0
    tensor_count: int = 0
    duration_seconds: float = 0.0


def successful_attempt_from_candidate(
    candidate: RlSourceCandidate,
    *,
    bytes_transferred: int,
    tensor_count: int,
    duration_seconds: float,
    lease_id: str = "",
) -> RlTransferAttempt:
    return RlTransferAttempt(
        mx_source_id=candidate.mx_source_id,
        worker_id=candidate.worker_id,
        worker_rank=candidate.worker_rank,
        role=candidate.metadata.role,
        model_version=candidate.metadata.model_version,
        success=True,
        lease_id=lease_id,
        source_status=candidate.status,
        source_updated_at=candidate.updated_at,
        bytes_transferred=bytes_transferred,
        tensor_count=tensor_count,
        duration_seconds=duration_seconds,
    )


def failed_attempt_from_candidate(
    candidate: RlSourceCandidate,
    exc: Exception,
) -> RlTransferAttempt:
    return RlTransferAttempt(
        mx_source_id=candidate.mx_source_id,
        worker_id=candidate.worker_id,
        worker_rank=candidate.worker_rank,
        role=candidate.metadata.role,
        model_version=candidate.metadata.model_version,
        success=False,
        error=str(exc),
        lease_id=_lease_id_for_exception(exc, candidate),
        source_status=candidate.status,
        source_updated_at=candidate.updated_at,
    )


@dataclass(frozen=True)
class RlTransferReport:
    """Summary of the latest RL receive attempt sequence."""

    requested_model_version: int | None
    resolved_model_version: int | None
    receiver_rank: int
    attempts: tuple[RlTransferAttempt, ...]

    @property
    def success(self) -> bool:
        return any(attempt.success for attempt in self.attempts)

    @property
    def retry_count(self) -> int:
        return sum(1 for attempt in self.attempts if not attempt.success)

    @property
    def source_role(self) -> RlSourceRole | None:
        for attempt in self.attempts:
            if attempt.success:
                return attempt.role
        return None

    @property
    def source_worker_id(self) -> str | None:
        for attempt in self.attempts:
            if attempt.success:
                return attempt.worker_id
        return None


@dataclass(frozen=True)
class _ReceiveCandidateResult:
    tensors: list[tuple[str, torch.Tensor]]
    bytes_transferred: int
    tensor_count: int
    duration_seconds: float
    lease_id: str = ""
    tensor_metadata: Mapping[str, Mapping[str, Any]] | None = None


@dataclass(frozen=True)
class _ReceiveSourcesResult:
    model_version: int
    tensors: list[tuple[str, torch.Tensor]]
    tensor_metadata: Mapping[str, Mapping[str, Any]] | None = None


def _lease_id_for_exception(exc: Exception, candidate: RlSourceCandidate) -> str:
    lease_id_for = getattr(exc, "lease_id_for", None)
    if lease_id_for is None:
        return ""
    try:
        return str(lease_id_for(candidate))
    except Exception:
        return ""
