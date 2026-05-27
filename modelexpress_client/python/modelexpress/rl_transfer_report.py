# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer result models for RL weight pulls."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from modelexpress.rl_metadata import RlSourceRole


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
    bytes_transferred: int = 0
    tensor_count: int = 0
    duration_seconds: float = 0.0


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
