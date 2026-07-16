# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinator-provided NCCL M2N bootstrap assignment."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from .. import m2n_bootstrap_pb2

NCCL_UNIQUE_ID_BYTES = 128
SHA256_DIGEST_BYTES = 32
MAX_BOOTSTRAP_TIMEOUT_S = 600.0


@dataclass(frozen=True)
class M2nBootstrapAssignment:
    """Immutable local view of a coordinator-defined communicator."""

    job_id: str
    attempt_id: str
    cohort_id: str
    participant_id: str
    uid_publisher_participant_id: str
    assigned_nccl_rank: int
    source_world_size: int
    destination_world_size: int
    roster_digest: bytes
    config_digest: bytes
    timeout_s: float = 120.0

    @property
    def world_size(self) -> int:
        return self.source_world_size + self.destination_world_size

    @property
    def is_uid_publisher(self) -> bool:
        return self.participant_id == self.uid_publisher_participant_id

    @property
    def ttl_ms(self) -> int:
        return max(1, int(self.timeout_s * 1000))

    def key_proto(self) -> m2n_bootstrap_pb2.M2nBootstrapKey:
        return m2n_bootstrap_pb2.M2nBootstrapKey(
            job_id=self.job_id,
            attempt_id=self.attempt_id,
            cohort_id=self.cohort_id,
        )

    def validate(self) -> None:
        for name, value in (
            ("job_id", self.job_id),
            ("attempt_id", self.attempt_id),
            ("cohort_id", self.cohort_id),
            ("participant_id", self.participant_id),
            ("uid_publisher_participant_id", self.uid_publisher_participant_id),
        ):
            if not value:
                raise ValueError(f"{name} is required")
        try:
            parsed_attempt_id = UUID(self.attempt_id)
        except ValueError as error:
            raise ValueError(
                "attempt_id must be a canonical random UUIDv4 and must never be reused"
            ) from error
        if parsed_attempt_id.version != 4 or str(parsed_attempt_id) != self.attempt_id:
            raise ValueError(
                "attempt_id must be a canonical random UUIDv4 and must never be reused"
            )
        if self.source_world_size <= 0 or self.destination_world_size <= 0:
            raise ValueError("source and destination world sizes must be positive")
        if not 0 <= self.assigned_nccl_rank < self.world_size:
            raise ValueError(
                f"assigned_nccl_rank {self.assigned_nccl_rank} is outside world size "
                f"{self.world_size}"
            )
        if self.is_uid_publisher != (self.assigned_nccl_rank == 0):
            raise ValueError("UID publisher participant must be assigned NCCL rank zero")
        if len(self.roster_digest) != SHA256_DIGEST_BYTES:
            raise ValueError("roster_digest must be a 32-byte SHA-256 digest")
        if len(self.config_digest) != SHA256_DIGEST_BYTES:
            raise ValueError("config_digest must be a 32-byte SHA-256 digest")
        if not 0 < self.timeout_s <= MAX_BOOTSTRAP_TIMEOUT_S:
            raise ValueError(
                f"timeout_s must be in the range (0, {MAX_BOOTSTRAP_TIMEOUT_S}]"
            )
