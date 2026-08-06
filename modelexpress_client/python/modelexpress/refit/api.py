# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Values consumed by the Miles publisher and SGLang receiver."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

ModelId = str
VersionId = str


class ReceiverRevisionState(Enum):
    """SGLang-local receiver outcomes, never persisted by the MX server."""

    BYTES_RECEIVED = "bytes_received"
    VERIFIED = "verified"
    FAILED = "failed"
    POISONED = "poisoned"


@dataclass(frozen=True)
class S3Config:
    """Direct S3 destination; credentials are resolved privately by boto3."""

    bucket: str
    prefix: str = ""
    endpoint_url: str | None = None
    region_name: str | None = None


@dataclass(frozen=True)
class PublisherConfig:
    model_id: ModelId
    catalog_endpoint: str
    s3: S3Config


@dataclass(frozen=True)
class WeightUpdateResult:
    success: bool
    receiver_id: str
    installed_version: VersionId | None
    state: ReceiverRevisionState
    target_digest: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class ReceiverStatus:
    receiver_id: str
    model_id: ModelId
    installed_version: VersionId | None = None
    state: ReceiverRevisionState | None = None
    detail: str = ""

    @property
    def recovery_required(self) -> bool:
        return self.state is ReceiverRevisionState.POISONED
