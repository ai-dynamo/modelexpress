# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal public delta weight-sync configuration, results, and lifecycle.

V0 has one production data path: a canonical exact-base delta stored in S3.
S3 bucket configuration is explicit rather than selected through a generic
transport interface. Encoding details live in the self-describing root index,
not in this public configuration or the revision-catalog API.

``PublicationMode`` remains publisher behavior. V0 exposes only ``BLOCK``:
the publisher waits by exact-get until Miles commits the revision. A publisher
never commits its own revision.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from .manifest import RevisionState


class PublicationMode(Enum):
    """Publisher wait behavior; this is not a server RPC field."""

    BLOCK = "block"


class ReceiverRevisionState(Enum):
    """SGLang-local receiver outcomes, never persisted by the MX server."""

    BYTES_RECEIVED = "bytes_received"
    VERIFIED = "verified"
    FAILED = "failed"
    POISONED = "poisoned"


RevisionLifecycleState = RevisionState

ModelId = str
VersionId = str


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
    publication_mode: PublicationMode = PublicationMode.BLOCK


@dataclass(frozen=True)
class ReceiverConfig:
    model_id: ModelId
    catalog_endpoint: str
    s3: S3Config


def normalize_layer_scope(layers: Iterable[str] | None) -> tuple[str, ...] | None:
    """Return the canonical layer scope: ``None`` for the complete model."""
    if layers is None:
        return None
    return tuple(sorted(set(layers)))


@dataclass(frozen=True)
class PreparedUpdate:
    """Immutable identity revalidated at an engine's mutation boundary."""

    model_id: ModelId
    base_version: VersionId
    base_digest: str
    target_version: VersionId
    target_digest: str
    format_digest: str
    receiver_incarnation: str
    model_generation: int
    layer_scope: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        scope = self.layer_scope
        if scope is None:
            return
        if not isinstance(scope, tuple):
            raise ValueError(f"layer_scope must be a tuple, got {type(scope).__name__}")
        if scope != normalize_layer_scope(scope):
            raise ValueError(f"layer_scope must be sorted and unique, got {scope}")


@dataclass(frozen=True)
class PublishResult:
    model_id: ModelId
    version: VersionId
    state: RevisionLifecycleState


@dataclass(frozen=True)
class PublisherStatus:
    model_id: ModelId
    current_version: VersionId | None = None
    state: RevisionLifecycleState | None = None
    publication_mode: PublicationMode = PublicationMode.BLOCK


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
        """A poisoned receiver must stop serving until replaced or restored."""
        return self.state is ReceiverRevisionState.POISONED


@runtime_checkable
class PublisherProtocol(Protocol):
    """Trainer-side lifecycle implemented by the ModelExpress publisher."""

    def initialize(self, config: PublisherConfig) -> None:
        """Initialize launch attestation, catalog access, and direct S3 upload."""

    def publish_version(
        self,
        version: VersionId,
        *,
        base_version: VersionId | None = None,
        gather_hf_buckets=None,
    ) -> PublishResult:
        """Publish launch metadata or a delta from Miles-gathered HF buckets."""

    def status(self) -> PublisherStatus:
        """Return the most recently observed exact revision state."""

    def deregister(self) -> None:
        """Release client-owned resources while no publication is active."""


@runtime_checkable
class ReceiverProtocol(Protocol):
    """Engine-native receiver lifecycle reserved for reset-plan Phase 2.2."""

    def initialize(self, config: ReceiverConfig) -> None:
        """Initialize engine-native exact-revision preparation."""

    def start_weight_update(self, version: VersionId) -> None:
        """Prepare one exact target without mutating the engine."""

    def update_weights(
        self,
        layers: Iterable[str] | None = None,
    ) -> WeightUpdateResult:
        """Install the internally prepared update through native loader mechanics."""

    def status(self) -> ReceiverStatus:
        """Return installed version and receiver-local health."""
