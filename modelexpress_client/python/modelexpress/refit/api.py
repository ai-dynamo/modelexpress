# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public delta weight-sync contract: configuration, results, and lifecycle.

This module freezes *what* a publisher and a receiver agree on; it deliberately
implements neither. `Publisher` and `Receiver` are structural protocols, so an
engine-native receiver (SGLang) conforms by shape alone and never inherits from
or composes a ModelExpress engine abstraction. Source capture, delta encoding,
byte transfer, reconstruction, engine installation, and recovery execution are
owned by later phases.

Publication contract carried by `PublicationMode`:

- `ASYNC` may return once immutable publication reaches `READY`;
- `BLOCK` waits read-only for `COMMITTED`;
- cancelling the wait stops waiting only and leaves the catalog revision
  `READY`.

The concrete wait, cancellation, and drain behavior is implemented by the
publisher that owns the data plane, not by this contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from .manifest import (
    DeltaTransferMethod,
    PublicationMode,
    ReceiverRevisionState,
    RevisionLifecycleState,
)

ModelId = str
VersionId = str


class TransportKind(Enum):
    """Location kinds a delta reference can take on the wire."""

    S3 = "s3"
    ZEROMQ = "zeromq"
    FILESYSTEM = "filesystem"


@dataclass(frozen=True)
class TransportConfig:
    """Normal delivery storage. Its data-plane methods land with the codecs."""

    kind: TransportKind
    root_uri: str


@dataclass(frozen=True)
class RecoveryStoreConfig:
    """Durable full anchors and deltas, independent from delivery transport."""

    kind: TransportKind
    root_uri: str


@dataclass(frozen=True)
class PublisherConfig:
    model_id: ModelId
    catalog_endpoint: str
    transport: TransportConfig
    delta_transfer_method: DeltaTransferMethod = DeltaTransferMethod.RANK_LOCAL
    recovery_store: RecoveryStoreConfig | None = None
    delta_method: str | None = None
    compression_algorithm: str | None = None
    full_anchor_interval: int | None = None
    publication_mode: PublicationMode = PublicationMode.BLOCK


@dataclass(frozen=True)
class ReceiverConfig:
    model_id: ModelId
    catalog_endpoint: str
    transport: TransportConfig
    delta_transfer_method: DeltaTransferMethod = DeltaTransferMethod.RANK_LOCAL
    recovery_store: RecoveryStoreConfig | None = None
    delta_method: str | None = None
    compression_algorithm: str | None = None
    max_delta_replay_length: int | None = None


def normalize_layer_scope(layers: Iterable[str] | None) -> tuple[str, ...] | None:
    """Return the canonical layer scope: ``None`` for the complete model."""
    if layers is None:
        return None
    return tuple(sorted(set(layers)))


@dataclass(frozen=True)
class PreparedUpdate:
    """Immutable identity of one prepared, exact-base target.

    Every field is revalidated at the engine's mutation boundary before the
    first weight write. The prepared payload representation stays private to
    the receiver that produced it and is never part of this identity.
    """

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
    created: bool = True


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
        """`POISONED` receivers must stop serving and recover from a complete model."""
        return self.state is ReceiverRevisionState.POISONED


@runtime_checkable
class Publisher(Protocol):
    """Trainer-side lifecycle. Implemented by the ModelExpress publisher."""

    def initialize(self, config: PublisherConfig) -> None:
        """Initialize catalog, delivery transport, recovery storage, and delta support."""

    def publish_version(
        self,
        version: VersionId,
        layers: Sequence[str] | None = None,
        *,
        base_version: VersionId | None = None,
    ) -> PublishResult:
        """Publish an exact-base CPU delta or a GPU-direct revision."""

    def status(self) -> PublisherStatus:
        """Return current version, readiness, transfer-reference, and lifetime state."""

    def deregister(self) -> None:
        """Drain live readers, mark the source stale, and release resources."""


@runtime_checkable
class Receiver(Protocol):
    """Rollout-side lifecycle. Implemented natively by the serving engine."""

    def initialize(self, config: ReceiverConfig) -> None:
        """Initialize engine, catalog, delivery transport, and recovery storage."""

    def start_weight_update(self, version: VersionId) -> None:
        """Select, fetch, verify, and prepare the target without mutating the engine."""

    def update_weights(
        self,
        layers: Sequence[str] | None = None,
    ) -> WeightUpdateResult:
        """Apply the internally prepared update and install the target version."""

    def recover(self, version: VersionId) -> WeightUpdateResult:
        """Replace unknown or poisoned state with a verified complete version."""

    def status(self) -> ReceiverStatus:
        """Return installed version, health, and recovery state."""
