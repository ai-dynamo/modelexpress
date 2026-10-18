# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport abstractions for file-backed ModelExpress artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .. import p2p_pb2
from .artifact_transfer import ArtifactBundle, ArtifactTransfer


@dataclass(frozen=True)
class ArtifactTransportContext:
    """Backend-independent values shared by artifact fetch and publish."""

    node_rank: int = 0
    accelerator: str = ""


@dataclass(frozen=True)
class StagedArtifact:
    """Artifact manifest header whose files are present in local staging."""

    header: p2p_pb2.GetArtifactManifestHeaderResponse
    transport: str


class ArtifactInstallStatus(Enum):
    """State persisted by the common artifact installation lifecycle."""

    ATTEMPTED = "attempted"
    MISS = "miss"
    INSTALLED = "installed"


@dataclass(frozen=True)
class ArtifactInstallState:
    """Shared install result used by transport-specific lifecycle policy."""

    status: ArtifactInstallStatus
    artifact_id: str | None = None
    transport: str | None = None
    owner_pid: int | None = None
    owner_starttime: str | None = None


class ArtifactInstallDisposition(Enum):
    """Action selected by a transport for an existing install state."""

    FETCH = "fetch"
    SKIP = "skip"
    CACHED_MISS = "cached_miss"


class PublicationHandle(Protocol):
    """Handle returned by a successful transport publication."""

    transport: str
    identifier: str

    def stop(self) -> None:
        """Stop any transport-side publication resources."""


class ArtifactTransport(Protocol):
    """Backend interface for moving one prepared artifact bundle."""

    name: str
    publish_requires_install_state: bool
    retry_publish_on_failure: bool

    def state_after_cache_miss(self) -> ArtifactInstallState:
        """Return the shared state to persist after a confirmed cache miss."""

    def resolve_install_state(
        self, state: ArtifactInstallState
    ) -> ArtifactInstallDisposition:
        """Choose how installation should handle an existing shared state."""

    def should_publish(self, state: ArtifactInstallState | None) -> bool:
        """Return whether this transport should publish for an install state."""

    def fetch(
        self,
        artifact: ArtifactTransfer,
        identity: p2p_pb2.SourceIdentity,
        context: ArtifactTransportContext,
    ) -> StagedArtifact:
        """Fetch an artifact into its target staging files."""

    def publish(
        self,
        artifact: ArtifactTransfer,
        identity: p2p_pb2.SourceIdentity,
        bundle: ArtifactBundle,
        context: ArtifactTransportContext,
    ) -> PublicationHandle:
        """Publish an already-prepared artifact bundle."""


class ArtifactCacheMiss(LookupError):
    """The transport has no compatible artifact for the requested identity."""


class ArtifactCacheStale(ArtifactCacheMiss):
    """A remote artifact exists but is incomplete, invalid, or corrupted."""


class ArtifactTransportUnavailable(RuntimeError):
    """The transport cannot currently serve or publish the artifact."""
