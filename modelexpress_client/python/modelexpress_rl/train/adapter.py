# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer-engine boundary for ModelExpress RL refit publication."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch


class NixlMetadataProvider(Protocol):
    """NIXL manager surface required to publish trainer manifests.

    Exposes the agent metadata every adapter needs plus ``register_tensors``.
    NIXL can only transfer registered memory, so an adapter registers whatever
    source buffers it owns (staging arenas, in-place local storage) before
    building its manifest, so the published ``nixl_metadata`` covers them.
    """

    @property
    def agent_name(self) -> str:
        """Return the local NIXL agent name."""
        ...

    @property
    def nixl_metadata(self) -> bytes:
        """Return serialized metadata for the local NIXL agent."""
        ...

    @property
    def listen_port(self) -> int | None:
        """Return the local NIXL metadata-listener port, when enabled."""
        ...

    def register_tensors(self, tensors: dict[str, torch.Tensor]) -> bytes:
        """Register buffers with NIXL and return the refreshed agent metadata."""
        ...


class TrainerStagingMode(str, Enum):
    """How a trainer adapter preserves a version's immutable source bytes."""

    UNSPECIFIED = "UNSPECIFIED"
    COPY_TO_DEVICE = "COPY_TO_DEVICE"
    COPY_TO_HOST = "COPY_TO_HOST"
    WRITE_TO_STORAGE = "WRITE_TO_STORAGE"
    IN_PLACE = "IN_PLACE"


class WeightPayloadFormat(str, Enum):
    """Weight representation fixed for one initialized client."""

    UNSPECIFIED = "UNSPECIFIED"
    FULL_TENSOR = "FULL_TENSOR"
    XOR_DELTA = "XOR_DELTA"


@dataclass(frozen=True)
class CompletionFence:
    """Blocking completion fence for an adapter-owned asynchronous operation."""

    _wait: Callable[[], None]

    def wait(self) -> None:
        """Block until the operation represented by this fence completes."""
        self._wait()


@dataclass(frozen=True)
class WeightVersionShardManifest:
    """Engine-neutral description of one trainer process's source buffers."""

    data: bytes
    tensor_count: int
    total_bytes: int
    transport: str

    def __post_init__(self) -> None:
        if not self.data:
            raise ValueError("manifest data must not be empty")
        if self.tensor_count <= 0:
            raise ValueError("tensor_count must be positive")
        if self.total_bytes <= 0:
            raise ValueError("total_bytes must be positive")
        if not self.transport:
            raise ValueError("transport must not be empty")

    @property
    def digest(self) -> str:
        """Return the SHA-256 digest advertised through RefitService."""
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True)
class StagedWeightVersionShardData:
    """Adapter-owned immutable buffers and their transfer manifest."""

    manifest: WeightVersionShardManifest
    publish_ready: CompletionFence
    buffer_owner: object | None = None


class TrainerEngineAdapter(ABC):
    """Engine-specific capture and staging boundary for trainer publication.

    ModelExpress owns worker registration, manifest serving, and control-plane
    publication. An implementation captures engine tensors into immutable
    source buffers and describes those buffers in a transfer manifest.
    """

    @property
    @abstractmethod
    def source_slot_id(self) -> str:
        """Return this rank's required logical contribution identifier."""

    @abstractmethod
    def bind_tensors(self, tensors: Any) -> str:
        """Bind stable engine tensors and return their logical source slot."""

    @property
    @abstractmethod
    def supported_staging_modes(self) -> frozenset[TrainerStagingMode]:
        """Return staging modes implemented by this engine adapter."""

    @property
    @abstractmethod
    def supported_payload_formats(self) -> frozenset[WeightPayloadFormat]:
        """Return payload formats implemented by this engine adapter."""

    @abstractmethod
    def stage_shard(
        self,
        *,
        tensors: Any,
        staging_mode: TrainerStagingMode,
        payload_format: WeightPayloadFormat,
    ) -> StagedWeightVersionShardData:
        """Capture one immutable, rank-local version shard."""


class WeightVersionShardManifestPublisher(Protocol):
    """Worker endpoint that makes a manifest retrievable before advertisement."""

    def publish_manifest(
        self,
        *,
        version_id: str,
        source_slot_id: str,
        manifest: WeightVersionShardManifest,
    ) -> str:
        """Publish ``manifest`` and return its ready, worker-local endpoint."""


__all__ = [
    "CompletionFence",
    "NixlMetadataProvider",
    "StagedWeightVersionShardData",
    "TrainerEngineAdapter",
    "TrainerStagingMode",
    "WeightPayloadFormat",
    "WeightVersionShardManifest",
    "WeightVersionShardManifestPublisher",
]
