# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer-engine boundary for ModelExpress RL refit publication."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Any, Protocol

from .. import refit_pb2

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
    """How a trainer adapter preserves a version's immutable source bytes.

    Prefer IN_PLACE for synchronous updates with stable, matching-dtype sources
    and no trainer-side conversion. Keep source bytes immutable until retirement.
    Prefer COPY_TO_HOST otherwise. COPY_TO_DEVICE trades substantial extra VRAM
    for lower latency and should be an explicit, measured exception.
    """

    UNSPECIFIED = "UNSPECIFIED"
    COPY_TO_DEVICE = "COPY_TO_DEVICE"
    COPY_TO_HOST = "COPY_TO_HOST"
    WRITE_TO_STORAGE = "WRITE_TO_STORAGE"
    IN_PLACE = "IN_PLACE"


class WeightPayloadFormat(str, Enum):
    """Weight representation used by one published version."""

    UNSPECIFIED = "UNSPECIFIED"
    FULL_TENSOR = "FULL_TENSOR"
    XOR_DELTA = "XOR_DELTA"
    FULL_HF_CHECKPOINT = "FULL_HF_CHECKPOINT"


@dataclass(frozen=True)
class CompletionFence:
    """Blocking completion fence for an adapter-owned asynchronous operation."""

    _wait: Callable[[], None]

    def wait(self) -> None:
        """Block until the operation represented by this fence completes."""
        self._wait()


@dataclass(frozen=True)
class TrainerShardMetadata:
    """Engine-neutral description of one trainer process's source buffers."""

    data: bytes
    tensor_count: int
    total_bytes: int
    transport: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "data", bytes(self.data))
        if not self.data:
            raise ValueError("metadata data must not be empty")
        if self.tensor_count <= 0:
            raise ValueError("tensor_count must be positive")
        if self.total_bytes <= 0:
            raise ValueError("total_bytes must be positive")
        if not self.transport:
            raise ValueError("transport must not be empty")

    @cached_property
    def digest(self) -> str:
        """Return the SHA-256 digest advertised through RefitService."""
        return hashlib.sha256(self.data).hexdigest()


@dataclass(frozen=True)
class StagedWeightVersionShardData:
    """Adapter-owned immutable buffers and their transfer manifest."""

    metadata: TrainerShardMetadata
    publish_ready: CompletionFence
    buffer_owner: object | None = None
    checksums: tuple[tuple[str, int, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "checksums", tuple(tuple(row) for row in self.checksums))


class TrainerEngineAdapter(ABC):
    """Engine-specific capture and staging boundary for trainer publication.

    ModelExpress owns worker registration, manifest serving, and control-plane
    publication. An implementation captures engine tensors into immutable
    source buffers and describes those buffers in a transfer manifest.
    """

    @property
    @abstractmethod
    def logical_shard_id(self) -> str:
        """Return this rank's required logical contribution identifier."""

    @abstractmethod
    def bind_tensors(self, tensors: Any) -> str:
        """Hash canonical wire coverage without staging or publishing weights."""

    @property
    @abstractmethod
    def bound_manifest(self) -> bytes:
        """Return canonical coverage computed by bind_tensors()."""

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


class TrainerShardMetadataPublisher(Protocol):
    """Worker endpoint that makes a manifest retrievable before advertisement."""

    @property
    def endpoint(self) -> str: ...

    def publish_binding(self, manifest: bytes) -> None:
        """Serve immutable tensor coverage before joining a trainer mesh."""

    def publish_metadata(self, *, metadata: TrainerShardMetadata) -> str:
        """Publish stable metadata and return its ready worker-local endpoint."""

    def release_metadata(self) -> None:
        """Stop serving the current physical metadata after publication retirement."""

    def publish_version_metadata(
        self, metadata: refit_pb2.WeightVersionShardMetadata
    ) -> None:
        """Publish immutable version checksums referencing already served metadata."""

    def release_version_metadata(self, *, version_id: str, logical_shard_id: str) -> None:
        """Stop serving a released version's checksums."""


__all__ = [
    "CompletionFence",
    "NixlMetadataProvider",
    "StagedWeightVersionShardData",
    "TrainerEngineAdapter",
    "TrainerStagingMode",
    "WeightPayloadFormat",
    "TrainerShardMetadata",
    "TrainerShardMetadataPublisher",
]
