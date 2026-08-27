# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator-engine boundary for ModelExpress RL refit installation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from modelexpress import p2p_pb2
from modelexpress.client import MxClientBase
from modelexpress_rl.object_storage import ObjectStorageSource
from modelexpress_rl.train import WeightPayloadFormat


class GeneratorEngineContext(ABC):
    """Typed rank-local inputs used to construct one engine adapter."""


@dataclass(frozen=True)
class NixlGeneratorSource:
    """Worker-hosted NIXL manifest for one source."""

    manifest_endpoint: str
    manifest: bytes


@dataclass(frozen=True)
class GeneratorSource:
    """One version-scoped source selected for a logical slot."""

    source_slot_id: str
    worker_id: str
    manifest_digest: str
    transport: NixlGeneratorSource

    @property
    def physical_fingerprint(self) -> tuple:
        """Return the transport identity that controls plan reuse."""
        return (
            "NIXL",
            self.transport.manifest_endpoint,
            self.manifest_digest,
        )


@dataclass(frozen=True)
class GeneratorTransferInputs:
    """Exact-version source metadata passed to one engine adapter."""

    version_id: str
    base_version_id: str | None
    layout_signature: str
    payload_format: WeightPayloadFormat
    sources: tuple[GeneratorSource, ...]
    object_storage: ObjectStorageSource | None = None

    @property
    def physical_fingerprint(self) -> tuple:
        """Return the physical assumptions whose drift invalidates a plan."""
        return (
            self.base_version_id,
            self.layout_signature,
            self.payload_format,
            self.object_storage,
            tuple(
                (
                    source.source_slot_id,
                    source.worker_id,
                    source.physical_fingerprint,
                )
                for source in self.sources
            ),
        )


class GeneratorEngineAdapter(ABC):
    """Engine-specific staging and installation boundary."""

    @property
    def worker_rank(self) -> int:
        """Return the rank used to match an inference P2P source."""
        raise NotImplementedError

    def build_p2p_identity(self, version_id: str) -> p2p_pb2.SourceIdentity:
        """Build the engine-compatible P2P identity for an exact version."""
        raise NotImplementedError

    def stage_peer_weight(self, source: p2p_pb2.WorkerMetadata) -> object:
        """Stage an exact version from a compatible inference peer."""
        raise NotImplementedError

    def publish_weight_version(
        self,
        *,
        version_id: str,
        staged: object,
        p2p_client: MxClientBase,
        worker_id: str,
    ) -> None:
        """Publish applied staging buffers as an exact-version P2P source."""
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_payload_formats(self) -> frozenset[WeightPayloadFormat]:
        """Return payload formats implemented by this adapter."""

    def create_transfer_plan(self, inputs: GeneratorTransferInputs) -> Any:
        """Compile a reusable rank-local plan from verified source manifests."""
        raise NotImplementedError

    def validate_transfer_plan(
        self,
        plan: Any,
        inputs: GeneratorTransferInputs,
    ) -> bool:
        """Return whether engine and transport assumptions still permit reuse."""
        raise NotImplementedError

    @abstractmethod
    def stage_weight(self, inputs: GeneratorTransferInputs) -> Any:
        """Transfer and verify one version without changing live weights."""

    @abstractmethod
    def apply_weight(self, staged: Any) -> Any:
        """Install a successfully verified staged version."""

    @abstractmethod
    def release_staged_weight(self, staged: Any) -> None:
        """Release adapter-owned local staging buffers."""

    @abstractmethod
    def close(self) -> None:
        """Release engine-adapter transport and worker resources."""


__all__ = [
    "GeneratorEngineContext",
    "GeneratorEngineAdapter",
    "GeneratorSource",
    "GeneratorTransferInputs",
    "NixlGeneratorSource",
]
