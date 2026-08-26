# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator-engine boundary for ModelExpress RL refit installation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from modelexpress import p2p_pb2
from modelexpress.client import MxClientBase
from modelexpress_rl.train import WeightPayloadFormat


class GeneratorEngineContext(ABC):
    """Typed rank-local inputs used to construct one engine adapter."""


@dataclass(frozen=True)
class GeneratorSource:
    """One verified, version-scoped source selected for a logical slot."""

    source_slot_id: str
    worker_id: str
    manifest_endpoint: str
    manifest_digest: str
    transport: str
    manifest: bytes


@dataclass(frozen=True)
class GeneratorTransferInputs:
    """Exact-version source metadata used to validate or build a transfer plan."""

    version_id: str
    layout_signature: str
    payload_format: WeightPayloadFormat
    sources: tuple[GeneratorSource, ...]

    @property
    def physical_fingerprint(self) -> tuple:
        """Return the physical assumptions whose drift invalidates a plan."""
        return (
            self.layout_signature,
            self.payload_format,
            tuple(
                (
                    source.source_slot_id,
                    source.worker_id,
                    source.manifest_endpoint,
                    source.manifest_digest,
                    source.transport,
                )
                for source in self.sources
            ),
        )


class GeneratorEngineAdapter(ABC):
    """Engine-specific transfer planning and installation boundary."""

    @property
    @abstractmethod
    def worker_rank(self) -> int:
        """Return the rank used to match an inference P2P source."""

    @abstractmethod
    def build_p2p_identity(self, version_id: str) -> p2p_pb2.SourceIdentity:
        """Build the engine-compatible P2P identity for an exact version."""

    @abstractmethod
    def stage_peer_weight(self, source: p2p_pb2.WorkerMetadata) -> object:
        """Stage an exact version from a compatible inference peer."""

    @abstractmethod
    def publish_weight_version(
        self,
        *,
        version_id: str,
        staged: object,
        p2p_client: MxClientBase,
        worker_id: str,
    ) -> None:
        """Publish applied staging buffers as an exact-version P2P source."""

    @property
    @abstractmethod
    def supported_payload_formats(self) -> frozenset[WeightPayloadFormat]:
        """Return payload formats implemented by this adapter."""

    @abstractmethod
    def create_transfer_plan(self, inputs: GeneratorTransferInputs) -> Any:
        """Compile a reusable rank-local plan from verified source manifests."""

    @abstractmethod
    def validate_transfer_plan(
        self,
        plan: Any,
        inputs: GeneratorTransferInputs,
    ) -> bool:
        """Return whether engine and transport assumptions still permit reuse."""

    @abstractmethod
    def stage_weight(self, plan: Any) -> Any:
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
]
