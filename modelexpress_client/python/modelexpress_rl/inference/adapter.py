# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator-engine boundary for ModelExpress RL refit installation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from modelexpress_rl.s3 import S3Object
from modelexpress_rl.train import WeightPayloadFormat


class GeneratorEngineContext(ABC):
    """Typed rank-local inputs used to construct one engine adapter."""


@dataclass(frozen=True)
class NixlGeneratorSource:
    """Worker-hosted NIXL manifest for one source."""

    manifest_endpoint: str
    manifest: bytes


@dataclass(frozen=True)
class S3GeneratorSource:
    """Canonical S3 root for one source."""

    location: S3Object


@dataclass(frozen=True)
class GeneratorSource:
    """One version-scoped source selected for a logical slot."""

    source_slot_id: str
    worker_id: str
    manifest_digest: str
    transport: NixlGeneratorSource | S3GeneratorSource

    @property
    def physical_fingerprint(self) -> tuple:
        """Return the transport identity that controls plan reuse."""
        if isinstance(self.transport, NixlGeneratorSource):
            return (
                "NIXL",
                self.transport.manifest_endpoint,
                self.manifest_digest,
            )
        location = self.transport.location
        return (
            "S3",
            location.bucket,
            location.key,
            location.object_version,
            location.checksum,
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

    @property
    def physical_fingerprint(self) -> tuple:
        """Return the physical assumptions whose drift invalidates a plan."""
        return (
            self.base_version_id,
            self.layout_signature,
            self.payload_format,
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
    @abstractmethod
    def supported_payload_formats(self) -> frozenset[WeightPayloadFormat]:
        """Return payload formats implemented by this adapter."""

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
    "S3GeneratorSource",
]
