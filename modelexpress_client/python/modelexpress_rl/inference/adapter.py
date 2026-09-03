# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator-engine boundary for ModelExpress RL refit installation."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass

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


__all__ = [
    "GeneratorEngineContext",
    "GeneratorSource",
    "GeneratorTransferInputs",
    "NixlGeneratorSource",
]
