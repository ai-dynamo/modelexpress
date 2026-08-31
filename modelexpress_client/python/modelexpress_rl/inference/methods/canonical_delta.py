# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical XOR-delta preparation from object storage."""

from __future__ import annotations

from ...control import WeightVersion
from ...object_storage import ObjectStorageType
from ...s3 import S3Client
from ...train import WeightPayloadFormat
from ..plan import (
    MethodCapabilities,
    ObjectStorageUpdateSource,
    PreparedArtifact,
    PreparedCheckpointArtifact,
    ResolvedSource,
    WeightSource,
    UpdateMethod,
)
from ..receiver import (
    ObjectStorageGeneratorConfig,
    _LocalCheckpoint,
    _S3Version,
)


class CanonicalDeltaUpdateMethod(UpdateMethod):
    """Reconstruct and verify a canonical checkpoint without engine mutation."""

    def __init__(
        self,
        *,
        model_name: str,
        config: ObjectStorageGeneratorConfig,
    ) -> None:
        if config.storage_type is not ObjectStorageType.S3:
            raise ValueError("only S3 object storage is currently supported")
        self._s3 = S3Client(
            endpoint_url=config.endpoint_url,
            region_name=config.region_name,
        )
        try:
            self._checkpoint = _LocalCheckpoint(
                model_name=model_name,
                config=config,
                s3=self._s3,
            )
            self._checkpoint.initialize()
        except Exception:
            self._s3.close()
            raise
        self._active: PreparedCheckpointArtifact | None = None

    @property
    def capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            payload_formats=frozenset({WeightPayloadFormat.XOR_DELTA}),
            sources=frozenset({WeightSource.OBJECT_STORAGE}),
            artifact_type=PreparedCheckpointArtifact,
            requires_base_version=True,
        )

    def prepare(
        self,
        *,
        version: WeightVersion,
        source: ResolvedSource,
    ) -> PreparedArtifact:
        if self._active is not None:
            raise RuntimeError("release staged weight before staging another version")
        if not isinstance(source, ObjectStorageUpdateSource):
            raise TypeError("canonical delta requires an object-storage source")
        storage = source.storage
        if storage.storage_type is not ObjectStorageType.S3:
            raise ValueError("canonical delta requires S3 object storage")
        assert version.base_version_id is not None
        # TODO: Make the default object-storage fallback reconstruct the target
        # from the most recent full canonical checkpoint followed by its ordered
        # delta chain. Until then, require the local checkpoint to be the exact
        # base for this single-delta update.
        if self._checkpoint.current_version not in {
            version.base_version_id,
            version.version_id,
        }:
            raise ValueError(
                "canonical delta target does not match the exact local base"
            )
        try:
            checkpoint = self._checkpoint.prepare(
                _S3Version(
                    version_id=version.version_id,
                    base_version_id=version.base_version_id,
                    uri=storage.uri,
                )
            )
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        self._active = PreparedCheckpointArtifact(checkpoint=checkpoint)
        return self._active

    def installation_context(self, prepared: PreparedArtifact):
        if prepared is not self._active:
            raise RuntimeError("canonical delta staged weight is no longer active")
        return self._checkpoint.installation_context(prepared.checkpoint)

    def release(self, prepared: PreparedArtifact) -> None:
        if prepared is not self._active:
            raise RuntimeError("canonical delta staged weight is no longer active")
        self._active = None

    def close(self) -> None:
        self._active = None
        self._s3.close()


__all__ = ["CanonicalDeltaUpdateMethod"]
