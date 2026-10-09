# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical checkpoint preparation from object storage."""

from __future__ import annotations

from dataclasses import replace

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
    bootstrap_s3_checkpoint,
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
        self._model_name = model_name
        self._config = config
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
            self._initialized = self._checkpoint.initialize(
                allow_unrecorded_seed=True,
            )
        except Exception:
            self._s3.close()
            raise
        self._active: PreparedCheckpointArtifact | None = None

    @property
    def requires_full_root(self) -> bool:
        return not self._initialized

    @property
    def capabilities(self) -> MethodCapabilities:
        return MethodCapabilities(
            payload_formats=frozenset(
                {
                    WeightPayloadFormat.XOR_DELTA,
                    WeightPayloadFormat.FULL_HF_CHECKPOINT,
                }
            ),
            sources=frozenset({WeightSource.OBJECT_STORAGE}),
            artifact_type=PreparedCheckpointArtifact,
        )

    def prepare(
        self,
        *,
        version: WeightVersion,
        source: ResolvedSource,
    ) -> PreparedArtifact:
        return self.prepare_chain(((version, source),))

    def prepare_chain(
        self,
        chain: tuple[tuple[WeightVersion, ResolvedSource], ...],
    ) -> PreparedArtifact:
        if self._active is not None:
            raise RuntimeError("release staged weight before staging another version")
        versions = [self._version(version, source) for version, source in chain]
        try:
            if not self._initialized:
                if (
                    not versions
                    or versions[0].payload_format
                    is not WeightPayloadFormat.FULL_HF_CHECKPOINT
                ):
                    raise RuntimeError(
                        "S3 fallback without a cached seed requires a full replay root"
                    )
                root = versions[0]
                seed = bootstrap_s3_checkpoint(
                    model_name=self._model_name,
                    version=root,
                    refit_checkpoint_dir=self._config.refit_checkpoint_dir,
                    refit_checkpoint_max_size_gb=self._config.refit_checkpoint_max_size_gb,
                    s3=self._s3,
                )
                checkpoint = _LocalCheckpoint(
                    model_name=self._model_name,
                    config=replace(
                        self._config,
                        initial_base_version_id=root.version_id,
                        seed_checkpoint_path=seed,
                    ),
                    s3=self._s3,
                )
                checkpoint.initialize()
                self._checkpoint = checkpoint
                self._initialized = True
            checkpoint = self._checkpoint.prepare_chain(tuple(versions))
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        self._active = PreparedCheckpointArtifact(checkpoint=checkpoint)
        return self._active

    @staticmethod
    def _version(version: WeightVersion, source: ResolvedSource) -> _S3Version:
        if not isinstance(source, ObjectStorageUpdateSource):
            raise TypeError("canonical checkpoint requires an object-storage source")
        storage = source.storage
        if storage.storage_type is not ObjectStorageType.S3:
            raise ValueError("canonical checkpoint requires S3 object storage")
        if version.payload_format is WeightPayloadFormat.XOR_DELTA:
            if version.base_version_id is None:
                raise ValueError("canonical delta is missing base_version_id")
        elif version.payload_format is WeightPayloadFormat.FULL_HF_CHECKPOINT:
            if version.base_version_id is not None:
                raise ValueError("FULL_HF_CHECKPOINT must not have base_version_id")
        else:
            raise ValueError("unsupported canonical S3 payload format")
        return _S3Version(
            version_id=version.version_id,
            base_version_id=version.base_version_id,
            payload_format=version.payload_format,
            uri=storage.uri,
        )

    def installation_context(
        self,
        prepared: PreparedArtifact,
        *,
        activate: bool = True,
    ):
        """Install a prepared checkpoint and optionally activate it afterward."""
        if prepared is not self._active:
            raise RuntimeError("canonical checkpoint is no longer active")
        return self._checkpoint.installation_context(
            prepared.checkpoint,
            activate=activate,
        )

    def preparation_failed(self) -> None:
        if self._initialized:
            self._checkpoint.recover_incomplete_preparation()

    def activate(self, prepared: PreparedArtifact) -> None:
        """Activate the prepared checkpoint after distributed loading succeeds."""
        if prepared is not self._active:
            raise RuntimeError("canonical checkpoint is no longer active")
        self._checkpoint.activate(prepared.checkpoint)

    def release(self, prepared: PreparedArtifact) -> None:
        if prepared is not self._active:
            raise RuntimeError("canonical checkpoint is no longer active")
        self._active = None

    def close(self) -> None:
        self._active = None
        self._s3.close()


__all__ = ["CanonicalDeltaUpdateMethod"]
