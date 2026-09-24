# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical checkpoint preparation from object storage."""

from __future__ import annotations

import time
from contextlib import contextmanager

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
    _parse_index_manifest,
    PreparedCheckpoint,
)

from ..streaming_checkpoint import StreamedCheckpoint


class CanonicalDeltaUpdateMethod(UpdateMethod):
    """Reconstruct and verify a canonical checkpoint without engine mutation."""

    def __init__(
        self,
        *,
        model_name: str,
        config: ObjectStorageGeneratorConfig,
        stream_full_checkpoints: bool = False,
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
        self._stream_full_checkpoints = stream_full_checkpoints
        self._streams: list[StreamedCheckpoint] = []

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
            checkpoint = None
            if (
                self._stream_full_checkpoints
                and len(versions) == 1
                and versions[0].payload_format is WeightPayloadFormat.FULL_HF_CHECKPOINT
            ):
                version = versions[0]
                started = time.perf_counter()
                index_data = self._s3.get(version.uri)
                metadata, weight_map = _parse_index_manifest(index_data, is_delta=False)
                if "checksum_format" not in metadata:
                    if set(weight_map) != set(self._checkpoint.tensor_metadata):
                        raise ValueError(
                            "full HF checkpoint tensor set differs from local checkpoint"
                        )
                    stream = StreamedCheckpoint(
                        version=version,
                        index_data=index_data,
                        weight_map=weight_map,
                        tensor_metadata=self._checkpoint.tensor_metadata,
                        store=self._checkpoint.store,
                    )
                    self._streams.append(stream)
                    checkpoint = PreparedCheckpoint(
                        target_version=version.version_id,
                        path=self._checkpoint.store.full_path(version.version_id),
                        metrics={
                            "perf/mx_receive_prepare_time": time.perf_counter()
                            - started
                        },
                        streaming=stream,
                    )
            if checkpoint is None:
                for stream in self._streams:
                    stream.wait_cache()
                self._streams.clear()
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

    @contextmanager
    def installation_context(
        self,
        prepared: PreparedArtifact,
        *,
        activate: bool = True,
    ):
        """Install a prepared checkpoint and optionally activate it afterward."""
        if prepared is not self._active:
            raise RuntimeError("canonical checkpoint is no longer active")
        stream = prepared.checkpoint.streaming
        if stream is not None:
            try:
                yield
                stream.finish_cache(success=True)
            except BaseException:
                stream.finish_cache(success=False)
                raise
        else:
            with self._checkpoint.installation_context(
                prepared.checkpoint, activate=activate
            ):
                yield

    def preparation_failed(self) -> None:
        self._checkpoint.recover_incomplete_preparation()

    def activate(self, prepared: PreparedArtifact) -> None:
        """Activate the prepared checkpoint after distributed loading succeeds."""
        if prepared is not self._active:
            raise RuntimeError("canonical checkpoint is no longer active")
        if prepared.checkpoint.streaming is None:
            self._checkpoint.activate(prepared.checkpoint)

    def release(self, prepared: PreparedArtifact) -> None:
        if prepared is not self._active:
            raise RuntimeError("canonical checkpoint is no longer active")
        self._active = None

    def close(self) -> None:
        self._active = None
        for stream in self._streams:
            stream.close()
        self._streams.clear()
        self._s3.close()


__all__ = ["CanonicalDeltaUpdateMethod"]
