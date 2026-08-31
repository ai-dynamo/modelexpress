# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer full-tensor publication over NIXL."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ... import refit_pb2, refit_pb2_grpc
from ...version import WeightVersionRef
from ..adapter import (
    StagedWeightVersionShardData,
    TrainerEngineAdapter,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionShardManifestPublisher,
)


class FullTensorNixlPublicationMethod:
    """Publish adapter-owned immutable shards and retrievable NIXL manifests."""

    def __init__(
        self,
        *,
        adapter: TrainerEngineAdapter,
        staging_mode: TrainerStagingMode,
        payload_format: WeightPayloadFormat,
        manifest_publisher: WeightVersionShardManifestPublisher,
        service: Callable[[], refit_pb2_grpc.RefitServiceStub],
        worker_id: str,
        rpc_timeout_seconds: float,
    ) -> None:
        if staging_mode not in adapter.supported_staging_modes:
            raise ValueError(
                f"adapter does not support staging mode {staging_mode.value}"
            )
        if payload_format not in adapter.supported_payload_formats:
            raise ValueError(
                f"adapter does not support payload format {payload_format.value}"
            )
        self._adapter = adapter
        self._staging_mode = staging_mode
        self._payload_format = payload_format
        self._manifest_publisher = manifest_publisher
        self._service = service
        self._worker_id = worker_id
        self._rpc_timeout_seconds = rpc_timeout_seconds
        self.published: dict[str, list[StagedWeightVersionShardData]] = {}

    @property
    def source_slot_id(self) -> str:
        return self._adapter.source_slot_id

    def bind_tensors(self, tensors: Any) -> str:
        return self._adapter.bind_tensors(tensors)

    def stage(
        self,
        *,
        version: WeightVersionRef,
        tensors: Any,
    ) -> StagedWeightVersionShardData:
        del version
        if tensors is None:
            raise ValueError("tensors is required for NIXL publication")
        return self._adapter.stage_shard(
            tensors=tensors,
            staging_mode=self._staging_mode,
            payload_format=self._payload_format,
        )

    def publish(
        self,
        *,
        version: WeightVersionRef,
        staged: object,
    ) -> None:
        if not isinstance(staged, StagedWeightVersionShardData):
            raise TypeError("full-tensor publication received an invalid shard")
        staged.publish_ready.wait()
        if staged.manifest.transport.upper() != "NIXL":
            raise ValueError(
                f"unsupported shard transport {staged.manifest.transport!r}"
            )
        endpoint = self._manifest_publisher.publish_manifest(
            version_id=version.version_id,
            source_slot_id=self.source_slot_id,
            manifest=staged.manifest,
        )
        if not endpoint.strip():
            raise ValueError("manifest_endpoint is required")
        shard = refit_pb2.WeightVersionShard(
            version_id=version.version_id,
            source_slot_id=self.source_slot_id,
            worker_id=self._worker_id,
            tensor_count=staged.manifest.tensor_count,
            total_bytes=staged.manifest.total_bytes,
            manifest_digest=staged.manifest.digest,
            manifest_endpoint=endpoint,
        )
        self._service().CreateWeightVersionShard(
            refit_pb2.CreateWeightVersionShardRequest(shard=shard),
            timeout=self._rpc_timeout_seconds,
        )
        self.published.setdefault(version.version_id, []).append(staged)

    def release(self, *, version: WeightVersionRef) -> None:
        if version.version_id not in self.published:
            return
        self._service().DeleteWeightVersionShard(
            refit_pb2.DeleteWeightVersionShardRequest(
                version_id=version.version_id,
                source_slot_id=self.source_slot_id,
                worker_id=self._worker_id,
            ),
            timeout=self._rpc_timeout_seconds,
        )
        del self.published[version.version_id]

    def close(self) -> None:
        self.published.clear()


__all__ = ["FullTensorNixlPublicationMethod"]
