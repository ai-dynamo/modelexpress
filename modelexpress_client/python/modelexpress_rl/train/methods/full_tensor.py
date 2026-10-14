# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer full-tensor publication over NIXL."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from time import monotonic, sleep

import grpc

from modelexpress.refit.timing import refit_span

from ... import refit_pb2, refit_pb2_grpc
from ...version import TrainerTensorsMetadata, WeightVersionRef
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
        self._binding: TrainerTensorsMetadata | None = None

    @property
    def logical_shard_id(self) -> str:
        if self._binding is None:
            raise RuntimeError("bind_tensors() must be called before logical_shard_id")
        return self._binding.logical_shard_id

    def bind_tensors(self, tensors: Any) -> TrainerTensorsMetadata:
        self._binding = TrainerTensorsMetadata(
            logical_shard_id=self._adapter.bind_tensors(tensors),
            metadata_endpoint=self._manifest_publisher.endpoint,
        )
        self._manifest_publisher.publish_binding(self._adapter.bound_manifest)
        return self._binding

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
        version_response = self._service().GetWeightVersion(
            refit_pb2.GetWeightVersionRequest(uid=version.version_id),
            timeout=self._rpc_timeout_seconds,
        )
        if not version_response.version.HasField("trainer_mesh_id"):
            raise RuntimeError("trainer publication requires trainer_mesh_id")
        if version_response.version.trainer_mesh_generation == 0:
            raise RuntimeError(
                "trainer publication requires a positive trainer_mesh_generation; recreate the version"
            )
        if self._binding is None:
            raise RuntimeError("mesh publication requires bind_tensors()")
        logical_shard_id = self._binding.logical_shard_id
        with refit_span(
            "source_preparation",
            metadata={"staging_syncs": 1},
            accumulate_metadata=True,
            duration_key="staging_sync_s",
        ):
            staged.publish_ready.wait()
        if staged.manifest.transport.upper() != "NIXL":
            raise ValueError(
                f"unsupported shard transport {staged.manifest.transport!r}"
            )
        # A cached property that hashes the whole manifest, so the first read is
        # the digest being computed and every later one is free.
        with refit_span(
            "source_preparation",
            metadata={"manifest_digests": 1},
            accumulate_metadata=True,
            duration_key="manifest_digest_s",
        ):
            manifest_digest = staged.manifest.digest
        with refit_span(
            "setup_registration",
            metadata={"manifest_publications": 1},
            accumulate_metadata=True,
            duration_key="manifest_publish_s",
        ):
            endpoint = self._manifest_publisher.publish_manifest(
                version_id=version.version_id,
                logical_shard_id=logical_shard_id,
                manifest=staged.manifest,
            )
        if not endpoint.strip():
            raise ValueError("manifest_endpoint is required")
        shard = refit_pb2.WeightVersionShard(
            version_id=version.version_id,
            logical_shard_id=logical_shard_id,
            worker_id=self._worker_id,
            tensor_count=staged.manifest.tensor_count,
            total_bytes=staged.manifest.total_bytes,
            manifest_digest=manifest_digest,
            manifest_endpoint=endpoint,
        )
        with refit_span(
            "setup_registration",
            metadata={"publication_rpcs": 1},
            accumulate_metadata=True,
            duration_key="publication_rpc_s",
        ):
            self._service().CreateWeightVersionShard(
                refit_pb2.CreateWeightVersionShardRequest(shard=shard),
                timeout=self._rpc_timeout_seconds,
            )
        self.published.setdefault(version.version_id, []).append(staged)

    def release(self, *, version: WeightVersionRef) -> None:
        if version.version_id not in self.published:
            return
        logical_shard_id = self.logical_shard_id
        request = refit_pb2.DeleteWeightVersionShardRequest(
            version_id=version.version_id,
            logical_shard_id=logical_shard_id,
            worker_id=self._worker_id,
        )
        deadline = monotonic() + self._rpc_timeout_seconds
        # Publication buffers must outlive every generator reader lease.
        while True:
            try:
                self._service().DeleteWeightVersionShard(
                    request, timeout=max(deadline - monotonic(), 0.001)
                )
                break
            except grpc.RpcError as error:
                if (
                    error.code() is not grpc.StatusCode.FAILED_PRECONDITION
                    or error.details() != "weight version has an active lease"
                    or monotonic() >= deadline
                ):
                    raise
                sleep(min(0.05, max(deadline - monotonic(), 0)))
        self._manifest_publisher.release_manifest(
            version_id=version.version_id,
            logical_shard_id=logical_shard_id,
        )
        del self.published[version.version_id]

    def close(self) -> None:
        self.published.clear()


__all__ = ["FullTensorNixlPublicationMethod"]
