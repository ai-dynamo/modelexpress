# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical trainer metadata and worker-local serving."""

from __future__ import annotations

import hashlib
import json
import threading

import grpc

from .. import refit_pb2, refit_pb2_grpc
from .adapter import TrainerShardMetadata


ShardChecksumKey = tuple[str, str, int, tuple[int, ...], tuple[int, ...]]


def version_metadata(
    *,
    version_id: str,
    worker_id: str,
    logical_shard_id: str,
    stable_metadata_digest: str,
    checksums: tuple[tuple[str, int, str], ...],
) -> refit_pb2.WeightVersionShardMetadata:
    return refit_pb2.WeightVersionShardMetadata(
        version_id=version_id,
        worker_id=worker_id,
        logical_shard_id=logical_shard_id,
        stable_metadata_digest=stable_metadata_digest,
        checksums=[
            refit_pb2.ShardChecksum(tensor_name=name, shard_index=index, digest=digest)
            for name, index, digest in sorted(checksums)
        ],
    )


def version_metadata_digest(metadata: refit_pb2.WeightVersionShardMetadata) -> str:
    keys = [(row.tensor_name, row.shard_index) for row in metadata.checksums]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise ValueError("version checksums must have unique canonical tensor/shard keys")
    return hashlib.sha256(metadata.SerializeToString(deterministic=True)).hexdigest()


def bound_tensor_manifest(tensor_coverage: list[dict]) -> bytes:
    """Canonical address- and content-independent coverage of one binding."""
    tensors = []
    for tensor in tensor_coverage:
        shards = sorted(
            (
                {"shard_offset": shard["shard_offset"], "shape": shard["shape"]}
                for shard in tensor["shards"]
            ),
            key=lambda shard: (shard["shard_offset"], shard["shape"]),
        )
        tensors.append(
            {
                "name": tensor["name"],
                "dtype": tensor["dtype"],
                "elsize": tensor["elsize"],
                "full_shape": tensor["full_shape"],
                "shards": shards,
            }
        )
    tensors.sort(key=lambda tensor: tensor["name"])
    return json.dumps(
        {"tensors": tensors}, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


class RefitWorkerService(refit_pb2_grpc.RefitWorkerServiceServicer):
    """Serve current stable metadata and releasable version checksums."""

    def __init__(self, *, endpoint: str) -> None:
        if not endpoint.strip():
            raise ValueError("endpoint is required")
        self.endpoint = endpoint
        self._metadata: TrainerShardMetadata | None = None
        self._binding: tuple[str, bytes] | None = None
        self._versions: dict[tuple[str, str], bytes] = {}
        self._lock = threading.Lock()

    def publish_binding(self, manifest: bytes) -> None:
        binding_id = hashlib.sha256(manifest).hexdigest()
        with self._lock:
            self._binding = (binding_id, manifest)

    def publish_metadata(self, *, metadata: TrainerShardMetadata) -> str:
        """Serve immutable physical metadata independently of any weight version."""
        with self._lock:
            if self._metadata is not None and self._metadata.digest != metadata.digest:
                raise ValueError("release current trainer metadata before replacing it")
            self._metadata = metadata
        return self.endpoint

    def release_metadata(self) -> None:
        """Drop current physical metadata after its version records are released."""
        with self._lock:
            if self._versions:
                raise RuntimeError("trainer metadata still has published version records")
            self._metadata = None

    def publish_version_metadata(
        self, metadata: refit_pb2.WeightVersionShardMetadata
    ) -> None:
        """Serve one version's checksums after its stable metadata is published."""
        if not metadata.version_id.strip():
            raise ValueError("version_id is required")
        if not metadata.logical_shard_id.strip():
            raise ValueError("logical_shard_id is required")
        if not metadata.worker_id.strip():
            raise ValueError("worker_id is required")
        version_metadata_digest(metadata)
        encoded = metadata.SerializeToString(deterministic=True)
        key = (metadata.version_id, metadata.logical_shard_id)
        with self._lock:
            if self._metadata is None or metadata.stable_metadata_digest != self._metadata.digest:
                raise ValueError("version metadata references unpublished stable metadata")
            existing = self._versions.get(key)
            if existing is not None and existing != encoded:
                raise ValueError(
                    "different version metadata is already published for "
                    f"version_id={metadata.version_id!r}, "
                    f"logical_shard_id={metadata.logical_shard_id!r}"
                )
            self._versions[key] = encoded

    def release_version_metadata(self, *, version_id: str, logical_shard_id: str) -> None:
        """Drop version checksums after the protected version is released."""
        key = (version_id, logical_shard_id)
        with self._lock:
            self._versions.pop(key, None)

    def GetTrainerShardMetadata(
        self, request, context
    ) -> refit_pb2.GetTrainerShardMetadataResponse:
        with self._lock:
            metadata = None
            if self._metadata is not None and request.metadata_digest == self._metadata.digest:
                metadata = self._metadata.data
            elif self._binding is not None and request.metadata_digest == self._binding[0]:
                metadata = self._binding[1]
        if metadata is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "metadata was not found")
        return refit_pb2.GetTrainerShardMetadataResponse(
            metadata=metadata, metadata_digest=request.metadata_digest
        )

    def GetWeightVersionShardMetadata(
        self, request, context
    ) -> refit_pb2.GetWeightVersionShardMetadataResponse:
        with self._lock:
            encoded = self._versions.get((request.version_id, request.logical_shard_id))
        if encoded is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "version metadata was not found")
        metadata = refit_pb2.WeightVersionShardMetadata.FromString(encoded)
        return refit_pb2.GetWeightVersionShardMetadataResponse(
            metadata=metadata, version_metadata_digest=version_metadata_digest(metadata)
        )


__all__ = ["RefitWorkerService"]
