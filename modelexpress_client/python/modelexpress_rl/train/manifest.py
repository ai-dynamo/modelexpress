# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-local serving for versioned trainer manifests."""

from __future__ import annotations

import hashlib
import json
import threading

import grpc

from .. import refit_pb2, refit_pb2_grpc
from .adapter import WeightVersionShardManifest


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


class WeightVersionShardManifestService(refit_pb2_grpc.RefitWorkerServiceServicer):
    """Publish and serve immutable manifests from one trainer process.

    The records intentionally share the worker process lifetime. Durable
    version and shard metadata remains in the central RefitService backend;
    large tensor buffers and their manifest remain worker-local.
    """

    def __init__(self, *, endpoint: str) -> None:
        if not endpoint.strip():
            raise ValueError("endpoint is required")
        self.endpoint = endpoint
        self._manifests: dict[tuple[str, str], WeightVersionShardManifest] = {}
        self._bindings: dict[str, bytes] = {}
        self._lock = threading.Lock()

    def publish_binding(self, manifest: bytes) -> None:
        binding_id = hashlib.sha256(manifest).hexdigest()
        with self._lock:
            self._bindings[binding_id] = manifest

    def publish_manifest(
        self,
        *,
        version_id: str,
        logical_shard_id: str,
        manifest: WeightVersionShardManifest,
    ) -> str:
        """Make one immutable manifest retrievable before returning its endpoint."""
        if not version_id.strip():
            raise ValueError("version_id is required")
        if not logical_shard_id.strip():
            raise ValueError("logical_shard_id is required")
        key = (version_id, logical_shard_id)
        with self._lock:
            existing = self._manifests.get(key)
            if existing is not None and existing != manifest:
                raise ValueError(
                    "a different manifest is already published for "
                    f"version_id={version_id!r}, logical_shard_id={logical_shard_id!r}"
                )
            self._manifests[key] = manifest
        return self.endpoint

    def release_manifest(self, *, version_id: str, logical_shard_id: str) -> None:
        """Drop a served manifest once its version is released.

        Without this the worker holds every manifest it ever published for the
        life of the process, which on a large MoE is megabytes per step.
        """
        key = (version_id, logical_shard_id)
        with self._lock:
            self._manifests.pop(key, None)

    def GetWeightVersionShardManifest(self, request, context):
        # Mesh admission reads bound coverage before a weight version exists.
        if not request.version_id:
            with self._lock:
                binding = self._bindings.get(request.logical_shard_id)
            if binding is None:
                context.abort(grpc.StatusCode.NOT_FOUND, "binding was not found")
            return refit_pb2.GetWeightVersionShardManifestResponse(
                manifest=binding, manifest_digest=request.logical_shard_id
            )
        key = (request.version_id, request.logical_shard_id)
        with self._lock:
            manifest = self._manifests.get(key)
        if manifest is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "manifest was not found")
        return refit_pb2.GetWeightVersionShardManifestResponse(
            manifest=manifest.data,
            manifest_digest=manifest.digest,
        )


__all__ = ["WeightVersionShardManifestService"]
