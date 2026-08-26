# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-local serving for versioned trainer manifests."""

from __future__ import annotations

import threading

import grpc

from .. import refit_pb2, refit_pb2_grpc
from .adapter import WeightVersionShardManifest


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
        self._lock = threading.Lock()

    def publish_manifest(
        self,
        *,
        version_id: str,
        source_slot_id: str,
        manifest: WeightVersionShardManifest,
    ) -> str:
        """Make one immutable manifest retrievable before returning its endpoint."""
        if not version_id.strip():
            raise ValueError("version_id is required")
        if not source_slot_id.strip():
            raise ValueError("source_slot_id is required")
        key = (version_id, source_slot_id)
        with self._lock:
            existing = self._manifests.get(key)
            if existing is not None and existing != manifest:
                raise ValueError(
                    "a different manifest is already published for "
                    f"version_id={version_id!r}, source_slot_id={source_slot_id!r}"
                )
            self._manifests[key] = manifest
        return self.endpoint

    def GetWeightVersionShardManifest(self, request, context):
        key = (request.version_id, request.source_slot_id)
        with self._lock:
            manifest = self._manifests.get(key)
        if manifest is None:
            context.abort(grpc.StatusCode.NOT_FOUND, "manifest was not found")
        return refit_pb2.GetWeightVersionShardManifestResponse(
            manifest=manifest.data,
            manifest_digest=manifest.digest,
        )


__all__ = ["WeightVersionShardManifestService"]
