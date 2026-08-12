# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed boundary over the three minimal revision-catalog RPCs."""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import grpc

from modelexpress import revision_pb2, revision_pb2_grpc

from .manifest import RevisionManifest, RevisionRecord


@runtime_checkable
class RevisionCatalog(Protocol):
    """Exact metadata operations available to the publisher and orchestrator."""

    def publish_revision(self, manifest: RevisionManifest) -> RevisionRecord: ...

    def get_revision(
        self, model_id: str, target_version: str
    ) -> RevisionRecord: ...

    def commit_revision(
        self, model_id: str, target_version: str
    ) -> RevisionRecord: ...


class GrpcRevisionCatalog:
    """Concrete :class:`RevisionCatalog` over the generated gRPC service."""

    def __init__(
        self,
        endpoint: str | None = None,
        stub=None,
        timeout: float = 10.0,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("catalog RPC timeout must be finite and positive")
        if (endpoint is None) == (stub is None):
            raise ValueError(
                "GrpcRevisionCatalog needs exactly one of endpoint or stub"
            )
        self._channel = None
        if stub is None:
            assert endpoint is not None
            if endpoint.startswith("https://"):
                target = endpoint.removeprefix("https://")
                self._channel = grpc.secure_channel(
                    target, grpc.ssl_channel_credentials()
                )
            else:
                target = endpoint.removeprefix("http://")
                self._channel = grpc.insecure_channel(target)
            stub = revision_pb2_grpc.RevisionCatalogServiceStub(self._channel)
        self._stub = stub
        self._timeout = timeout

    def __enter__(self) -> GrpcRevisionCatalog:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None

    def publish_revision(self, manifest: RevisionManifest) -> RevisionRecord:
        response = self._stub.PublishRevision(
            revision_pb2.PublishRevisionRequest(manifest=manifest.to_proto()),
            timeout=self._timeout,
        )
        return RevisionRecord.from_proto(response)

    def get_revision(self, model_id: str, target_version: str) -> RevisionRecord:
        response = self._stub.GetRevision(
            revision_pb2.GetRevisionRequest(
                model_id=model_id,
                target_version=target_version,
            ),
            timeout=self._timeout,
        )
        return RevisionRecord.from_proto(response)

    def commit_revision(self, model_id: str, target_version: str) -> RevisionRecord:
        response = self._stub.CommitRevision(
            revision_pb2.CommitRevisionRequest(
                model_id=model_id,
                target_version=target_version,
            ),
            timeout=self._timeout,
        )
        return RevisionRecord.from_proto(response)
