# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""P2P/NIXL implementation of the artifact transport interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .. import p2p_pb2
from .artifact_transfer import (
    ArtifactBundle,
    ArtifactTransfer,
    PublishedArtifactSource,
    discover_artifact_source,
    publish_artifact_source,
    transfer_artifact_from_worker,
)
from .artifact_transport import (
    ArtifactCacheMiss,
    ArtifactInstallDisposition,
    ArtifactInstallState,
    ArtifactInstallStatus,
    ArtifactTransport,
    ArtifactTransportContext,
    ArtifactTransportUnavailable,
    PublicationHandle,
    StagedArtifact,
)
from .publish import _get_worker_server

if TYPE_CHECKING:
    from ..nixl_transfer import NixlTransferManager
    from .worker_server import WorkerGrpcServer


@dataclass
class P2PPublicationHandle(PublicationHandle):
    """Adapter around the existing live P2P publication resources."""

    source: PublishedArtifactSource
    transport: str = "p2p"

    @property
    def identifier(self) -> str:
        return self.source.endpoint.mx_source_id

    def stop(self) -> None:
        self.source.stop()


class P2PArtifactTransport(ArtifactTransport):
    """Fetch and publish artifacts through ModelExpress/NIXL workers."""

    name = "p2p"
    # P2P always exposes the local artifact after startup, independently of
    # whether this worker fetched a remote copy or rebuilt it locally.
    publish_requires_install_state = False
    # Retry transient registration and source-publication failures while the
    # serving worker remains available.
    retry_publish_on_failure = True

    def state_after_cache_miss(self) -> ArtifactInstallState:
        """Suppress duplicate remote discovery after a P2P cache miss."""
        return ArtifactInstallState(ArtifactInstallStatus.ATTEMPTED)

    def resolve_install_state(
        self, state: ArtifactInstallState
    ) -> ArtifactInstallDisposition:
        """Any existing P2P marker suppresses another fetch in this pod."""
        return ArtifactInstallDisposition.SKIP

    def should_publish(self, state: ArtifactInstallState | None) -> bool:
        """A live P2P source is published after either a hit or a miss."""
        return True

    def __init__(
        self,
        *,
        mx_client: Any | None = None,
        nixl_manager: NixlTransferManager | None = None,
        worker_id: str = "",
        worker_rank: int = 0,
        device_id: int = 0,
        worker_grpc_server: WorkerGrpcServer | None = None,
    ) -> None:
        """Bind P2P-specific runtime dependencies to this adapter."""
        self._mx_client = mx_client
        self._nixl_manager = nixl_manager
        self._worker_id = worker_id
        self._worker_rank = worker_rank
        self._device_id = device_id
        self._worker_grpc_server = worker_grpc_server

    def fetch(
        self,
        artifact: ArtifactTransfer,
        identity: p2p_pb2.SourceIdentity,
        context: ArtifactTransportContext,
    ) -> StagedArtifact:
        if self._mx_client is None or self._nixl_manager is None:
            raise ArtifactTransportUnavailable(
                "P2P artifact fetch requires ModelExpress client and NIXL manager"
            )
        try:
            source = discover_artifact_source(
                self._mx_client,
                identity,
                worker_rank=None,
                node_rank=context.node_rank,
                accelerator=context.accelerator,
            )
            header = transfer_artifact_from_worker(
                source.worker_grpc_endpoint,
                source.mx_source_id,
                source.artifact_id,
                self._nixl_manager,
                target_file_paths=artifact.target_file_paths(),
            )
        except LookupError as exc:
            raise ArtifactCacheMiss(str(exc)) from exc
        except Exception as exc:
            raise ArtifactTransportUnavailable(
                f"P2P artifact fetch failed for {artifact.name}: {exc}"
            ) from exc
        return StagedArtifact(header=header, transport=self.name)

    def publish(
        self,
        artifact: ArtifactTransfer,
        identity: p2p_pb2.SourceIdentity,
        bundle: ArtifactBundle,
        context: ArtifactTransportContext,
    ) -> PublicationHandle:
        if self._mx_client is None or self._nixl_manager is None:
            raise ArtifactTransportUnavailable(
                "P2P artifact publish requires ModelExpress client and NIXL manager"
            )
        worker_grpc_server = self._worker_grpc_server or _get_worker_server(
            self._device_id
        )
        if worker_grpc_server is None:
            raise ArtifactTransportUnavailable(
                "P2P artifact publish requires a worker gRPC server"
            )
        try:
            source = publish_artifact_source(
                mx_client=self._mx_client,
                transfer=artifact,
                bundle=bundle,
                identity=identity,
                nixl_manager=self._nixl_manager,
                worker_id=self._worker_id,
                worker_grpc_server=worker_grpc_server,
                worker_rank=self._worker_rank,
                node_rank=context.node_rank,
                accelerator=context.accelerator,
            )
        except Exception as exc:
            raise ArtifactTransportUnavailable(
                f"P2P artifact publish failed for {artifact.name}: {exc}"
            ) from exc
        return P2PPublicationHandle(source)
