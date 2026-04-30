# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-worker gRPC server hosting peer-facing services.

When MX_P2P_METADATA=1, each source worker starts a WorkerGrpcServer that
serves:

* ``WorkerService.GetTensorManifest`` - tensor descriptors for the P2P
  NIXL/RDMA path. Avoids storing MB-scale tensor lists in the central
  metadata server.
* ``ModelService.{StreamModelFiles, ListModelFiles, EnsureModelDownloaded}``
  - peer-served local model cache. Lets other Python or Rust clients pull
  cached files directly from this worker without round-tripping through
  the central MX server. Backed by :class:`PeerModelServiceServicer`.
"""

from __future__ import annotations

import logging
from concurrent import futures
from pathlib import Path
from typing import Optional

import grpc

from . import model_pb2_grpc
from . import p2p_pb2
from . import p2p_pb2_grpc
from .peer_model_service import PeerModelServiceServicer

logger = logging.getLogger("modelexpress.worker_server")


# Default thread-pool size. One thread is held per concurrent streaming RPC,
# so this caps simultaneous fan-in. Bumped from 4 to 16 to support model-file
# streaming alongside tensor-manifest fetches.
DEFAULT_MAX_WORKERS = 16


class WorkerServiceServicer(p2p_pb2_grpc.WorkerServiceServicer):
    """Serves tensor descriptors for a single source worker."""

    def __init__(
        self,
        tensor_protos: list[p2p_pb2.TensorDescriptor],
        mx_source_id: str,
        metadata_endpoint: str = "",
        agent_name: str = "",
        worker_rank: int = 0,
    ):
        self._tensor_protos = tensor_protos
        self._mx_source_id = mx_source_id
        self._metadata_endpoint = metadata_endpoint
        self._agent_name = agent_name
        self._worker_rank = worker_rank

    def GetTensorManifest(self, request, context):
        if request.mx_source_id and request.mx_source_id != self._mx_source_id:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"mx_source_id mismatch: expected {self._mx_source_id}, "
                f"got {request.mx_source_id}",
            )
        return p2p_pb2.GetTensorManifestResponse(
            tensors=self._tensor_protos,
            mx_source_id=self._mx_source_id,
            metadata_endpoint=self._metadata_endpoint,
            agent_name=self._agent_name,
            worker_rank=self._worker_rank,
        )


class WorkerGrpcServer:
    """Manages the per-worker gRPC server.

    Hosts both ``WorkerService`` (tensor manifest) and ``ModelService``
    (peer file serving). The ModelService side reads from the local
    HuggingFace cache and serves chunked file streams identical to those
    emitted by the central MX server.
    """

    def __init__(
        self,
        tensor_protos: list[p2p_pb2.TensorDescriptor],
        mx_source_id: str,
        port: int = 0,
        metadata_endpoint: str = "",
        agent_name: str = "",
        worker_rank: int = 0,
        *,
        enable_model_service: bool = True,
        cache_root: Optional[Path] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
    ):
        self._tensor_protos = tensor_protos
        self._mx_source_id = mx_source_id
        self._requested_port = port
        self._metadata_endpoint = metadata_endpoint
        self._agent_name = agent_name
        self._worker_rank = worker_rank
        self._enable_model_service = enable_model_service
        self._cache_root = cache_root
        self._max_workers = max_workers
        self._server: grpc.Server | None = None
        self._port: int | None = None

    @property
    def port(self) -> int | None:
        return self._port

    def start(self) -> int:
        """Start the gRPC server. Returns the actual bound port."""
        self._server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self._max_workers)
        )

        worker_servicer = WorkerServiceServicer(
            tensor_protos=self._tensor_protos,
            mx_source_id=self._mx_source_id,
            metadata_endpoint=self._metadata_endpoint,
            agent_name=self._agent_name,
            worker_rank=self._worker_rank,
        )
        p2p_pb2_grpc.add_WorkerServiceServicer_to_server(
            worker_servicer, self._server
        )

        model_status = "disabled"
        if self._enable_model_service:
            try:
                model_servicer = PeerModelServiceServicer(
                    cache_root=self._cache_root
                )
                model_pb2_grpc.add_ModelServiceServicer_to_server(
                    model_servicer, self._server
                )
                model_status = f"cache={model_servicer._cache_root}"
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "ModelService init failed (%s); peer file serving disabled",
                    exc,
                )

        if self._requested_port:
            self._port = self._server.add_insecure_port(
                f"[::]:{self._requested_port}"
            )
        else:
            self._port = self._server.add_insecure_port("[::]:0")

        self._server.start()
        logger.info(
            f"WorkerGrpcServer started on port {self._port} "
            f"(mx_source_id={self._mx_source_id}, "
            f"{len(self._tensor_protos)} tensors, "
            f"max_workers={self._max_workers}, "
            f"ModelService={model_status})"
        )
        return self._port

    def stop(self, grace: float = 5.0) -> None:
        if self._server is not None:
            self._server.stop(grace)
            logger.info("WorkerGrpcServer stopped")


def fetch_tensor_manifest(
    endpoint: str,
    mx_source_id: str,
    timeout: float = 30.0,
) -> list[p2p_pb2.TensorDescriptor]:
    """Fetch tensor descriptors directly from a source worker's WorkerService."""
    channel = grpc.insecure_channel(endpoint)
    stub = p2p_pb2_grpc.WorkerServiceStub(channel)
    request = p2p_pb2.GetTensorManifestRequest(mx_source_id=mx_source_id)
    response = stub.GetTensorManifest(request, timeout=timeout)
    channel.close()
    logger.info(
        f"Fetched {len(response.tensors)} tensors from worker at {endpoint}"
    )
    return list(response.tensors)
