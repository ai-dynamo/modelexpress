# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-worker gRPC server for P2P tensor manifest exchange.

When MX_P2P_METADATA=1, each source worker starts a WorkerGrpcServer
that serves its tensor descriptors directly to target workers via the
GetTensorManifest RPC. This avoids storing MB-scale tensor lists in the
central metadata server.

In the peer-direct world (MX_METADATA_BACKEND=peer-direct), the same
server additionally answers ListWorkerSources so the peer-direct client
can enumerate what a discovered peer is serving without calling into a
central coordinator.
"""

from __future__ import annotations

import logging
import threading
from concurrent import futures

import grpc

from . import p2p_pb2
from . import p2p_pb2_grpc

logger = logging.getLogger("modelexpress.worker_server")


class WorkerServiceServicer(p2p_pb2_grpc.WorkerServiceServicer):
    """Serves tensor descriptors and source enumeration for a single worker.

    Holds one source's data for v1 (a vLLM worker loads one model per
    process). The ListWorkerSources response schema is plural so the
    structure can grow to multi-tenant workers without a protocol break.
    """

    def __init__(
        self,
        tensor_protos: list[p2p_pb2.TensorDescriptor],
        mx_source_id: str,
        worker_id: str = "",
        model_name: str = "",
        worker_rank: int = 0,
        status: "p2p_pb2.SourceStatus" = p2p_pb2.SOURCE_STATUS_UNKNOWN,
        worker_metadata: "p2p_pb2.WorkerMetadata | None" = None,
    ):
        self._tensor_protos = tensor_protos
        self._mx_source_id = mx_source_id
        self._worker_id = worker_id
        self._model_name = model_name
        self._worker_rank = worker_rank
        self._status = status
        self._worker_metadata = worker_metadata
        self._status_lock = threading.Lock()

    def set_status(self, status: "p2p_pb2.SourceStatus") -> None:
        """Update the advertised worker status (thread-safe)."""
        with self._status_lock:
            self._status = status

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
        )

    def ListWorkerSources(self, request, context):
        with self._status_lock:
            status = self._status
        ref = p2p_pb2.SourceInstanceRef(
            mx_source_id=self._mx_source_id,
            worker_id=self._worker_id,
            model_name=self._model_name,
            worker_rank=self._worker_rank,
            status=status,
        )
        return p2p_pb2.ListWorkerSourcesResponse(sources=[ref])

    def GetWorkerMetadata(self, request, context):
        """Return the full WorkerMetadata for a source served by this worker.

        Peer-direct analog of P2pService.GetMetadata. Callers use this to
        learn metadata_endpoint, agent_name, worker_grpc_endpoint, status,
        and tensors without a central server round-trip.
        """
        if request.mx_source_id and request.mx_source_id != self._mx_source_id:
            return p2p_pb2.GetWorkerMetadataResponse(
                found=False,
                mx_source_id=request.mx_source_id,
                worker_id=request.worker_id,
            )
        if request.worker_id and self._worker_id and request.worker_id != self._worker_id:
            return p2p_pb2.GetWorkerMetadataResponse(
                found=False,
                mx_source_id=request.mx_source_id,
                worker_id=request.worker_id,
            )
        if self._worker_metadata is None:
            # Construction-site didn't supply full metadata; we only know
            # enough for tensor/source enumeration, not for NIXL handshake.
            return p2p_pb2.GetWorkerMetadataResponse(
                found=False,
                mx_source_id=self._mx_source_id,
                worker_id=self._worker_id,
            )

        response_worker = p2p_pb2.WorkerMetadata()
        response_worker.CopyFrom(self._worker_metadata)
        with self._status_lock:
            response_worker.status = self._status
        # Tensors live in tensor_protos; populate if the stored metadata
        # doesn't already carry them (the usual case for P2P mode where
        # the registered metadata has tensors stripped).
        if not response_worker.tensors:
            response_worker.tensors.extend(self._tensor_protos)

        return p2p_pb2.GetWorkerMetadataResponse(
            found=True,
            worker=response_worker,
            mx_source_id=self._mx_source_id,
            worker_id=self._worker_id,
        )


class WorkerGrpcServer:
    """Manages a gRPC server for the WorkerService on a source worker."""

    def __init__(
        self,
        tensor_protos: list[p2p_pb2.TensorDescriptor],
        mx_source_id: str,
        port: int = 0,
        worker_id: str = "",
        model_name: str = "",
        worker_rank: int = 0,
        status: "p2p_pb2.SourceStatus" = p2p_pb2.SOURCE_STATUS_UNKNOWN,
        worker_metadata: "p2p_pb2.WorkerMetadata | None" = None,
    ):
        self._tensor_protos = tensor_protos
        self._mx_source_id = mx_source_id
        self._requested_port = port
        self._worker_id = worker_id
        self._model_name = model_name
        self._worker_rank = worker_rank
        self._initial_status = status
        self._worker_metadata = worker_metadata
        self._server: grpc.Server | None = None
        self._port: int | None = None
        self._servicer: WorkerServiceServicer | None = None

    @property
    def port(self) -> int | None:
        return self._port

    def set_status(self, status: "p2p_pb2.SourceStatus") -> None:
        """Update the status the servicer reports for ListWorkerSources."""
        if self._servicer is not None:
            self._servicer.set_status(status)
        else:
            # Server not started yet; apply on next start().
            self._initial_status = status

    def start(self) -> int:
        """Start the gRPC server. Returns the actual bound port."""
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        self._servicer = WorkerServiceServicer(
            tensor_protos=self._tensor_protos,
            mx_source_id=self._mx_source_id,
            worker_id=self._worker_id,
            model_name=self._model_name,
            worker_rank=self._worker_rank,
            status=self._initial_status,
            worker_metadata=self._worker_metadata,
        )
        p2p_pb2_grpc.add_WorkerServiceServicer_to_server(self._servicer, self._server)

        if self._requested_port:
            self._port = self._server.add_insecure_port(f"[::]:{self._requested_port}")
        else:
            self._port = self._server.add_insecure_port("[::]:0")

        self._server.start()
        logger.info(
            f"WorkerGrpcServer started on port {self._port} "
            f"(mx_source_id={self._mx_source_id}, "
            f"{len(self._tensor_protos)} tensors)"
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


def fetch_worker_sources(
    endpoint: str,
    timeout: float = 5.0,
) -> list[p2p_pb2.SourceInstanceRef]:
    """List the sources a discovered peer is currently serving.

    Intended for use by the peer-direct metadata client, which calls this
    once per discovered peer to populate its local peer-to-sources cache.
    """
    channel = grpc.insecure_channel(endpoint)
    try:
        stub = p2p_pb2_grpc.WorkerServiceStub(channel)
        request = p2p_pb2.ListWorkerSourcesRequest()
        response = stub.ListWorkerSources(request, timeout=timeout)
        return list(response.sources)
    finally:
        channel.close()


def fetch_worker_metadata(
    endpoint: str,
    mx_source_id: str,
    worker_id: str,
    timeout: float = 10.0,
) -> p2p_pb2.GetWorkerMetadataResponse:
    """Fetch full WorkerMetadata from a peer for one source.

    Peer-direct analog of MxClient.get_metadata - the peer itself is the
    scattered source of truth in this mode.
    """
    channel = grpc.insecure_channel(endpoint)
    try:
        stub = p2p_pb2_grpc.WorkerServiceStub(channel)
        request = p2p_pb2.GetWorkerMetadataRequest(
            mx_source_id=mx_source_id,
            worker_id=worker_id,
        )
        return stub.GetWorkerMetadata(request, timeout=timeout)
    finally:
        channel.close()
