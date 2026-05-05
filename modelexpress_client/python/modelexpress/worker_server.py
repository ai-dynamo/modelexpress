# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-worker gRPC server for P2P tensor manifest exchange.

When MX_P2P_METADATA=1, each source worker starts a WorkerGrpcServer
that serves its tensor descriptors directly to target workers via the
GetTensorManifest RPC. This avoids storing MB-scale tensor lists in the
central metadata server.
"""

from __future__ import annotations

import logging
from concurrent import futures

import grpc

from . import p2p_pb2
from . import p2p_pb2_grpc

logger = logging.getLogger("modelexpress.worker_server")


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
    """Manages a gRPC server for the WorkerService on a source worker."""

    def __init__(
        self,
        tensor_protos: list[p2p_pb2.TensorDescriptor],
        mx_source_id: str,
        port: int = 0,
        metadata_endpoint: str = "",
        agent_name: str = "",
        worker_rank: int = 0,
    ):
        self._tensor_protos = tensor_protos
        self._mx_source_id = mx_source_id
        self._requested_port = port
        self._metadata_endpoint = metadata_endpoint
        self._agent_name = agent_name
        self._worker_rank = worker_rank
        self._server: grpc.Server | None = None
        self._port: int | None = None

    @property
    def port(self) -> int | None:
        return self._port

    def start(self) -> int:
        """Start the gRPC server. Returns the actual bound port."""
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        servicer = WorkerServiceServicer(
            tensor_protos=self._tensor_protos,
            mx_source_id=self._mx_source_id,
            metadata_endpoint=self._metadata_endpoint,
            agent_name=self._agent_name,
            worker_rank=self._worker_rank,
        )
        p2p_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, self._server)

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
