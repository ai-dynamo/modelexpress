# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight per-worker gRPC server for serving tensor manifests.

Each source worker runs a WorkerGrpcServer that serves its tensor descriptor
list directly to targets via the WorkerService RPC. This keeps MB-scale tensor
metadata out of the centralized discovery layer (MX server / DHT).
"""

from __future__ import annotations

import logging
from concurrent import futures

import grpc

from . import p2p_pb2
from . import p2p_pb2_grpc

logger = logging.getLogger("modelexpress.worker_server")

_MX_WORKER_GRPC_PORT_DEFAULT = 6555


class _WorkerServiceServicer(p2p_pb2_grpc.WorkerServiceServicer):
    """Serves an immutable tensor descriptor list."""

    def __init__(self, tensor_protos: list[p2p_pb2.TensorDescriptor]):
        self._tensor_protos = tensor_protos

    def GetTensorManifest(self, request, context):
        return p2p_pb2.GetTensorManifestResponse(tensors=self._tensor_protos)


class WorkerGrpcServer:
    """Per-worker gRPC server that serves the tensor manifest.

    Args:
        tensor_protos: List of TensorDescriptor protos for this worker's GPU.
        port: Port to listen on.
        host: Bind address (default "0.0.0.0").
    """

    def __init__(
        self,
        tensor_protos: list[p2p_pb2.TensorDescriptor],
        port: int,
        host: str = "0.0.0.0",
    ):
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        servicer = _WorkerServiceServicer(tensor_protos)
        p2p_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, self._server)
        self._address = f"{host}:{port}"
        self._server.add_insecure_port(self._address)

    def start(self) -> None:
        """Start serving in the background."""
        self._server.start()
        logger.info("WorkerGrpcServer listening on %s", self._address)

    def stop(self, grace: float = 5.0) -> None:
        """Graceful shutdown."""
        self._server.stop(grace)
        logger.info("WorkerGrpcServer stopped")


def fetch_tensor_manifest(endpoint: str) -> list[p2p_pb2.TensorDescriptor]:
    """Fetch the tensor manifest from a source worker's gRPC server.

    Args:
        endpoint: "host:port" of the source worker's WorkerService.

    Returns:
        List of TensorDescriptor protos.
    """
    channel = grpc.insecure_channel(endpoint)
    stub = p2p_pb2_grpc.WorkerServiceStub(channel)
    try:
        response = stub.GetTensorManifest(
            p2p_pb2.GetTensorManifestRequest(), timeout=30
        )
        return list(response.tensors)
    finally:
        channel.close()
