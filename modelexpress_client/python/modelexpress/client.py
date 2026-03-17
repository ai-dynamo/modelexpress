# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress Client for P2P GPU Weight Transfers.

Orchestrates NIXL/RDMA transfers between vLLM workers. The client fetches
NIXL metadata from workers via ZMQ, queries the ModelExpress server for
existing sources, and instructs workers to receive weights if found.

NIXL agents live in vLLM workers (not here) because GPU memory must be
registered by the owning process for GPUDirect RDMA.
"""

import logging
import os

import grpc

from . import p2p_pb2
from . import p2p_pb2_grpc

logger = logging.getLogger("modelexpress.client")


def _parse_server_address(address: str) -> str:
    """Strip http:// or https:// prefix from server address for gRPC."""
    if address.startswith("http://"):
        return address[7:]
    elif address.startswith("https://"):
        return address[8:]
    return address


def _get_server_url(explicit_url: str | None = None) -> str:
    """
    Resolve the ModelExpress server URL.

    Priority:
    1. Explicit ``server_url`` argument
    2. ``MODEL_EXPRESS_URL`` env var (Dynamo-consistent)
    3. ``MX_SERVER_ADDRESS`` env var (backward compat)
    4. Default ``localhost:8001``
    """
    if explicit_url:
        return _parse_server_address(explicit_url)
    url = os.environ.get(
        "MODEL_EXPRESS_URL",
        os.environ.get("MX_SERVER_ADDRESS", "localhost:8001"),
    )
    return _parse_server_address(url)


class MxClient:
    """
    Lightweight gRPC client for ModelExpress server communication.

    Provides typed methods for every P2P RPC (``PublishMetadata``,
    ``GetMetadata``, ``UpdateStatus``) so that callers
    (loaders, coordinators) never need to create gRPC channels or
    stubs directly.

    The connection is created lazily on first use.

    Args:
        server_url: Explicit server address (``host:port``).  When
            *None* the address is resolved via ``MODEL_EXPRESS_URL``
            or ``MX_SERVER_ADDRESS`` env vars, falling back to
            ``localhost:8001``.
        max_message_size: Max send/receive message size in bytes.
    """

    def __init__(
        self,
        server_url: str | None = None,
        max_message_size: int = 100 * 1024 * 1024,  # 100 MB
    ):
        self.server_url = _get_server_url(server_url)
        self._max_message_size = max_message_size
        self._channel: grpc.Channel | None = None
        self._stub: p2p_pb2_grpc.P2pServiceStub | None = None

    # -- connection management ------------------------------------------------

    @property
    def stub(self) -> p2p_pb2_grpc.P2pServiceStub:
        """Return (and lazily create) the gRPC stub."""
        if self._channel is None:
            options = [
                ("grpc.max_send_message_length", self._max_message_size),
                ("grpc.max_receive_message_length", self._max_message_size),
            ]
            self._channel = grpc.insecure_channel(self.server_url, options=options)
            self._stub = p2p_pb2_grpc.P2pServiceStub(self._channel)
            logger.debug("MxClient connected to %s", self.server_url)
        return self._stub

    def close(self) -> None:
        """Close the underlying gRPC channel."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    # -- RPC wrappers ---------------------------------------------------------

    def publish_metadata(
        self,
        model_name: str,
        workers: list[p2p_pb2.WorkerMetadata],
    ) -> bool:
        """Publish worker metadata so targets can discover this source.

        Returns *True* on success.
        """
        request = p2p_pb2.PublishMetadataRequest(
            model_name=model_name,
            workers=workers,
        )
        response = self.stub.PublishMetadata(request, timeout=30)
        if not response.success:
            logger.error("PublishMetadata failed: %s", response.message)
        return response.success

    def get_metadata(
        self, model_name: str
    ) -> p2p_pb2.GetMetadataResponse:
        """Query for existing source metadata for *model_name*."""
        request = p2p_pb2.GetMetadataRequest(model_name=model_name)
        return self.stub.GetMetadata(request, timeout=30)

    def update_status(
        self,
        model_name: str,
        worker_id: int,
        status: "p2p_pb2.SourceStatus",
    ) -> bool:
        """Update worker status.  Returns *True* on success."""
        request = p2p_pb2.UpdateStatusRequest(
            model_name=model_name,
            worker_id=worker_id,
            status=status,
        )
        response = self.stub.UpdateStatus(request, timeout=30)
        if not response.success:
            logger.error("UpdateStatus failed: %s", response.message)
        return response.success

