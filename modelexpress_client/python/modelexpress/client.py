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
import time
import uuid
from dataclasses import dataclass

import grpc

from . import p2p_pb2
from . import p2p_pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
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
    ``GetMetadata``, ``PublishReady``, ``GetReady``) so that callers
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
        self.session_id: str = str(uuid.uuid4())

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

    def publish_ready(
        self,
        model_name: str,
        worker_id: int,
        session_id: str,
        metadata_hash: str,
        nixl_ready: bool = True,
        stability_verified: bool = True,
    ) -> bool:
        """Publish a source-ready flag.  Returns *True* on success."""
        request = p2p_pb2.PublishReadyRequest(
            model_name=model_name,
            worker_id=worker_id,
            session_id=session_id,
            metadata_hash=metadata_hash,
            nixl_ready=nixl_ready,
            stability_verified=stability_verified,
        )
        response = self.stub.PublishReady(request, timeout=30)
        if not response.success:
            logger.error("PublishReady failed: %s", response.message)
            return False
        return True

    def get_ready(
        self, model_name: str, worker_id: int
    ) -> p2p_pb2.GetReadyResponse:
        """Check whether the source is ready for *model_name* / *worker_id*."""
        request = p2p_pb2.GetReadyRequest(
            model_name=model_name,
            worker_id=worker_id,
        )
        return self.stub.GetReady(request, timeout=30)

    # -- coordination helpers -------------------------------------------------

    def wait_for_ready(
        self,
        model_name: str,
        worker_id: int,
        timeout_seconds: int = 7200,
        poll_interval: int = 10,
    ) -> tuple[bool, str | None, str | None]:
        """Poll until source is ready.

        Returns:
            (success, session_id, metadata_hash)
        """
        start_time = time.time()
        logger.info("[Worker %d] Waiting for source ready flag...", worker_id)

        while time.time() - start_time < timeout_seconds:
            try:
                response = self.get_ready(model_name, worker_id)
                if response.found and response.ready:
                    sid = response.session_id
                    mhash = response.metadata_hash
                    logger.info(
                        "[Worker %d] Source ready! session=%s, hash=%s",
                        worker_id,
                        sid[:8] if sid else "N/A",
                        mhash[:8] if mhash else "N/A",
                    )
                    return True, sid, mhash
            except Exception as e:
                logger.warning("[Worker %d] Error checking ready flag: %s", worker_id, e)

            time.sleep(poll_interval)
            elapsed = int(time.time() - start_time)
            if elapsed % 60 == 0:
                logger.info(
                    "[Worker %d] Still waiting for source ready (%ds/%ds)...",
                    worker_id, elapsed, timeout_seconds,
                )

        logger.error(
            "[Worker %d] Timeout waiting for source ready after %ds",
            worker_id, timeout_seconds,
        )
        return False, None, None

    def check_session_changed(
        self,
        model_name: str,
        worker_id: int,
        cached_session_id: str | None,
    ) -> tuple[bool, str | None]:
        """Check if source session changed (indicates restart).

        Returns:
            (changed, new_session_id)
        """
        if cached_session_id is None:
            return False, None

        try:
            response = self.get_ready(model_name, worker_id)
            if response.found:
                current_session = response.session_id
                if current_session and current_session != cached_session_id:
                    logger.warning(
                        "[Worker %d] Source restarted! cached=%s != current=%s",
                        worker_id,
                        cached_session_id[:8],
                        current_session[:8],
                    )
                    return True, current_session
        except Exception as e:
            logger.warning("[Worker %d] Error checking session: %s", worker_id, e)

        return False, cached_session_id


ZMQ_AVAILABLE = False
zmq = None
try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    logger.warning("ZMQ is not available - cannot connect to vLLM workers")


@dataclass
class TransferStats:
    """Statistics for a transfer operation."""
    total_bytes: int = 0
    total_tensors: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def bandwidth_gbps(self) -> float:
        if self.duration > 0:
            return (self.total_bytes * 8) / (self.duration * 1e9)
        return 0.0
