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
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional

import grpc

from . import p2p_pb2
from . import p2p_pb2_grpc
from .types import GetMetadataResponse, TensorDescriptor, WorkerMetadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("modelexpress.client")

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


class ModelExpressClient:
    """
    Client that coordinates NIXL transfers for vLLM model weights.

    This class:
    1. Connects to vLLM workers via ZMQ to get NIXL metadata
    2. Queries ModelExpress server for existing sources
    3. Orchestrates RDMA transfers (executed by vLLM workers)
    4. Publishes metadata to become a source

    Tensor Parallelism (TP > 1):
        For TP > 1, the client connects to multiple vLLM worker processes,
        one per GPU. Each worker has its own ZMQ socket address. Transfers
        are rank-matched: source worker rank 0 transfers to target rank 0.

    Args:
        model_name: Model identifier for metadata lookup
        server_address: ModelExpress gRPC server address
    """

    def __init__(
        self,
        model_name: str,
        server_address: str = "localhost:8001",
        engine_address: str = "http://localhost:8000",
    ):
        self.model_name = model_name
        self.server_address = server_address
        self.engine_address = engine_address.rstrip("/")

        # ZMQ connections to workers
        self._zmq_context: Any = None
        self._zmq_sockets: dict[int, Any] = {}

        # Worker metadata (from vLLM workers)
        self._worker_metadata: dict[int, WorkerMetadata] = {}

        # gRPC connection
        self._channel: Optional[grpc.Channel] = None
        self._stub: Optional[p2p_pb2_grpc.P2pServiceStub] = None
        self._connect_grpc()

        logger.info(f"Client initialized: model={model_name}")

    # =========================================================================
    # gRPC Connection
    # =========================================================================

    def _connect_grpc(self) -> None:
        """Establish connection to the ModelExpress server."""
        # Increase message size limits for large models like DeepSeek-V3
        options = [
            ('grpc.max_send_message_length', 100 * 1024 * 1024),  # 100MB
            ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
        ]
        self._channel = grpc.insecure_channel(self.server_address, options=options)
        self._stub = p2p_pb2_grpc.P2pServiceStub(self._channel)
        logger.info(f"Connected to ModelExpress at {self.server_address}")

    def _close_grpc(self) -> None:
        """Close the gRPC connection."""
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None

    @property
    def _grpc_stub(self) -> p2p_pb2_grpc.P2pServiceStub:
        """Get the gRPC stub, connecting if necessary."""
        if self._stub is None:
            self._connect_grpc()
        return self._stub

    def _get_metadata(self, model_name: str) -> GetMetadataResponse:
        """Query for existing source metadata for a model."""
        request = p2p_pb2.GetMetadataRequest(model_name=model_name)
        response = self._grpc_stub.GetMetadata(request)

        workers = []
        if response.found:
            for w in response.workers:
                tensors = [
                    TensorDescriptor(
                        name=t.name,
                        addr=t.addr,
                        size=t.size,
                        device_id=t.device_id,
                        dtype=t.dtype,
                    )
                    for t in w.tensors
                ]
                workers.append(WorkerMetadata(
                    worker_rank=w.worker_rank,
                    nixl_metadata=w.nixl_metadata,
                    tensors=tensors,
                ))

        return GetMetadataResponse(
            found=response.found,
            workers=workers,
        )

    def fetch_worker_metadata(self, worker_rank: int) -> WorkerMetadata:
        """Fetch NIXL metadata and tensor descriptors from a worker."""
        if worker_rank not in self._zmq_sockets:
            raise RuntimeError(f"Not connected to worker {worker_rank}")

        socket = self._zmq_sockets[worker_rank]
        socket.send_pyobj({"cmd": "get_metadata"})
        response = socket.recv_pyobj()

        if "error" in response:
            raise RuntimeError(f"Worker {worker_rank} error: {response['error']}")

        tensors = [
            TensorDescriptor(
                name=t["name"],
                addr=t["addr"],
                size=t["size"],
                device_id=t["device_id"],
                dtype=t["dtype"],
            )
            for t in response["tensors"]
        ]

        metadata = WorkerMetadata(
            worker_rank=response["worker_rank"],
            nixl_metadata=response["nixl_metadata"],
            tensors=tensors,
        )
        self._worker_metadata[worker_rank] = metadata

        total_size = sum(t.size for t in tensors)
        logger.info(
            f"Worker {worker_rank}: fetched metadata for {len(tensors)} tensors, "
            f"total size: {total_size / 1e9:.2f} GB"
        )
        return metadata

    def fetch_all_worker_metadata(self) -> None:
        """Fetch metadata from all connected workers."""
        for rank in self._zmq_sockets:
            self.fetch_worker_metadata(rank)

    def publish_metadata(self) -> bool:
        """Publish worker metadata to become a source."""
        workers_proto = []
        for rank in sorted(self._worker_metadata.keys()):
            worker = self._worker_metadata[rank]
            tensors_proto = [
                p2p_pb2.TensorDescriptor(
                    name=t.name,
                    addr=t.addr,
                    size=t.size,
                    device_id=t.device_id,
                    dtype=t.dtype,
                )
                for t in worker.tensors
            ]
            workers_proto.append(p2p_pb2.WorkerMetadata(
                worker_rank=rank,
                nixl_metadata=worker.nixl_metadata,
                tensors=tensors_proto,
            ))

        request = p2p_pb2.PublishMetadataRequest(
            model_name=self.model_name,
            workers=workers_proto,
        )
        response = self._grpc_stub.PublishMetadata(request)

        if response.success:
            logger.info(f"Published metadata for model '{self.model_name}' with {len(workers_proto)} workers")
        else:
            logger.error(f"Failed to publish metadata: {response.message}")
        return response.success

    def _receive_worker(
        self,
        worker_rank: int,
        source_worker: WorkerMetadata,
        timeout_seconds: float | None,
    ) -> tuple[int, int]:
        """
        Instruct a worker to receive from a source via RDMA.

        Args:
            worker_rank: Local worker rank
            source_worker: Source worker metadata
            timeout_seconds: Transfer timeout

        Returns:
            Tuple of (bytes_transferred, tensors_transferred)
        """
        if worker_rank not in self._zmq_sockets:
            raise RuntimeError(f"Not connected to worker {worker_rank}")

        socket = self._zmq_sockets[worker_rank]

        # Build source tensor list for the request
        source_tensors = [
            {
                "name": t.name,
                "addr": t.addr,
                "size": t.size,
                "device_id": t.device_id,
            }
            for t in source_worker.tensors
        ]

        # Send receive_from request to worker
        request = {
            "cmd": "receive_from",
            "nixl_metadata": source_worker.nixl_metadata,
            "tensors": source_tensors,
            "timeout_seconds": timeout_seconds,
        }
        socket.send_pyobj(request)
        response = socket.recv_pyobj()

        if not response.get("success", False):
            raise RuntimeError(f"Worker {worker_rank} transfer failed: {response.get('error', 'Unknown error')}")

        return response.get("bytes_transferred", 0), response.get("tensors_transferred", 0)

    def receive_from_source(
        self,
        source_workers: list[WorkerMetadata],
        timeout_seconds: float = 300,
    ) -> TransferStats:
        """
        Receive weights from a source via NIXL RDMA.

        Transfers are executed in parallel by the vLLM workers.

        Args:
            source_workers: Worker metadata from the source instance
            timeout_seconds: Maximum time to wait for transfer

        Returns:
            Transfer statistics
        """
        stats = TransferStats()
        stats.start_time = time.perf_counter()

        # Match source workers to local workers by rank
        source_by_rank = {w.worker_rank: w for w in source_workers}

        # Execute transfers in parallel
        errors = []
        with ThreadPoolExecutor(max_workers=len(self._zmq_sockets)) as executor:
            futures = {}
            for rank in self._zmq_sockets:
                if rank not in source_by_rank:
                    logger.warning(f"No source worker for rank {rank}")
                    continue
                future = executor.submit(
                    self._receive_worker,
                    rank,
                    source_by_rank[rank],
                    timeout_seconds,
                )
                futures[future] = rank

            for future in as_completed(futures):
                rank = futures[future]
                try:
                    worker_bytes, worker_tensors = future.result()
                    stats.total_bytes += worker_bytes
                    stats.total_tensors += worker_tensors
                    logger.info(f"Worker {rank}: transferred {worker_tensors} tensors")
                except Exception as e:
                    errors.append(f"Worker {rank}: {e}")

        if errors:
            raise RuntimeError(f"Transfer failed: {'; '.join(errors)}")

        stats.end_time = time.perf_counter()

        bandwidth_gbps = stats.bandwidth_gbps
        logger.info(
            f"Transfer complete: {stats.total_tensors} tensors, "
            f"{stats.total_bytes / 1e9:.2f} GB in {stats.duration:.2f}s "
            f"({bandwidth_gbps:.1f} Gbps)"
        )

        return stats

    def shutdown(self) -> None:
        """Clean up resources."""
        # Close ZMQ connections
        for rank, socket in self._zmq_sockets.items():
            try:
                socket.send_pyobj({"cmd": "done"})
                socket.recv_string()
            except Exception:
                pass
            socket.close()
        self._zmq_sockets.clear()

        if self._zmq_context:
            self._zmq_context.term()
            self._zmq_context = None

        # Close gRPC connection
        self._close_grpc()

        self._worker_metadata.clear()

        logger.info("Client shutdown complete")
