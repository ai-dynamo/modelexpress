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

import argparse
import json
import logging
import os
import stat
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

import grpc

from . import p2p_pb2
from . import p2p_pb2_grpc
from .types import GetMetadataResponse, TensorDescriptor, WorkerMetadata

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
        response = self.stub.PublishMetadata(request)
        if not response.success:
            logger.error("PublishMetadata failed: %s", response.message)
        return response.success

    def get_metadata(
        self, model_name: str
    ) -> p2p_pb2.GetMetadataResponse:
        """Query for existing source metadata for *model_name*."""
        request = p2p_pb2.GetMetadataRequest(model_name=model_name)
        return self.stub.GetMetadata(request)

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
        response = self.stub.PublishReady(request)
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
        return self.stub.GetReady(request)


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

    # =========================================================================
    # Connect to vLLM Workers
    # =========================================================================

    def _start_weight_server(self, zmq_address_base: str, timeout_s: int = 300) -> None:
        """Start weight server via collective_rpc."""
        url = f"{self.engine_address}/collective_rpc"
        payload = json.dumps({
            "method": "start_weight_server",
            "args": [zmq_address_base],
        }).encode()

        logger.info(f"Starting weight server via collective_rpc: {url}")
        start = time.time()

        while (time.time() - start) < timeout_s:
            try:
                req = Request(url, data=payload, method="POST")
                with urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        logger.info("Weight server started successfully")
                        return
            except URLError:
                pass
            except Exception as e:
                logger.debug(f"collective_rpc attempt failed: {e}")
            time.sleep(1)

        raise TimeoutError(f"Failed to start weight server after {timeout_s}s")

    def connect_to_workers(
        self,
        zmq_addresses: list[str],
        zmq_address_base: str,
        timeout_ms: int = 30000,
        wait_for_sockets: bool = True,
        wait_timeout_s: int = 300,
    ) -> None:
        """
        Connect to vLLM workers via ZMQ.

        Args:
            zmq_addresses: List of ZMQ addresses, one per worker
            zmq_address_base: Base ZMQ address pattern passed to workers
            timeout_ms: Connection timeout in milliseconds
            wait_for_sockets: If True, wait for IPC socket files to exist
            wait_timeout_s: Max seconds to wait for sockets (default 5 min)
        """
        if not ZMQ_AVAILABLE:
            raise RuntimeError("ZMQ is not available")

        # Start weight server via collective_rpc before waiting for sockets
        self._start_weight_server(zmq_address_base, wait_timeout_s)

        # Wait for IPC socket files to exist before connecting
        if wait_for_sockets:
            self._wait_for_ipc_sockets(zmq_addresses, wait_timeout_s)

        self._zmq_context = zmq.Context()

        for rank, addr in enumerate(zmq_addresses):
            socket = self._zmq_context.socket(zmq.REQ)
            socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
            socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
            socket.connect(addr)
            self._zmq_sockets[rank] = socket
            logger.info(f"Connected to worker {rank} at {addr}")

    def _wait_for_ipc_sockets(
        self,
        zmq_addresses: list[str],
        timeout_s: int = 300,
    ) -> None:
        """Wait for IPC socket files to exist."""
        ipc_paths = []
        for addr in zmq_addresses:
            if addr.startswith("ipc://"):
                ipc_paths.append((addr, Path(addr[6:])))

        if not ipc_paths:
            return

        logger.info(f"Waiting for {len(ipc_paths)} IPC socket(s) to be ready...")
        start = time.time()

        pending = list(ipc_paths)
        while pending and (time.time() - start) < timeout_s:
            still_pending = []
            for addr, path in pending:
                if path.exists() and stat.S_ISSOCK(path.stat().st_mode):
                    logger.info(f"Socket ready: {addr}")
                else:
                    still_pending.append((addr, path))
            pending = still_pending
            if pending:
                time.sleep(0.5)

        if pending:
            missing = [addr for addr, _ in pending]
            raise TimeoutError(
                f"Timed out waiting for IPC sockets after {timeout_s}s: {missing}"
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

    # =========================================================================
    # Main Run Loop
    # =========================================================================

    def run(
        self,
        zmq_addresses: list[str],
        zmq_address_base: str,
        source_only: bool = False,
    ) -> TransferStats | None:
        """
        Main client execution flow.

        Args:
            zmq_addresses: List of ZMQ addresses for vLLM workers
            zmq_address_base: Base ZMQ address pattern passed to workers
            source_only: If True, skip receiving and only publish metadata (source mode)

        Returns:
            Transfer stats if receiving, None if becoming source
        """
        stats = None

        # 1. Connect to vLLM workers and fetch metadata
        logger.info("Connecting to vLLM workers...")
        self.connect_to_workers(zmq_addresses, zmq_address_base)
        self.fetch_all_worker_metadata()

        if source_only:
            # Source-only mode: skip checking for existing sources
            logger.info(f"Source-only mode: publishing metadata for '{self.model_name}'")
        else:
            # 2. Query for existing source
            response = self._get_metadata(self.model_name)

            if response.found:
                # 3a. Receive from source
                logger.info(
                    f"Found existing source for '{self.model_name}'. "
                    "Receiving weights from source..."
                )
                stats = self.receive_from_source(response.workers)
            else:
                # 3b. No source found
                logger.info(
                    f"No existing source found for '{self.model_name}'. "
                    "Becoming source..."
                )

        # 4. Publish metadata
        self.publish_metadata()

        logger.info("Client ready - metadata published")
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


def generate_zmq_addresses(base: str, tp_size: int) -> list[str]:
    """Generate rank-specific ZMQ addresses from a base pattern.

    Args:
        base: Base ZMQ address (e.g., "ipc:///tmp/vllm.sock")
        tp_size: Number of workers (tensor parallel size)

    Returns:
        List of ZMQ addresses, one per worker rank
    """
    addresses = []
    for rank in range(tp_size):
        if base.startswith("ipc://"):
            base_path = base[6:]
            if "." in base_path:
                parts = base_path.rsplit(".", 1)
                addr = f"ipc://{parts[0]}-{rank}.{parts[1]}"
            else:
                addr = f"ipc://{base_path}-{rank}"
        else:
            addr = base.replace(":5555", f":{5555 + rank}")
        addresses.append(addr)
    return addresses


def main():
    parser = argparse.ArgumentParser(
        description="ModelExpress Client for P2P GPU Weight Transfers"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name for coordination (e.g., meta-llama/Llama-3.1-70B)",
    )

    # ZMQ address options (mutually exclusive convenience)
    zmq_group = parser.add_mutually_exclusive_group()
    zmq_group.add_argument(
        "--zmq-addresses",
        type=str,
        nargs="+",
        help="Explicit ZMQ addresses for vLLM workers",
    )
    zmq_group.add_argument(
        "--zmq-base",
        type=str,
        default=os.environ.get("MX_ZMQ_ADDRESS", "ipc:///tmp/vllm.sock"),
        help="Base ZMQ address pattern (used with --tp-size to auto-generate)",
    )

    parser.add_argument(
        "--tp-size",
        type=int,
        default=int(os.environ.get("MX_TP_SIZE", "1")),
        help="Tensor parallel size (number of GPU workers)",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=os.environ.get("MX_SERVER", "localhost:8001"),
        help="ModelExpress server address",
    )
    parser.add_argument(
        "--engine-address",
        type=str,
        default=os.environ.get("MX_ENGINE_ADDRESS", "http://localhost:8000"),
        help="vLLM engine HTTP address for collective_rpc",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        default=os.environ.get("MX_SOURCE_ONLY", "").lower() in ("1", "true", "yes"),
        help="Source-only mode: publish metadata but never try to receive from other sources",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine ZMQ base and addresses
    zmq_base = args.zmq_base
    if args.zmq_addresses:
        zmq_addresses = args.zmq_addresses
    else:
        zmq_addresses = generate_zmq_addresses(zmq_base, args.tp_size)

    logger.info(f"TP size: {args.tp_size}, ZMQ base: {zmq_base}, ZMQ addresses: {zmq_addresses}")

    client = ModelExpressClient(
        model_name=args.model_name,
        server_address=args.server_address,
        engine_address=args.engine_address,
    )

    try:
        stats = client.run(zmq_addresses, zmq_base, source_only=args.source_only)

        if stats:
            logger.info(
                "\n" + "=" * 60 + "\n" +
                "Transfer Summary" + "\n" +
                "=" * 60 + "\n" +
                f"Total tensors:       {stats.total_tensors}\n" +
                f"Total size:          {stats.total_bytes / 1e9:.2f} GB\n" +
                f"Total duration:      {stats.duration:.2f} s\n" +
                f"Aggregate bandwidth: {stats.bandwidth_gbps:.2f} Gbps\n" +
                "=" * 60 + "\n"
            )

        # Keep running to serve as source
        logger.info("Client is running. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
