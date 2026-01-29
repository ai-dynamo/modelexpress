# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress vLLM Worker Extension for GPU weight transfers.

Registers GPU tensors with NIXL for GPUDirect RDMA and serves metadata
to the ModelExpress client via ZMQ. Each worker creates its own NIXL agent
because GPU memory must be registered by the owning process.

Usage: --worker-extension-cls modelexpress.vllm_extension.MxWorkerExtension
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any

import torch

from .nixl_transfer import NixlTransferManager, is_nixl_available
from .types import TensorDescriptor

logger = logging.getLogger("modelexpress.vllm_extension")

ZMQ_AVAILABLE = False
zmq = None
try:
    import zmq
    ZMQ_AVAILABLE = True
except ImportError:
    pass


class MxWorkerExtension:
    """
    vLLM Worker Extension that exposes model weights for RDMA transfers.

    This class is loaded into vLLM workers via `worker_extension_cls`.
    It uses NixlTransferManager to register GPU tensors with NIXL for
    GPUDirect RDMA and provides tensor descriptors + NIXL metadata to
    the client via ZMQ.

    vLLM injects these attributes:
        - self.model_runner: The model runner with loaded model
        - self.device: The CUDA device for this worker
    """

    def start_weight_server(self, zmq_address_base: str | None = None) -> str | None:
        """
        Start ZMQ server to serve weight tensor metadata to client.

        This method:
        1. Builds weight info from model parameters
        2. Initializes NixlTransferManager and registers tensors (for GPUDirect RDMA)
        3. Starts ZMQ server to serve tensor descriptors + NIXL metadata

        Args:
            zmq_address_base: Base ZMQ address pattern. If None, reads from MX_ZMQ_ADDRESS.

        Returns:
            The rank-specific ZMQ address this worker is listening on, or None if failed.
        """
        if not ZMQ_AVAILABLE:
            logger.error("ZMQ not available, cannot start weight server")
            return None

        if getattr(self, "_server_thread", None) is not None:
            return getattr(self, "_zmq_address", None)

        zmq_address_base = zmq_address_base or os.environ.get("MX_ZMQ_ADDRESS")
        if not zmq_address_base:
            logger.error("No ZMQ address provided and MX_ZMQ_ADDRESS not set")
            return None

        worker_rank = self._get_worker_rank()
        self._worker_rank = worker_rank

        # Build weight infos and register with NIXL
        self._build_weight_infos()
        self._initialize_nixl_manager()

        # Create rank-specific ZMQ address
        zmq_address = self._create_zmq_address(zmq_address_base, worker_rank)
        self._zmq_address = zmq_address

        if getattr(self, "_zmq_context", None) is None:
            self._zmq_context = zmq.Context()

        self._zmq_socket = self._zmq_context.socket(zmq.REP)
        self._zmq_socket.bind(zmq_address)

        self._stop_event = threading.Event()

        self._server_thread = threading.Thread(
            target=self._serve_loop,
            daemon=True,
            name="mx-weight-server",
        )
        self._server_thread.start()

        logger.info(f"Worker {worker_rank}: Weight server started at {zmq_address}")
        return zmq_address

    def _create_zmq_address(self, zmq_address_base: str, worker_rank: int) -> str:
        """Create rank-specific ZMQ address from base pattern."""
        if zmq_address_base.startswith("ipc://"):
            base_path = zmq_address_base[6:]
            if "." in base_path:
                parts = base_path.rsplit(".", 1)
                zmq_address = f"ipc://{parts[0]}-{worker_rank}.{parts[1]}"
            else:
                zmq_address = f"ipc://{base_path}-{worker_rank}"
            socket_path = Path(zmq_address[6:])
            socket_path.parent.mkdir(parents=True, exist_ok=True)
            socket_path.unlink(missing_ok=True)
        else:
            zmq_address = zmq_address_base.replace(":5555", f":{5555 + worker_rank}")
        return zmq_address

    def _build_weight_infos(self) -> None:
        """Build weight info cache with tensor data."""
        if getattr(self, "_tensors", None) is not None:
            return

        self._tensors: dict[str, torch.Tensor] = {}
        model = self.model_runner.model

        for name, param in model.named_parameters():
            if not param.is_cuda:
                continue

            # CRITICAL: Use param.data directly, NOT .contiguous()!
            # .contiguous() can create a COPY, which means RDMA would write
            # to the copy instead of the actual model weights.
            # vLLM uses param.data for inference, so we MUST register the same memory.
            self._tensors[name] = param.data

        total_size = sum(t.numel() * t.element_size() for t in self._tensors.values())
        logger.info(
            f"Worker {self._worker_rank}: Built {len(self._tensors)} tensors, "
            f"total size: {total_size / 1e9:.2f} GB"
        )

    def _initialize_nixl_manager(self) -> None:
        """Initialize NixlTransferManager and register tensors for GPUDirect RDMA."""
        if getattr(self, "_nixl_manager", None) is not None:
            return

        if not is_nixl_available():
            logger.warning("NIXL not available, skipping RDMA registration")
            self._nixl_manager = None
            return

        device_id = self._tensors[next(iter(self._tensors))].device.index or 0

        try:
            self._nixl_manager = NixlTransferManager(
                agent_name=f"vllm-{uuid.uuid4().hex[:8]}-{self._worker_rank}",
                device_id=device_id,
            )
            self._nixl_manager.initialize()
            self._nixl_manager.register_tensors(self._tensors)

        except Exception as e:
            logger.error(f"Worker {self._worker_rank}: Failed to initialize NIXL: {e}")
            self._nixl_manager = None

    def _get_nixl_metadata(self) -> bytes:
        """Get NIXL metadata for this worker."""
        if self._nixl_manager is None:
            return b""
        return self._nixl_manager.nixl_metadata

    def _get_tensor_descriptors(self) -> list[TensorDescriptor]:
        """Get tensor descriptors for this worker."""
        if self._nixl_manager is None:
            return []
        return self._nixl_manager.tensor_descriptors

    def _serve_loop(self) -> None:
        """Background loop to serve ZMQ requests."""
        self._zmq_socket.setsockopt(zmq.RCVTIMEO, 100)

        while not self._stop_event.is_set():
            try:
                done = self._serve_once()
                if done:
                    break
            except zmq.Again:
                continue
            except zmq.ZMQError:
                break

    def _serve_once(self) -> bool:
        """Handle one request from the client."""
        request = self._zmq_socket.recv_pyobj()
        cmd = request if isinstance(request, str) else request.get("cmd")

        if cmd == "get_metadata":
            return self._handle_get_metadata()
        if cmd == "receive_from":
            return self._handle_receive_from(request)
        if cmd == "done":
            self._zmq_socket.send_string("ok")
            self._stop_event.set()
            return True

        self._zmq_socket.send_pyobj({"error": f"Unknown command: {cmd}"})
        return False

    def _handle_get_metadata(self) -> bool:
        """Handle get_metadata request."""
        tensor_descriptors = self._get_tensor_descriptors()
        response = {
            "worker_rank": self._worker_rank,
            "nixl_metadata": self._get_nixl_metadata(),
            "tensors": [
                {
                    "name": desc.name,
                    "addr": desc.addr,
                    "size": desc.size,
                    "device_id": desc.device_id,
                    "dtype": desc.dtype,
                }
                for desc in tensor_descriptors
            ],
        }
        self._zmq_socket.send_pyobj(response)
        return False

    def _handle_receive_from(self, request: dict) -> bool:
        """
        Handle receive_from request - RDMA READ from remote source.

        Request format:
        {
            "cmd": "receive_from",
            "nixl_metadata": bytes,  # Remote agent metadata
            "tensors": [{"name": str, "addr": int, "size": int, "device_id": int}, ...],
            "timeout_seconds": float | None,
        }
        """
        if self._nixl_manager is None:
            self._zmq_socket.send_pyobj({
                "success": False,
                "error": "NIXL not available",
            })
            return False

        try:
            source_tensors = [
                TensorDescriptor(
                    name=t["name"],
                    addr=t["addr"],
                    size=t["size"],
                    device_id=t["device_id"],
                    dtype=t.get("dtype", ""),
                )
                for t in request["tensors"]
            ]

            total_bytes, total_tensors, _ = self._nixl_manager.receive_from_source(
                source_metadata=request["nixl_metadata"],
                source_tensors=source_tensors,
                timeout_seconds=request.get("timeout_seconds"),
            )

            self._zmq_socket.send_pyobj({
                "success": True,
                "bytes_transferred": total_bytes,
                "tensors_transferred": total_tensors,
            })

        except Exception as e:
            self._zmq_socket.send_pyobj({
                "success": False,
                "error": str(e),
            })

        return False

    def _get_worker_rank(self) -> int:
        """Get the TP rank of this worker."""
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            return get_tensor_model_parallel_rank()
        except (ImportError, RuntimeError):
            pass

        if hasattr(self, "device") and hasattr(self.device, "index"):
            return self.device.index or 0

        return 0

    def stop_weight_server(self) -> None:
        """Stop the ZMQ weight server and cleanup NIXL resources."""
        stop_event = getattr(self, "_stop_event", None)
        if stop_event is not None:
            stop_event.set()

        server_thread = getattr(self, "_server_thread", None)
        if server_thread is not None:
            server_thread.join(timeout=2.0)
            self._server_thread = None

        zmq_socket = getattr(self, "_zmq_socket", None)
        if zmq_socket is not None:
            zmq_socket.close()
            self._zmq_socket = None

        zmq_context = getattr(self, "_zmq_context", None)
        if zmq_context is not None:
            zmq_context.term()
            self._zmq_context = None

        nixl_manager = getattr(self, "_nixl_manager", None)
        if nixl_manager is not None:
            nixl_manager.shutdown()
            self._nixl_manager = None
