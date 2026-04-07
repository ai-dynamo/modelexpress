# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Inference-side weight receiver for RL refit via ModelExpress.

Wraps NixlTransferManager + MxClient to discover updated weights
published by the training side, pull them via RDMA, and yield
``(name, tensor)`` pairs compatible with vLLM's ``model.load_weights()``.

Typical usage in a vLLM worker extension::

    receiver = MxRefitReceiver("inference-0", device_id=0, mx_server_url="mx-server:8001")
    receiver.initialize(model_tensors=dict(model.named_parameters()))

    source = receiver.poll_for_source(model_name="Qwen/Qwen2.5-1.5B")
    if source is not None:
        for name, tensor in receiver.receive_weights(source):
            ...  # load into model
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Iterator

import torch

from .client import MxClient
from .nixl_transfer import NixlTransferManager, is_nixl_available
from .types import TensorDescriptor
from . import p2p_pb2

logger = logging.getLogger("modelexpress.refit_receiver")


@dataclass
class SourceRef:
    """Lightweight handle to a discovered weight source on the MX Server."""
    mx_source_id: str
    worker_id: str
    model_name: str
    worker_rank: int
    training_step: int


class MxRefitReceiver:
    """Receives updated weights from a training process via ModelExpress RDMA.

    One instance per GPU rank on the inference side. Discovers training
    sources via the MX Server, pulls weight tensors over NIXL RDMA,
    and yields them for ``model.load_weights()``.

    Args:
        agent_name: Unique NIXL agent name (e.g. ``"inference-rank-0"``).
        device_id: CUDA device index for this inference rank.
        mx_server_url: gRPC address of the ModelExpress server.
        listen_port: Optional NIXL listen port for P2P metadata exchange.
    """

    def __init__(
        self,
        agent_name: str,
        device_id: int,
        mx_server_url: str = "localhost:8001",
        listen_port: int | None = None,
    ):
        self._agent_name = agent_name
        self._device_id = device_id
        self._mx_server_url = mx_server_url
        self._listen_port = listen_port

        self._nixl: NixlTransferManager | None = None
        self._client: MxClient | None = None
        self._initialized = False
        self._current_step = -1

    @property
    def current_step(self) -> int:
        """The most recently received training step."""
        return self._current_step

    def initialize(self, model_tensors: dict[str, torch.Tensor] | None = None) -> None:
        """Initialize NIXL agent, MX client, and optionally register receive buffers.

        Args:
            model_tensors: If provided, registers these tensors with NIXL as
                receive buffers. For tensor-name-matched transfers, the source's
                tensors are written directly into these buffers. If *None*,
                the caller must register tensors separately.
        """
        if not is_nixl_available():
            raise RuntimeError(
                "NIXL is not available. Install nixl or build from source."
            )

        self._nixl = NixlTransferManager(
            agent_name=self._agent_name,
            device_id=self._device_id,
            listen_port=self._listen_port,
        )
        self._nixl.initialize()

        if model_tensors is not None:
            self._nixl.register_tensors(model_tensors)
            logger.info(
                f"Registered {len(model_tensors)} receive buffers with NIXL"
            )

        self._client = MxClient(server_url=self._mx_server_url)
        self._initialized = True
        logger.info(
            f"MxRefitReceiver initialized: agent={self._agent_name}, "
            f"device={self._device_id}"
        )

    def poll_for_source(
        self,
        model_name: str,
        min_step: int | None = None,
        status_filter: int = p2p_pb2.SOURCE_STATUS_READY,
        timeout_seconds: float = 0,
    ) -> SourceRef | None:
        """Check the MX Server for a training source with updated weights.

        Args:
            model_name: Model name to filter on (must match publisher's identity).
            min_step: If set, only return sources with ``training_step >= min_step``.
                Defaults to ``current_step + 1`` to only find newer versions.
            timeout_seconds: If > 0, poll repeatedly until a source is found
                or timeout is reached. If 0, check once and return immediately.

        Returns:
            A :class:`SourceRef` if a matching source was found, else *None*.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before poll_for_source()")

        if min_step is None:
            min_step = self._current_step + 1

        identity = p2p_pb2.SourceIdentity(
            model_name=model_name,
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        )

        deadline = time.perf_counter() + timeout_seconds

        while True:
            try:
                response = self._client.list_sources(
                    identity=identity,
                    status_filter=status_filter,
                )
            except Exception as e:
                logger.warning(f"list_sources failed: {e}")
                if time.perf_counter() >= deadline:
                    return None
                time.sleep(0.5)
                continue

            for instance in response.instances:
                step_str = ""
                try:
                    meta_resp = self._client.get_metadata(
                        mx_source_id=instance.mx_source_id,
                        worker_id=instance.worker_id,
                    )
                    if meta_resp.found and meta_resp.worker:
                        worker = meta_resp.worker
                        if hasattr(worker, "tensors") and len(worker.tensors) > 0:
                            step_str = ""
                            for t in worker.tensors:
                                if t.name == "__training_step__":
                                    step_str = t.dtype
                                    break
                except Exception:
                    pass

                source_step = int(step_str) if step_str.isdigit() else 0

                if source_step >= min_step:
                    return SourceRef(
                        mx_source_id=instance.mx_source_id,
                        worker_id=instance.worker_id,
                        model_name=instance.model_name,
                        worker_rank=instance.worker_rank,
                        training_step=source_step,
                    )

            if time.perf_counter() >= deadline:
                return None
            time.sleep(0.5)

    def receive_weights(
        self,
        source: SourceRef,
        timeout_seconds: float = 300.0,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Receive weights from a discovered source via NIXL RDMA.

        Fetches the source's NIXL metadata and tensor descriptors from the
        MX Server, establishes an RDMA connection, and transfers weight
        tensors into locally registered buffers.

        Args:
            source: A :class:`SourceRef` obtained from :meth:`poll_for_source`.
            timeout_seconds: Maximum time to wait for the RDMA transfer.

        Yields:
            ``(name, tensor)`` pairs suitable for ``model.load_weights()``.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before receive_weights()")

        meta_resp = self._client.get_metadata(
            mx_source_id=source.mx_source_id,
            worker_id=source.worker_id,
        )
        if not meta_resp.found:
            raise RuntimeError(
                f"Source {source.mx_source_id}/{source.worker_id} not found on MX Server"
            )

        worker = meta_resp.worker
        source_tensors = [
            TensorDescriptor(
                name=t.name,
                addr=t.addr,
                size=t.size,
                device_id=t.device_id,
                dtype=t.dtype,
            )
            for t in worker.tensors
        ]

        transferred, skipped, elapsed = self._nixl.receive_from_source(
            source_metadata=worker.nixl_metadata,
            source_tensors=source_tensors,
            timeout_seconds=timeout_seconds,
        )

        logger.info(
            f"RDMA transfer complete: {transferred} bytes, "
            f"{len(source_tensors)} tensors, {elapsed:.2f}s "
            f"(step={source.training_step})"
        )

        self._current_step = source.training_step

        for td in source_tensors:
            if td.name in self._nixl._tensors:
                yield td.name, self._nixl._tensors[td.name]

    def receive_weights_from_metadata(
        self,
        nixl_metadata: bytes,
        source_tensors: list[TensorDescriptor],
        training_step: int,
        timeout_seconds: float = 300.0,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Receive weights when metadata is already available (bypasses MX Server query).

        Useful when the orchestrator passes metadata directly instead of
        having the worker poll the MX Server.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() first")

        transferred, skipped, elapsed = self._nixl.receive_from_source(
            source_metadata=nixl_metadata,
            source_tensors=source_tensors,
            timeout_seconds=timeout_seconds,
        )

        logger.info(
            f"RDMA transfer (direct metadata): {transferred} bytes, "
            f"{len(source_tensors)} tensors, {elapsed:.2f}s"
        )

        self._current_step = training_step

        for td in source_tensors:
            if td.name in self._nixl._tensors:
                yield td.name, self._nixl._tensors[td.name]

    def shutdown(self) -> None:
        """Release NIXL agent and close gRPC channel."""
        if self._nixl is not None:
            self._nixl.shutdown()
            self._nixl = None
        if self._client is not None:
            self._client.close()
            self._client = None
        self._initialized = False
        logger.info(f"MxRefitReceiver shut down: {self._agent_name}")
