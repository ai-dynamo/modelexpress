# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Training-side weight publisher for RL refit via ModelExpress.

Wraps NixlTransferManager + MxClient to register updated model weights
on the training GPU and publish metadata to the MX Server so that
inference workers can discover and pull them via RDMA.

Typical usage in an RL training loop::

    publisher = MxTrainingPublisher("trainer-0", device_id=0, mx_server_url="mx-server:8001")
    publisher.initialize(model_name="Qwen/Qwen2.5-1.5B")

    # After optimizer.step():
    for layer_idx, layer_sd in enumerate_layers(model):
        publisher.publish_layer(layer_sd, layer_idx, step=training_step)
    publisher.mark_ready()
"""

from __future__ import annotations

import logging
import uuid
from typing import Iterator

import torch

from .client import MxClient
from .nixl_transfer import NixlTransferManager, is_nixl_available
from .types import TensorDescriptor
from . import p2p_pb2

logger = logging.getLogger("modelexpress.training_publisher")


class MxTrainingPublisher:
    """Publishes updated model weights from a training process to ModelExpress.

    One instance per GPU rank. On the training side, after each optimizer step,
    the publisher registers weight tensors with NIXL and publishes metadata to
    the MX Server. Inference workers discover the source via ``ListSources``
    and pull weights via RDMA.

    Args:
        agent_name: Unique NIXL agent name (e.g. ``"trainer-rank-0"``).
        device_id: CUDA device index for this training rank.
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
        self._worker_id: str = str(uuid.uuid4())
        self._mx_source_id: str | None = None
        self._model_name: str = ""
        self._initialized = False

    @property
    def mx_source_id(self) -> str | None:
        return self._mx_source_id

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def initialize(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        expert_parallel_size: int = 1,
        dtype: str = "bfloat16",
    ) -> None:
        """Initialize NIXL agent and MX client.

        Must be called before any publish operations. Sets up the source
        identity that inference workers will use to filter compatible sources.
        """
        if not is_nixl_available():
            raise RuntimeError(
                "NIXL is not available. Install nixl or build from source."
            )

        self._model_name = model_name
        self._identity_kwargs = dict(
            model_name=model_name,
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
            backend_framework=p2p_pb2.BACKEND_FRAMEWORK_UNKNOWN,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            expert_parallel_size=expert_parallel_size,
            dtype=dtype,
        )

        self._nixl = NixlTransferManager(
            agent_name=self._agent_name,
            device_id=self._device_id,
            listen_port=self._listen_port,
        )
        self._nixl.initialize()

        self._client = MxClient(server_url=self._mx_server_url)
        self._initialized = True
        logger.info(
            f"MxTrainingPublisher initialized: agent={self._agent_name}, "
            f"device={self._device_id}, model={model_name}"
        )

    def _build_identity(self, step: int) -> p2p_pb2.SourceIdentity:
        """Build a SourceIdentity proto with the current training step."""
        return p2p_pb2.SourceIdentity(
            extra_parameters={
                "training_step": str(step),
                "training_framework": "prime_rl",
            },
            **self._identity_kwargs,
        )

    def _build_tensor_protos(
        self, descriptors: list[TensorDescriptor]
    ) -> list[p2p_pb2.TensorDescriptor]:
        return [
            p2p_pb2.TensorDescriptor(
                name=d.name,
                addr=d.addr,
                size=d.size,
                device_id=d.device_id,
                dtype=d.dtype,
            )
            for d in descriptors
        ]

    def publish_weights(
        self,
        named_tensors: dict[str, torch.Tensor],
        step: int,
        worker_rank: int = 0,
    ) -> str:
        """Register tensors with NIXL and publish metadata to MX Server.

        This is the all-at-once variant. For layer-by-layer streaming,
        use :meth:`publish_layer` instead.

        Args:
            named_tensors: Mapping of parameter name to GPU tensor.
            step: Current training step (used for version tracking).
            worker_rank: GPU rank of this worker within the training group.

        Returns:
            The ``mx_source_id`` (16-char hex) assigned by the server.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before publish_weights()")

        self._nixl.register_tensors(named_tensors)
        metadata = self._nixl.nixl_metadata
        descriptors = self._nixl.tensor_descriptors

        identity = self._build_identity(step)
        worker_meta = p2p_pb2.WorkerMetadata(
            worker_rank=worker_rank,
            nixl_metadata=metadata,
            tensors=self._build_tensor_protos(descriptors),
            status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
            agent_name=self._agent_name,
        )

        self._mx_source_id = self._client.publish_metadata(
            identity=identity,
            worker=worker_meta,
            worker_id=self._worker_id,
        )
        logger.info(
            f"Published {len(named_tensors)} tensors for step {step} "
            f"(mx_source_id={self._mx_source_id})"
        )
        return self._mx_source_id

    def publish_layer(
        self,
        layer_state_dict: dict[str, torch.Tensor],
        layer_idx: int,
        step: int,
        worker_rank: int = 0,
    ) -> str:
        """Publish a single layer's weights to MX Server.

        Designed for PRIME-RL's layer-by-layer streaming pattern where
        ``filter_state_dict_by_layers()`` yields one layer at a time.

        Layer tensors are registered with NIXL (overwriting previous
        registration), and metadata is published to the MX Server. The
        inference side accumulates all layers before loading.

        Args:
            layer_state_dict: Parameter name -> tensor for this layer.
            layer_idx: Layer index (-1 for non-layer weights like embeddings).
            step: Current training step.
            worker_rank: GPU rank of this worker.

        Returns:
            The ``mx_source_id`` assigned by the server.
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before publish_layer()")

        self._nixl.register_tensors(layer_state_dict)
        metadata = self._nixl.nixl_metadata
        descriptors = self._nixl.tensor_descriptors

        identity = self._build_identity(step)
        identity.extra_parameters["layer_idx"] = str(layer_idx)

        worker_meta = p2p_pb2.WorkerMetadata(
            worker_rank=worker_rank,
            nixl_metadata=metadata,
            tensors=self._build_tensor_protos(descriptors),
            status=p2p_pb2.SOURCE_STATUS_INITIALIZING,
            agent_name=self._agent_name,
        )

        self._mx_source_id = self._client.publish_metadata(
            identity=identity,
            worker=worker_meta,
            worker_id=self._worker_id,
        )
        logger.debug(
            f"Published layer {layer_idx} ({len(layer_state_dict)} tensors) "
            f"for step {step}"
        )
        return self._mx_source_id

    def mark_ready(self, worker_rank: int = 0) -> bool:
        """Signal that all layers/weights have been published and are ready.

        Inference workers filter on ``SOURCE_STATUS_READY`` when polling,
        so this must be called after all publish calls for a given step.
        """
        if self._mx_source_id is None:
            raise RuntimeError("No weights published yet; call publish_weights() first")

        return self._client.update_status(
            mx_source_id=self._mx_source_id,
            worker_id=self._worker_id,
            worker_rank=worker_rank,
            status=p2p_pb2.SOURCE_STATUS_READY,
        )

    def shutdown(self) -> None:
        """Release NIXL agent and close gRPC channel."""
        if self._nixl is not None:
            self._nixl.shutdown()
            self._nixl = None
        if self._client is not None:
            self._client.close()
            self._client = None
        self._initialized = False
        logger.info(f"MxTrainingPublisher shut down: {self._agent_name}")
