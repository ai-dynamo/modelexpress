# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes and shared helpers for loading strategies."""

from __future__ import annotations

import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from ..client import MxClient
from ..nixl_transfer import is_nixl_available
from ..tensor_utils import collect_module_tensors, log_tensor_summary
from ..metadata import publish_metadata_and_ready
from .. import p2p_pb2

if TYPE_CHECKING:
    from ..nixl_transfer import NixlTransferManager
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.load import LoadConfig

logger = logging.getLogger("modelexpress.load_strategy")


class SourceTransferError(Exception):
    """Raised when a failure is demonstrably from the remote source side.

    Only this exception triggers marking the source STALE. Target-local errors
    (OOM, process_weights_after_loading, warmup) are left as plain exceptions
    so they propagate without poisoning a healthy source.
    """


@dataclass
class LoadContext:
    """Shared state passed to all loading strategies.

    Rank semantics:
        global_rank: unique rank across all TP/PP groups (torch.distributed.get_rank()).
            Used for WorkerMetadata.worker_rank and source/target matching in RDMA.
        device_id: local CUDA device ordinal / TP rank (get_tensor_model_parallel_rank()).
            Used for CUDA device selection, port offsets (metadata_port + device_id).
    """

    vllm_config: VllmConfig
    model_config: ModelConfig
    load_config: LoadConfig
    target_device: torch.device
    global_rank: int
    device_id: int
    identity: p2p_pb2.SourceIdentity
    mx_client: MxClient
    worker_id: str
    nixl_manager: NixlTransferManager | None = None
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)


class LoadStrategy(ABC):
    """Base class for weight-loading strategies.

    Each strategy is fully self-contained: load() handles weight loading,
    post-processing, NIXL registration, and metadata publishing.
    Publish failures should not fail the worker.
    """

    name: str

    @abstractmethod
    def is_available(self, ctx: LoadContext) -> bool:
        """Check environment: is this strategy usable right now?"""

    @abstractmethod
    def load(self, model: nn.Module, ctx: LoadContext) -> bool:
        """Attempt to load weights. Return True on success, False to try next."""


# ---------------------------------------------------------------------------
# Shared helpers (used by strategy implementations)
# ---------------------------------------------------------------------------


def _init_nixl_manager(
    global_rank: int, device_id: int, role: str, listen_port: int = 0,
) -> NixlTransferManager:
    """Create and initialize a NIXL transfer manager."""
    from ..nixl_transfer import NixlTransferManager

    agent_name = f"mx-{role}-worker{global_rank}-{uuid.uuid4().hex[:8]}"
    logger.debug(f"[Worker {global_rank}] Initializing NIXL manager with agent_name={agent_name}")
    manager = NixlTransferManager(
        agent_name=agent_name,
        device_id=device_id,
        listen_port=listen_port,
    )
    manager.initialize()
    logger.debug(f"[Worker {global_rank}] NIXL manager initialized")
    return manager


def register_tensors(model: nn.Module, ctx: LoadContext) -> None:
    """Collect model tensors and register them with NIXL.

    Failures are logged but do not raise — the worker continues without
    P2P serving capability.
    """
    if not is_nixl_available():
        logger.warning(f"[Worker {ctx.global_rank}] NIXL not available, skipping registration")
        return

    try:
        ctx.tensors = collect_module_tensors(model)
        log_tensor_summary(ctx.tensors, ctx.global_rank, "Registering tensors")

        if ctx.nixl_manager is None:
            base_port = int(os.environ.get("MX_METADATA_PORT", "5555"))
            listen_port = base_port + ctx.device_id
            ctx.nixl_manager = _init_nixl_manager(
                ctx.global_rank, ctx.device_id, "auto", listen_port
            )

        if not ctx.nixl_manager.tensor_descriptors:
            logger.debug(f"[Worker {ctx.global_rank}] Registering tensors with NIXL...")
            ctx.nixl_manager.register_tensors(ctx.tensors)
            logger.debug(f"[Worker {ctx.global_rank}] Tensors registered with NIXL")
    except Exception as e:
        logger.warning(
            f"[Worker {ctx.global_rank}] NIXL registration failed, "
            f"worker will continue without P2P serving: {e}"
        )


def publish_metadata(ctx: LoadContext) -> None:
    """Publish metadata to the MX server. Failures are logged but do not raise."""
    if ctx.nixl_manager is None:
        logger.info(
            f"[Worker {ctx.global_rank}] No NIXL manager, skipping metadata publish"
        )
        return
    server_addr = os.environ.get("MODEL_EXPRESS_URL") or os.environ.get("MX_SERVER_ADDRESS")
    if not server_addr:
        logger.info(
            f"[Worker {ctx.global_rank}] No MX server configured, skipping metadata publish"
        )
        return
    try:
        publish_metadata_and_ready(
            ctx.mx_client, ctx.nixl_manager, ctx.tensors,
            ctx.global_rank, ctx.device_id, ctx.identity, ctx.worker_id,
        )
    except Exception as e:
        logger.warning(
            f"[Worker {ctx.global_rank}] Failed to publish metadata, "
            f"worker will continue without P2P serving: {e}"
        )
