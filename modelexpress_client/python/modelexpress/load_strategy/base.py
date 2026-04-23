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

from ..client import MxClientBase
from ..client_factory import create_metadata_client
from ..nixl_transfer import is_nixl_available
from ..tensor_utils import adopt_hidden_tensors, collect_module_tensors, log_tensor_summary
from ..metadata import build_source_identity, publish_metadata_and_ready
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
    mx_client: MxClientBase
    worker_id: str
    nixl_manager: NixlTransferManager | None = None
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)


def build_load_context(
    vllm_config: VllmConfig,
    model_config: ModelConfig,
) -> LoadContext:
    """Build a LoadContext from vLLM config objects.

    Resolves device, ranks, builds source identity, and creates MX client
    and worker ID. Used by both MxModelLoader and GMS loader to avoid
    duplicating context construction logic.

    Args:
        vllm_config: vLLM engine configuration.
        model_config: Model configuration (name, dtype, quantization).
    """
    from ..rank_utils import get_global_rank, get_worker_rank

    load_config = vllm_config.load_config
    load_device = (
        vllm_config.device_config.device
        if load_config.device is None
        else load_config.device
    )
    target_device = torch.device(load_device)
    device_id = get_worker_rank(target_device)
    global_rank = get_global_rank(target_device)

    return LoadContext(
        vllm_config=vllm_config,
        model_config=model_config,
        load_config=load_config,
        target_device=target_device,
        global_rank=global_rank,
        device_id=device_id,
        identity=build_source_identity(vllm_config, model_config),
        mx_client=create_metadata_client(worker_rank=global_rank),
        worker_id=uuid.uuid4().hex[:8],
    )


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

    def rollback(self, ctx: LoadContext) -> bool:
        """Clean up after a failed load attempt.

        Called by the chain when load() returns False or raises. Strategies
        that mutate the model before their failure point (e.g. RDMA runs
        process_weights_after_loading before the transfer) should override
        this to clean up their state and return True so the chain can
        re-initialize the model for the next strategy.

        Returns True if the model was mutated and needs re-initialization.
        """
        return False


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
        # Order matters: adopt first so hidden tensors appear in named_buffers()
        # before collect_module_tensors iterates them.
        adopt_hidden_tensors(model)
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
    # Decentralized backends (k8s-service, future DHT) have no central
    # server address; their metadata path is entirely peer-to-peer.
    # Only bail on missing MODEL_EXPRESS_URL / MX_SERVER_ADDRESS when the
    # client actually needs a central coordinator. Strict `is True`
    # check so MagicMock's auto-attribute doesn't masquerade as the flag.
    server_addr = os.environ.get("MODEL_EXPRESS_URL") or os.environ.get("MX_SERVER_ADDRESS")
    requires_p2p = getattr(ctx.mx_client, "REQUIRES_P2P_METADATA", False) is True
    if not server_addr and not requires_p2p:
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


def unpublish_metadata(ctx: LoadContext) -> None:
    """Stop heartbeat, stop worker gRPC server, and mark STALE on MX server.

    Call before memory becomes invalid (e.g., GMS unmap during sleep).
    The NIXL agent stays alive — only the P2P serving state is torn down.
    Call publish_metadata() again after memory is valid to re-enter the
    P2P network.
    """
    from ..metadata import _heartbeat_threads, _worker_servers

    hb = _heartbeat_threads.pop(ctx.global_rank, None)
    if hb is not None:
        try:
            hb.stop()  # also marks STALE on MX server
            logger.info(f"[Worker {ctx.global_rank}] Heartbeat stopped")
        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] Failed to stop heartbeat cleanly: {e}"
            )

    ws = _worker_servers.pop(ctx.device_id, None)
    if ws is not None:
        try:
            ws.stop()
            logger.info(f"[Worker {ctx.global_rank}] Worker gRPC server stopped")
        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] Failed to stop worker gRPC server cleanly: {e}"
            )
