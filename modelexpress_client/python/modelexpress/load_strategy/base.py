# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes and shared helpers for loading strategies."""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import torch.nn as nn

from ..nixl_transfer import is_nixl_available
from ..tensor_utils import log_tensor_summary
from ..metadata.publish import publish_metadata_and_ready
from .context import LoadContext, LoadResult

if TYPE_CHECKING:
    from ..nixl_transfer import NixlTransferManager

logger = logging.getLogger("modelexpress.load_strategy")


class SourceTransferError(Exception):
    """Raised when a failure is demonstrably from the remote source side.

    Only this exception triggers marking the source STALE. Target-local errors
    (OOM, process_weights_after_loading, warmup) are left as plain exceptions
    so they propagate without poisoning a healthy source.
    """


class LoadStrategy(ABC):
    """Base class for weight-loading strategies.

    Each strategy is fully self-contained for one loading path. Source
    publication is handled by the chain after a strategy succeeds.

    Contract:
      - return LoadResult only after successful loading
      - raise StrategyFailed(mutated=False) for expected fallback paths
      - raise StrategyFailed(mutated=True) after mutating the model
      - reserve unexpected errors for rare defensive fallback in the chain
    """

    name: str
    requires: ClassVar[tuple] = ()

    def is_available(self, ctx: LoadContext) -> bool:
        """Check environment: is this strategy usable right now?"""
        if not self.requires:
            return True
        if ctx.adapter is None:
            return False
        cls = type(ctx.adapter)
        return all(getattr(cls, m.__name__) is not m for m in self.requires)

    @abstractmethod
    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        """Attempt to load weights and return the updated result.

        Do not return booleans for fallback. Use StrategyFailed so the chain
        can distinguish clean misses from failures that require re-init.
        """

    def rollback(self, ctx: LoadContext) -> None:
        """Clean up strategy-owned state after a failed load attempt.

        This hook must not decide whether the model is dirty. Strategies report
        that through StrategyFailed(mutated=True).
        """
        return None


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


def _as_load_result(result_or_model: LoadResult | nn.Module) -> LoadResult:
    if isinstance(result_or_model, LoadResult):
        return result_or_model
    return LoadResult(value=result_or_model, model=result_or_model)


def _metadata_publication_configured(ctx: LoadContext) -> bool:
    """Return whether this worker has a metadata path for P2P serving."""
    if getattr(ctx, "metadata_server_url", None):
        return True
    return getattr(ctx.mx_client, "REQUIRES_P2P_METADATA", False) is True


def register_tensors(result_or_model: LoadResult | nn.Module, ctx: LoadContext) -> None:
    """Collect model tensors and register them with NIXL.

    Failures are logged but do not raise — the worker continues without
    P2P serving capability.
    """
    if not _metadata_publication_configured(ctx):
        logger.info(
            f"[Worker {ctx.global_rank}] No MX metadata path configured, "
            "skipping NIXL registration"
        )
        return
    if not is_nixl_available():
        logger.warning(f"[Worker {ctx.global_rank}] NIXL not available, skipping registration")
        return
    if ctx.adapter is None:
        raise RuntimeError("NIXL registration requires an engine adapter")

    try:
        result = _as_load_result(result_or_model)
        if result.model is None:
            logger.info(
                f"[Worker {ctx.global_rank}] No model available, skipping NIXL registration"
            )
            return

        ctx.tensors = ctx.adapter.discover_tensors(result)
        log_tensor_summary(ctx.tensors, ctx.global_rank, "Registering tensors")

        if ctx.nixl_manager is None:
            listen_port = ctx.metadata_port + ctx.device_id
            ctx.nixl_manager = _init_nixl_manager(
                ctx.global_rank, ctx.device_id, "auto", listen_port
            )

        if not ctx.nixl_manager.tensor_descriptors:
            if ctx.vmm_arena is not None:
                logger.debug(
                    f"[Worker {ctx.global_rank}] Registering arena with NIXL "
                    "(single MR via dmabuf)..."
                )
                ctx.nixl_manager.register_arena(ctx.vmm_arena, ctx.tensors)
            else:
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
    # Decentralized backends (k8s-service) have no central server address;
    # their metadata path is entirely peer-to-peer. Only bail on missing
    # metadata_server_url when the client actually needs a central coordinator.
    # Strict `is True` check so MagicMock's auto-attribute doesn't masquerade as
    # the flag.
    if not _metadata_publication_configured(ctx):
        logger.info(
            f"[Worker {ctx.global_rank}] No MX server configured, skipping metadata publish"
        )
        return
    try:
        publish_metadata_and_ready(
            ctx.mx_client, ctx.nixl_manager, ctx.tensors,
            ctx.worker_rank, ctx.device_id, ctx.identity, ctx.worker_id,
        )
    except Exception as e:
        logger.warning(
            f"[Worker {ctx.global_rank}] Failed to publish metadata, "
            f"worker will continue without P2P serving: {e}"
        )


def publish_loaded_model(result: LoadResult, ctx: LoadContext) -> None:
    """Register and publish an already-loaded model as a reusable source.

    Some engine lifecycles load weights outside the strategy chain and only
    have a publish callback after the model is ready. Keep the source context
    reachable from the model so its NIXL manager, identity, and metadata client
    remain alive while the engine owns the model object.
    """
    if not result.publishable:
        return

    register_tensors(result, ctx)
    publish_metadata(ctx)

    model = result.model
    if model is None:
        return

    _retain_source_runtime(model, ctx)


def _retain_source_runtime(model: nn.Module, ctx: LoadContext) -> None:
    """Keep source-serving runtime objects alive for the model lifetime.

    Publishing metadata advertises this process as a live source, but the actual
    source still depends on local runtime state. The context owns the NIXL
    manager, registered memory endpoints, worker identity, and metadata client.
    Some engine lifecycles, notably TRT-LLM post-load publish, create this
    context inside a short callback rather than storing it on a ModelExpress
    loader instance. Attach the current context to the model so Python does not
    collect the source runtime while the engine is still serving the model.
    """
    setattr(model, "_mx_load_context", ctx)


def publish_source_if_supported(result: LoadResult, ctx: LoadContext) -> None:
    """Best-effort source publication after a successful load."""
    if result.model_for_publish is None:
        return
    publish_loaded_model(result, ctx)


def unpublish_metadata(ctx: LoadContext) -> None:
    """Stop heartbeat, stop worker gRPC server, and mark STALE on MX server.

    Call before memory becomes invalid (e.g., VMM unmap during sleep).
    The NIXL agent stays alive — only the P2P serving state is torn down.
    Call publish_metadata() again after memory is valid to re-enter the
    P2P network.
    """
    from ..metadata.publish import _heartbeat_threads, _worker_servers

    hb = _heartbeat_threads.pop(ctx.worker_rank, None)
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
