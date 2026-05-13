# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes and shared helpers for loading strategies."""

from __future__ import annotations

import logging
import os
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import torch
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
    server_addr = os.environ.get("MODEL_EXPRESS_URL") or os.environ.get("MX_SERVER_ADDRESS")
    if server_addr:
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

        # Optional VMM compaction: move all tensors into a single contiguous
        # CUDA VA range so NIXL only needs one ibv_reg_mr call. Runs AFTER
        # discover_tensors (post-load, post-process_weights_after_loading) so
        # we operate on the final tensor set. Disabled by default; enable
        # with MX_VMM_COMPACT=1.
        vmm_range: tuple[int, int] | None = None
        if os.environ.get("MX_VMM_COMPACT", "0") == "1":
            from ..vmm_compact import compact_tensors

            va_base, va_size, ctx.tensors, ctx.vmm_arena = compact_tensors(
                result.model, ctx.tensors, ctx.device_id
            )
            if ctx.vmm_arena is not None:
                vmm_range = (va_base, va_size)

        if ctx.nixl_manager is None:
            base_port = int(os.environ.get("MX_METADATA_PORT", "5555"))
            listen_port = base_port + ctx.device_id
            ctx.nixl_manager = _init_nixl_manager(
                ctx.global_rank, ctx.device_id, "auto", listen_port
            )

        if not ctx.nixl_manager.tensor_descriptors:
            logger.debug(f"[Worker {ctx.global_rank}] Registering tensors with NIXL...")
            ctx.nixl_manager.register_tensors(ctx.tensors, vmm_range=vmm_range)
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
    # Decentralized backends (k8s-service) have no central server
    # address; their metadata path is entirely peer-to-peer.
    # Only bail on missing MODEL_EXPRESS_URL / MX_SERVER_ADDRESS when the
    # client actually needs a central coordinator. Strict `is True`
    # check so MagicMock's auto-attribute doesn't masquerade as the flag.
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


def publish_source_if_supported(result: LoadResult, ctx: LoadContext) -> None:
    """Best-effort source publication after a successful load."""
    if result.model_for_publish is None:
        return
    publish_metadata(ctx)


def unpublish_metadata(ctx: LoadContext) -> None:
    """Stop heartbeat, stop worker gRPC server, and mark STALE on MX server.

    Call before memory becomes invalid (e.g., GMS unmap during sleep).
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
