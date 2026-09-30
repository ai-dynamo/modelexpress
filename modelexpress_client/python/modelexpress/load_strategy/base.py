# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes and shared helpers for loading strategies."""

from __future__ import annotations

import logging
import traceback
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import torch.nn as nn

from .. import envs
from ..nixl_transfer import is_nixl_available
from ..tensor_utils import log_tensor_summary
from ..metadata.publish import extend_published_tensors, publish_metadata_and_ready
from .context import LoadContext, LoadResult

if TYPE_CHECKING:
    from ..accelerators import AcceleratorBackend
    from ..nixl_transfer import NixlTransferManager

logger = logging.getLogger("modelexpress.load_strategy")


def clear_exception_tracebacks(exc: BaseException) -> None:
    """Drop completed failure frames before releasing a mutated model.

    Transfer failures commonly retain target tensors through traceback frame
    locals (for example ``local_tensor`` in the NIXL matching loop). Clearing
    only ``LoadResult`` and ``LoadContext`` therefore does not guarantee that
    CUDA allocations become unreachable before retry initialization.
    """
    pending: list[BaseException] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
        if current.__traceback__ is not None:
            traceback.clear_frames(current.__traceback__)
            current.__traceback__ = None


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
    global_rank: int,
    device_id: int,
    role: str,
    listen_port: int = 0,
    accelerator_backend: AcceleratorBackend | None = None,
) -> NixlTransferManager:
    """Create and initialize a NIXL transfer manager."""
    from ..nixl_transfer import NixlTransferManager

    agent_name = f"mx-{role}-worker{global_rank}-{uuid.uuid4().hex[:8]}"
    logger.debug(f"[Worker {global_rank}] Initializing NIXL manager with agent_name={agent_name}")
    manager = NixlTransferManager(
        agent_name=agent_name,
        device_id=device_id,
        listen_port=listen_port,
        accelerator_backend=accelerator_backend,
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
    server_addr = (
        ctx.mx_server_url or envs.MODEL_EXPRESS_URL or envs.MX_SERVER_ADDRESS
    )
    if server_addr:
        return True
    return getattr(ctx.mx_client, "REQUIRES_P2P_METADATA", False) is True


def register_tensors(
    result_or_model: LoadResult | nn.Module,
    ctx: LoadContext,
    *,
    reuse_discovered: bool = False,
) -> None:
    """Collect model tensors and register them with NIXL.

    Failures are logged but do not raise — the worker continues without
    P2P serving capability.

    Args:
        reuse_discovered: If True, skip tensor discovery
            (``ctx.adapter.discover_tensors``) and re-register the
            previously collected set held in ``ctx.tensors``. Use on
            wake / CRIU-restore paths where the weight set is unchanged
            from initial load.

            Rationale: tensor discovery only needs to run once, when the
            model is in its final post-``process_weights_after_loading``
            state. Re-running it after vLLM's warmup / ``torch.compile`` /
            CUDA graph capture is both unnecessary (the weight set is
            frozen) and risky — post-warmup artifacts such as
            ``torch._dynamo.aot_compile.AOTCompiledFunction`` attach to
            ``model`` as plain attributes and reach framework objects
            (llvmlite, ``torch.distributed`` deprecation shims,
            HuggingFace ``_LazyAutoMapping``) that the walker was never
            designed to traverse. P2P targets receive only the weight
            tensors and run their own warmup / compile locally, so
            compile artifacts are never part of what P2P must expose.

            Falls back to discovery if ``ctx.tensors`` is empty so the
            flag is safe even if the caller misuses it.
    """
    if not ctx.p2p_enabled:
        return
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
    if (
        ctx.p2p_role == "draft"
        and ctx.nixl_manager is None
        and ctx.shared_nixl_manager is None
    ):
        # Without the main load's agent there is no publication to extend,
        # and binding a fresh agent would reuse MX_METADATA_PORT + device_id.
        logger.info(
            f"[Worker {ctx.global_rank}] Speculative draft pass has no main-load "
            "NIXL agent to share, skipping NIXL registration"
        )
        return

    try:
        if reuse_discovered and ctx.tensors:
            logger.info(
                f"[Worker {ctx.global_rank}] Re-registering "
                f"{len(ctx.tensors)} previously-discovered tensors "
                f"(skipping model walk)"
            )
        else:
            result = _as_load_result(result_or_model)
            if result.model is None:
                logger.info(
                    f"[Worker {ctx.global_rank}] No model available, skipping NIXL registration"
                )
                return

            ctx.tensors = ctx.adapter.discover_tensors(result)
            if ctx.p2p_role == "draft":
                ctx.tensors = {
                    ctx.draft_tensor_namespace + name: tensor
                    for name, tensor in ctx.tensors.items()
                }
            log_tensor_summary(ctx.tensors, ctx.global_rank, "Registering tensors")

        if ctx.nixl_manager is None:
            if ctx.shared_nixl_manager is not None:
                # Draft pass: rebinding MX_METADATA_PORT + device_id is what
                # produced "Address already in use [98]". Register into the
                # agent the main load already bound.
                ctx.nixl_manager = ctx.shared_nixl_manager
            else:
                base_port = envs.MX_METADATA_PORT
                listen_port = base_port + ctx.device_id
                ctx.nixl_manager = _init_nixl_manager(
                    ctx.global_rank,
                    ctx.device_id,
                    "auto",
                    listen_port,
                    ctx.accelerator_backend,
                )

        if ctx.p2p_role == "draft":
            # Reusing the target's manager means tensor_descriptors is already
            # populated; the draft must still register its own tensors, and
            # append rather than replace the target's catalog. Per-tensor (not
            # register_arena) because the arena MR range was captured at the
            # target's registration and cannot be grown without invalidating
            # the rkey peers already hold.
            # TODO: register_arena_extend (a second MR over the adjacent
            # sub-range of the arena reserve) would restore single-MR here.
            logger.debug(
                f"[Worker {ctx.global_rank}] Appending draft tensors to the "
                "shared NIXL registration..."
            )
            ctx.nixl_manager.register_additional_tensors(ctx.tensors)
        elif not ctx.nixl_manager.tensor_descriptors:
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
        if ctx.p2p_role == "draft":
            # Never publish draft descriptors that are not registered; the
            # shared agent itself stays with the main load.
            ctx.nixl_manager = None
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
    if ctx.p2p_role == "draft":
        # Same checkpoint, same SourceIdentity: extend the main load's
        # publication instead of starting a second worker gRPC server on
        # MX_WORKER_GRPC_PORT + device_id under a colliding mx_source_id.
        try:
            extended = extend_published_tensors(
                ctx.tensors,
                device_id=ctx.device_id,
                worker_rank=ctx.worker_rank,
            )
        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] Failed to extend the main "
                f"publication with draft tensors: {e}"
            )
            return
        if not extended:
            logger.info(
                f"[Worker {ctx.global_rank}] No main publication on device "
                f"{ctx.device_id} to extend, draft tensors are not served"
            )
        else:
            ctx.draft_published = True
        return
    try:
        publish_metadata_and_ready(
            ctx.mx_client, ctx.nixl_manager, ctx.tensors,
            ctx.worker_rank, ctx.device_id, ctx.identity, ctx.worker_id,
            accelerator=ctx.accelerator_backend.name,
            ready_fn=ctx.source_ready_fn,
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

    Call before memory becomes invalid (e.g., VMM unmap during sleep).
    The NIXL agent stays alive — only the P2P serving state is torn down.
    Call publish_metadata() again after memory is valid to re-enter the
    P2P network.
    """
    unpublish_metadata_for_worker(
        worker_rank=ctx.worker_rank,
        device_id=ctx.device_id,
    )


def drain_tensor_readers(ctx: LoadContext, *, timeout: float) -> None:
    """Block new tensor reads and wait for current readers before mutation."""
    from ..metadata.publish import _heartbeat_threads, _worker_servers

    worker_server = _worker_servers.get(ctx.device_id)
    if worker_server is not None:
        worker_server.drain_tensor_reads(timeout)
        return
    if _heartbeat_threads.get(ctx.worker_rank) is not None:
        raise RuntimeError(
            "active refit cannot safely mutate centrally published tensors; "
            "enable MX_P2P_METADATA"
        )


def unpublish_metadata_for_worker(*, worker_rank: int, device_id: int) -> None:
    """Stop one worker's publication without requiring a boot-load context."""
    from ..metadata.publish import (
        _central_manifests,
        _heartbeat_threads,
        _worker_servers,
    )

    _central_manifests.pop(device_id, None)
    hb = _heartbeat_threads.pop(worker_rank, None)
    if hb is not None:
        try:
            hb.stop()  # also marks STALE on MX server
            logger.info(f"[Worker {worker_rank}] Heartbeat stopped")
        except Exception as e:
            logger.warning(
                f"[Worker {worker_rank}] Failed to stop heartbeat cleanly: {e}"
            )

    ws = _worker_servers.pop(device_id, None)
    if ws is not None:
        try:
            ws.stop()
            logger.info(f"[Worker {worker_rank}] Worker gRPC server stopped")
        except Exception as e:
            logger.warning(
                f"[Worker {worker_rank}] Failed to stop worker gRPC server cleanly: {e}"
            )
