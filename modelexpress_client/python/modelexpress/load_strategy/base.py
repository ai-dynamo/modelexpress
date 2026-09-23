# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes and shared helpers for loading strategies."""

from __future__ import annotations

import logging
import traceback
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING, ClassVar

import torch.nn as nn

from .. import envs
from ..adapter import EngineAdapter, StrategyFailed
from ..nixl_transfer import is_nixl_available
from ..tensor_utils import log_tensor_summary
from ..metadata.publish import publish_metadata_and_ready
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


@contextmanager
def mutates_model():
    """Report a failure inside a phase as a mutated StrategyFailed.

    prepare(), finalize() and abort() all write into the model, so a strategy
    that falls through after one of them failed leaves the next needing a
    reinitialized model.
    """
    try:
        yield
    except StrategyFailed:
        raise
    except Exception as e:
        raise StrategyFailed(str(e), mutated=True) from e


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

    def prepare(self, result: LoadResult, ctx: LoadContext) -> None:
        """Bring the model to the state this strategy needs before loading.

        The default suits strategies that hand weights to the engine through
        its own load_weights() callbacks. A cold load needs nothing here. A
        reload enters the engine's layerwise reload, which puts parameters
        back on the meta device and materializes each layer as its weights
        arrive, so a reload never transiently holds two copies of the model.

        Works on ``result`` in place: it is the stable envelope the chain
        holds for the whole attempt.
        """
        if ctx.is_reload:
            with mutates_model():
                ctx.adapter.begin_streaming_reload(result)

    def finalize(self, result: LoadResult, ctx: LoadContext) -> None:
        """Bring the loaded weights to their final runtime layout.

        Runs only after `load` succeeded. On a cold load that is
        :meth:`post_process`. On a reload, leaving the layerwise reload
        entered by :meth:`prepare` already processed each layer as it
        materialized, so post-processing must not run again.

        Then registers the model's tensors with NIXL. Tensor discovery has to
        see the model in its final layout, which is only true from here on.
        An override that does not call this registers on its own terms, as
        RDMA does mid-transfer. Registration is best-effort and never writes
        into the model, so it sits outside :func:`mutates_model`.

        Works on ``result`` in place, like :meth:`prepare`.
        """
        with mutates_model():
            if ctx.is_reload:
                ctx.adapter.end_streaming_reload(result)
            else:
                self.post_process(result, ctx)
        if result.model_for_publish is not None:
            register_tensors(result, ctx)

    def abort(self, result: LoadResult, ctx: LoadContext) -> None:
        """Undo whatever :meth:`prepare` started, after `load` failed.

        Runs instead of :meth:`finalize`. A reload has to leave the layerwise
        reload here too, so a fallback never starts against a model still
        deferred to meta.
        """
        if ctx.is_reload:
            with mutates_model():
                ctx.adapter.end_streaming_reload(result)

    def post_process(self, result: LoadResult, ctx: LoadContext) -> None:
        """Convert the raw weights a cold load delivered into their runtime layout.

        Which adapter hook does that follows from how ``load`` handed the
        weights to the engine; see :class:`WeightsIteratorLoadStrategy` and
        :class:`NativeLoadStrategy`.
        """

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


class WeightsIteratorLoadStrategy(LoadStrategy):
    """A strategy whose load() streams weights through adapter.apply_weight_iter()."""

    requires: ClassVar[tuple] = (EngineAdapter.apply_weight_iter,)

    def post_process(self, result: LoadResult, ctx: LoadContext) -> None:
        ctx.adapter.after_weight_iter_load(result)


class NativeLoadStrategy(LoadStrategy):
    """A strategy whose load() hands weights to adapter.load_via_native()."""

    requires: ClassVar[tuple] = (EngineAdapter.load_via_native,)

    def post_process(self, result: LoadResult, ctx: LoadContext) -> None:
        ctx.adapter.after_native_load(result)


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
            log_tensor_summary(ctx.tensors, ctx.global_rank, "Registering tensors")

        if ctx.nixl_manager is None:
            base_port = envs.MX_METADATA_PORT
            listen_port = base_port + ctx.device_id
            ctx.nixl_manager = _init_nixl_manager(
                ctx.global_rank,
                ctx.device_id,
                "auto",
                listen_port,
                ctx.accelerator_backend,
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
        # Registration failed, so there is nothing for this worker to serve.
        # Drop the manager instead of leaving it half-initialized, for two
        # reasons:
        #
        # 1. The agent created above owns the metadata listener socket on
        #    MX_METADATA_PORT + device_id. Leaking it makes every subsequent
        #    attempt die on "Address already in use", turning a single
        #    recoverable failure into a permanent one for the life of the
        #    process -- the re-registration that runs after a CRIU restore
        #    never gets its port back.
        # 2. publish_metadata() treats a non-None manager as "ready to serve",
        #    so a half-initialized one gets advertised to the MX server and
        #    peers select a source that cannot transfer.
        if ctx.nixl_manager is not None:
            try:
                ctx.nixl_manager.shutdown()
            except Exception:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Failed to shut down NIXL manager "
                    "after registration failure; the metadata listener port may "
                    "stay bound",
                    exc_info=True,
                )
            finally:
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


def unpublish_metadata_for_worker(*, worker_rank: int, device_id: int) -> None:
    """Stop one worker's publication without requiring a boot-load context."""
    from ..metadata.publish import _heartbeat_threads, _worker_servers

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
