# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL OBJ loading strategy: S3-over-RDMA reads straight into device memory."""

from __future__ import annotations

import logging
from typing import Any

import torch

from .. import envs
from ..adapter import EngineAdapter, StrategyFailed, StrategyRecoveryError
from .base import LoadContext, LoadStrategy, _as_load_result, register_tensors
from .context import LoadResult

logger = logging.getLogger("modelexpress.strategy_obj")

_REASON_LIMIT = 300


def _world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


def _adapter_gathers_state(ctx: LoadContext) -> bool:
    return (
        ctx.adapter is not None
        and type(ctx.adapter).all_gather_state is not EngineAdapter.all_gather_state
    )


class ObjStrategy(LoadStrategy):
    """Load safetensors weights from an S3-over-RDMA endpoint via NIXL OBJ.

    Opt-in with MX_OBJ_ENABLED=1 and an s3:// MX_MODEL_URI. Placed ahead of
    ModelStreamer, which claims any MX_MODEL_URI. Every rank reaches the same
    decision at each phase, so a failure on one rank sends all ranks to the
    next strategy together instead of leaving one rank in a collective alone.
    """

    name = "obj"
    requires = (EngineAdapter.apply_weight_iter,)

    def is_available(self, ctx: LoadContext) -> bool:
        if not envs.MX_OBJ_ENABLED:
            return False
        model_uri = envs.MX_MODEL_URI or ""
        if not model_uri.lower().startswith("s3://"):
            logger.info(
                f"[Worker {ctx.global_rank}] MX_OBJ_ENABLED is set but MX_MODEL_URI "
                "is not an s3:// URI, skipping NIXL OBJ"
            )
            return False
        if not super().is_available(ctx):
            return False
        if envs.MX_VMM_ARENA:
            logger.info(f"[Worker {ctx.global_rank}] NIXL OBJ does not support MX_VMM_ARENA, skipping")
            return False
        if not ctx.accelerator_backend.supports_obj():
            logger.info(
                f"[Worker {ctx.global_rank}] NIXL OBJ not supported on "
                f"{ctx.accelerator_backend.name}, skipping"
            )
            return False
        if _world_size() > 1 and not _adapter_gathers_state(ctx):
            logger.info(
                f"[Worker {ctx.global_rank}] NIXL OBJ needs rank agreement, which this "
                "engine adapter does not provide for multi-rank runs, skipping"
            )
            return False
        from ..obj_transfer import is_obj_available

        available = is_obj_available()
        if not available:
            logger.info(f"[Worker {ctx.global_rank}] NIXL OBJ not available, skipping")
        return available

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        result = _as_load_result(result)
        from ..obj_loader import MxObjLoader, ObjIntegrityError, ObjLoadConfig

        model_uri = envs.MX_MODEL_URI
        logger.info(f"[Worker {ctx.global_rank}] Attempting NIXL OBJ loading from {model_uri}...")
        loader = None
        try:
            plan = None
            error: BaseException | None = None
            try:
                loader = MxObjLoader(ctx.accelerator_backend, ctx.device_id, ObjLoadConfig.from_env())
                plan = loader.prepare(model_uri)
            except Exception as exc:
                error = exc
            _agree(ctx, "prepared", error, plan.fingerprint if plan else None, mutated=False)

            error = None
            weights = loader.iter_tensors()
            try:
                result = ctx.adapter.apply_weight_iter(result, weights)
            except Exception as exc:
                error = exc
            finally:
                weights.close()
                finished = loader.finished
                loader.release()
            if error is None and not finished:
                error = ObjIntegrityError("engine stopped consuming weights before the last tensor")
            _agree(ctx, "streamed", error, None, mutated=True)
            logger.info(f"[Worker {ctx.global_rank}] NIXL OBJ weight loading complete")

            error = None
            try:
                result = ctx.adapter.after_weight_iter_load(result)
            except Exception as exc:
                error = exc
            _agree(ctx, "finalized", error, None, mutated=True)
        finally:
            if loader is not None:
                loader.shutdown()

        loader.log_timing()
        register_tensors(result, ctx)
        return result


def _agree(
    ctx: LoadContext,
    phase: str,
    error: BaseException | None,
    fingerprint: str | None,
    *,
    mutated: bool,
) -> None:
    """All-gather this rank's phase outcome; fail every rank if any rank failed."""
    ok = error is None
    reason = "" if ok else f"{type(error).__name__}: {error}"[:_REASON_LIMIT]
    if not ok:
        logger.warning(f"[Worker {ctx.global_rank}] NIXL OBJ {phase} phase failed: {reason}")
    state = (phase, ok, reason, fingerprint)
    try:
        states: tuple[Any, ...] = (
            ctx.adapter.all_gather_state(state) if ctx.adapter is not None else (state,)
        )
    except Exception as exc:
        raise StrategyRecoveryError(
            f"[Worker {ctx.global_rank}] NIXL OBJ ranks could not agree on phase {phase!r}: {exc}"
        ) from exc

    failures = []
    fingerprints = set()
    for rank, peer in enumerate(states):
        if not isinstance(peer, tuple) or len(peer) != 4:
            raise StrategyRecoveryError(f"invalid NIXL OBJ phase state from rank {rank}: {peer!r}")
        peer_phase, peer_ok, peer_reason, peer_fingerprint = peer
        if peer_phase != phase:
            raise StrategyRecoveryError(
                f"NIXL OBJ phase disagreement: local={phase!r}, rank {rank}={peer_phase!r}"
            )
        if not peer_ok:
            failures.append(f"rank {rank}: {peer_reason}")
        elif peer_fingerprint is not None:
            fingerprints.add(peer_fingerprint)
    if len(fingerprints) > 1:
        failures.append("ranks resolved different checkpoint objects")
    if failures:
        raise StrategyFailed(
            f"NIXL OBJ {phase} phase failed ({'; '.join(failures)})",
            mutated=mutated,
        ) from error
