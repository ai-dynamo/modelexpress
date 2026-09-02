# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
load_strategy: prioritized chain of model loading strategies.

Detects the environment and builds an ordered list of eligible loaders.
MxModelLoader iterates the chain until one succeeds.
"""

from __future__ import annotations

import logging
import time

import torch.nn as nn

from modelexpress.tracing import tracer

from ..adapter import StrategyFailed, StrategyRecoveryError, UnsupportedCapability
from ..metrics import metrics as load_metrics
from .base import (
    LoadContext,
    LoadResult,
    LoadStrategy,
    SourceTransferError,
    clear_exception_tracebacks,
    publish_source_if_supported,
    register_tensors,
    publish_metadata,
    unpublish_metadata,
)

__all__ = [
    "LoadContext",
    "LoadResult",
    "LoadStrategy",
    "LoadStrategyChain",
    "SourceTransferError",
    "register_tensors",
    "publish_metadata",
    "unpublish_metadata",
]

logger = logging.getLogger("modelexpress.load_strategy")


class LoadStrategyChain:
    """Prioritized chain of model loading strategies.

    Detects the environment, builds an ordered list of eligible loaders,
    and runs them until one succeeds.
    """

    @staticmethod
    def run(model: nn.Module, ctx: LoadContext) -> nn.Module:
        """Build the chain and execute strategies until one succeeds.

        Strategies return LoadResult on success. Expected misses raise
        StrategyFailed; mutated failures trigger adapter re-initialization
        before the next strategy runs. Unexpected exceptions are rolled back
        and treated as fallback to preserve the existing chain behavior.

        Returns the (possibly re-initialized) model on success.
        Raises RuntimeError if no strategy succeeds.
        """
        from .rdma_strategy import RdmaStrategy
        from .server_cache_strategy import ServerCacheStrategy
        from .instant_tensor_strategy import InstantTensorStrategy
        from .model_streamer_strategy import ModelStreamerStrategy
        from .gds_strategy import GdsStrategy
        from .default_strategy import DefaultStrategy

        all_strategies: list[LoadStrategy] = [
            RdmaStrategy(),
            ServerCacheStrategy(),
            InstantTensorStrategy(),
            ModelStreamerStrategy(),
            GdsStrategy(),
            DefaultStrategy(),
        ]
        # One evaluation, two uses. Asking each strategy for a reason rather
        # than a bool is what lets the skip counter exist: without it, a
        # strategy that recorded no attempt was either filtered out here or was
        # eligible and never reached because an earlier one succeeded, and those
        # are opposite conclusions drawn from the same absence of data.
        reasons = [(s, s.skip_reason(ctx)) for s in all_strategies]
        eligible = [s for s, reason in reasons if reason is None]
        for strategy, reason in reasons:
            if reason is not None:
                load_metrics.record_strategy_skipped(ctx.engine, strategy.name, reason)
        logger.info(f"Eligible loaders: {[s.name for s in eligible]}")

        result = LoadResult(value=model, model=model)
        with tracer.start_as_current_span("Load model") as span:
            span.set_attribute("model_name", ctx.identity.model_name)
            span.set_attribute("global_rank", ctx.global_rank)
            span.set_attribute("eligible_strategies", [s.name for s in eligible])

            for strategy in eligible:
                logger.info(f"[Worker {ctx.global_rank}] Trying strategy: {strategy.name}")
                # L2. The interval opens here rather than around strategy.load()
                # alone, so it charges a strategy for its own rollback and for
                # the _reinit_for_retry that its mutation forced -- `finally`
                # runs before the `continue`, so that re-init lands inside the
                # interval of the strategy that caused it. The tighter wrap is
                # easier to describe but leaves the most expensive operation in
                # the chain as an unattributed gap.
                #
                # `outcome` starts at "error" so that a BaseException the
                # handlers below do not catch -- KeyboardInterrupt, SystemExit,
                # CancelledError -- is still recorded rather than dropped.
                outcome = "error"
                started = time.perf_counter()
                try:
                    result = strategy.load(result, ctx)
                    publish_source_if_supported(result, ctx)
                    span.set_attribute("weight_loading_strategy", strategy.name)
                    outcome = "success"
                    return result.value
                except StrategyRecoveryError:
                    # Recovery already failed, so no later strategy can safely
                    # use the current model. Fail closed and retain the original
                    # recovery error as the exception cause.
                    outcome = "recovery_error"
                    strategy.rollback(ctx)
                    raise
                except StrategyFailed as e:
                    outcome = "fallback_dirty" if e.mutated else "fallback"
                    logger.warning(
                        f"[Worker {ctx.global_rank}] Strategy {strategy.name} failed, "
                        f"trying next: {e}"
                    )
                    strategy.rollback(ctx)
                    if e.mutated:
                        clear_exception_tracebacks(e)
                        result = LoadStrategyChain._reinit_for_retry(result, ctx, strategy)
                    continue
                except Exception as e:
                    # Unexpected strategy errors should be rare. Keep the engine
                    # alive by falling through to the next strategy; expected
                    # fallback paths should use StrategyFailed instead.
                    logger.warning(
                        f"[Worker {ctx.global_rank}] Strategy {strategy.name} "
                        f"raised unexpected error, trying next: {e}"
                    )
                    strategy.rollback(ctx)
                finally:
                    load_metrics.observe_load_strategy_seconds(
                        ctx.engine,
                        ctx.identity.model_name,
                        strategy.name,
                        outcome,
                        time.perf_counter() - started,
                    )

        raise RuntimeError(
            f"[Worker {ctx.global_rank}] No loading strategy succeeded "
            f"for model '{ctx.identity.model_name}'"
        )

    @staticmethod
    def _reinit_for_retry(
        result: LoadResult,
        ctx: LoadContext,
        strategy: LoadStrategy,
    ) -> LoadResult:
        if ctx.adapter is None:
            raise RuntimeError(
                f"[Worker {ctx.global_rank}] Strategy '{strategy.name}' mutated "
                "the model but no adapter can reinitialize it"
            )
        try:
            return ctx.adapter.reinit_for_retry(result)
        except UnsupportedCapability as exc:
            raise RuntimeError(
                f"[Worker {ctx.global_rank}] Strategy '{strategy.name}' mutated "
                "the model but adapter does not support retry reinitialization"
            ) from exc
