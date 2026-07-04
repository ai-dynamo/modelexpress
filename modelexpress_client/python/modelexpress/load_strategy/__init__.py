# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
load_strategy: prioritized chain of model loading strategies.

Detects the environment and builds an ordered list of eligible loaders.
MxModelLoader iterates the chain until one succeeds.
"""

from __future__ import annotations

import logging

import torch.nn as nn

from modelexpress.tracing import tracer

from ..adapter import StrategyFailed, UnsupportedCapability
from .base import (
    LoadContext,
    LoadResult,
    LoadStrategy,
    SourceTransferError,
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
        from .trainer_pull_strategy import TrainerPullStrategy
        from .rdma_strategy import RdmaStrategy
        from .model_streamer_strategy import ModelStreamerStrategy
        from .gds_strategy import GdsStrategy
        from .default_strategy import DefaultStrategy

        all_strategies: list[LoadStrategy] = [
            TrainerPullStrategy(),
            RdmaStrategy(),
            ModelStreamerStrategy(),
            GdsStrategy(),
            DefaultStrategy(),
        ]
        eligible = [s for s in all_strategies if s.is_available(ctx)]
        logger.info(f"Eligible loaders: {[s.name for s in eligible]}")

        result = LoadResult(value=model, model=model)
        with tracer.start_as_current_span("Load model") as span:
            span.set_attribute("model_name", ctx.identity.model_name)
            span.set_attribute("global_rank", ctx.global_rank)
            span.set_attribute("eligible_strategies", [s.name for s in eligible])

            for strategy in eligible:
                logger.info(f"[Worker {ctx.global_rank}] Trying strategy: {strategy.name}")
                try:
                    result = strategy.load(result, ctx)
                    publish_source_if_supported(result, ctx)
                    span.set_attribute("weight_loading_strategy", strategy.name)
                    return result.value
                except StrategyFailed as e:
                    logger.warning(
                        f"[Worker {ctx.global_rank}] Strategy {strategy.name} failed, "
                        f"trying next: {e}"
                    )
                    strategy.rollback(ctx)
                    if e.mutated:
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
