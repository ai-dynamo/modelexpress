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

from .base import (
    LoadContext,
    LoadStrategy,
    SourceTransferError,
    build_load_context,
    register_tensors,
    publish_metadata,
    unpublish_metadata,
)

__all__ = [
    "LoadContext",
    "LoadStrategy",
    "LoadStrategyChain",
    "SourceTransferError",
    "build_load_context",
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
    def run(model: nn.Module, ctx: LoadContext) -> None:
        """Build the chain and execute strategies until one succeeds.

        Raises RuntimeError if no strategy succeeds.
        """
        from .rdma_strategy import RdmaStrategy
        from .model_streamer_strategy import ModelStreamerStrategy
        from .gds_strategy import GdsStrategy
        from .default_strategy import DefaultStrategy

        all_strategies: list[LoadStrategy] = [
            RdmaStrategy(),
            ModelStreamerStrategy(),
            GdsStrategy(),
            DefaultStrategy(),
        ]
        eligible = [s for s in all_strategies if s.is_available(ctx)]
        logger.info(f"Eligible loaders: {[s.name for s in eligible]}")

        for strategy in eligible:
            logger.info(f"[Worker {ctx.global_rank}] Trying strategy: {strategy.name}")
            try:
                if strategy.load(model, ctx):
                    return
            except Exception as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Strategy {strategy.name} "
                    f"raised unexpected error, trying next: {e}"
                )

        raise RuntimeError(
            f"[Worker {ctx.global_rank}] No loading strategy succeeded "
            f"for model '{ctx.identity.model_name}'"
        )
