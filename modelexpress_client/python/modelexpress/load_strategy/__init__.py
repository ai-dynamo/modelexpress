# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
load_strategy: prioritized chain of model loading strategies.

Detects the environment and builds an ordered list of eligible loaders.
MxModelLoader iterates the chain until one succeeds.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import (
    LoadContext,
    LoadStrategy,
    SourceTransferError,
    register_tensors,
    publish_metadata,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "LoadContext",
    "LoadStrategy",
    "LoadStrategyChain",
    "SourceTransferError",
    "register_tensors",
    "publish_metadata",
]

logger = logging.getLogger("modelexpress.load_strategy")


class LoadStrategyChain:
    """Build a prioritized chain of eligible loading strategies."""

    @staticmethod
    def build(ctx: LoadContext) -> list[LoadStrategy]:
        """Return strategies ordered by priority, filtered to available ones."""
        from .rdma_strategy import RdmaStrategy
        from .gds_strategy import GdsStrategy
        from .default_strategy import DefaultStrategy

        all_strategies: list[LoadStrategy] = [
            RdmaStrategy(),
            GdsStrategy(),
            DefaultStrategy(),
        ]
        eligible = [s for s in all_strategies if s.is_available(ctx)]
        logger.info(f"Eligible loaders: {[s.name for s in eligible]}")
        return eligible
