# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default loading strategy: engine-native disk or hub loading."""

from __future__ import annotations

import logging

from ..adapter import EngineAdapter, StrategyFailed
from .base import LoadContext, LoadStrategy, _as_load_result, register_tensors
from .context import LoadResult

logger = logging.getLogger("modelexpress.strategy_default")


class DefaultStrategy(LoadStrategy):
    """Load weights via the engine's native fallback loader."""

    name = "default"
    requires = (EngineAdapter.load_via_native,)

    def is_available(self, ctx: LoadContext) -> bool:
        return super().is_available(ctx)

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        result = _as_load_result(result)
        logger.info(f"[Worker {ctx.global_rank}] Loading weights from disk...")
        try:
            result = ctx.adapter.load_via_native(result)
            logger.info(f"[Worker {ctx.global_rank}] Weights loaded from disk")

            result = ctx.adapter.after_native_load(result)
        except Exception as e:
            raise StrategyFailed(str(e), mutated=True) from e

        if result.model_for_publish is not None:
            register_tensors(result, ctx)
        return result
