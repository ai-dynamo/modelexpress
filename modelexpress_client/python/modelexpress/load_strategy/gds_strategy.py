# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDS loading strategy: GPUDirect Storage for direct file-to-GPU loading."""

from __future__ import annotations

import logging

from ..adapter import EngineAdapter, StrategyFailed
from .base import LoadContext, LoadStrategy, _as_load_result, register_tensors
from .context import LoadResult

logger = logging.getLogger("modelexpress.strategy_gds")


class GdsStrategy(LoadStrategy):
    """Load weights via GPUDirect Storage (direct file-to-GPU)."""

    name = "gds"
    requires = (EngineAdapter.apply_weight_iter,)

    def is_available(self, ctx: LoadContext) -> bool:
        if not super().is_available(ctx):
            return False
        if not ctx.accelerator_backend.supports_gds():
            logger.info(
                f"[Worker {ctx.global_rank}] GDS not supported on "
                f"{ctx.accelerator_backend.name}, skipping"
            )
            return False
        from ..gds_transfer import is_gds_available
        available = is_gds_available()
        if not available:
            logger.info(f"[Worker {ctx.global_rank}] GDS not available, skipping")
        return available

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        result = _as_load_result(result)
        from ..gds_loader import MxGdsLoader

        logger.info(f"[Worker {ctx.global_rank}] Attempting GDS loading...")
        gds_loader = MxGdsLoader(accelerator_backend=ctx.accelerator_backend)
        try:
            try:
                use_tqdm = getattr(ctx.load_config, "use_tqdm_on_load", True)
                revision = getattr(ctx.model_config, "revision", None)
                weights_iter = gds_loader.load_iter(
                    ctx.model_config.model, use_tqdm=use_tqdm, revision=revision
                )
            except Exception as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] GDS loading failed, falling through: {e}"
                )
                raise StrategyFailed(str(e), mutated=False) from e

            try:
                result = ctx.adapter.apply_weight_iter(result, weights_iter)
                logger.info(f"[Worker {ctx.global_rank}] GDS weight loading complete")
                result = ctx.adapter.after_weight_iter_load(result)
            except Exception as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] GDS loading failed, falling through: {e}"
                )
                raise StrategyFailed(str(e), mutated=True) from e
        finally:
            gds_loader.shutdown()

        register_tensors(result, ctx)
        return result
