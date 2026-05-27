# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelStreamer loading strategy: stream safetensors via runai-model-streamer.

Supports object storage (S3, GCS, Azure Blob) and local filesystem paths.
File resolution and tensor iteration are delegated to the engine adapter so
native loader integrations stay engine-specific.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Iterator

import torch

from ..adapter import EngineAdapter, StrategyFailed
from .base import LoadContext, LoadStrategy, _as_load_result, register_tensors
from .context import LoadResult

logger = logging.getLogger("modelexpress.strategy_model_streamer")


class ModelStreamerStrategy(LoadStrategy):
    """Load weights by streaming safetensors via runai-model-streamer.

    Activated by LoadContext.model_streamer_uri. Context builders preserve the
    legacy MX_MODEL_URI gate behavior and freeze the resolved model path before
    the strategy chain runs.
    """

    name = "model_streamer"
    requires = (
        EngineAdapter.apply_weight_iter,
        EngineAdapter.build_model_streamer_weight_iter,
    )

    def is_available(self, ctx: LoadContext) -> bool:
        if not super().is_available(ctx):
            return False
        if importlib.util.find_spec("runai_model_streamer") is None:
            logger.info(
                f"[Worker {ctx.global_rank}] runai_model_streamer not installed, skipping"
            )
            return False

        if not ctx.model_streamer_uri:
            logger.info(
                f"[Worker {ctx.global_rank}] ModelStreamer URI not configured, skipping"
            )
            return False
        return True

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        result = _as_load_result(result)
        model_uri = ctx.model_streamer_uri
        if not model_uri:
            raise StrategyFailed("ModelStreamer URI not configured", mutated=False)

        logger.info(f"[Worker {ctx.global_rank}] Attempting model streamer loading from {model_uri}")
        try:
            weights_iter = self._stream_weights(model_uri, ctx, result.model)
        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] Model streamer loading failed, falling through: {e}"
            )
            raise StrategyFailed(str(e), mutated=False) from e

        try:
            result = ctx.adapter.apply_weight_iter(result, weights_iter)
            logger.info(f"[Worker {ctx.global_rank}] Model streamer weight loading complete")
            result = ctx.adapter.after_weight_iter_load(result)
        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] Model streamer loading failed, falling through: {e}"
            )
            raise StrategyFailed(str(e), mutated=True) from e

        register_tensors(result, ctx)
        return result

    def _stream_weights(
        self,
        model_uri: str,
        ctx: LoadContext,
        model: torch.nn.Module | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        logger.info(f"[Worker {ctx.global_rank}] Streaming weights from {model_uri}")
        yield from ctx.adapter.build_model_streamer_weight_iter(
            model_uri,
            model=model,
        )
