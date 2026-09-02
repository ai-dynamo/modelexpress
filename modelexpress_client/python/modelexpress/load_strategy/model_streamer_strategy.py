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

from .. import envs
from ..adapter import EngineAdapter, StrategyFailed
from .base import LoadContext, LoadStrategy, _as_load_result, register_tensors
from .context import LoadResult

logger = logging.getLogger("modelexpress.strategy_model_streamer")


def _resolve_model_uri(ctx: LoadContext) -> str:
    """Resolve the model URI used by ModelStreamer across engine configs."""
    if model_uri := envs.MX_MODEL_URI:
        return model_uri
    for attr in ("model_weights", "model", "model_path"):
        value = getattr(ctx.model_config, attr, None)
        if value:
            return value
    return ""


class ModelStreamerStrategy(LoadStrategy):
    """Load weights by streaming safetensors via runai-model-streamer.

    Activated by setting MX_MODEL_URI, which is also the preferred streaming
    URI. Engine model configuration is used only when the environment value is
    unavailable. Engine adapters provide the concrete ModelStreamer iterator.
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

        model_uri = envs.MX_MODEL_URI or ""
        if not model_uri:
            logger.info(
                f"[Worker {ctx.global_rank}] MX_MODEL_URI not set, skipping model streamer"
            )
            return False
        return True

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        result = _as_load_result(result)
        model_uri = _resolve_model_uri(ctx)
        if not model_uri:
            raise StrategyFailed("ModelStreamer model URI is not configured", mutated=False)

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
