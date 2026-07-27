# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""InstantTensor loading strategy: fast local safetensors via the instanttensor library.

Loads the model's own safetensors directly onto CUDA using distributed loading,
pipelined prefetching, and direct I/O (with GPUDirect Storage when available).
Weight resolution and tensor iteration are delegated to the engine adapter,
which reuses vLLM's built-in ``--load-format instanttensor`` path.
"""

from __future__ import annotations

import importlib.util
import logging

from .. import envs
from ..adapter import EngineAdapter, StrategyFailed
from .base import LoadContext, LoadStrategy, _as_load_result, register_tensors
from .context import LoadResult

logger = logging.getLogger("modelexpress.strategy_instant_tensor")


class InstantTensorStrategy(LoadStrategy):
    """Load weights from local safetensors via the instanttensor library.

    Runs right after the RDMA (P2P) strategy: when no peer source is serving,
    InstantTensor is the fastest local-disk path before falling back to
    ModelStreamer, GDS, or the default loader. Unlike ModelStreamer it needs no
    streaming URI; the engine resolves the model's own weight files.

    Enabled by default and gated by ``MX_INSTANT_TENSOR``. Also requires the
    ``instanttensor`` package, a CUDA-like device, and an engine adapter that
    implements the InstantTensor iterator.
    """

    name = "instant_tensor"
    requires = (
        EngineAdapter.apply_weight_iter,
        EngineAdapter.build_instanttensor_weight_iter,
    )

    def is_available(self, ctx: LoadContext) -> bool:
        if not super().is_available(ctx):
            return False
        if not envs.MX_INSTANT_TENSOR:
            logger.info(
                f"[Worker {ctx.global_rank}] MX_INSTANT_TENSOR disabled, skipping instant tensor"
            )
            return False
        if importlib.util.find_spec("instanttensor") is None:
            logger.info(
                f"[Worker {ctx.global_rank}] instanttensor not installed, skipping instant tensor"
            )
            return False
        if ctx.adapter is None or not ctx.adapter.is_cuda_alike():
            logger.info(
                f"[Worker {ctx.global_rank}] InstantTensor requires a CUDA device, skipping"
            )
            return False
        return True

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        result = _as_load_result(result)
        logger.info(f"[Worker {ctx.global_rank}] Attempting InstantTensor loading")
        try:
            weights_iter = ctx.adapter.build_instanttensor_weight_iter(model=result.model)
        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] InstantTensor iterator setup failed, "
                f"falling through: {e}"
            )
            raise StrategyFailed(str(e), mutated=False) from e

        try:
            result = ctx.adapter.apply_weight_iter(result, weights_iter)
            logger.info(f"[Worker {ctx.global_rank}] InstantTensor weight loading complete")
            result = ctx.adapter.after_weight_iter_load(result)
        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] InstantTensor loading failed, falling through: {e}"
            )
            raise StrategyFailed(str(e), mutated=True) from e

        register_tensors(result, ctx)
        return result
