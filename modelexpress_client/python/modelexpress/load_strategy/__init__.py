# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
load_strategy: prioritized chain of model loading strategies.

Detects the environment and builds an ordered list of eligible loaders.
MxModelLoader iterates the chain until one succeeds.
"""

from __future__ import annotations

import logging

import torch
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


def _reset_vllm_compilation_state(compilation_config) -> None:
    """Reset per-model mutable state on vLLM's CompilationConfig.

    vLLM registers attention / MLA / Mamba / FusedMoE layers and accumulates
    compile stats into dicts / sets / counters on ``compilation_config``
    during ``initialize_model()``. These live on the config object, not the
    model, so they survive ``del model`` and either crash the next
    ``initialize_model()`` (``static_forward_context`` has an explicit
    duplicate-name guard) or silently corrupt subsequent state (MoE layer
    list, custom op counters, traced files, compilation time).

    Called from the chain's re-init path so the next strategy sees a clean
    config. Audited against vLLM 0.17.1; newer vLLM versions may add
    additional ``init=False`` fields on ``CompilationConfig`` that need
    similar treatment.
    """
    compilation_config.static_forward_context.clear()
    compilation_config.static_all_moe_layers.clear()
    compilation_config.enabled_custom_ops.clear()
    compilation_config.disabled_custom_ops.clear()
    compilation_config.traced_files.clear()
    compilation_config.compilation_time = 0.0


class LoadStrategyChain:
    """Prioritized chain of model loading strategies.

    Detects the environment, builds an ordered list of eligible loaders,
    and runs them until one succeeds.
    """

    @staticmethod
    def run(model: nn.Module, ctx: LoadContext) -> nn.Module:
        """Build the chain and execute strategies until one succeeds.

        Returns the (possibly re-initialized) model on success.
        Raises RuntimeError if no strategy succeeds.
        """
        from vllm.model_executor.model_loader.utils import initialize_model
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
                    return model
            except Exception as e:
                logger.warning(
                    f"[Worker {ctx.global_rank}] Strategy {strategy.name} "
                    f"raised unexpected error, trying next: {e}"
                )

            if strategy.rollback(ctx):
                del model
                torch.cuda.empty_cache()
                _reset_vllm_compilation_state(ctx.vllm_config.compilation_config)
                logger.info(
                    f"[Worker {ctx.global_rank}] Re-initializing model after "
                    f"failed strategy '{strategy.name}'"
                )
                with ctx.target_device:
                    model = initialize_model(
                        vllm_config=ctx.vllm_config,
                        model_config=ctx.model_config,
                    )

        raise RuntimeError(
            f"[Worker {ctx.global_rank}] No loading strategy succeeded "
            f"for model '{ctx.identity.model_name}'"
        )
