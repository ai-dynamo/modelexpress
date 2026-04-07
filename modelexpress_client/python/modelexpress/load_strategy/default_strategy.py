# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default loading strategy: vLLM DefaultModelLoader (CPU-staged, HF Hub download)."""

from __future__ import annotations

import copy
import logging

import torch.nn as nn

from .base import LoadContext, LoadStrategy, register_tensors, publish_metadata
from ..tensor_utils import capture_tensor_attrs

logger = logging.getLogger("modelexpress.strategy_default")


class DefaultStrategy(LoadStrategy):
    """Load weights via vLLM DefaultModelLoader (CPU-staged, HF Hub download)."""

    name = "default"

    def is_available(self, ctx: LoadContext) -> bool:
        return True

    def load(self, model: nn.Module, ctx: LoadContext) -> bool:
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        disk_config = copy.copy(ctx.load_config)
        try:
            disk_config.load_format = "auto"
        except AttributeError:
            object.__setattr__(disk_config, "load_format", "auto")

        logger.info(f"[Worker {ctx.global_rank}] Loading weights from disk...")
        default_loader = DefaultModelLoader(disk_config)
        default_loader.load_weights(model, ctx.model_config)
        logger.info(f"[Worker {ctx.global_rank}] Weights loaded from disk")

        with capture_tensor_attrs():
            process_weights_after_loading(model, ctx.model_config, ctx.target_device)

        register_tensors(model, ctx)
        publish_metadata(ctx)
        return True
