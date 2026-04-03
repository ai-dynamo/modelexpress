# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDS loading strategy: GPUDirect Storage for direct file-to-GPU loading."""

from __future__ import annotations

import logging

import torch.nn as nn

from .base import LoadContext, LoadStrategy, register_tensors, publish_metadata
from ..tensor_utils import capture_tensor_attrs

logger = logging.getLogger("modelexpress.strategy_gds")


class GdsStrategy(LoadStrategy):
    """Load weights via GPUDirect Storage (direct file-to-GPU)."""

    name = "gds"

    def is_available(self, ctx: LoadContext) -> bool:
        from ..gds_transfer import is_gds_available
        available = is_gds_available()
        if not available:
            logger.info(f"[Worker {ctx.global_rank}] GDS not available, skipping")
        return available

    def load(self, model: nn.Module, ctx: LoadContext) -> bool:
        from ..gds_loader import MxGdsLoader
        from vllm.model_executor.model_loader.utils import process_weights_after_loading

        logger.info(f"[Worker {ctx.global_rank}] Attempting GDS loading...")
        gds_loader = MxGdsLoader()
        try:
            use_tqdm = getattr(ctx.load_config, "use_tqdm_on_load", True)
            revision = getattr(ctx.model_config, "revision", None)
            weights_iter = gds_loader.load_iter(
                ctx.model_config.model, use_tqdm=use_tqdm, revision=revision
            )
            model.load_weights(weights_iter)
            logger.info(f"[Worker {ctx.global_rank}] GDS weight loading complete")

            with capture_tensor_attrs():
                process_weights_after_loading(model, ctx.model_config, ctx.target_device)

            register_tensors(model, ctx)
            publish_metadata(ctx)
            return True
        except Exception as e:
            logger.warning(
                f"[Worker {ctx.global_rank}] GDS loading failed, falling through: {e}"
            )
            return False
        finally:
            gds_loader.shutdown()
