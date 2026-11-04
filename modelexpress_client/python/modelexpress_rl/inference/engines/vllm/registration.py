# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Installed vLLM plugin for ModelExpress RL."""

import logging

from modelexpress import envs, register_modelexpress
from .. import configure_engine_logging

logger = logging.getLogger(__name__)
_WEIGHT_TRANSFER_BACKEND = "modelexpress"


def register() -> None:
    """Install base loading and the optional RL engine integration."""
    configure_engine_logging("vllm")
    register_modelexpress()
    if envs.MX_LOAD_STRATEGY_CHAIN == "RL":
        from vllm.model_executor import model_loader
        from .loader import RLMxModelLoader

        for load_format in ("modelexpress", "mx"):
            model_loader._LOAD_FORMAT_TO_MODEL_LOADER[load_format] = RLMxModelLoader
    register_plugin_weight_transfer_engine()


def register_plugin_weight_transfer_engine() -> None:
    """Register the ModelExpress backend when vLLM exposes weight transfer."""
    try:
        from vllm.distributed.weight_transfer.factory import (
            WeightTransferEngineFactory,
        )
    except ImportError:
        logger.debug("vLLM does not expose the weight-transfer engine factory")
        return

    if _WEIGHT_TRANSFER_BACKEND in WeightTransferEngineFactory._registry:
        return
    WeightTransferEngineFactory.register_engine(
        _WEIGHT_TRANSFER_BACKEND,
        "modelexpress_rl.inference.engines.vllm.weight_transfer_engine",
        "ModelExpressWeightTransferEngine",
    )
