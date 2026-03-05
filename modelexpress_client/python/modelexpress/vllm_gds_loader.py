# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM integration for the GDS model loader.

Thin wrapper that registers MxGdsLoader as a vLLM model loader (``mx-gds``).
The actual GDS logic lives in ``gds_loader.py`` (framework-agnostic).

Usage::

    vllm serve deepseek-ai/DeepSeek-V3 --load-format mx-gds
"""

from __future__ import annotations

import logging

import torch.nn as nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader

from .gds_loader import MxGdsLoader

logger = logging.getLogger("modelexpress.vllm_gds_loader")


@register_model_loader("mx-gds")
class MxGdsVllmLoader(BaseModelLoader):
    """
    vLLM model loader that loads weights via GPUDirect Storage.

    Uses :class:`MxGdsLoader` to read safetensors files directly into GPU
    memory through NIXL's GDS backend, then feeds the tensors through vLLM's
    standard ``model.load_weights()`` pipeline (which handles sharding,
    renaming, QKV merge, etc.).

    The base class ``load_model()`` already orchestrates::

        initialize_model → load_weights → process_weights_after_loading

    We only implement ``load_weights`` and ``download_model``.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)

    def download_model(self, model_config: ModelConfig) -> None:
        """Download HuggingFace model if not already cached locally."""
        from pathlib import Path
        if not Path(model_config.model).is_dir():
            from huggingface_hub import snapshot_download
            snapshot_download(model_config.model)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights from safetensors files directly to GPU via GDS."""
        model_path = model_config.model
        logger.info("Loading weights via GDS from %s", model_path)

        gds_loader = MxGdsLoader()
        try:
            model.load_weights(gds_loader.load_iter(model_path))
        finally:
            gds_loader.shutdown()

        logger.info("GDS weight loading complete")
