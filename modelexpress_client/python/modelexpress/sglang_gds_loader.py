# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
sglang integration for the GDS model loader.

Thin wrapper that provides an sglang-compatible GDS loader class.
The actual GDS logic lives in ``gds_loader.py`` (framework-agnostic).

Usage::

    python -m sglang.launch_server --model Qwen/Qwen2.5-7B --load-format mx_gds
"""

from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

from .gds_loader import MxGdsLoader as _MxGdsLoader

logger = logging.getLogger("modelexpress.sglang_gds_loader")


def create_sglang_gds_loader_class():
    """Return an sglang ``BaseModelLoader`` subclass for GDS loading.

    Uses late imports so modelexpress can be installed without sglang.
    """
    from sglang.srt.model_loader.loader import BaseModelLoader, _initialize_model
    from sglang.srt.model_loader.utils import post_load_weights, set_default_torch_dtype

    class MxGdsSglangLoader(BaseModelLoader):
        """sglang model loader that loads weights via GPUDirect Storage.

        Uses :class:`~modelexpress.gds_loader.MxGdsLoader` to read safetensors
        files directly into GPU memory through NIXL's GDS backend, then feeds
        the tensors through sglang's standard ``model.load_weights()`` pipeline.
        """

        def download_model(self, model_config) -> None:
            """Download HuggingFace model if not already cached locally."""
            if not os.path.isdir(model_config.model_path):
                from sglang.srt.model_loader.weight_utils import (
                    download_weights_from_hf,
                )

                download_weights_from_hf(
                    model_config.model_path,
                    self.load_config.download_dir,
                    allow_patterns=["*.safetensors"],
                    revision=model_config.revision,
                    ignore_patterns=self.load_config.ignore_patterns,
                )

        def load_model(self, *, model_config, device_config) -> nn.Module:
            """Load model using MxGdsLoader for direct storage-to-GPU transfer."""
            target_device = torch.device(device_config.device)

            with set_default_torch_dtype(model_config.dtype):
                with target_device:
                    model = _initialize_model(model_config, self.load_config)

            # Ensure MxGdsLoader sees the correct CUDA device.
            # device_config.device may be "cuda" without index; fall back
            # to current device which sglang has already configured.
            if target_device.index is not None:
                torch.cuda.set_device(target_device)

            model_path = model_config.model_path
            logger.info("Loading weights via GDS from %s", model_path)

            gds_loader = _MxGdsLoader()
            try:
                for name, loaded_weight in gds_loader.load_iter(model_path):
                    model.load_weights([(name, loaded_weight)])
            finally:
                gds_loader.shutdown()

            # Post-processing (matches sglang DefaultModelLoader)
            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    if (
                        hasattr(module, "is_weights_quantized")
                        and module.is_weights_quantized()
                    ):
                        continue
                    quant_method.process_weights_after_loading(module)

            post_load_weights(model, model_config)

            logger.info("GDS weight loading complete")
            return model.eval()

    return MxGdsSglangLoader
