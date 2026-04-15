# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
ModelExpress Custom Model Loader for vLLM.

This loader hooks into vLLM's weight loading pipeline to perform RDMA transfers
of fully-processed model tensors. Registration happens AFTER
process_weights_after_loading() so that all final tensors are captured.
Tensor discovery uses named_parameters() and named_buffers(); bare tensor
attributes created during post-processing (e.g. FP8 scales, MLA projections)
are auto-promoted to non-persistent buffers via capture_tensor_attrs().

Uses LoadStrategyChain to auto-detect the best loading strategy:
    1. RDMA (P2P GPU transfer via NIXL) - if a source is already serving
    2. ModelStreamer (S3 streaming via runai-model-streamer) - stream to GPU, no disk
    3. GDS (GPUDirect Storage) - direct file-to-GPU, bypassing CPU
    4. Default (vLLM DefaultModelLoader) - standard CPU-staged loading

Usage:
    --load-format mx  (auto-detect: RDMA -> ModelStreamer -> GDS -> default)
"""

from __future__ import annotations

import logging
import time
import torch
import torch.nn as nn

from . import configure_vllm_logging
from .load_strategy import build_load_context, LoadContext, LoadStrategyChain
from .nixl_transfer import NixlTransferManager

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.utils.torch_utils import set_default_torch_dtype

logger = logging.getLogger("modelexpress.vllm_loader")


# Global storage for tensor metadata, keyed by device_id (local CUDA ordinal).
_tensor_registry: dict[int, dict[str, torch.Tensor]] = {}
_nixl_managers: dict[int, NixlTransferManager] = {}


@register_model_loader("mx")
class MxModelLoader(BaseModelLoader):
    """
    Auto-detecting model loader for ModelExpress.

    Uses LoadStrategyChain to find the best available loading strategy
    (RDMA P2P, GDS, or default disk loading), then registers tensors
    with NIXL and publishes metadata so future nodes can discover this
    one as a source.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        configure_vllm_logging()
        self._ctx: LoadContext | None = None

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        """Load model, auto-detecting the best loading strategy."""
        load_start = time.perf_counter()

        ctx = build_load_context(vllm_config, model_config)
        self._ctx = ctx

        logger.info(f"[Worker {ctx.global_rank}] MxModelLoader starting (model={ctx.identity.model_name})")

        with set_default_torch_dtype(model_config.dtype):
            with ctx.target_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )

            LoadStrategyChain.run(model, ctx)

            # Update global registries
            _tensor_registry[ctx.device_id] = ctx.tensors
            if ctx.nixl_manager is not None:
                _nixl_managers[ctx.device_id] = ctx.nixl_manager

        total_time = time.perf_counter() - load_start
        logger.info(
            f"[Worker {ctx.global_rank}] MxModelLoader.load_model() COMPLETE "
            f"in {total_time:.2f}s"
        )
        return model.eval()

    def download_model(self, model_config: ModelConfig) -> None:
        """Download the model so it can be loaded immediately."""
        import copy
        disk_config = copy.copy(self.load_config)
        try:
            disk_config.load_format = "auto"
        except AttributeError:
            object.__setattr__(disk_config, "load_format", "auto")
        DefaultModelLoader(disk_config).download_model(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into an already-initialized model (standalone API)."""
        import copy
        disk_config = copy.copy(self.load_config)
        try:
            disk_config.load_format = "auto"
        except AttributeError:
            object.__setattr__(disk_config, "load_format", "auto")
        DefaultModelLoader(disk_config).load_weights(model, model_config)

    @property
    def nixl_manager(self) -> NixlTransferManager | None:
        """Access the NIXL manager for external use."""
        if self._ctx is not None:
            return self._ctx.nixl_manager
        return None

    @property
    def tensors(self) -> dict[str, torch.Tensor]:
        """Access the registered tensor dict."""
        if self._ctx is not None:
            return self._ctx.tensors
        return {}
