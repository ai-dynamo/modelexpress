# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM generator integration."""

from __future__ import annotations

from ...adapter import GeneratorEngineContext
from ...receiver import ObjectStorageGeneratorConfig
from .adapter import VllmGeneratorAdapter
from .context import VllmGeneratorContext


def _create_vllm_adapter(
    engine_context: GeneratorEngineContext,
    worker_id: str,
    object_storage: ObjectStorageGeneratorConfig | None = None,
) -> VllmGeneratorAdapter:
    if not isinstance(engine_context, VllmGeneratorContext):
        raise TypeError("VLLM requires a VllmGeneratorContext")

    from torch.nn import Module
    from vllm.config import ModelConfig, VllmConfig

    model = engine_context.model
    vllm_config = engine_context.vllm_config
    model_config = vllm_config.model_config
    if not isinstance(model, Module):
        raise TypeError("vLLM engine context model must be a torch Module")
    if not isinstance(vllm_config, VllmConfig):
        raise TypeError("vLLM engine context vllm_config must be a VllmConfig")
    if not isinstance(model_config, ModelConfig):
        raise TypeError("vLLM engine context model_config must be a ModelConfig")
    return VllmGeneratorAdapter(
        model=model,
        vllm_config=vllm_config,
        model_config=model_config,
        worker_id=worker_id,
        object_storage=object_storage,
    )


__all__ = ["VllmGeneratorAdapter", "VllmGeneratorContext"]
