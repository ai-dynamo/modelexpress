# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator-engine implementations for ModelExpress RL refit."""

from __future__ import annotations

from ..adapter import GeneratorEngineAdapter, GeneratorEngineContext
from ..receiver import ObjectStorageGeneratorConfig
from .sglang import SglangGeneratorContext, _create_sglang_adapter
from .vllm import VllmGeneratorContext, _create_vllm_adapter


def _create_generator_adapter(
    *,
    engine_context: GeneratorEngineContext,
    worker_id: str,
    object_storage: ObjectStorageGeneratorConfig | None,
) -> GeneratorEngineAdapter:
    """Construct the adapter selected by the concrete engine context."""
    if isinstance(engine_context, SglangGeneratorContext):
        return _create_sglang_adapter(engine_context, object_storage)
    if isinstance(engine_context, VllmGeneratorContext):
        return _create_vllm_adapter(
            engine_context,
            worker_id,
            object_storage,
        )
    raise TypeError(f"unsupported generator context {type(engine_context).__name__}")


__all__: list[str] = []
