# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator-engine implementations for ModelExpress RL refit."""

from __future__ import annotations

from ..adapter import GeneratorEngineContext
from ..runtime import EngineRuntime
from .sglang import SglangGeneratorContext, _create_sglang_engine_runtime
from .vllm import VllmGeneratorContext, _create_vllm_engine_runtime


def _create_engine_runtime(engine_context: GeneratorEngineContext) -> EngineRuntime:
    """Expose engine-specific installation and target geometry."""
    if isinstance(engine_context, SglangGeneratorContext):
        return _create_sglang_engine_runtime(engine_context)
    if isinstance(engine_context, VllmGeneratorContext):
        return _create_vllm_engine_runtime(engine_context)
    raise TypeError(f"unsupported generator context {type(engine_context).__name__}")


__all__: list[str] = []
