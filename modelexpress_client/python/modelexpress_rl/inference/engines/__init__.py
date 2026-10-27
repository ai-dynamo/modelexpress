# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator-engine implementations for ModelExpress RL refit."""

from __future__ import annotations

from ..adapter import GeneratorEngineContext
from ..runtime import EngineRuntime
from .sglang import SglangGeneratorContext
from .vllm import VllmGeneratorContext


def _create_engine_runtime(engine_context: GeneratorEngineContext) -> EngineRuntime:
    """Expose engine-specific installation and target geometry."""
    if isinstance(engine_context, SglangGeneratorContext):
        from .sglang import _create_sglang_engine_runtime

        return _create_sglang_engine_runtime(engine_context)
    if isinstance(engine_context, VllmGeneratorContext):
        from .vllm import _create_vllm_engine_runtime

        return _create_vllm_engine_runtime(engine_context)
    raise TypeError(f"unsupported generator context {type(engine_context).__name__}")


__all__: list[str] = []


def configure_engine_logging(engine: str) -> None:
    """Retry attaching engine handlers when worker logging becomes available."""
    from modelexpress import _configure_engine_logging

    _configure_engine_logging(engine, ("modelexpress", "modelexpress_rl"))
