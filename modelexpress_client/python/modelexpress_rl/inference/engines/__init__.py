# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator-engine implementations for ModelExpress RL refit."""

from __future__ import annotations

from collections.abc import Callable

from ..adapter import GeneratorEngineAdapter, GeneratorEngineContext
from .vllm import _create_vllm_adapter


_ENGINE_FACTORIES: dict[
    str, Callable[[GeneratorEngineContext, str], GeneratorEngineAdapter]
] = {
    "VLLM": _create_vllm_adapter,
}


def _create_generator_adapter(
    *,
    engine: str,
    engine_context: GeneratorEngineContext,
    worker_id: str,
) -> GeneratorEngineAdapter:
    """Construct the configured private engine adapter."""
    try:
        factory = _ENGINE_FACTORIES[engine]
    except KeyError as error:
        raise ValueError(f"unsupported MX_GENERATOR_ENGINE={engine!r}") from error
    if engine_context.engine_name != engine:
        raise ValueError(
            f"engine context {engine_context.engine_name!r} does not match "
            f"MX_GENERATOR_ENGINE={engine!r}"
        )
    return factory(engine_context, worker_id)


__all__: list[str] = []
