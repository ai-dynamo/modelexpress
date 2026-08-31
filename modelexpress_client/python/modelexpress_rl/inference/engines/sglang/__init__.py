# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang generator integration for ModelExpress RL refit."""

from ...adapter import GeneratorEngineContext
from ...runtime import EngineRuntime
from .context import SglangGeneratorContext


def _create_sglang_engine_runtime(
    engine_context: GeneratorEngineContext,
) -> EngineRuntime:
    if not isinstance(engine_context, SglangGeneratorContext):
        raise TypeError("SGLang requires a SglangGeneratorContext")
    from .installer import _SglangInstaller

    runner = engine_context.model_runner
    return EngineRuntime(
        model_name=runner.model_config.model_path,
        installer=_SglangInstaller(runner),
    )


__all__ = ["SglangGeneratorContext"]
