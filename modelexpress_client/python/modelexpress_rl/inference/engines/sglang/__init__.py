# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang generator integration for ModelExpress RL refit."""

from ...adapter import GeneratorEngineContext
from ...receiver import ObjectStorageGeneratorConfig
from .adapter import SGLangGeneratorAdapter
from .context import SglangGeneratorContext


def _create_sglang_adapter(
    engine_context: GeneratorEngineContext,
    object_storage: ObjectStorageGeneratorConfig | None,
) -> SGLangGeneratorAdapter:
    if not isinstance(engine_context, SglangGeneratorContext):
        raise TypeError("SGLang requires a SglangGeneratorContext")
    if object_storage is None:
        raise ValueError("SGLang requires object storage")
    return SGLangGeneratorAdapter(
        context=engine_context,
        config=object_storage,
    )


__all__ = [
    "SGLangGeneratorAdapter",
    "SglangGeneratorContext",
]
