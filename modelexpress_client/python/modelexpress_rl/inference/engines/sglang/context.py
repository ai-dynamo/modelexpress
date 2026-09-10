# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public rank-local context required by the SGLang generator integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...adapter import GeneratorEngineContext


@dataclass(frozen=True)
class SglangGeneratorContext(GeneratorEngineContext):
    """Live SGLang runner; full-tensor refit is opt-in for validated BF16 models."""

    model_runner: Any
    enable_full_tensor: bool = False


__all__ = ["SglangGeneratorContext"]
