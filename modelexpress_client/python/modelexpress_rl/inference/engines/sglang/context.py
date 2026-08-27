# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public rank-local context required by the SGLang generator adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...adapter import GeneratorEngineContext


@dataclass(frozen=True)
class SglangGeneratorContext(GeneratorEngineContext):
    """Live SGLang model runner used to install prepared checkpoints."""

    model_runner: Any


__all__ = ["SglangGeneratorContext"]
