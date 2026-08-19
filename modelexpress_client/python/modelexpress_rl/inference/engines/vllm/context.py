# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public rank-local context required by the vLLM generator adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...adapter import GeneratorEngineContext

if TYPE_CHECKING:
    from torch.nn import Module
    from vllm.config import ModelConfig, VllmConfig


@dataclass(frozen=True)
class VllmGeneratorContext(GeneratorEngineContext):
    """Live vLLM objects required to install weights on one generator rank."""

    model: Module
    vllm_config: VllmConfig
    model_config: ModelConfig

    @property
    def engine_name(self) -> str:
        return "VLLM"


__all__ = ["VllmGeneratorContext"]
