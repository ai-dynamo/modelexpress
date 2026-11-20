# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public rank-local context required by the vLLM generator integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...adapter import GeneratorEngineContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.nn import Module
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class VllmGeneratorContext(GeneratorEngineContext):
    """Live vLLM objects required to install weights on one generator rank."""

    model: Module
    vllm_config: VllmConfig
    # Maps a trainer-native source state_dict to HF names/layout before capture.
    convert_native_to_hf: Callable[[dict], dict] | None = None
    # Install trainer refits through vLLM's own load_weights instead of replaying
    # traced loader writes: each source is staged whole, so any model vLLM can
    # reload installs without its loader being traceable.
    install_through_loader: bool = False


__all__ = ["VllmGeneratorContext"]
