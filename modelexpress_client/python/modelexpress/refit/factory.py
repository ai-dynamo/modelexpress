# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Delta receiver construction, dispatched on the rollout backend."""

from __future__ import annotations

from enum import Enum
from typing import Any

from .receiver import ModelExpressWeightReceiver


class RolloutBackend(str, Enum):
    """The inference engine a rollout worker runs."""

    SGLANG = "sglang"
    VLLM = "vllm"


def build_delta_receiver(
    backend: RolloutBackend | str, **kwargs: Any
) -> ModelExpressWeightReceiver:
    """Build the delta receiver for ``backend``.

    Each branch imports its own module, because importing either one pulls in
    that engine. Keyword arguments are the chosen receiver's constructor
    arguments: ``config`` and ``receiver_id`` for both, plus ``model_runner``
    for SGLang or ``engine`` for vLLM.
    """
    backend = RolloutBackend(backend)
    if backend is RolloutBackend.SGLANG:
        from ..engines.sglang.refit.receiver import SglangWeightReceiver

        return SglangWeightReceiver(**kwargs)
    from ..engines.vllm.refit.delta_receiver import VllmWeightReceiver

    return VllmWeightReceiver(**kwargs)
