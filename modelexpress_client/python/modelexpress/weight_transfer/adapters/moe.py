# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE stacked-expert naming adapter.

Expands stacked [num_experts, out, in] tensors into per-expert HF weight names
and renames the router gate (.mlp.router.gate. → .mlp.gate.).
"""

from __future__ import annotations

import re
from typing import Iterator

from .base import ModelAdapter
from ..engine.lazy import LazyWeight
from ..protocol.ops import OpSpec
from ..protocol.types import TrainerTable


class MoEAdapter(ModelAdapter):
    """HuggingFace MoE naming adapter (prime-rl / standard HF layout)."""

    _default_pattern = re.compile(r"^(.*\.mlp\.experts)\.(w1|w2|w3)$")
    _default_proj_map = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    _default_router_pattern = re.compile(r"\.mlp\.router\.gate\.")
    _default_router_replacement = ".mlp.gate."

    def __init__(
        self,
        num_experts: int,
        stacked_pattern: re.Pattern | None = None,
        projection_map: dict[str, str] | None = None,
        router_pattern: re.Pattern | None = None,
        router_replacement: str | None = None,
    ) -> None:
        self._num_experts = num_experts
        self._stacked_pattern = stacked_pattern or self._default_pattern
        self._projection_map = projection_map or self._default_proj_map
        self._router_pattern = router_pattern or self._default_router_pattern
        self._router_replacement = router_replacement or self._default_router_replacement

    def adapt_lazy_weights(
        self, lazy_weights: Iterator[tuple[str, LazyWeight]], table: TrainerTable
    ) -> Iterator[tuple[str, LazyWeight]]:
        for name, lazy in lazy_weights:
            hf_name = self._router_pattern.sub(self._router_replacement, name)
            m = self._stacked_pattern.match(hf_name)
            if m is None:
                yield hf_name, lazy
                continue
            prefix, proj_key = m.group(1), m.group(2)
            hf_proj = self._projection_map.get(proj_key)
            if hf_proj is None:
                yield hf_name, lazy
                continue
            import torch
            for j in range(self._num_experts):
                yield (
                    f"{prefix}.{j}.{hf_proj}.weight",
                    LazyWeight(
                        name=lazy._lazy_name,
                        shape=torch.Size(lazy.shape[1:]),
                        dtype=lazy.dtype,
                        op_chain=lazy._lazy_chain + (OpSpec(name="__getitem__", args=(j,), kwargs={}),),
                    ),
                )
