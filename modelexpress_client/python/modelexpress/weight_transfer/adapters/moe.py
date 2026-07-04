# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MoE stacked-expert naming adapter.

Training frameworks typically store all experts in a single stacked tensor:

    model.layers.N.mlp.experts.w1  shape: [num_experts, out, in]

HuggingFace checkpoints (and vLLM's loader) expect per-expert tensors:

    model.layers.N.mlp.experts.0.gate_proj.weight  shape: [out, in]
    model.layers.N.mlp.experts.1.gate_proj.weight
    ...

This adapter handles the explosion (one stacked -> N per-expert) and the
router gate renaming (.mlp.router.gate. -> .mlp.gate.).

To support a different MoE convention, subclass MoEAdapter and override
``_projection_map`` and ``_stacked_expert_pattern``.
"""

from __future__ import annotations

import re
from typing import Iterator

from .base import ModelAdapter
from ..engine.lazy import LazyWeight
from ..protocol.ops import OpSpec
from ..protocol.types import TrainerTable


class MoEAdapter(ModelAdapter):
    """HuggingFace MoE naming adapter.

    Args:
        num_experts: Total number of experts in the model.
        stacked_pattern: Regex matching stacked expert tensor names.
            Must have two groups: (prefix, projection_key).
        projection_map: Maps training projection keys to HF weight names.
        router_pattern: Regex for router gate renaming (matched on full name).
        router_replacement: Replacement string for router_pattern.
    """

    # Default: prime-rl / standard HF MoE layout
    _default_pattern = re.compile(r"^(.*\.mlp\.experts)\.(w1|w2|w3)$")
    _default_proj_map = {
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
    }
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
        self,
        lazy_weights: Iterator[tuple[str, LazyWeight]],
        table: TrainerTable,
    ) -> Iterator[tuple[str, LazyWeight]]:
        """Yield HF-named lazy weights, expanding stacked expert tensors."""
        for name, lazy in lazy_weights:
            # Router gate renaming (no shape change)
            hf_name = self._router_pattern.sub(self._router_replacement, name)

            m = self._stacked_pattern.match(hf_name)
            if m is None:
                yield hf_name, lazy
                continue

            prefix = m.group(1)
            proj_key = m.group(2)
            hf_proj = self._projection_map.get(proj_key)
            if hf_proj is None:
                yield hf_name, lazy
                continue

            # Expand stacked [num_experts, out, in] -> N x [out, in]
            import torch
            for j in range(self._num_experts):
                expert_name = f"{prefix}.{j}.{hf_proj}.weight"
                slice_op = OpSpec(name="__getitem__", args=(j,), kwargs={})
                expert_lazy = LazyWeight(
                    name=lazy._lazy_name,
                    shape=torch.Size(lazy.shape[1:]),
                    dtype=lazy.dtype,
                    op_chain=lazy._lazy_chain + (slice_op,),
                )
                yield expert_name, expert_lazy
