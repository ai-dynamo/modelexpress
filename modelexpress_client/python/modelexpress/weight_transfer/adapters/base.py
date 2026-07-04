# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model naming adapter interface.

Inference engines and training frameworks frequently use different
parameter naming conventions.  A ModelAdapter translates names and
shapes between the two worlds without touching the weight loader logic.

Example adapters to implement:
  - MoEAdapter        stacked [num_experts, ...] -> per-expert HF names
  - MegatronAdapter   Megatron-LM column/row parallel naming
  - DeepSpeedAdapter  ZeRO stage 3 flat buffer naming
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from ..engine.lazy import LazyWeight
    from ..protocol.types import TrainerTable


class ModelAdapter(ABC):
    """Translate trainer tensor names/shapes to engine naming convention.

    Implement ``adapt_lazy_weights`` to convert the raw LazyWeight iterator
    (keyed by trainer names) into the iterator the engine's ``load_weights``
    expects.  A single trainer tensor may expand into multiple engine tensors
    (e.g. stacked expert -> per-expert) or be renamed/reshaped.
    """

    @abstractmethod
    def adapt_lazy_weights(
        self,
        lazy_weights: Iterator[tuple[str, LazyWeight]],
        table: TrainerTable,
    ) -> Iterator[tuple[str, LazyWeight]]:
        """Yield (engine_param_name, LazyWeight) pairs the engine can consume.

        Args:
            lazy_weights: Raw (trainer_name, LazyWeight) pairs from the table.
            table: Full TrainerTable, available for shape/dtype lookups.
        """
