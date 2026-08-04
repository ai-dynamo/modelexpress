# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model naming adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from ..engine.lazy import LazyWeight
    from ..protocol.types import TrainerTable


class ModelAdapter(ABC):
    """Translate trainer tensor names/shapes to engine naming convention."""

    @abstractmethod
    def adapt_lazy_weights(
        self, lazy_weights: Iterator[tuple[str, LazyWeight]], table: TrainerTable
    ) -> Iterator[tuple[str, LazyWeight]]:
        """Yield (engine_param_name, LazyWeight) pairs the engine can consume."""
