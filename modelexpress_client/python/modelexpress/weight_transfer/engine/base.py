# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract weight loader adapter."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    import torch.nn as nn
    from ..protocol.types import TrainerTable


class WeightLoaderAdapter(ABC):
    """Plug a new inference engine into the trainer pull/push workflow."""

    @abstractmethod
    def iter_lazy_weights(self, table: TrainerTable) -> Iterator[tuple[str, Any]]:
        """Yield (param_name, LazyWeight) pairs for the bake pass."""

    @abstractmethod
    def iter_param_shards(self, model: nn.Module) -> Iterator[tuple[str, Any]]:
        """Yield (param_name, tensor) pairs for building an InferenceTable."""

    def post_pull_hook(self, model: nn.Module) -> None:
        """Called after a PULL completes; override for FP8 repack etc."""

    def post_push_hook(self, model: nn.Module) -> None:
        """Called after a PUSH completes on the inference worker side."""
