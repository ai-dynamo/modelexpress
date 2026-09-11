# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit trainer-engine selection for full-tensor publication."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class TrainerEngineContext:
    """Identifies the trainer engine that owns the published tensors."""


@dataclass(frozen=True)
class FSDPTrainerContext(TrainerEngineContext):
    """Select FSDP/DTensor tensor capture and geometry."""

    wire_dtype_overrides: Mapping[str, torch.dtype] = field(default_factory=dict)
    """Exact state-dict names to transfer as FP16, BF16 or FP32; other tensors use BF16."""


@dataclass(frozen=True)
class MegatronTrainerContext(TrainerEngineContext):
    """Select Megatron tensor capture and geometry."""


__all__ = ["FSDPTrainerContext", "MegatronTrainerContext", "TrainerEngineContext"]
