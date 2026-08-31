# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit trainer-engine selection for full-tensor publication."""

from dataclasses import dataclass


class TrainerEngineContext:
    """Identifies the trainer engine that owns the published tensors."""


@dataclass(frozen=True)
class FSDPTrainerContext(TrainerEngineContext):
    """Select FSDP/DTensor tensor capture and geometry."""


@dataclass(frozen=True)
class MegatronTrainerContext(TrainerEngineContext):
    """Select Megatron tensor capture and geometry."""


__all__ = ["FSDPTrainerContext", "MegatronTrainerContext", "TrainerEngineContext"]
