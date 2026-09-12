# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal shard-layout records for NCCL M2N mesh inference.

Shared layout records for the collective path. They describe global tensor
geometry and per-rank 2-D tile ownership so ``mesh.shard_dim_from_layout`` can
classify REPLICATE vs row/col sharding.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LayoutShard:
    """One rank's ownership of a 2-D tile within a parameter tensor."""

    agent_index: int
    row_start: int
    row_end: int
    device_addr: int = 0
    row_bytes: int = 0
    device_id: int = 0
    col_start: int = 0
    col_end: int = -1  # -1 = full width; resolved against LayoutTensor.shape


@dataclass
class LayoutTensor:
    """Shard descriptors for one parameter tensor across ranks."""

    name: str
    dtype: str  # e.g. "torch.bfloat16"
    shape: list[int]
    shards: list[LayoutShard] = field(default_factory=list)

    def resolved_col_end(self, shard: LayoutShard) -> int:
        """Return the effective col_end, resolving -1 to the full column count."""
        if shard.col_end == -1:
            return self.shape[1] if len(self.shape) > 1 else 1
        return shard.col_end


@dataclass
class LayoutTable:
    """Shared layout for one collective reshard step."""

    tensors: list[LayoutTensor]
    step: int = 0
