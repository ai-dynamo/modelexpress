# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-agnostic RL fan-out source policies."""

from __future__ import annotations

from dataclasses import dataclass

from modelexpress.rl_metadata import RlSourceRole


@dataclass(frozen=True)
class RlTreeFanoutPolicy:
    """Source policy for a bounded-fanout replica tree.

    Trainer rank ``root_source_rank`` seeds the first ``fanout`` replica ranks.
    Every later replica pulls from one earlier inference replica and can
    republish after refit. This keeps the policy independent of veRL, SLIME, or
    Miles; framework adapters only map their rollout rank to ``receiver_rank``.
    """

    receiver_rank: int
    replica_world_size: int
    fanout: int = 2
    root_source_rank: int = 0

    def __post_init__(self) -> None:
        if self.replica_world_size <= 0:
            raise ValueError("replica_world_size must be positive")
        if not 0 <= self.receiver_rank < self.replica_world_size:
            raise ValueError("receiver_rank must be within replica_world_size")
        if self.fanout <= 0:
            raise ValueError("fanout must be positive")
        if self.root_source_rank < 0:
            raise ValueError("root_source_rank must be non-negative")

    @property
    def parent_replica_rank(self) -> int | None:
        """Return the replica parent rank, or None when the trainer is parent."""
        if self.receiver_rank < self.fanout:
            return None
        return (self.receiver_rank - self.fanout) // self.fanout

    @property
    def roles(self) -> tuple[RlSourceRole, ...]:
        """Return source roles to query, ordered by preference."""
        if self.parent_replica_rank is None:
            return (RlSourceRole.TRAINER,)
        return (RlSourceRole.INFERENCE_REPLICA,)

    @property
    def source_ranks_by_role(self) -> dict[RlSourceRole, tuple[int, ...]]:
        """Return the only source rank this receiver should pull from."""
        parent_rank = self.parent_replica_rank
        if parent_rank is None:
            return {RlSourceRole.TRAINER: (self.root_source_rank,)}
        return {RlSourceRole.INFERENCE_REPLICA: (parent_rank,)}
