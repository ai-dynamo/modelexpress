# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-agnostic description of a worker's parallelism placement."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("modelexpress.topology")

_AXES = ("dp", "tp", "pp", "ep")


@dataclass(frozen=True)
class ParallelTopology:
    """Per-axis placement of one worker inside a parallel deployment.

    Every field is ``int | None``. A value of None means the engine did not
    tell us: it exposes no accessor for that axis, or it is not initialized
    yet. Unknown is kept rather than defaulted, because a fabricated rank 0
    silently collides every worker on that axis, and a fabricated size 1
    silently asserts a constraint the engine never stated.

    An axis the engine reports as unused is size 1 and rank 0. That is a
    read value, not a placeholder, and is distinct from None.

    Ranks are validated against their own size only when both are known, so
    an unread size never invalidates a rank that was read.
    """

    dp_rank: int | None = 0
    dp_size: int | None = 1
    tp_rank: int | None = 0
    tp_size: int | None = 1
    pp_rank: int | None = 0
    pp_size: int | None = 1
    ep_rank: int | None = 0
    ep_size: int | None = 1

    def __post_init__(self) -> None:
        for axis in _AXES:
            size = getattr(self, f"{axis}_size")
            rank = getattr(self, f"{axis}_rank")
            if size is not None and (not isinstance(size, int) or size < 1):
                raise ValueError(f"{axis}_size must be at least 1, got {size!r}")
            if rank is None:
                continue
            if not isinstance(rank, int) or rank < 0:
                raise ValueError(f"{axis}_rank must not be negative, got {rank!r}")
            if size is not None and rank >= size:
                raise ValueError(
                    f"{axis}_rank {rank!r} is outside [0, {size}) for axis {axis}"
                )


def build_topology(**axes: int | None) -> ParallelTopology:
    """Build a topology, downgrading any out-of-range rank to None.

    Engine adapters call this from publish paths, where a worker can be
    observed mid-initialization with a rank and a size that disagree.
    Raising there would take down a publish, so the inconsistent rank is
    dropped to unknown while the rest of the placement is kept. Every drop is
    logged at warning level, since an inconsistent placement usually means an
    engine accessor moved and an unreadable axis would otherwise look exactly
    like an absent one. Callers that want the inconsistency to surface should
    construct ParallelTopology directly.
    """
    try:
        return ParallelTopology(**axes)
    except ValueError:
        pass
    kept = dict(axes)
    for axis in _AXES:
        size = kept.get(f"{axis}_size", 1)
        rank = kept.get(f"{axis}_rank", 0)
        if size is not None and (not isinstance(size, int) or size < 1):
            logger.warning(
                "Dropping %s_size %r (with %s_rank %r): size must be at least 1",
                axis, size, axis, rank,
            )
            kept[f"{axis}_size"] = None
            size = None
        if rank is None:
            continue
        if not isinstance(rank, int) or rank < 0:
            logger.warning(
                "Dropping %s_rank %r (with %s_size %r): rank must not be negative",
                axis, rank, axis, size,
            )
            kept[f"{axis}_rank"] = None
        elif size is not None and rank >= size:
            logger.warning(
                "Dropping %s_rank %r (with %s_size %r): rank is outside [0, %d)",
                axis, rank, axis, size, size,
            )
            kept[f"{axis}_rank"] = None
    return ParallelTopology(**kept)


def flat_shard_rank(topology: ParallelTopology) -> int:
    """Return the flat model-shard key derived from a topology.

    The key identifies a distinct set of model weights, so data-parallel
    replicas share it and pipeline/tensor shards do not. Expert parallelism
    is not folded in: on the engines ModelExpress supports today the expert
    axis is colocated with the tensor axis, so ep_rank carries no
    information tp_rank does not already carry, and adding it would change
    published shard keys.

    Only tp_rank, tp_size and pp_rank contribute. pp_size does not, so an
    engine that reports a pipeline rank without a pipeline size still gets
    the same key it would have got from the tensor and pipeline ranks alone.

    Raises ValueError when a contributing value is unknown, since guessing
    would silently pair workers that own different weights.
    """
    for field in ("tp_rank", "tp_size", "pp_rank"):
        if getattr(topology, field) is None:
            raise ValueError(f"cannot derive a shard key with an unknown {field}")
    return topology.pp_rank * topology.tp_size + topology.tp_rank
