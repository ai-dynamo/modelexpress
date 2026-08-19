# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan validation, digesting, and the opt-in default derivation.

Two things live here, and the split matters:

*Mechanism* is the coverage gate and the digest. Every deployment gets them,
and neither can be opted out of, because both failures they prevent are silent:
a parameter in neither list keeps serving its previous value while the refit
reports success, and a plan that drifted from the one a generator cached moves
bytes laid out for a model that no longer exists.

*Policy* is the default mesh, placement and bulk-set derivation at the bottom
of this module. It reproduces the conventional Megatron-style layout so the
common case stays cheap, and a Publisher whose ranks are laid out differently
declares its own and never calls it.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter

from .types import MESH_MAX_NDIM, MeshSpec, MiscParam, ParamPlan, Placement, ReshardPlan

# FFN projection weights. Column-parallel projections shard the output dim,
# the row-parallel one shards the input dim.
_COLUMN_PARALLEL_SUFFIXES = ("gate_proj.weight", "up_proj.weight")
_ROW_PARALLEL_SUFFIXES = ("down_proj.weight",)
_EXPERT_MARKER = ".experts."
_SHARED_EXPERT_MARKER = "shared_expert"

_GROUPED_EXPERT_RE = re.compile(r"(.+\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")


class PlanCoverageError(ValueError):
    """The declared plan does not partition the model.

    Deliberately not a generic ValueError at the call site: a coverage failure
    means the refit would have silently served a mixed model version, which is
    a different class of problem from a malformed record.
    """


def validate_coverage(plan: ReshardPlan, expected: list[str]) -> None:
    """Prove the plan names every expected parameter exactly once.

    ``expected`` is the model's full canonical parameter list, as the Publisher
    sees it. This is a gate rather than a guideline because both failure modes
    are silent at refit time:

    - a missing parameter never moves, and the destination keeps serving its
      previous value while the refit reports success;
    - a duplicated parameter is applied twice, once through each path, with the
      later write deciding.
    """
    declared = plan.parameter_names()
    counts = Counter(declared)

    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise PlanCoverageError(
            f"the plan names {len(duplicates)} parameter(s) more than once: "
            f"{', '.join(duplicates[:5])}"
        )

    expected_set = set(expected)
    declared_set = set(declared)

    missing = sorted(expected_set - declared_set)
    if missing:
        raise PlanCoverageError(
            f"the plan does not cover {len(missing)} parameter(s): {', '.join(missing[:5])}"
        )

    unknown = sorted(declared_set - expected_set)
    if unknown:
        raise PlanCoverageError(
            f"the plan names {len(unknown)} parameter(s) the model does not have: "
            f"{', '.join(unknown[:5])}"
        )


def plan_digest(plan: ReshardPlan) -> str:
    """A canonical digest over the whole plan.

    Every participant computes this independently and reports it when it joins.
    MX admits the group only when they all agree, and a change bumps the group
    epoch, which drops the cached plan and the cached communicator together.

    The bulk set is hashed order-independently, because two Publishers may
    enumerate parameters differently and still describe the same transfer. The
    misc list is hashed *in order*, because its order is the payload layout.
    """
    hasher = hashlib.sha256()
    hasher.update(b"mx-nccl-m2n-plan-v1\0")
    hasher.update(f"{plan.source_partition_count}\0".encode())

    hasher.update(b"bulk\0")
    for canonical in sorted(entry.canonical() for entry in plan.bulk):
        hasher.update(canonical.encode())
        hasher.update(b"\0")

    hasher.update(b"misc\0")
    for entry in plan.misc:
        hasher.update(entry.canonical().encode())
        hasher.update(b"\0")

    return hasher.hexdigest()


def is_bulk_param(name: str) -> bool:
    """Default bulk classification: FFN projection weights.

    A profiling result about MoE models rather than a property of the
    transport, which is why a Publisher can override it by classifying
    parameters itself. Shared-expert weights are excluded because their layout
    does not follow the per-expert convention this default assumes.
    """
    if _SHARED_EXPERT_MARKER in name:
        return False
    return name.endswith(_COLUMN_PARALLEL_SUFFIXES + _ROW_PARALLEL_SUFFIXES)


def default_shard_dim(name: str) -> int | None:
    """Tensor dim the default derivation shards, or None to replicate."""
    if name.endswith(_COLUMN_PARALLEL_SUFFIXES):
        return 0
    if name.endswith(_ROW_PARALLEL_SUFFIXES):
        return 1
    return None


def is_expert_param(name: str) -> bool:
    return _EXPERT_MARKER in name


def grouped_expert_name(name: str) -> str | None:
    """Collapse ``experts.N.gate_proj.weight`` into its grouped entry.

    Per-expert parameters are moved as one stacked ``[E, ...]`` tensor, so the
    plan names the group rather than each expert. Returns None for a name that
    is not a per-expert weight.
    """
    match = _GROUPED_EXPERT_RE.match(name)
    if match is None:
        return None
    return f"{match.group(1)}.{match.group(3)}.weight"


def build_mesh(
    *,
    rank_count: int,
    rank_offset: int = 0,
    tp_size: int = 1,
    ep_size: int = 1,
    dp_size: int = 1,
    pp_size: int = 1,
) -> tuple[MeshSpec, dict[str, int]]:
    """Default mesh derivation, and the axis each parallelism landed on.

    Dims are emitted in the order ``(tp, ep, dp, pp)``, size-1 dims are dropped,
    and the survivors are reversed into a row-major grid, so the first
    surviving dim becomes the innermost, fastest-varying axis. That reproduces
    the conventional Megatron rank layout, which is the only reason this
    function can exist at all: it is a guess, correct for one family of
    trainers, and a Publisher that lays its ranks out differently must declare
    its own mesh rather than call this.
    """
    if rank_count <= 0:
        raise ValueError("rank_count must be positive")
    declared = tp_size * ep_size * dp_size * pp_size
    if declared != rank_count:
        raise ValueError(
            f"tp*ep*dp*pp = {declared} does not account for {rank_count} ranks"
        )

    ordered = [("tp", tp_size), ("ep", ep_size), ("dp", dp_size), ("pp", pp_size)]
    active = [(axis, size) for axis, size in ordered if size > 1]
    if not active:
        return MeshSpec(shape=(rank_count,), rank_offset=rank_offset), {}

    reversed_active = list(reversed(active))
    if len(reversed_active) > MESH_MAX_NDIM:
        raise ValueError(
            f"{len(reversed_active)} parallelism axes are larger than one "
            f"({', '.join(name for name, _ in active)}), but a mesh may carry at "
            f"most {MESH_MAX_NDIM}. Declare the mesh explicitly, or collapse the "
            "axes so at most two are larger than one."
        )
    shape = tuple(size for _, size in reversed_active)
    axis_of = {name: index for index, (name, _) in enumerate(reversed_active)}
    return MeshSpec(shape=shape, rank_offset=rank_offset), axis_of


def default_placements(name: str, axis_of: dict[str, int], ndim: int) -> tuple[Placement, ...]:
    """Default placement derivation for one parameter.

    One-dimensional parameters replicate. Expert parameters shard the leading
    expert dim on the EP axis, which pushes any tensor-parallel split one dim
    to the right because the grouped tensor carries the expert dim in front.
    """
    axis_count = max(len(axis_of), 1)
    placements: list[Placement] = [Placement.replicate() for _ in range(axis_count)]
    if ndim < 2:
        return tuple(placements)

    if is_expert_param(name):
        if "ep" in axis_of:
            placements[axis_of["ep"]] = Placement.shard(0)
        elif "tp" in axis_of:
            shard_dim = default_shard_dim(name)
            if shard_dim is not None:
                placements[axis_of["tp"]] = Placement.shard(shard_dim + 1)
        return tuple(placements)

    shard_dim = default_shard_dim(name)
    if shard_dim is not None and "tp" in axis_of:
        placements[axis_of["tp"]] = Placement.shard(shard_dim)
    return tuple(placements)


def build_param_plan(
    *,
    name: str,
    global_shape: tuple[int, ...],
    dtype: str,
    partition_id: int,
    src_mesh: MeshSpec,
    src_axis_of: dict[str, int],
    dst_mesh: MeshSpec,
    dst_axis_of: dict[str, int],
    group_key: str | None = None,
) -> ParamPlan:
    """Assemble one bulk entry using the default placement derivation."""
    ndim = len(global_shape)
    return ParamPlan(
        name=name,
        global_shape=global_shape,
        dtype=dtype,
        partition_id=partition_id,
        src_mesh=src_mesh,
        src_placements=default_placements(name, src_axis_of, ndim),
        dst_mesh=dst_mesh,
        dst_placements=default_placements(name, dst_axis_of, ndim),
        group_key=group_key,
    )


def generator_rank_offset(trainer_count: int, source_partition_count: int) -> int:
    """Where generator ranks begin inside a reshard lane.

    Trainers occupy the low ranks of a lane and generators follow, so a
    generator's ``dst_mesh`` starts after that partition's trainer ranks. This
    mirrors the server's assignment rule; the two must agree or every rank
    builds a mesh that disagrees with the rank it was actually given.
    """
    if source_partition_count <= 0:
        raise ValueError("source_partition_count must be positive")
    if trainer_count % source_partition_count != 0:
        raise ValueError(
            f"trainer count {trainer_count} is not divisible by "
            f"{source_partition_count} partitions"
        )
    return trainer_count // source_partition_count
