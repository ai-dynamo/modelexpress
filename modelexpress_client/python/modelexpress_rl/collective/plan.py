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
    plan.validate()
    declared = plan.parameter_names()
    counts = Counter(declared)

    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise PlanCoverageError(
            f"the plan names {len(duplicates)} parameter(s) more than once: "
            f"{', '.join(duplicates[:5])}"
        )

    expected_counts = Counter(expected)
    expected_duplicates = sorted(name for name, count in expected_counts.items() if count > 1)
    if expected_duplicates:
        raise PlanCoverageError(
            "the model parameter list names "
            f"{len(expected_duplicates)} parameter(s) more than once: "
            f"{', '.join(expected_duplicates[:5])}"
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


DEFAULT_RECEIVER_PROTOCOL = "mx-nccl-m2n-bootstrap-v1"


def reshard_run_key(entry: ParamPlan) -> tuple[int, str, str]:
    """The run an entry belongs to: its lane, source mesh and destination mesh.

    Entries sharing a key go to the same communicator with the same topology,
    which is what nccl.m2n can PACK-fuse inside one ``m2n.group()``.
    """
    return (entry.partition_id, entry.src_mesh.canonical(), entry.dst_mesh.canonical())


def submission_order(entries: list[ParamPlan]) -> list[ParamPlan]:
    """The order one layer group's reshards are recorded in, on every rank.

    Sorted by run key, then by the entry's canonical form, so runs are
    contiguous and in ascending lane order. A rank on one lane records a
    subsequence of the same total order, which is what M2N needs from buckets
    whose communicator memberships overlap: one common order on every rank.
    """
    return sorted(entries, key=lambda entry: (reshard_run_key(entry), entry.canonical()))


def reshard_runs(entries: list[ParamPlan]) -> list[list[ParamPlan]]:
    """Split one layer group, in submission order, into its runs."""
    runs: list[list[ParamPlan]] = []
    for entry in submission_order(entries):
        if runs and reshard_run_key(runs[-1][-1]) == reshard_run_key(entry):
            runs[-1].append(entry)
        else:
            runs.append([entry])
    return runs


def plan_digest(
    plan: ReshardPlan,
    *,
    receiver_protocol: str = DEFAULT_RECEIVER_PROTOCOL,
    m2n_abi_version: str = "",
    layer_groups: list[list[str]] | None = None,
    blocking_data_plane: bool = True,
    grouped_submission: bool = False,
) -> str:
    """A canonical digest over the whole plan.

    Every participant computes this independently and reports it when it joins.
    MX admits the group only when they all agree, and a change bumps the group
    epoch, which drops the cached plan and the cached communicator together.

    ``receiver_protocol`` names the bootstrap encoding both sides must share
    and ``m2n_abi_version`` the M2N build they must be compatible on. Peers
    that disagree on either produce different digests, so the group never
    reaches READY instead of meeting inside a collective and finding out.

    Both lists are hashed in order. Misc order is the broadcast payload layout,
    so it is part of the wire contract. Bulk order is not: the backend sorts
    bulk entries canonically before issuing them, so two orderings of the same
    entries produce identical wire traffic. Bulk is still hashed in order as a
    conservative admission rule, which means peers that enumerate the same
    entries differently are refused rather than admitted.

    ``layer_groups`` is the bulk grouping from ``setup_layer_groups``. Each
    layer group id names a set of wire operations, so peers that split bulk
    differently would issue different operations under the same id. Group
    order is hashed; names within a group are not, since the backend issues a
    group in canonical order.

    The run boundaries of every layer group are hashed too: the lane, meshes
    and length of each run, in submission order. Each layer group is recorded
    as one ``m2n.group()`` whose per-communicator contents must match on every
    rank, so peers that would split or order runs differently - a different
    grouping rule, not only a different grouping - are refused at READY.

    ``blocking_data_plane`` and ``grouped_submission`` are the execution modes
    the client froze from ``MX_NCCL_REFIT_BLOCKING_DATA_PLANE`` and
    ``MX_NCCL_REFIT_GROUPED_SUBMISSION``. The first decides whether a rank
    enters ``Communicator.split()`` and runs on the child, the second whether
    a layer group is one ``m2n.group()`` or one call per entry. Peers that
    disagree on either would diverge inside native collectives, so they are
    refused at READY too.
    """
    plan.validate()
    hasher = hashlib.sha256()
    hasher.update(b"mx-nccl-m2n-plan-v3\0")
    # The digest is what readiness compares, so anything two peers must agree
    # on before they meet inside a collective belongs in it. The plan says what
    # moves; these say whether both sides can speak about it at all. They are
    # supplied by the caller rather than probed here, because this module is
    # deliberately importable without nccl4py present.
    hasher.update(f"{receiver_protocol}\0".encode())
    hasher.update(f"{m2n_abi_version}\0".encode())
    hasher.update(f"{plan.source_partition_count}\0".encode())
    hasher.update(
        f"mode\0blocking={int(blocking_data_plane)}|grouped={int(grouped_submission)}\0".encode()
    )

    hasher.update(b"bulk\0")
    for entry in plan.bulk:
        hasher.update(entry.canonical().encode())
        hasher.update(b"\0")

    hasher.update(b"misc\0")
    for entry in plan.misc:
        hasher.update(entry.canonical().encode())
        hasher.update(b"\0")

    if layer_groups is not None:
        hasher.update(f"groups\0{len(layer_groups)}\0".encode())
        for group in layer_groups:
            hasher.update(f"{len(group)}\0".encode())
            for name in sorted(group):
                encoded = name.encode()
                hasher.update(f"{len(encoded)}:".encode())
                hasher.update(encoded)

    by_name = {entry.name: entry for entry in plan.bulk}
    groups = (
        [list(plan.bulk)]
        if layer_groups is None
        else [[by_name[name] for name in group if name in by_name] for group in layer_groups]
    )
    hasher.update(f"runs\0{len(groups)}\0".encode())
    for group in groups:
        runs = reshard_runs(group)
        hasher.update(f"{len(runs)}\0".encode())
        for run in runs:
            lane, src_mesh, dst_mesh = reshard_run_key(run[0])
            hasher.update(f"{lane}|{src_mesh}|{dst_mesh}|{len(run)}\0".encode())

    return hasher.hexdigest()


def validate_layer_groups(plan: ReshardPlan, groupings: list[list[str]]) -> None:
    """Refuse a grouping the plan cannot execute.

    Every bulk parameter must be in exactly one group, and every name must be
    a bulk parameter of this plan. The client runs this on the captured plan
    before it joins, so a worker never reaches READY and bootstraps a cohort
    with a grouping it would only reject afterwards.
    """
    bulk_names = {entry.name for entry in plan.bulk}
    seen: set[str] = set()
    for group_id, names in enumerate(groupings):
        name_set = set(names)
        if len(name_set) != len(names):
            raise ValueError(f"layer group {group_id} names a parameter more than once")
        overlap = seen & name_set
        if overlap:
            raise ValueError(f"{min(overlap)} appears in more than one layer group")
        unknown = sorted(name_set - bulk_names)
        if unknown:
            raise KeyError(f"{unknown[0]} is not a bulk parameter in this plan")
        seen.update(name_set)

    uncovered = sorted(bulk_names - seen)
    if uncovered:
        raise ValueError(
            f"layer groups leave {len(uncovered)} bulk parameter(s) uncovered: "
            f"{', '.join(uncovered[:5])}"
        )


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
    if any(size <= 0 for size in (tp_size, ep_size, dp_size, pp_size)):
        raise ValueError("parallelism sizes must be positive")
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
        # A grouped expert tensor carries the expert dimension in front. EP
        # shards that leading dimension, while TP still shards the shifted
        # projection dimension when both axes are present.
        if "ep" in axis_of:
            placements[axis_of["ep"]] = Placement.shard(0)
        if "tp" in axis_of:
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
    if trainer_count <= 0:
        raise ValueError("trainer_count must be positive")
    if source_partition_count <= 0:
        raise ValueError("source_partition_count must be positive")
    if trainer_count % source_partition_count != 0:
        raise ValueError(
            f"trainer count {trainer_count} is not divisible by "
            f"{source_partition_count} partitions"
        )
    return trainer_count // source_partition_count
