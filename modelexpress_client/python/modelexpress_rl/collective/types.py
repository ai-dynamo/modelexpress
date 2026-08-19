# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch-free records for the NCCL M2N collective refit plan.

Nothing here imports torch, NCCL or CUDA, so the whole plan contract is
testable on a laptop. The backend converts these into the tensors and
placements the wire op wants; keeping the conversion at that boundary is what
lets the planning rules be checked without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    """Which side of the transfer a worker is on."""

    TRAINER = "TRAINER"
    GENERATOR = "GENERATOR"


class PlacementKind(str, Enum):
    REPLICATE = "REPLICATE"
    SHARD = "SHARD"


@dataclass(frozen=True)
class Placement:
    """How one mesh axis maps onto a tensor.

    Mirrors DTensor's ``Shard(dim)`` / ``Replicate()``. Kept as a plain record
    rather than the torch type so a plan can be built, hashed and compared
    without importing torch.
    """

    kind: PlacementKind
    dim: int | None = None

    def __post_init__(self) -> None:
        if self.kind is PlacementKind.SHARD:
            if self.dim is None or self.dim < 0:
                raise ValueError("a SHARD placement needs a non-negative tensor dim")
        elif self.dim is not None:
            raise ValueError("a REPLICATE placement must not name a tensor dim")

    @staticmethod
    def shard(dim: int) -> Placement:
        return Placement(PlacementKind.SHARD, dim)

    @staticmethod
    def replicate() -> Placement:
        return Placement(PlacementKind.REPLICATE)

    def canonical(self) -> str:
        """Stable text form, used by the plan digest."""
        return "R" if self.kind is PlacementKind.REPLICATE else f"S{self.dim}"

    def to_dtensor(self):
        """Convert to the torch DTensor placement the reshard op expects.

        torch is imported here rather than at module scope so the plan
        contract stays importable, and testable, without it.
        """
        from torch.distributed.tensor.placement_types import (  # noqa: PLC0415
            Replicate,
            Shard,
        )

        return Replicate() if self.kind is PlacementKind.REPLICATE else Shard(self.dim)


@dataclass(frozen=True)
class MeshSpec:
    """A rank grid, as a shape plus the rank the grid starts at.

    Ranks are lane-local: every reshard lane numbers its own participants from
    zero, trainers first. ``rank_offset`` is therefore where this side's ranks
    begin inside that lane, not a global rank.
    """

    shape: tuple[int, ...]
    rank_offset: int = 0

    def __post_init__(self) -> None:
        if not self.shape or any(extent <= 0 for extent in self.shape):
            raise ValueError("mesh shape must be non-empty and positive")
        if self.rank_offset < 0:
            raise ValueError("rank_offset must not be negative")

    @property
    def size(self) -> int:
        total = 1
        for extent in self.shape:
            total *= extent
        return total

    def ranks(self) -> list[int]:
        """The grid flattened in row-major order."""
        return [self.rank_offset + i for i in range(self.size)]

    def nested(self) -> list:
        """The grid nested to ``shape``, which is the form the reshard op takes.

        NeMo RL passes ``mesh.mesh.tolist()`` of a tensor already reshaped to
        the mesh shape, so a flat list would silently describe a different
        topology for any mesh with more than one axis.
        """
        flat = self.ranks()

        def build(dims: tuple[int, ...], offset: int):
            if len(dims) == 1:
                return flat[offset : offset + dims[0]]
            stride = 1
            for extent in dims[1:]:
                stride *= extent
            return [build(dims[1:], offset + i * stride) for i in range(dims[0])]

        return build(self.shape, 0)

    def canonical(self) -> str:
        return f"{'x'.join(str(e) for e in self.shape)}@{self.rank_offset}"


@dataclass(frozen=True)
class ParamPlan:
    """One bulk parameter's declared geometry on both sides.

    Declared by the Publisher, never inferred from the parameter name by the
    shared core. That inversion is what keeps the path portable across trainer
    frameworks, and it removes the failure mode where a mis-guessed mesh moves
    wrong bytes without erroring.
    """

    name: str
    global_shape: tuple[int, ...]
    dtype: str
    partition_id: int
    src_mesh: MeshSpec
    src_placements: tuple[Placement, ...]
    dst_mesh: MeshSpec
    dst_placements: tuple[Placement, ...]
    group_key: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a param plan needs a name")
        if not self.global_shape or any(extent <= 0 for extent in self.global_shape):
            raise ValueError(f"{self.name}: global_shape must be non-empty and positive")
        if self.partition_id < 0:
            raise ValueError(f"{self.name}: partition_id must not be negative")
        _check_placements(self.name, "src", self.src_mesh, self.src_placements, self.global_shape)
        _check_placements(self.name, "dst", self.dst_mesh, self.dst_placements, self.global_shape)

    def canonical(self) -> str:
        """Stable text form, used by the plan digest."""
        return "|".join(
            (
                self.name,
                ",".join(str(e) for e in self.global_shape),
                self.dtype,
                str(self.partition_id),
                self.src_mesh.canonical(),
                "".join(p.canonical() for p in self.src_placements),
                self.dst_mesh.canonical(),
                "".join(p.canonical() for p in self.dst_placements),
                self.group_key or "",
            )
        )


def _check_placements(
    name: str,
    side: str,
    mesh: MeshSpec,
    placements: tuple[Placement, ...],
    global_shape: tuple[int, ...],
) -> None:
    if len(placements) != len(mesh.shape):
        raise ValueError(
            f"{name}: {side} has {len(placements)} placements for a "
            f"{len(mesh.shape)}-dimensional mesh"
        )
    sharded: list[int] = []
    for axis, placement in enumerate(placements):
        if placement.kind is not PlacementKind.SHARD:
            continue
        dim = placement.dim
        if dim is None or dim >= len(global_shape):
            raise ValueError(
                f"{name}: {side} shards tensor dim {dim}, which the "
                f"{len(global_shape)}-dimensional parameter does not have"
            )
        if dim in sharded:
            # Two mesh axes sharding one tensor dim is a 2-D tile. NCCL can
            # express it, but the extent split between the axes is ambiguous
            # from this record alone, so it is rejected rather than guessed.
            raise ValueError(f"{name}: {side} shards tensor dim {dim} on two mesh axes")
        sharded.append(dim)
        extent = global_shape[dim]
        axis_size = mesh.shape[axis]
        if extent % axis_size != 0:
            raise ValueError(
                f"{name}: {side} splits dim {dim} of size {extent} across "
                f"{axis_size} ranks, which does not divide evenly"
            )


@dataclass(frozen=True)
class MiscParam:
    """One parameter that rides the packed broadcast instead of the reshard."""

    name: str
    global_shape: tuple[int, ...]
    dtype: str

    def canonical(self) -> str:
        return f"{self.name}|{','.join(str(e) for e in self.global_shape)}|{self.dtype}"


@dataclass
class ReshardPlan:
    """The full declared plan: bulk parameters plus the ordered misc list.

    The misc order is load-bearing. Producer and consumer walk it in lockstep
    during the packed broadcast, so a plan whose order differs between the two
    sides mismatches the payload without either side detecting it.
    """

    bulk: list[ParamPlan] = field(default_factory=list)
    misc: list[MiscParam] = field(default_factory=list)
    source_partition_count: int = 1

    def parameter_names(self) -> list[str]:
        return [p.name for p in self.bulk] + [p.name for p in self.misc]
