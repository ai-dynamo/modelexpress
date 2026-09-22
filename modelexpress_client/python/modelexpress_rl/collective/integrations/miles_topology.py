# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure MILES topology projection into the ModelExpress reshard plan."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from ..types import MeshSpec, ParamPlan, Placement, ReshardPlan

MilesTensorFamily = Literal["dense", "routed_expert"]


@dataclass(frozen=True)
class MilesTrainerTopology:
    """One trainer rank's MILES parallel coordinates."""

    world_rank: int
    pp_rank: int
    pp_size: int
    tp_rank: int
    tp_size: int
    cp_rank: int
    cp_size: int
    dense_dp_rank: int
    dense_dp_size: int
    ep_rank: int
    ep_size: int
    etp_rank: int
    etp_size: int
    expert_dp_rank: int
    expert_dp_size: int
    independent_dp_rank: int
    independent_dp_size: int


@dataclass(frozen=True)
class MilesTensorSpec:
    """One native source contribution to a canonical rollout parameter."""

    world_rank: int
    source_name: str
    canonical_name: str
    family: MilesTensorFamily
    global_shape: tuple[int, ...]
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if not self.source_name or not self.canonical_name:
            raise ValueError("MILES tensor names must not be empty")
        if self.family not in ("dense", "routed_expert"):
            raise ValueError(f"unsupported MILES tensor family {self.family!r}")
        if not self.global_shape or any(dim <= 0 for dim in self.global_shape):
            raise ValueError(
                f"{self.canonical_name}: global_shape must be non-empty and positive"
            )
        if not self.dtype:
            raise ValueError(f"{self.canonical_name}: dtype must not be empty")


@dataclass(frozen=True)
class MilesSourceRoute:
    """Native owners and names that prepare one bulk plan entry."""

    canonical_name: str
    family: MilesTensorFamily
    partition_id: int
    source_world_ranks: tuple[int, ...]
    source_names_by_world: tuple[tuple[int, tuple[str, ...]], ...]


@dataclass(frozen=True)
class MilesReshardTopologyPlan:
    """The collective plan plus MILES-local routing and scheduling metadata."""

    plan: ReshardPlan
    trainer_lanes: tuple[tuple[int, ...], ...]
    routes: tuple[MilesSourceRoute, ...]
    residual_update_units: tuple[tuple[str, ...], ...]
    waves: tuple[tuple[int, ...], ...]
    layer_groups: tuple[tuple[str, ...], ...]


_SIZE_FIELDS = (
    "pp",
    "tp",
    "cp",
    "dense_dp",
    "ep",
    "etp",
    "expert_dp",
    "independent_dp",
)
_DECODER_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


def _validate_topology(
    records: Sequence[MilesTrainerTopology],
) -> dict[str, int]:
    if not records:
        raise ValueError("cannot build a MILES plan without trainer topology")
    world_ranks = [record.world_rank for record in records]
    if len(world_ranks) != len(set(world_ranks)):
        raise ValueError("trainer topology contains duplicate world ranks")

    sizes: dict[str, int] = {}
    for field in _SIZE_FIELDS:
        values = {getattr(record, f"{field}_size") for record in records}
        if len(values) != 1:
            raise ValueError(f"inconsistent trainer {field} sizes: {sorted(values)}")
        size = values.pop()
        if size <= 0:
            raise ValueError(f"trainer {field} size must be positive")
        sizes[field] = size
        for record in records:
            rank = getattr(record, f"{field}_rank")
            if not 0 <= rank < size:
                raise ValueError(
                    f"invalid trainer {field} rank {rank} for size {size} "
                    f"on world rank {record.world_rank}"
                )
    if sizes["etp"] != 1:
        raise ValueError(f"MILES NCCL M2N requires trainer ETP=1, got {sizes['etp']}")

    coordinate_views = {
        "dense": ("pp", "tp", "cp", "dense_dp", "independent_dp"),
        "expert": ("pp", "ep", "etp", "expert_dp", "independent_dp"),
    }
    for family, fields in coordinate_views.items():
        expected = 1
        for field in fields:
            expected *= sizes[field]
        coordinates = {
            tuple(getattr(record, f"{field}_rank") for field in fields)
            for record in records
        }
        if len(records) != expected or len(coordinates) != expected:
            raise ValueError(
                f"MILES {family} topology describes {expected} ranks across "
                f"{fields}, but received {len(records)} trainer ranks with "
                f"{len(coordinates)} unique coordinates"
            )

    return sizes


def _owners_by_partition(
    records: Sequence[MilesTrainerTopology],
    sizes: dict[str, int],
) -> dict[tuple[int, MilesTensorFamily], tuple[int, ...]]:
    owners: dict[tuple[int, MilesTensorFamily], tuple[int, ...]] = {}
    for pp_rank in range(sizes["pp"]):
        dense = tuple(
            record.world_rank
            for record in sorted(records, key=lambda item: item.tp_rank)
            if record.pp_rank == pp_rank
            and record.independent_dp_rank == 0
            and record.dense_dp_rank == 0
            and record.cp_rank == 0
        )
        expert = tuple(
            record.world_rank
            for record in sorted(
                records,
                key=lambda item: (item.ep_rank, item.etp_rank),
            )
            if record.pp_rank == pp_rank
            and record.independent_dp_rank == 0
            and record.expert_dp_rank == 0
        )
        if len(dense) != sizes["tp"]:
            raise ValueError(
                f"PP={pp_rank} requires {sizes['tp']} canonical dense owners, "
                f"got {dense}"
            )
        if len(expert) != sizes["ep"] * sizes["etp"]:
            raise ValueError(
                f"PP={pp_rank} requires {sizes['ep'] * sizes['etp']} canonical "
                f"expert owners, got {expert}"
            )
        owners[(pp_rank, "dense")] = dense
        owners[(pp_rank, "routed_expert")] = expert
    return owners


def _build_trainer_lanes(
    records: Sequence[MilesTrainerTopology],
    sizes: dict[str, int],
    owners: dict[tuple[int, MilesTensorFamily], tuple[int, ...]],
) -> tuple[tuple[int, ...], ...]:
    topology_by_world = {record.world_rank: record for record in records}
    lanes: list[tuple[int, ...]] = []
    for pp_rank in range(sizes["pp"]):
        stage = [record for record in records if record.pp_rank == pp_rank]
        semantic_order = tuple(
            record.world_rank
            for record in sorted(
                stage,
                key=lambda item: (
                    item.independent_dp_rank,
                    item.expert_dp_rank,
                    item.ep_rank,
                    item.etp_rank,
                    item.dense_dp_rank,
                    item.cp_rank,
                    item.tp_rank,
                ),
            )
        )
        dense = owners[(pp_rank, "dense")]
        expert = owners[(pp_rank, "routed_expert")]
        candidates = (
            tuple(dict.fromkeys((*expert, *dense, *semantic_order))),
            tuple(dict.fromkeys((*dense, *expert, *semantic_order))),
            semantic_order,
        )
        for lane in candidates:
            if len(lane) != len(stage):
                continue
            try:
                _source_rank_offset(dense, lane, f"PP={pp_rank} dense owners")
                _source_rank_offset(expert, lane, f"PP={pp_rank} expert owners")
            except ValueError:
                continue
            lanes.append(lane)
            break
        else:
            dense_coordinates = [topology_by_world[rank].tp_rank for rank in dense]
            expert_coordinates = [
                (
                    topology_by_world[rank].ep_rank,
                    topology_by_world[rank].etp_rank,
                )
                for rank in expert
            ]
            raise ValueError(
                f"PP={pp_rank} cannot form one coordinate-stable trainer lane "
                f"for dense TP owners {dense_coordinates} and expert EP/ETP "
                f"owners {expert_coordinates}"
            )
    lane_sizes = {len(lane) for lane in lanes}
    if len(lane_sizes) != 1:
        raise ValueError(
            f"trainer PP lane sizes are inconsistent: {sorted(lane_sizes)}"
        )
    return tuple(lanes)


def _projection_shard_dim(name: str, family: MilesTensorFamily) -> int:
    if family == "routed_expert":
        if ".experts." not in name:
            raise ValueError(f"{name}: routed expert parameter lacks .experts.")
        return 0
    if name.endswith(("gate_proj.weight", "up_proj.weight")):
        return 0
    if name.endswith("down_proj.weight"):
        return 1
    raise ValueError(f"{name}: unsupported dense MILES projection")


def _decoder_layer(name: str) -> int:
    match = _DECODER_LAYER_RE.match(name)
    if match is None:
        raise ValueError(f"{name}: unsupported canonical decoder parameter name")
    return int(match.group(1))


def _required_canonical_outputs(name: str) -> frozenset[str]:
    for component in ("gate", "up"):
        suffix = f"{component}_proj.weight"
        if name.endswith(suffix):
            prefix = name.removesuffix(suffix)
            return frozenset(
                {
                    f"{prefix}gate_proj.weight",
                    f"{prefix}up_proj.weight",
                }
            )
    if name.endswith("down_proj.weight"):
        return frozenset({name})
    raise ValueError(f"{name}: unsupported canonical FFN projection")


def _partition_for_specs(
    specs: Sequence[MilesTensorSpec],
    topology_by_world: dict[int, MilesTrainerTopology],
) -> int:
    try:
        partitions = {topology_by_world[spec.world_rank].pp_rank for spec in specs}
    except KeyError as exc:
        raise ValueError(
            f"tensor spec names unknown trainer world rank {exc.args[0]}"
        ) from exc
    if len(partitions) != 1:
        raise ValueError(
            f"{specs[0].canonical_name}: source owners span PP partitions "
            f"{sorted(partitions)}"
        )
    return partitions.pop()


def _source_rank_offset(
    owner_ranks: tuple[int, ...],
    lane: tuple[int, ...],
    name: str,
) -> int:
    lane_positions = [lane.index(rank) for rank in owner_ranks]
    expected = list(range(lane_positions[0], lane_positions[0] + len(lane_positions)))
    if lane_positions != expected:
        raise ValueError(
            f"{name}: source owners must be contiguous in the trainer lane; "
            f"got positions {lane_positions}"
        )
    return lane_positions[0]


def _normalize_update_units(
    update_units: Iterable[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    normalized: list[tuple[str, ...]] = []
    for unit in update_units:
        names = tuple(str(name) for name in unit)
        if not names or any(not name for name in names):
            raise ValueError("MILES update units must contain non-empty names")
        if len(names) != len(set(names)):
            raise ValueError(f"MILES update unit contains duplicate names: {names}")
        normalized.append(names)
    if len(normalized) != len(set(normalized)):
        raise ValueError("MILES update units must not be duplicated")
    return tuple(sorted(normalized))


def _route_source_names(route: MilesSourceRoute) -> frozenset[str]:
    return frozenset(
        source_name
        for _, source_names in route.source_names_by_world
        for source_name in source_names
    )


def _fused_projection(
    name: str,
) -> tuple[str, Literal["gate", "up"]] | None:
    for component in ("gate", "up"):
        suffix = f"{component}_proj.weight"
        if name.endswith(suffix):
            return name.removesuffix(suffix), component
    return None


def _validate_fused_projection_pairs(
    candidates: Sequence[tuple[ParamPlan, MilesSourceRoute]],
) -> None:
    pairs: dict[
        str,
        dict[Literal["gate", "up"], tuple[ParamPlan, MilesSourceRoute]],
    ] = {}
    for candidate in candidates:
        projection = _fused_projection(candidate[0].name)
        if projection is None:
            continue
        prefix, component = projection
        pairs.setdefault(prefix, {})[component] = candidate

    for prefix, pair in sorted(pairs.items()):
        if pair.keys() != {"gate", "up"}:
            continue
        gate_entry, gate_route = pair["gate"]
        up_entry, up_route = pair["up"]
        label = prefix.rstrip(".")
        if gate_entry.global_shape != up_entry.global_shape:
            raise ValueError(
                f"{label}: fused gate/up projections have incompatible global_shape"
            )
        if gate_entry.dtype != up_entry.dtype:
            raise ValueError(
                f"{label}: fused gate/up projections have incompatible dtype"
            )
        if gate_route.source_names_by_world != up_route.source_names_by_world:
            raise ValueError(
                f"{label}: fused gate/up projections have incompatible source mapping"
            )
        if (
            gate_route.family != up_route.family
            or gate_route.partition_id != up_route.partition_id
            or gate_route.source_world_ranks != up_route.source_world_ranks
        ):
            raise ValueError(
                f"{label}: fused gate/up projections have incompatible source ownership"
            )
        if (
            gate_entry.src_mesh != up_entry.src_mesh
            or gate_entry.src_placements != up_entry.src_placements
            or gate_entry.dst_mesh != up_entry.dst_mesh
            or gate_entry.dst_placements != up_entry.dst_placements
            or gate_entry.group_key != up_entry.group_key
        ):
            raise ValueError(
                f"{label}: fused gate/up projections have incompatible geometry"
            )


def _routed_atomic_units(
    retained: Sequence[tuple[ParamPlan, MilesSourceRoute]],
    update_units: Sequence[tuple[str, ...]],
) -> tuple[
    tuple[tuple[str, ...], ...],
    set[tuple[str, tuple[tuple[int, tuple[str, ...]], ...]]],
]:
    routed_units: list[tuple[str, ...]] = []
    routed_candidates: set[tuple[str, tuple[tuple[int, tuple[str, ...]], ...]]] = set()
    for unit in update_units:
        unit_names = frozenset(unit)
        candidates = [
            candidate
            for candidate in retained
            if _route_source_names(candidate[1]).issubset(unit_names)
        ]
        covered_names = frozenset(
            source_name
            for _, route in candidates
            for source_name in _route_source_names(route)
        )
        if covered_names != unit_names:
            continue
        available_outputs = {entry.name for entry, _ in candidates}
        required_outputs = frozenset(
            required
            for entry, _ in candidates
            for required in _required_canonical_outputs(entry.name)
        )
        if not required_outputs or not required_outputs.issubset(available_outputs):
            continue
        _validate_fused_projection_pairs(candidates)
        routed_units.append(unit)
        routed_candidates.update(
            (entry.name, route.source_names_by_world) for entry, route in candidates
        )
    return tuple(routed_units), routed_candidates


def build_miles_reshard_plan(
    topologies: Sequence[MilesTrainerTopology],
    tensor_specs: Sequence[MilesTensorSpec],
    *,
    update_units: Iterable[Sequence[str]],
    rollout_engine_sizes: Sequence[int],
    pp_wave_size: int,
) -> MilesReshardTopologyPlan:
    """Project native MILES ownership into PR 795 collective records."""

    records = tuple(topologies)
    sizes = _validate_topology(records)
    owners = _owners_by_partition(records, sizes)
    trainer_lanes = _build_trainer_lanes(records, sizes, owners)
    topology_by_world = {record.world_rank: record for record in records}

    engine_sizes = tuple(int(size) for size in rollout_engine_sizes)
    if not engine_sizes or any(size <= 0 for size in engine_sizes):
        raise ValueError(
            f"MILES requires positive rollout engine sizes, got {engine_sizes}"
        )
    if len(set(engine_sizes)) != 1:
        raise ValueError(
            f"MILES requires homogeneous rollout engine sizes, got {engine_sizes}"
        )
    if pp_wave_size <= 0:
        raise ValueError("pp_wave_size must be positive")

    grouped: dict[str, list[MilesTensorSpec]] = {}
    for spec in tensor_specs:
        grouped.setdefault(spec.canonical_name, []).append(spec)

    candidates: list[tuple[ParamPlan, MilesSourceRoute]] = []
    layer_owners: dict[int, int] = {}
    for canonical_name in sorted(grouped):
        specs = sorted(
            grouped[canonical_name],
            key=lambda item: (item.world_rank, item.source_name),
        )
        families = {spec.family for spec in specs}
        shapes = {spec.global_shape for spec in specs}
        dtypes = {spec.dtype for spec in specs}
        if len(families) != 1 or len(shapes) != 1 or len(dtypes) != 1:
            raise ValueError(f"{canonical_name}: source tensor specs are inconsistent")
        family = families.pop()
        global_shape = shapes.pop()
        dtype = dtypes.pop()
        partition_id = _partition_for_specs(specs, topology_by_world)
        layer = _decoder_layer(canonical_name)
        previous_partition = layer_owners.setdefault(layer, partition_id)
        if previous_partition != partition_id:
            raise ValueError(
                f"global decoder layer {layer} is reported by PP partitions "
                f"{previous_partition} and {partition_id}"
            )
        expected_owners = owners[(partition_id, family)]
        actual_owner_set = {spec.world_rank for spec in specs}
        if actual_owner_set != set(expected_owners):
            raise ValueError(
                f"{canonical_name}: expected source owners {expected_owners}, "
                f"got {tuple(sorted(actual_owner_set))}"
            )
        actual_owners = expected_owners
        names_by_world = tuple(
            (
                world_rank,
                tuple(
                    sorted(
                        {
                            spec.source_name
                            for spec in specs
                            if spec.world_rank == world_rank
                        }
                    )
                ),
            )
            for world_rank in actual_owners
        )
        shard_dim = _projection_shard_dim(canonical_name, family)
        lane = trainer_lanes[partition_id]
        source_offset = _source_rank_offset(actual_owners, lane, canonical_name)
        source_mesh = MeshSpec(
            shape=(1, len(actual_owners)),
            rank_offset=source_offset,
        )
        destination_mesh = MeshSpec(
            shape=(len(engine_sizes), engine_sizes[0]),
            rank_offset=len(lane),
        )
        destination_placements = (
            Placement.replicate(),
            Placement.shard(shard_dim),
        )
        entry = ParamPlan(
            name=canonical_name,
            global_shape=global_shape,
            dtype=dtype,
            partition_id=partition_id,
            src_mesh=source_mesh,
            src_placements=(
                Placement.replicate(),
                Placement.shard(shard_dim),
            ),
            dst_mesh=destination_mesh,
            dst_placements=destination_placements,
            group_key=f"pp-wave-{partition_id // pp_wave_size}",
        )
        route = MilesSourceRoute(
            canonical_name=canonical_name,
            family=family,
            partition_id=partition_id,
            source_world_ranks=actual_owners,
            source_names_by_world=names_by_world,
        )
        candidates.append((entry, route))

    normalized_units = _normalize_update_units(update_units)
    retained = list(candidates)
    while True:
        routed_units, routed_candidates = _routed_atomic_units(
            retained,
            normalized_units,
        )
        next_retained = [
            candidate
            for candidate in retained
            if (candidate[0].name, candidate[1].source_names_by_world)
            in routed_candidates
        ]
        if len(next_retained) == len(retained):
            break
        retained = next_retained
    if not retained:
        raise ValueError("no complete atomic MILES update unit can use NCCL M2N")

    retained.sort(key=lambda item: (item[0].partition_id, item[0].name))
    entries = [entry for entry, _ in retained]
    routes = tuple(route for _, route in retained)
    plan = ReshardPlan(
        bulk=entries,
        source_partition_count=sizes["pp"],
    )
    waves = tuple(
        tuple(range(start, min(start + pp_wave_size, sizes["pp"])))
        for start in range(0, sizes["pp"], pp_wave_size)
    )
    layer_groups = tuple(
        tuple(entry.name for entry in entries if entry.partition_id in wave)
        for wave in waves
    )
    routed_unit_set = set(routed_units)
    residual_units = tuple(
        unit for unit in normalized_units if unit not in routed_unit_set
    )
    return MilesReshardTopologyPlan(
        plan=plan,
        trainer_lanes=trainer_lanes,
        routes=routes,
        residual_update_units=residual_units,
        waves=waves,
        layer_groups=layer_groups,
    )
