# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PR 3304 manifest projection and normal MILES protocol integration."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import grpc

from modelexpress import auth

from ... import refit_collective_pb2 as collective_pb
from .. import envs
from ..rendezvous import CollectiveRendezvous
from ..types import MeshSpec, ParamPlan, Placement, ReshardPlan
from ._common import _endpoint, _local_shape, _text
from .miles import (
    CollectiveTopology,
    MilesTrainerSession,
    MilesTransferCoordinator,
)
from .miles_native import (
    MilesNativePublisher,
    MilesNativeTensorRecord,
    MilesSourceRecipe,
    inventory_miles_native_tensors,
)
from .miles_topology import (
    MilesReshardTopologyPlan,
    MilesSourceRoute,
    MilesTrainerTopology,
)
from .wire import (
    CollectiveControl,
    DestinationManifestEntry,
    DestinationQuantization,
    encode_control,
)

logger = logging.getLogger("modelexpress_rl.collective.integrations.miles_pr3304")

SEMANTIC_MANIFEST_VERSION = "miles-nccl-m2n-manifest-v1"
_TERMINAL_STATES = frozenset(
    {
        collective_pb.COLLECTIVE_TRANSFER_STATE_COMPLETE,
        collective_pb.COLLECTIVE_TRANSFER_STATE_FAILED,
        collective_pb.COLLECTIVE_TRANSFER_STATE_ABORTED,
    }
)


@dataclass(frozen=True)
class MilesPr3304Projection:
    """One verified MILES manifest translated into ModelExpress records."""

    manifest: dict[str, Any]
    semantic_manifest_version: str
    semantic_manifest_digest: str
    topology_plan: MilesReshardTopologyPlan
    destination_manifest: tuple[DestinationManifestEntry, ...]


def canonical_pr3304_manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Return the exact canonical digest used by MILES PR 3304."""

    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer_list(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty integer list")
    return tuple(_integer(item, f"{name}[]") for item in value)


def _mesh_rows(value: object, name: str) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list of mesh rows")
    rows = tuple(_integer_list(row, f"{name}[]") for row in value)
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be rectangular")
    flat = tuple(rank for row in rows for rank in row)
    if len(flat) != len(set(flat)):
        raise ValueError(f"{name} contains duplicate ranks")
    return rows


def _placement(value: object, name: str) -> Placement:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    kind = value.get("type")
    if kind == "replicate":
        if set(value) != {"type"}:
            raise ValueError(f"{name}: replicate placement has unknown fields")
        return Placement.replicate()
    if kind == "shard":
        if set(value) != {"type", "dim"}:
            raise ValueError(f"{name}: shard placement has unknown fields")
        return Placement.shard(_integer(value.get("dim"), f"{name}.dim"))
    raise ValueError(f"{name}: unsupported placement type {kind!r}")


def _placements(value: object, name: str, ndim: int) -> tuple[Placement, ...]:
    if not isinstance(value, list) or len(value) != ndim:
        raise ValueError(f"{name} must contain exactly {ndim} placements")
    return tuple(
        _placement(item, f"{name}[{index}]") for index, item in enumerate(value)
    )


def _topology_order(record: MilesTrainerTopology) -> tuple[int, ...]:
    return (
        record.independent_dp_rank,
        record.expert_dp_rank,
        record.ep_rank,
        record.etp_rank,
        record.dense_dp_rank,
        record.cp_rank,
        record.tp_rank,
        record.world_rank,
    )


def _owner_sequences_by_partition(
    entries: Sequence[Mapping[str, Any]],
    comm_to_world: Mapping[int, int],
    pp_size: int,
) -> dict[int, tuple[tuple[int, ...], ...]]:
    result: dict[int, list[tuple[int, ...]]] = {pp: [] for pp in range(pp_size)}
    for index, entry in enumerate(entries):
        pp_rank = _integer(entry.get("pp_rank"), f"entries[{index}].pp_rank")
        if pp_rank >= pp_size:
            raise ValueError(f"entries[{index}].pp_rank exceeds trainer PP size")
        source = entry.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"entries[{index}].source must be an object")
        source_rows = _mesh_rows(source.get("mesh"), f"entries[{index}].source.mesh")
        try:
            owners = tuple(comm_to_world[rank] for row in source_rows for rank in row)
        except KeyError as error:
            raise ValueError(
                f"entries[{index}] source rank {error.args[0]} is not a trainer"
            ) from error
        if owners not in result[pp_rank]:
            result[pp_rank].append(owners)
    return {pp: tuple(sequences) for pp, sequences in result.items()}


def _lane_supports(
    lane: tuple[int, ...],
    owner_sequences: Sequence[tuple[int, ...]],
) -> bool:
    for owners in owner_sequences:
        try:
            positions = tuple(lane.index(owner) for owner in owners)
        except ValueError:
            return False
        if positions != tuple(range(positions[0], positions[0] + len(positions))):
            return False
    return True


def _trainer_lanes(
    topologies: Sequence[MilesTrainerTopology],
    owner_sequences: Mapping[int, Sequence[tuple[int, ...]]],
) -> tuple[tuple[int, ...], ...]:
    if not topologies:
        raise ValueError("trainer topology must not be empty")
    pp_sizes = {record.pp_size for record in topologies}
    if len(pp_sizes) != 1:
        raise ValueError(f"trainer PP sizes disagree: {sorted(pp_sizes)}")
    pp_size = pp_sizes.pop()
    world_ranks = [record.world_rank for record in topologies]
    if len(world_ranks) != len(set(world_ranks)):
        raise ValueError("trainer topology contains duplicate world ranks")

    lanes = []
    for pp_rank in range(pp_size):
        stage = tuple(record for record in topologies if record.pp_rank == pp_rank)
        if not stage:
            raise ValueError(f"trainer topology has no ranks for PP={pp_rank}")
        stage_world = {record.world_rank for record in stage}
        sequences = tuple(owner_sequences.get(pp_rank, ()))
        if not sequences:
            raise ValueError(f"manifest has no routed entries for PP={pp_rank}")
        if any(not set(sequence).issubset(stage_world) for sequence in sequences):
            raise ValueError(f"manifest source owners escape PP={pp_rank}")
        semantic = tuple(
            record.world_rank for record in sorted(stage, key=_topology_order)
        )
        candidates = [semantic]
        for sequence_order in itertools.permutations(sequences):
            candidates.append(
                tuple(
                    dict.fromkeys(
                        (
                            *(rank for sequence in sequence_order for rank in sequence),
                            *semantic,
                        )
                    )
                )
            )
        lane = next(
            (
                candidate
                for candidate in candidates
                if len(candidate) == len(stage)
                and set(candidate) == stage_world
                and _lane_supports(candidate, sequences)
            ),
            None,
        )
        if lane is None:
            raise ValueError(
                f"PP={pp_rank} cannot project manifest owners into one "
                "contiguous ModelExpress trainer lane"
            )
        lanes.append(lane)
    widths = {len(lane) for lane in lanes}
    if len(widths) != 1:
        raise ValueError(f"trainer PP lane widths disagree: {sorted(widths)}")
    return tuple(lanes)


def _project_mesh(
    rows: tuple[tuple[int, ...], ...],
    rank_map: Mapping[int, int],
    name: str,
) -> MeshSpec:
    try:
        projected = tuple(tuple(rank_map[rank] for rank in row) for row in rows)
    except KeyError as error:
        raise ValueError(
            f"{name} rank {error.args[0]} has no projected rank"
        ) from error
    flat = tuple(rank for row in projected for rank in row)
    expected = tuple(range(flat[0], flat[0] + len(flat)))
    if flat != expected:
        raise ValueError(
            f"{name} must project to one contiguous row-major interval; got {flat}"
        )
    return MeshSpec((len(projected), len(projected[0])), rank_offset=flat[0])


def _normalize_update_units(
    units: Iterable[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    normalized = []
    for unit in units:
        names = tuple(_string(name, "update unit name") for name in unit)
        if not names or len(names) != len(set(names)):
            raise ValueError(f"invalid MILES update unit {names}")
        normalized.append(names)
    if len(normalized) != len(set(normalized)):
        raise ValueError("MILES update units contain duplicates")
    return tuple(sorted(normalized))


def _destination_quantization(
    value: object,
) -> DestinationQuantization | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "quant_method",
        "activation_scheme",
        "weight_block_size",
        "weight_dtype",
        "scale_dtype",
        "scale_format",
    }:
        raise ValueError("PR 3304 quantization metadata is invalid")
    block_size = _integer_list(
        value.get("weight_block_size"),
        "quantization.weight_block_size",
    )
    if len(block_size) != 2:
        raise ValueError("quantization.weight_block_size must contain two dimensions")
    return DestinationQuantization(
        quant_method=_string(value.get("quant_method"), "quantization.quant_method"),
        activation_scheme=_string(
            value.get("activation_scheme"),
            "quantization.activation_scheme",
        ),
        weight_block_size=block_size,
        weight_dtype=_string(value.get("weight_dtype"), "quantization.weight_dtype"),
        scale_dtype=_string(value.get("scale_dtype"), "quantization.scale_dtype"),
        scale_format=_string(value.get("scale_format"), "quantization.scale_format"),
    )


def project_pr3304_manifest(
    manifest: Mapping[str, Any],
    trainer_topologies: Sequence[MilesTrainerTopology],
    *,
    update_units: Iterable[Sequence[str]],
    pp_wave_size: int,
) -> MilesPr3304Projection:
    """Translate an already-decided PR 3304 manifest without re-deriving policy."""

    frozen = deepcopy(dict(manifest))
    if set(frozen) - {
        "schema_version",
        "source_world_ranks",
        "trainer_world_to_comm_rank",
        "communicator_world_size",
        "routed_update_units",
        "entries",
        "quantization",
        "manifest_hash",
    }:
        raise ValueError("PR 3304 manifest contains unknown top-level fields")
    if _integer(frozen.get("schema_version"), "schema_version", minimum=1) != 1:
        raise ValueError("unsupported PR 3304 manifest schema version")
    manifest_hash = _string(frozen.get("manifest_hash"), "manifest_hash")
    if (
        len(manifest_hash) != 64
        or any(character not in "0123456789abcdef" for character in manifest_hash)
        or manifest_hash != canonical_pr3304_manifest_digest(frozen)
    ):
        raise ValueError("PR 3304 manifest hash validation failed")
    entries = frozen.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("PR 3304 manifest entries must be a non-empty list")
    quantization = _destination_quantization(frozen.get("quantization"))

    source_world = _integer_list(frozen.get("source_world_ranks"), "source_world_ranks")
    raw_world_to_comm = frozen.get("trainer_world_to_comm_rank")
    if not isinstance(raw_world_to_comm, dict):
        raise ValueError("trainer_world_to_comm_rank must be an object")
    world_to_comm = {
        int(world): _integer(comm, f"trainer_world_to_comm_rank[{world!r}]")
        for world, comm in raw_world_to_comm.items()
    }
    if tuple(world_to_comm) != source_world:
        raise ValueError(
            "trainer_world_to_comm_rank order must match source_world_ranks"
        )
    if tuple(world_to_comm.values()) != tuple(range(len(source_world))):
        raise ValueError("trainer communicator ranks must be contiguous from zero")
    comm_to_world = {comm: world for world, comm in world_to_comm.items()}

    records = tuple(trainer_topologies)
    pp_sizes = {record.pp_size for record in records}
    if len(pp_sizes) != 1:
        raise ValueError("trainer topology must agree on PP size")
    pp_size = pp_sizes.pop()
    if pp_wave_size <= 0:
        raise ValueError("pp_wave_size must be positive")
    owner_sequences = _owner_sequences_by_partition(entries, comm_to_world, pp_size)
    trainer_lanes = _trainer_lanes(records, owner_sequences)

    communicator_world_size = _integer(
        frozen.get("communicator_world_size"),
        "communicator_world_size",
        minimum=1,
    )
    destination_count = communicator_world_size - len(source_world)
    if destination_count <= 0:
        raise ValueError("PR 3304 manifest contains no destination ranks")

    waves = tuple(
        tuple(range(start, min(start + pp_wave_size, pp_size)))
        for start in range(0, pp_size, pp_wave_size)
    )
    wave_by_partition = {
        partition: wave_index
        for wave_index, wave in enumerate(waves)
        for partition in wave
    }

    plan_entries = []
    routes = []
    destination_manifest = []
    seen_names = set()
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"entries[{index}] must be an object")
        name = _string(raw_entry.get("name"), f"entries[{index}].name")
        if name in seen_names:
            raise ValueError(f"duplicate PR 3304 manifest entry {name!r}")
        seen_names.add(name)
        family = raw_entry.get("family")
        if family not in ("dense", "routed_expert"):
            raise ValueError(f"{name}: unsupported MILES tensor family {family!r}")
        partition_id = _integer(raw_entry.get("pp_rank"), f"{name}.pp_rank")
        if partition_id >= pp_size:
            raise ValueError(f"{name}: PP rank exceeds trainer topology")
        global_shape = _integer_list(
            raw_entry.get("global_shape"), f"{name}.global_shape"
        )
        dtype = _string(raw_entry.get("dtype"), f"{name}.dtype")
        source = raw_entry.get("source")
        destination = raw_entry.get("destination")
        if not isinstance(source, dict) or not isinstance(destination, dict):
            raise ValueError(f"{name}: source and destination must be objects")
        source_rows = _mesh_rows(source.get("mesh"), f"{name}.source.mesh")
        source_local_shape = _integer_list(
            source.get("local_shape"),
            f"{name}.source.local_shape",
        )
        destination_rows = _mesh_rows(
            destination.get("mesh"), f"{name}.destination.mesh"
        )
        destination_local_shape = _integer_list(
            destination.get("local_shape"),
            f"{name}.destination.local_shape",
        )
        destination_parameter = _string(
            destination.get("parameter"),
            f"{name}.destination.parameter",
        )
        destination_recipe = _string(
            destination.get("recipe"),
            f"{name}.destination.recipe",
        )
        pair_id = raw_entry.get("pair_id")
        tensor_role = raw_entry.get("tensor_role")
        if (pair_id is None) != (tensor_role is None):
            raise ValueError(f"{name}: pair_id and tensor_role must be set together")
        if pair_id is not None:
            pair_id = _string(pair_id, f"{name}.pair_id")
            if tensor_role not in ("weight", "scale"):
                raise ValueError(f"{name}: unsupported tensor_role {tensor_role!r}")
            if family != "routed_expert" or quantization is None:
                raise ValueError(
                    f"{name}: paired tensors require routed_expert quantization"
                )
        elif quantization is not None:
            raise ValueError(
                f"{name}: quantized manifests require pair metadata on every entry"
            )
        lane = trainer_lanes[partition_id]
        source_rank_map = {
            comm_rank: lane.index(world_rank)
            for comm_rank, world_rank in comm_to_world.items()
            if world_rank in lane
        }
        destination_rank_map = {
            len(source_world) + generator_rank: len(lane) + generator_rank
            for generator_rank in range(destination_count)
        }
        src_mesh = _project_mesh(
            source_rows,
            source_rank_map,
            f"{name}.source.mesh",
        )
        dst_mesh = _project_mesh(
            destination_rows,
            destination_rank_map,
            f"{name}.destination.mesh",
        )
        source_names = source.get("names_by_rank")
        if not isinstance(source_names, dict):
            raise ValueError(f"{name}.source.names_by_rank must be an object")
        names_by_world = []
        for raw_rank, raw_names in source_names.items():
            comm_rank = int(raw_rank)
            if comm_rank not in comm_to_world:
                raise ValueError(
                    f"{name}: source recipe names non-trainer rank {comm_rank}"
                )
            if not isinstance(raw_names, list) or not raw_names:
                raise ValueError(f"{name}: source names must be a non-empty list")
            names_by_world.append(
                (
                    comm_to_world[comm_rank],
                    tuple(_string(item, f"{name}.source name") for item in raw_names),
                )
            )
        names_by_world.sort(key=lambda item: source_rank_map[world_to_comm[item[0]]])
        owners = tuple(world for world, _ in names_by_world)
        mesh_owners = tuple(comm_to_world[rank] for row in source_rows for rank in row)
        if owners != mesh_owners:
            raise ValueError(
                f"{name}: source names and source mesh owner order disagree"
            )
        src_placements = _placements(
            source.get("placements"),
            f"{name}.source.placements",
            len(src_mesh.shape),
        )
        dst_placements = _placements(
            destination.get("placements"),
            f"{name}.destination.placements",
            len(dst_mesh.shape),
        )
        projected_source_shape = _local_shape(
            global_shape,
            src_mesh,
            src_placements,
        )
        if source_local_shape != projected_source_shape:
            raise ValueError(
                f"{name}: source local shape {source_local_shape} differs from "
                f"projected collective shape {projected_source_shape}"
            )
        projected_destination_shape = _local_shape(
            global_shape,
            dst_mesh,
            dst_placements,
        )
        if destination_local_shape != projected_destination_shape:
            raise ValueError(
                f"{name}: destination local shape {destination_local_shape} differs "
                f"from projected collective shape {projected_destination_shape}"
            )
        plan_entries.append(
            ParamPlan(
                name=name,
                global_shape=global_shape,
                dtype=dtype,
                partition_id=partition_id,
                src_mesh=src_mesh,
                src_placements=src_placements,
                dst_mesh=dst_mesh,
                dst_placements=dst_placements,
                group_key=f"pp-wave-{wave_by_partition[partition_id]}",
            )
        )
        routes.append(
            MilesSourceRoute(
                canonical_name=name,
                family=family,
                partition_id=partition_id,
                source_world_ranks=owners,
                source_names_by_world=tuple(names_by_world),
            )
        )
        destination_manifest.append(
            DestinationManifestEntry(
                name=name,
                dtype=dtype,
                local_shape=destination_local_shape,
                parameter=destination_parameter,
                recipe=destination_recipe,
                family=family,
                pair_id=pair_id,
                tensor_role=tensor_role,
                quantization=quantization,
            )
        )

    plan = ReshardPlan(bulk=plan_entries, source_partition_count=pp_size)
    normalized_units = _normalize_update_units(update_units)
    routed_units = _normalize_update_units(frozen.get("routed_update_units", ()))
    unknown_routed = sorted(set(routed_units) - set(normalized_units))
    if unknown_routed:
        raise ValueError(
            f"manifest routes unknown MILES update units {unknown_routed[:3]}"
        )
    residual_units = tuple(
        unit for unit in normalized_units if unit not in set(routed_units)
    )
    layer_groups = tuple(
        tuple(entry.name for entry in plan_entries if entry.partition_id in wave)
        for wave in waves
    )
    topology_plan = MilesReshardTopologyPlan(
        plan=plan,
        trainer_lanes=trainer_lanes,
        routes=tuple(routes),
        residual_update_units=residual_units,
        waves=waves,
        layer_groups=layer_groups,
    )
    return MilesPr3304Projection(
        manifest=frozen,
        semantic_manifest_version=SEMANTIC_MANIFEST_VERSION,
        semantic_manifest_digest=manifest_hash,
        topology_plan=topology_plan,
        destination_manifest=tuple(destination_manifest),
    )


def _source_recipe(value: object, name: str) -> MilesSourceRecipe:
    recipe = _string(value, f"{name}.source.recipe")
    if recipe.endswith("_scale"):
        recipe = recipe.removesuffix("_scale")
    try:
        return {
            "dense_fc1_0": MilesSourceRecipe.DENSE_FC1_GATE,
            "dense_fc1_1": MilesSourceRecipe.DENSE_FC1_UP,
            "dense_fc2": MilesSourceRecipe.DENSE_FC2,
            "expert_fc1_0": MilesSourceRecipe.EXPERT_FC1_GATE,
            "expert_fc1_1": MilesSourceRecipe.EXPERT_FC1_UP,
            "expert_fc2": MilesSourceRecipe.EXPERT_FC2,
        }[recipe]
    except KeyError as error:
        raise ValueError(
            f"{name}: unsupported PR 3304 source recipe {recipe!r}"
        ) from error


def validate_native_publisher_manifest(
    publisher: MilesNativePublisher,
    manifest: Mapping[str, Any],
    *,
    source_world_rank: int,
) -> None:
    """Prove the publisher will execute the exact source semantics that were hashed."""

    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("PR 3304 manifest entries must be a list")
    manifest_by_name = {
        _string(entry.get("name"), "manifest entry name"): entry
        for entry in entries
        if isinstance(entry, dict)
    }
    bindings = publisher.source_bindings()
    for binding in bindings:
        try:
            entry = manifest_by_name[binding.canonical_name]
        except KeyError as error:
            raise ValueError(
                f"{binding.canonical_name}: publisher binding is absent from manifest"
            ) from error
        source = entry.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"{binding.canonical_name}: manifest source is invalid")
        names_by_rank = source.get("names_by_rank")
        if not isinstance(names_by_rank, dict):
            raise ValueError(
                f"{binding.canonical_name}: manifest source names are invalid"
            )
        raw_world_to_comm = manifest.get("trainer_world_to_comm_rank")
        if not isinstance(raw_world_to_comm, dict):
            raise ValueError("trainer_world_to_comm_rank must be an object")
        try:
            comm_rank = int(raw_world_to_comm[str(source_world_rank)])
        except KeyError as error:
            raise ValueError(
                f"source world rank {source_world_rank} is absent from manifest"
            ) from error
        manifest_names = tuple(names_by_rank.get(str(comm_rank), ()))
        if (
            binding.source_names != manifest_names
            or binding.recipe
            is not _source_recipe(
                source.get("recipe"),
                binding.canonical_name,
            )
            or binding.pair_id != entry.get("pair_id")
            or binding.tensor_role != entry.get("tensor_role")
        ):
            raise ValueError(
                f"{binding.canonical_name}: native publisher semantics differ "
                "from the hashed PR 3304 manifest"
            )


def _check_response(response: Any) -> Any:
    success = (
        response.get("success")
        if isinstance(response, dict)
        else getattr(response, "success", None)
    )
    message = (
        response.get("message", "")
        if isinstance(response, dict)
        else getattr(response, "message", "")
    )
    if success is not True:
        raise RuntimeError(str(message or "SGLang rejected the collective command"))
    return response


def _arg(args: Any, name: str, default: Any = None) -> Any:
    value = getattr(args, name, None)
    return default if value is None else value


def _pr3304_hf_exclusions(manifest: Mapping[str, Any]) -> set[str]:
    names = set()
    for entry in manifest["entries"]:
        if entry["family"] == "routed_expert":
            prefix, suffix = entry["name"].split(".experts.", 1)
            names.update(
                f"{prefix}.experts.{expert}.{suffix}"
                for expert in range(entry["global_shape"][0])
            )
        else:
            names.add(entry["name"])
    return names


def _gloo_group():
    from miles.utils.distributed_utils import get_gloo_group

    return get_gloo_group()


class MilesPr3304ProtocolCore:
    """ModelExpress routed phase embedded in the normal PR 3304 lifecycle."""

    supports_lora = False
    use_weight_update_session = True

    def __init__(self, args: Any) -> None:
        self.args = args
        self._iterator = None
        self._model = None
        self._inventory: tuple[MilesNativeTensorRecord, ...] = ()
        self._local_payload: dict[str, Any] | None = None
        self._quantization: dict[str, Any] | None = None
        self._projection: MilesPr3304Projection | None = None
        self._topology: CollectiveTopology | None = None
        self._publisher: MilesNativePublisher | None = None
        self._session: MilesTrainerSession | None = None
        self._coordinator: MilesTransferCoordinator | None = None
        self._rendezvous: CollectiveRendezvous | None = None
        self._channel = None
        self._generator_endpoint: str | None = None
        self._engine_gpu_counts: tuple[int, ...] = ()
        self._engine_gpu_offsets: tuple[int, ...] = ()
        self._pending_version: str | None = None
        self._deferred_update_error: str | None = None
        self._closed = False
        self._generators_prepared = False

    def configure_model(self, iterator: Any) -> None:
        import torch
        from miles.backends.megatron_utils.megatron_to_hf.processors import (
            quantizer_fp8,
        )
        from miles.backends.megatron_utils.named_weights import (
            named_params_and_buffers,
        )
        from miles.backends.megatron_utils.parallel import (
            get_expert_data_parallel_rank_and_size,
        )
        from miles.backends.megatron_utils.sglang import per_block_cast_to_fp8
        from miles.backends.training_utils.parallel import get_parallel_state
        from miles.backends.training_utils.weight_update.protocols.nccl_m2n_manifest import (
            _fp8_manifest_quantization,
            _local_source_spec,
        )

        self._iterator = iterator
        self._model = iterator.model
        global_named = list(named_params_and_buffers(self.args, self._model))
        local_names = (
            [
                name
                for name, _ in named_params_and_buffers(
                    self.args,
                    self._model,
                    convert_to_global_name=False,
                )
            ]
            if self.args.megatron_to_hf_mode == "bridge"
            else [name for name, _ in global_named]
        )
        source_keys = {
            global_name: local_name
            for (global_name, _), local_name in zip(
                global_named,
                local_names,
                strict=True,
            )
        }
        self._inventory = inventory_miles_native_tensors(
            global_named,
            source_keys=source_keys,
        )
        quantization = _fp8_manifest_quantization(iterator.quantization_config)
        if quantization is not None:
            scale_format = quantizer_fp8._get_scale_format(
                self.args,
                quantization["weight_block_size"],
                is_expert=True,
            )
            quantization["scale_format"] = (
                "ue8m0_unpacked" if scale_format == "ue8m0" else "canonical"
            )
            if (
                quantization["scale_format"] == "ue8m0_unpacked"
                and per_block_cast_to_fp8 is None
            ):
                raise RuntimeError(
                    "ModelExpress MILES FP8 requires the selected MILES quantizer"
                )
        self._quantization = quantization

        ps = get_parallel_state()
        expert_dp_rank, expert_dp_size = get_expert_data_parallel_rank_and_size()
        specs = []
        update_units = []
        for name, tensor in global_named:
            update_units.append([name])
            spec = _local_source_spec(name, tensor)
            if spec is not None:
                specs.append(spec)
        self._local_payload = {
            "topology": {
                "world_rank": torch.distributed.get_rank(),
                "tp_rank": ps.tp.rank,
                "tp_size": ps.tp.size,
                "pp_rank": ps.pp.rank,
                "pp_size": ps.pp.size,
                "cp_rank": ps.cp.rank,
                "cp_size": ps.cp.size,
                "dense_dp_rank": ps.intra_dp.rank,
                "dense_dp_size": ps.intra_dp.size,
                "ep_rank": ps.ep.rank,
                "ep_size": ps.ep.size,
                "etp_rank": ps.etp.rank,
                "etp_size": ps.etp.size,
                "expert_dp_rank": expert_dp_rank,
                "expert_dp_size": expert_dp_size,
                "independent_dp_rank": ps.indep_dp.rank,
                "independent_dp_size": ps.indep_dp.size,
            },
            "specs": specs,
            "update_units": update_units,
        }

    def _negotiate_projection(
        self,
        engine_gpu_counts: Sequence[int],
    ) -> MilesPr3304Projection:
        import torch.distributed as dist
        from miles.backends.training_utils.weight_update.protocols.nccl_m2n_manifest import (
            _build_manifest,
        )

        if self._local_payload is None:
            raise RuntimeError("configure_model must run before connect")
        gathered: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(
            gathered,
            {
                "payload": self._local_payload,
                "quantization": self._quantization,
            },
            group=_gloo_group(),
        )
        result: list[tuple[dict[str, Any] | None, str]] = [(None, "")]
        if dist.get_rank() == 0:
            try:
                records = [item for item in gathered if item is not None]
                if any(item["quantization"] != self._quantization for item in records):
                    raise ValueError(
                        "trainer ranks selected different MILES quantization formats"
                    )
                payloads = [item["payload"] for item in records]
                manifest = _build_manifest(
                    payloads,
                    engine_gpu_counts,
                    quantization_config=self._quantization,
                    destination_ep_size=self.args.sglang_ep_size,
                )
                topologies = [
                    MilesTrainerTopology(**payload["topology"]) for payload in payloads
                ]
                update_units = {
                    tuple(unit)
                    for payload in payloads
                    for unit in payload["update_units"]
                }
                projection = project_pr3304_manifest(
                    manifest,
                    topologies,
                    update_units=update_units,
                    pp_wave_size=int(_arg(self.args, "m2n_pp_concurrency", 2)),
                )
                result[0] = (
                    {
                        "manifest": projection.manifest,
                        "topologies": [record.__dict__ for record in topologies],
                        "update_units": [list(unit) for unit in update_units],
                    },
                    "",
                )
            except BaseException as error:
                result[0] = (None, repr(error))
        dist.broadcast_object_list(result, src=0, group=_gloo_group())
        payload, error = result[0]
        if error:
            raise RuntimeError(f"PR 3304 manifest negotiation failed: {error}")
        if payload is None:
            raise RuntimeError("PR 3304 manifest negotiation returned no payload")
        return project_pr3304_manifest(
            payload["manifest"],
            [MilesTrainerTopology(**item) for item in payload["topologies"]],
            update_units=payload["update_units"],
            pp_wave_size=int(_arg(self.args, "m2n_pp_concurrency", 2)),
        )

    def _agree_run_id(self) -> str:
        import torch.distributed as dist

        requested = _arg(
            self.args,
            "modelexpress_m2n_run_id",
            os.environ.get("MX_MILES_RUN_ID"),
        )
        candidates = [None] * dist.get_world_size()
        dist.all_gather_object(candidates, requested, group=_gloo_group())
        configured = {str(item) for item in candidates if item is not None}
        if configured:
            if len(configured) != 1 or any(item is None for item in candidates):
                raise ValueError(
                    "modelexpress_m2n_run_id must be set identically on every rank"
                )
            return _text(next(iter(configured)), "modelexpress_m2n_run_id")
        generated = [None] * dist.get_world_size()
        dist.all_gather_object(generated, uuid4().hex, group=_gloo_group())
        return _text(generated[0], "modelexpress_m2n_run_id")

    def _validate_selector(self, selector: str) -> None:
        if self.args.sglang_speculative_algorithm and selector != "target":
            raise ValueError(
                "NCCL M2N supports speculation only with a frozen draft "
                "and target-only weight updates; disable trainer MTP layers."
            )

    def connect_modelexpress(
        self,
        rollout_engines: Sequence[Any],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: Any,
    ) -> None:
        import torch
        import torch.distributed as dist
        from miles.backends.training_utils.weight_update.protocols.nccl_m2n import (
            _quantize_block_fp8,
        )

        if engine_gpu_counts is None or engine_gpu_offsets is None:
            raise ValueError(
                "ModelExpress MILES requires explicit engine GPU counts and offsets"
            )
        counts = tuple(int(item) for item in engine_gpu_counts)
        offsets = tuple(int(item) for item in engine_gpu_offsets)
        if len(counts) != len(rollout_engines) or len(offsets) != len(counts):
            raise ValueError("engine GPU topology must cover every rollout engine")
        self._engine_gpu_counts = counts
        self._engine_gpu_offsets = offsets
        self._projection = self._negotiate_projection(counts)
        run_id = self._agree_run_id()
        trainer_slots_by_world = {
            world_rank: f"{run_id}:trainer-{world_rank}"
            for lane in self._projection.topology_plan.trainer_lanes
            for world_rank in lane
        }
        generator_slots = tuple(
            f"{run_id}:generator-{offset + local_rank}"
            for offset, count in zip(offsets, counts, strict=True)
            for local_rank in range(count)
        )
        self._topology = CollectiveTopology.from_miles_reshard_plan(
            model_name=str(_arg(self.args, "model", self._iterator.model_name)),
            miles_plan=self._projection.topology_plan,
            trainer_slots_by_world_rank=trainer_slots_by_world,
            generator_slots=generator_slots,
            m2n_abi_version=str(
                _arg(
                    self.args,
                    "modelexpress_m2n_abi_version",
                    "miles-pr3304-v1",
                )
            ),
        )
        world_rank = dist.get_rank()
        source_partition = int(parallel_state.pp.rank)
        fp8_quantizer = None
        if self._quantization is not None:
            scale_format = self._quantization["scale_format"]

            def fp8_quantizer(tensor):
                return _quantize_block_fp8(tensor, scale_format=scale_format)

        self._publisher = MilesNativePublisher(
            topology=self._projection.topology_plan,
            collective_topology=self._topology,
            source_partition=source_partition,
            source_world_rank=world_rank,
            inventory=self._inventory,
            device=f"cuda:{torch.cuda.current_device()}",
            fp8_quantizer=fp8_quantizer,
            source_recipes={
                entry["name"]: _source_recipe(
                    entry["source"]["recipe"],
                    entry["name"],
                )
                for entry in self._projection.manifest["entries"]
            },
        )
        validate_native_publisher_manifest(
            self._publisher,
            self._projection.manifest,
            source_world_rank=world_rank,
        )
        endpoint = _endpoint(
            _arg(
                self.args,
                "modelexpress_server_address",
                os.environ.get("MX_SERVER_ADDRESS", "127.0.0.1:50051"),
            )
        )
        channel = auth.with_auth(grpc.insecure_channel(endpoint))
        rendezvous = CollectiveRendezvous(channel)
        slot_id = trainer_slots_by_world[world_rank]
        index_in_role = self._topology.trainer_slots.index(slot_id)
        session = MilesTrainerSession.create(
            rendezvous=rendezvous,
            topology=self._topology,
            publisher=self._publisher,
            source_partition=source_partition,
            slot_id=slot_id,
            worker_id=f"miles-{slot_id}-{uuid4().hex}",
            index_in_role=index_in_role,
            semantic_manifest_version=self._projection.semantic_manifest_version,
            semantic_manifest_digest=self._projection.semantic_manifest_digest,
            layer_groups=self._projection.topology_plan.layer_groups,
            device=f"cuda:{torch.cuda.current_device()}",
        )
        self._channel = channel
        self._rendezvous = rendezvous
        self._session = session
        self._coordinator = MilesTransferCoordinator(rendezvous, self._topology)
        self._generator_endpoint = endpoint

    def _prepare_modelexpress_sessions(self) -> None:
        import torch.distributed as dist

        if self._generators_prepared:
            return
        if self._session is None or self._generator_endpoint is None:
            raise RuntimeError("ModelExpress MILES connection is not ready")
        local_error = ""
        generator_futures = []
        try:
            generator_futures = self._generator_futures(
                "prepare",
                endpoint=self._generator_endpoint,
            )
        except BaseException as error:
            local_error = repr(error)
        failures = self._collect_errors(local_error)
        if failures:
            self.close_modelexpress()
            raise RuntimeError(
                "ModelExpress MILES generator preparation dispatch failed: "
                + "; ".join(failures[:4])
            )
        try:
            self._session.prepare()
        except BaseException as error:
            local_error = repr(error)
        if dist.get_rank() == 0 and not local_error:
            try:
                self._wait_futures(generator_futures)
            except BaseException as error:
                local_error = repr(error)
        failures = self._collect_errors(local_error)
        if failures:
            self.close_modelexpress()
            raise RuntimeError(
                "ModelExpress MILES session preparation failed: "
                + "; ".join(failures[:4])
            )
        self._generators_prepared = True

    async def _send_control(self, client: Any, control: CollectiveControl):
        response = await client.update_weights_from_distributed(
            names=[],
            dtypes=[],
            shapes=[],
            group_name=encode_control(control),
            flush_cache=False,
            selector="target",
        )
        return _check_response(response)

    async def _send_controls(
        self,
        requests: Sequence[tuple[Any, CollectiveControl]],
    ):
        return await asyncio.gather(
            *(self._send_control(client, control) for client, control in requests)
        )

    def _generator_futures(
        self,
        action: str,
        *,
        endpoint: str | None = None,
        **kwargs,
    ):
        import torch.distributed as dist
        from miles.utils import async_utils

        if dist.get_rank() != 0:
            return []
        requests = []
        offset = 0
        for client, count in zip(
            self.rollout_engines,
            self._engine_gpu_counts,
            strict=True,
        ):
            requests.append(
                (
                    client,
                    CollectiveControl(
                        action=action,
                        plan=(
                            self._projection.topology_plan.plan
                            if action == "prepare"
                            else None
                        ),
                        topology=self._topology if action == "prepare" else None,
                        generator_slot_offset=offset if action == "prepare" else None,
                        endpoint=endpoint if action == "prepare" else None,
                        semantic_manifest_version=(
                            self._projection.semantic_manifest_version
                            if action == "prepare"
                            else None
                        ),
                        semantic_manifest_digest=(
                            self._projection.semantic_manifest_digest
                            if action == "prepare"
                            else None
                        ),
                        destination_manifest=(
                            self._projection.destination_manifest
                            if action == "prepare"
                            else None
                        ),
                        **kwargs,
                    ),
                )
            )
            offset += count
        if not requests:
            return []
        coroutine = self._send_controls(requests)
        try:
            return [async_utils.submit(coroutine)]
        except BaseException:
            coroutine.close()
            raise

    @staticmethod
    def _wait_futures(futures) -> None:
        if futures:
            from miles.utils import async_utils

            async_utils.wait_futures(futures)

    @staticmethod
    def _collect_errors(error: str) -> list[str]:
        import torch.distributed as dist

        errors = [""] * dist.get_world_size()
        dist.all_gather_object(errors, error, group=_gloo_group())
        return [item for item in errors if item]

    def begin_sync(
        self,
        weight_version: int,
        _iter_buckets: Callable[..., Any],
    ) -> bool:
        if self._pending_version is not None:
            raise RuntimeError("a ModelExpress MILES round is already pending")
        self._deferred_update_error = None
        self._pending_version = str(weight_version)
        return True

    def _run_engine_session_collectively(
        self,
        operation: Callable[[], None],
        driver_runner: Callable[[Callable[[], None]], None],
    ) -> None:
        local_error = ""
        try:
            driver_runner(operation)
        except BaseException as error:
            local_error = f"rollout session: {type(error).__name__}: {error}"
        failures = self._collect_errors(local_error)
        if failures:
            raise RuntimeError(
                "ModelExpress MILES rollout session failed: " + "; ".join(failures[:4])
            )

    def _send_residual_bucket(
        self,
        bucket: list[tuple[str, Any]],
        sender: Callable[[list[tuple[str, Any]]], None],
    ) -> None:
        if self._deferred_update_error is not None:
            bucket.clear()
            return
        try:
            sender(bucket)
        except BaseException as error:
            try:
                import torch.distributed as dist

                rank = dist.get_rank()
            except BaseException:
                rank = "unknown"
            self._deferred_update_error = (
                f"trainer rank {rank} residual update: {type(error).__name__}: {error}"
            )
            bucket.clear()

    def _finish_residual_stream(self) -> None:
        failures = self._collect_errors(self._deferred_update_error or "")
        if failures:
            raise RuntimeError(
                "ModelExpress MILES residual update failed: " + "; ".join(failures[:4])
            )

    def before_base_weights(self, weights: Mapping[str, Any]) -> None:
        import torch.distributed as dist

        if (
            self._publisher is None
            or self._session is None
            or self._coordinator is None
            or self._pending_version is None
        ):
            raise RuntimeError("ModelExpress MILES protocol is not ready")
        try:
            self._prepare_modelexpress_sessions()
        except BaseException:
            self._pending_version = None
            raise
        refresh_error = ""
        try:
            self._publisher.refresh(weights)
        except BaseException as error:
            refresh_error = repr(error)
        failures = self._collect_errors(refresh_error)
        if failures:
            self._pending_version = None
            raise RuntimeError(
                "ModelExpress MILES source refresh failed: " + "; ".join(failures[:4])
            )

        version = self._pending_version
        operation: list[tuple[str | None, str]] = [(None, "")]
        if dist.get_rank() == 0:
            try:
                group_id = self._session.membership.group_id
                transfer = self._coordinator.create(
                    version,
                    idempotency_key=f"miles-{group_id}-weight-version-{version}",
                )
                operation[0] = (str(transfer.operation_id), "")
            except BaseException as error:
                operation[0] = (None, repr(error))
        dist.broadcast_object_list(operation, src=0, group=_gloo_group())
        operation_id, create_error = operation[0]
        if create_error or operation_id is None:
            self._pending_version = None
            raise RuntimeError(
                "ModelExpress MILES operation creation failed: "
                + (create_error or "missing operation ID")
            )

        generator_futures = []
        submission_error = ""
        if dist.get_rank() == 0:
            try:
                generator_futures = self._generator_futures(
                    "run_round",
                    version=version,
                    operation_id=operation_id,
                )
            except BaseException as error:
                submission_error = repr(error)
        failures = self._collect_errors(submission_error)
        if failures:
            cleanup_errors = []
            if dist.get_rank() == 0:
                try:
                    self._session.report_failure(
                        operation_id=operation_id,
                        error=RuntimeError("; ".join(failures[:4])),
                    )
                except BaseException as error:
                    cleanup_errors.append(f"terminal failure report: {error!r}")
                try:
                    self._wait_futures(generator_futures)
                except BaseException as error:
                    cleanup_errors.append(f"receiver future settlement: {error!r}")
                terminal = None
                try:
                    terminal = self._wait_terminal(operation_id)
                    self._require_complete(operation_id, terminal)
                except BaseException as error:
                    cleanup_errors.append(f"terminal wait: {error!r}")
                if terminal is not None:
                    try:
                        self._coordinator.delete(operation_id)
                    except BaseException as error:
                        cleanup_errors.append(f"terminal delete: {error!r}")
            cleanup_status = ["; ".join(cleanup_errors)]
            dist.broadcast_object_list(
                cleanup_status,
                src=0,
                group=_gloo_group(),
            )
            self._pending_version = None
            failures.extend(item for item in cleanup_status if item)
            raise RuntimeError(
                "ModelExpress MILES receiver launch failed: " + "; ".join(failures[:4])
            )

        local_error = ""
        try:
            self._session.run_round(version=version, operation_id=operation_id)
        except BaseException as error:
            local_error = repr(error)
        trainer_failures = self._collect_errors(local_error)
        coordinator_error = ""
        if dist.get_rank() == 0:
            terminal = None
            try:
                self._wait_futures(generator_futures)
            except BaseException as error:
                coordinator_error = repr(error)
            try:
                terminal = self._wait_terminal(operation_id)
                self._require_complete(operation_id, terminal)
            except BaseException as error:
                terminal_error = repr(error)
                coordinator_error = (
                    f"{coordinator_error}; {terminal_error}"
                    if coordinator_error
                    else terminal_error
                )
            finally:
                if terminal is not None:
                    try:
                        self._coordinator.delete(operation_id)
                    except BaseException as error:
                        if not coordinator_error:
                            coordinator_error = repr(error)
        status = [coordinator_error]
        dist.broadcast_object_list(status, src=0, group=_gloo_group())
        self._pending_version = None
        failures = [*trainer_failures, *[item for item in status if item]]
        if failures:
            raise RuntimeError(
                "ModelExpress MILES routed update failed: " + "; ".join(failures[:4])
            )

    @staticmethod
    def _require_complete(operation_id: str, transfer: Any) -> None:
        if transfer.state != collective_pb.COLLECTIVE_TRANSFER_STATE_COMPLETE:
            raise RuntimeError(
                transfer.failure_message
                or f"collective operation {operation_id} ended in state "
                f"{transfer.state}"
            )

    def _wait_terminal(self, operation_id: str) -> Any:
        timeout_s = envs.MX_NCCL_REFIT_TRANSFER_TIMEOUT_S
        deadline = time.monotonic() + timeout_s
        while True:
            transfer = self._coordinator.get(operation_id)
            if transfer.state in _TERMINAL_STATES:
                return transfer
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"collective operation {operation_id} did not finish "
                    f"within {timeout_s:.0f}s"
                )
            time.sleep(0.1)

    def close_modelexpress(self) -> None:
        if self._closed:
            return
        first_error = None
        session = self._session
        try:
            futures = self._generator_futures("close")
            self._wait_futures(futures)
        except BaseException as error:
            first_error = error
        try:
            if self._session is not None:
                self._session.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
        finally:
            self._session = None
            self._coordinator = None
            if self._rendezvous is not None and session is None:
                try:
                    self._rendezvous.close()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            self._rendezvous = None
            if self._channel is not None:
                try:
                    self._channel.close()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            self._channel = None
            self._generator_endpoint = None
            self._closed = True
        if first_error is not None:
            raise first_error


def build_protocol(args: Any):
    """Build a normal PR 3304 protocol with ModelExpress as its routed phase."""

    from miles.backends.training_utils.weight_update.hf_weight_iterator import (
        WeightUpdatePlacement,
    )
    from miles.backends.training_utils.weight_update.protocols.broadcast import (
        UpdateWeightFromDistributed,
        disconnect_rollout_engines_from_distributed,
    )

    class MilesModelExpressProtocol(
        MilesPr3304ProtocolCore,
        UpdateWeightFromDistributed,
    ):
        required_placement = WeightUpdatePlacement(gather_pp=False)

        def __init__(self, protocol_args: Any) -> None:
            UpdateWeightFromDistributed.__init__(self, protocol_args)
            MilesPr3304ProtocolCore.__init__(self, protocol_args)

        def configure_model(self, iterator: Any) -> None:
            MilesPr3304ProtocolCore.configure_model(self, iterator)

        def connect(
            self,
            rollout_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
            parallel_state,
            placement,
            selector,
        ) -> None:
            self._validate_selector(selector)
            self.rollout_engines = rollout_engines
            try:
                self.connect_modelexpress(
                    rollout_engines,
                    engine_gpu_counts,
                    engine_gpu_offsets,
                    parallel_state,
                )
                projection = self._projection
                routed_names = {
                    name
                    for unit in projection.manifest["routed_update_units"]
                    for name in unit
                }
                if hasattr(self._iterator, "excluded_native_names"):
                    self._iterator.excluded_native_names = set(routed_names)
                self._iterator.excluded_hf_names = _pr3304_hf_exclusions(
                    projection.manifest
                )
                UpdateWeightFromDistributed.connect(
                    self,
                    rollout_engines,
                    engine_gpu_counts,
                    engine_gpu_offsets,
                    parallel_state,
                    placement,
                    selector,
                )
            except BaseException:
                try:
                    self.close_modelexpress()
                finally:
                    if hasattr(self._iterator, "excluded_native_names"):
                        self._iterator.excluded_native_names = set()
                    self._iterator.excluded_hf_names = set()
                raise

        def begin_sync(self, weight_version, iter_buckets) -> bool:
            return MilesPr3304ProtocolCore.begin_sync(
                self,
                weight_version,
                iter_buckets,
            )

        def before_base_weights(self, weights) -> None:
            MilesPr3304ProtocolCore.before_base_weights(self, weights)

        def run_engine_session(self, operation) -> None:
            from miles.backends.training_utils.weight_update.protocol import (
                WeightTransferProtocol,
            )

            self._run_engine_session_collectively(
                operation,
                lambda callback: WeightTransferProtocol.run_engine_session(
                    self,
                    callback,
                ),
            )

        def send_bucket(self, bucket) -> None:
            self._send_residual_bucket(
                bucket,
                lambda pending: UpdateWeightFromDistributed.send_bucket(
                    self,
                    pending,
                ),
            )

        def after_base_weights(self) -> None:
            self._finish_residual_stream()

        def close(self) -> None:
            first_error = None
            try:
                self.close_modelexpress()
            except BaseException as error:
                first_error = error
            if self.is_sender and self._model_update_groups is not None:
                try:
                    disconnect_rollout_engines_from_distributed(
                        self.args,
                        self.group_name,
                        self._model_update_groups,
                        self.rollout_engines,
                    )
                    self._model_update_groups = None
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            if first_error is not None:
                raise first_error

    return MilesModelExpressProtocol(args)


__all__ = [
    "MilesPr3304Projection",
    "SEMANTIC_MANIFEST_VERSION",
    "build_protocol",
    "canonical_pr3304_manifest_digest",
    "project_pr3304_manifest",
    "validate_native_publisher_manifest",
]
