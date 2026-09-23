# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Megatron source bindings for MILES collective refit."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from ..spi import LocalParamSpec
from ..types import ReshardPlan
from ._common import _check_stable, _local_shape, _tensor_signature, _text
from .miles_topology import MilesReshardTopologyPlan, MilesSourceRoute

_DENSE_RE = re.compile(
    r"^module\.module\.decoder\.layers\.(\d+)\.mlp\.linear_fc([12])\.weight$"
)
_EXPERT_RE = re.compile(
    r"^module\.module\.decoder\.layers\.(\d+)\.mlp\.experts\."
    r"linear_fc([12])\.weight(\d+)$"
)
_CANONICAL_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")
_FP8_BLOCK_SIZE = (128, 128)

MilesTensorFamily = Literal["dense", "routed_expert"]
MilesTensorRole = Literal["weight", "scale"]


class MilesSourceRecipe(str, Enum):
    """Source-only logical tensor recipes from MILES PR 3304."""

    DENSE_FC1_GATE = "dense_fc1_gate"
    DENSE_FC1_UP = "dense_fc1_up"
    DENSE_FC2 = "dense_fc2"
    EXPERT_FC1_GATE = "expert_fc1_gate"
    EXPERT_FC1_UP = "expert_fc1_up"
    EXPERT_FC2 = "expert_fc2"


@dataclass(frozen=True)
class MilesNativeTensorRecord:
    """Metadata-only description of one local Megatron tensor."""

    native_name: str
    source_key: str
    family: MilesTensorFamily
    layer: int
    projection: Literal["fc1", "fc2"]
    expert_id: int | None
    shape: tuple[int, ...]
    dtype: str
    partition_dim: int
    partition_stride: int

    def __post_init__(self) -> None:
        if not self.native_name or not self.source_key:
            raise ValueError("native_name and source_key must not be empty")
        if self.layer < 0:
            raise ValueError("MILES layer must not be negative")
        if not self.shape or any(dim <= 0 for dim in self.shape):
            raise ValueError(f"{self.native_name}: shape must be positive")
        if not self.dtype:
            raise ValueError(f"{self.native_name}: dtype must not be empty")
        if self.family not in ("dense", "routed_expert"):
            raise ValueError(f"{self.native_name}: unsupported family {self.family!r}")
        if self.projection not in ("fc1", "fc2"):
            raise ValueError(
                f"{self.native_name}: unsupported projection {self.projection!r}"
            )
        if self.family == "dense" and self.expert_id is not None:
            raise ValueError(f"{self.native_name}: dense tensors cannot name an expert")
        if self.family == "routed_expert" and (
            self.expert_id is None or self.expert_id < 0
        ):
            raise ValueError(f"{self.native_name}: expert tensors must name an expert")


@dataclass(frozen=True)
class MilesSourceBinding:
    """Frozen local recipe for one canonical collective entry."""

    canonical_name: str
    source_names: tuple[str, ...]
    source_keys: tuple[str, ...]
    recipe: MilesSourceRecipe
    expected_shape: tuple[int, ...]
    dtype: str
    pair_id: str | None = None
    tensor_role: MilesTensorRole | None = None


def _dtype_label(value: object) -> str:
    return str(value).removeprefix("torch.")


def _normalize_partition_metadata(
    native_name: str,
    projection: Literal["fc1", "fc2"],
    partition_dim: int,
    partition_stride: int,
) -> tuple[int, int]:
    if projection == "fc1":
        if partition_stride not in (1, 2):
            raise ValueError(
                f"{native_name}: FC1 partition_stride must be 1 or 2, "
                f"got {partition_stride}"
            )
        if partition_dim < 0:
            partition_dim = 0
        elif partition_dim != 0:
            raise ValueError(
                f"{native_name}: FC1 partition_dim must be 0 or a negative "
                f"sentinel, got {partition_dim}"
            )
        return partition_dim, 2

    if partition_stride != 1:
        raise ValueError(
            f"{native_name}: FC2 partition_stride must be 1, got {partition_stride}"
        )
    if partition_dim <= 0:
        partition_dim = 1
    elif partition_dim != 1:
        raise ValueError(
            f"{native_name}: FC2 partition_dim must be 1 or a non-positive "
            f"sentinel, got {partition_dim}"
        )
    return partition_dim, partition_stride


def inventory_miles_native_tensors(
    named_tensors: Iterable[tuple[str, Any]],
    *,
    source_keys: Mapping[str, str] | None = None,
) -> tuple[MilesNativeTensorRecord, ...]:
    """Inventory PR 3304 FFN tensors without touching their storage."""

    records: list[MilesNativeTensorRecord] = []
    seen_names: set[str] = set()
    seen_keys: set[str] = set()
    key_map = {} if source_keys is None else dict(source_keys)
    for native_name, tensor in named_tensors:
        dense_match = _DENSE_RE.fullmatch(native_name)
        expert_match = _EXPERT_RE.fullmatch(native_name)
        if dense_match is not None:
            layer_text, projection_text = dense_match.groups()
            family: MilesTensorFamily = "dense"
            expert_id = None
        elif expert_match is not None:
            layer_text, projection_text, expert_text = expert_match.groups()
            family = "routed_expert"
            expert_id = int(expert_text)
        else:
            continue

        source_key = str(key_map.get(native_name, native_name))
        if native_name in seen_names:
            raise ValueError(f"duplicate MILES native tensor {native_name}")
        if source_key in seen_keys:
            raise ValueError(f"duplicate MILES source key {source_key}")
        seen_names.add(native_name)
        seen_keys.add(source_key)

        projection = f"fc{projection_text}"
        partition_dim = int(getattr(tensor, "partition_dim", -1))
        partition_stride = int(getattr(tensor, "partition_stride", 1))
        partition_dim, partition_stride = _normalize_partition_metadata(
            native_name,
            projection,
            partition_dim,
            partition_stride,
        )
        records.append(
            MilesNativeTensorRecord(
                native_name=native_name,
                source_key=source_key,
                family=family,
                layer=int(layer_text),
                projection=projection,
                expert_id=expert_id,
                shape=tuple(int(dim) for dim in tensor.shape),
                dtype=_dtype_label(tensor.dtype),
                partition_dim=partition_dim,
                partition_stride=partition_stride,
            )
        )
    return tuple(sorted(records, key=lambda record: record.native_name))


def _recipe(
    canonical_name: str,
    family: MilesTensorFamily,
) -> MilesSourceRecipe:
    base_name = canonical_name.removesuffix("_scale_inv")
    if base_name.endswith("gate_proj.weight"):
        component = "gate"
    elif base_name.endswith("up_proj.weight"):
        component = "up"
    elif base_name.endswith("down_proj.weight"):
        component = "down"
    else:
        raise ValueError(
            f"{canonical_name}: unsupported MILES canonical source projection"
        )
    if family == "dense":
        return {
            "gate": MilesSourceRecipe.DENSE_FC1_GATE,
            "up": MilesSourceRecipe.DENSE_FC1_UP,
            "down": MilesSourceRecipe.DENSE_FC2,
        }[component]
    return {
        "gate": MilesSourceRecipe.EXPERT_FC1_GATE,
        "up": MilesSourceRecipe.EXPERT_FC1_UP,
        "down": MilesSourceRecipe.EXPERT_FC2,
    }[component]


def _pair_metadata(
    canonical_name: str,
    dtype: str,
    family: MilesTensorFamily,
) -> tuple[str | None, MilesTensorRole | None]:
    dtype = _dtype_label(dtype)
    if dtype == "bfloat16":
        return None, None
    if family != "routed_expert":
        raise ValueError(
            f"{canonical_name}: quantized MILES routes must be routed experts"
        )
    if dtype == "float8_e4m3fn" and canonical_name.endswith(".weight"):
        return canonical_name, "weight"
    if dtype == "float32" and canonical_name.endswith(".weight_scale_inv"):
        return canonical_name.removesuffix("_scale_inv"), "scale"
    raise ValueError(f"{canonical_name}: unsupported MILES source dtype/role {dtype!r}")


def _record_projection(recipe: MilesSourceRecipe) -> Literal["fc1", "fc2"]:
    if recipe in (
        MilesSourceRecipe.DENSE_FC1_GATE,
        MilesSourceRecipe.DENSE_FC1_UP,
        MilesSourceRecipe.EXPERT_FC1_GATE,
        MilesSourceRecipe.EXPERT_FC1_UP,
    ):
        return "fc1"
    return "fc2"


def _canonical_layer(canonical_name: str) -> int:
    match = _CANONICAL_LAYER_RE.match(canonical_name)
    if match is None:
        raise ValueError(f"{canonical_name}: unsupported canonical MILES layer name")
    return int(match.group(1))


def _expected_native_shape(
    binding: MilesSourceBinding,
    source_count: int,
) -> tuple[int, ...]:
    output = binding.expected_shape
    if binding.recipe in (
        MilesSourceRecipe.DENSE_FC1_GATE,
        MilesSourceRecipe.DENSE_FC1_UP,
    ):
        if len(output) != 2 or source_count != 1:
            raise ValueError(
                f"{binding.canonical_name}: dense FC1 requires one 2-D source"
            )
        return (output[0] * 2, output[1])
    if binding.recipe is MilesSourceRecipe.DENSE_FC2:
        if len(output) != 2 or source_count != 1:
            raise ValueError(
                f"{binding.canonical_name}: dense FC2 requires one 2-D source"
            )
        return output
    if len(output) != 3 or output[0] != source_count:
        raise ValueError(
            f"{binding.canonical_name}: expert output {output} requires "
            f"{output[0] if output else 0} ordered sources, got {source_count}"
        )
    if binding.recipe in (
        MilesSourceRecipe.EXPERT_FC1_GATE,
        MilesSourceRecipe.EXPERT_FC1_UP,
    ):
        return (output[1] * 2, output[2])
    return (output[1], output[2])


def _component(pair_id: str) -> tuple[str, str]:
    for component in ("gate", "up", "down"):
        suffix = f".{component}_proj.weight"
        if pair_id.endswith(suffix):
            return pair_id.removesuffix(suffix), component
    raise ValueError(f"invalid MILES FP8 pair id {pair_id!r}")


def _fp8_scale_shape(shape: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(int(dim) for dim in shape)
    if len(result) < 2 or any(
        dim % block for dim, block in zip(result[-2:], _FP8_BLOCK_SIZE, strict=True)
    ):
        raise ValueError(
            f"{name}: FP8 shape {result} is not aligned to {_FP8_BLOCK_SIZE}"
        )
    return (
        *result[:-2],
        result[-2] // _FP8_BLOCK_SIZE[0],
        result[-1] // _FP8_BLOCK_SIZE[1],
    )


def _prepare_logical_source(
    tensors: Sequence[Any],
    recipe: MilesSourceRecipe,
) -> Any:
    if recipe in (
        MilesSourceRecipe.DENSE_FC1_GATE,
        MilesSourceRecipe.DENSE_FC1_UP,
    ):
        index = 0 if recipe is MilesSourceRecipe.DENSE_FC1_GATE else 1
        return tensors[0].chunk(2, dim=0)[index].contiguous()
    if recipe is MilesSourceRecipe.DENSE_FC2:
        return tensors[0].contiguous()

    import torch  # noqa: PLC0415

    if recipe in (
        MilesSourceRecipe.EXPERT_FC1_GATE,
        MilesSourceRecipe.EXPERT_FC1_UP,
    ):
        index = 0 if recipe is MilesSourceRecipe.EXPERT_FC1_GATE else 1
        return torch.stack(
            [tensor.chunk(2, dim=0)[index] for tensor in tensors]
        ).contiguous()
    if recipe is MilesSourceRecipe.EXPERT_FC2:
        return torch.stack(list(tensors)).contiguous()
    raise AssertionError(f"unhandled MILES source recipe {recipe}")


class MilesNativePublisher:
    """Bind current native MILES shards to stable collective source specs."""

    def __init__(
        self,
        *,
        topology: MilesReshardTopologyPlan,
        collective_topology: Any,
        source_partition: int,
        source_world_rank: int,
        inventory: Sequence[MilesNativeTensorRecord],
        device: str,
        fp8_quantizer: Callable[[Any], tuple[Any, Any]] | None = None,
        source_recipes: Mapping[str, MilesSourceRecipe] | None = None,
    ) -> None:
        self._topology = copy.deepcopy(topology)
        self._plan = copy.deepcopy(topology.plan)
        self._plan.validate()
        if self._plan.misc:
            raise ValueError("native MILES publisher supports bulk routes only")
        if not 0 <= source_partition < self._plan.source_partition_count:
            raise ValueError(
                f"source_partition must be in [0, "
                f"{self._plan.source_partition_count}), got {source_partition}"
            )
        self._source_partition = source_partition
        if isinstance(source_world_rank, bool) or not isinstance(
            source_world_rank, int
        ):
            raise TypeError("source_world_rank must be an integer")
        self._source_world_rank = source_world_rank
        try:
            lane = self._topology.trainer_lanes[source_partition]
        except IndexError as exc:
            raise ValueError(
                f"MILES topology has no trainer lane for partition {source_partition}"
            ) from exc
        if self._source_world_rank not in lane:
            raise ValueError(
                f"source world rank {self._source_world_rank} is not in trainer "
                f"lane {source_partition}: {lane}"
            )
        self.validate_topology(collective_topology)
        device = _text(device, "device")
        kind, separator, index = device.partition(":")
        if kind != "cuda" or not separator or not index.isdecimal():
            raise ValueError(f"device must name an indexed CUDA device, got {device!r}")
        self._device = device
        self._fp8_quantizer = fp8_quantizer

        entries = {entry.name: entry for entry in self._plan.bulk}
        if len(entries) != len(self._plan.bulk):
            raise ValueError("MILES plan must not contain duplicate parameters")
        routes = {route.canonical_name: route for route in self._topology.routes}
        if set(routes) != set(entries):
            raise ValueError(
                "MILES routes must exactly cover bulk plan entries "
                f"(missing={sorted(set(entries) - set(routes))[:5]}, "
                f"unknown={sorted(set(routes) - set(entries))[:5]})"
            )
        if source_recipes is None:
            recipe_by_name = {
                name: _recipe(name, route.family) for name, route in routes.items()
            }
        else:
            recipe_by_name = dict(source_recipes)
            if set(recipe_by_name) != set(entries):
                raise ValueError(
                    "MILES source recipes must exactly cover bulk plan entries "
                    f"(missing={sorted(set(entries) - set(recipe_by_name))[:5]}, "
                    f"unknown={sorted(set(recipe_by_name) - set(entries))[:5]})"
                )
            if any(
                not isinstance(recipe, MilesSourceRecipe)
                for recipe in recipe_by_name.values()
            ):
                raise ValueError(
                    "MILES source recipes must be MilesSourceRecipe values"
                )
        for name, route in routes.items():
            entry = entries[name]
            if route.partition_id != entry.partition_id:
                raise ValueError(f"{name}: route partition differs from the plan")
            mapped_owners = tuple(
                world_rank for world_rank, _ in route.source_names_by_world
            )
            if mapped_owners != route.source_world_ranks:
                raise ValueError(f"{name}: route source ownership mapping drifted")
            source_ranks = entry.src_mesh.ranks()
            lane = self._topology.trainer_lanes[route.partition_id]
            if any(rank >= len(lane) for rank in source_ranks):
                raise ValueError(f"{name}: source mesh exceeds its trainer lane")
            expected_owners = tuple(lane[rank] for rank in source_ranks)
            if route.source_world_ranks != expected_owners:
                raise ValueError(
                    f"{name}: route source ownership does not match source mesh; "
                    f"expected {expected_owners}, got {route.source_world_ranks}"
                )

        record_by_name = {record.native_name: record for record in inventory}
        if len(record_by_name) != len(inventory):
            raise ValueError("MILES inventory contains duplicate native names")
        self._inventory = tuple(inventory)
        self._bindings: list[MilesSourceBinding] = []
        for entry in self._plan.bulk:
            route = routes[entry.name]
            if (
                entry.partition_id != source_partition
                or self._source_world_rank not in route.source_world_ranks
            ):
                continue
            source_names = dict(route.source_names_by_world)[self._source_world_rank]
            try:
                records = [record_by_name[name] for name in source_names]
            except KeyError as exc:
                raise ValueError(
                    f"{entry.name}: owned native source {exc.args[0]!r} "
                    "is missing from inventory"
                ) from exc
            if not records:
                raise ValueError(f"{entry.name}: owned route has no native sources")
            recipe = recipe_by_name[entry.name]
            projection = _record_projection(recipe)
            layer = _canonical_layer(entry.name)
            if any(
                record.family != route.family
                or record.projection != projection
                or record.layer != layer
                for record in records
            ):
                raise ValueError(
                    f"{entry.name}: native source family/projection/layer drifted"
                )
            if route.family == "routed_expert":
                records.sort(key=lambda record: int(record.expert_id))
                expert_ids = [int(record.expert_id) for record in records]
                if expert_ids != list(
                    range(expert_ids[0], expert_ids[0] + len(expert_ids))
                ):
                    raise ValueError(
                        f"{entry.name}: local expert ids must be contiguous, "
                        f"got {expert_ids}"
                    )
            pair_id, tensor_role = _pair_metadata(
                entry.name,
                entry.dtype,
                route.family,
            )
            binding = MilesSourceBinding(
                canonical_name=entry.name,
                source_names=tuple(record.native_name for record in records),
                source_keys=tuple(record.source_key for record in records),
                recipe=recipe,
                expected_shape=_local_shape(
                    entry.global_shape,
                    entry.src_mesh,
                    entry.src_placements,
                ),
                dtype=_dtype_label(entry.dtype),
                pair_id=pair_id,
                tensor_role=tensor_role,
            )
            expected_native = (
                None
                if tensor_role == "scale"
                else _expected_native_shape(binding, len(records))
            )
            for record in records:
                if expected_native is not None and record.shape != expected_native:
                    raise ValueError(
                        f"{entry.name}: native source {record.native_name} shape "
                        f"{record.shape} does not match {expected_native}"
                    )
                if record.dtype != "bfloat16":
                    raise ValueError(
                        f"{entry.name}: native source {record.native_name} must "
                        f"be bfloat16, got {record.dtype}"
                    )
                if record.partition_dim != (
                    0 if projection == "fc1" else 1
                ) or record.partition_stride != (2 if projection == "fc1" else 1):
                    family = "dense" if route.family == "dense" else "expert"
                    raise ValueError(
                        f"{entry.name}: {family} native partition metadata drifted"
                    )
            self._bindings.append(binding)

        self._fp8_pairs = self._validate_fp8_bindings(entries, routes)
        if any(binding.pair_id is not None for binding in self._bindings):
            if self._fp8_quantizer is None:
                raise ValueError("FP8 MILES routes require an fp8_quantizer")
        self._specs = {
            binding.canonical_name: LocalParamSpec() for binding in self._bindings
        }
        self._current_sources: dict[str, Any] = {}
        self._prepared: dict[str, Any] = {}
        self._signatures: dict[str, Any] = {}
        self._refresh_generation = 0
        self._started_generation = 0

    @property
    def source_partition_count(self) -> int:
        return self._plan.source_partition_count

    @property
    def device(self) -> str:
        return self._device

    def _validate_fp8_bindings(
        self,
        entries: Mapping[str, Any],
        routes: Mapping[str, MilesSourceRoute],
    ) -> dict[str, tuple[MilesSourceBinding, MilesSourceBinding]]:
        pairs: dict[str, dict[MilesTensorRole, MilesSourceBinding]] = {}
        modules: dict[str, set[str]] = {}
        for binding in self._bindings:
            if binding.pair_id is None or binding.tensor_role is None:
                continue
            role_bindings = pairs.setdefault(binding.pair_id, {})
            if binding.tensor_role in role_bindings:
                raise ValueError(
                    f"{binding.pair_id}: duplicate FP8 {binding.tensor_role}"
                )
            role_bindings[binding.tensor_role] = binding
            module, component = _component(binding.pair_id)
            modules.setdefault(module, set()).add(component)
        for pair_id, pair in pairs.items():
            if set(pair) != {"weight", "scale"}:
                raise ValueError(
                    f"MILES FP8 pair {pair_id!r} is incomplete: {sorted(pair)}"
                )
            weight = pair["weight"]
            scale = pair["scale"]
            weight_entry = entries[weight.canonical_name]
            scale_entry = entries[scale.canonical_name]
            weight_route = routes[weight.canonical_name]
            scale_route = routes[scale.canonical_name]
            if (
                scale.canonical_name != f"{pair_id}_scale_inv"
                or weight.source_names != scale.source_names
                or weight.source_keys != scale.source_keys
                or weight.recipe is not scale.recipe
                or weight_route.family != scale_route.family
                or weight_route.partition_id != scale_route.partition_id
                or weight_route.source_world_ranks != scale_route.source_world_ranks
                or weight_route.source_names_by_world
                != scale_route.source_names_by_world
                or weight_entry.partition_id != scale_entry.partition_id
                or weight_entry.src_mesh != scale_entry.src_mesh
                or weight_entry.src_placements != scale_entry.src_placements
                or scale.expected_shape
                != _fp8_scale_shape(weight.expected_shape, pair_id)
            ):
                raise ValueError(f"MILES FP8 pair {pair_id!r} is inconsistent")
        for module, components in modules.items():
            if components != {"gate", "up", "down"}:
                raise ValueError(
                    f"MILES FP8 expert module {module!r} is incomplete: "
                    f"{sorted(components)}"
                )
        return {
            pair_id: (pair["weight"], pair["scale"]) for pair_id, pair in pairs.items()
        }

    def validate_topology(self, topology: Any) -> None:
        if self.source_partition_count != topology.source_partition_count:
            raise ValueError(
                "plan source_partition_count does not match collective topology"
            )
        if len(self._topology.trainer_lanes) != self.source_partition_count:
            raise ValueError("MILES trainer lanes do not match plan source partitions")
        lane_sizes = {len(lane) for lane in self._topology.trainer_lanes}
        if len(lane_sizes) != 1 or not lane_sizes or 0 in lane_sizes:
            raise ValueError("MILES trainer lanes must be non-empty and equally sized")
        trainers_per_lane = (
            len(topology.trainer_slots) // topology.source_partition_count
        )
        if trainers_per_lane != next(iter(lane_sizes)):
            raise ValueError(
                "MILES trainer lane width does not match collective topology"
            )
        expected_dst = list(
            range(
                trainers_per_lane,
                trainers_per_lane + len(topology.generator_slots),
            )
        )
        for entry in self._plan.bulk:
            src_ranks = entry.src_mesh.ranks()
            if (
                not src_ranks
                or src_ranks != list(range(src_ranks[0], src_ranks[0] + len(src_ranks)))
                or src_ranks[-1] >= trainers_per_lane
            ):
                raise ValueError(
                    f"{entry.name}: source mesh is not a contiguous trainer subset"
                )
            if entry.dst_mesh.ranks() != expected_dst:
                raise ValueError(
                    f"{entry.name}: destination mesh does not match generators"
                )

    def capture(self) -> ReshardPlan:
        return copy.deepcopy(self._plan)

    def parameter_names(self) -> list[str]:
        return self._plan.parameter_names()

    def local_params(self) -> dict[str, LocalParamSpec]:
        return dict(self._specs)

    def source_bindings(self) -> tuple[MilesSourceBinding, ...]:
        return tuple(self._bindings)

    def refresh(self, weights: Mapping[str, Any]) -> None:
        current_sources: dict[str, Any] = {}
        records = {record.native_name: record for record in self._inventory}
        for binding in self._bindings:
            for native_name, source_key in zip(
                binding.source_names,
                binding.source_keys,
                strict=True,
            ):
                if source_key not in weights:
                    raise KeyError(
                        f"{binding.canonical_name}: current weights are missing "
                        f"source key {source_key!r}"
                    )
                tensor = weights[source_key]
                record = records[native_name]
                _tensor_signature(
                    native_name,
                    tensor,
                    expected_shape=record.shape,
                    expected_dtype=record.dtype,
                )
                if str(tensor.device) != self._device:
                    raise ValueError(
                        f"{native_name}: tensor device {tensor.device} does not "
                        f"match publisher device {self._device}"
                    )
                current_sources[native_name] = tensor

        prepared: dict[str, Any] = {}
        for binding in self._bindings:
            tensors = [current_sources[name] for name in binding.source_names]
            if binding.pair_id is None:
                prepared[binding.canonical_name] = _prepare_logical_source(
                    tensors,
                    binding.recipe,
                )

        quantizer = self._fp8_quantizer
        for weight_binding, scale_binding in self._fp8_pairs.values():
            tensors = [current_sources[name] for name in weight_binding.source_names]
            logical = _prepare_logical_source(tensors, weight_binding.recipe)
            if quantizer is None:
                raise RuntimeError("FP8 MILES publisher has no quantizer")
            result = quantizer(logical)
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("fp8_quantizer must return a (weight, scale) tuple")
            prepared[weight_binding.canonical_name] = result[0]
            prepared[scale_binding.canonical_name] = result[1]

        signatures = {}
        for binding in self._bindings:
            tensor = prepared[binding.canonical_name]
            signature = _tensor_signature(
                binding.canonical_name,
                tensor,
                expected_shape=binding.expected_shape,
                expected_dtype=binding.dtype,
            )
            if signature.device != self._device:
                raise ValueError(
                    f"{binding.canonical_name}: prepared device "
                    f"{signature.device} does not match {self._device}"
                )
            signatures[binding.canonical_name] = signature

        for name, spec in self._specs.items():
            spec.base = prepared[name]
        self._current_sources = current_sources
        self._prepared = prepared
        self._signatures = signatures
        self._refresh_generation += 1

    def start_new_round(self, version: str) -> None:
        _text(version, "version")
        if self._refresh_generation == self._started_generation:
            raise RuntimeError(
                "refresh must bind current native MILES weights before each round"
            )
        for binding in self._bindings:
            _check_stable(
                binding.canonical_name,
                self._prepared[binding.canonical_name],
                self._signatures[binding.canonical_name],
                expected_shape=binding.expected_shape,
                expected_dtype=binding.dtype,
            )
        self._started_generation = self._refresh_generation

    def cleanup(self) -> None:
        for spec in self._specs.values():
            spec.base = None
        self._current_sources.clear()
        self._prepared.clear()
        self._signatures.clear()


__all__ = [
    "MilesNativePublisher",
    "MilesNativeTensorRecord",
    "MilesSourceBinding",
    "MilesSourceRecipe",
    "inventory_miles_native_tensors",
]
