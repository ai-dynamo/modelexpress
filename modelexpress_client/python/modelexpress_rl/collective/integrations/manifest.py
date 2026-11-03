# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical JSON wire for the MILES/SGLang ``m2n_manifest`` (plan and topology)."""

from __future__ import annotations

from typing import Any

from ..types import MeshSpec, ParamPlan, Placement, PlacementKind, ReshardPlan
from .miles import CollectiveTopology

#: The ``m2n_manifest`` schema the public SGLang receiver factory accepts.
MANIFEST_SCHEMA = "modelexpress.reshard_plan/v1"


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    """Read a mandatory wire field; a control boundary must not invent one."""
    if key not in mapping:
        raise ValueError(f"{context} is missing {key!r}")
    return mapping[key]


def _require_int(mapping: dict[str, Any], key: str, context: str) -> int:
    """Read a mandatory integer wire field, rejecting bools and coercions."""
    value = _require(mapping, key, context)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} field {key!r} must be an integer")
    return value


def _optional_int(mapping: dict[str, Any], key: str, context: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} field {key!r} must be an integer or null")
    return value


def _require_str(mapping: dict[str, Any], key: str, context: str) -> str:
    value = _require(mapping, key, context)
    if not isinstance(value, str):
        raise ValueError(f"{context} field {key!r} must be a string")
    return value


def _require_list(mapping: dict[str, Any], key: str, context: str) -> list[Any]:
    """Read a mandatory list wire field; a string would iterate into characters."""
    value = _require(mapping, key, context)
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} field {key!r} must be a list")
    return list(value)


def _require_int_list(
    mapping: dict[str, Any], key: str, context: str
) -> tuple[int, ...]:
    items = []
    for item in _require_list(mapping, key, context):
        if not isinstance(item, int) or isinstance(item, bool):
            raise ValueError(f"{context} field {key!r} must be a list of integers")
        items.append(item)
    return tuple(items)


def _require_str_list(
    mapping: dict[str, Any], key: str, context: str
) -> tuple[str, ...]:
    items = []
    for item in _require_list(mapping, key, context):
        if not isinstance(item, str):
            raise ValueError(f"{context} field {key!r} must be a list of strings")
        items.append(item)
    return tuple(items)


def _placement_to_wire(placement: Placement) -> dict[str, Any]:
    return {"kind": placement.kind.value, "dim": placement.dim}


def _placement_from_wire(value: object) -> Placement:
    if not isinstance(value, dict):
        raise ValueError("placement must be an object")
    kind = PlacementKind(_require_str(value, "kind", "placement"))
    return Placement(kind=kind, dim=_optional_int(value, "dim", "placement"))


def plan_to_wire(plan: ReshardPlan) -> dict[str, Any]:
    plan.validate()
    if plan.misc:
        raise ValueError("the MILES bridge supports all-bulk plans only")
    return {
        "source_partition_count": plan.source_partition_count,
        "bulk": [
            {
                "name": entry.name,
                "global_shape": list(entry.global_shape),
                "dtype": entry.dtype,
                "partition_id": entry.partition_id,
                "src_mesh": {
                    "shape": list(entry.src_mesh.shape),
                    "rank_offset": entry.src_mesh.rank_offset,
                },
                "src_placements": [
                    _placement_to_wire(value) for value in entry.src_placements
                ],
                "dst_mesh": {
                    "shape": list(entry.dst_mesh.shape),
                    "rank_offset": entry.dst_mesh.rank_offset,
                },
                "dst_placements": [
                    _placement_to_wire(value) for value in entry.dst_placements
                ],
                "group_key": entry.group_key,
            }
            for entry in plan.bulk
        ],
    }


def plan_from_wire(value: object) -> ReshardPlan:
    if not isinstance(value, dict):
        raise ValueError("plan must be an object")
    raw_bulk = _require(value, "bulk", "plan")
    if not isinstance(raw_bulk, list):
        raise ValueError("plan.bulk must be a list")
    bulk = []
    for raw in raw_bulk:
        if not isinstance(raw, dict):
            raise ValueError("plan.bulk entries must be objects")
        context = f"plan.bulk entry {raw.get('name')!r}"
        src_mesh = _require(raw, "src_mesh", context)
        dst_mesh = _require(raw, "dst_mesh", context)
        if not isinstance(src_mesh, dict) or not isinstance(dst_mesh, dict):
            raise ValueError("plan meshes must be objects")
        bulk.append(
            ParamPlan(
                name=_require_str(raw, "name", context),
                global_shape=_require_int_list(raw, "global_shape", context),
                dtype=_require_str(raw, "dtype", context),
                partition_id=_require_int(raw, "partition_id", context),
                src_mesh=MeshSpec(
                    _require_int_list(src_mesh, "shape", context),
                    _require_int(src_mesh, "rank_offset", context),
                ),
                src_placements=tuple(
                    _placement_from_wire(item)
                    for item in _require_list(raw, "src_placements", context)
                ),
                dst_mesh=MeshSpec(
                    _require_int_list(dst_mesh, "shape", context),
                    _require_int(dst_mesh, "rank_offset", context),
                ),
                dst_placements=tuple(
                    _placement_from_wire(item)
                    for item in _require_list(raw, "dst_placements", context)
                ),
                group_key=_optional_str(_require(raw, "group_key", context)),
            )
        )
    plan = ReshardPlan(
        bulk=bulk,
        source_partition_count=_require_int(value, "source_partition_count", "plan"),
    )
    plan.validate()
    return plan


def topology_to_wire(topology: CollectiveTopology) -> dict[str, Any]:
    return {
        "model_name": topology.model_name,
        "trainer_slots": list(topology.trainer_slots),
        "generator_slots": list(topology.generator_slots),
        "source_partition_count": topology.source_partition_count,
        "m2n_abi_version": topology.m2n_abi_version,
        "receiver_protocol": topology.receiver_protocol,
    }


def topology_from_wire(value: object) -> CollectiveTopology:
    if not isinstance(value, dict):
        raise ValueError("topology must be an object")
    return CollectiveTopology(
        model_name=_require_str(value, "model_name", "topology"),
        trainer_slots=_require_str_list(value, "trainer_slots", "topology"),
        generator_slots=_require_str_list(value, "generator_slots", "topology"),
        source_partition_count=_require_int(
            value, "source_partition_count", "topology"
        ),
        m2n_abi_version=_require_str(value, "m2n_abi_version", "topology"),
        receiver_protocol=_require_str(value, "receiver_protocol", "topology"),
    )


def manifest_to_wire(plan: ReshardPlan, topology: CollectiveTopology) -> dict[str, Any]:
    """The ``m2n_manifest`` the trainer sends with ``init_weights_update_group``."""
    return {
        "schema": MANIFEST_SCHEMA,
        "plan": plan_to_wire(plan),
        "topology": topology_to_wire(topology),
    }


def manifest_from_wire(value: object) -> tuple[ReshardPlan, CollectiveTopology]:
    """Decode an ``m2n_manifest``; any other schema or key is refused."""
    if not isinstance(value, dict):
        raise ValueError("m2n_manifest must be an object")
    schema = value.get("schema")
    if schema != MANIFEST_SCHEMA:
        raise ValueError(
            f"m2n_manifest schema must be {MANIFEST_SCHEMA!r}, got {schema!r}"
        )
    unexpected = sorted(set(value) - {"schema", "plan", "topology"})
    if unexpected:
        raise ValueError(f"m2n_manifest has unexpected fields {unexpected}")
    return (
        plan_from_wire(_require(value, "plan", "m2n_manifest")),
        topology_from_wire(_require(value, "topology", "m2n_manifest")),
    )


__all__ = [
    "MANIFEST_SCHEMA",
    "manifest_from_wire",
    "manifest_to_wire",
    "plan_from_wire",
    "plan_to_wire",
    "topology_from_wire",
    "topology_to_wire",
]
