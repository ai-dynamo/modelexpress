# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical JSON control wire for the MILES/SGLang collective bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..types import MeshSpec, ParamPlan, Placement, PlacementKind, ReshardPlan
from .miles import CollectiveTopology

CONTROL_PREFIX = "modelexpress:miles-m2n:v1:"
_ACTIONS = frozenset({"prepare", "run_round", "close"})
MAX_CONTROL_CHARS = 32 * 1024 * 1024


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


@dataclass(frozen=True)
class CollectiveControl:
    """One scheduler-local bridge command transported through SGLang."""

    action: str
    plan: ReshardPlan | None = None
    topology: CollectiveTopology | None = None
    generator_slot_offset: int | None = None
    version: str | None = None
    operation_id: str | None = None
    endpoint: str | None = None

    def __post_init__(self) -> None:
        if self.action not in _ACTIONS:
            raise ValueError(f"unsupported collective action {self.action!r}")
        if self.action == "prepare":
            if self.plan is None or self.topology is None:
                raise ValueError("prepare requires a plan and topology")
            if self.generator_slot_offset is None or self.generator_slot_offset < 0:
                raise ValueError(
                    "prepare requires a non-negative generator slot offset"
                )
            if not str(self.endpoint or "").strip():
                raise ValueError("prepare requires a ModelExpress endpoint")
        elif self.action == "run_round":
            if not str(self.version or "").strip():
                raise ValueError("run_round requires a version")
            if not str(self.operation_id or "").strip():
                raise ValueError("run_round requires an operation_id")


def encode_control(
    control: CollectiveControl,
    *,
    plan_wire: dict[str, Any] | None = None,
    topology_wire: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {"action": control.action}
    if control.plan is not None:
        payload["plan"] = plan_to_wire(control.plan) if plan_wire is None else plan_wire
    if control.topology is not None:
        payload["topology"] = (
            topology_to_wire(control.topology)
            if topology_wire is None
            else topology_wire
        )
    if control.generator_slot_offset is not None:
        payload["generator_slot_offset"] = control.generator_slot_offset
    if control.version is not None:
        payload["version"] = control.version
    if control.operation_id is not None:
        payload["operation_id"] = control.operation_id
    if control.endpoint is not None:
        payload["endpoint"] = control.endpoint
    encoded = CONTROL_PREFIX + json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    if len(encoded) > MAX_CONTROL_CHARS:
        raise ValueError(
            f"collective control exceeds the {MAX_CONTROL_CHARS} character limit"
        )
    return encoded


def decode_control(value: object) -> CollectiveControl | None:
    if not isinstance(value, str) or not value.startswith(CONTROL_PREFIX):
        return None
    if len(value) > MAX_CONTROL_CHARS:
        raise ValueError(
            f"collective control exceeds the {MAX_CONTROL_CHARS} character limit"
        )
    raw = json.loads(value.removeprefix(CONTROL_PREFIX))
    if not isinstance(raw, dict):
        raise ValueError("collective control payload must be an object")
    return CollectiveControl(
        action=_require_str(raw, "action", "collective control"),
        plan=plan_from_wire(raw["plan"]) if "plan" in raw else None,
        topology=topology_from_wire(raw["topology"]) if "topology" in raw else None,
        # The encoder only writes this field when it is set, so a present
        # null is malformed rather than absent.
        generator_slot_offset=(
            _require_int(raw, "generator_slot_offset", "collective control")
            if "generator_slot_offset" in raw
            else None
        ),
        version=_optional_str(raw.get("version")),
        operation_id=_optional_str(raw.get("operation_id")),
        endpoint=_optional_str(raw.get("endpoint")),
    )


__all__ = [
    "CONTROL_PREFIX",
    "MAX_CONTROL_CHARS",
    "CollectiveControl",
    "decode_control",
    "encode_control",
    "plan_from_wire",
    "plan_to_wire",
    "topology_from_wire",
    "topology_to_wire",
]
