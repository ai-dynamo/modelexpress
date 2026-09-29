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


def _placement_to_wire(placement: Placement) -> dict[str, Any]:
    return {"kind": placement.kind.value, "dim": placement.dim}


def _placement_from_wire(value: object) -> Placement:
    if not isinstance(value, dict):
        raise ValueError("placement must be an object")
    kind = PlacementKind(str(value.get("kind", "")))
    dim = value.get("dim")
    return Placement(kind=kind, dim=None if dim is None else int(dim))


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
    raw_bulk = value.get("bulk")
    if not isinstance(raw_bulk, list):
        raise ValueError("plan.bulk must be a list")
    bulk = []
    for raw in raw_bulk:
        if not isinstance(raw, dict):
            raise ValueError("plan.bulk entries must be objects")
        src_mesh = raw.get("src_mesh")
        dst_mesh = raw.get("dst_mesh")
        if not isinstance(src_mesh, dict) or not isinstance(dst_mesh, dict):
            raise ValueError("plan meshes must be objects")
        bulk.append(
            ParamPlan(
                name=str(raw.get("name", "")),
                global_shape=tuple(int(dim) for dim in raw.get("global_shape", [])),
                dtype=str(raw.get("dtype", "")),
                partition_id=int(raw.get("partition_id", -1)),
                src_mesh=MeshSpec(
                    tuple(int(dim) for dim in src_mesh.get("shape", [])),
                    int(src_mesh.get("rank_offset", 0)),
                ),
                src_placements=tuple(
                    _placement_from_wire(item) for item in raw.get("src_placements", [])
                ),
                dst_mesh=MeshSpec(
                    tuple(int(dim) for dim in dst_mesh.get("shape", [])),
                    int(dst_mesh.get("rank_offset", 0)),
                ),
                dst_placements=tuple(
                    _placement_from_wire(item) for item in raw.get("dst_placements", [])
                ),
                group_key=raw.get("group_key"),
            )
        )
    return ReshardPlan(
        bulk=bulk,
        source_partition_count=int(value.get("source_partition_count", 0)),
    )


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
        model_name=str(value.get("model_name", "")),
        trainer_slots=tuple(str(slot) for slot in value.get("trainer_slots", [])),
        generator_slots=tuple(str(slot) for slot in value.get("generator_slots", [])),
        source_partition_count=int(value.get("source_partition_count", 0)),
        m2n_abi_version=str(value.get("m2n_abi_version", "")),
        receiver_protocol=str(value.get("receiver_protocol", "")),
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


def encode_control(control: CollectiveControl) -> str:
    payload: dict[str, Any] = {"action": control.action}
    if control.plan is not None:
        payload["plan"] = plan_to_wire(control.plan)
    if control.topology is not None:
        payload["topology"] = topology_to_wire(control.topology)
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
        action=str(raw.get("action", "")),
        plan=plan_from_wire(raw["plan"]) if "plan" in raw else None,
        topology=topology_from_wire(raw["topology"]) if "topology" in raw else None,
        generator_slot_offset=(
            int(raw["generator_slot_offset"])
            if "generator_slot_offset" in raw
            else None
        ),
        version=_optional_str(raw.get("version")),
        operation_id=_optional_str(raw.get("operation_id")),
        endpoint=_optional_str(raw.get("endpoint")),
    )


__all__ = [
    "CONTROL_PREFIX",
    "CollectiveControl",
    "MAX_CONTROL_CHARS",
    "decode_control",
    "encode_control",
    "plan_from_wire",
    "plan_to_wire",
    "topology_from_wire",
    "topology_to_wire",
]
