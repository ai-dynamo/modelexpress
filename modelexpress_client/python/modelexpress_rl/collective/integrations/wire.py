# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical JSON control wire for the MILES/SGLang collective bridge."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from ..plan import validate_semantic_manifest_identity
from ..types import MeshSpec, ParamPlan, Placement, PlacementKind, ReshardPlan
from .miles import CollectiveTopology

CONTROL_PREFIX = "modelexpress:miles-m2n:v1:"
_ACTIONS = frozenset({"prepare", "run_round", "close"})
_TENSOR_DIGEST = re.compile(r"[0-9a-f]{64}")
MAX_CONTROL_CHARS = 32 * 1024 * 1024
MAX_TENSOR_DIGESTS = 100_000
MAX_TENSOR_NAME_CHARS = 1024
MAX_DESTINATION_ENTRIES = 100_000
MAX_DESTINATION_NDIM = 8

TensorDigestMap = tuple[tuple[str, str], ...]


def _bounded_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > MAX_TENSOR_NAME_CHARS:
        raise ValueError(f"{name} exceeds the {MAX_TENSOR_NAME_CHARS} character limit")
    return value


@dataclass(frozen=True)
class DestinationQuantization:
    """Transport-independent quantization contract for one destination."""

    quant_method: str
    activation_scheme: str
    weight_block_size: tuple[int, int]
    weight_dtype: str
    scale_dtype: str
    scale_format: str

    def __post_init__(self) -> None:
        for name in (
            "quant_method",
            "activation_scheme",
            "weight_dtype",
            "scale_dtype",
            "scale_format",
        ):
            _bounded_text(getattr(self, name), f"destination quantization {name}")
        if (
            not isinstance(self.weight_block_size, tuple)
            or len(self.weight_block_size) != 2
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                for dim in self.weight_block_size
            )
        ):
            raise ValueError(
                "destination quantization weight_block_size must contain two "
                "positive integers"
            )


@dataclass(frozen=True)
class DestinationManifestEntry:
    """Exact receiver-side semantics selected by an integration manifest."""

    name: str
    dtype: str
    local_shape: tuple[int, ...]
    parameter: str
    recipe: str
    family: str = "dense"
    pair_id: str | None = None
    tensor_role: str | None = None
    quantization: DestinationQuantization | None = None

    def __post_init__(self) -> None:
        _bounded_text(self.name, "destination manifest name")
        _bounded_text(self.dtype, "destination manifest dtype")
        _bounded_text(self.parameter, "destination manifest parameter")
        _bounded_text(self.recipe, "destination manifest recipe")
        if self.family not in ("dense", "routed_expert"):
            raise ValueError(f"unsupported destination manifest family {self.family!r}")
        if (self.pair_id is None) != (self.tensor_role is None):
            raise ValueError(
                "destination manifest pair_id and tensor_role must be set together"
            )
        if self.tensor_role not in (None, "weight", "scale"):
            raise ValueError(
                f"unsupported destination tensor_role {self.tensor_role!r}"
            )
        if self.pair_id is not None:
            _bounded_text(self.pair_id, "destination manifest pair_id")
            if self.family != "routed_expert" or self.quantization is None:
                raise ValueError(
                    "paired destination tensors require routed_expert family "
                    "and quantization"
                )
        elif self.quantization is not None:
            raise ValueError("destination quantization requires atomic pair metadata")
        if (
            not isinstance(self.local_shape, tuple)
            or not self.local_shape
            or len(self.local_shape) > MAX_DESTINATION_NDIM
            or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
                for dim in self.local_shape
            )
        ):
            raise ValueError(
                "destination manifest local_shape must contain one to "
                f"{MAX_DESTINATION_NDIM} positive integer dimensions"
            )


def validate_destination_manifest(
    value: object,
) -> tuple[DestinationManifestEntry, ...]:
    if not isinstance(value, tuple) or not value:
        raise ValueError("destination manifest must be a non-empty tuple")
    if len(value) > MAX_DESTINATION_ENTRIES:
        raise ValueError(
            f"destination manifest exceeds the {MAX_DESTINATION_ENTRIES} entry limit"
        )
    if not all(isinstance(entry, DestinationManifestEntry) for entry in value):
        raise ValueError(
            "destination manifest entries must be DestinationManifestEntry values"
        )
    names = [entry.name for entry in value]
    if len(names) != len(set(names)):
        raise ValueError("destination manifest names must be unique")
    return value


def validate_tensor_digests(value: object) -> TensorDigestMap:
    if not isinstance(value, tuple) or not value:
        raise ValueError("run_round requires tensor digests")
    if len(value) > MAX_TENSOR_DIGESTS:
        raise ValueError(f"tensor digests exceed the {MAX_TENSOR_DIGESTS} entry limit")
    normalized: list[tuple[str, str]] = []
    for item in value:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(part, str) for part in item)
        ):
            raise ValueError("tensor digests must be (name, digest) string pairs")
        name, digest = item
        if not name:
            raise ValueError("tensor digest names must not be empty")
        if len(name) > MAX_TENSOR_NAME_CHARS:
            raise ValueError(
                f"tensor digest name exceeds the {MAX_TENSOR_NAME_CHARS} "
                "character limit"
            )
        if _TENSOR_DIGEST.fullmatch(digest) is None:
            raise ValueError(
                f"{name}: tensor digest must be 64 lowercase hexadecimal characters"
            )
        normalized.append((name, digest))
    names = [name for name, _digest in normalized]
    if len(names) != len(set(names)):
        raise ValueError("tensor digests must contain unique tensor names")
    if names != sorted(names):
        raise ValueError("tensor digests must be sorted by tensor name")
    return tuple(normalized)


def tensor_equality_receipt(
    *,
    version: str,
    operation_id: str,
    tensor_digests: TensorDigestMap,
) -> str:
    tensor_digests = validate_tensor_digests(tensor_digests)
    canonical = json.dumps(
        {
            "operation_id": operation_id,
            "tensor_digests": tensor_digests,
            "version": version,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    tensor_digests: TensorDigestMap | None = None
    semantic_manifest_version: str | None = None
    semantic_manifest_digest: str | None = None
    destination_manifest: tuple[DestinationManifestEntry, ...] | None = None

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
            validate_semantic_manifest_identity(
                version=self.semantic_manifest_version,
                digest=self.semantic_manifest_digest,
            )
            if self.destination_manifest is not None:
                validate_destination_manifest(self.destination_manifest)
                if self.semantic_manifest_version is None:
                    raise ValueError(
                        "destination manifest requires semantic manifest identity"
                    )
        elif self.action == "run_round":
            if not str(self.version or "").strip():
                raise ValueError("run_round requires a version")
            if not str(self.operation_id or "").strip():
                raise ValueError("run_round requires an operation_id")
            if self.tensor_digests is not None:
                validate_tensor_digests(self.tensor_digests)
        if self.action != "run_round" and self.tensor_digests is not None:
            raise ValueError("tensor digests are valid only for run_round")
        if self.action != "prepare" and (
            self.semantic_manifest_version is not None
            or self.semantic_manifest_digest is not None
        ):
            raise ValueError("semantic manifest identity is valid only for prepare")
        if self.action != "prepare" and self.destination_manifest is not None:
            raise ValueError("destination manifest is valid only for prepare")


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
    if control.tensor_digests is not None:
        payload["tensor_digests"] = [list(item) for item in control.tensor_digests]
    if control.semantic_manifest_version is not None:
        payload["semantic_manifest_version"] = control.semantic_manifest_version
        payload["semantic_manifest_digest"] = control.semantic_manifest_digest
    if control.destination_manifest is not None:
        payload["destination_manifest"] = [
            {
                "name": entry.name,
                "dtype": entry.dtype,
                "local_shape": list(entry.local_shape),
                "parameter": entry.parameter,
                "recipe": entry.recipe,
                "family": entry.family,
                "pair_id": entry.pair_id,
                "tensor_role": entry.tensor_role,
                "quantization": (
                    {
                        "quant_method": entry.quantization.quant_method,
                        "activation_scheme": entry.quantization.activation_scheme,
                        "weight_block_size": list(entry.quantization.weight_block_size),
                        "weight_dtype": entry.quantization.weight_dtype,
                        "scale_dtype": entry.quantization.scale_dtype,
                        "scale_format": entry.quantization.scale_format,
                    }
                    if entry.quantization is not None
                    else None
                ),
            }
            for entry in control.destination_manifest
        ]
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
    raw_tensor_digests = raw.get("tensor_digests")
    tensor_digests = None
    if raw_tensor_digests is not None:
        if not isinstance(raw_tensor_digests, list):
            raise ValueError("tensor_digests must be a list")
        if len(raw_tensor_digests) > MAX_TENSOR_DIGESTS:
            raise ValueError(
                f"tensor digests exceed the {MAX_TENSOR_DIGESTS} entry limit"
            )
        normalized = []
        for item in raw_tensor_digests:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or not all(isinstance(part, str) for part in item)
            ):
                raise ValueError(
                    "tensor digests must be two-element string lists on the wire"
                )
            normalized.append((item[0], item[1]))
        tensor_digests = tuple(normalized)
    raw_destination_manifest = raw.get("destination_manifest")
    destination_manifest = None
    if raw_destination_manifest is not None:
        if not isinstance(raw_destination_manifest, list):
            raise ValueError("destination_manifest must be a list")
        if len(raw_destination_manifest) > MAX_DESTINATION_ENTRIES:
            raise ValueError(
                "destination manifest exceeds the "
                f"{MAX_DESTINATION_ENTRIES} entry limit"
            )
        normalized_destinations = []
        expected_fields = {
            "name",
            "dtype",
            "local_shape",
            "parameter",
            "recipe",
            "family",
            "pair_id",
            "tensor_role",
            "quantization",
        }
        for raw_entry in raw_destination_manifest:
            if not isinstance(raw_entry, dict) or set(raw_entry) != expected_fields:
                raise ValueError(
                    "destination manifest entries must contain exactly "
                    f"{sorted(expected_fields)}"
                )
            raw_shape = raw_entry["local_shape"]
            if not isinstance(raw_shape, list):
                raise ValueError("destination manifest local_shape must be a list")
            raw_quantization = raw_entry["quantization"]
            quantization = None
            if raw_quantization is not None:
                quantization_fields = {
                    "quant_method",
                    "activation_scheme",
                    "weight_block_size",
                    "weight_dtype",
                    "scale_dtype",
                    "scale_format",
                }
                if (
                    not isinstance(raw_quantization, dict)
                    or set(raw_quantization) != quantization_fields
                    or not isinstance(
                        raw_quantization["weight_block_size"],
                        list,
                    )
                ):
                    raise ValueError(
                        "destination quantization must contain exactly "
                        f"{sorted(quantization_fields)}"
                    )
                quantization = DestinationQuantization(
                    quant_method=raw_quantization["quant_method"],
                    activation_scheme=raw_quantization["activation_scheme"],
                    weight_block_size=tuple(raw_quantization["weight_block_size"]),
                    weight_dtype=raw_quantization["weight_dtype"],
                    scale_dtype=raw_quantization["scale_dtype"],
                    scale_format=raw_quantization["scale_format"],
                )
            normalized_destinations.append(
                DestinationManifestEntry(
                    name=raw_entry["name"],
                    dtype=raw_entry["dtype"],
                    local_shape=tuple(raw_shape),
                    parameter=raw_entry["parameter"],
                    recipe=raw_entry["recipe"],
                    family=raw_entry["family"],
                    pair_id=raw_entry["pair_id"],
                    tensor_role=raw_entry["tensor_role"],
                    quantization=quantization,
                )
            )
        destination_manifest = tuple(normalized_destinations)
    return CollectiveControl(
        action=str(raw.get("action", "")),
        plan=plan_from_wire(raw["plan"]) if "plan" in raw else None,
        topology=topology_from_wire(raw["topology"]) if "topology" in raw else None,
        generator_slot_offset=(
            int(raw["generator_slot_offset"])
            if "generator_slot_offset" in raw
            else None
        ),
        version=raw.get("version"),
        operation_id=raw.get("operation_id"),
        endpoint=raw.get("endpoint"),
        tensor_digests=tensor_digests,
        semantic_manifest_version=raw.get("semantic_manifest_version"),
        semantic_manifest_digest=raw.get("semantic_manifest_digest"),
        destination_manifest=destination_manifest,
    )


__all__ = [
    "CONTROL_PREFIX",
    "CollectiveControl",
    "DestinationManifestEntry",
    "DestinationQuantization",
    "MAX_CONTROL_CHARS",
    "MAX_DESTINATION_ENTRIES",
    "MAX_DESTINATION_NDIM",
    "MAX_TENSOR_DIGESTS",
    "MAX_TENSOR_NAME_CHARS",
    "TensorDigestMap",
    "decode_control",
    "encode_control",
    "plan_from_wire",
    "plan_to_wire",
    "topology_from_wire",
    "topology_to_wire",
    "tensor_equality_receipt",
    "validate_destination_manifest",
    "validate_tensor_digests",
]
