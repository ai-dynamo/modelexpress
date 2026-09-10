# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Miles tensor-spec translation and public RL client binding (no Ray imports)."""

from __future__ import annotations

import atexit
import ipaddress
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import torch
from modelexpress import envs
from modelexpress_rl.control import ModelExpressControlClient, WeightVersionState
from modelexpress_rl.train import ModelExpressTrainerClient, ModelExpressTrainerConfig
from modelexpress_rl.train.adapter import TrainerStagingMode, WeightPayloadFormat
from modelexpress_rl.train.context import MegatronTrainerContext
from modelexpress_rl.train.engines.megatron.aliases import (
    MegatronTensorSpec,
    build_hf_aliases,
)
from modelexpress_rl.version import WeightVersionRef


def _require_text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Miles ModelExpress registration requires non-empty {label}")
    return text


def _routable_worker_host() -> str:
    host = _require_text(
        envs.MX_WORKER_HOST,
        "MX_WORKER_HOST (a receiver-routable trainer hostname or IP)",
    )
    lowered = host.lower().rstrip(".")
    if lowered in {"localhost", "localhost.localdomain"}:
        raise ValueError("MX_WORKER_HOST must be receiver-routable, not localhost")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if address.version == 6:
        raise ValueError(
            "MX_WORKER_HOST IPv6 literals are unsupported by the current "
            "ModelExpress host:port rendezvous contract; use a routable DNS name"
        )
    if address.is_loopback or address.is_unspecified:
        raise ValueError(f"MX_WORKER_HOST must be receiver-routable, got {host!r}")
    return host


def _tensor_signature(tensors: Mapping[str, Any]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            name,
            int(tensor.data_ptr()),
            tuple(int(dim) for dim in tensor.shape),
            str(tensor.dtype),
        )
        for name, tensor in sorted(tensors.items())
    )


def _request_tensors(request: Any) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    for spec in request.tensors:
        name = _require_text(spec.native_name, "tensor native_name")
        if name in tensors:
            raise ValueError(f"duplicate Miles native tensor name {name!r}")
        tensor = spec.tensor
        if not tensor.is_contiguous():
            raise ValueError(f"{name}: native Megatron storage must be contiguous")
        tensors[name] = tensor
    if not tensors:
        raise ValueError("Miles ModelExpress publish request contains no tensors")
    return tensors


def _alias_by_name(spec: Any) -> dict[str, Any]:
    aliases: dict[str, Any] = {}
    for alias in spec.aliases:
        name = _require_text(alias.hf_name, f"{spec.native_name} alias hf_name")
        if name in aliases:
            raise ValueError(f"{spec.native_name}: duplicate HF alias {name!r}")
        aliases[name] = alias
    if tuple(aliases) != tuple(spec.hf_names):
        raise ValueError(
            f"{spec.native_name}: hf_names {tuple(spec.hf_names)!r} do not match "
            f"alias names {tuple(aliases)!r}"
        )
    return aliases


def _validate_alias_output(
    spec: Any, aliases: Mapping[str, Any], published: Sequence[Any]
) -> None:
    actual = {item.name: item for item in published}
    if set(actual) != set(aliases) or len(actual) != len(published):
        raise ValueError(
            f"{spec.native_name}: generated aliases {sorted(actual)!r} do not "
            f"match Miles aliases {sorted(aliases)!r}"
        )
    for name, expected in aliases.items():
        item = actual[name]
        expected_shape = tuple(int(dim) for dim in expected.global_shape)
        if tuple(item.full_shape) != expected_shape:
            raise ValueError(
                f"{spec.native_name}/{name}: generated global shape "
                f"{tuple(item.full_shape)} != Miles {expected_shape}"
            )
        axis = expected.shard_axis
        shard_range = expected.local_shard_range
        if (axis is None) != (shard_range is None):
            raise ValueError(
                f"{spec.native_name}/{name}: incomplete Miles shard geometry"
            )
        if axis is None:
            if len(item.shards) != 1 or tuple(item.shards[0].shape) != expected_shape:
                raise ValueError(
                    f"{spec.native_name}/{name}: invalid replicated alias geometry"
                )
            continue
        axis = int(axis)
        lo, hi = (int(value) for value in shard_range)
        intervals = []
        for shard in item.shards:
            offset = tuple(int(value) for value in shard.shard_offset)
            shape = tuple(int(value) for value in shard.shape)
            if len(shape) != len(expected_shape) or any(
                offset[index] != 0 or shape[index] != expected_shape[index]
                for index in range(len(shape))
                if index != axis
            ):
                raise ValueError(
                    f"{spec.native_name}/{name}: generated shard is not axis-{axis} aligned"
                )
            intervals.append((offset[axis], offset[axis] + shape[axis]))
        intervals.sort()
        cursor = lo
        for start, end in intervals:
            if start != cursor or end <= start:
                raise ValueError(
                    f"{spec.native_name}/{name}: generated shards do not cover Miles range"
                )
            cursor = end
        if cursor != hi:
            raise ValueError(
                f"{spec.native_name}/{name}: generated range {(lo, cursor)} != Miles {(lo, hi)}"
            )


def _generic_alias_input(spec: Any, alias: Any) -> MegatronTensorSpec | None:
    tensor = spec.tensor
    full_shape = tuple(int(dim) for dim in alias.global_shape)
    shard_axis = None if alias.shard_axis is None else int(alias.shard_axis)
    shard_range = (
        None
        if alias.local_shard_range is None
        else tuple(int(value) for value in alias.local_shard_range)
    )
    if (shard_axis is None) != (shard_range is None):
        raise ValueError(f"{spec.native_name}: incomplete generic alias shard geometry")

    metadata = dict(spec.conversion_metadata)
    if metadata.get("layout") == "padded_vocab":
        if shard_axis != 0 or shard_range is None:
            raise ValueError(
                f"{spec.native_name}: padded vocabulary must be sharded on axis 0"
            )
        native_range = spec.local_shard_range
        if native_range is None:
            if tuple(tensor.shape) != full_shape:
                tensor = tensor.narrow(0, 0, full_shape[0])
        else:
            native_lo, native_hi = (int(value) for value in native_range)
            lo, hi = shard_range
            if lo != native_lo or not native_lo <= hi <= native_hi:
                raise ValueError(
                    f"{spec.native_name}: invalid padded vocabulary canonical range"
                )
            valid_rows = hi - lo
            if valid_rows == 0:
                return None
            tensor = tensor.narrow(0, 0, valid_rows)

    if shard_axis is not None:
        lo, hi = shard_range
        if not 0 <= shard_axis < tensor.ndim or hi - lo != int(
            tensor.shape[shard_axis]
        ):
            raise ValueError(
                f"{spec.native_name}: alias shard range does not match local storage"
            )
        placement = "SHARD"
    else:
        if tuple(int(dim) for dim in tensor.shape) != full_shape:
            raise ValueError(
                f"{spec.native_name}: replicated alias shape does not match storage"
            )
        placement = "REPLICATE"

    if metadata.get("layout") == "padded_vocab":
        role = "vocab"
    elif placement == "REPLICATE":
        role = "replicated"
    elif shard_axis == 1:
        role = "row"
    else:
        role = str(spec.role)
    return MegatronTensorSpec(
        name=str(spec.native_name),
        tensor=tensor,
        role=role,
        hf_names=(str(alias.hf_name),),
        global_shape=full_shape,
        placement_kind=placement,
        shard_axis=shard_axis,
        local_shard_range=shard_range,
    )


def _alias_input(spec: Any) -> tuple[MegatronTensorSpec | None, Mapping[str, Any]]:
    aliases = _alias_by_name(spec)
    role = str(spec.role)
    placement = "SHARD" if spec.shard_axis is not None else "REPLICATE"
    if role == "qkv":
        metadata = dict(spec.conversion_metadata)
        required = ("head_dim", "local_query_groups", "query_heads_per_group")
        missing = [name for name in required if name not in metadata]
        if missing:
            raise ValueError(
                f"{spec.native_name}: QKV conversion metadata missing {missing}"
            )
        item = MegatronTensorSpec(
            name=str(spec.native_name),
            tensor=spec.tensor,
            role="qkv_column",
            hf_names=tuple(str(name) for name in spec.hf_names),
            global_shape=tuple(int(dim) for dim in spec.global_shape),
            placement_kind=placement,
            shard_axis=None if spec.shard_axis is None else int(spec.shard_axis),
            local_shard_range=(
                None
                if spec.local_shard_range is None
                else tuple(int(value) for value in spec.local_shard_range)
            ),
            extras={
                "head_dim": str(metadata["head_dim"]),
                "num_heads_local": str(
                    int(metadata["local_query_groups"])
                    * int(metadata["query_heads_per_group"])
                ),
                "num_kv_heads_local": str(metadata["local_query_groups"]),
            },
        )
        return item, aliases
    if role == "gate_up":
        item = MegatronTensorSpec(
            name=str(spec.native_name),
            tensor=spec.tensor,
            role="gated_mlp_column",
            hf_names=tuple(str(name) for name in spec.hf_names),
            global_shape=tuple(int(dim) for dim in spec.global_shape),
            placement_kind=placement,
            shard_axis=None if spec.shard_axis is None else int(spec.shard_axis),
            local_shard_range=(
                None
                if spec.local_shard_range is None
                else tuple(int(value) for value in spec.local_shard_range)
            ),
            extras={"gated_mlp_order": "gate_then_up"},
        )
        return item, aliases
    if len(aliases) != 1:
        raise ValueError(
            f"{spec.native_name}: generic aliasing requires exactly one HF name"
        )
    return _generic_alias_input(spec, next(iter(aliases.values()))), aliases


def _build_specs(request: Any) -> list[MegatronTensorSpec]:
    inputs: list[MegatronTensorSpec] = []
    expected_aliases = []
    all_names: set[str] = set()
    for spec in request.tensors:
        item, aliases = _alias_input(spec)
        duplicate = all_names.intersection(aliases)
        if duplicate:
            raise ValueError(
                f"duplicate HF aliases across Miles tensors: {sorted(duplicate)!r}"
            )
        all_names.update(aliases)
        if item is not None:
            inputs.append(item)
            expected_aliases.append((spec, aliases))
    published = build_hf_aliases(inputs, agent_name="validation")
    cursor = 0
    for spec, aliases in expected_aliases:
        count = len(aliases)
        _validate_alias_output(spec, aliases, published[cursor : cursor + count])
        cursor += count
    return inputs


class MilesModelExpressPublisher:
    """Implement Miles' publisher protocol with one persistent trainer client."""

    def __init__(self):
        self._client = None
        self._control = None
        self._registration = None
        self._signature = None
        self._specs = None
        atexit.register(self.close)

    def configure(self, registration):
        if self._registration is not None:
            if (
                registration.model_name,
                registration.worker_id,
                registration.source_geometry,
            ) != (
                self._registration.model_name,
                self._registration.worker_id,
                self._registration.source_geometry,
            ):
                raise RuntimeError(
                    "Miles trainer geometry changed; restart the trainer"
                )
        # A changed rollout cohort does not invalidate the trainer's stable storage.
        self._registration = registration

    def prepare(self, tensors):
        if self._registration is None:
            raise RuntimeError("configure the Miles publisher before preparing")
        request = SimpleNamespace(tensors=tensors)
        signature = _tensor_signature(_request_tensors(request))
        if self._client is not None:
            if signature != self._signature:
                raise RuntimeError("Miles trainer storage changed after binding")
            return self._client.source_slot_id
        _routable_worker_host()
        specs = _build_specs(request)
        registration = self._registration
        self._client = ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                model_name=registration.model_name,
                device_id=torch.cuda.current_device(),
                staging_mode=TrainerStagingMode.IN_PLACE,
                payload_format=WeightPayloadFormat.FULL_TENSOR,
                engine_context=MegatronTrainerContext(),
            )
        )
        try:
            slot = self._client.bind_tensors(specs)
        except BaseException:
            self._client.close()
            self._client = None
            raise
        self._specs = specs
        self._signature = signature
        return slot

    def create_version(self, *, source_slots, step, update_id):
        if self._control is None:
            self._control = ModelExpressControlClient.connect()
        version = self._control.create_weight_version(
            model_name=self._registration.model_name,
            idempotency_key=update_id,
            payload_format=WeightPayloadFormat.FULL_TENSOR,
            expected_source_slots=list(source_slots),
        )
        return version.version_id

    def publish_and_execute(self, request):
        self.prepare(request.tensors)
        if request.cohort_id != self._registration.cohort_id:
            raise RuntimeError("Miles update cohort changed after selection")
        torch.cuda.synchronize()
        self._client.publish_version(version=WeightVersionRef(request.version))

    def mark_ready(self, version_id):
        self._control.update_weight_version_state(version_id, WeightVersionState.READY)

    def retire(self, version_id):
        self._control.delete_weight_version(version_id)

    def release(self, version_id):
        if self._client is not None:
            self._client.release_version(version=WeightVersionRef(version_id))

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None
        if self._control is not None:
            self._control.close()
            self._control = None


def create_miles_publisher():
    return MilesModelExpressPublisher()
