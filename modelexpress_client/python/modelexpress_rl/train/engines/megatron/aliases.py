# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expose native Megatron storage as HF-canonical RL refit source shards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modelexpress.refit.reshard.rendezvous import PublishedShard, PublishedTensor
from modelexpress.refit.reshard.verify import published_digest


@dataclass(frozen=True)
class MegatronTensorSpec:
    """One native Megatron tensor and its logical transfer representation."""

    name: str
    tensor: Any
    role: str
    hf_names: tuple[str, ...]
    global_shape: tuple[int, ...]
    placement_kind: str
    shard_axis: int | None
    local_shard_range: tuple[int, int] | None
    extras: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.hf_names:
            raise ValueError("name and hf_names are required")
        if len(set(self.hf_names)) != len(self.hf_names):
            raise ValueError(f"{self.name}: hf_names must be unique")
        if not self.global_shape or any(int(dim) <= 0 for dim in self.global_shape):
            raise ValueError(f"{self.name}: invalid global shape {self.global_shape}")
        if self.placement_kind not in {"SHARD", "REPLICATE"}:
            raise ValueError(f"{self.name}: unsupported placement_kind")
        has_shard_layout = (
            self.shard_axis is not None and self.local_shard_range is not None
        )
        if self.placement_kind == "SHARD" and not has_shard_layout:
            raise ValueError(
                f"{self.name}: SHARD requires shard_axis and local_shard_range"
            )
        if self.placement_kind == "REPLICATE" and (
            self.shard_axis is not None or self.local_shard_range is not None
        ):
            raise ValueError(f"{self.name}: REPLICATE cannot set shard layout")


def _source_rank_and_size(item: MegatronTensorSpec, axis: int) -> tuple[int, int]:
    local_extent = int(item.tensor.shape[axis])
    global_extent = int(item.global_shape[axis])
    if item.placement_kind != "SHARD":
        return 0, 1
    if item.local_shard_range is None:
        raise ValueError(f"{item.name}: SHARD has no local range")
    lo, hi = (int(value) for value in item.local_shard_range)
    # A range can pass every check below and still lie outside the tensor it
    # claims part of: (16, 24) against a global extent of 16 has the right width,
    # divides evenly, and yields source rank 2 of a 2-rank group. The alias that
    # follows would then address bytes the full tensor does not have.
    if not 0 <= lo < hi <= global_extent:
        raise ValueError(
            f"{item.name}: source shard range {(lo, hi)} is outside the global "
            f"extent {global_extent} on axis {axis}"
        )
    if hi - lo != local_extent or global_extent % local_extent:
        raise ValueError(f"{item.name}: inconsistent source shard geometry")
    if lo % local_extent:
        raise ValueError(f"{item.name}: non-uniform source shard is unsupported")
    return lo // local_extent, global_extent // local_extent


def _one_shard(
    *,
    name: str,
    tensor: Any,
    full_shape: tuple[int, ...],
    agent_name: str,
    shard_axis: int | None,
    shard_range: tuple[int, int] | None,
) -> PublishedTensor:
    local_shape = tuple(int(dim) for dim in tensor.shape)
    offset = [0] * len(local_shape)
    if shard_axis is not None:
        if shard_range is None:
            raise ValueError(f"{name}: shard axis has no range")
        lo, hi = shard_range
        if hi - lo != local_shape[shard_axis]:
            raise ValueError(f"{name}: shard range does not match local shape")
        offset[shard_axis] = lo
    elif local_shape != full_shape:
        raise ValueError(f"{name}: replicated shape mismatch")
    return PublishedTensor(
        name=name,
        dtype=str(tensor.dtype),
        elsize=int(tensor.element_size()),
        full_shape=full_shape,
        shards=[
            PublishedShard(
                agent_name=agent_name,
                device_id=int(tensor.device.index or 0),
                addr=int(tensor.data_ptr()),
                shard_offset=tuple(offset),
                shape=local_shape,
                digest=published_digest(tensor),
            )
        ],
    )


GATE_THEN_UP = "gate_then_up"


def _build_gated_aliases(
    item: MegatronTensorSpec, agent_name: str
) -> list[PublishedTensor]:
    if len(item.hf_names) != 2:
        raise ValueError(f"{item.name}: gated tensor requires gate/up HF names")
    # The halves are assigned to hf_names positionally, so the storage order is
    # what decides which HF tensor each half becomes. Getting it wrong publishes
    # the gate projection's bytes under the up projection's name, which no digest
    # gate can see: both names receive the bytes their publisher advertised. The
    # order is therefore required rather than assumed.
    order = item.extras.get("gated_mlp_order")
    if order != GATE_THEN_UP:
        raise ValueError(
            f"{item.name}: fused gate/up aliasing requires extras"
            f"['gated_mlp_order'] == {GATE_THEN_UP!r}, got {order!r}"
        )
    axis = int(item.shard_axis if item.shard_axis is not None else 0)
    local_extent = int(item.tensor.shape[axis])
    if local_extent % 2:
        raise ValueError(f"{item.name}: fused gate/up extent must be even")
    half = local_extent // 2
    gate = item.tensor.narrow(axis, 0, half)
    up = item.tensor.narrow(axis, half, half)
    if not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError(
            f"{item.name}: fused gate/up aliases are not contiguous on axis {axis}"
        )
    source_rank, source_size = _source_rank_and_size(item, axis)
    full_shape = list(item.tensor.shape)
    full_shape[axis] = half * source_size
    expected = [int(dim) for dim in item.global_shape]
    if expected[axis] % 2:
        raise ValueError(f"{item.name}: fused global extent must be even")
    expected[axis] //= 2
    if expected != [int(dim) for dim in full_shape]:
        raise ValueError(
            f"{item.name}: derived gate/up shape {tuple(full_shape)} disagrees "
            f"with declared global shape {item.global_shape}"
        )
    shard_range = (
        (source_rank * half, (source_rank + 1) * half) if source_size > 1 else None
    )
    return [
        _one_shard(
            name=hf_name,
            tensor=tensor,
            full_shape=tuple(int(dim) for dim in full_shape),
            agent_name=agent_name,
            shard_axis=axis if shard_range is not None else None,
            shard_range=shard_range,
        )
        for hf_name, tensor in zip(item.hf_names, (gate, up), strict=True)
    ]


def _build_qkv_aliases(
    item: MegatronTensorSpec, agent_name: str
) -> list[PublishedTensor]:
    if len(item.hf_names) != 3 or item.tensor.ndim != 2:
        raise ValueError(f"{item.name}: QKV aliasing requires 2D q/k/v weights")
    required = ("head_dim", "num_heads_local", "num_kv_heads_local")
    missing = [key for key in required if key not in item.extras]
    if missing:
        raise ValueError(f"{item.name}: QKV aliasing requires extras {missing}")
    head_dim = int(item.extras["head_dim"])
    q_heads_local = int(item.extras["num_heads_local"])
    kv_heads_local = int(item.extras["num_kv_heads_local"])
    if kv_heads_local < 1 or q_heads_local % kv_heads_local:
        raise ValueError(f"{item.name}: invalid local Q/KV head geometry")
    rows_per_group = (q_heads_local // kv_heads_local + 2) * head_dim
    if rows_per_group * kv_heads_local != int(item.tensor.shape[0]):
        raise ValueError(f"{item.name}: QKV rows disagree with head metadata")
    source_rank, source_size = _source_rank_and_size(item, 0)
    hidden = int(item.tensor.shape[1])
    q_heads_per_group = q_heads_local // kv_heads_local
    q_shards = []
    k_shards = []
    v_shards = []
    for local_group in range(kv_heads_local):
        group = item.tensor.narrow(0, local_group * rows_per_group, rows_per_group)
        q_rows = q_heads_per_group * head_dim
        q = group.narrow(0, 0, q_rows)
        k = group.narrow(0, q_rows, head_dim)
        v = group.narrow(0, q_rows + head_dim, head_dim)
        global_group = source_rank * kv_heads_local + local_group
        for tensor, shards, start in (
            (q, q_shards, global_group * q_rows),
            (k, k_shards, global_group * head_dim),
            (v, v_shards, global_group * head_dim),
        ):
            shards.append(
                PublishedShard(
                    agent_name=agent_name,
                    device_id=int(tensor.device.index or 0),
                    addr=int(tensor.data_ptr()),
                    shard_offset=(start, 0),
                    shape=tuple(int(dim) for dim in tensor.shape),
                    # The narrow, not the fused parent: this is the box a receiver
                    # reads from ``addr``, so it is the box whose bytes must match.
                    digest=published_digest(tensor),
                )
            )
    return [
        PublishedTensor(
            name=name,
            dtype=str(item.tensor.dtype),
            elsize=int(item.tensor.element_size()),
            full_shape=(rows, hidden),
            shards=shards,
        )
        for name, rows, shards in (
            (item.hf_names[0], q_heads_local * source_size * head_dim, q_shards),
            (item.hf_names[1], kv_heads_local * source_size * head_dim, k_shards),
            (item.hf_names[2], kv_heads_local * source_size * head_dim, v_shards),
        )
    ]


def build_hf_aliases(
    items: list[MegatronTensorSpec], *, agent_name: str
) -> list[PublishedTensor]:
    """Build zero-copy HF aliases whose addresses remain in registered storage."""

    aliases = []
    for item in items:
        # Every alias published below is a base address plus a shape, which tells
        # a reader the bytes run contiguously from that address. A strided view
        # satisfies neither, and nothing downstream can detect it: the read simply
        # lands on whatever sits between the elements the view meant to select.
        if not item.tensor.is_contiguous():
            raise ValueError(
                f"{item.name}: aliasing publishes an address and a shape, which "
                f"requires contiguous storage; got a non-contiguous tensor of "
                f"shape {tuple(int(dim) for dim in item.tensor.shape)}"
            )
        if item.role == "qkv_column":
            aliases.extend(_build_qkv_aliases(item, agent_name))
            continue
        if (
            item.role in {"gated_mlp_column", "expert_column"}
            and len(item.hf_names) == 2
        ):
            aliases.extend(_build_gated_aliases(item, agent_name))
            continue
        if len(item.hf_names) != 1:
            raise ValueError(
                f"{item.name}: role {item.role!r} cannot map to "
                f"{len(item.hf_names)} HF tensors"
            )
        aliases.append(
            _one_shard(
                name=item.hf_names[0],
                tensor=item.tensor,
                full_shape=tuple(item.global_shape),
                agent_name=agent_name,
                shard_axis=(
                    int(item.shard_axis) if item.placement_kind == "SHARD" else None
                ),
                shard_range=(
                    tuple(item.local_shard_range)
                    if item.placement_kind == "SHARD"
                    and item.local_shard_range is not None
                    else None
                ),
            )
        )
    return aliases


# Compatibility name for the existing reshard API.
MegatronAliasInput = MegatronTensorSpec


__all__ = ["MegatronAliasInput", "MegatronTensorSpec", "build_hf_aliases"]
