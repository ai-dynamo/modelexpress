# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expose native Megatron storage as HF-canonical reshard source shards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from modelexpress.refit.reshard.rendezvous import PublishedShard, PublishedTensor
from modelexpress.refit.reshard.verify import published_digest


@dataclass(frozen=True)
class MegatronAliasInput:
    name: str
    tensor: Any
    role: str
    hf_names: tuple[str, ...]
    global_shape: tuple[int, ...]
    placement_kind: str
    shard_axis: int | None
    local_shard_range: tuple[int, int] | None
    extras: dict[str, str] = field(default_factory=dict)


def _source_rank_and_size(item: MegatronAliasInput, axis: int) -> tuple[int, int]:
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
    item: MegatronAliasInput, agent_name: str
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
    item: MegatronAliasInput, agent_name: str
) -> list[PublishedTensor]:
    if len(item.hf_names) != 3 or item.tensor.ndim != 2:
        raise ValueError(f"{item.name}: QKV aliasing requires 2D q/k/v weights")
    has_global_q = "num_heads" in item.extras
    has_global_kv = "num_kv_heads" in item.extras
    if has_global_q != has_global_kv:
        raise ValueError(
            f"{item.name}: global QKV metadata requires both num_heads and num_kv_heads"
        )
    if has_global_q:
        return _build_global_qkv_aliases(item, agent_name)
    return _build_legacy_qkv_aliases(item, agent_name)


def _qkv_source_interval(item: MegatronAliasInput) -> tuple[int, int]:
    """Return this rank's row interval in the global fused QKV tensor."""
    local_rows = int(item.tensor.shape[0])
    global_rows = int(item.global_shape[0])
    if item.placement_kind != "SHARD":
        if local_rows != global_rows:
            raise ValueError(f"{item.name}: replicated QKV shape mismatch")
        return 0, global_rows
    if item.shard_axis != 0 or item.local_shard_range is None:
        raise ValueError(f"{item.name}: QKV shards must carry a row range")
    lo, hi = (int(value) for value in item.local_shard_range)
    if not 0 <= lo < hi <= global_rows or hi - lo != local_rows:
        raise ValueError(f"{item.name}: inconsistent QKV source row interval")
    return lo, hi


def _build_global_qkv_aliases(
    item: MegatronAliasInput, agent_name: str
) -> list[PublishedTensor]:
    """Map a raw TP row interval through Megatron's global interleaved layout."""
    if item.extras.get("qkv_interleave") != "by_head":
        raise ValueError(
            f"{item.name}: global QKV aliasing requires qkv_interleave='by_head'"
        )
    head_dim = int(item.extras["head_dim"])
    q_heads = int(item.extras["num_heads"])
    kv_heads = int(item.extras["num_kv_heads"])
    if head_dim < 1 or q_heads < 1 or kv_heads < 1 or q_heads % kv_heads:
        raise ValueError(f"{item.name}: invalid global Q/KV head geometry")

    hidden = int(item.tensor.shape[1])
    if len(item.global_shape) != 2 or int(item.global_shape[1]) != hidden:
        raise ValueError(f"{item.name}: QKV hidden dimension mismatch")
    q_heads_per_group = q_heads // kv_heads
    q_rows_per_group = q_heads_per_group * head_dim
    group_rows = q_rows_per_group + 2 * head_dim
    expected_global_rows = kv_heads * group_rows
    if int(item.global_shape[0]) != expected_global_rows:
        raise ValueError(f"{item.name}: global QKV rows disagree with head metadata")

    source_lo, source_hi = _qkv_source_interval(item)
    shards: list[list[PublishedShard]] = [[], [], []]
    mapped_rows = 0
    for group in range(kv_heads):
        group_lo = group * group_rows
        parts = (
            (group_lo, q_rows_per_group, group * q_rows_per_group),
            (group_lo + q_rows_per_group, head_dim, group * head_dim),
            (group_lo + q_rows_per_group + head_dim, head_dim, group * head_dim),
        )
        for part_index, (part_lo, part_rows, destination_lo) in enumerate(parts):
            part_hi = part_lo + part_rows
            overlap_lo = max(source_lo, part_lo)
            overlap_hi = min(source_hi, part_hi)
            if overlap_lo >= overlap_hi:
                continue
            rows = overlap_hi - overlap_lo
            tensor = item.tensor.narrow(0, overlap_lo - source_lo, rows)
            shards[part_index].append(
                PublishedShard(
                    agent_name=agent_name,
                    device_id=int(tensor.device.index or 0),
                    addr=int(tensor.data_ptr()),
                    shard_offset=(destination_lo + overlap_lo - part_lo, 0),
                    shape=tuple(int(dim) for dim in tensor.shape),
                    digest=published_digest(tensor),
                )
            )
            mapped_rows += rows

    if mapped_rows != int(item.tensor.shape[0]):
        raise ValueError(
            f"{item.name}: QKV interval mapping covered {mapped_rows} of "
            f"{int(item.tensor.shape[0])} source rows"
        )

    aliases = []
    for name, rows, tensor_shards in zip(
        item.hf_names,
        (q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim),
        shards,
        strict=True,
    ):
        # With KV heads below TP, most ranks legitimately own no K or V rows.
        # Do not advertise an empty source tensor; other ranks publish those
        # destination intervals during rendezvous.
        if tensor_shards:
            aliases.append(
                PublishedTensor(
                    name=name,
                    dtype=str(item.tensor.dtype),
                    elsize=int(item.tensor.element_size()),
                    full_shape=(rows, hidden),
                    shards=tensor_shards,
                )
            )
    return aliases


def _build_legacy_qkv_aliases(
    item: MegatronAliasInput, agent_name: str
) -> list[PublishedTensor]:
    """Compatibility path for divisible descriptors with local head counts."""
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
    items: list[MegatronAliasInput], *, agent_name: str
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


__all__ = ["MegatronAliasInput", "build_hf_aliases"]
