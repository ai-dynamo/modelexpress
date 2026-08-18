# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from modelexpress.refit.reshard.megatron_aliases import (
    MegatronAliasInput,
    build_hf_aliases,
)

HF_NAMES = ("q_proj.weight", "k_proj.weight", "v_proj.weight")


def _global_fused(q_heads: int, kv_heads: int, head_dim: int, hidden: int):
    rows = (q_heads + 2 * kv_heads) * head_dim
    return torch.arange(rows * hidden, dtype=torch.float32).reshape(rows, hidden)


def _global_extras(q_heads: int, kv_heads: int, head_dim: int):
    return {
        "qkv_interleave": "by_head",
        "num_heads": str(q_heads),
        "num_kv_heads": str(kv_heads),
        "head_dim": str(head_dim),
    }


def _publish_all_ranks(
    q_heads: int, kv_heads: int, tp_size: int, head_dim: int, hidden: int = 3
):
    fused = _global_fused(q_heads, kv_heads, head_dim, hidden)
    assert fused.shape[0] % tp_size == 0
    local_rows = fused.shape[0] // tp_size
    published = []
    locals_by_agent = {}
    for rank in range(tp_size):
        lo = rank * local_rows
        hi = lo + local_rows
        local = fused[lo:hi].clone()
        agent = f"tp{rank}"
        locals_by_agent[agent] = local
        published.extend(
            build_hf_aliases(
                [
                    MegatronAliasInput(
                        name="linear_qkv.weight",
                        tensor=local,
                        role="qkv_column",
                        hf_names=HF_NAMES,
                        global_shape=tuple(fused.shape),
                        placement_kind="SHARD" if tp_size > 1 else "REPLICATE",
                        shard_axis=0 if tp_size > 1 else None,
                        local_shard_range=(lo, hi) if tp_size > 1 else None,
                        extras=_global_extras(q_heads, kv_heads, head_dim),
                    )
                ],
                agent_name=agent,
            )
        )
    return fused, locals_by_agent, published


def _expected_hf(fused: torch.Tensor, q_heads: int, kv_heads: int, head_dim: int):
    q_per_group = q_heads // kv_heads
    q_rows = q_per_group * head_dim
    group_rows = q_rows + 2 * head_dim
    q, k, v = [], [], []
    for group in range(kv_heads):
        group_lo = group * group_rows
        q.append(fused[group_lo : group_lo + q_rows])
        k.append(fused[group_lo + q_rows : group_lo + q_rows + head_dim])
        v.append(fused[group_lo + q_rows + head_dim : group_lo + group_rows])
    return tuple(torch.cat(parts) for parts in (q, k, v))


def _reconstruct(published, locals_by_agent):
    by_name = {name: [] for name in HF_NAMES}
    full_shapes = {}
    for tensor in published:
        by_name[tensor.name].extend(tensor.shards)
        full_shapes[tensor.name] = tensor.full_shape

    actual = []
    for name in HF_NAMES:
        assert name in full_shapes
        destination = torch.empty(full_shapes[name], dtype=torch.float32)
        coverage = torch.zeros(full_shapes[name][0], dtype=torch.int32)
        for shard in by_name[name]:
            local = locals_by_agent[shard.agent_name]
            row_bytes = local.shape[1] * local.element_size()
            byte_offset = shard.addr - local.data_ptr()
            assert byte_offset % row_bytes == 0
            source_lo = byte_offset // row_bytes
            rows = shard.shape[0]
            destination_lo = shard.shard_offset[0]
            destination[destination_lo : destination_lo + rows].copy_(
                local[source_lo : source_lo + rows]
            )
            coverage[destination_lo : destination_lo + rows] += 1
        assert torch.all(coverage == 1), (name, coverage)
        actual.append(destination)

    for agent, local in locals_by_agent.items():
        source_coverage = torch.zeros(local.shape[0], dtype=torch.int32)
        for shards in by_name.values():
            for shard in shards:
                if shard.agent_name != agent:
                    continue
                row_bytes = local.shape[1] * local.element_size()
                source_lo = (shard.addr - local.data_ptr()) // row_bytes
                source_coverage[source_lo : source_lo + shard.shape[0]] += 1
        assert torch.all(source_coverage == 1), (agent, source_coverage)
    return tuple(actual)


@pytest.mark.parametrize(
    ("q_heads", "kv_heads", "tp_size", "head_dim"),
    [
        (32, 4, 2, 4),  # Topology A's divisible TP geometry.
        (32, 4, 1, 4),  # Topology B's unsharded dense geometry.
        (64, 2, 8, 128),  # Nemotron Ultra: KV heads below trainer TP.
        (24, 6, 4, 2),  # KV heads exceed TP but are not divisible by it.
    ],
)
def test_global_interval_aliases_cover_qkv_without_gaps_or_overlaps(
    q_heads: int, kv_heads: int, tp_size: int, head_dim: int
):
    fused, locals_by_agent, published = _publish_all_ranks(
        q_heads, kv_heads, tp_size, head_dim
    )

    actual = _reconstruct(published, locals_by_agent)

    expected = _expected_hf(fused, q_heads, kv_heads, head_dim)
    assert all(
        torch.equal(got, want) for got, want in zip(actual, expected, strict=True)
    )


def test_kv_below_tp_only_advertises_kv_on_ranks_that_own_it():
    _, _, published = _publish_all_ranks(64, 2, 8, 128)
    names_by_agent = {}
    for tensor in published:
        for shard in tensor.shards:
            names_by_agent.setdefault(shard.agent_name, set()).add(tensor.name)

    assert names_by_agent["tp0"] == {"q_proj.weight"}
    assert names_by_agent["tp3"] == set(HF_NAMES)
    assert names_by_agent["tp4"] == {"q_proj.weight"}
    assert names_by_agent["tp7"] == set(HF_NAMES)


def test_two_layers_may_use_different_qkv_geometry():
    fixtures = [(64, 2, 8, 128), (32, 8, 8, 64)]
    for q_heads, kv_heads, tp_size, head_dim in fixtures:
        fused, locals_by_agent, published = _publish_all_ranks(
            q_heads, kv_heads, tp_size, head_dim
        )
        actual = _reconstruct(published, locals_by_agent)
        expected = _expected_hf(fused, q_heads, kv_heads, head_dim)
        assert all(
            torch.equal(got, want) for got, want in zip(actual, expected, strict=True)
        )


def test_divisible_global_descriptors_match_legacy_aliases_byte_for_byte():
    q_heads, kv_heads, tp_size, head_dim, hidden = 32, 4, 2, 4, 3
    fused = _global_fused(q_heads, kv_heads, head_dim, hidden)
    local_rows = fused.shape[0] // tp_size
    for rank in range(tp_size):
        lo = rank * local_rows
        hi = lo + local_rows
        local = fused[lo:hi].clone()
        common = {
            "name": "linear_qkv.weight",
            "tensor": local,
            "role": "qkv_column",
            "hf_names": HF_NAMES,
            "global_shape": tuple(fused.shape),
            "placement_kind": "SHARD",
            "shard_axis": 0,
            "local_shard_range": (lo, hi),
        }
        legacy = build_hf_aliases(
            [
                MegatronAliasInput(
                    **common,
                    extras={
                        "num_heads_local": str(q_heads // tp_size),
                        "num_kv_heads_local": str(kv_heads // tp_size),
                        "head_dim": str(head_dim),
                    },
                )
            ],
            agent_name=f"tp{rank}",
        )
        global_aliases = build_hf_aliases(
            [
                MegatronAliasInput(
                    **common,
                    extras=_global_extras(q_heads, kv_heads, head_dim),
                )
            ],
            agent_name=f"tp{rank}",
        )
        assert global_aliases == legacy


@pytest.mark.parametrize(
    ("extras", "global_rows", "match"),
    [
        (
            {"num_heads": "64", "head_dim": "128", "qkv_interleave": "by_head"},
            8704,
            "requires both",
        ),
        (_global_extras(64, 2, 128), 8192, "rows disagree"),
        (
            {**_global_extras(64, 2, 128), "qkv_interleave": "unsupported"},
            8704,
            "qkv_interleave",
        ),
    ],
)
def test_unrecoverable_global_geometry_fails_closed(extras, global_rows, match):
    local = torch.zeros(global_rows // 8, 3)
    with pytest.raises(ValueError, match=match):
        build_hf_aliases(
            [
                MegatronAliasInput(
                    name="linear_qkv.weight",
                    tensor=local,
                    role="qkv_column",
                    hf_names=HF_NAMES,
                    global_shape=(global_rows, 3),
                    placement_kind="SHARD",
                    shard_axis=0,
                    local_shard_range=(0, local.shape[0]),
                    extras=extras,
                )
            ],
            agent_name="tp0",
        )
