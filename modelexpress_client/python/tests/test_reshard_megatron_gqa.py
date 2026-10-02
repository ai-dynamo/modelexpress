# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from modelexpress.refit.reshard.rendezvous import build_sources, merge_shard_tables
from modelexpress.refit.reshard.transfer_plan import plan_transfer
from modelexpress.refit.reshard.types import CaptureResult, RecordedCopy
from modelexpress_rl.train.engines.megatron import (
    MegatronTensorSpec,
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
                    MegatronTensorSpec(
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
        sample = next(iter(locals_by_agent.values()))
        destination = torch.empty(
            full_shapes[name], dtype=sample.dtype, device=sample.device
        )
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


def _tables_by_agent(published):
    tables = {}
    for tensor in published:
        assert tensor.shards
        agent = tensor.shards[0].agent_name
        assert all(shard.agent_name == agent for shard in tensor.shards)
        tables.setdefault(agent, []).append(tensor)
    return [tables[name] for name in sorted(tables)]


def _full_copy(name: str, shape: tuple[int, int]) -> RecordedCopy:
    return RecordedCopy(
        src_name=name,
        op_chain=(),
        param_name=name,
        dest_offset=0,
        dest_shape=shape,
        dest_stride=(shape[1], 1),
        dest_dtype=torch.float32,
    )


@pytest.mark.parametrize(
    ("q_heads", "kv_heads", "tp_size", "head_dim"),
    [
        (32, 4, 2, 4),
        (32, 4, 1, 4),
        (64, 2, 8, 128),
        (24, 6, 4, 2),
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


def test_global_qkv_bias_aliases_preserve_interleaved_head_order():
    q_heads, kv_heads, head_dim = 4, 2, 2
    fused = torch.arange(16, dtype=torch.float32)
    aliases = build_hf_aliases(
        [
            MegatronTensorSpec(
                name="linear_qkv.bias",
                tensor=fused,
                role="qkv_column",
                hf_names=("q_proj.bias", "k_proj.bias", "v_proj.bias"),
                global_shape=tuple(fused.shape),
                placement_kind="REPLICATE",
                shard_axis=None,
                local_shard_range=None,
                extras=_global_extras(q_heads, kv_heads, head_dim),
            )
        ],
        agent_name="trainer-tp0",
    )

    actual = []
    for alias in aliases:
        destination = torch.empty(alias.full_shape, dtype=fused.dtype)
        for shard in alias.shards:
            source_start = (shard.addr - fused.data_ptr()) // fused.element_size()
            destination_start = shard.shard_offset[0]
            destination.narrow(0, destination_start, shard.shape[0]).copy_(
                fused.narrow(0, source_start, shard.shape[0])
            )
        actual.append(destination)

    expected = _expected_hf(fused, q_heads, kv_heads, head_dim)
    assert [alias.full_shape for alias in aliases] == [(8,), (4,), (4,)]
    assert all(
        torch.equal(got, want) for got, want in zip(actual, expected, strict=True)
    )


def test_kv_below_tp_only_advertises_kv_on_ranks_that_own_it():
    _, _, published = _publish_all_ranks(64, 2, 8, 128)
    names_by_agent = {f"tp{rank}": set() for rank in range(8)}
    for tensor in published:
        for shard in tensor.shards:
            names_by_agent[shard.agent_name].add(tensor.name)

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
                MegatronTensorSpec(
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
                MegatronTensorSpec(
                    **common,
                    extras=_global_extras(q_heads, kv_heads, head_dim),
                )
            ],
            agent_name=f"tp{rank}",
        )
        assert global_aliases == legacy


def test_sparse_kv_tables_merge_into_a_complete_bounded_plan():
    _, _, published = _publish_all_ranks(64, 2, 8, 128)
    merged = merge_shard_tables(_tables_by_agent(published))
    sources, _, _ = build_sources(merged)
    capture = CaptureResult(
        copies=[
            _full_copy(name, tuple(sources[name].global_shape)) for name in HF_NAMES
        ]
    )

    plan = plan_transfer(capture, sources, max_segments_per_copy=1)

    assert plan.fallback == []
    assert {pull.src_name for pull in plan.full_pulls} == set(HF_NAMES)
    assert plan.bytes_planned() == sum(
        tensor.numel() * tensor.element_size()
        for tensor in _expected_hf(_global_fused(64, 2, 128, 3), 64, 2, 128)
    )


def test_missing_kv_publishers_fail_closed_instead_of_silently_falling_back():
    _, _, published = _publish_all_ranks(64, 2, 8, 128)
    q_only = [tensor for tensor in published if tensor.name == HF_NAMES[0]]
    sources, _, _ = build_sources(merge_shard_tables(_tables_by_agent(q_only)))
    capture = CaptureResult(
        copies=[
            _full_copy(HF_NAMES[0], (64 * 128, 3)),
            _full_copy(HF_NAMES[1], (2 * 128, 3)),
            _full_copy(HF_NAMES[2], (2 * 128, 3)),
        ]
    )

    plan = plan_transfer(capture, sources)

    assert set(plan.fallback) == {HF_NAMES[1], HF_NAMES[2]}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA tensors")
def test_cuda_64q_2kv_tp8_noop_and_moving_weight_refits_match_logits():
    device = torch.device("cuda", 0)
    q_heads, kv_heads, tp_size, head_dim, hidden = 64, 2, 8, 128, 64
    rows = (q_heads + 2 * kv_heads) * head_dim
    generator = torch.Generator(device=device).manual_seed(20260818)
    base = torch.randn(
        (rows, hidden),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    activation = torch.randn(
        (4, hidden),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )

    def refit(fused):
        local_rows = fused.shape[0] // tp_size
        published = []
        locals_by_agent = {}
        for rank in range(tp_size):
            lo = rank * local_rows
            hi = lo + local_rows
            agent = f"tp{rank}"
            local = fused[lo:hi].clone()
            locals_by_agent[agent] = local
            published.extend(
                build_hf_aliases(
                    [
                        MegatronTensorSpec(
                            name="linear_qkv.weight",
                            tensor=local,
                            role="qkv_column",
                            hf_names=HF_NAMES,
                            global_shape=tuple(fused.shape),
                            placement_kind="SHARD",
                            shard_axis=0,
                            local_shard_range=(lo, hi),
                            extras=_global_extras(q_heads, kv_heads, head_dim),
                        )
                    ],
                    agent_name=agent,
                )
            )
        actual = _reconstruct(published, locals_by_agent)
        expected = _expected_hf(fused, q_heads, kv_heads, head_dim)
        assert all(
            torch.equal(got, want)
            for got, want in zip(actual, expected, strict=True)
        )
        actual_logits = tuple(activation @ weight.T for weight in actual)
        expected_logits = tuple(activation @ weight.T for weight in expected)
        assert all(
            torch.equal(got, want)
            for got, want in zip(actual_logits, expected_logits, strict=True)
        )
        return actual

    no_op = refit(base)
    moving = refit(base + torch.tensor(0.125, dtype=base.dtype, device=device))

    assert any(
        not torch.equal(before, after)
        for before, after in zip(no_op, moving, strict=True)
    )


@pytest.mark.parametrize(
    ("extras", "global_rows", "match"),
    [
        (
            {"num_heads": "64", "head_dim": "128", "qkv_interleave": "by_head"},
            8704,
            "requires both",
        ),
        (
            {
                "num_heads": "64",
                "num_kv_heads": "2",
                "qkv_interleave": "by_head",
            },
            8704,
            "head_dim",
        ),
        (_global_extras(64, 2, 128), 8192, "rows disagree"),
        (
            {**_global_extras(64, 2, 128), "qkv_interleave": "unsupported"},
            8704,
            "qkv_interleave",
        ),
        (_global_extras(63, 2, 128), 8576, "invalid global"),
    ],
)
def test_unrecoverable_global_geometry_fails_closed(extras, global_rows, match):
    assert global_rows % 8 == 0
    local = torch.zeros(global_rows // 8, 3)
    with pytest.raises(ValueError, match=match):
        build_hf_aliases(
            [
                MegatronTensorSpec(
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
