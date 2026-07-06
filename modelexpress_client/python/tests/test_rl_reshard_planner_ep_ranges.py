# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity: ranges-only EP coverage == the deprecated expert shim.

The core planner is meant to be range-only — an MoE expert shard is just a
``SHARD`` range on the expert axis, and ordinary range intersection produces
the correct EP coverage without the ``is_expert`` / ``owned_expert_ids`` /
``required_experts`` special-case. These tests prove that: for every common EP
placement, the plan built from **plain ranges** (via
:func:`expert_ids_to_contiguous_ranges`, no expert fields set) covers exactly
the same experts from exactly the same source ranks as the plan built with the
legacy shim. This is the evidence that lets the shim be removed later without
changing behavior.
"""

from __future__ import annotations

from modelexpress.rl_expert_layout import (
    compute_local_expert_ids,
    expert_ids_to_contiguous_ranges,
)
from modelexpress.rl_reshard_planner import plan_coverage
from modelexpress.rl_slice_descriptors import SliceOwnership, SliceRequest

_TENSOR = "model.layers.0.experts.w13_weight"
_NE = 128
_EXPERT_BYTES = 4096 * 2048 * 2  # per-expert bf16 footprint


# ---- builders: legacy shim vs ranges-only -------------------------------


def _shim_source(rank: int, owned: tuple[int, ...], ne: int = _NE) -> SliceOwnership:
    lo, hi = owned[0], owned[-1] + 1
    return SliceOwnership(
        model_name="m", tensor_name=_TENSOR, global_shape=(ne, 4096, 2048),
        dtype="torch.bfloat16", placement_kind="SHARD", shard_axis=0,
        local_shard_range=(lo, hi), worker_rank=rank,
        nixl_addr=0x1000 + rank, byte_size=len(owned) * _EXPERT_BYTES,
        is_expert=True, expert_axis=0, owned_expert_ids=owned,
    )


def _shim_request(rank: int, required: frozenset[int]) -> list[SliceRequest]:
    # one request per contiguous run (matches how callers express non-linear EP)
    reqs = []
    for lo, hi in expert_ids_to_contiguous_ranges(required):
        reqs.append(SliceRequest(
            tensor_name=_TENSOR, global_range=(lo, hi), shard_axis=0,
            dtype="torch.bfloat16", receiver_rank=rank, target_addr=0xF000,
            required_experts=required,
        ))
    return reqs


def _range_source(rank: int, owned: tuple[int, ...], ne: int = _NE) -> list[SliceOwnership]:
    # ranges-only: one SHARD entry per contiguous run, NO expert fields
    out = []
    for lo, hi in expert_ids_to_contiguous_ranges(owned):
        out.append(SliceOwnership(
            model_name="m", tensor_name=_TENSOR, global_shape=(ne, 4096, 2048),
            dtype="torch.bfloat16", placement_kind="SHARD", shard_axis=0,
            local_shard_range=(lo, hi), worker_rank=rank,
            nixl_addr=0x1000 + rank, byte_size=(hi - lo) * _EXPERT_BYTES,
        ))
    return out


def _range_request(rank: int, required: frozenset[int]) -> list[SliceRequest]:
    # ranges-only: one request per contiguous run, NO required_experts
    reqs = []
    for lo, hi in expert_ids_to_contiguous_ranges(required):
        reqs.append(SliceRequest(
            tensor_name=_TENSOR, global_range=(lo, hi), shard_axis=0,
            dtype="torch.bfloat16", receiver_rank=rank, target_addr=0xF000,
        ))
    return reqs


def _coverage(plan) -> tuple[set[int], set[int]]:
    """(covered expert ids, source ranks used) from a plan's segments."""
    experts: set[int] = set()
    ranks: set[int] = set()
    for seg in plan.segments:
        req_lo = seg.request.global_range[0]
        t_lo, t_hi = seg.target_range
        experts.update(range(req_lo + t_lo, req_lo + t_hi))
        ranks.add(seg.source.worker_rank)
    return experts, ranks


def _assert_parity(sources_shim, requests_shim, sources_rng, requests_rng) -> None:
    plan_shim = plan_coverage(sources_shim, requests_shim)
    plan_rng = plan_coverage(sources_rng, requests_rng)
    assert plan_shim.complete and plan_rng.complete
    assert _coverage(plan_shim) == _coverage(plan_rng)


# ---- parity across placements -------------------------------------------


def test_parity_linear_ep8():
    for rr in range(8):
        owned = {r: compute_local_expert_ids(r, 8, _NE, "linear") for r in range(8)}
        req_ids = frozenset(compute_local_expert_ids(rr, 8, _NE, "linear"))
        shim_src = [_shim_source(r, owned[r]) for r in range(8)]
        rng_src = [s for r in range(8) for s in _range_source(r, owned[r])]
        _assert_parity(shim_src, _shim_request(rr, req_ids),
                       rng_src, _range_request(rr, req_ids))


def test_parity_mixed_ep_trainer8_infer4():
    owned = {r: compute_local_expert_ids(r, 8, _NE, "linear") for r in range(8)}
    shim_src = [_shim_source(r, owned[r]) for r in range(8)]
    rng_src = [s for r in range(8) for s in _range_source(r, owned[r])]
    for rr in range(4):
        req_ids = frozenset(compute_local_expert_ids(rr, 4, _NE, "linear"))
        _assert_parity(shim_src, _shim_request(rr, req_ids),
                       rng_src, _range_request(rr, req_ids))


def test_parity_round_robin_ep4_noncontiguous():
    ne = 16
    owned = {r: compute_local_expert_ids(r, 4, ne, "round_robin") for r in range(4)}
    shim_src = [_shim_source(r, owned[r], ne) for r in range(4)]
    rng_src = [s for r in range(4) for s in _range_source(r, owned[r], ne)]
    for rr in range(4):
        req_ids = frozenset(compute_local_expert_ids(rr, 4, ne, "round_robin"))
        # sanity: this really is non-contiguous → multiple runs
        assert len(expert_ids_to_contiguous_ranges(req_ids)) > 1
        _assert_parity(shim_src, _shim_request(rr, req_ids),
                       rng_src, _range_request(rr, req_ids))


def test_ranges_adapter_runs():
    assert expert_ids_to_contiguous_ranges([0, 1, 2, 3]) == ((0, 4),)
    assert expert_ids_to_contiguous_ranges([0, 2, 4, 6]) == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert expert_ids_to_contiguous_ranges([4, 5, 6, 9, 10]) == ((4, 7), (9, 11))
    assert expert_ids_to_contiguous_ranges([]) == ()
