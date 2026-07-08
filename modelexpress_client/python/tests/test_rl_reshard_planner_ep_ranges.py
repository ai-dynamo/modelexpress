# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ranges-only EP coverage in the reshard planner.

The core planner is range-only: an MoE expert shard is just a ``SHARD`` range
on the expert axis, and ordinary range intersection produces the correct EP
coverage — there is no ``is_expert`` / ``owned_expert_ids`` / ``required_experts``
special-case. These tests express every common EP placement as plain ranges
(via :func:`expert_ids_to_contiguous_ranges`) and assert the plan pulls exactly
the wanted experts from exactly the right source ranks.

(Previously these were parity tests against a deprecated expert shim; the shim
was proven equivalent and removed, so these are now the canonical EP tests.)
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


# ---- ranges-only builders (no expert fields) ----------------------------


def _range_source(rank: int, owned: tuple[int, ...], ne: int = _NE) -> list[SliceOwnership]:
    """One SHARD entry per contiguous run of owned expert ids."""
    out = []
    for lo, hi in expert_ids_to_contiguous_ranges(owned):
        out.append(SliceOwnership(
            model_name="m", tensor_name=_TENSOR, global_shape=(ne, 4096, 2048),
            dtype="torch.bfloat16", placement_kind="SHARD", shard_axis=0,
            local_shard_range=(lo, hi), worker_rank=rank,
            nixl_addr=0x1000 + rank, byte_size=(hi - lo) * _EXPERT_BYTES,
        ))
    return out


def _range_request(rank: int, required) -> list[SliceRequest]:
    """One request per contiguous run of required expert ids."""
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


# ---- coverage across placements ------------------------------------------


def test_linear_ep8_pulls_own_experts_from_own_rank():
    owned = {r: compute_local_expert_ids(r, 8, _NE, "linear") for r in range(8)}
    sources = [s for r in range(8) for s in _range_source(r, owned[r])]
    for rr in range(8):
        want = compute_local_expert_ids(rr, 8, _NE, "linear")
        plan = plan_coverage(sources, _range_request(rr, want))
        assert plan.complete
        experts, ranks = _coverage(plan)
        assert experts == set(want)
        # linear EP: rank rr's experts live entirely on source rank rr
        assert ranks == {rr}


def test_mixed_ep_trainer8_infer4_pulls_from_two_sources():
    owned = {r: compute_local_expert_ids(r, 8, _NE, "linear") for r in range(8)}
    sources = [s for r in range(8) for s in _range_source(r, owned[r])]
    for rr in range(4):
        want = compute_local_expert_ids(rr, 4, _NE, "linear")
        plan = plan_coverage(sources, _range_request(rr, want))
        assert plan.complete
        experts, ranks = _coverage(plan)
        assert experts == set(want)
        # each infer rank's 32 experts span exactly two trainer ranks
        assert ranks == {rr * 2, rr * 2 + 1}


def test_round_robin_ep4_noncontiguous():
    ne = 16
    owned = {r: compute_local_expert_ids(r, 4, ne, "round_robin") for r in range(4)}
    sources = [s for r in range(4) for s in _range_source(r, owned[r], ne)]
    for rr in range(4):
        want = compute_local_expert_ids(rr, 4, ne, "round_robin")
        # sanity: really non-contiguous → multiple runs
        assert len(expert_ids_to_contiguous_ranges(want)) > 1
        plan = plan_coverage(sources, _range_request(rr, want))
        assert plan.complete
        experts, ranks = _coverage(plan)
        assert experts == set(want)
        assert ranks == {rr}


def test_disjoint_request_pulls_nothing():
    # source owns experts [0,16); receiver wants [32,48) → no overlap, no pull
    sources = _range_source(0, tuple(range(16)))
    plan = plan_coverage(sources, _range_request(2, frozenset(range(32, 48))))
    experts, ranks = _coverage(plan)
    assert experts == set()
    assert ranks == set()
    assert not plan.complete


def test_ranges_adapter_runs():
    assert expert_ids_to_contiguous_ranges([0, 1, 2, 3]) == ((0, 4),)
    assert expert_ids_to_contiguous_ranges([0, 2, 4, 6]) == ((0, 1), (2, 3), (4, 5), (6, 7))
    assert expert_ids_to_contiguous_ranges([4, 5, 6, 9, 10]) == ((4, 7), (9, 11))
    assert expert_ids_to_contiguous_ranges([]) == ()
