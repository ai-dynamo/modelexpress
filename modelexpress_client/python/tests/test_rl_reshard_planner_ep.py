# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the planner's per-expert filtering logic.

Covers the new ``SliceRequest.required_experts`` field and the
extension of ``_source_matches_request`` that consumes it. Validates:

- Matched expert filter emits the expected SegmentPlans.
- Mismatched expert filter emits zero SegmentPlans (no false pulls).
- Cross-expert pulls are refused with a clear diagnostic.
- ``required_experts=None`` (default) preserves pre-EP planner behavior.
- The helper output composes correctly with the planner.
"""

from __future__ import annotations

from modelexpress.rl_expert_layout import compute_local_expert_ids
from modelexpress.rl_reshard_planner import plan_coverage
from modelexpress.rl_slice_descriptors import SliceOwnership, SliceRequest


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _make_expert_source(
    *,
    rank: int,
    owned: tuple[int, ...],
    tensor_name: str = "model.layers.0.experts.w13_weight",
    num_experts: int = 128,
    addr_base: int = 0x10000000,
) -> SliceOwnership:
    """Build a SliceOwnership for a contiguous slice of expert tensors."""
    # Each rank owns a contiguous block along axis 0 (expert axis).
    # Pretend each expert is a 4096x2048 bf16 tensor → 16 MiB per expert.
    expert_bytes = 4096 * 2048 * 2
    lo, hi = owned[0], owned[-1] + 1
    return SliceOwnership(
        model_name="ep-test-model",
        tensor_name=tensor_name,
        global_shape=(num_experts, 4096, 2048),
        dtype="torch.bfloat16",
        placement_kind="SHARD",
        shard_axis=0,
        local_shard_range=(lo, hi),
        worker_rank=rank,
        nixl_addr=addr_base + rank * 0x10000000,
        byte_size=len(owned) * expert_bytes,
        device_id=0,
        is_expert=True,
        expert_axis=0,
        owned_expert_ids=owned,
    )


def _make_expert_request(
    *,
    receiver_rank: int,
    required: frozenset[int] | None,
    tensor_name: str = "model.layers.0.experts.w13_weight",
    num_experts: int = 128,
    target_addr: int = 0xF0000000,
) -> SliceRequest:
    """Build a SliceRequest covering the contiguous range that holds ``required``.

    For an EP receiver, the request range should describe exactly the
    expert-axis sub-range the receiver wants to land. For linear
    placement this is a single contiguous block (min..max+1). For
    round-robin or external placements with non-contiguous required
    sets, the caller should issue multiple SliceRequests — one per
    contiguous sub-range — and this helper is not the right shape.
    Falls back to ``(0, num_experts)`` when ``required`` is None
    (back-compat path).
    """
    if required is None:
        rng = (0, num_experts)
    else:
        lo, hi = min(required), max(required) + 1
        # Validate contiguity — round-robin / external go through a
        # different code path (multiple requests).
        assert frozenset(range(lo, hi)) == required, (
            f"_make_expert_request: non-contiguous required={sorted(required)}; "
            f"build multiple SliceRequests for non-linear placements"
        )
        rng = (lo, hi)
    return SliceRequest(
        tensor_name=tensor_name,
        global_range=rng,
        shard_axis=0,
        dtype="torch.bfloat16",
        receiver_rank=receiver_rank,
        target_addr=target_addr,
        required_experts=required,
    )


# ---------------------------------------------------------------------------
# matched / mismatched / cross-expert
# ---------------------------------------------------------------------------


class TestMatchedExpertFilter:
    """Receiver wants experts a publisher actually owns."""

    def test_single_source_full_overlap(self) -> None:
        # Source owns experts [0, 16); receiver wants [0, 16).
        own = _make_expert_source(rank=0, owned=tuple(range(16)))
        req = _make_expert_request(
            receiver_rank=0, required=frozenset(range(16))
        )
        plan = plan_coverage([own], [req])
        assert len(plan.segments) == 1
        assert plan.segments[0].source.worker_rank == 0
        assert plan.complete
        assert not plan.missing

    def test_multi_source_partition(self) -> None:
        # 8 sources owning 16 experts each = 128 experts total.
        # Receiver wants experts [0, 16) — should pull only from source 0.
        sources = [
            _make_expert_source(rank=r, owned=tuple(range(r * 16, (r + 1) * 16)))
            for r in range(8)
        ]
        req = _make_expert_request(
            receiver_rank=0, required=frozenset(range(16))
        )
        plan = plan_coverage(sources, [req])
        assert len(plan.segments) == 1
        assert plan.segments[0].source.worker_rank == 0
        # No other source should appear in the plan.
        ranks_in_plan = {seg.source.worker_rank for seg in plan.segments}
        assert ranks_in_plan == {0}


class TestMismatchedExpertFilter:
    """Receiver wants experts no publisher owns — planner emits nothing."""

    def test_disjoint_no_segments(self) -> None:
        # Source owns experts [0, 16); receiver wants [32, 48).
        own = _make_expert_source(rank=0, owned=tuple(range(16)))
        req = _make_expert_request(
            receiver_rank=2, required=frozenset(range(32, 48))
        )
        plan = plan_coverage([own], [req])
        assert plan.segments == ()
        # The tensor name was matched but no eligible source covered it
        # after the expert filter rejected the only candidate. The
        # missing-entry rejection reason should mention the expert mismatch.
        assert len(plan.missing) == 1
        _, _, reason = plan.missing[0]
        assert "experts" in reason.lower()

    def test_partial_overlap_picks_only_intersecting_sources(self) -> None:
        # 8 publishers each owning 16 experts. Receiver wants experts
        # [32, 48) (EP rank 2 of 8 in linear placement).
        sources = [
            _make_expert_source(rank=r, owned=tuple(range(r * 16, (r + 1) * 16)))
            for r in range(8)
        ]
        req = _make_expert_request(
            receiver_rank=2, required=frozenset(range(32, 48))
        )
        plan = plan_coverage(sources, [req])
        # Only source rank 2 (which owns experts [32, 48)) should appear.
        ranks_in_plan = {seg.source.worker_rank for seg in plan.segments}
        assert ranks_in_plan == {2}


class TestNonExpertSourceRejected:
    """A source that doesn't declare is_expert cannot satisfy an EP request."""

    def test_non_expert_source_against_ep_request(self) -> None:
        # Source has shape (128, 4096, 2048) and shard [0, 16) but is_expert=False.
        own = SliceOwnership(
            model_name="ep-test-model",
            tensor_name="model.layers.0.experts.w13_weight",
            global_shape=(128, 4096, 2048),
            dtype="torch.bfloat16",
            placement_kind="SHARD",
            shard_axis=0,
            local_shard_range=(0, 16),
            worker_rank=0,
            byte_size=16 * 4096 * 2048 * 2,
            is_expert=False,  # ← the key: not marked as expert
        )
        req = _make_expert_request(
            receiver_rank=0, required=frozenset(range(16))
        )
        plan = plan_coverage([own], [req])
        assert plan.segments == ()
        assert len(plan.missing) == 1
        _, _, reason = plan.missing[0]
        assert "not marked is_expert" in reason


# ---------------------------------------------------------------------------
# back-compat: required_experts=None
# ---------------------------------------------------------------------------


class TestBackCompatNoneFilter:
    """``required_experts=None`` (default) preserves existing planner behavior."""

    def test_non_ep_request_still_works(self) -> None:
        # Non-MoE source + request without expert filter. Should plan
        # exactly as before EP support landed.
        own = SliceOwnership(
            model_name="test-model",
            tensor_name="model.embed_tokens.weight",
            global_shape=(50000, 4096),
            dtype="torch.bfloat16",
            placement_kind="SHARD",
            shard_axis=0,
            local_shard_range=(0, 25000),
            worker_rank=0,
            byte_size=25000 * 4096 * 2,
        )
        req = SliceRequest(
            tensor_name="model.embed_tokens.weight",
            global_range=(0, 25000),
            shard_axis=0,
            dtype="torch.bfloat16",
            receiver_rank=0,
            # required_experts not set → default None → no EP filtering
        )
        plan = plan_coverage([own], [req])
        assert len(plan.segments) == 1
        assert plan.complete

    def test_non_ep_request_ignores_expert_source_metadata(self) -> None:
        # A source declares itself as expert with owned_expert_ids,
        # but the request doesn't set required_experts. The planner
        # should NOT filter — required_experts=None means "I don't care
        # about EP." (Useful for a receiver that wants all experts
        # whether the source declares EP metadata or not.)
        own = _make_expert_source(rank=0, owned=tuple(range(16)))
        req = SliceRequest(
            tensor_name="model.layers.0.experts.w13_weight",
            global_range=(0, 16),
            shard_axis=0,
            dtype="torch.bfloat16",
            receiver_rank=0,
            # required_experts not set
        )
        plan = plan_coverage([own], [req])
        assert len(plan.segments) == 1


# ---------------------------------------------------------------------------
# helper-driven integration
# ---------------------------------------------------------------------------


class TestHelperDrivenIntegration:
    """Helper output composes correctly with the planner across all
    common EP placements."""

    def test_linear_ep_8_world_128_experts(self) -> None:
        # 8 publishers (linear EP=8 over 128 experts). 8 receivers
        # (same shape, linear EP=8). Each receiver should pull from
        # exactly one source (its rank-matched publisher).
        sources = [
            _make_expert_source(
                rank=r,
                owned=compute_local_expert_ids(r, 8, 128, "linear"),
            )
            for r in range(8)
        ]
        for receiver_rank in range(8):
            local = frozenset(compute_local_expert_ids(receiver_rank, 8, 128, "linear"))
            req = _make_expert_request(
                receiver_rank=receiver_rank, required=local
            )
            plan = plan_coverage(sources, [req])
            ranks_pulled_from = {seg.source.worker_rank for seg in plan.segments}
            assert ranks_pulled_from == {receiver_rank}, (
                f"linear EP=8 receiver rank {receiver_rank} should pull only "
                f"from publisher rank {receiver_rank}; pulled from {ranks_pulled_from}"
            )
            assert plan.complete

    def test_mixed_ep_trainer_8_inference_4(self) -> None:
        # 8 publishers (trainer EP=8); 4 receivers (inference EP=4).
        # Each inference rank owns 32 experts (linear EP=4 over 128).
        # Each receive request should match exactly two publishers
        # (since trainer's 16 experts per rank × 2 = 32 = inference's
        # per-rank ownership).
        sources = [
            _make_expert_source(
                rank=r,
                owned=compute_local_expert_ids(r, 8, 128, "linear"),
            )
            for r in range(8)
        ]
        for receiver_rank in range(4):
            local = frozenset(compute_local_expert_ids(receiver_rank, 4, 128, "linear"))
            req = _make_expert_request(
                receiver_rank=receiver_rank, required=local
            )
            plan = plan_coverage(sources, [req])
            ranks_pulled_from = {seg.source.worker_rank for seg in plan.segments}
            # Mixed EP 8→4: each inference rank pulls from exactly two
            # trainer ranks. E.g. inference rank 0 wants experts [0, 32),
            # which is held by trainer ranks 0 and 1.
            expected = {receiver_rank * 2, receiver_rank * 2 + 1}
            assert ranks_pulled_from == expected, (
                f"mixed EP trainer=8 inference=4 receiver rank {receiver_rank} "
                f"should pull from publisher ranks {expected}; pulled from "
                f"{ranks_pulled_from}"
            )

    def test_round_robin_ep_4_world_16_experts(self) -> None:
        # Round-robin placement: experts are interleaved across ranks.
        # Receiver rank r wants {r, r+4, r+8, r+12} — non-contiguous on
        # the expert axis, so each wanted expert needs its own SliceRequest.
        # This is the architecturally correct way to express non-linear
        # EP layouts to the planner today (the alternative would be a
        # planner extension for set-valued requested ranges, which is
        # out of scope for this PR).
        sources = [
            _make_expert_source(
                rank=r,
                owned=compute_local_expert_ids(r, 4, 16, "round_robin"),
                num_experts=16,
            )
            for r in range(4)
        ]
        for receiver_rank in range(4):
            local = compute_local_expert_ids(receiver_rank, 4, 16, "round_robin")
            # One SliceRequest per wanted expert (since they're not contiguous).
            requests = [
                SliceRequest(
                    tensor_name="model.layers.0.experts.w13_weight",
                    global_range=(eid, eid + 1),
                    shard_axis=0,
                    dtype="torch.bfloat16",
                    receiver_rank=receiver_rank,
                    target_addr=0xF0000000 + i * (4096 * 2048 * 2),
                    required_experts=frozenset({eid}),
                )
                for i, eid in enumerate(local)
            ]
            plan = plan_coverage(sources, requests)
            ranks_pulled_from = {seg.source.worker_rank for seg in plan.segments}
            # Same-rank match: receiver r owns same indices as publisher r
            # under symmetric round-robin EP=4.
            assert ranks_pulled_from == {receiver_rank}
            assert plan.complete
