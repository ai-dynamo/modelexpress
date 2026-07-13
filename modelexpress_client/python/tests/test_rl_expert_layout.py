# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the EP layout helpers (``modelexpress.rl_expert_layout``).

Covers the three placement strategies (linear / round_robin / external),
edge cases (empty model, single rank, uneven splits), partition
validation, and the round-trip property that ``compute_local_expert_ids``
outputs compose into plain SHARD ranges on the substrate's expert axis.
"""

from __future__ import annotations

import pytest

from modelexpress.rl_expert_layout import (
    compute_local_expert_ids,
    expert_ids_to_contiguous_ranges,
    validate_placement_partition,
)
from modelexpress.rl_slice_descriptors import SliceOwnership, SliceRequest


# ---------------------------------------------------------------------------
# linear placement
# ---------------------------------------------------------------------------


class TestLinear:
    """Linear placement — contiguous blocks per rank. The vLLM/SGLang default."""

    def test_two_ranks_eight_experts(self) -> None:
        assert compute_local_expert_ids(0, 2, 8, "linear") == (0, 1, 2, 3)
        assert compute_local_expert_ids(1, 2, 8, "linear") == (4, 5, 6, 7)

    def test_eight_ranks_qwen3_moe_30b_shape(self) -> None:
        # 128 experts (Qwen3-MoE-30B-A3B shape) across EP=8 → 16 each.
        for rank in range(8):
            owned = compute_local_expert_ids(rank, 8, 128, "linear")
            assert owned == tuple(range(rank * 16, (rank + 1) * 16))
            assert len(owned) == 16

    def test_qwen3_235b_a22b_shape(self) -> None:
        # 192 experts (Qwen3-235B-A22B) across EP=8 → 24 each.
        owned = compute_local_expert_ids(2, 8, 192, "linear")
        assert owned == tuple(range(48, 72))
        assert len(owned) == 24

    def test_single_rank_owns_everything(self) -> None:
        assert compute_local_expert_ids(0, 1, 16, "linear") == tuple(range(16))

    def test_uneven_split_raises(self) -> None:
        # 7 experts across 2 ranks doesn't divide evenly.
        with pytest.raises(ValueError, match="divisible"):
            compute_local_expert_ids(0, 2, 7, "linear")

    def test_zero_experts(self) -> None:
        # Edge case: dense (non-MoE) model with EP world > 1 should
        # produce empty ownership for every rank.
        assert compute_local_expert_ids(0, 4, 0, "linear") == ()
        assert compute_local_expert_ids(3, 4, 0, "linear") == ()


# ---------------------------------------------------------------------------
# round_robin placement
# ---------------------------------------------------------------------------


class TestRoundRobin:
    """Round-robin placement — interleaved. Megatron-Core's rebalancer pattern."""

    def test_two_ranks_eight_experts(self) -> None:
        assert compute_local_expert_ids(0, 2, 8, "round_robin") == (0, 2, 4, 6)
        assert compute_local_expert_ids(1, 2, 8, "round_robin") == (1, 3, 5, 7)

    def test_four_ranks_sixteen_experts(self) -> None:
        assert compute_local_expert_ids(0, 4, 16, "round_robin") == (0, 4, 8, 12)
        assert compute_local_expert_ids(2, 4, 16, "round_robin") == (2, 6, 10, 14)

    def test_uneven_total_handled_gracefully(self) -> None:
        # 7 experts across 2 ranks: rank 0 gets 4, rank 1 gets 3.
        # Round-robin doesn't require divisibility.
        assert compute_local_expert_ids(0, 2, 7, "round_robin") == (0, 2, 4, 6)
        assert compute_local_expert_ids(1, 2, 7, "round_robin") == (1, 3, 5)


# ---------------------------------------------------------------------------
# external placement (EPLB-style)
# ---------------------------------------------------------------------------


class TestExternal:
    """External map — caller-supplied per-rank expert assignments."""

    def test_explicit_map(self) -> None:
        em = {0: [0, 1, 4, 7], 1: [2, 3, 5, 6]}
        assert compute_local_expert_ids(0, 2, 8, "external", em) == (0, 1, 4, 7)
        assert compute_local_expert_ids(1, 2, 8, "external", em) == (2, 3, 5, 6)

    def test_external_output_sorted(self) -> None:
        # External map values can come in any order; output is sorted.
        em = {0: [7, 1, 4, 0]}
        assert compute_local_expert_ids(0, 1, 8, "external", em) == (0, 1, 4, 7)

    def test_external_missing_map_raises(self) -> None:
        with pytest.raises(ValueError, match="external_map to be provided"):
            compute_local_expert_ids(0, 2, 8, "external")

    def test_external_missing_rank_raises(self) -> None:
        with pytest.raises(ValueError, match="no entry for ep_rank=1"):
            compute_local_expert_ids(1, 2, 8, "external", {0: [0, 1, 2, 3]})

    def test_external_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="outside \\[0, 8\\)"):
            compute_local_expert_ids(0, 1, 8, "external", {0: [0, 1, 9]})

    def test_external_overlap_allowed(self) -> None:
        # Overlapping ownership (e.g. for replicated-expert deployments)
        # is permitted — the helper validates inputs but doesn't enforce
        # disjoint partitioning. validate_placement_partition does.
        em = {0: [0, 1, 2], 1: [2, 3, 4]}
        assert compute_local_expert_ids(0, 2, 5, "external", em) == (0, 1, 2)
        assert compute_local_expert_ids(1, 2, 5, "external", em) == (2, 3, 4)


# ---------------------------------------------------------------------------
# argument validation
# ---------------------------------------------------------------------------


class TestArgumentValidation:
    def test_ep_world_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="ep_world_size must be >= 1"):
            compute_local_expert_ids(0, 0, 8, "linear")

    def test_ep_rank_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="ep_rank=4 out of range"):
            compute_local_expert_ids(4, 2, 8, "linear")

    def test_negative_ep_rank_raises(self) -> None:
        with pytest.raises(ValueError, match="ep_rank=-1 out of range"):
            compute_local_expert_ids(-1, 2, 8, "linear")

    def test_negative_num_experts_raises(self) -> None:
        with pytest.raises(ValueError, match="num_experts must be >= 0"):
            compute_local_expert_ids(0, 2, -1, "linear")

    def test_unknown_placement_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown placement"):
            # Pass an unknown string; ignore the static type warning.
            compute_local_expert_ids(0, 2, 8, "bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# validate_placement_partition
# ---------------------------------------------------------------------------


class TestValidatePartition:
    def test_linear_covers_all(self) -> None:
        # Should not raise.
        validate_placement_partition(8, 128, "linear")

    def test_round_robin_covers_all(self) -> None:
        validate_placement_partition(4, 16, "round_robin")

    def test_external_with_gap_raises(self) -> None:
        # External map that misses expert 5.
        em = {0: [0, 1, 2, 3, 4], 1: [6, 7]}
        with pytest.raises(ValueError, match=r"\[5\]"):
            validate_placement_partition(2, 8, "external", em)

    def test_external_with_overlap_is_fine(self) -> None:
        # Overlapping ownership covers every expert; no error.
        em = {0: [0, 1, 2, 3, 4], 1: [4, 5, 6, 7]}
        validate_placement_partition(2, 8, "external", em)


# ---------------------------------------------------------------------------
# round-trip with substrate dataclasses
# ---------------------------------------------------------------------------


class TestSubstrateRoundTrip:
    """Helper output composes into plain SHARD ranges on the expert axis —
    there is no expert-specific descriptor field."""

    def test_owned_ids_become_a_shard_range(self) -> None:
        owned = compute_local_expert_ids(2, 8, 128, "linear")  # -> (32..47)
        (lo, hi), = expert_ids_to_contiguous_ranges(owned)
        own = SliceOwnership(
            model_name="test-model",
            tensor_name="model.layers.0.experts.w13_weight",
            global_shape=(128, 4096, 2048),
            dtype="torch.bfloat16",
            placement_kind="SHARD",
            shard_axis=0,
            local_shard_range=(lo, hi),
            worker_rank=2,
            byte_size=(hi - lo) * 4096 * 2048 * 2,
        )
        assert own.local_shard_range == (32, 48)

    def test_wanted_ids_become_request_ranges(self) -> None:
        owned = compute_local_expert_ids(2, 8, 128, "linear")
        reqs = [
            SliceRequest(
                tensor_name="model.layers.0.experts.w13_weight",
                global_range=(lo, hi),
                shard_axis=0,
                dtype="torch.bfloat16",
                receiver_rank=2,
            )
            for lo, hi in expert_ids_to_contiguous_ranges(owned)
        ]
        assert [r.global_range for r in reqs] == [(32, 48)]

    def test_noncontiguous_ids_become_multiple_ranges(self) -> None:
        owned = compute_local_expert_ids(0, 4, 16, "round_robin")  # {0,4,8,12}
        ranges = expert_ids_to_contiguous_ranges(owned)
        assert len(ranges) == 4  # one SHARD entry per interleaved expert
