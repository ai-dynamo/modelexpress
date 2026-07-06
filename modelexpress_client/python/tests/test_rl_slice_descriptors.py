# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :mod:`modelexpress.rl_slice_descriptors`.

Pure-Python (no torch, no NIXL) — exercises the descriptor invariants
that the rank-to-rank no-allgather contract relies on.
"""

from __future__ import annotations

import pytest

from modelexpress.rl_slice_descriptors import (
    CoveragePlan,
    PlanIncompleteError,
    SliceOwnership,
    SliceRequest,
)


# ---------------------------------------------------------------------------
# SliceOwnership validation
# ---------------------------------------------------------------------------


def test_replicate_ownership_is_valid_without_shard_info():
    """REPLICATE doesn't need shard_axis or local_shard_range."""
    own = SliceOwnership(
        model_name="m", tensor_name="ln.weight",
        global_shape=(1024,), dtype="torch.bfloat16",
        placement_kind="REPLICATE",
    )
    assert own.covers((0, 1024)) == (0, 1024)


def test_shard_ownership_requires_shard_axis_and_range():
    """SHARD without shard_axis/range should fail at construction."""
    with pytest.raises(ValueError, match="SHARD requires shard_axis"):
        SliceOwnership(
            model_name="m", tensor_name="t", global_shape=(8,), dtype="torch.bfloat16",
            placement_kind="SHARD",
        )
    with pytest.raises(ValueError, match="SHARD requires local_shard_range"):
        SliceOwnership(
            model_name="m", tensor_name="t", global_shape=(8,), dtype="torch.bfloat16",
            placement_kind="SHARD", shard_axis=0,
        )


def test_shard_ownership_rejects_out_of_bounds_range():
    """local_shard_range hi > global_shape[shard_axis] should fail."""
    with pytest.raises(ValueError, match="exceeds global_shape"):
        SliceOwnership(
            model_name="m", tensor_name="t", global_shape=(8,), dtype="torch.bfloat16",
            placement_kind="SHARD", shard_axis=0, local_shard_range=(0, 16),
        )


def test_shard_ownership_rejects_empty_range():
    with pytest.raises(ValueError, match="empty or negative"):
        SliceOwnership(
            model_name="m", tensor_name="t", global_shape=(8,), dtype="torch.bfloat16",
            placement_kind="SHARD", shard_axis=0, local_shard_range=(4, 4),
        )


def test_partial_placement_is_not_yet_supported():
    """PARTIAL placement should raise — would require all-reduce on receiver."""
    with pytest.raises(NotImplementedError, match="PARTIAL placement"):
        SliceOwnership(
            model_name="m", tensor_name="t", global_shape=(8,), dtype="torch.bfloat16",
            placement_kind="PARTIAL",
        )


# ---------------------------------------------------------------------------
# covers() — the intersection primitive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("own_range", "req_range", "expected"),
    [
        ((0, 100), (0, 100), (0, 100)),       # exact match
        ((0, 100), (20, 80), (20, 80)),       # request is interior
        ((20, 80), (0, 100), (20, 80)),       # ownership is interior
        ((0, 50), (40, 100), (40, 50)),       # partial overlap on right
        ((50, 100), (0, 60), (50, 60)),       # partial overlap on left
        ((0, 50), (50, 100), None),           # adjacent, no overlap
        ((0, 50), (60, 80), None),            # disjoint
    ],
)
def test_covers_intersection(own_range, req_range, expected):
    own = SliceOwnership(
        model_name="m", tensor_name="t", global_shape=(100,), dtype="torch.bfloat16",
        placement_kind="SHARD", shard_axis=0, local_shard_range=own_range,
    )
    assert own.covers(req_range) == expected


# ---------------------------------------------------------------------------
# SliceRequest validation
# ---------------------------------------------------------------------------


def test_request_rejects_empty_range():
    with pytest.raises(ValueError, match="empty or negative"):
        SliceRequest(
            tensor_name="t", global_range=(10, 10), shard_axis=0,
            dtype="torch.bfloat16",
        )


# ---------------------------------------------------------------------------
# CoveragePlan completeness
# ---------------------------------------------------------------------------


def test_empty_plan_is_complete():
    plan = CoveragePlan(segments=())
    assert plan.complete
    plan.raise_if_incomplete()  # no-op


def test_plan_with_missing_is_incomplete():
    plan = CoveragePlan(
        segments=(),
        missing=(("t", (0, 100), "no source"),),
    )
    assert not plan.complete
    with pytest.raises(PlanIncompleteError, match="no source"):
        plan.raise_if_incomplete()
