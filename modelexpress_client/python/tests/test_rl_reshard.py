# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import modelexpress
from modelexpress.rl_reshard import (
    TensorReceiveSpec,
    TensorShardSpec,
    plan_exact_transfers,
)


def _source(name="w", rank=0, **kwargs):
    values = {
        "name": name,
        "worker_rank": rank,
        "shape": (2, 4),
        "dtype": "torch.bfloat16",
    }
    values.update(kwargs)
    return TensorShardSpec(**values)


def _target(name="w", rank=0, **kwargs):
    values = {
        "name": name,
        "receiver_rank": rank,
        "shape": (2, 4),
        "dtype": "torch.bfloat16",
    }
    values.update(kwargs)
    return TensorReceiveSpec(**values)


def test_plan_exact_transfers_matches_same_rank_dense_tensor():
    plan = plan_exact_transfers(
        [_source("w", rank=0)],
        [_target("w", rank=0)],
    )

    assert plan.complete
    assert len(plan.entries) == 1
    assert plan.entries[0].tensor_name == "w"
    assert plan.entries[0].source_worker_rank == 0
    assert plan.entries[0].receiver_rank == 0


def test_plan_exact_transfers_reports_missing_tensor():
    plan = plan_exact_transfers(
        [_source("other", rank=0)],
        [_target("w", rank=0)],
    )

    assert not plan.complete
    assert plan.missing[0].target.name == "w"
    assert plan.missing[0].reason == "tensor not found"
    with pytest.raises(ValueError, match="incomplete RL reshard plan"):
        plan.raise_if_incomplete()


def test_plan_exact_transfers_reports_shape_or_dtype_mismatch():
    plan = plan_exact_transfers(
        [_source("w", rank=0, shape=(4, 4))],
        [_target("w", rank=0, shape=(2, 4))],
    )

    assert not plan.complete
    assert "shape, dtype, layout, or expert ownership differs" in plan.missing[0].reason


def test_plan_exact_transfers_can_fall_back_to_different_rank():
    sources = [
        _source("w", rank=1),
        _source("w", rank=2),
    ]
    target = _target("w", rank=0)

    same_rank_plan = plan_exact_transfers(sources, [target])
    fallback_plan = plan_exact_transfers(sources, [target], same_rank_only=False)

    assert same_rank_plan.missing[0].reason == "compatible source exists on a different rank"
    assert fallback_plan.complete
    assert fallback_plan.entries[0].source_worker_rank == 1


def test_plan_exact_transfers_prefers_same_rank_when_fallback_allowed():
    plan = plan_exact_transfers(
        [_source("w", rank=2), _source("w", rank=0)],
        [_target("w", rank=0)],
        same_rank_only=False,
    )

    assert plan.complete
    assert plan.entries[0].source_worker_rank == 0


def test_plan_exact_transfers_filters_by_tp_and_pp_coordinates():
    plan = plan_exact_transfers(
        [
            _source("w", rank=0, tensor_parallel_rank=0, pipeline_parallel_rank=1),
            _source("w", rank=0, tensor_parallel_rank=1, pipeline_parallel_rank=1),
        ],
        [_target("w", rank=0, tensor_parallel_rank=1, pipeline_parallel_rank=1)],
    )

    assert plan.complete
    assert plan.entries[0].source.tensor_parallel_rank == 1
    assert plan.entries[0].source.pipeline_parallel_rank == 1


def test_plan_exact_transfers_filters_by_moe_expert_ownership():
    plan = plan_exact_transfers(
        [
            _source("experts.w", rank=0, expert_ids=frozenset({0, 1})),
            _source("experts.w", rank=0, expert_ids=frozenset({2, 3})),
        ],
        [_target("experts.w", rank=0, expert_ids=frozenset({3}))],
    )

    assert plan.complete
    assert plan.entries[0].source.expert_ids == frozenset({2, 3})


def test_plan_exact_transfers_requires_matching_expert_metadata():
    plan = plan_exact_transfers(
        [_source("experts.w", rank=0, expert_ids=frozenset({0, 1}))],
        [_target("experts.w", rank=0)],
    )

    assert not plan.complete
    assert "expert ownership differs" in plan.missing[0].reason


def test_reshard_planner_types_are_exported_from_package():
    assert modelexpress.TensorShardSpec is TensorShardSpec
    assert modelexpress.TensorReceiveSpec is TensorReceiveSpec
    assert modelexpress.plan_exact_transfers is plan_exact_transfers
