# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import modelexpress
from modelexpress.rl_reshard import (
    TensorReceiveSpec,
    TensorShardSpec,
    TensorSlice,
    plan_dense_reshard_transfers,
    plan_exact_transfers,
    receive_specs_from_shape_registry,
    receive_specs_from_tensors,
    source_specs_from_shape_registry,
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


def test_plan_exact_transfers_rejects_matching_local_shape_at_different_offsets():
    plan = plan_exact_transfers(
        [
            _source(
                "w",
                rank=0,
                shape=(2, 4),
                global_shape=(4, 4),
                shard_offsets=(0, 0),
            )
        ],
        [
            _target(
                "w",
                rank=0,
                shape=(2, 4),
                global_shape=(4, 4),
                shard_offsets=(2, 0),
            )
        ],
    )

    assert not plan.complete
    assert "layout" in plan.missing[0].reason


def test_plan_dense_reshard_transfers_splits_full_target_across_source_shards():
    plan = plan_dense_reshard_transfers(
        [
            _source(
                "w",
                rank=0,
                shape=(2, 4),
                global_shape=(4, 4),
                shard_offsets=(0, 0),
            ),
            _source(
                "w",
                rank=1,
                shape=(2, 4),
                global_shape=(4, 4),
                shard_offsets=(2, 0),
            ),
        ],
        [
            _target(
                "w",
                rank=0,
                shape=(4, 4),
                global_shape=(4, 4),
                shard_offsets=(0, 0),
            )
        ],
    )

    assert plan.complete
    assert [entry.source_worker_rank for entry in plan.entries] == [0, 1]
    assert [entry.source_slice for entry in plan.entries] == [
        TensorSlice((0, 0), (2, 4)),
        TensorSlice((0, 0), (2, 4)),
    ]
    assert [entry.target_slice for entry in plan.entries] == [
        TensorSlice((0, 0), (2, 4)),
        TensorSlice((2, 0), (2, 4)),
    ]


def test_plan_dense_reshard_transfers_targets_partial_shard_from_full_source():
    plan = plan_dense_reshard_transfers(
        [
            _source(
                "w",
                rank=0,
                shape=(8,),
                global_shape=(8,),
                shard_offsets=(0,),
            )
        ],
        [
            _target(
                "w",
                rank=2,
                shape=(2,),
                global_shape=(8,),
                shard_offsets=(4,),
            )
        ],
    )

    assert plan.complete
    assert plan.entries[0].source_slice == TensorSlice((4,), (2,))
    assert plan.entries[0].target_slice == TensorSlice((0,), (2,))


def test_plan_dense_reshard_transfers_reports_incomplete_coverage():
    plan = plan_dense_reshard_transfers(
        [
            _source(
                "w",
                rank=0,
                shape=(2,),
                global_shape=(8,),
                shard_offsets=(0,),
            )
        ],
        [_target("w", rank=0, shape=(4,), global_shape=(8,), shard_offsets=(0,))],
    )

    assert not plan.complete
    assert plan.missing[0].reason == "source coverage is incomplete or overlapping"


def test_plan_dense_reshard_transfers_rejects_overlapping_coverage():
    plan = plan_dense_reshard_transfers(
        [
            _source("w", rank=0, shape=(2,), global_shape=(4,), shard_offsets=(0,)),
            _source("w", rank=1, shape=(2,), global_shape=(4,), shard_offsets=(0,)),
        ],
        [_target("w", rank=0, shape=(4,), global_shape=(4,), shard_offsets=(0,))],
    )

    assert not plan.complete
    assert plan.missing[0].reason == "source coverage is incomplete or overlapping"


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
    assert modelexpress.TensorSlice is TensorSlice
    assert modelexpress.plan_exact_transfers is plan_exact_transfers
    assert modelexpress.plan_dense_reshard_transfers is plan_dense_reshard_transfers
    assert modelexpress.source_specs_from_shape_registry is source_specs_from_shape_registry
    assert modelexpress.receive_specs_from_shape_registry is receive_specs_from_shape_registry
    assert modelexpress.receive_specs_from_tensors is receive_specs_from_tensors


def test_source_specs_from_shape_registry_preserves_layout_metadata():
    specs = source_specs_from_shape_registry(
        {
            "experts.w": {
                "shape": [2, 4],
                "dtype": "torch.bfloat16",
                "tensor_parallel_rank": 1,
                "pipeline_parallel_rank": 2,
                "global_shape": [4, 4],
                "shard_offsets": [2, 0],
                "expert_ids": [3, 7],
            }
        },
        worker_rank=5,
    )

    assert specs == (
        TensorShardSpec(
            name="experts.w",
            worker_rank=5,
            shape=(2, 4),
            dtype="torch.bfloat16",
            global_shape=(4, 4),
            shard_offsets=(2, 0),
            tensor_parallel_rank=1,
            pipeline_parallel_rank=2,
            expert_ids=frozenset({3, 7}),
        ),
    )


def test_receive_specs_from_shape_registry_sets_receiver_rank():
    specs = receive_specs_from_shape_registry(
        {"w": {"shape": [2, 4], "dtype": "torch.float16"}},
        receiver_rank=2,
    )

    assert specs == (
        TensorReceiveSpec(
            name="w",
            receiver_rank=2,
            shape=(2, 4),
            dtype="torch.float16",
        ),
    )


def test_receive_specs_from_tensors_builds_default_dense_specs():
    specs = receive_specs_from_tensors(
        {"w": torch.zeros((2, 4), dtype=torch.float32)},
        receiver_rank=1,
    )

    assert specs == (
        TensorReceiveSpec(
            name="w",
            receiver_rank=1,
            shape=(2, 4),
            dtype="torch.float32",
        ),
    )


def test_shape_registry_spec_conversion_rejects_invalid_expert_metadata():
    with pytest.raises(ValueError, match="expert_ids must be a list"):
        source_specs_from_shape_registry(
            {"w": {"shape": [1], "dtype": "torch.float32", "expert_ids": "bad"}},
            worker_rank=0,
        )
