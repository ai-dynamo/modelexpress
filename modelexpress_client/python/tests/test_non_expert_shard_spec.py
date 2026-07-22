# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for explicit non-expert shard publishing (NonExpertShardSpec).

Covers the path where a plain (non-DTensor) buffer is published as a
SHARD of a larger logical tensor so receivers can pull + reassemble
across ranks instead of the publisher pre-gathering. This is what lets
prime-rl's post-conversion ShardedSlot buffers publish per-rank shards
without a DTensor.full_tensor() allgather.
"""

from __future__ import annotations

import pytest
import torch

from modelexpress.shape_descriptors import (
    PLACEMENT_REPLICATE,
    PLACEMENT_SHARD,
    NonExpertShardSpec,
    describe_tensor,
)
from modelexpress.rl_reshard_planner import plan_coverage
from modelexpress.rl_slice_descriptors import SliceOwnership, SliceRequest


class TestShardSpecDescriptor:
    def test_plain_tensor_without_spec_is_replicate(self) -> None:
        # Baseline: a plain buffer with no spec is REPLICATE (the current
        # behavior that collapses shards into full replicas).
        t = torch.zeros(4, 8)
        d = describe_tensor(name="w", tensor=t, rank=0, fsdp_world_size=4)
        assert d.placement_kind == PLACEMENT_REPLICATE
        assert d.global_shape == (4, 8)

    def test_plain_tensor_with_spec_is_shard(self) -> None:
        # 4-way row shard of a (16, 8) tensor; rank 1 holds rows [4, 8).
        t = torch.zeros(4, 8)
        spec = NonExpertShardSpec(
            global_shape=(16, 8), shard_axis=0, local_shard_range=(4, 8)
        )
        d = describe_tensor(
            name="w", tensor=t, rank=1, fsdp_world_size=4, shard_spec=spec
        )
        assert d.placement_kind == PLACEMENT_SHARD
        assert d.global_shape == (16, 8)
        assert d.shard_axis == 0
        assert d.local_shard_range == (4, 8)

    def test_spec_extent_mismatch_raises(self) -> None:
        t = torch.zeros(4, 8)  # extent 4 on axis 0
        spec = NonExpertShardSpec(
            global_shape=(16, 8), shard_axis=0, local_shard_range=(4, 10)  # width 6 != 4
        )
        with pytest.raises(ValueError, match="width 6 != tensor extent 4"):
            describe_tensor(
                name="w", tensor=t, rank=1, fsdp_world_size=4, shard_spec=spec
            )

    def test_spec_range_out_of_bounds_raises(self) -> None:
        t = torch.zeros(4, 8)
        spec = NonExpertShardSpec(
            global_shape=(16, 8), shard_axis=0, local_shard_range=(14, 18)
        )
        with pytest.raises(ValueError, match="outside global_shape"):
            describe_tensor(
                name="w", tensor=t, rank=3, fsdp_world_size=4, shard_spec=spec
            )


class TestShardSpecReassembly:
    """The whole point: 4 per-rank shards published via spec should let the
    planner produce a full-coverage plan for a receiver wanting the full
    tensor — no allgather on the publisher side."""

    def test_four_rank_shards_cover_full_tensor(self) -> None:
        global_rows = 16
        n_ranks = 4
        chunk = global_rows // n_ranks

        # Build 4 ownerships from the descriptors (as the receiver would
        # reconstruct them from published metadata).
        ownerships = []
        for r in range(n_ranks):
            lo, hi = r * chunk, (r + 1) * chunk
            ownerships.append(
                SliceOwnership(
                    model_name="m",
                    tensor_name="model.layers.0.self_attn.q_proj.weight",
                    global_shape=(global_rows, 8),
                    dtype="torch.bfloat16",
                    placement_kind="SHARD",
                    shard_axis=0,
                    local_shard_range=(lo, hi),
                    worker_rank=r,
                    nixl_addr=0x1000 * (r + 1),
                    byte_size=chunk * 8 * 2,
                    device_id=0,
                )
            )

        # Receiver wants the full tensor (gather emulation).
        req = SliceRequest(
            tensor_name="model.layers.0.self_attn.q_proj.weight",
            global_range=(0, global_rows),
            shard_axis=0,
            dtype="torch.bfloat16",
            receiver_rank=0,
            target_addr=0xF000,
        )
        plan = plan_coverage(ownerships, [req])
        assert plan.complete
        # Every rank contributes exactly one segment; union covers [0, 16).
        assert len(plan.segments) == n_ranks
        ranks = sorted(seg.source.worker_rank for seg in plan.segments)
        assert ranks == [0, 1, 2, 3]
