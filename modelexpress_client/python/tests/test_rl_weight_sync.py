# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RL-specific integration tests for weight_transfer.

Covers the trainer-inference weight sync protocol as it is used inside an RL
training loop (GRPO / PPO / PrimeRL style):

  1. Trainer takes gradient step → increments TrainerTable.step.
  2. Inference workers pull updated weights via PullRole.
  3. Inference runs rollouts → rewards fed back to trainer.
  4. Loop continues; plan is cached and only invalidated on reshard.

All tests use pure-Python stubs (no NIXL, no real GPU, no real model).
"""

import math
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest
import torch

from modelexpress.weight_transfer.planner.local import LocalPlanner
from modelexpress.weight_transfer.planner.m2n_planner import M2nPlanner
from modelexpress.weight_transfer.planner.router import route_regions
from modelexpress.weight_transfer.protocol.types import (
    M2nDescriptor,
    RdmaDescriptor,
    ResolvedRegion,
    TrainerShard,
    TrainerTable,
    TrainerTensor,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_row_shard(agent_index, row_start, row_end, shape, base_addr=0x10000):
    """Return a row-only TrainerShard (FSDP / DP sharding)."""
    elem_size = 2  # bfloat16
    row_bytes = math.prod(shape[1:]) * elem_size if len(shape) > 1 else elem_size
    addr = base_addr + row_start * row_bytes
    return TrainerShard(
        agent_index=agent_index,
        row_start=row_start,
        row_end=row_end,
        device_addr=addr,
        row_bytes=row_bytes,
        device_id=agent_index,
    )


def _make_col_shard(agent_index, row_start, row_end, col_start, col_end, full_cols, base_addr=0x20000):
    """Return a column (2-D tile) TrainerShard for TP row-parallel layers."""
    elem_size = 2
    shard_cols = col_end - col_start
    row_bytes = shard_cols * elem_size
    # Each shard holds a contiguous [rows, shard_cols] tile.
    addr = base_addr + agent_index * (row_end - row_start) * row_bytes
    return TrainerShard(
        agent_index=agent_index,
        row_start=row_start,
        row_end=row_end,
        col_start=col_start,
        col_end=col_end,
        device_addr=addr,
        row_bytes=row_bytes,
        device_id=agent_index,
    )


def _simple_table(step=1):
    """Single 4×4 tensor, two row-shards (FSDP DP=2)."""
    shape = [4, 4]
    shards = [
        _make_row_shard(0, 0, 2, shape, base_addr=0x1000),
        _make_row_shard(1, 2, 4, shape, base_addr=0x2000),
    ]
    tensors = [TrainerTensor(name="layer.weight", dtype="torch.bfloat16", shape=shape, shards=shards)]
    return TrainerTable(agents=[b"rank0", b"rank1"], tensors=tensors, step=step)


def _simple_regions():
    return [ResolvedRegion(
        tensor_name="layer.weight",
        src_elem_runs=[0, 16],
        dst_addr=0xABCD0000,
        dst_elem_runs=[0, 16],
        element_size=2,
        dst_device_id=0,
    )]


# ---------------------------------------------------------------------------
# TestRLSyncCycleProtocol
#
# Validates that the planner correctly handles multi-step RL loops:
# plan is built once and cached; invalidated when TrainerTable.step changes.
# ---------------------------------------------------------------------------


class TestRLSyncCycleProtocol:
    def test_local_planner_caches_across_steps(self):
        """Same plan_key → cached result, router called exactly once."""
        planner = LocalPlanner()
        table = _simple_table(step=1)
        regions = _simple_regions()

        descs1 = planner.build(regions, table, "plan-rl-v1")
        descs2 = planner.build(regions, table, "plan-rl-v1")
        assert descs1 is descs2

    def test_local_planner_invalidate_triggers_rebuild(self):
        """After invalidate(), a second build() re-routes (different object)."""
        planner = LocalPlanner()
        table = _simple_table(step=1)
        regions = _simple_regions()

        descs1 = planner.build(regions, table, "plan-v1")
        planner.invalidate("plan-v1")
        descs2 = planner.build(regions, table, "plan-v1")
        assert descs1 is not descs2
        assert descs1 == descs2  # same math → same values

    def test_rl_step_advance_invalidates_and_rebuilds(self):
        """Simulate PullRole.refresh() pattern: step advance → invalidate → rebuild."""
        planner = LocalPlanner()
        regions = _simple_regions()

        for step in range(1, 6):
            table = _simple_table(step=step)
            plan_key = f"model-rank0-step{step}"
            if step > 1:
                planner.invalidate(f"model-rank0-step{step - 1}")
            descs = planner.build(regions, table, plan_key)
            total = sum(d.nbytes for d in descs)
            assert total == 16 * 2  # 16 elements × 2 bytes bfloat16

    def test_multiple_workers_build_independent_plans(self):
        """Two RL rollout workers build plans independently; no state sharing."""
        table = _simple_table(step=1)
        regions = _simple_regions()
        planner0, planner1 = LocalPlanner(), LocalPlanner()

        descs0 = planner0.build(regions, table, "plan-worker0")
        descs1 = planner1.build(regions, table, "plan-worker1")
        # Same table, same regions → same descriptors (different objects)
        assert descs0 is not descs1
        assert descs0 == descs1

    def test_grpo_multi_rollout_workers_all_get_full_weights(self):
        """N rollout workers each get descriptors covering all model bytes."""
        NUM_WORKERS = 4
        shape = [8, 8]
        shards = [
            _make_row_shard(0, 0, 4, shape, base_addr=0x1000),
            _make_row_shard(1, 4, 8, shape, base_addr=0x2000),
        ]
        tensors = [TrainerTensor(name="mlp.weight", dtype="torch.bfloat16", shape=shape, shards=shards)]
        table = TrainerTable(agents=[b"r0", b"r1"], tensors=tensors, step=1)
        regions = [ResolvedRegion(
            tensor_name="mlp.weight",
            src_elem_runs=[0, 64],
            dst_addr=0xDEAD0000,
            dst_elem_runs=[0, 64],
            element_size=2,
            dst_device_id=0,
        )]
        expected_bytes = 64 * 2

        for rank in range(NUM_WORKERS):
            planner = LocalPlanner()
            descs = planner.build(regions, table, f"grpo-worker{rank}")
            total = sum(d.nbytes for d in descs)
            assert total == expected_bytes, f"worker {rank} missing bytes"


# ---------------------------------------------------------------------------
# TestRLTPSharding
#
# Verifies correct routing for TP=2 column-sharded (row-parallel) layers as
# they appear in a real PrimeRL / Megatron-LM trainer setup.
# ---------------------------------------------------------------------------


class TestRLTPSharding:
    def _tp2_table(self, step=1):
        """TP=2 table: one row-parallel layer (col-sharded, dim-1 split)."""
        shape = [4, 8]  # 4 rows × 8 cols, split at col 4
        shards = [
            _make_col_shard(0, 0, 4, 0, 4, 8, base_addr=0x1000),
            _make_col_shard(1, 0, 4, 4, 8, 8, base_addr=0x2000),
        ]
        tensors = [TrainerTensor(name="o_proj.weight", dtype="torch.bfloat16", shape=shape, shards=shards)]
        return TrainerTable(agents=[b"tp0", b"tp1"], tensors=tensors, step=step)

    def test_tp2_col_shards_cover_all_bytes(self):
        """Sum of descriptor bytes across both TP shards == full tensor size."""
        table = self._tp2_table()
        regions = [ResolvedRegion(
            tensor_name="o_proj.weight",
            src_elem_runs=[0, 32],
            dst_addr=0xC0000000,
            dst_elem_runs=[0, 32],
            element_size=2,
            dst_device_id=0,
        )]
        descs = route_regions(regions, table)
        total = sum(d.nbytes for d in descs)
        assert total == 32 * 2

    def test_tp2_col_shards_split_across_both_agents(self):
        """Descriptors come from both TP ranks (agent_index 0 and 1)."""
        table = self._tp2_table()
        regions = [ResolvedRegion(
            tensor_name="o_proj.weight",
            src_elem_runs=[0, 32],
            dst_addr=0xC0000000,
            dst_elem_runs=[0, 32],
            element_size=2,
            dst_device_id=0,
        )]
        descs = route_regions(regions, table)
        agents_used = {d.agent_index for d in descs}
        assert agents_used == {0, 1}

    def test_tp2_each_agent_contributes_half_bytes(self):
        """Each TP shard contributes exactly half the tensor bytes."""
        table = self._tp2_table()
        regions = [ResolvedRegion(
            tensor_name="o_proj.weight",
            src_elem_runs=[0, 32],
            dst_addr=0xC0000000,
            dst_elem_runs=[0, 32],
            element_size=2,
            dst_device_id=0,
        )]
        descs = route_regions(regions, table)
        by_agent = {0: 0, 1: 0}
        for d in descs:
            by_agent[d.agent_index] += d.nbytes
        assert by_agent[0] == by_agent[1] == 32  # 16 elems × 2 bytes each

    def test_tp2_col_sharding_step_advance(self):
        """TP=2 plan survives a step advance in an RL loop."""
        planner = LocalPlanner()
        regions = [ResolvedRegion(
            tensor_name="o_proj.weight",
            src_elem_runs=[0, 32],
            dst_addr=0xC0000000,
            dst_elem_runs=[0, 32],
            element_size=2,
            dst_device_id=0,
        )]
        for step in range(1, 4):
            table = self._tp2_table(step=step)
            plan_key = f"tp2-step{step}"
            descs = planner.build(regions, table, plan_key)
            assert sum(d.nbytes for d in descs) == 64

    def test_moe_expert_parallel_sharding(self):
        """MoE: each expert shard is row-sharded across EP ranks."""
        num_experts = 8
        expert_rows = 4
        expert_cols = 16
        shape = [num_experts * expert_rows, expert_cols]
        # EP=2: 4 experts per rank
        shards = [
            _make_row_shard(0, 0, num_experts // 2 * expert_rows, shape, base_addr=0x1000),
            _make_row_shard(1, num_experts // 2 * expert_rows, num_experts * expert_rows, shape, base_addr=0x2000),
        ]
        tensors = [TrainerTensor(
            name="moe.experts.gate_proj.weight",
            dtype="torch.bfloat16",
            shape=shape,
            shards=shards,
        )]
        table = TrainerTable(agents=[b"ep0", b"ep1"], tensors=tensors, step=1)
        total_elems = math.prod(shape)
        regions = [ResolvedRegion(
            tensor_name="moe.experts.gate_proj.weight",
            src_elem_runs=[0, total_elems],
            dst_addr=0xF0000000,
            dst_elem_runs=[0, total_elems],
            element_size=2,
            dst_device_id=0,
        )]
        descs = route_regions(regions, table)
        assert sum(d.nbytes for d in descs) == total_elems * 2


# ---------------------------------------------------------------------------
# TestM2nPlannerRLLoop
#
# Verifies that M2nPlanner handles the RL-loop lifecycle correctly:
# plan is built once per step, invalidated before the next step.
# ---------------------------------------------------------------------------


def _make_m2n_client(plan_id="pid-1"):
    client = MagicMock()
    reg_resp = MagicMock()
    reg_resp.m2n_plan_id = plan_id
    client.register_m2n_worker.return_value = reg_resp

    from unittest.mock import MagicMock as MM
    get_resp = MM()
    get_resp.ready = True
    get_resp.descriptors = []
    client.get_m2n_plan.return_value = get_resp
    return client


class TestM2nPlannerRLLoop:
    def test_plan_rebuilt_per_rl_step(self):
        """After invalidate(), M2nPlanner re-registers with server for each step."""
        client = _make_m2n_client()
        planner = M2nPlanner(
            mx_client=client,
            model_key="policy",
            worker_rank=0,
            total_workers=2,
            nixl_metadata=b"meta",
        )
        table = _simple_table(step=1)
        regions = _simple_regions()

        planner.build_m2n(regions, table, "step1")
        planner.invalidate("step1")
        planner.build_m2n(regions, table, "step2")
        assert client.register_m2n_worker.call_count == 2

    def test_invalidate_notifies_server(self):
        """invalidate() calls invalidate_m2n_plan on the MX server."""
        client = _make_m2n_client()
        planner = M2nPlanner(
            mx_client=client,
            model_key="policy",
            worker_rank=0,
            total_workers=1,
            nixl_metadata=b"",
        )
        planner.build_m2n(_simple_regions(), _simple_table(), "pk")
        planner.invalidate("pk")
        client.invalidate_m2n_plan.assert_called_once_with(model_key="policy")

    def test_fallback_used_when_server_down_during_rl_step(self):
        """If MX server fails mid-RL-run, LocalPlanner keeps training unblocked."""
        client = MagicMock()
        client.register_m2n_worker.side_effect = ConnectionError("server unavailable")

        planner = M2nPlanner(
            mx_client=client,
            model_key="policy",
            worker_rank=0,
            total_workers=2,
            nixl_metadata=b"",
        )
        descs = planner.build(_simple_regions(), _simple_table(), "pk")
        assert isinstance(descs, list)
        assert all(isinstance(d, RdmaDescriptor) for d in descs)
        assert sum(d.nbytes for d in descs) == 16 * 2

    def test_worker_rank_forwarded_to_server(self):
        """Each rollout worker sends its own rank to the M2N barrier."""
        for rank in range(4):
            client = _make_m2n_client(plan_id=f"plan-{rank}")
            planner = M2nPlanner(
                mx_client=client,
                model_key="policy",
                worker_rank=rank,
                total_workers=4,
                nixl_metadata=b"meta",
            )
            planner.build_m2n(_simple_regions(), _simple_table(), f"pk-{rank}")
            kwargs = client.register_m2n_worker.call_args.kwargs
            assert kwargs["worker_rank"] == rank
            assert kwargs["total_workers"] == 4
