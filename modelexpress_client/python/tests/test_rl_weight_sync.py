# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RL integration tests for weight_transfer (no NIXL, no GPU, no real model)."""

import math
from unittest.mock import MagicMock

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


def _make_row_shard(agent_index, row_start, row_end, shape, base_addr=0x10000):
    elem_size = 2
    row_bytes = math.prod(shape[1:]) * elem_size if len(shape) > 1 else elem_size
    addr = base_addr + row_start * row_bytes
    return TrainerShard(
        agent_index=agent_index, row_start=row_start, row_end=row_end,
        device_addr=addr, row_bytes=row_bytes, device_id=agent_index,
    )


def _make_col_shard(agent_index, row_start, row_end, col_start, col_end, full_cols, base_addr=0x20000):
    elem_size = 2
    row_bytes = (col_end - col_start) * elem_size
    addr = base_addr + agent_index * (row_end - row_start) * row_bytes
    return TrainerShard(
        agent_index=agent_index, row_start=row_start, row_end=row_end,
        col_start=col_start, col_end=col_end,
        device_addr=addr, row_bytes=row_bytes, device_id=agent_index,
    )


def _simple_table(step=1):
    shape = [4, 4]
    shards = [
        _make_row_shard(0, 0, 2, shape, base_addr=0x1000),
        _make_row_shard(1, 2, 4, shape, base_addr=0x2000),
    ]
    tensors = [TrainerTensor(name="layer.weight", dtype="torch.bfloat16", shape=shape, shards=shards)]
    return TrainerTable(agents=[b"rank0", b"rank1"], tensors=tensors, step=step)


def _simple_regions():
    return [ResolvedRegion(
        tensor_name="layer.weight", src_elem_runs=[0, 16],
        dst_addr=0xABCD0000, dst_elem_runs=[0, 16], element_size=2, dst_device_id=0,
    )]


class TestRLSyncCycleProtocol:
    def test_local_planner_caches_across_steps(self):
        planner = LocalPlanner()
        table, regions = _simple_table(step=1), _simple_regions()
        descs1 = planner.build(regions, table, "plan-rl-v1")
        descs2 = planner.build(regions, table, "plan-rl-v1")
        assert descs1 is descs2

    def test_local_planner_invalidate_triggers_rebuild(self):
        """After invalidate(), build() re-routes: different object, same values."""
        planner = LocalPlanner()
        table, regions = _simple_table(step=1), _simple_regions()
        descs1 = planner.build(regions, table, "plan-v1")
        planner.invalidate("plan-v1")
        descs2 = planner.build(regions, table, "plan-v1")
        assert descs1 is not descs2
        assert descs1 == descs2

    def test_rl_step_advance_invalidates_and_rebuilds(self):
        planner = LocalPlanner()
        regions = _simple_regions()
        for step in range(1, 6):
            table = _simple_table(step=step)
            plan_key = f"model-rank0-step{step}"
            if step > 1:
                planner.invalidate(f"model-rank0-step{step - 1}")
            descs = planner.build(regions, table, plan_key)
            assert sum(d.nbytes for d in descs) == 16 * 2

    def test_multiple_workers_build_independent_plans(self):
        table, regions = _simple_table(step=1), _simple_regions()
        planner0, planner1 = LocalPlanner(), LocalPlanner()
        descs0 = planner0.build(regions, table, "plan-worker0")
        descs1 = planner1.build(regions, table, "plan-worker1")
        assert descs0 is not descs1
        assert descs0 == descs1

    def test_grpo_multi_rollout_workers_all_get_full_weights(self):
        NUM_WORKERS = 4
        shape = [8, 8]
        shards = [
            _make_row_shard(0, 0, 4, shape, base_addr=0x1000),
            _make_row_shard(1, 4, 8, shape, base_addr=0x2000),
        ]
        tensors = [TrainerTensor(name="mlp.weight", dtype="torch.bfloat16", shape=shape, shards=shards)]
        table = TrainerTable(agents=[b"r0", b"r1"], tensors=tensors, step=1)
        regions = [ResolvedRegion(
            tensor_name="mlp.weight", src_elem_runs=[0, 64],
            dst_addr=0xDEAD0000, dst_elem_runs=[0, 64], element_size=2, dst_device_id=0,
        )]
        expected_bytes = 64 * 2
        for rank in range(NUM_WORKERS):
            planner = LocalPlanner()
            descs = planner.build(regions, table, f"grpo-worker{rank}")
            assert sum(d.nbytes for d in descs) == expected_bytes, f"worker {rank} missing bytes"


class TestRLTPSharding:
    def _tp2_table(self, step=1):
        shape = [4, 8]  # 4 rows × 8 cols, split at col 4
        shards = [
            _make_col_shard(0, 0, 4, 0, 4, 8, base_addr=0x1000),
            _make_col_shard(1, 0, 4, 4, 8, 8, base_addr=0x2000),
        ]
        tensors = [TrainerTensor(name="o_proj.weight", dtype="torch.bfloat16", shape=shape, shards=shards)]
        return TrainerTable(agents=[b"tp0", b"tp1"], tensors=tensors, step=step)

    def _o_proj_region(self):
        return [ResolvedRegion(
            tensor_name="o_proj.weight", src_elem_runs=[0, 32],
            dst_addr=0xC0000000, dst_elem_runs=[0, 32], element_size=2, dst_device_id=0,
        )]

    def test_tp2_col_shards_cover_all_bytes(self):
        descs = route_regions(self._o_proj_region(), self._tp2_table())
        assert sum(d.nbytes for d in descs) == 32 * 2

    def test_tp2_col_shards_split_across_both_agents(self):
        descs = route_regions(self._o_proj_region(), self._tp2_table())
        assert {d.agent_index for d in descs} == {0, 1}

    def test_tp2_each_agent_contributes_half_bytes(self):
        descs = route_regions(self._o_proj_region(), self._tp2_table())
        by_agent = {0: 0, 1: 0}
        for d in descs:
            by_agent[d.agent_index] += d.nbytes
        assert by_agent[0] == by_agent[1] == 32

    def test_tp2_col_sharding_step_advance(self):
        planner = LocalPlanner()
        regions = self._o_proj_region()
        for step in range(1, 4):
            table = self._tp2_table(step=step)
            descs = planner.build(regions, table, f"tp2-step{step}")
            assert sum(d.nbytes for d in descs) == 64

    def test_moe_expert_parallel_sharding(self):
        num_experts, expert_rows, expert_cols = 8, 4, 16
        shape = [num_experts * expert_rows, expert_cols]
        shards = [
            _make_row_shard(0, 0, num_experts // 2 * expert_rows, shape, base_addr=0x1000),
            _make_row_shard(1, num_experts // 2 * expert_rows, num_experts * expert_rows, shape, base_addr=0x2000),
        ]
        tensors = [TrainerTensor(
            name="moe.experts.gate_proj.weight", dtype="torch.bfloat16", shape=shape, shards=shards,
        )]
        table = TrainerTable(agents=[b"ep0", b"ep1"], tensors=tensors, step=1)
        total_elems = math.prod(shape)
        regions = [ResolvedRegion(
            tensor_name="moe.experts.gate_proj.weight", src_elem_runs=[0, total_elems],
            dst_addr=0xF0000000, dst_elem_runs=[0, total_elems], element_size=2, dst_device_id=0,
        )]
        assert sum(d.nbytes for d in route_regions(regions, table)) == total_elems * 2


def _make_m2n_client(plan_id="pid-1"):
    client = MagicMock()
    reg_resp = MagicMock()
    reg_resp.m2n_plan_id = plan_id
    client.register_m2n_worker.return_value = reg_resp
    get_resp = MagicMock()
    get_resp.ready = True
    get_resp.descriptors = []
    client.get_m2n_plan.return_value = get_resp
    return client


class TestM2nPlannerRLLoop:
    def test_plan_rebuilt_per_rl_step(self):
        """After invalidate(), M2nPlanner re-registers with server for each step."""
        client = _make_m2n_client()
        planner = M2nPlanner(
            mx_client=client, model_key="policy",
            worker_rank=0, total_workers=2, nixl_metadata=b"meta",
        )
        planner.build_m2n(_simple_regions(), _simple_table(step=1), "step1")
        planner.invalidate("step1")
        planner.build_m2n(_simple_regions(), _simple_table(step=1), "step2")
        assert client.register_m2n_worker.call_count == 2

    def test_invalidate_notifies_server(self):
        client = _make_m2n_client()
        planner = M2nPlanner(
            mx_client=client, model_key="policy",
            worker_rank=0, total_workers=1, nixl_metadata=b"",
        )
        planner.build_m2n(_simple_regions(), _simple_table(), "pk")
        planner.invalidate("pk")
        client.invalidate_m2n_plan.assert_called_once_with(model_key="policy")

    def test_fallback_used_when_server_down_during_rl_step(self):
        """If MX server fails mid-RL-run, LocalPlanner keeps training unblocked."""
        client = MagicMock()
        client.register_m2n_worker.side_effect = ConnectionError("server unavailable")
        planner = M2nPlanner(
            mx_client=client, model_key="policy",
            worker_rank=0, total_workers=2, nixl_metadata=b"",
        )
        descs = planner.build(_simple_regions(), _simple_table(), "pk")
        assert isinstance(descs, list)
        assert all(isinstance(d, RdmaDescriptor) for d in descs)
        assert sum(d.nbytes for d in descs) == 16 * 2

    def test_worker_rank_forwarded_to_server(self):
        for rank in range(4):
            client = _make_m2n_client(plan_id=f"plan-{rank}")
            planner = M2nPlanner(
                mx_client=client, model_key="policy",
                worker_rank=rank, total_workers=4, nixl_metadata=b"meta",
            )
            planner.build_m2n(_simple_regions(), _simple_table(), f"pk-{rank}")
            kwargs = client.register_m2n_worker.call_args.kwargs
            assert kwargs["worker_rank"] == rank
            assert kwargs["total_workers"] == 4
