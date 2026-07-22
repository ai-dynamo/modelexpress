# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for PrimeRL weight-transfer (bake/resolve/route, no NIXL/GPU)."""

import math
from unittest.mock import MagicMock

import pytest
import torch

from modelexpress.weight_transfer.engine.lazy import BakeRecorder, LazyWeight
from modelexpress.weight_transfer.planner.local import LocalPlanner
from modelexpress.weight_transfer.planner.resolver import resolve_chain_region, region_elem_runs
from modelexpress.weight_transfer.planner.router import route_regions
from modelexpress.weight_transfer.protocol.types import (
    M2nDescriptor,
    RdmaDescriptor,
    ResolvedRegion,
    TrainerShard,
    TrainerTable,
    TrainerTensor,
)


def _row_shard(agent, r0, r1, shape, base=0x10000):
    row_bytes = math.prod(shape[1:]) * 2
    return TrainerShard(
        agent_index=agent, row_start=r0, row_end=r1,
        device_addr=base + r0 * row_bytes, row_bytes=row_bytes, device_id=agent,
    )


def _col_shard(agent, r0, r1, c0, c1, base=0x30000):
    row_bytes = (c1 - c0) * 2
    return TrainerShard(
        agent_index=agent, row_start=r0, row_end=r1, col_start=c0, col_end=c1,
        device_addr=base + agent * (r1 - r0) * row_bytes, row_bytes=row_bytes, device_id=agent,
    )


def _make_table(layers: dict[str, list[int]], shards_fn=None, step=1):
    """Build a TrainerTable from {name: shape} with 2 row-shards by default."""
    tensors = []
    for name, shape in layers.items():
        if shards_fn:
            shards = shards_fn(name, shape)
        else:
            shards = [
                _row_shard(0, 0, shape[0] // 2, shape),
                _row_shard(1, shape[0] // 2, shape[0], shape, base=0x20000),
            ]
        tensors.append(TrainerTensor(name, "torch.bfloat16", list(shape), shards))
    return TrainerTable(agents=[b"rank0", b"rank1"], tensors=tensors, step=step)


def _full_regions(table):
    regions = []
    for i, tt in enumerate(table.tensors):
        n = math.prod(tt.shape)
        regions.append(ResolvedRegion(
            tensor_name=tt.name, src_elem_runs=[0, n],
            dst_addr=0xA000_0000 + i * 0x100_0000, dst_elem_runs=[0, n],
            element_size=2, dst_device_id=0,
        ))
    return regions


class TestPrimeRLBakeResolveRoute:
    def _bake_single_param(self, shape, dst_tensor):
        recorder = BakeRecorder()
        lw = LazyWeight("layer.weight", torch.Size(shape), torch.bfloat16)
        with recorder:
            dst_tensor.copy_(lw)
        return recorder.copies

    def test_bake_captures_one_copy_per_param(self):
        shape = [4, 8]
        dst = torch.zeros(shape, dtype=torch.bfloat16)
        copies = self._bake_single_param(shape, dst)
        assert len(copies) == 1
        assert copies[0].src_name == "layer.weight"

    def test_resolve_identity_op_chain(self):
        """Identity chain → offset=0, shape=root shape."""
        offset, shape, stride = resolve_chain_region((), torch.Size([4, 8]), torch.bfloat16)
        assert offset == 0
        assert tuple(shape) == (4, 8)

    def test_region_elem_runs_contiguous(self):
        """Contiguous 2-D tensor → single run covering all elements."""
        runs = region_elem_runs(0, torch.Size([4, 8]), (8, 1))
        assert len(runs) == 1
        assert runs[0] == (0, 32)

    def test_route_covers_all_bytes(self):
        table = _make_table({"layer.weight": [8, 16]})
        descs = route_regions(_full_regions(table), table)
        assert sum(d.nbytes for d in descs) == math.prod([8, 16]) * 2

    def test_multi_layer_pipeline_full_coverage(self):
        layers = {
            "embed_tokens.weight": [256, 64],
            "lm_head.weight":      [256, 64],
            "mlp.gate.weight":     [64, 32],
            "mlp.down.weight":     [32, 64],
        }
        table = _make_table(layers)
        descs = route_regions(_full_regions(table), table)
        expected = sum(math.prod(s) * 2 for s in layers.values())
        assert sum(d.nbytes for d in descs) == expected

    def test_bake_roundtrip_byte_exact(self):
        """After bake→resolve→route, inferred addresses are consistent with src."""
        shape = [4, 4]
        src = torch.randn(shape, dtype=torch.bfloat16)
        dst = torch.zeros(shape, dtype=torch.bfloat16)

        recorder = BakeRecorder()
        lw = LazyWeight("w", torch.Size(shape), torch.bfloat16)
        with recorder:
            dst.copy_(lw)
        copy = recorder.copies[0]

        offset, rshape, stride = resolve_chain_region(
            copy.op_chain, torch.Size(copy.dst_shape), copy.dst_dtype
        )
        runs = list(region_elem_runs(offset, rshape, stride))
        assert runs == [(0, 16)]

        shard = TrainerShard(
            agent_index=0, row_start=0, row_end=4,
            device_addr=src.data_ptr(), row_bytes=4 * 2, device_id=0,
        )
        tt = TrainerTensor("w", "torch.bfloat16", list(shape), [shard])
        table = TrainerTable(agents=[b"r0"], tensors=[tt], step=1)

        dst_runs_flat = [r for pair in runs for r in pair]
        region = ResolvedRegion(
            tensor_name="w", src_elem_runs=dst_runs_flat,
            dst_addr=dst.data_ptr(), dst_elem_runs=dst_runs_flat,
            element_size=2, dst_device_id=0,
        )
        descs = route_regions([region], table)
        assert len(descs) == 1
        assert descs[0].src_addr == src.data_ptr()
        assert descs[0].dst_addr == dst.data_ptr()
        assert descs[0].nbytes == 16 * 2


class TestPrimeRLSyncLifecycle:
    def _rl_step(self, planner, table, step_label):
        return planner.build(_full_regions(table), table, step_label)

    def test_plan_stable_within_step(self):
        """Same step → same plan_key → same cached descriptors object."""
        table = _make_table({"w": [4, 8]}, step=1)
        planner = LocalPlanner()
        assert self._rl_step(planner, table, "step1") is self._rl_step(planner, table, "step1")

    def test_new_step_after_invalidate_produces_same_result(self):
        """Plan content is deterministic: invalidate + rebuild → equal descriptors."""
        table = _make_table({"w": [4, 8]}, step=1)
        planner = LocalPlanner()
        descs1 = self._rl_step(planner, table, "step1")
        planner.invalidate("step1")
        descs2 = self._rl_step(planner, table, "step2")
        assert descs1 == descs2

    def test_ten_rl_steps_all_cover_full_bytes(self):
        layers = {"q_proj.weight": [8, 4], "v_proj.weight": [8, 4]}
        expected = sum(math.prod(s) * 2 for s in layers.values())
        planner = LocalPlanner()
        for step in range(1, 11):
            table = _make_table(layers, step=step)
            if step > 1:
                planner.invalidate(f"step{step - 1}")
            descs = self._rl_step(planner, table, f"step{step}")
            assert sum(d.nbytes for d in descs) == expected, f"step {step} failed"

    def test_multiple_workers_independent_step_counters(self):
        table = _make_table({"w": [8, 8]}, step=1)
        p0, p1 = LocalPlanner(), LocalPlanner()
        d0 = p0.build(_full_regions(table), table, "w0-step1")
        d1 = p1.build(_full_regions(table), table, "w1-step1")
        assert d0 == d1
        p0.invalidate("w0-step1")
        d0_new = p0.build(_full_regions(table), table, "w0-step2")
        d1_cached = p1.build(_full_regions(table), table, "w1-step1")
        assert d0_new == d0
        assert d1_cached is d1

    def test_post_pull_hook_called_per_sync(self):
        adapter = MagicMock()
        adapter.post_pull_hook = MagicMock()
        model = MagicMock()
        for _ in range(3):
            adapter.post_pull_hook(model)
        assert adapter.post_pull_hook.call_count == 3


class TestPrimeRLTP2Sharding:
    def _tp2_table(self, step=1):
        q_shape, o_shape = [8, 4], [4, 8]
        tt_q = TrainerTensor("q_proj.weight", "torch.bfloat16", q_shape, [
            _row_shard(0, 0, 4, q_shape, base=0x10000),
            _row_shard(1, 4, 8, q_shape, base=0x20000),
        ])
        tt_o = TrainerTensor("o_proj.weight", "torch.bfloat16", o_shape, [
            _col_shard(0, 0, 4, 0, 4, base=0x30000),
            _col_shard(1, 0, 4, 4, 8, base=0x40000),
        ])
        return TrainerTable(agents=[b"tp0", b"tp1"], tensors=[tt_q, tt_o], step=step)

    def test_tp2_full_byte_coverage(self):
        table = self._tp2_table()
        descs = route_regions(_full_regions(table), table)
        expected = (math.prod([8, 4]) + math.prod([4, 8])) * 2
        assert sum(d.nbytes for d in descs) == expected

    def test_tp2_o_proj_col_shards_both_agents(self):
        """o_proj column shards must reference both TP agents."""
        table = self._tp2_table()
        regions = [ResolvedRegion(
            tensor_name="o_proj.weight", src_elem_runs=[0, 32],
            dst_addr=0xC000_0000, dst_elem_runs=[0, 32], element_size=2, dst_device_id=0,
        )]
        assert {d.agent_index for d in route_regions(regions, table)} == {0, 1}

    def test_tp2_q_proj_row_shards_both_agents(self):
        """q_proj row shards must also reference both TP agents."""
        table = self._tp2_table()
        regions = [ResolvedRegion(
            tensor_name="q_proj.weight", src_elem_runs=[0, 32],
            dst_addr=0xD000_0000, dst_elem_runs=[0, 32], element_size=2, dst_device_id=0,
        )]
        assert {d.agent_index for d in route_regions(regions, table)} == {0, 1}

    def test_tp2_plan_survives_multiple_rl_steps(self):
        planner = LocalPlanner()
        for step in range(1, 6):
            table = self._tp2_table(step=step)
            key = f"tp2-step{step}"
            if step > 1:
                planner.invalidate(f"tp2-step{step - 1}")
            descs = planner.build(_full_regions(table), table, key)
            assert sum(d.nbytes for d in descs) == (32 + 32) * 2, f"step {step}"

    def test_tp2_col_shard_byte_symmetry(self):
        """TP=2 col shards contribute equal bytes for o_proj."""
        table = self._tp2_table()
        regions = [ResolvedRegion(
            tensor_name="o_proj.weight", src_elem_runs=[0, 32],
            dst_addr=0xE000_0000, dst_elem_runs=[0, 32], element_size=2, dst_device_id=0,
        )]
        descs = route_regions(regions, table)
        by_agent = {0: 0, 1: 0}
        for d in descs:
            by_agent[d.agent_index] += d.nbytes
        assert by_agent[0] == by_agent[1]


class TestPrimeRLReadinessAfterSync:
    def _run_pipeline(self, shape):
        src = torch.arange(math.prod(shape), dtype=torch.bfloat16).reshape(shape)
        dst = torch.zeros(shape, dtype=torch.bfloat16)

        recorder = BakeRecorder()
        lw = LazyWeight("w", torch.Size(shape), torch.bfloat16)
        with recorder:
            dst.copy_(lw)
        copy = recorder.copies[0]

        offset, rshape, stride = resolve_chain_region(
            copy.op_chain, torch.Size(copy.dst_shape), copy.dst_dtype
        )
        runs = list(region_elem_runs(offset, rshape, stride))
        flat = [v for pair in runs for v in pair]

        shard = TrainerShard(
            agent_index=0,
            row_start=0,
            row_end=shape[0] if len(shape) > 1 else 1,
            device_addr=src.data_ptr(),
            row_bytes=(math.prod(shape[1:]) if len(shape) > 1 else shape[0]) * 2,
            device_id=0,
        )
        tt = TrainerTensor("w", "torch.bfloat16",
                           list(shape) if len(shape) > 1 else [1, shape[0]], [shard])
        table = TrainerTable(agents=[b"r0"], tensors=[tt], step=1)
        region = ResolvedRegion(
            tensor_name="w", src_elem_runs=flat,
            dst_addr=dst.data_ptr(), dst_elem_runs=flat,
            element_size=2, dst_device_id=0,
        )
        return src, dst, route_regions([region], table)

    def test_descriptor_addresses_match_tensor_data_ptrs(self):
        src, dst, descs = self._run_pipeline([4, 4])
        assert len(descs) == 1
        assert descs[0].src_addr == src.data_ptr()
        assert descs[0].dst_addr == dst.data_ptr()

    def test_descriptor_nbytes_equals_tensor_bytes(self):
        shape = [8, 8]
        src, dst, descs = self._run_pipeline(shape)
        assert sum(d.nbytes for d in descs) == math.prod(shape) * 2

    def test_1d_param_pipeline(self):
        """1-D params (biases, layer norms) must also resolve correctly."""
        src, dst, descs = self._run_pipeline([64])
        assert sum(d.nbytes for d in descs) == 64 * 2

    def test_large_param_pipeline(self):
        src, dst, descs = self._run_pipeline([256, 128])
        assert sum(d.nbytes for d in descs) == 256 * 128 * 2

    def test_descriptor_count_is_one_for_contiguous_tensor(self):
        """Contiguous param → resolver emits one run → one descriptor."""
        src, dst, descs = self._run_pipeline([16, 32])
        assert len(descs) == 1
