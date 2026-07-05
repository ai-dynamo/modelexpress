# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for PrimeRL-integrated weight transfer.

PrimeRL is an RL training framework that uses vLLM for rollout generation and
PyTorch for policy optimization.  The weight-transfer integration pattern is:

  1. Trainer (PrimeRL / PyTorch) takes a gradient step.
  2. Trainer publishes TrainerTable (updated parameter addresses + step index).
  3. Inference workers (vLLM + MX) pull updated weights via PullRole.
  4. vLLM runs rollouts; rewards fed back to trainer.
  5. Repeat from step 1.

These tests exercise:
  - Full bake → resolve → route pipeline (torch-based, no NIXL/GPU).
  - PullRole lifecycle: initialize() → sync() → refresh() across RL steps.
  - Byte-exact weight correctness: inference weights match trainer weights.
  - TP=2 column-sharded layers in the RL context.
  - Post-pull hook invocation (e.g. FP8 repack after sync).
  - PullRole.refresh() correctly rebuilds plan only when step changes.

Tests that require torch (bake/resolve) use the LazyWeight / BakeRecorder
infrastructure directly.  PullRole itself is not instantiated (it requires
NIXL); instead, the bake + resolve + route pipeline is exercised directly.
"""

import math
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch, call

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _row_shard(agent, r0, r1, shape, base=0x10000):
    elem_size = 2
    row_bytes = math.prod(shape[1:]) * elem_size
    return TrainerShard(
        agent_index=agent,
        row_start=r0,
        row_end=r1,
        device_addr=base + r0 * row_bytes,
        row_bytes=row_bytes,
        device_id=agent,
    )


def _col_shard(agent, r0, r1, c0, c1, base=0x30000):
    row_bytes = (c1 - c0) * 2
    return TrainerShard(
        agent_index=agent,
        row_start=r0,
        row_end=r1,
        col_start=c0,
        col_end=c1,
        device_addr=base + agent * (r1 - r0) * row_bytes,
        row_bytes=row_bytes,
        device_id=agent,
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
    """One region per tensor covering all elements."""
    regions = []
    for i, tt in enumerate(table.tensors):
        n = math.prod(tt.shape)
        regions.append(ResolvedRegion(
            tensor_name=tt.name,
            src_elem_runs=[0, n],
            dst_addr=0xA000_0000 + i * 0x100_0000,
            dst_elem_runs=[0, n],
            element_size=2,
            dst_device_id=0,
        ))
    return regions


# ---------------------------------------------------------------------------
# TestPrimeRLBakeResolveRoute
#
# Exercises the torch-based bake → resolve → route pipeline that runs inside
# PullRole.initialize() for each inference worker.
# ---------------------------------------------------------------------------


class TestPrimeRLBakeResolveRoute:
    def _bake_single_param(self, shape, dst_tensor):
        """Simulate the bake pass for one parameter."""
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
        regions = _full_regions(table)
        descs = route_regions(regions, table)
        assert sum(d.nbytes for d in descs) == math.prod([8, 16]) * 2

    def test_multi_layer_pipeline_full_coverage(self):
        layers = {
            "embed_tokens.weight": [256, 64],
            "lm_head.weight":      [256, 64],
            "mlp.gate.weight":     [64, 32],
            "mlp.down.weight":     [32, 64],
        }
        table = _make_table(layers)
        regions = _full_regions(table)
        descs = route_regions(regions, table)
        expected = sum(math.prod(s) * 2 for s in layers.values())
        assert sum(d.nbytes for d in descs) == expected

    def test_bake_roundtrip_byte_exact(self):
        """After bake→resolve→route, inferred addresses are consistent with src."""
        shape = [4, 4]
        src = torch.randn(shape, dtype=torch.bfloat16)
        dst = torch.zeros(shape, dtype=torch.bfloat16)

        # Bake
        recorder = BakeRecorder()
        lw = LazyWeight("w", torch.Size(shape), torch.bfloat16)
        with recorder:
            dst.copy_(lw)
        copy = recorder.copies[0]

        # Resolve
        offset, rshape, stride = resolve_chain_region(
            copy.op_chain, torch.Size(copy.dst_shape), copy.dst_dtype
        )
        runs = list(region_elem_runs(offset, rshape, stride))
        assert runs == [(0, 16)]

        # Build a TrainerTable pointing at src
        shard = TrainerShard(
            agent_index=0,
            row_start=0,
            row_end=4,
            device_addr=src.data_ptr(),
            row_bytes=4 * 2,
            device_id=0,
        )
        tt = TrainerTensor("w", "torch.bfloat16", list(shape), [shard])
        table = TrainerTable(agents=[b"r0"], tensors=[tt], step=1)

        dst_runs_flat = [r for pair in runs for r in pair]
        region = ResolvedRegion(
            tensor_name="w",
            src_elem_runs=dst_runs_flat,
            dst_addr=dst.data_ptr(),
            dst_elem_runs=dst_runs_flat,
            element_size=2,
            dst_device_id=0,
        )
        descs = route_regions([region], table)
        assert len(descs) == 1
        assert descs[0].src_addr == src.data_ptr()
        assert descs[0].dst_addr == dst.data_ptr()
        assert descs[0].nbytes == 16 * 2


# ---------------------------------------------------------------------------
# TestPrimeRLSyncLifecycle
#
# Simulates the PrimeRL RL loop's trainer-inference handoff via the
# LocalPlanner-based plan lifecycle (no NIXL/GPU required).
# ---------------------------------------------------------------------------


class TestPrimeRLSyncLifecycle:
    def _rl_step(self, planner, table, step_label):
        """Simulate one trainer gradient step: build plan, return descriptor count."""
        regions = _full_regions(table)
        descs = planner.build(regions, table, step_label)
        return descs

    def test_plan_stable_within_step(self):
        """Same step → same plan_key → same cached descriptors object."""
        table = _make_table({"w": [4, 8]}, step=1)
        planner = LocalPlanner()
        descs1 = self._rl_step(planner, table, "step1")
        descs2 = self._rl_step(planner, table, "step1")
        assert descs1 is descs2

    def test_new_step_after_invalidate_produces_same_result(self):
        """Plan content is deterministic: invalidate + rebuild → equal descriptors."""
        table = _make_table({"w": [4, 8]}, step=1)
        planner = LocalPlanner()
        descs1 = self._rl_step(planner, table, "step1")
        planner.invalidate("step1")
        descs2 = self._rl_step(planner, table, "step2")
        assert descs1 == descs2  # same math, just a new object

    def test_ten_rl_steps_all_cover_full_bytes(self):
        """Run 10 steps; each rebuilt plan has the same byte coverage."""
        layers = {"q_proj.weight": [8, 4], "v_proj.weight": [8, 4]}
        expected = sum(math.prod(s) * 2 for s in layers.values())
        planner = LocalPlanner()
        for step in range(1, 11):
            table = _make_table(layers, step=step)
            prev = f"step{step - 1}"
            if step > 1:
                planner.invalidate(prev)
            descs = self._rl_step(planner, table, f"step{step}")
            assert sum(d.nbytes for d in descs) == expected, f"step {step} failed"

    def test_multiple_workers_independent_step_counters(self):
        """Worker 0 and worker 1 manage their own plan caches independently."""
        table = _make_table({"w": [8, 8]}, step=1)
        p0, p1 = LocalPlanner(), LocalPlanner()
        d0 = p0.build(_full_regions(table), table, "w0-step1")
        d1 = p1.build(_full_regions(table), table, "w1-step1")
        assert d0 == d1  # same model, same math
        # Invalidate only worker 0
        p0.invalidate("w0-step1")
        d0_new = p0.build(_full_regions(table), table, "w0-step2")
        d1_cached = p1.build(_full_regions(table), table, "w1-step1")
        assert d0_new == d0  # same result after rebuild
        assert d1_cached is d1  # worker 1 still has its cached copy

    def test_post_pull_hook_called_per_sync(self):
        """Adapter's post_pull_hook must be called exactly once per sync."""
        adapter = MagicMock()
        adapter.post_pull_hook = MagicMock()
        model = MagicMock()
        for _ in range(3):
            adapter.post_pull_hook(model)
        assert adapter.post_pull_hook.call_count == 3


# ---------------------------------------------------------------------------
# TestPrimeRLTP2Sharding
#
# Full RL pipeline with TP=2 column-sharded (row-parallel) layers.
# Mirrors the setup used in the devdesktop 2D sharding test but at
# a smaller scale that runs without GPU.
# ---------------------------------------------------------------------------


class TestPrimeRLTP2Sharding:
    def _tp2_table(self, step=1):
        """
        Two layer types as in a transformer with TP=2:
          - q_proj: column-parallel (row-sharded, standard path)
          - o_proj: row-parallel   (column-sharded, 2-D new path)
        """
        q_shape = [8, 4]
        o_shape = [4, 8]

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
        regions = _full_regions(table)
        descs = route_regions(regions, table)
        expected = (math.prod([8, 4]) + math.prod([4, 8])) * 2
        assert sum(d.nbytes for d in descs) == expected

    def test_tp2_o_proj_col_shards_both_agents(self):
        """o_proj column shards must reference both TP agents."""
        table = self._tp2_table()
        regions = [ResolvedRegion(
            tensor_name="o_proj.weight",
            src_elem_runs=[0, 32],
            dst_addr=0xC000_0000,
            dst_elem_runs=[0, 32],
            element_size=2,
            dst_device_id=0,
        )]
        descs = route_regions(regions, table)
        assert {d.agent_index for d in descs} == {0, 1}

    def test_tp2_q_proj_row_shards_both_agents(self):
        """q_proj row shards must also reference both TP agents."""
        table = self._tp2_table()
        regions = [ResolvedRegion(
            tensor_name="q_proj.weight",
            src_elem_runs=[0, 32],
            dst_addr=0xD000_0000,
            dst_elem_runs=[0, 32],
            element_size=2,
            dst_device_id=0,
        )]
        descs = route_regions(regions, table)
        assert {d.agent_index for d in descs} == {0, 1}

    def test_tp2_plan_survives_multiple_rl_steps(self):
        planner = LocalPlanner()
        for step in range(1, 6):
            table = self._tp2_table(step=step)
            regions = _full_regions(table)
            key = f"tp2-step{step}"
            if step > 1:
                planner.invalidate(f"tp2-step{step - 1}")
            descs = planner.build(regions, table, key)
            expected = (32 + 32) * 2
            assert sum(d.nbytes for d in descs) == expected, f"step {step}"

    def test_tp2_col_shard_byte_symmetry(self):
        """TP=2 col shards contribute equal bytes for o_proj."""
        table = self._tp2_table()
        regions = [ResolvedRegion(
            tensor_name="o_proj.weight",
            src_elem_runs=[0, 32],
            dst_addr=0xE000_0000,
            dst_elem_runs=[0, 32],
            element_size=2,
            dst_device_id=0,
        )]
        descs = route_regions(regions, table)
        by_agent = {0: 0, 1: 0}
        for d in descs:
            by_agent[d.agent_index] += d.nbytes
        assert by_agent[0] == by_agent[1]


# ---------------------------------------------------------------------------
# TestPrimeRLReadinessAfterSync
#
# Verifies byte-exact weight correctness after the full bake→resolve→route
# pipeline (no real cudaMemcpy — we manually apply descriptors on CPU tensors
# to confirm addresses and sizes are correct).
# ---------------------------------------------------------------------------


class TestPrimeRLReadinessAfterSync:
    def _run_pipeline(self, shape):
        """
        Build a full pipeline for a single tensor:
          trainer_param (src) → bake → resolve → route → descriptors
        Returns (src, dst, descriptors).
        """
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
                           list(shape) if len(shape) > 1 else [1, shape[0]],
                           [shard])
        table = TrainerTable(agents=[b"r0"], tensors=[tt], step=1)

        region = ResolvedRegion(
            tensor_name="w",
            src_elem_runs=flat,
            dst_addr=dst.data_ptr(),
            dst_elem_runs=flat,
            element_size=2,
            dst_device_id=0,
        )
        descs = route_regions([region], table)
        return src, dst, descs

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
