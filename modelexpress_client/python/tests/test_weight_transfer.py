# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the weight_transfer package (lazy bake + pull plan)."""

import math
import pytest
import torch

from modelexpress.weight_transfer.protocol.ops import OpSpec
from modelexpress.weight_transfer.protocol.types import (
    TrainerShard,
    TrainerTensor,
    TrainerTable,
)
from modelexpress.weight_transfer.protocol.serialization import (
    encode_trainer_table,
    decode_trainer_table,
)
from modelexpress.weight_transfer.engine.lazy import (
    BakeRecorder,
    LazyWeight,
    RecordedCopy,
)
from modelexpress.weight_transfer.planner.resolver import (
    apply_chain,
    resolve_chain_region,
    region_elem_runs,
)
from modelexpress.weight_transfer.planner.router import route_regions
from modelexpress.weight_transfer.protocol.types import ResolvedRegion


# ---------------------------------------------------------------------------
# protocol/ops.py + planner/resolver.py
# ---------------------------------------------------------------------------


class TestApplyChain:
    def test_empty_chain_returns_input(self):
        t = torch.randn(4, 8)
        assert apply_chain(t, ()).data_ptr() == t.data_ptr()

    def test_narrow(self):
        t = torch.randn(8, 16)
        chain = (OpSpec("narrow", (0, 2, 4), {}),)
        out = apply_chain(t, chain)
        assert out.shape == torch.Size([4, 16])

    def test_transpose(self):
        t = torch.randn(4, 8)
        chain = (OpSpec("transpose", (0, 1), {}),)
        out = apply_chain(t, chain)
        assert out.shape == torch.Size([8, 4])


class TestResolveChainRegion:
    def test_identity(self):
        offset, shape, stride = resolve_chain_region(
            (), torch.Size([4, 8]), torch.float32
        )
        assert offset == 0
        assert tuple(shape) == (4, 8)

    def test_narrow_dim0(self):
        chain = (OpSpec("narrow", (0, 2, 2), {}),)
        offset, shape, stride = resolve_chain_region(
            chain, torch.Size([8, 16]), torch.bfloat16
        )
        # narrow(dim=0, start=2, length=2) on shape (8,16) with stride (16,1)
        assert offset == 2 * 16  # 32
        assert tuple(shape) == (2, 16)

    def test_view_preserves_elements(self):
        chain = (OpSpec("view", (32,), {}),)
        offset, shape, stride = resolve_chain_region(
            chain, torch.Size([4, 8]), torch.float32
        )
        assert math.prod(shape) == 32


class TestRegionElemRuns:
    def test_contiguous_single_run(self):
        runs = region_elem_runs(0, torch.Size([4]), (1,))
        assert runs == [(0, 4)]

    def test_offset_contiguous(self):
        runs = region_elem_runs(10, torch.Size([3]), (1,))
        assert runs == [(10, 3)]

    def test_strided_produces_multiple_runs(self):
        # stride=2 along dim 0, so elements at 0, 2, 4
        runs = region_elem_runs(0, torch.Size([3]), (2,))
        assert len(runs) == 3
        assert runs == [(0, 1), (2, 1), (4, 1)]

    def test_2d_contiguous(self):
        runs = region_elem_runs(0, torch.Size([2, 3]), (3, 1))
        assert runs == [(0, 6)]


# ---------------------------------------------------------------------------
# engine/lazy.py
# ---------------------------------------------------------------------------


class TestLazyWeight:
    def test_shape_and_dtype(self):
        lw = LazyWeight("foo", torch.Size([4, 8]), torch.bfloat16)
        assert lw.shape == torch.Size([4, 8])
        assert lw.dtype == torch.bfloat16

    def test_narrow_records_op(self):
        lw = LazyWeight("foo", torch.Size([8, 4]), torch.float32)
        out = lw.narrow(0, 2, 3)
        assert isinstance(out, LazyWeight)
        assert out.shape == torch.Size([3, 4])
        assert len(out._lazy_chain) == 1
        assert out._lazy_chain[0].name == "narrow"

    def test_chain_preserves_name(self):
        lw = LazyWeight("myweight", torch.Size([4, 4]), torch.float32)
        out = lw.narrow(0, 0, 2).view(8)
        assert out._lazy_name == "myweight"

    def test_copy_to_real_tensor_is_recorded(self):
        lw = LazyWeight("w", torch.Size([4]), torch.float32)
        dst = torch.zeros(4)
        recorder = BakeRecorder()
        with recorder:
            dst.copy_(lw)
        assert len(recorder.copies) == 1
        copy = recorder.copies[0]
        assert copy.src_name == "w"
        assert copy.dst_addr == dst.data_ptr()

    def test_no_copy_outside_recorder(self):
        lw = LazyWeight("w", torch.Size([4]), torch.float32)
        dst = torch.zeros(4)
        dst.copy_(lw)  # should not raise

    def test_bake_recorder_nested(self):
        lw = LazyWeight("w", torch.Size([4]), torch.float32)
        dst = torch.zeros(4)
        outer = BakeRecorder()
        inner = BakeRecorder()
        with outer:
            with inner:
                dst.copy_(lw)
        # Inner recorder (top of stack) should have the copy
        assert len(inner.copies) == 1
        assert len(outer.copies) == 0


# ---------------------------------------------------------------------------
# protocol/serialization.py
# ---------------------------------------------------------------------------


class TestTrainerTableSerialization:
    def _make_table(self) -> TrainerTable:
        shards = [
            TrainerShard(agent_index=0, row_start=0, row_end=512, device_addr=0x1000, row_bytes=256, device_id=0),
            TrainerShard(agent_index=1, row_start=512, row_end=1024, device_addr=0x2000, row_bytes=256, device_id=1),
        ]
        tensors = [
            TrainerTensor(name="model.lm_head.weight", dtype="torch.bfloat16", shape=[1024, 128], shards=shards),
        ]
        agents = [b"agent0_nixl_bytes", b"agent1_nixl_bytes"]
        return TrainerTable(agents=agents, tensors=tensors, step=1)

    def test_roundtrip(self):
        table = self._make_table()
        encoded = encode_trainer_table(table)
        decoded = decode_trainer_table(encoded)

        assert len(decoded.agents) == 2
        assert decoded.agents[0] == b"agent0_nixl_bytes"
        assert len(decoded.tensors) == 1
        tt = decoded.tensors[0]
        assert tt.name == "model.lm_head.weight"
        assert tt.shape == [1024, 128]
        assert len(tt.shards) == 2
        assert tt.shards[0].row_start == 0
        assert tt.shards[1].row_end == 1024

    def test_shard_for_row(self):
        table = self._make_table()
        tt = table.tensors[0]
        assert tt.shard_for_row(0).agent_index == 0
        assert tt.shard_for_row(511).agent_index == 0
        assert tt.shard_for_row(512).agent_index == 1
        assert tt.shard_for_row(1023).agent_index == 1
        assert tt.shard_for_row(1024) is None


# ---------------------------------------------------------------------------
# planner/router.py
# ---------------------------------------------------------------------------


class TestRouteRegions:
    def _make_table(self) -> TrainerTable:
        # 8×4 weight split at row 4 across two trainer shards.
        # row_bytes = 4 cols * 2 bytes (bfloat16) = 8
        shards = [
            TrainerShard(agent_index=0, row_start=0, row_end=4, device_addr=0x0000, row_bytes=8, device_id=0),
            TrainerShard(agent_index=1, row_start=4, row_end=8, device_addr=0x1000, row_bytes=8, device_id=1),
        ]
        tensors = [
            TrainerTensor(name="model.weight", dtype="torch.bfloat16", shape=[8, 4], shards=shards),
        ]
        return TrainerTable(agents=[b"a0", b"a1"], tensors=tensors, step=1)

    def test_single_shard_region(self):
        # Inference needs rows 0-3 (first 4 rows, all in shard 0).
        # src_elem_runs: flat [offset, count] pairs covering rows 0-3 = elems 0..15
        region = ResolvedRegion(
            tensor_name="model.weight",
            src_elem_runs=[0, 16],   # one run: offset=0, count=16
            dst_addr=0xABCD0000,
            dst_elem_runs=[0, 16],
            element_size=2,          # bfloat16
            dst_device_id=0,
        )
        table = self._make_table()
        descriptors = route_regions([region], table)
        agents = {d.agent_index for d in descriptors}
        assert agents == {0}
        total_bytes = sum(d.nbytes for d in descriptors)
        assert total_bytes == 16 * 2  # 16 elements × 2 bytes

    def test_cross_shard_region(self):
        # Inference needs rows 2-5 (spans both shards).
        # Elements: rows 2,3 from shard 0, rows 4,5 from shard 1. Each row = 4 elems.
        # src_elem_runs: [8, 16] (offset=8 → row 2, count=16 → rows 2-5)
        region = ResolvedRegion(
            tensor_name="model.weight",
            src_elem_runs=[8, 16],
            dst_addr=0xDEAD0000,
            dst_elem_runs=[0, 16],
            element_size=2,
            dst_device_id=0,
        )
        table = self._make_table()
        descriptors = route_regions([region], table)
        agents = {d.agent_index for d in descriptors}
        assert 0 in agents
        assert 1 in agents
        total_bytes = sum(d.nbytes for d in descriptors)
        assert total_bytes == 16 * 2  # 16 elements × 2 bytes


# ---------------------------------------------------------------------------
# adapters/moe.py
# ---------------------------------------------------------------------------


class TestMoEAdapter:
    def test_passthrough_non_expert(self):
        from modelexpress.weight_transfer.adapters.moe import MoEAdapter
        adapter = MoEAdapter(num_experts=8)
        table = TrainerTable(agents=[], tensors=[], step=0)
        lw = LazyWeight("model.embed_tokens.weight", torch.Size([32000, 4096]), torch.bfloat16)
        lazy_weights = {"model.embed_tokens.weight": lw}
        result = dict(adapter.adapt_lazy_weights(lazy_weights.items(), table))
        assert "model.embed_tokens.weight" in result
        assert len(result) == 1

    def test_explodes_stacked_experts(self):
        from modelexpress.weight_transfer.adapters.moe import MoEAdapter
        adapter = MoEAdapter(num_experts=8)
        table = TrainerTable(agents=[], tensors=[], step=0)
        lw = LazyWeight("model.layers.0.mlp.experts.w1", torch.Size([8, 1024, 4096]), torch.bfloat16)
        lazy_weights = {"model.layers.0.mlp.experts.w1": lw}
        result = dict(adapter.adapt_lazy_weights(lazy_weights.items(), table))
        assert len(result) == 8
        for j in range(8):
            key = f"model.layers.0.mlp.experts.{j}.gate_proj.weight"
            assert key in result
            assert result[key].shape == torch.Size([1024, 4096])

    def test_router_gate_renaming(self):
        from modelexpress.weight_transfer.adapters.moe import MoEAdapter
        adapter = MoEAdapter(num_experts=8)
        table = TrainerTable(agents=[], tensors=[], step=0)
        lw = LazyWeight("model.layers.0.mlp.router.gate.weight", torch.Size([8, 64]), torch.bfloat16)
        lazy_weights = {"model.layers.0.mlp.router.gate.weight": lw}
        result = dict(adapter.adapt_lazy_weights(lazy_weights.items(), table))
        assert "model.layers.0.mlp.gate.weight" in result
        assert "model.layers.0.mlp.router.gate.weight" not in result
