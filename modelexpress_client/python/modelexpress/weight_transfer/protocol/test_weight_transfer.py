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


# ---------------------------------------------------------------------------
# planner/router.py — 2-D tile sharding (TP trainer support)
# ---------------------------------------------------------------------------


class TestRouteRegions2D:
    """Tests for column-parallel and 2-D tile (TP × FSDP) sharding."""

    # ── helpers ──────────────────────────────────────────────────────────────

    def _region(
        self,
        name: str,
        src_elem_runs: list[int],
        dst_addr: int = 0xDEAD_0000,
        dst_elem_runs: list[int] | None = None,
        element_size: int = 2,
    ) -> ResolvedRegion:
        n_elems = sum(src_elem_runs[i + 1] for i in range(0, len(src_elem_runs), 2))
        if dst_elem_runs is None:
            dst_elem_runs = [0, n_elems]
        return ResolvedRegion(
            tensor_name=name,
            src_elem_runs=src_elem_runs,
            dst_addr=dst_addr,
            dst_elem_runs=dst_elem_runs,
            element_size=element_size,
        )

    # ── column-parallel TP (row-parallel weight split along dim-1) ───────────

    def test_column_parallel_tp2_full_tensor(self):
        """Row-parallel TP=2: o_proj weight [8, 8] split into two [8, 4] column shards.

        The full tensor lives as:
          Shard 0 (cols 0-3): row_bytes=8 (4 cols × 2 bytes)
          Shard 1 (cols 4-7): row_bytes=8

        We request all 64 elements (the full tensor).
        Expected: 32 elements from shard 0, 32 from shard 1 with interleaving
        at every row boundary.
        """
        # 8 rows × 8 cols, TP=2 along columns
        # Shard 0 owns cols [0:4], shard 1 owns cols [4:8]
        shards = [
            TrainerShard(
                agent_index=0, row_start=0, row_end=8,
                col_start=0, col_end=4,
                device_addr=0x1000, row_bytes=8, device_id=0,
            ),
            TrainerShard(
                agent_index=1, row_start=0, row_end=8,
                col_start=4, col_end=8,
                device_addr=0x2000, row_bytes=8, device_id=1,
            ),
        ]
        table = TrainerTable(
            agents=[b"a0", b"a1"],
            tensors=[TrainerTensor(name="weight", dtype="torch.bfloat16",
                                   shape=[8, 8], shards=shards)],
            step=0,
        )
        # Request all 64 elements
        region = self._region("weight", [0, 64])
        descs = route_regions([region], table)
        total_bytes = sum(d.nbytes for d in descs)
        assert total_bytes == 64 * 2
        # Both shards must be referenced
        agents = {d.agent_index for d in descs}
        assert agents == {0, 1}
        # Contributions must be equal (32 elements each)
        bytes_per_agent = {a: 0 for a in agents}
        for d in descs:
            bytes_per_agent[d.agent_index] += d.nbytes
        assert bytes_per_agent[0] == 32 * 2
        assert bytes_per_agent[1] == 32 * 2

    def test_column_parallel_tp2_partial_row(self):
        """Request only the first row (cols 0-7) of a column-split tensor.

        First row spans both column shards: cols 0-3 in shard 0, cols 4-7 in shard 1.
        """
        shards = [
            TrainerShard(
                agent_index=0, row_start=0, row_end=4,
                col_start=0, col_end=4,
                device_addr=0x1000, row_bytes=8, device_id=0,
            ),
            TrainerShard(
                agent_index=1, row_start=0, row_end=4,
                col_start=4, col_end=8,
                device_addr=0x2000, row_bytes=8, device_id=1,
            ),
        ]
        table = TrainerTable(
            agents=[b"a0", b"a1"],
            tensors=[TrainerTensor(name="w", dtype="torch.bfloat16",
                                   shape=[4, 8], shards=shards)],
            step=0,
        )
        # First row: elements 0-7
        region = self._region("w", [0, 8])
        descs = route_regions([region], table)
        assert sum(d.nbytes for d in descs) == 8 * 2
        assert {d.agent_index for d in descs} == {0, 1}
        # Each shard contributes exactly 4 elements (one half-row)
        for d in descs:
            assert d.nbytes == 4 * 2

    def test_column_parallel_single_shard_hit(self):
        """When the requested elements land entirely in one column shard."""
        shards = [
            TrainerShard(
                agent_index=0, row_start=0, row_end=4,
                col_start=0, col_end=4,
                device_addr=0x0000, row_bytes=8, device_id=0,
            ),
            TrainerShard(
                agent_index=1, row_start=0, row_end=4,
                col_start=4, col_end=8,
                device_addr=0x1000, row_bytes=8, device_id=1,
            ),
        ]
        table = TrainerTable(
            agents=[b"a0", b"a1"],
            tensors=[TrainerTensor(name="w", dtype="torch.bfloat16",
                                   shape=[4, 8], shards=shards)],
            step=0,
        )
        # Elements 4-7: second half of row 0 → shard 1 only
        region = self._region("w", [4, 4])
        descs = route_regions([region], table)
        assert len(descs) == 1
        assert descs[0].agent_index == 1
        assert descs[0].nbytes == 4 * 2
        # Local offset in shard 1: col 4 is local col 0 → shard offset 0
        assert descs[0].src_addr == 0x1000

    # ── 2-D tile sharding (TP × FSDP) ────────────────────────────────────────

    def test_2d_tile_tp2_fsdp2(self):
        """TP=2 × FSDP=2 tiling of a [8, 8] weight into four [4, 4] tiles.

        Tile layout (trainer_rank → tile):
          rank 0: rows [0:4], cols [0:4]
          rank 1: rows [0:4], cols [4:8]
          rank 2: rows [4:8], cols [0:4]
          rank 3: rows [4:8], cols [4:8]

        All tiles have device_addr=0 (we only check byte counts and coverage).
        """
        shards = [
            TrainerShard(agent_index=0, row_start=0, row_end=4, col_start=0, col_end=4,
                         device_addr=0x0000, row_bytes=8, device_id=0),
            TrainerShard(agent_index=1, row_start=0, row_end=4, col_start=4, col_end=8,
                         device_addr=0x1000, row_bytes=8, device_id=1),
            TrainerShard(agent_index=2, row_start=4, row_end=8, col_start=0, col_end=4,
                         device_addr=0x2000, row_bytes=8, device_id=0),
            TrainerShard(agent_index=3, row_start=4, row_end=8, col_start=4, col_end=8,
                         device_addr=0x3000, row_bytes=8, device_id=1),
        ]
        table = TrainerTable(
            agents=[b"a0", b"a1", b"a2", b"a3"],
            tensors=[TrainerTensor(name="w", dtype="torch.bfloat16",
                                   shape=[8, 8], shards=shards)],
            step=0,
        )
        # Request all 64 elements
        region = self._region("w", [0, 64])
        descs = route_regions([region], table)
        total_bytes = sum(d.nbytes for d in descs)
        assert total_bytes == 64 * 2
        # All four shards must be referenced
        assert {d.agent_index for d in descs} == {0, 1, 2, 3}
        # Each tile contributes exactly 16 elements
        per_agent = {i: 0 for i in range(4)}
        for d in descs:
            per_agent[d.agent_index] += d.nbytes // 2
        for agent, count in per_agent.items():
            assert count == 16, f"Agent {agent} contributed {count} elements, expected 16"

    def test_2d_tile_local_offset_correctness(self):
        """Verify that src_addr is correct for a known element in a 2-D tile.

        Tensor [4, 4], split into 2 column shards of [4, 2].
        Request element (row=1, col=2) = flat offset 6.
        In shard 1 (cols 2-3): local row=1, local col=0 → local offset=2.
        src_addr should be shard1.device_addr + 2 * elem_size.
        """
        elem_size = 2
        shard1_addr = 0x5000
        shards = [
            TrainerShard(agent_index=0, row_start=0, row_end=4, col_start=0, col_end=2,
                         device_addr=0x4000, row_bytes=4, device_id=0),
            TrainerShard(agent_index=1, row_start=0, row_end=4, col_start=2, col_end=4,
                         device_addr=shard1_addr, row_bytes=4, device_id=1),
        ]
        table = TrainerTable(
            agents=[b"a0", b"a1"],
            tensors=[TrainerTensor(name="w", dtype="torch.bfloat16",
                                   shape=[4, 4], shards=shards)],
            step=0,
        )
        # Flat offset 6 = row 1, col 2 → lands in shard 1, local offset 2
        region = self._region("w", [6, 1], element_size=elem_size)
        descs = route_regions([region], table)
        assert len(descs) == 1
        assert descs[0].agent_index == 1
        assert descs[0].nbytes == 1 * elem_size
        assert descs[0].src_addr == shard1_addr + 2 * elem_size

    # ── backward compatibility ────────────────────────────────────────────────

    def test_row_only_shards_unchanged(self):
        """Existing row-only shards (no col_start/col_end) continue to work."""
        shards = [
            TrainerShard(agent_index=0, row_start=0, row_end=4,
                         device_addr=0x0000, row_bytes=8, device_id=0),
            TrainerShard(agent_index=1, row_start=4, row_end=8,
                         device_addr=0x1000, row_bytes=8, device_id=1),
        ]
        table = TrainerTable(
            agents=[b"a0", b"a1"],
            tensors=[TrainerTensor(name="w", dtype="torch.bfloat16",
                                   shape=[8, 4], shards=shards)],
            step=0,
        )
        # Cross-shard region: rows 2-5 (spans both shards)
        region = self._region("w", [8, 16])  # offset=8=row2, count=16=4rows
        descs = route_regions([region], table)
        assert sum(d.nbytes for d in descs) == 16 * 2
        assert {d.agent_index for d in descs} == {0, 1}

    def test_shard_for_elem_on_trainer_tensor(self):
        """TrainerTensor.shard_for_elem correctly dispatches for 2-D tiles."""
        shards = [
            TrainerShard(agent_index=0, row_start=0, row_end=2, col_start=0, col_end=4,
                         device_addr=0, row_bytes=8, device_id=0),
            TrainerShard(agent_index=1, row_start=0, row_end=2, col_start=4, col_end=8,
                         device_addr=0, row_bytes=8, device_id=1),
        ]
        from modelexpress.weight_transfer.protocol.types import TrainerTensor as TT
        tt = TT(name="w", dtype="torch.bfloat16", shape=[2, 8], shards=shards)
        assert tt.shard_for_elem(0, 0).agent_index == 0
        assert tt.shard_for_elem(0, 3).agent_index == 0
        assert tt.shard_for_elem(0, 4).agent_index == 1
        assert tt.shard_for_elem(0, 7).agent_index == 1
        assert tt.shard_for_elem(1, 0).agent_index == 0
        assert tt.shard_for_elem(1, 4).agent_index == 1
        assert tt.shard_for_elem(1, 8) is None   # out of range
