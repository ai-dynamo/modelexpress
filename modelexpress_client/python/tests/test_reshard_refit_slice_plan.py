# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Slice intersection - pure index/stride arithmetic, no torch.

Feeds captured ``RecordedCopy`` records + published ``Shard``s through
``plan_pull`` and checks the emitted byte segments against a hand-computed
reference. Covers: contiguous column-block, strided column-slice (multi-run),
fused per-shard dest offset, multi-shard fan-in, disjoint shard skip, and
UnsupportedReshard detection (non-box op, dtype mismatch).

Run: pytest tests/test_reshard_refit_slice_plan.py
"""

import pytest

from modelexpress.refit.reshard.slice_plan import (
    Shard,
    intersect,
    op_chain_to_box,
    plan_pull,
)
from modelexpress.refit.reshard.types import RecordedCopy, UnsupportedReshard

F32 = "f32"
EL = 4  # bytes per f32 element


def _copy(op_chain, param_name, dest_offset, dest_shape, dest_stride):
    return RecordedCopy(
        src_name="src",
        op_chain=op_chain,
        param_name=param_name,
        dest_offset=dest_offset,
        dest_shape=dest_shape,
        dest_stride=dest_stride,
        dest_dtype=F32,
    )


def test_column_block_single_contiguous_run():
    # ColumnParallel rank 0: full [8,4], need rows [0:4] -> contiguous block.
    copy = _copy((("narrow", (0, 0, 4), ()),), "col", 0, (4, 4), (4, 1))
    shard = Shard(shard_offset=(0, 0), shape=(8, 4), session="s0", addr=1000, elsize=EL)
    segs = plan_pull(
        copy, global_shape=(8, 4), src_dtype=F32, elsize=EL, shards=[shard]
    )
    assert len(segs) == 1
    assert (
        segs[0].src_addr == 1000 and segs[0].dst_byte == 0 and segs[0].nbytes == 16 * EL
    )


def test_column_slice_is_strided_multi_run():
    # RowParallel rank 0: full [4,8], need cols [0:4] -> strided, one run per row.
    copy = _copy((("narrow", (1, 0, 4), ()),), "row", 0, (4, 4), (4, 1))
    shard = Shard(shard_offset=(0, 0), shape=(4, 8), session="s0", addr=2000, elsize=EL)
    segs = plan_pull(
        copy, global_shape=(4, 8), src_dtype=F32, elsize=EL, shards=[shard]
    )
    assert len(segs) == 4  # 4 rows, each a 4-element run
    # Row r: src starts at r*8 elements, dst at r*4 elements, 4 elems each.
    assert [(s.src_addr - 2000) // EL for s in segs] == [0, 8, 16, 24]
    assert [s.dst_byte // EL for s in segs] == [0, 4, 8, 12]
    assert all(s.nbytes == 4 * EL for s in segs)


def test_fused_shard_dest_offset():
    # Fused qkv 'k' shard: full-copy source [2,4] into dest param 'qkv' at row 4
    # -> dest_offset 16 (4 rows * 4 cols).
    copy = _copy((), "qkv", 16, (2, 4), (4, 1))
    shard = Shard(shard_offset=(0, 0), shape=(2, 4), session="s0", addr=3000, elsize=EL)
    segs = plan_pull(
        copy, global_shape=(2, 4), src_dtype=F32, elsize=EL, shards=[shard]
    )
    assert len(segs) == 1
    assert segs[0].src_addr == 3000
    assert segs[0].dst_byte == 16 * EL and segs[0].nbytes == 8 * EL
    assert segs[0].param_name == "qkv"


def test_multi_shard_fan_in():
    # Need full [4,8]; published as two column-halves [.,0:4] and [.,4:8].
    copy = _copy((), "w", 0, (4, 8), (8, 1))
    left = Shard(shard_offset=(0, 0), shape=(4, 4), session="a", addr=100, elsize=EL)
    right = Shard(shard_offset=(0, 4), shape=(4, 4), session="b", addr=200, elsize=EL)
    segs = plan_pull(
        copy, global_shape=(4, 8), src_dtype=F32, elsize=EL, shards=[left, right]
    )
    # Each half is strided in the full row -> 4 runs per shard, 8 total.
    assert len(segs) == 8
    assert {s.session for s in segs} == {"a", "b"}
    left_dst = sorted(s.dst_byte // EL for s in segs if s.session == "a")
    assert left_dst == [0, 8, 16, 24]  # left half lands at col 0 of each 8-wide row
    right_dst = sorted(s.dst_byte // EL for s in segs if s.session == "b")
    assert right_dst == [4, 12, 20, 28]  # right half lands at col 4


def test_disjoint_shard_skipped():
    copy = _copy((("narrow", (0, 0, 4), ()),), "col", 0, (4, 4), (4, 1))
    far = Shard(shard_offset=(4, 0), shape=(4, 4), session="s0", addr=9000, elsize=EL)
    segs = plan_pull(copy, global_shape=(8, 4), src_dtype=F32, elsize=EL, shards=[far])
    assert segs == []


def test_rank_mismatched_geometry_raises():
    with pytest.raises(ValueError):
        intersect([(0, 4), (0, 4)], [(0, 4)])

    copy = _copy((), "w", 0, (4, 4), (4, 1))
    malformed = Shard(
        shard_offset=(0,),
        shape=(4, 4),
        session="s0",
        addr=0,
        elsize=EL,
    )
    with pytest.raises(ValueError):
        plan_pull(
            copy,
            global_shape=(4, 4),
            src_dtype=F32,
            elsize=EL,
            shards=[malformed],
        )


def test_getitem_slice_supported():
    # Row-range slice: x[0:4] on dim 0.
    box = op_chain_to_box((("__getitem__", (slice(0, 4),), ()),), (8, 4))
    assert box == [(0, 4), (0, 4)]


def _reconstruct(src_2d, copy, shards, segs):
    """Apply the planned segments (src buffer -> dest buffer) and return the dest
    reshaped, so a test can assert it equals the loader's ground-truth view."""
    import torch

    src_buf = src_2d.reshape(-1).contiguous()
    dest_numel = 1
    for s in copy.dest_shape:
        dest_numel *= s
    dst_buf = torch.zeros(copy.dest_offset + dest_numel, dtype=src_2d.dtype)
    addr_of = {sh.session: sh.addr for sh in shards}
    for seg in segs:
        s_el = (seg.src_addr - addr_of[seg.session]) // EL
        d_el = seg.dst_byte // EL
        n = seg.nbytes // EL
        dst_buf[d_el : d_el + n] = src_buf[s_el : s_el + n]
    return dst_buf[copy.dest_offset : copy.dest_offset + dest_numel].reshape(
        copy.dest_shape
    )


def test_transpose_reconstructs_bit_for_bit():
    # Loader does source.transpose(0,1); 4a replays it to a permuted box and slices.
    import torch

    src = torch.arange(3 * 4, dtype=torch.float32).reshape(3, 4)
    ground_truth = src.transpose(0, 1)  # [4,3]
    copy = _copy((("transpose", (0, 1), ()),), "wT", 0, (4, 3), (3, 1))
    shard = Shard(shard_offset=(0, 0), shape=(3, 4), session="s0", addr=0, elsize=EL)
    segs = plan_pull(
        copy, global_shape=(3, 4), src_dtype=F32, elsize=EL, shards=[shard]
    )
    assert torch.equal(_reconstruct(src, copy, [shard], segs), ground_truth)


def test_permute_3d_reconstructs_bit_for_bit():
    import torch

    src = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    ground_truth = src.permute(2, 0, 1)  # [4,2,3]
    copy = _copy((("permute", (2, 0, 1), ()),), "wP", 0, (4, 2, 3), (6, 3, 1))
    shard = Shard(
        shard_offset=(0, 0, 0), shape=(2, 3, 4), session="s0", addr=0, elsize=EL
    )
    segs = plan_pull(
        copy, global_shape=(2, 3, 4), src_dtype=F32, elsize=EL, shards=[shard]
    )
    assert torch.equal(_reconstruct(src, copy, [shard], segs), ground_truth)


def test_transpose_of_narrowed_block_reconstructs():
    # narrow then transpose: source[1:3, :].transpose(0,1) -> exercises box_lo unravel.
    import torch

    src = torch.arange(4 * 5, dtype=torch.float32).reshape(4, 5)
    ground_truth = src.narrow(0, 1, 2).transpose(0, 1)  # [5,2]
    copy = _copy(
        (("narrow", (0, 1, 2), ()), ("transpose", (0, 1), ())), "wNT", 0, (5, 2), (2, 1)
    )
    shard = Shard(shard_offset=(0, 0), shape=(4, 5), session="s0", addr=0, elsize=EL)
    segs = plan_pull(
        copy, global_shape=(4, 5), src_dtype=F32, elsize=EL, shards=[shard]
    )
    assert torch.equal(_reconstruct(src, copy, [shard], segs), ground_truth)


def test_dim_merge_reshape_still_unsupported():
    # A rank-changing (dim-merge) reshape is not a permuted box -> full-pull.
    copy = _copy((("reshape", ((12,),), ()),), "w", 0, (12,), (1,))
    shard = Shard(shard_offset=(0, 0), shape=(3, 4), session="s0", addr=0, elsize=EL)
    with pytest.raises(UnsupportedReshard):
        plan_pull(copy, global_shape=(3, 4), src_dtype=F32, elsize=EL, shards=[shard])


def test_transpose_then_contiguous_materializes_falls_back():
    # transpose().contiguous() on a SQUARE tensor: the materialized copy has
    # source-matching strides (n,1)@0, so the box/permutation math alone would
    # wrongly treat it as the identity box and emit UNtransposed bytes. Only the
    # materialization (base) check saves us -> must fall back.
    copy = _copy(
        (("transpose", (0, 1), ()), ("contiguous", (), ())), "w", 0, (4, 4), (4, 1)
    )
    shard = Shard(shard_offset=(0, 0), shape=(4, 4), session="s0", addr=0, elsize=EL)
    with pytest.raises(UnsupportedReshard):
        plan_pull(copy, global_shape=(4, 4), src_dtype=F32, elsize=EL, shards=[shard])


def test_dtype_mismatch_raises():
    copy = _copy((), "w", 0, (4, 4), (4, 1))
    shard = Shard(shard_offset=(0, 0), shape=(4, 4), session="s0", addr=0, elsize=EL)
    with pytest.raises(UnsupportedReshard):
        plan_pull(
            copy, global_shape=(4, 4), src_dtype="bf16", elsize=EL, shards=[shard]
        )


def test_int_index_collapse_unsupported():
    with pytest.raises(UnsupportedReshard):
        op_chain_to_box((("__getitem__", (0,), ()),), (8, 4))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
