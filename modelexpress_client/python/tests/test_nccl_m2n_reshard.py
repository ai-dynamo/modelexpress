# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-GPU tests for the nccl_m2n reshard transport's layout math.

These validate the pure-math half (mesh inference + tile shapes) against a
gather+reslice golden reference, which ``nccl.m2n.reshard`` must match
byte-for-byte on GPU. No CUDA or NCCL required.
"""

from __future__ import annotations

import torch

from modelexpress.weight_transfer.planner.mesh import (
    REPLICATE,
    build_tp_meshes,
    shard_dim_from_trainer_tensor,
    tile_shape,
)
from modelexpress.weight_transfer.protocol.types import (
    TrainerShard,
    TrainerTensor,
)


def _row_sharded_tensor(name: str, rows: int, cols: int, n: int) -> TrainerTensor:
    step = rows // n
    shards = [
        TrainerShard(
            agent_index=i,
            row_start=i * step,
            row_end=(i + 1) * step,
            device_addr=0,
            row_bytes=cols * 2,
            device_id=i,
        )
        for i in range(n)
    ]
    return TrainerTensor(name=name, dtype="torch.bfloat16", shape=[rows, cols], shards=shards)


def _col_sharded_tensor(name: str, rows: int, cols: int, n: int) -> TrainerTensor:
    step = cols // n
    shards = [
        TrainerShard(
            agent_index=i,
            row_start=0,
            row_end=rows,
            device_addr=0,
            row_bytes=step * 2,
            device_id=i,
            col_start=i * step,
            col_end=(i + 1) * step,
        )
        for i in range(n)
    ]
    return TrainerTensor(name=name, dtype="torch.bfloat16", shape=[rows, cols], shards=shards)


def _replicated_tensor(name: str, rows: int, cols: int) -> TrainerTensor:
    shard = TrainerShard(
        agent_index=0, row_start=0, row_end=rows, device_addr=0, row_bytes=cols * 2, device_id=0
    )
    return TrainerTensor(name=name, dtype="torch.bfloat16", shape=[rows, cols], shards=[shard])


# ---- shard-dim inference ---------------------------------------------------
def test_shard_dim_row():
    assert shard_dim_from_trainer_tensor(_row_sharded_tensor("w", 8, 4, 2)) == 0


def test_shard_dim_col():
    assert shard_dim_from_trainer_tensor(_col_sharded_tensor("w", 4, 8, 2)) == 1


def test_shard_dim_replicate():
    assert shard_dim_from_trainer_tensor(_replicated_tensor("norm", 4, 4)) == REPLICATE


def test_shard_dim_both_dims_raises():
    shards = [
        TrainerShard(
            agent_index=0,
            row_start=0,
            row_end=4,
            device_addr=0,
            row_bytes=4,
            device_id=0,
            col_start=0,
            col_end=2,
        ),
        TrainerShard(
            agent_index=1,
            row_start=4,
            row_end=8,
            device_addr=0,
            row_bytes=4,
            device_id=1,
            col_start=2,
            col_end=4,
        ),
    ]
    tensor = TrainerTensor(
        name="w2d", dtype="torch.bfloat16", shape=[8, 4], shards=shards
    )
    try:
        shard_dim_from_trainer_tensor(tensor)
    except ValueError as exc:
        assert "shards both dims" in str(exc)
    else:
        raise AssertionError("expected ValueError for 2-D tile shards")


def test_shard_dim_partial_single_shard_raises():
    shard = TrainerShard(
        agent_index=0,
        row_start=0,
        row_end=2,
        device_addr=0,
        row_bytes=8,
        device_id=0,
    )
    tensor = TrainerTensor(
        name="partial", dtype="torch.bfloat16", shape=[4, 4], shards=[shard]
    )
    try:
        shard_dim_from_trainer_tensor(tensor)
    except ValueError as exc:
        assert "does not cover the full tensor" in str(exc)
    else:
        raise AssertionError("expected ValueError for partial single shard")


# ---- tile shapes -----------------------------------------------------------
def test_tile_shape_shard_dim0():
    src, dst = build_tp_meshes(shard_dim=0, tp_src=4, tp_dst=2)
    assert tile_shape((8, 16), src) == (2, 16)  # 8 rows / 4
    assert tile_shape((8, 16), dst) == (4, 16)  # 8 rows / 2


def test_tile_shape_replicate_returns_full():
    src, dst = build_tp_meshes(shard_dim=REPLICATE, tp_src=4, tp_dst=2)
    assert tile_shape((8, 16), src) == (8, 16)
    assert tile_shape((8, 16), dst) == (8, 16)


# ---- golden reshard parity (gather+reslice) --------------------------------
def _reference_reshard(global_tensor, shard_dim, tp_src, tp_dst):
    """Golden: independent src tiles -> gather to full -> reslice to dst tiles."""
    src_tiles = [
        tile.clone() for tile in torch.chunk(global_tensor, tp_src, dim=shard_dim)
    ]
    full = torch.cat(src_tiles, dim=shard_dim)
    return list(torch.chunk(full, tp_dst, dim=shard_dim))


def test_golden_reshard_dim0_4to2():
    torch.manual_seed(0)
    g = torch.randn(8, 16)
    dst_shards = _reference_reshard(g, shard_dim=0, tp_src=4, tp_dst=2)
    expected = list(torch.chunk(g, 2, dim=0))
    # dst tile shape from mesh math must match the golden shard shape.
    _, dst_mesh = build_tp_meshes(shard_dim=0, tp_src=4, tp_dst=2)
    assert tuple(dst_shards[0].shape) == tile_shape((8, 16), dst_mesh)
    for got, exp in zip(dst_shards, expected, strict=True):
        assert torch.equal(got, exp)


def test_golden_reshard_dim1_2to4():
    torch.manual_seed(1)
    g = torch.randn(16, 8)
    dst_shards = _reference_reshard(g, shard_dim=1, tp_src=2, tp_dst=4)
    expected = list(torch.chunk(g, 4, dim=1))
    _, dst_mesh = build_tp_meshes(shard_dim=1, tp_src=2, tp_dst=4)
    assert tuple(dst_shards[0].shape) == tile_shape((16, 8), dst_mesh)
    for got, exp in zip(dst_shards, expected, strict=True):
        assert torch.equal(got, exp)
