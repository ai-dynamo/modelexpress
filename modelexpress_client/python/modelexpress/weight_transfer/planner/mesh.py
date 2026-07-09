# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mesh mapping for the nccl_m2n reshard transport.

Pure-math (no torch) helpers that translate PR #481's ``TrainerTensor`` shard
layout into the 2-D ``ncclMesh_t`` descriptors ncclReshardWithWindow consumes.
Mirrors the PoC ``reshard_ref.Mesh`` math and extends it to read the richer
``TrainerShard`` (row/col tile) layout.

First slice: TP-only, same-dim reshards (one axis SHARD, the other REPLICATE).
Cross-dim (transpose) and 2-D TP x FSDP tiles are later slices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..protocol.types import TrainerTensor

REPLICATE = -1  # NCCL_RESHARD_REPLICATE


def SHARD(dim: int) -> int:  # NCCL_RESHARD_SHARD(d)
    return dim


@dataclass
class Mesh:
    """Mirrors ncclMesh_t. 2-D mesh; product(dims) == number of ranks."""

    dims: tuple[int, int]
    start_rank: int
    placement: tuple[int, int]  # each entry REPLICATE or a sharded tensor-dim

    @property
    def nranks(self) -> int:
        return self.dims[0] * self.dims[1]

    @property
    def shard_dim(self) -> int:
        sh = [p for p in self.placement if p != REPLICATE]
        if len(sh) != 1:
            raise ValueError(f"mesh must shard exactly one dim, got placement={self.placement}")
        return sh[0]

    @property
    def shard_count(self) -> int:
        for i, p in enumerate(self.placement):
            if p != REPLICATE:
                return self.dims[i]
        raise ValueError("no shard axis")


def local_shape(global_shape: tuple[int, ...], mesh: Mesh) -> tuple[int, ...]:
    """Per-rank tile shape for ``mesh`` given the global tensor shape."""
    d = mesh.shard_dim
    n = mesh.shard_count
    if global_shape[d] % n != 0:
        raise ValueError(f"global dim {d}={global_shape[d]} not divisible by shard_count {n}")
    ls = list(global_shape)
    ls[d] = global_shape[d] // n
    return tuple(ls)


def tile_shape(global_shape: tuple[int, ...], mesh: Mesh) -> tuple[int, ...]:
    """Per-rank tile shape, returning the full shape for a fully-replicated mesh."""
    if all(p == REPLICATE for p in mesh.placement):
        return tuple(global_shape)
    return local_shape(global_shape, mesh)


def build_tp_meshes(shard_dim: int, tp_src: int, tp_dst: int) -> tuple[Mesh, Mesh]:
    """Src (trainer TP) and dst (generator TP) 1xN meshes for one param.

    Src group = ranks ``[0, tp_src)``; dst group = disjoint
    ``[tp_src, tp_src + tp_dst)``.  A replicated param (``shard_dim == REPLICATE``)
    replicates on both axes; a sharded param shards the same tensor dim on both
    sides with possibly different TP sizes (the M!=N reshard).
    """
    if shard_dim == REPLICATE:
        src = Mesh(dims=(1, tp_src), start_rank=0, placement=(REPLICATE, REPLICATE))
        dst = Mesh(dims=(1, tp_dst), start_rank=tp_src, placement=(REPLICATE, REPLICATE))
        return src, dst
    src = Mesh(dims=(1, tp_src), start_rank=0, placement=(REPLICATE, SHARD(shard_dim)))
    dst = Mesh(dims=(1, tp_dst), start_rank=tp_src, placement=(REPLICATE, SHARD(shard_dim)))
    return src, dst


def shard_dim_from_trainer_tensor(tensor: TrainerTensor) -> int:
    """Infer the single sharded tensor dim from a ``TrainerTensor``'s shards.

    Returns ``REPLICATE`` when a single shard covers the full tensor, ``0`` when
    the shards partition rows (row-parallel-loaded / column-parallel weights),
    or ``1`` when they partition columns (row-parallel weights, 2-D-col tiles).
    Raises when the layout shards both dims at once (a later 2-D slice).
    """
    shards = tensor.shards
    if not shards or len(shards) == 1:
        return REPLICATE

    rows = tensor.shape[0] if tensor.shape else 0
    cols = tensor.shape[1] if len(tensor.shape) > 1 else 1

    rows_partitioned = any(s.row_start != 0 or s.row_end != rows for s in shards)
    cols_partitioned = any(
        s.col_start != 0 or tensor._resolved_col_end(s) != cols for s in shards
    )

    if rows_partitioned and cols_partitioned:
        raise ValueError(
            f"tensor {tensor.name!r} shards both dims (2-D tile); not supported in the first slice"
        )
    if cols_partitioned:
        return 1
    if rows_partitioned:
        return 0
    return REPLICATE
