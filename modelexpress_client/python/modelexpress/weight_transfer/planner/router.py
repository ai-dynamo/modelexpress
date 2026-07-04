# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-math region router.  No torch dependency.

This module contains the same arithmetic as the Rust WeightSyncService
router (modelexpress_server/src/weight_sync/router.rs).  Both must stay in
sync: the Python version is used by LocalPlanner as a fallback; the Rust
version runs inside the MX server for ServerPlanner.

Algorithm
---------
Each ResolvedRegion describes a parameter slice as flat element runs on
both the source (trainer shard) side and the destination (vLLM parameter)
side.  The router:

  1. Iterates source element runs and maps each element offset to the
     trainer shard that owns it (dim-0 row sharding).
  2. Computes the GPU byte address inside that shard.
  3. Zips source and destination byte addresses into RdmaDescriptor pairs.
"""

from __future__ import annotations

import math

from ..protocol.types import RdmaDescriptor, ResolvedRegion, TrainerShard, TrainerTable


def _shard_for_elem(
    elem_offset: int,
    elems_per_row: int,
    shards: list[TrainerShard],
) -> tuple[TrainerShard, int]:
    """Return the shard that owns *elem_offset* and the offset within that shard.

    Args:
        elem_offset: Element offset from the start of the trainer tensor storage.
        elems_per_row: Number of elements per dim-0 row.
        shards: Sorted (by row_start) list of TrainerShard for this tensor.

    Returns:
        (shard, shard_relative_elem_offset)
    """
    row = elem_offset // elems_per_row if elems_per_row > 0 else 0
    col = elem_offset % elems_per_row if elems_per_row > 0 else 0
    for shard in shards:
        if shard.row_start <= row < shard.row_end:
            return shard, (row - shard.row_start) * elems_per_row + col
    raise ValueError(
        f"Element at offset {elem_offset} (row {row}) is not covered by any shard"
    )


def _split_run_across_shards(
    run_offset: int,
    run_count: int,
    elems_per_row: int,
    shards: list[TrainerShard],
) -> list[tuple[TrainerShard, int, int]]:
    """Split an element run at shard boundaries.

    Returns a list of (shard, shard_relative_offset, count) triples.
    """
    result: list[tuple[TrainerShard, int, int]] = []
    pos = run_offset
    remaining = run_count

    while remaining > 0:
        shard, shard_rel = _shard_for_elem(pos, elems_per_row, shards)
        row = pos // elems_per_row if elems_per_row > 0 else 0
        col = pos % elems_per_row if elems_per_row > 0 else 0
        elems_until_shard_end = (shard.row_end - row) * elems_per_row - col
        count = min(remaining, elems_until_shard_end)
        result.append((shard, shard_rel, count))
        pos += count
        remaining -= count

    return result


def _unpack_runs(flat: list[int]) -> list[tuple[int, int]]:
    """Unpack flat [offset, count, ...] into [(offset, count), ...]."""
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def _zip_src_dst(
    src_triples: list[tuple[TrainerShard, int, int]],
    dst_runs: list[tuple[int, int]],
    dst_base_addr: int,
    element_size: int,
) -> list[RdmaDescriptor]:
    """Zip source shard runs with destination runs into RdmaDescriptors.

    Both sides are already decomposed into element runs of matching total
    size.  We walk them in lockstep, splitting at whichever side's boundary
    comes first.
    """
    descriptors: list[RdmaDescriptor] = []

    src_iter = iter(src_triples)
    dst_iter = iter(dst_runs)

    cur_shard, src_rel, src_rem = next(src_iter, (None, 0, 0))
    dst_off, dst_rem = next(dst_iter, (0, 0))

    while cur_shard is not None and dst_rem > 0:
        count = min(src_rem, dst_rem)
        src_addr = cur_shard.device_addr + src_rel * element_size
        dst_addr = dst_base_addr + dst_off * element_size
        descriptors.append(RdmaDescriptor(
            agent_index=cur_shard.agent_index,
            src_addr=src_addr,
            dst_addr=dst_addr,
            nbytes=count * element_size,
        ))

        src_rel += count
        src_rem -= count
        dst_off += count
        dst_rem -= count

        if src_rem == 0:
            nxt = next(src_iter, None)
            if nxt is None:
                break
            cur_shard, src_rel, src_rem = nxt
        if dst_rem == 0:
            dst_off, dst_rem = next(dst_iter, (0, 0))

    return descriptors


def route_regions(
    regions: list[ResolvedRegion],
    table: TrainerTable,
) -> list[RdmaDescriptor]:
    """Route resolved element-run regions to NIXL RDMA descriptors.

    This is the pure-math step that maps every element offset to a trainer
    shard GPU address and pairs it with the destination vLLM parameter
    address.

    The same algorithm is implemented in Rust in
    modelexpress_server/src/weight_sync/router.rs.

    Args:
        regions: Output of resolver.resolve_copies() -- one per parameter slice.
        table: TrainerTable with shard layout.

    Returns:
        Flat list of RdmaDescriptor ready for NIXL.
    """
    descriptors: list[RdmaDescriptor] = []

    for region in regions:
        trainer_tensor = table.tensor_by_name(region.tensor_name)
        if trainer_tensor is None:
            continue

        elems_per_row = math.prod(trainer_tensor.shape[1:]) if len(trainer_tensor.shape) > 1 else 1
        shards = sorted(trainer_tensor.shards, key=lambda s: s.row_start)

        src_runs = _unpack_runs(region.src_elem_runs)
        dst_runs = _unpack_runs(region.dst_elem_runs)

        # Split all src runs across shard boundaries
        src_triples: list[tuple[TrainerShard, int, int]] = []
        for run_off, run_count in src_runs:
            src_triples.extend(_split_run_across_shards(run_off, run_count, elems_per_row, shards))

        descs = _zip_src_dst(
            src_triples,
            dst_runs,
            region.dst_addr,
            region.element_size,
        )
        descriptors.extend(descs)

    return descriptors
