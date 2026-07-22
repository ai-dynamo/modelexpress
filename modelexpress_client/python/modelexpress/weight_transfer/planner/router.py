# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure-math region router.  No torch dependency.

Maps ResolvedRegion element runs to trainer shard GPU addresses and produces
RdmaDescriptor pairs.  Mirrors the Rust router in
modelexpress_server/src/weight_sync/router.rs — both must stay in sync.
"""

from __future__ import annotations

import math

from ..protocol.types import RdmaDescriptor, ResolvedRegion, TrainerShard, TrainerTable


def _resolved_col_end(shard: TrainerShard, full_width: int) -> int:
    """Return the effective col_end for a shard, resolving -1 to full_width."""
    return full_width if shard.col_end == -1 else shard.col_end


def _shard_width(shard: TrainerShard, full_width: int) -> int:
    return _resolved_col_end(shard, full_width) - shard.col_start


def _is_row_only(shards: list[TrainerShard], full_width: int) -> bool:
    """True when all shards span the full column range (row-only sharding)."""
    return all(
        s.col_start == 0 and _resolved_col_end(s, full_width) == full_width
        for s in shards
    )


def _shard_for_elem_2d(
    row: int,
    col: int,
    shards: list[TrainerShard],
    full_width: int,
) -> tuple[TrainerShard, int]:
    """Return the shard owning (row, col) and the element offset within it.

    For 2-D tile shards the local offset is:
        (row - row_start) * shard_width + (col - col_start)
    This invariant ensures device_addr + local_off*elem_size is the correct
    byte address for any element in a tile.
    """
    for shard in shards:
        col_end = _resolved_col_end(shard, full_width)
        if shard.row_start <= row < shard.row_end and shard.col_start <= col < col_end:
            w = col_end - shard.col_start
            local_off = (row - shard.row_start) * w + (col - shard.col_start)
            return shard, local_off
    raise ValueError(
        f"Element at (row={row}, col={col}) is not covered by any shard"
    )


def _split_run_row_only(
    run_offset: int,
    run_count: int,
    elems_per_row: int,
    shards: list[TrainerShard],
) -> list[tuple[TrainerShard, int, int]]:
    """Fast path for row-only sharding.  Splits runs at row-shard boundaries."""
    result: list[tuple[TrainerShard, int, int]] = []
    pos = run_offset
    remaining = run_count

    while remaining > 0:
        row = pos // elems_per_row if elems_per_row > 0 else 0
        col = pos % elems_per_row if elems_per_row > 0 else 0
        shard: TrainerShard | None = None
        for s in shards:
            if s.row_start <= row < s.row_end:
                shard = s
                break
        if shard is None:
            raise ValueError(
                f"Element at offset {pos} (row {row}) is not covered by any shard"
            )
        shard_rel = (row - shard.row_start) * elems_per_row + col
        elems_until_shard_end = (shard.row_end - row) * elems_per_row - col
        count = min(remaining, elems_until_shard_end)
        result.append((shard, shard_rel, count))
        pos += count
        remaining -= count

    return result


def _split_run_2d(
    run_offset: int,
    run_count: int,
    full_width: int,
    shards: list[TrainerShard],
) -> list[tuple[TrainerShard, int, int]]:
    """Split an element run at row AND column shard boundaries (2-D tiling).

    We advance pos one column-segment at a time, stopping at the earlier of
    the column boundary, row boundary, or end of run.  Stopping at the
    original-tensor row end is required because crossing it jumps to
    row+1 col=0, which may land in a different column shard.
    """
    result: list[tuple[TrainerShard, int, int]] = []
    pos = run_offset
    remaining = run_count

    while remaining > 0:
        row = pos // full_width if full_width > 0 else 0
        col = pos % full_width if full_width > 0 else 0

        shard, local_off = _shard_for_elem_2d(row, col, shards, full_width)
        col_end = _resolved_col_end(shard, full_width)

        # Elements remaining in this row before hitting a column boundary
        elems_to_col_boundary = col_end - col
        # Elements remaining in the original tensor row (full_width - col)
        elems_to_row_end = full_width - col

        # We must stop at the earlier of: col_end or end-of-original-row
        # because crossing the original row end means we jump to row+1 col=0,
        # which may land in a different column shard.
        elems_this_segment = min(remaining, elems_to_col_boundary, elems_to_row_end)

        result.append((shard, local_off, elems_this_segment))
        pos += elems_this_segment
        remaining -= elems_this_segment

    return result


def _split_run_across_shards(
    run_offset: int,
    run_count: int,
    full_width: int,
    shards: list[TrainerShard],
    row_only: bool,
) -> list[tuple[TrainerShard, int, int]]:
    if row_only:
        return _split_run_row_only(run_offset, run_count, full_width, shards)
    return _split_run_2d(run_offset, run_count, full_width, shards)


def _unpack_runs(flat: list[int]) -> list[tuple[int, int]]:
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


def _zip_src_dst(
    src_triples: list[tuple[TrainerShard, int, int]],
    dst_runs: list[tuple[int, int]],
    dst_base_addr: int,
    element_size: int,
) -> list[RdmaDescriptor]:
    """Zip source shard runs with destination runs into RdmaDescriptors.

    Both sides are decomposed into element runs of matching total size.
    Walk them in lockstep, splitting at whichever side's boundary comes first.
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

    Maps every element offset to a trainer shard GPU address and pairs it
    with the destination vLLM parameter address.  The same algorithm is
    implemented in Rust in modelexpress_server/src/weight_sync/router.rs.
    """
    descriptors: list[RdmaDescriptor] = []

    for region in regions:
        trainer_tensor = table.tensor_by_name(region.tensor_name)
        if trainer_tensor is None:
            continue

        full_width = math.prod(trainer_tensor.shape[1:]) if len(trainer_tensor.shape) > 1 else 1
        # Sort shards: primary by row_start, secondary by col_start for 2-D tiles
        shards = sorted(trainer_tensor.shards, key=lambda s: (s.row_start, s.col_start))
        row_only = _is_row_only(shards, full_width)

        src_runs = _unpack_runs(region.src_elem_runs)
        dst_runs = _unpack_runs(region.dst_elem_runs)

        src_triples: list[tuple[TrainerShard, int, int]] = []
        for run_off, run_count in src_runs:
            src_triples.extend(
                _split_run_across_shards(run_off, run_count, full_width, shards, row_only)
            )

        descs = _zip_src_dst(
            src_triples,
            dst_runs,
            region.dst_addr,
            region.element_size,
        )
        descriptors.extend(descs)

    return descriptors
