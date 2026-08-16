# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Current-M2N end-to-end reshard driver.

Launch one process per rank. Ranks ``[0, tp_src)`` are trainers; remaining
ranks are generators. A Gloo process group broadcasts only NCCL bootstrap
bytes. The transfer itself uses a runtime-owned NCCL4Py communicator, one
explicit CUDA stream, and public ``nccl.m2n.reshard``.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

from modelexpress.refit.reshard.transport.nccl_m2n.executor import (
    NcclM2nExecutor,
    ReshardParam,
)
from modelexpress.refit.reshard.transport.nccl_m2n.runtime import (
    _M2nLaneSpec,
    _M2nRuntime,
)


def _reference_dst_tile(rows, cols, dtype, shard_dim, tp_dst, dst_index):
    global_tensor = torch.arange(rows * cols, dtype=dtype).reshape(rows, cols)
    return list(torch.chunk(global_tensor, tp_dst, dim=shard_dim))[dst_index]


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    tp_src = int(os.environ.get("TP_SRC", "1"))
    tp_dst = int(os.environ.get("TP_DST", "1"))
    if tp_src + tp_dst != world:
        raise SystemExit(
            f"tp_src+tp_dst ({tp_src}+{tp_dst}) != world {world}"
        )

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    is_src = rank < tp_src

    runtime = _M2nRuntime(local_rank, max_cta=8)
    bootstrap: list[bytes | None] = [
        runtime.new_unique_id_bytes() if rank == 0 else None
    ]
    dist.broadcast_object_list(bootstrap, src=0)
    if bootstrap[0] is None:
        raise RuntimeError("failed to broadcast NCCL unique id")
    lane = runtime.create_lane(
        _M2nLaneSpec(
            lane_id="weights-0",
            key=(0, 0),
            unique_id=bootstrap[0],
            nranks=world,
            comm_rank=rank,
            device_id=local_rank,
        ),
    )

    rows, cols = 8, 16
    dtype = torch.float32
    shard_dim = 0
    if rows % tp_src != 0 or rows % tp_dst != 0:
        raise SystemExit(
            f"rows {rows} must be divisible by tp_src={tp_src} and tp_dst={tp_dst}"
        )
    src_rows = rows // tp_src
    dst_rows = rows // tp_dst

    tiles = []
    offsets = (0, 10_000)
    for offset in offsets:
        if is_src:
            global_tensor = torch.arange(rows * cols, dtype=dtype).reshape(rows, cols)
            tile = global_tensor[
                rank * src_rows : (rank + 1) * src_rows
            ].contiguous().cuda() + offset
        else:
            tile = torch.zeros(dst_rows, cols, dtype=dtype, device="cuda")
        tiles.append(tile)

    executor = NcclM2nExecutor(
        runtime,
        lane,
        tp_src=tp_src,
        tp_dst=tp_dst,
    )
    executor.execute(
        [
            ReshardParam(
                name=f"w{index}",
                global_shape=(rows, cols),
                shard_dim=shard_dim,
                local_tensor=tile,
            )
            for index, tile in enumerate(tiles)
        ]
    )

    rc = 0
    if not is_src:
        dst_index = rank - tp_src
        results = []
        for index, (tile, offset) in enumerate(zip(tiles, offsets, strict=True)):
            expected = _reference_dst_tile(
                rows,
                cols,
                dtype,
                shard_dim,
                tp_dst,
                dst_index,
            ) + offset
            got = tile.cpu()
            ok = torch.equal(got, expected)
            results.append(ok)
            print(
                f"[rank {rank}] RESHARD {'PASS' if ok else 'FAIL'} "
                f"param={index} dst_index={dst_index} "
                f"got[0,:4]={got.flatten()[:4].tolist()} "
                f"exp[0,:4]={expected.flatten()[:4].tolist()}",
                flush=True,
            )
        rc = 0 if all(results) else 1

    executor.teardown()
    dist.barrier()
    runtime.close()
    dist.barrier()
    dist.destroy_process_group()
    return rc


if __name__ == "__main__":
    sys.exit(main())
