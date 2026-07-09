# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""2-GPU end-to-end driver for the nccl_m2n reshard transport.

Launched as one process per rank (srun/torchrun).  Rank layout: ranks
``[0, tp_src)`` = trainer / source mesh, ``[tp_src, world)`` = generator /
destination mesh.  This exercises the full path -- fresh-comm bootstrap,
symmetric window, staging, and ``ncclReshardWithWindow`` -- through the
ModelExpress binding + ``NcclM2nExecutor``, and byte-compares the destination
tile against a gather+reslice golden.

Env: ``NCCL_SO`` / ``M2N_SO`` / ``CUDART_SO`` point ctypes at the vendored,
from-source libraries; ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` set the ranks;
``TP_SRC`` / ``TP_DST`` (default 1/1) size the two meshes.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

from modelexpress.weight_transfer.transport import _nccl_m2n_bind as binding
from modelexpress.weight_transfer.transport.nccl_m2n_executor import (
    NcclM2nExecutor,
    ReshardParam,
    torch_dtype_to_nccl,
)


def _reference_dst_tile(rows, cols, dtype, shard_dim, tp_src, tp_dst, dst_index):
    """Golden: full global tensor resliced into the dst mesh's tile for dst_index."""
    g = torch.arange(rows * cols, dtype=dtype).reshape(rows, cols)
    return list(torch.chunk(g, tp_dst, dim=shard_dim))[dst_index]


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", rank))
    tp_src = int(os.environ.get("TP_SRC", "1"))
    tp_dst = int(os.environ.get("TP_DST", "1"))
    assert tp_src + tp_dst == world, f"tp_src+tp_dst ({tp_src}+{tp_dst}) != world {world}"

    torch.cuda.set_device(local)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    is_src = rank < tp_src

    m2n = binding.M2N(
        nccl_path=os.environ.get("NCCL_SO"),
        m2n_path=os.environ.get("M2N_SO"),
        cudart_path=os.environ.get("CUDART_SO"),
    )
    comm = binding.bootstrap_comm_from_torch(m2n, tp_src, tp_dst, device_id=local)
    if rank == 0:
        print(f"[rank {rank}] comm bootstrapped, world={world} tp_src={tp_src} tp_dst={tp_dst}", flush=True)

    # Global tensor sharded along dim 0.  rows must divide by both tp sizes.
    rows, cols = 8, 16
    dtype = torch.float32
    shard_dim = 0
    src_rows = rows // tp_src
    dst_rows = rows // tp_dst

    if is_src:
        # This src rank owns rows [rank*src_rows, (rank+1)*src_rows).
        g = torch.arange(rows * cols, dtype=dtype).reshape(rows, cols)
        tile = g[rank * src_rows : (rank + 1) * src_rows].contiguous().cuda()
    else:
        tile = torch.zeros(dst_rows, cols, dtype=dtype, device="cuda")

    p = ReshardParam(
        name="w",
        global_shape=(rows, cols),
        ndims=2,
        shard_dim=shard_dim,
        dtype_nccl=torch_dtype_to_nccl(dtype),
        local_ptr=tile.data_ptr(),
        local_nbytes=tile.numel() * tile.element_size(),
    )
    # window must fit the larger of the src/dst tile (world-consistent).
    window_bytes = max(src_rows, dst_rows) * cols * tile.element_size()

    ex = NcclM2nExecutor(m2n, comm, rank, tp_src, tp_dst, device_id=local, max_cta=8)
    ex.execute([p], window_bytes)
    torch.cuda.synchronize()

    rc = 0
    if not is_src:
        dst_index = rank - tp_src
        expected = _reference_dst_tile(rows, cols, dtype, shard_dim, tp_src, tp_dst, dst_index)
        got = tile.cpu()
        ok = torch.equal(got, expected)
        print(
            f"[rank {rank}] RESHARD {'PASS' if ok else 'FAIL'} "
            f"dst_index={dst_index} got[0,:4]={got.flatten()[:4].tolist()} "
            f"exp[0,:4]={expected.flatten()[:4].tolist()}",
            flush=True,
        )
        rc = 0 if ok else 1

    ex.teardown()
    dist.barrier()
    dist.destroy_process_group()
    return rc


if __name__ == "__main__":
    sys.exit(main())
