# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute a trainer->generator weight reshard via nccl_m2n.

``NcclM2nExecutor`` is the collective counterpart to ``NixlExecutor``.  Instead
of a list of one-sided ``RdmaDescriptor``s it consumes per-parameter mesh
descriptors and drives ``ncclReshardWithWindow`` -- a single GPU collective that
both the source (trainer / PushRole) and destination (generator / PullRole) ranks
enter together over one shared ``ncclComm_t``.

Because the library routes internally from the src/dst meshes, the router /
resolver / RdmaDescriptor machinery is not on this path.

Window contract: both the src and dst tiles must live at the window base
(zero-offset).  Live parameters generally do not, so each param is staged through
a symmetric ``ncclMemAlloc`` buffer:

    src rank:  param tile -> window base -> reshard
    dst rank:  reshard -> window base -> param tile
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from typing import TYPE_CHECKING

from ..planner.mesh import build_tp_meshes, shard_dim_from_trainer_tensor, tile_shape
from . import _nccl_m2n_bind as binding

if TYPE_CHECKING:
    from ..protocol.types import TrainerTable

logger = logging.getLogger("modelexpress.weight_transfer.nccl_m2n_executor")


@dataclass
class ReshardParam:
    """One parameter to reshard, from this rank's point of view.

    ``local_ptr`` is the owned tile on this rank: the src tile on trainer ranks,
    the dst tile on generator ranks.  ``shard_dim`` is the single sharded tensor
    dim (or ``mesh.REPLICATE``); ``dtype_nccl`` is the ncclDataType_t enum.
    """

    name: str
    global_shape: tuple[int, ...]
    ndims: int
    shard_dim: int
    dtype_nccl: int
    local_ptr: int
    local_nbytes: int


class NcclM2nExecutor:
    """Drive per-param trainer<->generator reshards over a shared comm + window."""

    def __init__(
        self,
        m2n: binding.M2N,
        comm: int,
        rank: int,
        tp_src: int,
        tp_dst: int,
        device_id: int,
        stream: int = 0,
        max_cta: int | None = None,
    ) -> None:
        self._m2n = m2n
        self._comm = comm
        self._rank = rank
        self._tp_src = tp_src
        self._tp_dst = tp_dst
        self._device_id = device_id
        self._stream = stream
        self._is_src = rank < tp_src

        self._m2n.init(max_cta)

        self._window_buf: int | None = None
        self._window: int | None = None
        self._window_bytes: int = 0

    def _ensure_window(self, nbytes: int) -> None:
        """(Re)allocate the symmetric window if it is too small.

        The window is collectively registered, so every rank must size it to the
        same value -- callers pass a world-consistent worst case.
        """
        if self._window is not None and self._window_bytes >= nbytes:
            return
        if self._window is not None:
            self._m2n.window_deregister(self._comm, self._window)
            self._m2n.mem_free(self._window_buf)
        self._window_buf = self._m2n.mem_alloc(nbytes)
        self._window = self._m2n.window_register(self._comm, self._window_buf, nbytes)
        self._window_bytes = nbytes

    def execute(self, params: list[ReshardParam], window_bytes: int) -> tuple[int, float]:
        """Reshard every param.  ``window_bytes`` is the world-consistent window size.

        Returns ``(total_bytes_moved, elapsed_seconds)``.
        """
        if not params:
            return 0, 0.0

        self._ensure_window(window_bytes)
        assert self._window is not None and self._window_buf is not None

        start = time.perf_counter()
        total_bytes = 0

        for p in params:
            src_mesh_dc, dst_mesh_dc = build_tp_meshes(p.shard_dim, self._tp_src, self._tp_dst)
            src_mesh = binding.make_mesh(
                src_mesh_dc.dims, src_mesh_dc.start_rank, src_mesh_dc.placement
            )
            dst_mesh = binding.make_mesh(
                dst_mesh_dc.dims, dst_mesh_dc.start_rank, dst_mesh_dc.placement
            )

            src_local = tile_shape(p.global_shape, src_mesh_dc)
            dst_local = tile_shape(p.global_shape, dst_mesh_dc)

            # Stage the owned tile into the window base before the collective.
            if self._is_src:
                self._m2n.memcpy_dtod(self._window_buf, p.local_ptr, p.local_nbytes)

            src_ptr = self._window_buf if self._is_src else 0
            dst_ptr = self._window_buf if not self._is_src else 0
            src_t = binding.make_tensor_desc(
                src_ptr, src_local, p.ndims, p.dtype_nccl, src_mesh
            )
            dst_t = binding.make_tensor_desc(
                dst_ptr, dst_local, p.ndims, p.dtype_nccl, dst_mesh
            )

            rc = self._m2n.reshard(self._comm, self._window, src_t, dst_t, self._stream)
            if rc != binding.ncclSuccess:
                raise RuntimeError(f"ncclReshardWithWindow({p.name!r}) rc={rc}")

            # Copy the resharded tile out of the window into the live param.
            if not self._is_src:
                self._m2n.device_synchronize()
                self._m2n.memcpy_dtod(p.local_ptr, self._window_buf, p.local_nbytes)

            total_bytes += p.local_nbytes

        self._m2n.device_synchronize()
        elapsed = time.perf_counter() - start
        gbps = (total_bytes * 8) / (elapsed * 1e9) if elapsed > 0 else 0.0
        logger.info(
            "reshard complete: %d params, %.2f GB in %.3fs (%.1f Gbps)",
            len(params),
            total_bytes / 1e9,
            elapsed,
            gbps,
        )
        return total_bytes, elapsed

    def teardown(self) -> None:
        if self._window is not None:
            self._m2n.window_deregister(self._comm, self._window)
            self._m2n.mem_free(self._window_buf)
            self._window = None
            self._window_buf = None
            self._window_bytes = 0
        self._m2n.finalize()


def torch_dtype_to_nccl(dtype) -> int:
    """Map a torch dtype to its ncclDataType_t enum (src and dst must match)."""
    import torch

    table = {
        torch.float32: binding.ncclFloat32,
        torch.float16: binding.ncclFloat16,
        torch.bfloat16: binding.ncclBfloat16,
        torch.float64: binding.ncclFloat64,
        torch.int8: binding.ncclInt8,
        torch.uint8: binding.ncclUint8,
        torch.int32: binding.ncclInt32,
        torch.int64: binding.ncclInt64,
    }
    if dtype not in table:
        raise ValueError(f"unsupported dtype for reshard: {dtype}")
    return table[dtype]


def build_reshard_params(
    model,
    table: TrainerTable,
    tp_src: int,
    tp_dst: int,
) -> tuple[list[ReshardParam], int]:
    """Build this rank's ``ReshardParam`` list plus the world-consistent window size.

    ``local_ptr`` is taken from the local model's live parameter (the src tile on
    trainer ranks, the dst tile on generator ranks).  The sharded dim, global
    shape, and dtype come from the shared ``TrainerTable`` so both sides agree.
    """
    from math import prod

    named = dict(model.named_parameters())
    params: list[ReshardParam] = []
    window_bytes = 0

    for tt in table.tensors:
        param = named.get(tt.name)
        if param is None:
            logger.debug("param %s not in local model, skipping", tt.name)
            continue

        shard_dim = shard_dim_from_trainer_tensor(tt)
        global_shape = tuple(tt.shape)
        ndims = len(global_shape)
        elem = param.element_size()

        params.append(
            ReshardParam(
                name=tt.name,
                global_shape=global_shape,
                ndims=ndims,
                shard_dim=shard_dim,
                dtype_nccl=torch_dtype_to_nccl(param.dtype),
                local_ptr=param.data_ptr(),
                local_nbytes=param.numel() * elem,
            )
        )

        src_mesh, dst_mesh = build_tp_meshes(shard_dim, tp_src, tp_dst)
        src_bytes = prod(tile_shape(global_shape, src_mesh)) * elem
        dst_bytes = prod(tile_shape(global_shape, dst_mesh)) * elem
        window_bytes = max(window_bytes, src_bytes, dst_bytes)

    return params, window_bytes
