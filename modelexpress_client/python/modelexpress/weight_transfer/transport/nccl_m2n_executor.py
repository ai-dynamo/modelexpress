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
    dst rank:  reshard -> window base -> version staging

Destination ranks copy version staging into live parameters only after every
reshard succeeds.  ``execute`` does not expose separate stage and commit phases,
so the caller must quiesce serving across the destination cohort before entering
``execute`` and must not resume until every destination rank reports success.
Executor poisoning is local transfer state; it does not stop the serving engine.
On any commit failure, the caller must keep the cohort stopped and reinitialize
the affected model and executor.
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
        self._staging_buf: int | None = None
        self._staging_bytes: int = 0
        self._poisoned = False

    def _ensure_window(self, nbytes: int) -> None:
        """(Re)allocate the symmetric window if it is too small.

        The window is collectively registered, so every rank must size it to the
        same value -- callers pass a world-consistent worst case.
        """
        if self._window is not None and self._window_bytes >= nbytes:
            return
        self._release_window()
        self._window_buf = self._m2n.mem_alloc(nbytes)
        self._window = self._m2n.window_register(self._comm, self._window_buf, nbytes)
        self._window_bytes = nbytes

    def _release_window(self) -> None:
        if self._window is not None:
            self._m2n.window_deregister(self._comm, self._window)
            self._window = None
        if self._window_buf is not None:
            self._m2n.mem_free(self._window_buf)
            self._window_buf = None
        self._window_bytes = 0

    def _ensure_staging(self, nbytes: int) -> None:
        """Allocate destination-local storage for one complete model version."""
        if self._is_src or (self._staging_buf is not None and self._staging_bytes >= nbytes):
            return
        self._release_staging()
        self._staging_buf = self._m2n.mem_alloc(nbytes)
        self._staging_bytes = nbytes

    def _release_staging(self) -> None:
        if self._staging_buf is not None:
            self._m2n.mem_free(self._staging_buf)
            self._staging_buf = None
        self._staging_bytes = 0

    def _check_async_error(self, name: str) -> None:
        state = self._m2n.comm_get_async_error(self._comm)
        if state != binding.ncclSuccess:
            raise RuntimeError(f"nccl async error after reshard({name!r}): {state}")

    def execute(self, params: list[ReshardParam], window_bytes: int) -> tuple[int, float]:
        """Reshard every param.  ``window_bytes`` is the world-consistent window size.

        On destination ranks, all parameters are staged before any live parameter
        is changed.  Stage and commit are not separate public phases, so serving
        must remain quiesced from before this call until every destination rank
        reports success.  A commit failure makes this executor unusable because
        the live model may contain a partial version; poisoning this executor does
        not itself stop the serving engine.

        Returns ``(total_bytes_moved, elapsed_seconds)``.
        """
        if self._poisoned:
            raise RuntimeError(
                "nccl_m2n executor is unusable after a failed model commit; "
                "reinitialize the model and executor before serving or transferring again"
            )
        if not params:
            return 0, 0.0

        self._ensure_window(window_bytes)
        self._ensure_staging(sum(p.local_nbytes for p in params))

        start = time.perf_counter()

        try:
            total_bytes = self._reshard_all(params)
        except BaseException:
            self._release_window()
            raise

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

    def _reshard_all(self, params: list[ReshardParam]) -> int:
        total_bytes = 0
        staging_offset = 0

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

            self._m2n.reshard(self._comm, self._window, src_t, dst_t, self._stream)

            # Preserve the live model until the complete version is available.
            if not self._is_src:
                self._m2n.device_synchronize()
                self._check_async_error(p.name)
                self._m2n.memcpy_dtod(
                    self._staging_buf + staging_offset,
                    self._window_buf,
                    p.local_nbytes,
                )
                staging_offset += p.local_nbytes

            total_bytes += p.local_nbytes

        self._m2n.device_synchronize()
        if params:
            self._check_async_error(params[-1].name)
        if not self._is_src:
            self._commit_staged(params)
        return total_bytes

    def _commit_staged(self, params: list[ReshardParam]) -> None:
        """Install a complete staged version while serving remains quiesced."""
        self._poisoned = True
        staging_offset = 0
        try:
            for p in params:
                self._m2n.memcpy_dtod(
                    p.local_ptr,
                    self._staging_buf + staging_offset,
                    p.local_nbytes,
                )
                staging_offset += p.local_nbytes
            self._m2n.device_synchronize()
        except BaseException as exc:
            raise RuntimeError(
                "failed to commit staged model version; live parameters may contain "
                "mixed versions, so serving must remain stopped and the model and "
                "executor must be reinitialized"
            ) from exc
        self._poisoned = False

    def teardown(self) -> None:
        self._release_window()
        self._release_staging()
        if self._comm is not None:
            self._m2n.comm_destroy(self._comm)
            self._comm = None
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
    for attr, enum in (
        ("float8_e4m3fn", binding.ncclFloat8e4m3),
        ("float8_e5m2", binding.ncclFloat8e5m2),
    ):
        torch_dtype = getattr(torch, attr, None)
        if torch_dtype is not None:
            table[torch_dtype] = enum
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
            raise RuntimeError(
                f"trainer table parameter {tt.name!r} is not present in the local model; "
                "every rank in a reshard cohort must contribute the same parameter set, "
                "since reshard and window registration are collectives"
            )

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
