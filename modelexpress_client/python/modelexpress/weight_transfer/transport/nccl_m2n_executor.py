# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-specific preparation and whole-version staging for NCCL M2N.

Executors prepare lane batches; :class:`_M2nRuntime` is the only component that
submits M2N calls. On processes owning multiple PP-pair lanes, callers must use
``execute_batch`` so the runtime can enqueue lanes by canonical key.

Whole-version consistency is local to one destination process. The caller must
keep serving quiesced across the complete destination cohort until every rank
reports a successful update. If a commit fails after live copies start, serving
must remain disabled until the model and runtime are reinitialized.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from ..planner.mesh import build_tp_meshes, shard_dim_from_trainer_tensor, tile_shape
from .nccl_m2n_runtime import (
    _M2nCall,
    _M2nLane,
    _M2nLaneBatch,
    _M2nRuntime,
)

if TYPE_CHECKING:
    from ..protocol.types import TrainerTable

logger = logging.getLogger("modelexpress.weight_transfer.nccl_m2n_executor")


@dataclass
class ReshardParam:
    """One local parameter tile plus globally shared layout metadata."""

    name: str
    global_shape: tuple[int, ...]
    shard_dim: int
    local_tensor: Any

    @property
    def local_nbytes(self) -> int:
        return int(self.local_tensor.numel() * self.local_tensor.element_size())


class NcclM2nExecutor:
    """Prepare complete model versions for one runtime-owned lane."""

    def __init__(
        self,
        runtime: _M2nRuntime,
        lane: _M2nLane,
        *,
        tp_src: int,
        tp_dst: int,
    ) -> None:
        if tp_src <= 0 or tp_dst <= 0:
            raise ValueError(f"tp_src and tp_dst must be positive, got {tp_src}/{tp_dst}")
        if lane.nranks != tp_src + tp_dst:
            raise ValueError(
                f"lane size {lane.nranks} != tp_src {tp_src} + tp_dst {tp_dst}"
            )

        self._runtime = runtime
        self._lane = lane
        self._tp_src = int(tp_src)
        self._tp_dst = int(tp_dst)
        self._is_src = lane.comm_rank < tp_src
        self._staged: list[Any] = []
        self._staging_signature: tuple[Any, ...] | None = None
        self._poisoned = False
        self._stream_failed = False
        self._commit_started = False
        self._prepare_lock = threading.Lock()

    def execute(self, params: list[ReshardParam]) -> tuple[int, float]:
        """Execute one update when this process owns exactly one lane."""
        return self.execute_batch([(self, params)])[0]

    @staticmethod
    def execute_batch(
        updates: Sequence[tuple["NcclM2nExecutor", list[ReshardParam]]],
    ) -> list[tuple[int, float]]:
        """Prepare and canonically submit every active lane for one update."""
        if not updates:
            return []
        executors = [executor for executor, _ in updates]
        if len(set(executors)) != len(executors):
            raise ValueError("an executor may appear only once in an M2N update batch")
        runtime = executors[0]._runtime
        if any(executor._runtime is not runtime for executor in executors):
            raise ValueError("all M2N update executors must share one _M2nRuntime")

        ordered_updates = sorted(updates, key=lambda update: update[0]._lane.key)
        acquired: list[NcclM2nExecutor] = []
        try:
            # Preparation state is executor-local. Canonical lock acquisition
            # prevents two racing model updates from deadlocking.
            for executor, _ in ordered_updates:
                executor._prepare_lock.acquire()
                acquired.append(executor)

            batches = [
                executor._prepare_lane_batch(params)
                for executor, params in ordered_updates
            ]
            if all(not batch.calls for batch in batches):
                return [(0, 0.0) for _ in updates]

            start = time.perf_counter()
            byte_counts = runtime.dispatch_batch(batches)
            elapsed = time.perf_counter() - start
            by_executor = {
                executor: (byte_counts[executor._lane.lane_id], elapsed)
                for executor, _ in ordered_updates
            }
            for executor, _ in ordered_updates:
                total_bytes, duration = by_executor[executor]
                gbps = (total_bytes * 8) / (duration * 1e9) if duration > 0 else 0.0
                logger.info(
                    "reshard complete: lane=%s key=%s %.2f GB in %.3fs (%.1f Gbps)",
                    executor._lane.lane_id,
                    executor._lane.key,
                    total_bytes / 1e9,
                    duration,
                    gbps,
                )
            return [by_executor[executor] for executor, _ in updates]
        finally:
            for executor in reversed(acquired):
                executor._prepare_lock.release()

    def _prepare_lane_batch(self, params: list[ReshardParam]) -> _M2nLaneBatch:
        if self._poisoned:
            raise RuntimeError(
                "nccl_m2n executor is unusable after a failed model commit; "
                "reinitialize the model and executor before serving or transferring again"
            )
        if self._stream_failed:
            raise RuntimeError(
                "nccl_m2n executor stream could not be drained after a transfer failure; "
                "reinitialize the runtime before transferring again"
            )
        if not params:
            return _M2nLaneBatch(lane=self._lane, calls=(), total_bytes=0)

        self._validate_local_tiles(params)
        self._ensure_staging(params)
        calls: list[_M2nCall] = []
        for index, param in enumerate(params):
            src_mesh, dst_mesh = build_tp_meshes(
                param.shard_dim,
                self._tp_src,
                self._tp_dst,
            )
            calls.append(
                _M2nCall(
                    src=param.local_tensor if self._is_src else None,
                    dst=None if self._is_src else self._staged[index],
                    src_mesh=src_mesh,
                    dst_mesh=dst_mesh,
                    src_local_shape=tile_shape(param.global_shape, src_mesh),
                    dst_local_shape=tile_shape(param.global_shape, dst_mesh),
                    dtype=param.local_tensor.dtype,
                )
            )

        self._commit_started = False
        return _M2nLaneBatch(
            lane=self._lane,
            calls=tuple(calls),
            total_bytes=sum(param.local_nbytes for param in params),
            commit=None if self._is_src else lambda: self._enqueue_commit(params),
            on_complete=self._mark_complete,
            on_failure=self._mark_failure,
        )

    def _validate_local_tiles(self, params: list[ReshardParam]) -> None:
        for param in params:
            if not param.global_shape:
                raise ValueError(f"parameter {param.name!r} has an empty global shape")
            if len(param.global_shape) > 3:
                raise ValueError(
                    f"parameter {param.name!r} rank {len(param.global_shape)} exceeds M2N limit 3"
                )
            if not param.local_tensor.is_contiguous():
                raise ValueError(f"parameter {param.name!r} must be contiguous for M2N")

            src_mesh, dst_mesh = build_tp_meshes(
                param.shard_dim,
                self._tp_src,
                self._tp_dst,
            )
            expected = tile_shape(
                param.global_shape,
                src_mesh if self._is_src else dst_mesh,
            )
            actual = tuple(int(dim) for dim in param.local_tensor.shape)
            if actual != expected:
                side = "source" if self._is_src else "destination"
                raise ValueError(
                    f"{side} tile shape mismatch for {param.name!r}: "
                    f"local={actual}, expected={expected}"
                )

    def _ensure_staging(self, params: list[ReshardParam]) -> None:
        if self._is_src:
            return
        signature = tuple(
            (
                tuple(int(dim) for dim in param.local_tensor.shape),
                param.local_tensor.dtype,
                param.local_tensor.device,
            )
            for param in params
        )
        if signature == self._staging_signature:
            return

        import torch

        with self._runtime.stream_context(self._lane):
            staged = [
                torch.empty_like(param.local_tensor, memory_format=torch.preserve_format)
                for param in params
            ]
        for tensor in staged:
            if tensor.is_cuda:
                tensor.record_stream(self._lane.stream)
        self._staged = staged
        self._staging_signature = signature

    def _copy_into_live(self, param: ReshardParam, staged: Any) -> None:
        param.local_tensor.copy_(staged, non_blocking=True)

    def _enqueue_commit(self, params: list[ReshardParam]) -> None:
        """Enqueue live-model copies; the runtime synchronizes every lane later."""
        import torch

        self._poisoned = True
        self._commit_started = True
        try:
            with torch.no_grad(), self._runtime.stream_context(self._lane):
                for param, staged in zip(params, self._staged, strict=True):
                    self._copy_into_live(param, staged)
        except BaseException as exc:
            raise RuntimeError(
                "nccl_m2n model commit failed after live-model modification began; "
                "serving must remain stopped until the model and runtime are reinitialized"
            ) from exc

    def _mark_complete(self) -> None:
        self._poisoned = False
        self._commit_started = False

    def _mark_failure(self, exc: BaseException) -> None:
        if self._commit_started:
            self._poisoned = True
        if "stream" in str(exc).lower():
            self._stream_failed = True

    def teardown(self) -> None:
        """Release model staging; runtime/lane ownership remains external."""
        try:
            self._runtime.synchronize_lane(self._lane)
        except BaseException as exc:
            self._stream_failed = True
            self._runtime.poison_lane(self._lane)
            raise RuntimeError(
                "cannot safely tear down nccl_m2n executor because its lane stream "
                "could not be drained; staging tensors were retained"
            ) from exc
        self._staged = []
        self._staging_signature = None


def build_reshard_params(
    model: Any,
    table: TrainerTable,
    tp_src: int,
    tp_dst: int,
) -> list[ReshardParam]:
    """Build local tensors plus globally shared M2N layout metadata."""
    if tp_src <= 0 or tp_dst <= 0:
        raise ValueError(f"tp_src and tp_dst must be positive, got {tp_src}/{tp_dst}")

    named = dict(model.named_parameters())
    params: list[ReshardParam] = []
    for tensor in table.tensors:
        param = named.get(tensor.name)
        if param is None:
            raise RuntimeError(
                f"trainer table parameter {tensor.name!r} is not present in the local model; "
                "every rank in a reshard cohort must contribute the same parameter set, "
                "since reshard operations are collectives"
            )
        params.append(
            ReshardParam(
                name=tensor.name,
                global_shape=tuple(tensor.shape),
                shard_dim=shard_dim_from_trainer_tensor(tensor),
                local_tensor=param,
            )
        )
    return params


__all__ = ["NcclM2nExecutor", "ReshardParam", "build_reshard_params"]
