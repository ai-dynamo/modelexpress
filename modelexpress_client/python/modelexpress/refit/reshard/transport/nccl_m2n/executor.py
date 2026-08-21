# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo-style tensor preparation and whole-version staging for NCCL M2N.

One process-level executor prepares every local PP transfer group. Runtime then
submits one official M2N group in canonical PP-group order. MX owns PP parent
communicators and CUDA streams in this first integration.

Source contract: caller invokes :meth:`execute` after producer work has been
enqueued on current CUDA stream. If producers use other streams, caller must
make current stream wait for them first. Runtime records one readiness event on
current stream and makes every source PP stream wait for it. Source tensors
must remain allocated and unmodified until :meth:`execute` returns.

Whole-version consistency is local to one destination process. Caller must keep
serving quiesced across complete destination cohort until every rank reports a
successful update. If live commit fails, serving stays disabled until reload.

Known collective contract: every rank in one PP group must provide identical
parameter count/order. This PR documents but does not solve cross-rank plan
agreement. Source-only overlap across different PP groups is intentionally
allowed; within-group overlap and any cross-group destination overlap are not.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .mesh import REPLICATE, build_tp_meshes, tile_shape
from .runtime import (
    _M2nCall,
    _M2nPPGroup,
    _M2nPPGroupBatch,
    _M2nRuntime,
    _PPGroupKey,
)

logger = logging.getLogger("modelexpress.refit.reshard.nccl_m2n_executor")


@dataclass(frozen=True)
class ReshardParam:
    """One local NeMo/Megatron tensor plus shared logical layout."""

    name: str
    global_shape: tuple[int, ...]
    shard_dim: int
    local_tensor: Any
    local_shard_index: int | None = None

    @property
    def local_nbytes(self) -> int:
        return int(self.local_tensor.numel() * self.local_tensor.element_size())


@dataclass
class _PPGroupModelState:
    pp_group: _M2nPPGroup
    staged: list[Any] = field(default_factory=list)
    staging_signature: tuple[Any, ...] | None = None
    commit_started: bool = False


@dataclass(frozen=True)
class _StorageRegion:
    pp_group_key: _PPGroupKey
    param_name: str
    is_destination: bool
    device: tuple[str, int | None]
    start: int
    end: int

    def overlaps(self, other: _StorageRegion) -> bool:
        return (
            self.device == other.device
            and self.start < other.end
            and other.start < self.end
        )


class NcclM2nExecutor:
    """Prepare and execute complete updates for every local PP group."""

    def __init__(self, runtime: _M2nRuntime) -> None:
        self._runtime = runtime
        pp_groups = runtime.freeze_pp_groups()
        self._states = {
            pp_group.key: _PPGroupModelState(pp_group=pp_group)
            for pp_group in pp_groups
        }
        self._execute_lock = threading.Lock()
        self._poisoned = False
        self._stream_failed = False

    @property
    def pp_group_keys(self) -> tuple[_PPGroupKey, ...]:
        return tuple(sorted(self._states))

    def execute(
        self,
        updates_by_pp_group: Mapping[_PPGroupKey, Sequence[ReshardParam]],
    ) -> dict[_PPGroupKey, tuple[int, float]]:
        """Prepare and canonically submit one complete local model update."""
        with self._execute_lock:
            self._require_usable()
            self._validate_update_keys(updates_by_pp_group)
            ordered_updates = [
                (self._states[key], list(updates_by_pp_group[key]))
                for key in self.pp_group_keys
            ]
            self._validate_storage_overlap(ordered_updates)
            batches = [
                self._prepare_pp_group_batch(state, params)
                for state, params in ordered_updates
            ]

            start = time.perf_counter()
            byte_counts = self._runtime.submit_model_update(batches)
            elapsed = time.perf_counter() - start
            results = {key: (byte_counts[key], elapsed) for key in self.pp_group_keys}
            for key, (total_bytes, duration) in results.items():
                gbps = (total_bytes * 8) / (duration * 1e9) if duration > 0 else 0.0
                logger.info(
                    "reshard complete: pp_group=%s %.2f GB in %.3fs (%.1f Gbps)",
                    key,
                    total_bytes / 1e9,
                    duration,
                    gbps,
                )
            return results

    def _require_usable(self) -> None:
        if self._poisoned:
            raise RuntimeError(
                "nccl_m2n executor is unusable after a failed model commit or "
                "M2N submission; reinitialize model, runtime, and executor"
            )
        if self._stream_failed:
            raise RuntimeError(
                "nccl_m2n executor stream could not be drained after failure; "
                "reinitialize runtime before transferring again"
            )

    def _validate_update_keys(
        self,
        updates_by_pp_group: Mapping[_PPGroupKey, Sequence[ReshardParam]],
    ) -> None:
        expected = set(self._states)
        provided = set(updates_by_pp_group)
        if provided != expected:
            raise ValueError(
                "M2N update must contain every local PP group exactly once; "
                f"missing={sorted(expected - provided)}, "
                f"unexpected={sorted(provided - expected)}"
            )

    def _prepare_pp_group_batch(
        self,
        state: _PPGroupModelState,
        params: list[ReshardParam],
    ) -> _M2nPPGroupBatch:
        pp_group = state.pp_group
        if not params:
            return _M2nPPGroupBatch(
                pp_group=pp_group,
                calls=(),
                total_bytes=0,
            )

        self._validate_local_tiles(pp_group, params)
        self._ensure_staging(state, params)
        calls: list[_M2nCall] = []
        for index, param in enumerate(params):
            src_mesh, dst_mesh = build_tp_meshes(
                param.shard_dim,
                pp_group.source_size,
                pp_group.destination_size,
            )
            calls.append(
                _M2nCall.from_param(
                    self._runtime.m2n,
                    name=param.name,
                    src_buffer=param.local_tensor if pp_group.is_source else None,
                    dst_buffer=(None if pp_group.is_source else state.staged[index]),
                    src_mesh=src_mesh,
                    dst_mesh=dst_mesh,
                    src_local_shape=tile_shape(param.global_shape, src_mesh),
                    dst_local_shape=tile_shape(param.global_shape, dst_mesh),
                    dtype=param.local_tensor.dtype,
                )
            )

        state.commit_started = False
        return _M2nPPGroupBatch(
            pp_group=pp_group,
            calls=tuple(calls),
            total_bytes=sum(param.local_nbytes for param in params),
            commit=(
                None
                if pp_group.is_source
                else lambda: self._enqueue_commit(state, params)
            ),
            on_complete=lambda: self._mark_complete(state),
            on_failure=lambda exc: self._mark_failure(state, exc),
        )

    @staticmethod
    def _validate_local_tiles(
        pp_group: _M2nPPGroup,
        params: Sequence[ReshardParam],
    ) -> None:
        names: set[str] = set()
        for param in params:
            if not param.name:
                raise ValueError("M2N parameter name must not be empty")
            if param.name in names:
                raise ValueError(
                    f"duplicate parameter {param.name!r} in PP group {pp_group.key}"
                )
            names.add(param.name)
            if not param.global_shape:
                raise ValueError(f"parameter {param.name!r} has an empty global shape")
            if len(param.global_shape) > 3:
                raise ValueError(
                    f"parameter {param.name!r} rank {len(param.global_shape)} "
                    "exceeds M2N limit 3"
                )
            if any(int(dim) <= 0 for dim in param.global_shape):
                raise ValueError(
                    f"parameter {param.name!r} has invalid global shape "
                    f"{param.global_shape}"
                )
            if not param.local_tensor.is_contiguous():
                raise ValueError(f"parameter {param.name!r} must be contiguous for M2N")
            if param.shard_dim == REPLICATE:
                if param.local_shard_index is not None:
                    raise ValueError(
                        f"replicated parameter {param.name!r} cannot set a local "
                        "shard index"
                    )
            else:
                if not 0 <= param.shard_dim < len(param.global_shape):
                    raise ValueError(
                        f"parameter {param.name!r} has invalid shard dim "
                        f"{param.shard_dim}"
                    )
                if param.local_shard_index is None:
                    raise ValueError(
                        f"sharded parameter {param.name!r} requires local_shard_index"
                    )
                expected_shard_index = (
                    pp_group.comm_rank
                    if pp_group.is_source
                    else pp_group.comm_rank - pp_group.source_size
                )
                if param.local_shard_index != expected_shard_index:
                    raise ValueError(
                        f"parameter {param.name!r} local shard index "
                        f"{param.local_shard_index} does not match PP group "
                        f"communicator shard index {expected_shard_index}"
                    )

            src_mesh, dst_mesh = build_tp_meshes(
                param.shard_dim,
                pp_group.source_size,
                pp_group.destination_size,
            )
            expected = tile_shape(
                param.global_shape,
                src_mesh if pp_group.is_source else dst_mesh,
            )
            actual = tuple(int(dim) for dim in param.local_tensor.shape)
            if actual != expected:
                side = "source" if pp_group.is_source else "destination"
                raise ValueError(
                    f"{side} tile shape mismatch for {param.name!r} in PP group "
                    f"{pp_group.key}: local={actual}, expected={expected}"
                )

    def _ensure_staging(
        self,
        state: _PPGroupModelState,
        params: Sequence[ReshardParam],
    ) -> None:
        if state.pp_group.is_source:
            return
        signature = tuple(
            (
                param.name,
                tuple(int(dim) for dim in param.local_tensor.shape),
                param.local_tensor.dtype,
                param.local_tensor.device,
            )
            for param in params
        )
        if signature == state.staging_signature:
            return

        import torch

        with self._runtime.stream_context(state.pp_group):
            staged = [
                torch.empty_like(
                    param.local_tensor,
                    memory_format=torch.preserve_format,
                )
                for param in params
            ]
        for tensor in staged:
            if tensor.is_cuda:
                tensor.record_stream(state.pp_group.stream)
        state.staged = staged
        state.staging_signature = signature

    @staticmethod
    def _copy_into_live(param: ReshardParam, staged: Any) -> None:
        param.local_tensor.copy_(staged, non_blocking=True)

    def _enqueue_commit(
        self,
        state: _PPGroupModelState,
        params: Sequence[ReshardParam],
    ) -> None:
        """Enqueue live copies after every PP group completed staging."""
        import torch

        self._poisoned = True
        state.commit_started = True
        try:
            with torch.no_grad(), self._runtime.stream_context(state.pp_group):
                for param, staged in zip(params, state.staged, strict=True):
                    self._copy_into_live(param, staged)
        except BaseException as exc:
            raise RuntimeError(
                "nccl_m2n model commit failed after live-model modification began; "
                "serving must remain stopped until model and runtime are reinitialized"
            ) from exc

    def _mark_complete(self, state: _PPGroupModelState) -> None:
        state.commit_started = False
        if not any(item.commit_started for item in self._states.values()):
            self._poisoned = False

    def _mark_failure(
        self,
        state: _PPGroupModelState,
        exc: BaseException,
    ) -> None:
        # Any failure after M2N enqueue poisons runtime. Commit failure can leave
        # live weights partial. Keep executor unusable in both cases.
        self._poisoned = True
        if "stream" in str(exc).lower():
            self._stream_failed = True
        if state.commit_started:
            self._poisoned = True

    @staticmethod
    def _storage_region(
        pp_group: _M2nPPGroup,
        param: ReshardParam,
    ) -> _StorageRegion | None:
        nbytes = param.local_nbytes
        if nbytes == 0:
            return None
        device = param.local_tensor.device
        return _StorageRegion(
            pp_group_key=pp_group.key,
            param_name=param.name,
            is_destination=pp_group.is_destination,
            device=(str(device.type), device.index),
            start=int(param.local_tensor.data_ptr()),
            end=int(param.local_tensor.data_ptr()) + nbytes,
        )

    def _validate_storage_overlap(
        self,
        updates: Sequence[tuple[_PPGroupModelState, Sequence[ReshardParam]]],
    ) -> None:
        regions: list[_StorageRegion] = []
        for state, params in updates:
            group_regions = [
                region
                for param in params
                if (region := self._storage_region(state.pp_group, param)) is not None
            ]
            self._reject_overlaps(
                group_regions,
                context=f"within PP group {state.pp_group.key}",
            )
            regions.extend(group_regions)

        for index, left in enumerate(regions):
            for right in regions[index + 1 :]:
                if left.pp_group_key == right.pp_group_key:
                    continue
                if not left.overlaps(right):
                    continue
                if not left.is_destination and not right.is_destination:
                    # User-approved contract: source storage may feed multiple
                    # M2N buckets, remains immutable, and is held through drain.
                    continue
                raise ValueError(
                    "M2N destination storage overlap across PP groups is unsupported: "
                    f"{left.pp_group_key}/{left.param_name!r} overlaps "
                    f"{right.pp_group_key}/{right.param_name!r}"
                )

    @staticmethod
    def _reject_overlaps(
        regions: Sequence[_StorageRegion],
        *,
        context: str,
    ) -> None:
        for index, left in enumerate(regions):
            for right in regions[index + 1 :]:
                if left.overlaps(right):
                    raise ValueError(
                        f"M2N tensor storage overlap {context}: "
                        f"{left.param_name!r} overlaps {right.param_name!r}"
                    )

    def teardown(self) -> None:
        """Drain PP streams and release whole-version staging."""
        with self._execute_lock:
            for state in self._states.values():
                try:
                    self._runtime.synchronize_pp_group(state.pp_group)
                except BaseException as exc:
                    self._stream_failed = True
                    raise RuntimeError(
                        "cannot safely tear down nccl_m2n executor because PP stream "
                        "could not drain; staging tensors retained"
                    ) from exc
            for state in self._states.values():
                state.staged = []
                state.staging_signature = None


def build_reshard_params(tensors: Sequence[Any]) -> list[ReshardParam]:
    """Translate NeMo/MegatronTensorSpec inputs into M2N planner inputs."""
    from modelexpress.refit.reshard.megatron_aliases import MegatronTensorSpec

    if not isinstance(tensors, Sequence) or isinstance(tensors, (str, bytes)):
        raise TypeError("tensors must be a sequence of MegatronTensorSpec")

    params: list[ReshardParam] = []
    names: set[str] = set()
    for item in tensors:
        if not isinstance(item, MegatronTensorSpec):
            raise TypeError("tensors must contain only MegatronTensorSpec")
        if item.name in names:
            raise ValueError(f"duplicate Megatron tensor name {item.name!r}")
        names.add(item.name)
        if not item.tensor.is_contiguous():
            raise ValueError(f"Megatron tensor {item.name!r} must be contiguous")
        local_shape = tuple(int(dim) for dim in item.tensor.shape)
        if len(local_shape) != len(item.global_shape):
            raise ValueError(
                f"Megatron tensor {item.name!r} local/global ranks disagree: "
                f"{local_shape} vs {item.global_shape}"
            )

        shard_dim = REPLICATE
        local_shard_index = None
        if item.placement_kind == "SHARD":
            if item.shard_axis is None or item.local_shard_range is None:
                raise ValueError(f"Megatron tensor {item.name!r} has no shard geometry")
            shard_dim = int(item.shard_axis)
            if not 0 <= shard_dim < len(item.global_shape):
                raise ValueError(
                    f"Megatron tensor {item.name!r} has invalid shard axis {shard_dim}"
                )
            lo, hi = (int(value) for value in item.local_shard_range)
            local_extent = int(item.tensor.shape[shard_dim])
            global_extent = int(item.global_shape[shard_dim])
            if local_extent <= 0:
                raise ValueError(
                    f"Megatron tensor {item.name!r} has an empty local shard"
                )
            if not 0 <= lo < hi <= global_extent or hi - lo != local_extent:
                raise ValueError(
                    f"Megatron tensor {item.name!r} has inconsistent local shard range"
                )
            if global_extent % local_extent or lo % local_extent:
                raise ValueError(
                    f"Megatron tensor {item.name!r} uses non-uniform sharding"
                )
            local_shard_index = lo // local_extent

        params.append(
            ReshardParam(
                name=item.name,
                global_shape=tuple(int(dim) for dim in item.global_shape),
                shard_dim=shard_dim,
                local_tensor=item.tensor,
                local_shard_index=local_shard_index,
            )
        )
    return params


def run_reshard(
    tensors_by_pp_group: Mapping[_PPGroupKey, Sequence[Any]],
    executor: NcclM2nExecutor,
) -> dict[_PPGroupKey, tuple[int, float]]:
    """Translate NeMo inputs and execute one complete PP-group update."""
    updates = {
        key: build_reshard_params(tensors)
        for key, tensors in tensors_by_pp_group.items()
    }
    return executor.execute(updates)


__all__ = [
    "NcclM2nExecutor",
    "ReshardParam",
    "build_reshard_params",
    "run_reshard",
]
