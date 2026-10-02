# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NeMo-style tensor preparation and whole-version staging for NCCL M2N.

One process-level executor prepares every local PP transfer group. Runtime then
submits one official M2N group in canonical PP-group order. MX owns PP parent
communicators and CUDA streams in this first integration.

Source contract: caller invokes :meth:`stage` after producer work has been
enqueued on current CUDA stream. If producers use other streams, caller must
make current stream wait for them first. Runtime records one readiness event on
current stream and makes every source PP stream wait for it. Source tensors
must remain allocated and unmodified until the staged update is released.

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
from enum import Enum, auto
from numbers import Integral
from typing import Any

from .mesh import REPLICATE, build_tp_meshes, tile_shape
from .runtime import (
    M2nCohortRestartRequired,
    _M2nCall,
    _M2nPPGroup,
    _M2nPPGroupBatch,
    _M2nPPGroupSpec,
    _M2nRuntime,
    _PPGroupKey,
)

logger = logging.getLogger("modelexpress.refit.reshard.nccl_m2n_executor")
_EXECUTOR_FACTORY_TOKEN = object()


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


@dataclass(frozen=True)
class M2nPPGroupBootstrap:
    """Control-plane material needed to create one M2N PP communicator."""

    group_id: str
    key: _PPGroupKey
    unique_id: bytes
    source_size: int
    destination_size: int
    comm_rank: int


class _StagedUpdateState(Enum):
    STAGED = auto()
    APPLIED = auto()
    RELEASED = auto()


class _ExecutorState(Enum):
    OPEN = auto()
    CLOSING = auto()
    POISONED = auto()
    CLOSED = auto()


@dataclass(eq=False)
class M2nStagedUpdate:
    """Opaque whole-version token returned by :meth:`NcclM2nExecutor.stage`."""

    _owner: NcclM2nExecutor
    _ordered_updates: tuple[tuple[_PPGroupModelState, tuple[ReshardParam, ...]], ...]
    _batches: tuple[_M2nPPGroupBatch, ...]
    _results: dict[_PPGroupKey, tuple[int, float]]
    _state: _StagedUpdateState = _StagedUpdateState.STAGED

    @property
    def results(self) -> dict[_PPGroupKey, tuple[int, float]]:
        """Return per-PP-group staged bytes and elapsed staging time."""
        return dict(self._results)


@dataclass
class _PPGroupModelState:
    pp_group: _M2nPPGroup
    staged: list[Any] = field(default_factory=list)
    staging_signature: tuple[Any, ...] | None = None


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
    """Stage and safely apply complete updates for every local PP group."""

    def __init__(
        self,
        runtime: _M2nRuntime,
        *,
        _factory_token: object | None = None,
        _enforce_cuda_tensors: bool = True,
    ) -> None:
        if _factory_token is not _EXECUTOR_FACTORY_TOKEN:
            raise TypeError("use NcclM2nExecutor.create()")
        self._runtime = runtime
        self._enforce_cuda_tensors = _enforce_cuda_tensors
        self._execute_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._lifecycle_state = _ExecutorState.OPEN
        self._poisoned = False
        self._stream_failed = False
        self._torn_down = False
        self._pending_update: M2nStagedUpdate | None = None
        self._states: dict[_PPGroupKey, _PPGroupModelState] = {}
        try:
            pp_groups = runtime._freeze_and_attach_executor(self)
            self._states = {
                pp_group.key: _PPGroupModelState(pp_group=pp_group)
                for pp_group in pp_groups
            }
        except BaseException:
            runtime._detach_executor(self)
            raise

    @classmethod
    def _create_for_tests(cls, runtime: _M2nRuntime) -> NcclM2nExecutor:
        """Private CPU/fake-backend constructor; production must use create()."""
        return cls(
            runtime,
            _factory_token=_EXECUTOR_FACTORY_TOKEN,
            _enforce_cuda_tensors=False,
        )

    @classmethod
    def create(
        cls,
        device_id: int,
        pp_groups: Sequence[M2nPPGroupBootstrap],
        *,
        max_cta: int | None = None,
        comm_init_timeout_s: float = 120.0,
        transfer_timeout_s: float = 900.0,
        finalize_timeout_s: float = 300.0,
    ) -> NcclM2nExecutor:
        """Create the process-owned runtime and all PP groups in one batch."""
        if not isinstance(pp_groups, Sequence) or isinstance(pp_groups, (str, bytes)):
            raise TypeError("pp_groups must be a sequence of M2nPPGroupBootstrap")
        items = tuple(pp_groups)
        device = cls._validate_bootstraps(device_id, items)
        restart_scope = tuple(
            sorted(
                (
                    (item.group_id, (int(item.key[0]), int(item.key[1])))
                    for item in items
                ),
                key=lambda item: item[1],
            )
        )
        runtime = _M2nRuntime(
            device,
            max_cta=max_cta,
            comm_init_timeout_s=comm_init_timeout_s,
            transfer_timeout_s=transfer_timeout_s,
            finalize_timeout_s=finalize_timeout_s,
            _restart_scope=restart_scope,
        )
        communicators_initialized = False
        try:
            runtime.create_pp_groups(
                tuple(
                    _M2nPPGroupSpec(
                        group_id=item.group_id,
                        key=(int(item.key[0]), int(item.key[1])),
                        unique_id=item.unique_id,
                        source_size=int(item.source_size),
                        destination_size=int(item.destination_size),
                        comm_rank=int(item.comm_rank),
                        device_id=device,
                    )
                    for item in items
                )
            )
            communicators_initialized = True
            return cls(
                runtime,
                _factory_token=_EXECUTOR_FACTORY_TOKEN,
                _enforce_cuda_tensors=True,
            )
        except M2nCohortRestartRequired:
            raise
        except BaseException as exc:
            if communicators_initialized:
                runtime._enter_fail_stop(runtime.pp_groups, (), exc)
                runtime._start_abort_worker()
                runtime._raise_restart_required(
                    operation="create",
                    phase="executor_attach",
                    reason="executor construction failed after PP-group creation",
                    cause=exc,
                )
            runtime.close()
            raise

    @staticmethod
    def _validate_bootstraps(
        device_id: int,
        pp_groups: Sequence[M2nPPGroupBootstrap],
    ) -> int:
        if not isinstance(device_id, Integral) or isinstance(device_id, bool):
            raise TypeError("M2N device_id must be an integer")
        device = int(device_id)
        if device < 0:
            raise ValueError("M2N device_id must be non-negative")
        if not isinstance(pp_groups, Sequence) or isinstance(pp_groups, (str, bytes)):
            raise TypeError("pp_groups must be a sequence of M2nPPGroupBootstrap")
        if not pp_groups:
            raise ValueError("at least one M2N PP group is required")

        ids: set[str] = set()
        keys: set[_PPGroupKey] = set()
        unique_ids: set[bytes] = set()
        for item in pp_groups:
            if not isinstance(item, M2nPPGroupBootstrap):
                raise TypeError(
                    "pp_groups must contain only M2nPPGroupBootstrap values"
                )
            if not isinstance(item.group_id, str):
                raise TypeError("M2N PP group_id must be a string")
            if not item.group_id:
                raise ValueError("M2N PP group_id must not be empty")
            if item.group_id in ids:
                raise ValueError(f"duplicate M2N PP group ID {item.group_id!r}")
            ids.add(item.group_id)
            if (
                not isinstance(item.key, tuple)
                or len(item.key) != 2
                or any(
                    not isinstance(stage, Integral)
                    or isinstance(stage, bool)
                    or int(stage) < 0
                    for stage in item.key
                )
            ):
                raise ValueError(
                    "M2N PP group key must be a pair of non-negative stage IDs"
                )
            if item.key in keys:
                raise ValueError(f"duplicate M2N PP group key {item.key}")
            keys.add(item.key)
            if not isinstance(item.unique_id, bytes):
                raise TypeError("M2N PP group unique_id must be bytes")
            if not item.unique_id:
                raise ValueError("M2N PP group unique_id must not be empty")
            if item.unique_id in unique_ids:
                raise ValueError("duplicate M2N PP group unique_id")
            unique_ids.add(item.unique_id)
            integral_fields = {
                "source_size": item.source_size,
                "destination_size": item.destination_size,
                "comm_rank": item.comm_rank,
            }
            for name, value in integral_fields.items():
                if not isinstance(value, Integral) or isinstance(value, bool):
                    raise TypeError(f"M2N PP group {name} must be an integer")
            source_size = int(item.source_size)
            destination_size = int(item.destination_size)
            comm_rank = int(item.comm_rank)
            if source_size <= 0 or destination_size <= 0:
                raise ValueError(
                    "M2N PP group source/destination sizes must be positive"
                )
            nranks = source_size + destination_size
            if not 0 <= comm_rank < nranks:
                raise ValueError(
                    f"invalid M2N PP group communicator rank " f"{comm_rank}/{nranks}"
                )
        return device

    @property
    def pp_group_keys(self) -> tuple[_PPGroupKey, ...]:
        return tuple(sorted(self._states))

    @staticmethod
    def _log_staged_result(
        key: _PPGroupKey,
        total_bytes: int,
        duration: float,
    ) -> None:
        """Keep optional logging failures outside the transfer contract."""
        gbps = (total_bytes * 8) / (duration * 1e9) if duration > 0 else 0.0
        try:
            logger.info(
                "reshard staged: pp_group=%s %.2f GB in %.3fs (%.1f Gbps)",
                key,
                total_bytes / 1e9,
                duration,
                gbps,
            )
        except Exception:  # noqa: BLE001 - logging must not fail a staged update.
            return

    def stage(
        self,
        updates_by_pp_group: Mapping[_PPGroupKey, Sequence[ReshardParam]],
    ) -> M2nStagedUpdate:
        """Receive one whole version without mutating destination live weights."""
        with self._execute_lock:
            with self._lifecycle_lock:
                self._require_usable_locked()
                if self._pending_update is not None:
                    raise RuntimeError(
                        "release the current M2N staged update before staging another"
                    )
            with self._runtime._active_operation():
                updates = self._snapshot_updates(updates_by_pp_group)
                self._validate_update_keys(updates)
                ordered_updates = tuple(
                    (self._states[key], updates[key]) for key in self.pp_group_keys
                )
                self._validate_update_preflight(ordered_updates)
                self._validate_storage_overlap(ordered_updates)
                batches = tuple(
                    self._prepare_pp_group_batch(state, params)
                    for state, params in ordered_updates
                )

                start = time.perf_counter()
                try:
                    byte_counts = self._runtime.submit_model_update(batches)
                except M2nCohortRestartRequired:
                    self._mark_poisoned()
                    raise
                elapsed = time.perf_counter() - start
                results = {
                    key: (byte_counts[key], elapsed) for key in self.pp_group_keys
                }
                update = M2nStagedUpdate(
                    _owner=self,
                    _ordered_updates=ordered_updates,
                    _batches=batches,
                    _results=results,
                )
                for key, (total_bytes, duration) in results.items():
                    self._log_staged_result(key, total_bytes, duration)
                with self._lifecycle_lock:
                    if self._lifecycle_state is _ExecutorState.POISONED:
                        self._require_usable_locked()
                    if self._lifecycle_state is _ExecutorState.CLOSED:
                        raise RuntimeError("nccl_m2n executor is closed")
                    # CLOSING preserves this already-admitted stage. close()
                    # waits for _execute_lock, then returns this token to caller.
                    self._pending_update = update
                return update

    @staticmethod
    def _snapshot_updates(
        updates_by_pp_group: Mapping[_PPGroupKey, Sequence[ReshardParam]],
    ) -> dict[_PPGroupKey, tuple[ReshardParam, ...]]:
        if not isinstance(updates_by_pp_group, Mapping):
            raise TypeError("updates_by_pp_group must be a mapping")
        snapshot: dict[_PPGroupKey, tuple[ReshardParam, ...]] = {}
        for raw_key, raw_params in tuple(updates_by_pp_group.items()):
            if (
                not isinstance(raw_key, tuple)
                or len(raw_key) != 2
                or any(
                    not isinstance(stage, Integral) or isinstance(stage, bool)
                    for stage in raw_key
                )
            ):
                raise TypeError(
                    "M2N update keys must be (trainer_stage, generator_stage) "
                    "integer pairs"
                )
            key = tuple(int(stage) for stage in raw_key)
            if key in snapshot:
                raise ValueError(f"duplicate normalized M2N PP group key {key}")
            if not isinstance(raw_params, Sequence) or isinstance(
                raw_params, (str, bytes)
            ):
                raise TypeError(f"M2N PP group {key} parameters must be a sequence")
            params = tuple(raw_params)
            for param in params:
                if not isinstance(param, ReshardParam):
                    raise TypeError(
                        f"M2N PP group {key} must contain only ReshardParam values"
                    )
            snapshot[key] = params
        return snapshot

    def _validate_update_preflight(
        self,
        updates: Sequence[tuple[_PPGroupModelState, Sequence[ReshardParam]]],
    ) -> None:
        """Purely validate the complete local plan before staging allocation."""
        for state, params in updates:
            for param in params:
                self._validate_param_protocol(param)
            self._validate_local_tiles(state.pp_group, params)

    def _validate_param_protocol(self, param: ReshardParam) -> None:
        if not isinstance(param.name, str):
            raise TypeError("M2N parameter name must be a string")
        if not param.name:
            raise ValueError("M2N parameter name must not be empty")
        if not isinstance(param.global_shape, tuple):
            raise TypeError(f"parameter {param.name!r} global_shape must be a tuple")
        if not param.global_shape:
            raise ValueError(f"parameter {param.name!r} has an empty global shape")
        for dim in param.global_shape:
            if not isinstance(dim, Integral) or isinstance(dim, bool):
                raise TypeError(
                    f"parameter {param.name!r} global_shape must contain integers"
                )
            if int(dim) <= 0:
                raise ValueError(
                    f"parameter {param.name!r} has invalid global shape "
                    f"{param.global_shape}"
                )
        if not isinstance(param.shard_dim, Integral) or isinstance(
            param.shard_dim, bool
        ):
            raise TypeError(f"parameter {param.name!r} shard_dim must be an integer")
        if param.local_shard_index is not None and (
            not isinstance(param.local_shard_index, Integral)
            or isinstance(param.local_shard_index, bool)
        ):
            raise TypeError(
                f"parameter {param.name!r} local_shard_index must be an integer"
            )

        tensor = param.local_tensor
        try:
            shape_value = tensor.shape
            dtype = tensor.dtype
            device = tensor.device
            is_cuda = tensor.is_cuda
        except AttributeError as exc:
            raise TypeError(
                f"parameter {param.name!r} tensor lacks " "shape/dtype/device/is_cuda"
            ) from exc
        try:
            shape = tuple(shape_value)
        except TypeError as exc:
            raise TypeError(
                f"parameter {param.name!r} tensor shape is not iterable"
            ) from exc
        for dim in shape:
            if not isinstance(dim, Integral) or isinstance(dim, bool):
                raise TypeError(
                    f"parameter {param.name!r} tensor shape must contain integers"
                )
            if int(dim) < 0:
                raise ValueError(
                    f"parameter {param.name!r} tensor shape must be non-negative"
                )
        if dtype is None:
            raise TypeError(f"parameter {param.name!r} tensor dtype is missing")
        try:
            device_type = device.type
            device_index = device.index
        except AttributeError as exc:
            raise TypeError(
                f"parameter {param.name!r} tensor device lacks type/index"
            ) from exc
        if not isinstance(device_type, str) or not device_type:
            raise TypeError(
                f"parameter {param.name!r} tensor device type must be a string"
            )
        if device_index is not None and (
            not isinstance(device_index, Integral) or isinstance(device_index, bool)
        ):
            raise TypeError(
                f"parameter {param.name!r} tensor device index must be an integer"
            )
        if not isinstance(is_cuda, bool):
            raise TypeError(f"parameter {param.name!r} tensor is_cuda must be bool")
        if self._enforce_cuda_tensors:
            if not is_cuda:
                raise ValueError(f"parameter {param.name!r} must be a CUDA tensor")
            if device_type != "cuda" or device_index is None:
                raise ValueError(
                    f"parameter {param.name!r} must identify a concrete CUDA device"
                )
            if int(device_index) != self._runtime.device_id:
                raise ValueError(
                    f"parameter {param.name!r} is on CUDA device "
                    f"{device_index}, expected {self._runtime.device_id}"
                )

        methods = {
            name: getattr(tensor, name, None)
            for name in (
                "numel",
                "element_size",
                "is_contiguous",
                "data_ptr",
                "copy_",
            )
        }
        missing = [name for name, method in methods.items() if not callable(method)]
        if missing:
            raise TypeError(
                f"parameter {param.name!r} tensor lacks callable " + ", ".join(missing)
            )
        try:
            numel = methods["numel"]()
            element_size = methods["element_size"]()
            contiguous = methods["is_contiguous"]()
            data_ptr = methods["data_ptr"]()
        except Exception as exc:
            raise TypeError(
                f"parameter {param.name!r} tensor protocol call failed"
            ) from exc
        integral_results = {
            "numel": numel,
            "element_size": element_size,
            "data_ptr": data_ptr,
        }
        for name, value in integral_results.items():
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise TypeError(
                    f"parameter {param.name!r} tensor {name}() must return an integer"
                )
        if int(numel) < 0 or int(element_size) <= 0 or int(data_ptr) < 0:
            raise ValueError(
                f"parameter {param.name!r} tensor protocol returned invalid sizes"
            )
        if not isinstance(contiguous, bool):
            raise TypeError(
                f"parameter {param.name!r} tensor is_contiguous() must return bool"
            )
        if not contiguous:
            raise ValueError(f"parameter {param.name!r} must be contiguous for M2N")

    def apply(
        self,
        update: M2nStagedUpdate,
    ) -> dict[_PPGroupKey, tuple[int, float]]:
        """Apply a staged version to destination live weights at a safe point."""
        with self._execute_lock:
            with self._lifecycle_lock:
                self._require_usable_locked()
                self._require_pending(update, _StagedUpdateState.STAGED)
            destination_updates = tuple(
                (state, params)
                for state, params in update._ordered_updates
                if state.pp_group.is_destination
            )
            apply_started = False
            try:
                with self._runtime._active_operation():
                    import torch

                    with torch.no_grad():
                        for state, params in destination_updates:
                            if not params:
                                continue
                            with self._runtime.stream_context(state.pp_group):
                                for param, staged in zip(
                                    params, state.staged, strict=True
                                ):
                                    apply_started = True
                                    self._copy_into_live(param, staged)
                    applied_groups = tuple(
                        state.pp_group
                        for state, params in destination_updates
                        if params
                    )
                    if applied_groups:
                        self._runtime.wait_for_pp_groups(
                            applied_groups,
                            operation="model-version apply",
                        )
            except BaseException as exc:
                if apply_started or isinstance(exc, M2nCohortRestartRequired):
                    self._mark_poisoned()
                    pp_groups = tuple(
                        state.pp_group for state, _ in destination_updates
                    )
                    self._runtime._enter_fail_stop(pp_groups, update._batches, exc)
                    self._runtime._start_abort_worker()
                    if isinstance(exc, M2nCohortRestartRequired):
                        raise
                    raise self._runtime._restart_error(
                        operation="apply",
                        phase="live_weight_copy",
                        reason="destination live-weight apply failed",
                    ) from exc
                raise
            update._state = _StagedUpdateState.APPLIED
            return update.results

    def release(self, update: M2nStagedUpdate) -> None:
        """Release one applied update or explicitly discard it before apply."""
        with self._execute_lock, self._lifecycle_lock:
            self._require_usable_locked()
            self._require_pending(
                update,
                _StagedUpdateState.STAGED,
                _StagedUpdateState.APPLIED,
            )
            update._ordered_updates = ()
            update._batches = ()
            update._state = _StagedUpdateState.RELEASED
            self._pending_update = None

    def _require_pending(
        self,
        update: M2nStagedUpdate,
        *allowed_states: _StagedUpdateState,
    ) -> None:
        if not isinstance(update, M2nStagedUpdate):
            raise TypeError("update must be an M2nStagedUpdate")
        if update._owner is not self:
            raise ValueError("M2N staged update belongs to another executor")
        if self._pending_update is not update:
            raise ValueError("M2N staged update is stale or already released")
        if update._state not in allowed_states:
            raise ValueError(
                f"M2N staged update is already {update._state.name.lower()}"
            )

    def _require_usable(self) -> None:
        with self._lifecycle_lock:
            self._require_usable_locked()

    def _require_usable_locked(self) -> None:
        if self._lifecycle_state is _ExecutorState.POISONED or self._poisoned:
            raise self._runtime._restart_error(
                operation="executor_operation",
                phase="admission",
                reason="NCCL M2N executor is in fail-stop state",
            )
        if self._stream_failed:
            raise self._runtime._restart_error(
                operation="executor_operation",
                phase="stream_drain",
                reason="NCCL M2N executor stream could not be drained",
            )
        if self._lifecycle_state is _ExecutorState.CLOSING:
            raise RuntimeError("nccl_m2n executor is closing")
        if self._lifecycle_state is _ExecutorState.CLOSED or self._torn_down:
            raise RuntimeError("nccl_m2n executor is closed")

    def _mark_poisoned(self) -> None:
        with self._lifecycle_lock:
            self._poisoned = True
            self._lifecycle_state = _ExecutorState.POISONED

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
        params: Sequence[ReshardParam],
    ) -> _M2nPPGroupBatch:
        pp_group = state.pp_group
        if not params:
            return _M2nPPGroupBatch(
                pp_group=pp_group,
                calls=(),
                total_bytes=0,
            )

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

        return _M2nPPGroupBatch(
            pp_group=pp_group,
            calls=tuple(calls),
            total_bytes=sum(param.local_nbytes for param in params),
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

    def close(self) -> None:
        """Close sequentially and idempotently; concurrent close fails fast."""
        if not self._close_lock.acquire(blocking=False):
            raise RuntimeError("nccl_m2n executor is already closing")
        execute_acquired = False
        try:
            with self._lifecycle_lock:
                if self._lifecycle_state is _ExecutorState.CLOSED:
                    return
                # Fatal state wins over the ordinary pending-token error. The
                # token and all tensor references remain quarantined.
                if (
                    self._lifecycle_state is _ExecutorState.POISONED
                    or self._poisoned
                    or self._stream_failed
                ):
                    self._require_usable_locked()
                if self._lifecycle_state is _ExecutorState.CLOSING:
                    if not self._runtime._native_closed:
                        raise RuntimeError("nccl_m2n executor is already closing")
                else:
                    self._lifecycle_state = _ExecutorState.CLOSING

            deadline = time.monotonic() + self._runtime._finalize_timeout_s
            if not self._runtime._native_closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._execute_lock.acquire(timeout=remaining):
                    cause = TimeoutError(
                        "timed out waiting for an admitted M2N executor operation"
                    )
                    self._runtime._enter_fail_stop(
                        self._runtime.pp_groups,
                        (),
                        cause,
                    )
                    self._runtime._start_abort_worker()
                    self._mark_poisoned()
                    self._runtime._raise_restart_required(
                        operation="close",
                        phase="executor_operation_drain",
                        reason=(self._runtime._fail_stop_reason or str(cause)),
                        cause=cause,
                    )
                execute_acquired = True

                with self._lifecycle_lock:
                    if self._lifecycle_state is _ExecutorState.POISONED:
                        self._require_usable_locked()
                    if self._pending_update is not None:
                        self._lifecycle_state = _ExecutorState.OPEN
                        raise RuntimeError(
                            "release the current M2N staged update before closing"
                        )

            try:
                self._runtime.close(owner=self, _deadline=deadline)
            except M2nCohortRestartRequired:
                self._mark_poisoned()
                raise
            except BaseException:
                with self._lifecycle_lock:
                    self._lifecycle_state = (
                        _ExecutorState.CLOSING
                        if self._runtime._native_closed
                        else _ExecutorState.OPEN
                    )
                raise

            for state in self._states.values():
                state.staged = []
                state.staging_signature = None
            with self._lifecycle_lock:
                self._torn_down = True
                self._lifecycle_state = _ExecutorState.CLOSED
        finally:
            if execute_acquired:
                self._execute_lock.release()
            self._close_lock.release()


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


__all__ = [
    "M2nCohortRestartRequired",
    "M2nPPGroupBootstrap",
    "M2nStagedUpdate",
    "NcclM2nExecutor",
    "ReshardParam",
    "build_reshard_params",
]
