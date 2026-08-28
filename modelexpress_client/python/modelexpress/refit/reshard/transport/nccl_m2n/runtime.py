# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal process-level owner for NCCL M2N PP transfer groups.

Current M2N keeps process-global state. MX therefore owns one explicit M2N
handle per process/GPU and retains every parent communicator until M2N
finalization. PP transfer groups use globally stable
``(trainer_stage, generator_stage)`` keys.

Preparation may be concurrent; M2N submission is single-dispatcher and
canonically ordered. One model update records every call inside one official
``nccl.m2n.group()``. Calls for each PP group remain parameter ordered; PP
groups are submitted in ascending key order. Each group has its own MX-owned
CUDA stream, so GPU work may overlap after host enqueue.
"""

from __future__ import annotations

import importlib
import logging
import math
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from numbers import Integral
from typing import Any

from .mesh import REPLICATE
from .mesh import Mesh as PlannerMesh

_PPGroupKey = tuple[int, int]
_NCCL_SUCCESS = 0
_NCCL_IN_PROGRESS = 7
logger = logging.getLogger("modelexpress.refit.reshard.nccl_m2n_runtime")


class _RuntimeState(Enum):
    OPEN = auto()
    CLOSING = auto()
    POISONED = auto()
    CLOSED = auto()


class _PPGroupState(Enum):
    OPEN = auto()
    POISONED = auto()
    CLOSED = auto()


@dataclass(frozen=True)
class _M2nPPGroupSpec:
    """Bootstrap material for one MX-owned PP-pair communicator."""

    group_id: str
    key: _PPGroupKey
    unique_id: bytes
    source_size: int
    destination_size: int
    comm_rank: int
    device_id: int

    @property
    def nranks(self) -> int:
        return self.source_size + self.destination_size


@dataclass(eq=False)
class _M2nPPGroup:
    """One runtime-owned parent communicator and explicit CUDA stream."""

    group_id: str
    key: _PPGroupKey
    communicator: Any
    source_size: int
    destination_size: int
    comm_rank: int
    device_id: int
    stream: Any
    state: _PPGroupState = _PPGroupState.OPEN
    abort_attempted: bool = False
    aborted: bool = False

    @property
    def nranks(self) -> int:
        return self.source_size + self.destination_size

    @property
    def is_source(self) -> bool:
        return self.comm_rank < self.source_size

    @property
    def is_destination(self) -> bool:
        return not self.is_source


def _convert_layout(
    m2n: Any,
    mesh: PlannerMesh,
    tensor_ndim: int,
) -> tuple[Any, tuple[Any, Any]]:
    m2n_mesh = m2n.Mesh(mesh.dims, start_rank=mesh.start_rank)
    if all(placement == REPLICATE for placement in mesh.placement):
        if tensor_ndim < 1 or mesh.dims[0] != 1:
            raise ValueError(
                "fully replicated M2N layout requires a size-one mesh axis"
            )
        return m2n_mesh, (m2n.Shard(0), m2n.Replicate())

    placements = tuple(
        m2n.Replicate() if placement == REPLICATE else m2n.Shard(placement)
        for placement in mesh.placement
    )
    return m2n_mesh, placements


@dataclass(frozen=True)
class _M2nCall:
    """Official M2N descriptors for one parameter collective."""

    name: str
    src: Any
    dst: Any

    @classmethod
    def from_param(
        cls,
        m2n: Any,
        *,
        name: str,
        src_buffer: Any | None,
        dst_buffer: Any | None,
        src_mesh: PlannerMesh,
        dst_mesh: PlannerMesh,
        src_local_shape: tuple[int, ...],
        dst_local_shape: tuple[int, ...],
        dtype: Any,
    ) -> _M2nCall:
        src_m2n_mesh, src_placements = _convert_layout(
            m2n,
            src_mesh,
            len(src_local_shape),
        )
        dst_m2n_mesh, dst_placements = _convert_layout(
            m2n,
            dst_mesh,
            len(dst_local_shape),
        )
        return cls(
            name=name,
            src=m2n.DistTensor(
                src_buffer,
                src_local_shape,
                dtype,
                mesh=src_m2n_mesh,
                placements=src_placements,
            ),
            dst=m2n.DistTensor(
                dst_buffer,
                dst_local_shape,
                dtype,
                mesh=dst_m2n_mesh,
                placements=dst_placements,
            ),
        )


@dataclass(frozen=True)
class _M2nPPGroupBatch:
    """Ordered parameter calls for one PP transfer group and model update."""

    pp_group: _M2nPPGroup
    calls: tuple[_M2nCall, ...]
    total_bytes: int
    commit: Callable[[], None] | None = None
    on_complete: Callable[[], None] | None = None
    on_failure: Callable[[BaseException], None] | None = None


def _version_tuple(version: object) -> tuple[int, int, int]:
    if isinstance(version, Integral) and not isinstance(version, bool):
        encoded = int(version)
        return encoded // 10_000, encoded // 100 % 100, encoded % 100

    release = getattr(version, "release", None)
    if not release:
        raise TypeError(
            "loaded libnccl version must be a packed integer or expose a "
            f"non-empty .release tuple, got {version!r}"
        )
    try:
        components = [int(value) for value in release]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"loaded libnccl version has an invalid .release tuple: {release!r}"
        ) from exc
    return tuple((components + [0, 0, 0])[:3])


class _M2nRuntime:
    """Own one M2N handle, all PP groups, and all M2N submission."""

    _singleton_lock = threading.Lock()
    _live_runtime: _M2nRuntime | None = None

    def __init__(
        self,
        device_id: int,
        *,
        max_cta: int | None = None,
        comm_init_timeout_s: float = 120.0,
        transfer_timeout_s: float = 900.0,
        finalize_timeout_s: float = 300.0,
        _poll_interval_s: float = 0.002,
        _m2n_module: Any | None = None,
        _nccl_module: Any | None = None,
        _torch_module: Any | None = None,
        _enforce_singleton: bool = True,
    ) -> None:
        self._device_id = int(device_id)
        self._comm_init_timeout_s = self._positive_finite_timeout(
            "comm_init_timeout_s",
            comm_init_timeout_s,
        )
        self._transfer_timeout_s = self._positive_finite_timeout(
            "transfer_timeout_s",
            transfer_timeout_s,
        )
        self._finalize_timeout_s = float(finalize_timeout_s)
        self._positive_finite_timeout(
            "finalize_timeout_s",
            self._finalize_timeout_s,
        )
        self._poll_interval_s = self._positive_finite_timeout(
            "_poll_interval_s",
            _poll_interval_s,
        )
        self._m2n = _m2n_module or self._import_backend("nccl.m2n")
        self._nccl = _nccl_module or self._import_backend("nccl.core")
        self._torch = _torch_module or self._import_backend("torch")
        self._enforce_singleton = _enforce_singleton

        self._state = _RuntimeState.OPEN
        self._state_cv = threading.Condition()
        self._dispatcher_lock = threading.Lock()
        self._active_operations = 0
        self._operation_local = threading.local()
        self._close_abandoned = False
        self._restart_required = False
        self._pp_groups: dict[_PPGroupKey, _M2nPPGroup] = {}
        self._topology_frozen = False
        self._handle: Any | None = None
        self._handle_quarantined = False
        self._quarantined_batches: tuple[_M2nPPGroupBatch, ...] = ()
        self._fail_stop_reason: str | None = None
        self._abort_lock = threading.Lock()
        self._abort_thread: threading.Thread | None = None
        self._abort_done = threading.Event()

        if _enforce_singleton:
            with self._singleton_lock:
                live = type(self)._live_runtime
                if live is not None and live._state is not _RuntimeState.CLOSED:
                    raise RuntimeError(
                        "only one _M2nRuntime may be active in a process"
                    )
                type(self)._live_runtime = self

        try:
            self._validate_nccl_version()
            self._validate_nccl_api()
            self._validate_m2n_api()
            self._torch.cuda.set_device(self._device_id)
            config = (
                self._m2n.Config()
                if max_cta is None
                else self._m2n.Config(max_cta=max_cta)
            )
            self._handle = self._m2n.init(config)
            if not all(
                callable(getattr(self._handle, name, None))
                for name in ("reshard", "destroy")
            ):
                destroy = getattr(self._handle, "destroy", None)
                if callable(destroy):
                    destroy()
                self._handle = None
                raise RuntimeError(
                    "NCCL M2N package lacks current Handle.reshard()/destroy() API"
                )
        except BaseException:
            self._clear_singleton()
            raise

    @staticmethod
    def _import_backend(module: str) -> Any:
        try:
            return importlib.import_module(module)
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "NCCL M2N requires the current nccl-extensions package and NCCL4Py"
            ) from exc

    @staticmethod
    def _positive_finite_timeout(name: str, value: float) -> float:
        timeout = float(value)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")
        return timeout

    def _validate_nccl_version(self) -> None:
        get_version = getattr(self._nccl, "get_version", None)
        if not callable(get_version):
            raise TypeError("NCCL4Py does not expose nccl.core.get_version()")
        version_info = get_version()
        if isinstance(version_info, Integral) and not isinstance(version_info, bool):
            libnccl_version = version_info
        else:
            libnccl = getattr(version_info, "libnccl", None)
            if libnccl is None:
                raise RuntimeError(
                    "NCCL M2N requires NCCL >= 2.30.5, but NCCL4Py could not "
                    f"identify a loaded libnccl: {version_info!r}"
                )
            libnccl_version = getattr(libnccl, "version", None)
            if libnccl_version is None:
                raise TypeError(
                    "NCCL4Py VersionInfo.libnccl does not expose .version"
                )
        if _version_tuple(libnccl_version) < (2, 30, 5):
            raise RuntimeError(
                "NCCL M2N requires NCCL >= 2.30.5, found "
                f"{libnccl_version}"
            )

    def _validate_nccl_api(self) -> None:
        config_type = getattr(self._nccl, "NCCLConfig", None)
        communicator_type = getattr(self._nccl, "Communicator", None)
        required_methods = ("initialize", "get_async_error", "abort")
        missing = [
            f"Communicator.{name}"
            for name in required_methods
            if not callable(getattr(communicator_type, name, None))
        ]
        if not callable(config_type):
            missing.insert(0, "NCCLConfig")
        if missing:
            raise RuntimeError(
                "NCCL M2N fault-tolerant mode requires current NCCL4Py APIs: "
                + ", ".join(missing)
            )
        try:
            config_type(blocking=False)
        except BaseException as exc:
            raise RuntimeError(
                "NCCL4Py cannot construct NCCLConfig(blocking=False)"
            ) from exc

    def _new_nccl_config(self) -> Any:
        return self._nccl.NCCLConfig(blocking=False)

    def _validate_m2n_api(self) -> None:
        required = (
            "Config",
            "DistTensor",
            "Mesh",
            "Replicate",
            "Shard",
            "group",
            "init",
        )
        missing = [name for name in required if not hasattr(self._m2n, name)]
        if missing:
            raise RuntimeError(
                "NCCL M2N package lacks current Python API: " + ", ".join(missing)
            )

    @contextmanager
    def _active_operation(
        self,
        *,
        allow_poisoned: bool = False,
    ) -> Iterator[None]:
        """Keep runtime resources alive for one reentrant top-level operation."""
        depth = getattr(self._operation_local, "depth", 0)
        if depth:
            with self._state_cv:
                self._require_admitted_operation_locked(
                    allow_poisoned=getattr(
                        self._operation_local,
                        "allow_poisoned",
                        False,
                    )
                )
            self._operation_local.depth = depth + 1
            try:
                yield
            finally:
                self._operation_local.depth = depth
            return

        with self._state_cv:
            if (
                allow_poisoned
                and self._state is _RuntimeState.POISONED
                and not self._close_abandoned
            ):
                pass
            else:
                self._require_open_locked()
            self._active_operations += 1
        self._operation_local.depth = 1
        self._operation_local.allow_poisoned = allow_poisoned
        try:
            yield
        finally:
            del self._operation_local.depth
            del self._operation_local.allow_poisoned
            with self._state_cv:
                self._active_operations -= 1
                self._state_cv.notify_all()

    def _require_admitted_operation(self) -> None:
        with self._state_cv:
            self._require_admitted_operation_locked()

    def _require_admitted_operation_locked(
        self,
        *,
        allow_poisoned: bool = False,
    ) -> None:
        if self._state in (_RuntimeState.OPEN, _RuntimeState.CLOSING):
            return
        if (
            allow_poisoned
            and self._state is _RuntimeState.POISONED
            and not self._close_abandoned
        ):
            return
        raise RuntimeError(f"M2N runtime is {self._state.name.lower()}")

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def handle(self) -> Any:
        if self._handle is None:
            raise RuntimeError("M2N runtime handle is not available")
        return self._handle

    @property
    def m2n(self) -> Any:
        """Private backend access used only to construct official descriptors."""
        return self._m2n

    @property
    def pp_groups(self) -> tuple[_M2nPPGroup, ...]:
        return tuple(sorted(self._pp_groups.values(), key=lambda group: group.key))

    def new_unique_id_bytes(self) -> bytes:
        with self._active_operation():
            unique_id = self._nccl.get_unique_id()
            payload = getattr(unique_id, "as_bytes", None)
            if payload is None:
                raise TypeError("NCCL4Py UniqueId does not expose .as_bytes")
            return bytes(payload)

    def create_pp_groups(
        self,
        specs: Sequence[_M2nPPGroupSpec],
    ) -> tuple[_M2nPPGroup, ...]:
        """Collectively create every local PP group in canonical order.

        Creation is intentionally batch-only. Thread arrival or unordered map
        iteration must never choose parent-communicator first-use order.
        """
        with self._active_operation():
            ordered_specs = tuple(sorted(specs, key=lambda spec: spec.key))
            if not ordered_specs:
                return ()
            self._validate_pp_group_specs(ordered_specs)

            with self._dispatcher_lock:
                self._require_admitted_operation()
                with self._state_cv:
                    if self._topology_frozen:
                        raise RuntimeError("M2N PP-group topology is frozen")
                    if self._pp_groups:
                        raise RuntimeError(
                            "all local M2N PP groups must be created in one "
                            "canonical batch"
                        )

                created: list[_M2nPPGroup] = []
                try:
                    self._torch.cuda.set_device(self._device_id)
                    for spec in ordered_specs:
                        unique_id = self._nccl.UniqueId.from_bytes(spec.unique_id)
                        stream = self._torch.cuda.Stream(device=self._device_id)
                        try:
                            communicator = self._nccl.Communicator()
                        except BaseException:
                            close = getattr(stream, "close", None)
                            if callable(close):
                                close()
                            raise
                        pp_group = _M2nPPGroup(
                            group_id=spec.group_id,
                            key=spec.key,
                            communicator=communicator,
                            source_size=spec.source_size,
                            destination_size=spec.destination_size,
                            comm_rank=spec.comm_rank,
                            device_id=spec.device_id,
                            stream=stream,
                        )
                        created.append(pp_group)
                        communicator.initialize(
                            spec.nranks,
                            spec.comm_rank,
                            unique_id,
                            self._new_nccl_config(),
                        )
                    self._poll_communicators_ready(
                        created,
                        operation="PP-group initialization",
                        deadline=time.monotonic() + self._comm_init_timeout_s,
                    )
                except BaseException as exc:
                    with self._state_cv:
                        close_abandoned = self._close_abandoned
                        if close_abandoned:
                            self._pp_groups.update(
                                {group.key: group for group in created}
                            )
                            for group in created:
                                group.state = _PPGroupState.POISONED
                    if not close_abandoned:
                        self._enter_fail_stop(created, (), exc)
                        self._start_abort_worker()
                    raise

                with self._state_cv:
                    self._pp_groups = {
                        pp_group.key: pp_group for pp_group in created
                    }
                    if self._state is _RuntimeState.POISONED:
                        for pp_group in created:
                            pp_group.state = _PPGroupState.POISONED
                    self._require_admitted_operation_locked()
            return tuple(created)

    def _validate_pp_group_specs(
        self,
        specs: Sequence[_M2nPPGroupSpec],
    ) -> None:
        ids: set[str] = set()
        keys: set[_PPGroupKey] = set()
        for spec in specs:
            if spec.device_id != self._device_id:
                raise ValueError(
                    f"PP group device {spec.device_id} does not match runtime "
                    f"device {self._device_id}"
                )
            if not spec.group_id:
                raise ValueError("PP group_id must not be empty")
            if spec.group_id in ids:
                raise ValueError(f"duplicate M2N PP group ID {spec.group_id!r}")
            ids.add(spec.group_id)
            if len(spec.key) != 2 or any(int(stage) < 0 for stage in spec.key):
                raise ValueError(
                    "PP group key must contain non-negative trainer/generator "
                    f"stage IDs, got {spec.key}"
                )
            if spec.key in keys:
                raise ValueError(f"duplicate M2N PP group key {spec.key}")
            keys.add(spec.key)
            if spec.source_size <= 0 or spec.destination_size <= 0:
                raise ValueError(
                    "M2N PP group source/destination sizes must be positive, got "
                    f"{spec.source_size}/{spec.destination_size}"
                )
            if not 0 <= spec.comm_rank < spec.nranks:
                raise ValueError(
                    f"invalid M2N PP group communicator rank "
                    f"{spec.comm_rank}/{spec.nranks}"
                )

    def freeze_pp_groups(self) -> tuple[_M2nPPGroup, ...]:
        with self._active_operation(), self._dispatcher_lock:
            with self._state_cv:
                self._require_admitted_operation_locked()
                if not self._pp_groups:
                    raise RuntimeError("cannot freeze an empty M2N PP-group topology")
                self._topology_frozen = True
                return tuple(
                    sorted(self._pp_groups.values(), key=lambda group: group.key)
                )

    def submit_model_update(
        self,
        batches: Sequence[_M2nPPGroupBatch],
    ) -> dict[_PPGroupKey, int]:
        """Submit one complete local model update in canonical PP-group order."""
        with self._active_operation():
            return self._submit_model_update(batches)

    def _submit_model_update(
        self,
        batches: Sequence[_M2nPPGroupBatch],
    ) -> dict[_PPGroupKey, int]:
        ordered = tuple(sorted(batches, key=lambda batch: batch.pp_group.key))
        if not ordered:
            return {}

        with self._state_cv:
            self._require_admitted_operation_locked()
            if not self._topology_frozen:
                raise RuntimeError(
                    "M2N PP-group topology must be frozen before updates"
                )
            expected = self.pp_groups
            provided = tuple(batch.pp_group for batch in ordered)
            if len(set(provided)) != len(provided):
                raise ValueError("an M2N update may contain each PP group only once")
            if provided != expected:
                raise RuntimeError(
                    "M2N model update must contain every local PP group; "
                    f"expected={[group.key for group in expected]}, "
                    f"provided={[group.key for group in provided]}"
                )
            for pp_group in provided:
                if pp_group.state is not _PPGroupState.OPEN:
                    raise RuntimeError(
                        f"M2N PP group {pp_group.group_id!r} is "
                        f"{pp_group.state.name.lower()}"
                    )

        with self._dispatcher_lock:
            submission_started = False
            completion_wait_started = False
            try:
                # Another queued update may have poisoned runtime while this
                # caller waited for dispatcher ownership.
                self._require_admitted_operation()
                self._wait_for_source_readiness(ordered)
                if any(batch.calls for batch in ordered):
                    submission_started = True
                    with self._m2n.group():
                        for batch in ordered:
                            for call in batch.calls:
                                self.handle.reshard(
                                    batch.pp_group.communicator,
                                    call.src,
                                    call.dst,
                                    stream=batch.pp_group.stream,
                                )

                # This deadline bounds MX-owned stream completion after the
                # native grouped submission returns. Current official M2N may
                # still block inside group_end(); Python cannot interrupt it.
                deadline = time.monotonic() + self._transfer_timeout_s
                completion_wait_started = True
                self._poll_pp_groups_completion(
                    provided,
                    operation="model-version staging",
                    deadline=deadline,
                )

                for batch in ordered:
                    if batch.commit is not None:
                        batch.commit()
                committed = tuple(
                    batch.pp_group for batch in ordered if batch.commit is not None
                )
                if committed:
                    self._poll_pp_groups_completion(
                        committed,
                        operation="model-version commit",
                        deadline=deadline,
                    )
                self._raise_if_restart_required("model-version completion")
                for batch in ordered:
                    if batch.on_complete is not None:
                        batch.on_complete()
            except BaseException as exc:
                fail_stop = submission_started or completion_wait_started
                if fail_stop:
                    self._enter_fail_stop(provided, ordered, exc)
                for batch in ordered:
                    if batch.on_failure is not None:
                        try:
                            batch.on_failure(exc)
                        except Exception:
                            logger.warning(
                                "M2N failure callback raised while preserving "
                                "original transfer error",
                                exc_info=True,
                            )
                if fail_stop:
                    self._start_abort_worker()
                raise

        return {batch.pp_group.key: batch.total_bytes for batch in ordered}

    def _wait_for_source_readiness(
        self,
        batches: Sequence[_M2nPPGroupBatch],
    ) -> None:
        source_batches = [
            batch for batch in batches if batch.calls and batch.pp_group.is_source
        ]
        if not source_batches:
            return

        self._torch.cuda.set_device(self._device_id)
        producer_stream = self._torch.cuda.current_stream(self._device_id)
        ready = self._torch.cuda.Event()
        ready.record(producer_stream)
        for batch in source_batches:
            batch.pp_group.stream.wait_event(ready)

    @contextmanager
    def stream_context(self, pp_group: _M2nPPGroup) -> Iterator[None]:
        with self._active_operation():
            if pp_group.stream is None:
                raise RuntimeError(
                    f"M2N PP group {pp_group.group_id!r} stream has been released"
                )
            with self._torch.cuda.stream(pp_group.stream):
                yield

    def wait_for_pp_groups(
        self,
        pp_groups: Sequence[_M2nPPGroup],
        *,
        operation: str,
    ) -> None:
        """Bound one healthy drain; failure makes the process epoch fail-stop."""
        with self._active_operation():
            self._raise_if_restart_required(operation)
            while not self._dispatcher_lock.acquire(
                timeout=self._poll_interval_s,
            ):
                self._raise_if_restart_required(operation)
            try:
                self._raise_if_restart_required(operation)
                ordered = tuple(sorted(pp_groups, key=lambda group: group.key))
                try:
                    self._poll_pp_groups_completion(
                        ordered,
                        operation=operation,
                        deadline=time.monotonic() + self._transfer_timeout_s,
                    )
                except BaseException as exc:
                    self._enter_fail_stop(ordered, (), exc)
                    self._start_abort_worker()
                    raise
            finally:
                self._dispatcher_lock.release()

    def synchronize_pp_group(self, pp_group: _M2nPPGroup) -> None:
        """Compatibility wrapper using the bounded completion path."""
        self.wait_for_pp_groups(
            (pp_group,),
            operation=f"PP-group {pp_group.group_id!r} synchronization",
        )

    def _poll_communicators_ready(
        self,
        pp_groups: Sequence[_M2nPPGroup],
        *,
        operation: str,
        deadline: float,
    ) -> None:
        pending = list(sorted(pp_groups, key=lambda group: group.key))
        while pending:
            self._raise_if_restart_required(operation)
            for pp_group in tuple(pending):
                state = self._communicator_state(pp_group, operation)
                if state == _NCCL_SUCCESS:
                    pending.remove(pp_group)
            if pending:
                self._poll_pause(pending, operation=operation, deadline=deadline)
        self._raise_if_restart_required(operation)

    def _poll_pp_groups_completion(
        self,
        pp_groups: Sequence[_M2nPPGroup],
        *,
        operation: str,
        deadline: float,
    ) -> None:
        self._torch.cuda.set_device(self._device_id)
        pending = list(sorted(pp_groups, key=lambda group: group.key))
        while pending:
            self._raise_if_restart_required(operation)
            for pp_group in tuple(pending):
                state = self._communicator_state(pp_group, operation)
                if state == _NCCL_IN_PROGRESS:
                    continue
                stream = pp_group.stream
                if stream is None:
                    raise RuntimeError(
                        f"M2N PP group {pp_group.group_id!r} stream was released "
                        f"during {operation}"
                    )
                try:
                    ready = bool(stream.query())
                except BaseException as exc:
                    raise RuntimeError(
                        f"CUDA stream query failed during {operation} on PP group "
                        f"{pp_group.group_id!r}"
                    ) from exc
                if ready:
                    pending.remove(pp_group)
            if pending:
                self._poll_pause(pending, operation=operation, deadline=deadline)
        self._raise_if_restart_required(operation)

    def _communicator_state(
        self,
        pp_group: _M2nPPGroup,
        operation: str,
    ) -> int:
        try:
            state = pp_group.communicator.get_async_error()
            value = int(state)
        except BaseException as exc:
            raise RuntimeError(
                f"NCCL status query failed during {operation} on PP group "
                f"{pp_group.group_id!r}"
            ) from exc
        if value not in (_NCCL_SUCCESS, _NCCL_IN_PROGRESS):
            raise RuntimeError(
                f"NCCL async error during {operation} on PP group "
                f"{pp_group.group_id!r}: status={value}"
            )
        return value

    def _poll_pause(
        self,
        pending: Sequence[_M2nPPGroup],
        *,
        operation: str,
        deadline: float,
    ) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"timed out during {operation}; pending PP groups="
                f"{[group.key for group in pending]}"
            )
        time.sleep(min(self._poll_interval_s, remaining))

    def _raise_if_restart_required(self, operation: str) -> None:
        with self._state_cv:
            if self._restart_required:
                raise RuntimeError(
                    f"M2N runtime entered fail-stop during {operation}; "
                    "resources were retained and process restart is required"
                )

    def _enter_fail_stop(
        self,
        pp_groups: Sequence[_M2nPPGroup],
        batches: Sequence[_M2nPPGroupBatch],
        cause: BaseException,
    ) -> None:
        ordered_groups = tuple(sorted(pp_groups, key=lambda group: group.key))
        with self._state_cv:
            if self._state is _RuntimeState.CLOSED:
                return
            for pp_group in ordered_groups:
                self._pp_groups.setdefault(pp_group.key, pp_group)
            self._state = _RuntimeState.POISONED
            self._restart_required = True
            self._handle_quarantined = self._handle is not None
            self._quarantined_batches += tuple(batches)
            self._fail_stop_reason = f"{type(cause).__name__}: {cause}"
            for pp_group in self._pp_groups.values():
                if pp_group.state is not _PPGroupState.CLOSED:
                    pp_group.state = _PPGroupState.POISONED
            self._state_cv.notify_all()
        logger.error(
            "NCCL M2N runtime entered fail-stop; process restart required: %s",
            self._fail_stop_reason,
        )

    def _start_abort_worker(self) -> None:
        with self._abort_lock:
            if self._abort_thread is not None or self._abort_done.is_set():
                return
            pp_groups = tuple(
                group
                for group in self.pp_groups
                if group.state is not _PPGroupState.CLOSED
                and not group.abort_attempted
            )
            if not pp_groups:
                self._abort_done.set()
                return
            thread = threading.Thread(
                target=self._abort_pp_groups,
                args=(pp_groups,),
                name="mx-m2n-abort",
                daemon=True,
            )
            self._abort_thread = thread
            try:
                thread.start()
            except BaseException:
                self._abort_done.set()
                logger.exception("failed to start NCCL M2N abort worker")

    def _abort_pp_groups(
        self,
        pp_groups: Sequence[_M2nPPGroup],
    ) -> None:
        try:
            try:
                self._torch.cuda.set_device(self._device_id)
            except BaseException:
                logger.warning(
                    "failed to select CUDA device before NCCL communicator abort",
                    exc_info=True,
                )
            for pp_group in pp_groups:
                pp_group.abort_attempted = True
                try:
                    pp_group.communicator.abort()
                except BaseException:
                    logger.warning(
                        "NCCL communicator abort failed on PP group %r",
                        pp_group.group_id,
                        exc_info=True,
                    )
                else:
                    pp_group.aborted = True
        finally:
            self._abort_done.set()

    def close(self) -> None:
        """Drain groups, finalize M2N, then destroy parent comms canonically."""
        if getattr(self._operation_local, "depth", 0):
            raise RuntimeError(
                "cannot close M2N runtime from an active runtime operation"
            )

        with self._state_cv:
            if self._state is _RuntimeState.CLOSED:
                return
            if self._restart_required or self._close_abandoned:
                raise RuntimeError(
                    "M2N runtime is fail-stop; resources were retained and "
                    "process restart is required"
                )
            if self._state is _RuntimeState.CLOSING:
                raise RuntimeError("M2N runtime is already closing")
            self._state = _RuntimeState.CLOSING
            self._state_cv.notify_all()
            deadline = time.monotonic() + self._finalize_timeout_s
            while self._active_operations:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._state = _RuntimeState.POISONED
                    self._close_abandoned = True
                    self._restart_required = True
                    self._handle_quarantined = self._handle is not None
                    for pp_group in self._pp_groups.values():
                        if pp_group.state is not _PPGroupState.CLOSED:
                            pp_group.state = _PPGroupState.POISONED
                    self._state_cv.notify_all()
                    raise TimeoutError(
                        "timed out waiting for active M2N operations; resources "
                        "were retained and process restart is required"
                    )
                self._state_cv.wait(timeout=remaining)
            if self._restart_required:
                raise RuntimeError(
                    "M2N runtime entered fail-stop while shutdown waited; "
                    "resources were retained and process restart is required"
                )

        try:
            with self._dispatcher_lock:
                pp_groups = sorted(
                    self._pp_groups.values(), key=lambda group: group.key
                )
                try:
                    self._poll_pp_groups_completion(
                        pp_groups,
                        operation="runtime shutdown drain",
                        deadline=deadline,
                    )
                except BaseException as exc:
                    self._enter_fail_stop(pp_groups, (), exc)
                    self._start_abort_worker()
                    raise

                # M2N cache cleanup runs while every parent communicator remains
                # valid.
                if self._handle is not None and not self._handle_quarantined:
                    self._handle.destroy()
                    self._handle = None

            # nccl-rl issue #76: each process destroys a sorted subsequence of
            # one global PP-pair order, preventing communicator wait cycles.
            for pp_group in pp_groups:
                self._release_pp_group_stream(pp_group)
                pp_group.communicator.finalize()
                self._wait_for_finalize(
                    pp_group,
                    deadline=time.monotonic() + self._finalize_timeout_s,
                )
                pp_group.communicator.destroy()
                pp_group.state = _PPGroupState.CLOSED
        except BaseException:
            with self._state_cv:
                self._state = _RuntimeState.POISONED
            raise

        self._pp_groups.clear()
        with self._state_cv:
            self._state = _RuntimeState.CLOSED
        self._clear_singleton()

    @staticmethod
    def _release_pp_group_stream(pp_group: _M2nPPGroup) -> None:
        """Release MX ownership of a PyTorch stream before its communicator."""
        stream = pp_group.stream
        pp_group.stream = None
        if stream is not None:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

    def _wait_for_finalize(
        self,
        pp_group: _M2nPPGroup,
        *,
        deadline: float,
    ) -> None:
        while True:
            state = pp_group.communicator.get_async_error()
            value = int(state)
            if value == _NCCL_SUCCESS:
                return
            if value != _NCCL_IN_PROGRESS:
                raise RuntimeError(
                    "NCCL communicator finalize failed on PP group "
                    f"{pp_group.group_id!r}: {state}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "timed out finalizing NCCL communicator PP group "
                    f"{pp_group.group_id!r}"
                )
            time.sleep(0.001)

    def _require_open(self) -> None:
        with self._state_cv:
            self._require_open_locked()

    def _require_open_locked(self) -> None:
        if self._restart_required:
            raise RuntimeError(
                "M2N runtime is fail-stop; resources were retained and "
                "process restart is required"
            )
        if self._state is not _RuntimeState.OPEN:
            raise RuntimeError(f"M2N runtime is {self._state.name.lower()}")

    def _clear_singleton(self) -> None:
        if not self._enforce_singleton:
            return
        with self._singleton_lock:
            if type(self)._live_runtime is self:
                type(self)._live_runtime = None


__all__ = [
    "_M2nCall",
    "_M2nPPGroup",
    "_M2nPPGroupBatch",
    "_M2nPPGroupSpec",
    "_M2nRuntime",
]
