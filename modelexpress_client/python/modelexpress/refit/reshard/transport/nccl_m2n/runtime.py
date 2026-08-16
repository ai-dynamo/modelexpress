# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal process-level owner and deterministic dispatcher for NCCL M2N.

Current M2N keeps process-global caches. MX therefore creates one explicit
handle per process and retains every parent communicator until handle
finalization. Each PP-pair communicator is a lane keyed by the globally stable
``(trainer_stage, generator_stage)`` pair used by nccl-rl issue #76.

Preparation may be concurrent; M2N submission is single-dispatcher and
canonically ordered. A model-update batch submits every locally active lane in
ascending key order. Each lane has its own CUDA stream, so GPU work may overlap
after sequential host enqueue.
"""

from __future__ import annotations

import importlib
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Sequence

from .mesh import Mesh as PlannerMesh
from .mesh import REPLICATE

_LaneKey = tuple[int, int]


class _RuntimeState(Enum):
    OPEN = auto()
    CLOSING = auto()
    POISONED = auto()
    CLOSED = auto()


class _LaneState(Enum):
    OPEN = auto()
    POISONED = auto()
    CLOSED = auto()


@dataclass(frozen=True)
class _M2nLaneSpec:
    """Bootstrap material for one PP-pair communicator lane."""

    lane_id: str
    key: _LaneKey
    unique_id: bytes
    nranks: int
    comm_rank: int
    device_id: int


@dataclass(eq=False)
class _M2nLane:
    lane_id: str
    key: _LaneKey
    communicator: Any
    nranks: int
    comm_rank: int
    device_id: int
    stream: Any
    owns_stream: bool
    state: _LaneState = _LaneState.OPEN


@dataclass(frozen=True)
class _M2nCall:
    """One parameter collective already prepared for host submission."""

    src: Any | None
    dst: Any | None
    src_mesh: PlannerMesh
    dst_mesh: PlannerMesh
    src_local_shape: tuple[int, ...]
    dst_local_shape: tuple[int, ...]
    dtype: Any


@dataclass(frozen=True)
class _M2nLaneBatch:
    """All ordered parameter calls for one lane in one model update."""

    lane: _M2nLane
    calls: tuple[_M2nCall, ...]
    total_bytes: int
    commit: Callable[[], None] | None = None
    on_complete: Callable[[], None] | None = None
    on_failure: Callable[[BaseException], None] | None = None


def _version_tuple(version: object) -> tuple[int, int, int]:
    numbers = [int(value) for value in re.findall(r"\d+", str(version))[:3]]
    return tuple((numbers + [0, 0, 0])[:3])


class _M2nRuntime:
    """Own one M2N handle, all lanes, and the only M2N submission path."""

    _singleton_lock = threading.Lock()
    _live_runtime: _M2nRuntime | None = None

    def __init__(
        self,
        device_id: int,
        *,
        max_cta: int | None = None,
        finalize_timeout_s: float = 300.0,
        _m2n_module: Any | None = None,
        _nccl_module: Any | None = None,
        _torch_module: Any | None = None,
        _enforce_singleton: bool = True,
    ) -> None:
        self._device_id = int(device_id)
        self._finalize_timeout_s = float(finalize_timeout_s)
        self._m2n = _m2n_module or self._import_backend("nccl.m2n")
        self._nccl = _nccl_module or self._import_backend("nccl.core")
        self._torch = _torch_module or self._import_backend("torch")
        self._enforce_singleton = _enforce_singleton

        self._state = _RuntimeState.OPEN
        self._state_cv = threading.Condition()
        self._dispatcher_lock = threading.Lock()
        self._active_batches = 0
        self._lanes: dict[str, _M2nLane] = {}
        self._lane_keys: set[_LaneKey] = set()
        self._handle: Any | None = None

        if _enforce_singleton:
            with self._singleton_lock:
                live = type(self)._live_runtime
                if live is not None and live._state is not _RuntimeState.CLOSED:
                    raise RuntimeError("only one _M2nRuntime may be active in a process")
                type(self)._live_runtime = self

        try:
            self._validate_nccl_version()
            self._torch.cuda.set_device(self._device_id)
            config = (
                self._m2n.Config()
                if max_cta is None
                else self._m2n.Config(max_cta=max_cta)
            )
            self._handle = self._m2n.init(config)
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

    def _validate_nccl_version(self) -> None:
        get_version = getattr(self._nccl, "get_version", None)
        if not callable(get_version):
            raise RuntimeError("NCCL4Py does not expose nccl.core.get_version()")
        version = get_version()
        if _version_tuple(version) < (2, 30, 5):
            raise RuntimeError(f"NCCL M2N requires NCCL >= 2.30.5, found {version}")

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def handle(self) -> Any:
        if self._handle is None:
            raise RuntimeError("M2N runtime handle is not available")
        return self._handle

    @property
    def lanes(self) -> tuple[_M2nLane, ...]:
        return tuple(sorted(self._lanes.values(), key=lambda lane: lane.key))

    def new_unique_id_bytes(self) -> bytes:
        self._require_open()
        return bytes(self._nccl.get_unique_id())

    def create_lane(
        self,
        spec: _M2nLaneSpec,
        *,
        stream: Any | None = None,
    ) -> _M2nLane:
        """Collectively create and register one NCCL communicator lane."""
        if spec.device_id != self._device_id:
            raise ValueError(
                f"lane device {spec.device_id} does not match runtime device {self._device_id}"
            )
        unique_id = self._nccl.UniqueId.from_bytes(spec.unique_id)
        self._torch.cuda.set_device(self._device_id)
        communicator = self._nccl.Communicator.init(
            spec.nranks,
            spec.comm_rank,
            unique_id,
        )
        return self.register_lane(
            lane_id=spec.lane_id,
            key=spec.key,
            communicator=communicator,
            nranks=spec.nranks,
            comm_rank=spec.comm_rank,
            stream=stream,
        )

    def register_lane(
        self,
        *,
        lane_id: str,
        key: _LaneKey,
        communicator: Any,
        nranks: int,
        comm_rank: int,
        stream: Any | None = None,
    ) -> _M2nLane:
        """Register an already-created communicator under runtime ownership."""
        if len(key) != 2 or any(int(stage) < 0 for stage in key):
            raise ValueError(f"lane key must contain two non-negative stage IDs, got {key}")
        key = (int(key[0]), int(key[1]))
        with self._state_cv:
            self._require_open_locked()
            if not lane_id:
                raise ValueError("lane_id must not be empty")
            if lane_id in self._lanes:
                raise ValueError(f"M2N lane {lane_id!r} is already registered")
            if key in self._lane_keys:
                raise ValueError(f"M2N lane key {key} is already registered")
            if nranks <= 0 or not 0 <= comm_rank < nranks:
                raise ValueError(f"invalid communicator rank {comm_rank}/{nranks}")

            owns_stream = stream is None
            if stream is None:
                self._torch.cuda.set_device(self._device_id)
                stream = self._torch.cuda.Stream(device=self._device_id)

            lane = _M2nLane(
                lane_id=lane_id,
                key=key,
                communicator=communicator,
                nranks=int(nranks),
                comm_rank=int(comm_rank),
                device_id=self._device_id,
                stream=stream,
                owns_stream=owns_stream,
            )
            self._lanes[lane_id] = lane
            self._lane_keys.add(key)
            return lane

    def dispatch_batch(self, batches: Sequence[_M2nLaneBatch]) -> dict[str, int]:
        """Submit one complete model update in canonical lane order.

        A multi-lane process must provide every registered lane in one batch.
        Independent per-lane caller threads are rejected because mutex arrival
        order is not a distributed ordering contract.
        """
        ordered = sorted(batches, key=lambda batch: batch.lane.key)
        if not ordered:
            return {}

        with self._state_cv:
            self._require_open_locked()
            expected = self.lanes
            provided = tuple(batch.lane for batch in ordered)
            if len(set(provided)) != len(provided):
                raise ValueError("an M2N batch may contain each lane only once")
            if provided != expected:
                raise RuntimeError(
                    "M2N model-update batch must contain every locally registered lane "
                    f"in canonical order; expected={[lane.key for lane in expected]}, "
                    f"provided={[lane.key for lane in provided]}"
                )
            for lane in provided:
                if lane.state is not _LaneState.OPEN:
                    raise RuntimeError(f"M2N lane {lane.lane_id!r} is {lane.state.name.lower()}")
            self._active_batches += 1

        # One dispatcher owns the full host-submission sequence. Holding it for
        # the batch also prevents racing model versions from interleaving.
        with self._dispatcher_lock:
            started: set[_M2nLane] = set()
            try:
                for batch in ordered:
                    for call in batch.calls:
                        started.add(batch.lane)
                        self._submit_call(batch.lane, call)

                # All lane streams now have work in flight and may overlap.
                for batch in ordered:
                    self.synchronize_lane(batch.lane)
                    self.check_async_error(batch.lane, "model-version staging")

                # Enqueue destination commits on every lane before waiting.
                for batch in ordered:
                    if batch.commit is not None:
                        batch.commit()
                for batch in ordered:
                    if batch.commit is not None:
                        self.synchronize_lane(batch.lane)
                        self.check_async_error(batch.lane, "model-version commit")
                for batch in ordered:
                    if batch.on_complete is not None:
                        batch.on_complete()
            except BaseException as exc:
                for lane in started:
                    self.poison_lane(lane)
                for batch in ordered:
                    if batch.on_failure is not None:
                        try:
                            batch.on_failure(exc)
                        except BaseException:
                            pass
                for batch in ordered:
                    try:
                        self.synchronize_lane(batch.lane)
                    except BaseException:
                        self.poison_lane(batch.lane)
                raise
            finally:
                with self._state_cv:
                    self._active_batches -= 1
                    self._state_cv.notify_all()

        return {batch.lane.lane_id: batch.total_bytes for batch in ordered}

    def _submit_call(self, lane: _M2nLane, call: _M2nCall) -> None:
        src_mesh, src_placements = self._convert_layout(
            call.src_mesh,
            len(call.src_local_shape),
        )
        dst_mesh, dst_placements = self._convert_layout(
            call.dst_mesh,
            len(call.dst_local_shape),
        )
        self._torch.cuda.set_device(self._device_id)
        self._m2n.reshard(
            src=call.src,
            dst=call.dst,
            comm=lane.communicator,
            stream=lane.stream,
            src_mesh=src_mesh,
            src_placements=src_placements,
            src_local_shape=call.src_local_shape,
            src_dtype=call.dtype,
            dst_mesh=dst_mesh,
            dst_placements=dst_placements,
            dst_local_shape=call.dst_local_shape,
            dst_dtype=call.dtype,
            handle=self.handle,
        )

    def _convert_layout(
        self,
        mesh: PlannerMesh,
        tensor_ndim: int,
    ) -> tuple[Any, tuple[Any, Any]]:
        m2n_mesh = self._m2n.Mesh(mesh.dims, start_rank=mesh.start_rank)
        if all(placement == REPLICATE for placement in mesh.placement):
            if tensor_ndim < 1 or mesh.dims[0] != 1:
                raise ValueError(
                    "fully replicated M2N layout requires a size-one mesh axis"
                )
            return m2n_mesh, (self._m2n.Shard(0), self._m2n.Replicate())

        placements = tuple(
            self._m2n.Replicate()
            if placement == REPLICATE
            else self._m2n.Shard(placement)
            for placement in mesh.placement
        )
        return m2n_mesh, placements

    def stream_context(self, lane: _M2nLane) -> Any:
        if lane.stream is None:
            raise RuntimeError(f"M2N lane {lane.lane_id!r} stream has been released")
        return self._torch.cuda.stream(lane.stream)

    def synchronize_lane(self, lane: _M2nLane) -> None:
        self._torch.cuda.set_device(self._device_id)
        if lane.stream is None:
            raise RuntimeError(f"M2N lane {lane.lane_id!r} stream has been released")
        lane.stream.synchronize()

    def check_async_error(self, lane: _M2nLane, operation: str) -> None:
        state = lane.communicator.get_async_error()
        if int(state) != 0:
            raise RuntimeError(
                f"NCCL async error after {operation} on lane {lane.lane_id!r}: {state}"
            )

    def poison_lane(self, lane: _M2nLane) -> None:
        with self._state_cv:
            if self._lanes.get(lane.lane_id) is not lane:
                raise RuntimeError(f"M2N lane {lane.lane_id!r} is not registered")
            if lane.state is not _LaneState.CLOSED:
                lane.state = _LaneState.POISONED

    def close(self) -> None:
        """Drain lanes, finalize M2N, then finalize/destroy comms canonically."""
        with self._state_cv:
            if self._state is _RuntimeState.CLOSED:
                return
            if self._state is not _RuntimeState.OPEN:
                raise RuntimeError(f"cannot close M2N runtime in state {self._state.name.lower()}")
            self._state = _RuntimeState.CLOSING
            deadline = time.monotonic() + self._finalize_timeout_s
            while self._active_batches:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._state = _RuntimeState.POISONED
                    raise TimeoutError("timed out waiting for active M2N batches")
                self._state_cv.wait(timeout=remaining)

        lanes = list(self.lanes)
        try:
            for lane in lanes:
                self.synchronize_lane(lane)

            # M2N cache cleanup runs while every parent communicator is valid.
            with self._dispatcher_lock:
                if self._handle is not None:
                    self._handle.destroy()
                    self._handle = None

            # Issue #76 ordering: each process tears down a sorted subsequence
            # of the same global PP-pair order. Processes owning multiple lanes
            # therefore cannot form a communicator-destruction wait cycle.
            for lane in lanes:
                self._release_lane_stream(lane)
                lane.communicator.finalize()
                self._wait_for_finalize(
                    lane,
                    deadline=time.monotonic() + self._finalize_timeout_s,
                )
                lane.communicator.destroy()
                lane.state = _LaneState.CLOSED
        except BaseException:
            with self._state_cv:
                self._state = _RuntimeState.POISONED
            raise

        self._lanes.clear()
        self._lane_keys.clear()
        with self._state_cv:
            self._state = _RuntimeState.CLOSED
        self._clear_singleton()

    @staticmethod
    def _release_lane_stream(lane: _M2nLane) -> None:
        """Release a runtime-owned stream before its parent communicator.

        ``torch.cuda.Stream`` uses PyTorch's native stream pool and has no public
        destroy method, so dropping MX's owning reference is its supported
        release operation. Other stream implementations may expose ``close``;
        use it when present. Caller-owned streams are never destroyed by MX.
        """
        stream = lane.stream
        lane.stream = None
        if lane.owns_stream and stream is not None:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

    def _wait_for_finalize(self, lane: _M2nLane, *, deadline: float) -> None:
        while True:
            state = lane.communicator.get_async_error()
            value = int(state)
            if value == 0:
                return
            if value != 7:
                raise RuntimeError(
                    f"NCCL communicator finalize failed on lane {lane.lane_id!r}: {state}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"timed out finalizing NCCL communicator lane {lane.lane_id!r}"
                )
            time.sleep(0.001)

    def _require_open(self) -> None:
        with self._state_cv:
            self._require_open_locked()

    def _require_open_locked(self) -> None:
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
    "_M2nLane",
    "_M2nLaneBatch",
    "_M2nLaneSpec",
    "_M2nRuntime",
]
