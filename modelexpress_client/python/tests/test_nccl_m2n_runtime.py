# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-GPU tests for deterministic process-level NCCL M2N dispatch."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from itertools import pairwise
from types import SimpleNamespace

import pytest
from modelexpress.refit.reshard.transport.nccl_m2n import (
    runtime as runtime_module,
)
from modelexpress.refit.reshard.transport.nccl_m2n.mesh import (
    REPLICATE,
    build_tp_meshes,
    tile_shape,
)
from modelexpress.refit.reshard.transport.nccl_m2n.runtime import (
    M2nCohortRestartRequired,
    _M2nCall,
    _M2nPPGroupBatch,
    _M2nPPGroupSpec,
    _M2nRuntime,
    _PPGroupState,
    _RuntimeState,
)


class FakeEvent:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events
        self.name = "weights-ready"

    def record(self, stream: FakeStream) -> None:
        self.events.append(("event_record", self.name, stream.name))


class FakeStream:
    def __init__(self, name: str, events: list[tuple]) -> None:
        self.name = name
        self.events = events
        self.query_results: list[bool | BaseException] = [True]

    def wait_event(self, event: FakeEvent) -> None:
        self.events.append(("stream_wait_event", self.name, event.name))

    def set_query_results(
        self,
        *results: bool | BaseException,
    ) -> None:
        self.query_results = list(results)

    def query(self) -> bool:
        if not self.query_results:
            result: bool | BaseException = True
        elif len(self.query_results) == 1:
            result = self.query_results[0]
        else:
            result = self.query_results.pop(0)
        if isinstance(result, BaseException):
            self.events.append(("stream_query_error", self.name))
            raise result
        ready = bool(result)
        self.events.append(("stream_query", self.name, ready))
        return ready

    def synchronize(self) -> None:
        self.events.append(("stream_sync", self.name))

    def close(self) -> None:
        self.events.append(("stream_destroy", self.name))


class FakeCuda:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events
        self.stream_count = 0
        self.streams: list[FakeStream] = []
        self.producer_stream = FakeStream("producer", events)

    def set_device(self, device: int) -> None:
        self.events.append(("set_device", device))

    def Stream(self, *, device: int) -> FakeStream:
        stream = FakeStream(f"owned-{self.stream_count}", self.events)
        self.stream_count += 1
        self.streams.append(stream)
        self.events.append(("stream_create", stream.name, device))
        return stream

    def Event(self) -> FakeEvent:
        self.events.append(("event_create",))
        return FakeEvent(self.events)

    def current_stream(self, device: int) -> FakeStream:
        self.events.append(("current_stream", device))
        return self.producer_stream

    def stream(self, stream: FakeStream):
        return nullcontext()


class FakeComm:
    def __init__(self, name: str, events: list[tuple]) -> None:
        self.name = name
        self.events = events
        self.finalized = False
        self.async_states: list[int | BaseException] = [0]
        self.abort_error: BaseException | None = None
        self.abort_entered: threading.Event | None = None
        self.abort_release: threading.Event | None = None

    def set_async_states(self, *states: int | BaseException) -> None:
        self.async_states = list(states)

    def get_async_error(self) -> int:
        if len(self.async_states) > 1:
            state = self.async_states.pop(0)
        else:
            state = self.async_states[0]
        if isinstance(state, BaseException):
            raise state
        self.events.append(("comm_async", self.name, state))
        return state

    def abort(self) -> None:
        self.events.append(
            ("comm_abort_start", self.name, threading.current_thread().daemon)
        )
        if self.abort_entered is not None:
            self.abort_entered.set()
        if self.abort_release is not None:
            assert self.abort_release.wait(timeout=10)
        if self.abort_error is not None:
            raise self.abort_error
        self.events.append(("comm_abort_done", self.name))

    def finalize(self) -> None:
        self.events.append(("comm_finalize", self.name))
        self.finalized = True

    def destroy(self) -> None:
        assert self.finalized
        self.events.append(("comm_destroy", self.name))


class FakeUniqueId:
    def __init__(self, value: bytes) -> None:
        self.value = bytes(value)

    @staticmethod
    def from_bytes(value: bytes) -> FakeUniqueId:
        return FakeUniqueId(value)

    @property
    def as_bytes(self) -> bytes:
        return self.value


@dataclass(frozen=True)
class FakeVersion:
    release: tuple[int, ...]

    def __str__(self) -> str:
        return ".".join(str(value) for value in self.release)

    def __repr__(self) -> str:
        return f"<Version({str(self)!r})>"


@dataclass(frozen=True)
class FakeLibraryInfo:
    version: FakeVersion


@dataclass(frozen=True)
class FakeVersionInfo:
    nccl4py: FakeVersion
    nccl_bindings: FakeVersion
    libnccl: FakeLibraryInfo | None


@dataclass(frozen=True)
class FakeNCCLConfig:
    blocking: bool | None = None


class FakeNccl:
    UniqueId = FakeUniqueId
    NCCLConfig = FakeNCCLConfig

    def __init__(self, events: list[tuple]) -> None:
        self.events = events
        self.next_uid = b"new-uid"
        self.async_scripts: dict[str, list[int | BaseException]] = {}
        owner = self

        class Communicator(FakeComm):
            def __init__(self) -> None:
                super().__init__("<uninitialized>", owner.events)

            def initialize(
                self,
                nranks: int,
                rank: int,
                unique_id: FakeUniqueId,
                config: FakeNCCLConfig,
            ) -> None:
                name = unique_id.value.decode()
                self.name = name
                self.set_async_states(*owner.async_scripts.get(name, [0]))
                owner.events.append(("comm_init", name, nranks, rank, config.blocking))

        self.Communicator = Communicator

    def get_version(self) -> FakeVersionInfo:
        return FakeVersionInfo(
            nccl4py=FakeVersion((0, 4, 1)),
            nccl_bindings=FakeVersion((2, 30, 5)),
            libnccl=FakeLibraryInfo(FakeVersion((2, 30, 5))),
        )

    def get_unique_id(self) -> FakeUniqueId:
        return FakeUniqueId(self.next_uid)


@dataclass(frozen=True)
class FakeConfig:
    max_cta: int | None = None


@dataclass(frozen=True)
class FakeMesh:
    dims: tuple[int, int]
    start_rank: int = 0


@dataclass(frozen=True)
class FakeShard:
    dim: int


@dataclass(frozen=True)
class FakeReplicate:
    pass


@dataclass(frozen=True)
class FakeDistTensor:
    buffer: object
    local_shape: tuple[int, ...]
    dtype: object
    mesh: FakeMesh
    placements: tuple[object, ...]

    def __init__(
        self,
        buffer,
        local_shape,
        dtype,
        *,
        mesh,
        placements,
    ) -> None:
        object.__setattr__(self, "buffer", buffer)
        object.__setattr__(self, "local_shape", tuple(local_shape))
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "placements", tuple(placements))


class FakeHandle:
    def __init__(
        self,
        events: list[tuple],
        *,
        host_delay: float = 0.0,
        fail_at: int | None = None,
    ) -> None:
        self.events = events
        self.host_delay = host_delay
        self.fail_at = fail_at
        self.calls: list[dict] = []

    def reshard(self, comm, src, dst, *, stream) -> None:
        index = len(self.calls)
        self.calls.append({"comm": comm, "src": src, "dst": dst, "stream": stream})
        self.events.append(("reshard", comm.name, stream.name))
        if index == self.fail_at:
            raise RuntimeError("injected reshard failure")
        if self.host_delay:
            time.sleep(self.host_delay)

    def destroy(self) -> None:
        self.events.append(("handle_destroy",))


class FakeM2n:
    Config = FakeConfig
    Mesh = FakeMesh
    Shard = FakeShard
    Replicate = FakeReplicate
    DistTensor = FakeDistTensor

    def __init__(
        self,
        events: list[tuple],
        *,
        host_delay: float = 0.0,
        fail_at: int | None = None,
    ) -> None:
        self.events = events
        self.handle = FakeHandle(
            events,
            host_delay=host_delay,
            fail_at=fail_at,
        )

    def init(self, config: FakeConfig) -> FakeHandle:
        self.events.append(("m2n_init", config.max_cta))
        return self.handle

    @contextmanager
    def group(self):
        self.events.append(("group_start",))
        try:
            yield
        except BaseException:
            self.events.append(("group_abort",))
            raise
        else:
            self.events.append(("group_end",))


def make_runtime(
    *,
    host_delay: float = 0.0,
    fail_at: int | None = None,
    comm_init_timeout_s: float = 120.0,
    transfer_timeout_s: float = 900.0,
    finalize_timeout_s: float = 300.0,
    poll_interval_s: float = 0.002,
    enforce_singleton: bool = False,
):
    events: list[tuple] = []
    m2n = FakeM2n(events, host_delay=host_delay, fail_at=fail_at)
    nccl = FakeNccl(events)
    runtime = _M2nRuntime(
        0,
        max_cta=8,
        comm_init_timeout_s=comm_init_timeout_s,
        transfer_timeout_s=transfer_timeout_s,
        finalize_timeout_s=finalize_timeout_s,
        _poll_interval_s=poll_interval_s,
        _m2n_module=m2n,
        _nccl_module=nccl,
        _torch_module=SimpleNamespace(cuda=FakeCuda(events)),
        _enforce_singleton=enforce_singleton,
    )
    return runtime, m2n, nccl, events


def start_thread(target, *, name: str | None = None):
    results = []
    errors = []

    def run() -> None:
        try:
            results.append(target())
        except BaseException as exc:  # noqa: BLE001 - thread failures are assertions.
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True, name=name)
    thread.start()
    return thread, results, errors


def join_thread(thread: threading.Thread) -> None:
    thread.join(timeout=10)
    assert not thread.is_alive()


def wait_for_runtime_state(
    runtime: _M2nRuntime,
    state: _RuntimeState,
) -> None:
    deadline = time.monotonic() + 10
    with runtime._state_cv:
        while runtime._state is not state:
            remaining = deadline - time.monotonic()
            assert remaining > 0, f"runtime did not reach {state.name}"
            runtime._state_cv.wait(timeout=remaining)


def assert_active_operations(runtime: _M2nRuntime, expected: int) -> None:
    with runtime._state_cv:
        assert runtime._active_operations == expected


def make_spec(
    key: tuple[int, int],
    *,
    rank: int = 0,
    source_size: int = 1,
    destination_size: int = 1,
) -> _M2nPPGroupSpec:
    name = f"{key[0]}-{key[1]}"
    return _M2nPPGroupSpec(
        group_id=name,
        key=key,
        unique_id=name.encode(),
        source_size=source_size,
        destination_size=destination_size,
        comm_rank=rank,
        device_id=0,
    )


def create_groups(runtime: _M2nRuntime, keys, *, rank: int = 0):
    groups = runtime.create_pp_groups([make_spec(key, rank=rank) for key in keys])
    runtime.freeze_pp_groups()
    return groups


def make_call(runtime: _M2nRuntime, shard_dim: int = 0) -> _M2nCall:
    src_mesh, dst_mesh = build_tp_meshes(shard_dim, 1, 1)
    return _M2nCall.from_param(
        runtime.m2n,
        name="weight",
        src_buffer=1,
        dst_buffer=None,
        src_mesh=src_mesh,
        dst_mesh=dst_mesh,
        src_local_shape=tile_shape((4,), src_mesh),
        dst_local_shape=tile_shape((4,), dst_mesh),
        dtype="float32",
    )


def make_batch(runtime: _M2nRuntime, pp_group) -> _M2nPPGroupBatch:
    return _M2nPPGroupBatch(
        pp_group=pp_group,
        calls=(make_call(runtime),),
        total_bytes=16,
    )


def test_create_pp_groups_creates_comms_and_streams_in_canonical_order():
    runtime, _, _, events = make_runtime()
    groups = create_groups(runtime, [(1, 0), (0, 0)])

    assert [group.key for group in groups] == [(0, 0), (1, 0)]
    assert [event[1] for event in events if event[0] == "comm_init"] == [
        "0-0",
        "1-0",
    ]
    runtime.close()


def test_create_failure_after_native_init_is_fail_stop_for_full_intended_scope():
    runtime, _, nccl, events = make_runtime()
    original_initialize = nccl.Communicator.initialize

    def fail_second_initialize(self, nranks, rank, unique_id, config):
        if unique_id.value == b"1-0":
            raise ValueError("injected second communicator init failure")
        return original_initialize(self, nranks, rank, unique_id, config)

    nccl.Communicator.initialize = fail_second_initialize
    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        runtime.create_pp_groups([make_spec((1, 0)), make_spec((0, 0))])

    assert exc_info.value.operation == "create_pp_groups"
    assert exc_info.value.phase == "communicator_init"
    assert exc_info.value.group_ids == ("0-0", "1-0")
    assert exc_info.value.pp_group_keys == ((0, 0), (1, 0))
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "second communicator init failure" in str(exc_info.value.__cause__)
    assert [group.key for group in runtime.pp_groups] == [(0, 0), (1, 0)]
    assert all(group.state is _PPGroupState.POISONED for group in runtime.pp_groups)
    assert runtime._abort_done.wait(timeout=10)
    assert all(group.abort_attempted for group in runtime.pp_groups)
    assert sum(event[0] == "comm_abort_start" for event in events) == 2
    assert not any(
        event[0]
        in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
        for event in events
    )


def test_submit_uses_one_group_and_canonical_pp_group_order():
    runtime, _, _, events = make_runtime()
    early, late = create_groups(runtime, [(1, 0), (0, 0)])

    runtime.submit_model_update([make_batch(runtime, late), make_batch(runtime, early)])

    pipeline = [
        event
        for event in events
        if event[0] in {"group_start", "reshard", "group_end", "stream_query"}
    ]
    assert pipeline[:6] == [
        ("group_start",),
        ("reshard", "0-0", "owned-0"),
        ("reshard", "1-0", "owned-1"),
        ("group_end",),
        ("stream_query", "owned-0", True),
        ("stream_query", "owned-1", True),
    ]
    runtime.close()


def test_two_caller_threads_cannot_change_recorded_pp_group_order():
    runtime, _, _, events = make_runtime(host_delay=0.01)
    early, late = create_groups(runtime, [(1, 0), (0, 0)])
    barrier = threading.Barrier(3)

    def submit() -> None:
        barrier.wait()
        runtime.submit_model_update(
            [make_batch(runtime, late), make_batch(runtime, early)]
        )

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert [event[1] for event in events if event[0] == "reshard"] == [
        "0-0",
        "1-0",
        "0-0",
        "1-0",
    ]
    runtime.close()


def test_partial_pp_group_submission_is_rejected():
    runtime, _, _, _ = make_runtime()
    early, _ = create_groups(runtime, [(0, 0), (1, 0)])

    with pytest.raises(RuntimeError, match="every local PP group"):
        runtime.submit_model_update([make_batch(runtime, early)])
    runtime.close()


def test_all_empty_update_skips_native_submission_and_completion_polling():
    runtime, _, _, events = make_runtime()
    early, late = create_groups(runtime, [(1, 0), (0, 0)])
    batches = [
        _M2nPPGroupBatch(pp_group=late, calls=(), total_bytes=0),
        _M2nPPGroupBatch(pp_group=early, calls=(), total_bytes=0),
    ]
    events.clear()

    assert runtime.submit_model_update(batches) == {(0, 0): 0, (1, 0): 0}
    assert not any(
        event[0]
        in {
            "event_create",
            "group_start",
            "group_end",
            "reshard",
            "comm_async",
            "stream_query",
        }
        for event in events
    )
    runtime.close()


def test_two_multi_group_processes_record_same_sorted_sequence():
    sequences = []
    for insertion_order in (
        ((2, 0), (0, 0), (1, 0)),
        ((1, 0), (2, 0), (0, 0)),
    ):
        runtime, _, _, events = make_runtime()
        groups = create_groups(runtime, insertion_order)
        runtime.submit_model_update(
            [make_batch(runtime, group) for group in reversed(groups)]
        )
        sequences.append([event[1] for event in events if event[0] == "reshard"])
        runtime.close()
    assert sequences == [["0-0", "1-0", "2-0"], ["0-0", "1-0", "2-0"]]


def test_source_streams_wait_for_current_stream_readiness():
    runtime, _, _, events = make_runtime()
    (pp_group,) = create_groups(runtime, [(0, 0)], rank=0)

    runtime.submit_model_update([make_batch(runtime, pp_group)])

    ready = [
        event
        for event in events
        if event[0] in {"event_record", "stream_wait_event", "reshard"}
    ]
    assert ready == [
        ("event_record", "weights-ready", "producer"),
        ("stream_wait_event", "owned-0", "weights-ready"),
        ("reshard", "0-0", "owned-0"),
    ]
    runtime.close()


def test_fully_replicated_layout_builds_official_dist_tensors():
    runtime, m2n, _, _ = make_runtime()
    (pp_group,) = create_groups(runtime, [(0, 0)])

    call = make_call(runtime, REPLICATE)
    runtime.submit_model_update(
        [_M2nPPGroupBatch(pp_group=pp_group, calls=(call,), total_bytes=16)]
    )

    submitted = m2n.handle.calls[0]
    assert isinstance(submitted["src"], FakeDistTensor)
    assert submitted["src"].placements == (FakeShard(0), FakeReplicate())
    assert submitted["dst"].placements == (FakeShard(0), FakeReplicate())
    runtime.close()


def test_group_exception_aborts_and_poisons_runtime():
    runtime, _, _, events = make_runtime(fail_at=0)
    (pp_group,) = create_groups(runtime, [(0, 0)])

    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        runtime.submit_model_update([make_batch(runtime, pp_group)])
    assert exc_info.value.operation == "stage"
    assert exc_info.value.phase == "submission"
    assert exc_info.value.group_ids == ("0-0",)
    assert exc_info.value.pp_group_keys == ((0, 0),)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "injected reshard failure" in str(exc_info.value.__cause__)
    assert ("group_abort",) in events
    assert runtime._abort_done.wait(timeout=10)
    assert ("comm_abort_start", "0-0", True) in events
    with pytest.raises(M2nCohortRestartRequired) as repeated:
        runtime.submit_model_update([make_batch(runtime, pp_group)])
    assert repeated.value.__cause__ is exc_info.value.__cause__
    with pytest.raises(M2nCohortRestartRequired):
        runtime.close()
    assert ("handle_destroy",) not in events


def test_shutdown_finalizes_m2n_before_comms_in_canonical_order():
    runtime, _, _, events = make_runtime()
    create_groups(runtime, [(1, 0), (0, 0)])

    runtime.close()

    lifecycle = [
        event
        for event in events
        if event[0]
        in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
    ]
    assert lifecycle == [
        ("handle_destroy",),
        ("stream_destroy", "owned-0"),
        ("comm_finalize", "0-0"),
        ("comm_destroy", "0-0"),
        ("stream_destroy", "owned-1"),
        ("comm_finalize", "1-0"),
        ("comm_destroy", "1-0"),
    ]


@pytest.mark.parametrize(
    ("failure", "expected_phase", "expected_key"),
    [
        ("handle-destroy", "m2n-finalize", None),
        ("stream-close", "stream-release", (0, 0)),
        ("comm-finalize", "comm-finalize", (0, 0)),
        ("finalize-wait-error", "comm-finalize-wait", (0, 0)),
        ("finalize-wait-timeout", "comm-finalize-wait", (0, 0)),
        ("comm-destroy", "comm-destroy", (0, 0)),
    ],
)
def test_post_drain_teardown_failure_is_terminal_and_not_retried(
    failure: str,
    expected_phase: str,
    expected_key: tuple[int, int] | None,
):
    runtime, m2n, _, events = make_runtime(
        finalize_timeout_s=0.003,
        poll_interval_s=0.0005,
    )
    (pp_group,) = create_groups(runtime, [(0, 0)])
    original_stream = pp_group.stream

    if failure == "handle-destroy":

        def fail_handle_destroy() -> None:
            events.append(("handle_destroy_error",))
            raise RuntimeError("injected handle destroy failure")

        m2n.handle.destroy = fail_handle_destroy
    elif failure == "stream-close":

        def fail_stream_close() -> None:
            events.append(("stream_destroy_error", original_stream.name))
            raise RuntimeError("injected stream close failure")

        original_stream.close = fail_stream_close
    elif failure == "comm-finalize":

        def fail_comm_finalize() -> None:
            events.append(("comm_finalize_error", pp_group.group_id))
            raise RuntimeError("injected communicator finalize failure")

        pp_group.communicator.finalize = fail_comm_finalize
    elif failure in {"finalize-wait-error", "finalize-wait-timeout"}:
        original_get_async_error = pp_group.communicator.get_async_error

        def fail_finalize_wait() -> int:
            if not pp_group.communicator.finalized:
                return original_get_async_error()
            if failure == "finalize-wait-timeout":
                events.append(("comm_async", pp_group.group_id, 7))
                return 7
            events.append(("comm_async_error", pp_group.group_id))
            raise RuntimeError("injected finalize wait failure")

        pp_group.communicator.get_async_error = fail_finalize_wait
    else:

        def fail_comm_destroy() -> None:
            events.append(("comm_destroy_error", pp_group.group_id))
            raise RuntimeError("injected communicator destroy failure")

        pp_group.communicator.destroy = fail_comm_destroy

    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        runtime.close()
    assert exc_info.value.operation == "close"
    assert exc_info.value.phase == expected_phase
    assert exc_info.value.group_ids == ("0-0",)
    assert exc_info.value.pp_group_keys == ((0, 0),)
    assert exc_info.value.__cause__ is not None

    assert runtime._state is _RuntimeState.POISONED
    assert runtime._restart_required
    assert runtime._close_abandoned
    assert runtime._teardown_failure_phase == expected_phase
    assert runtime._teardown_failure_key == expected_key
    assert f"phase={expected_phase}" in runtime._fail_stop_reason
    if expected_key is None:
        assert "pp_group=" not in runtime._fail_stop_reason
    else:
        assert f"pp_group={expected_key}" in runtime._fail_stop_reason
    assert pp_group.state is _PPGroupState.POISONED
    assert runtime.pp_groups == (pp_group,)
    assert runtime._abort_thread is None
    assert not any(event[0] == "comm_abort_start" for event in events)

    if failure == "handle-destroy":
        assert runtime._handle is m2n.handle
        assert runtime._handle_quarantined
        assert pp_group.stream is original_stream
    else:
        assert runtime._handle is None
        if failure == "stream-close":
            assert pp_group.stream is original_stream
        else:
            assert pp_group.stream is None

    events_before_retry = tuple(events)
    with pytest.raises(M2nCohortRestartRequired):
        runtime.close()
    assert tuple(events) == events_before_retry


def test_partial_teardown_after_earlier_group_closed_is_not_replayed():
    runtime, _, _, events = make_runtime()
    early, late = create_groups(runtime, [(0, 0), (1, 0)])

    def fail_late_finalize() -> None:
        events.append(("comm_finalize_error", late.group_id))
        raise RuntimeError("injected later-group finalize failure")

    late.communicator.finalize = fail_late_finalize
    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        runtime.close()
    assert exc_info.value.phase == "comm-finalize"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "later-group finalize failure" in str(exc_info.value.__cause__)

    assert early.state is _PPGroupState.CLOSED
    assert late.state is _PPGroupState.POISONED
    assert runtime._teardown_failure_phase == "comm-finalize"
    assert runtime._teardown_failure_key == (1, 0)
    assert ("comm_destroy", early.group_id) in events
    assert not any(event[0] == "comm_abort_start" for event in events)

    events_before_retry = tuple(events)
    with pytest.raises(M2nCohortRestartRequired):
        runtime.close()
    assert tuple(events) == events_before_retry


def test_pre_destructive_close_exception_restores_open_state_and_can_retry():
    runtime, _, _, events = make_runtime()
    create_groups(runtime, [(0, 0)])
    dispatcher = runtime._dispatcher_lock

    class FailBeforeDrain:
        def __enter__(self):
            raise RuntimeError("injected pre-destructive close failure")

        def __exit__(self, exc_type, exc, traceback):
            return False

    runtime._dispatcher_lock = FailBeforeDrain()
    with pytest.raises(RuntimeError, match="pre-destructive close failure"):
        runtime.close()

    assert runtime._state is _RuntimeState.OPEN
    assert not runtime._restart_required
    assert not runtime._close_abandoned
    assert runtime._teardown_failure_phase is None
    assert not any(
        event[0]
        in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
        for event in events
    )

    runtime._dispatcher_lock = dispatcher
    runtime.close()
    assert runtime._state is _RuntimeState.CLOSED


def test_close_wait_exception_restores_open_state_and_can_retry():
    runtime, _, _, events = make_runtime()
    create_groups(runtime, [(0, 0)])
    operation_entered = threading.Event()
    release_operation = threading.Event()

    def hold_active_operation() -> None:
        with runtime._active_operation():
            operation_entered.set()
            assert release_operation.wait(timeout=10)

    operation_thread, _, operation_errors = start_thread(hold_active_operation)
    assert operation_entered.wait(timeout=10)
    original_wait = runtime._state_cv.wait

    def fail_wait(*, timeout=None):
        del timeout
        raise RuntimeError("injected close condition-wait failure")

    runtime._state_cv.wait = fail_wait
    with pytest.raises(RuntimeError, match="condition-wait failure"):
        runtime.close()

    assert runtime._state is _RuntimeState.OPEN
    assert not runtime._restart_required
    assert not runtime._close_abandoned
    assert not any(
        event[0]
        in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
        for event in events
    )

    runtime._state_cv.wait = original_wait
    release_operation.set()
    join_thread(operation_thread)
    assert not operation_errors
    runtime.close()
    assert runtime._state is _RuntimeState.CLOSED


def test_close_notification_exception_restores_open_state_and_can_retry():
    runtime, _, _, events = make_runtime()
    create_groups(runtime, [(0, 0)])
    original_notify_all = runtime._state_cv.notify_all
    notify_attempts = 0

    def fail_first_notify() -> None:
        nonlocal notify_attempts
        notify_attempts += 1
        if notify_attempts == 1:
            raise RuntimeError("injected close notification failure")
        original_notify_all()

    runtime._state_cv.notify_all = fail_first_notify
    with pytest.raises(RuntimeError, match="notification failure"):
        runtime.close()

    assert runtime._state is _RuntimeState.OPEN
    assert not runtime._restart_required
    assert not runtime._close_abandoned
    assert not any(
        event[0]
        in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
        for event in events
    )

    runtime._state_cv.notify_all = original_notify_all
    runtime.close()
    assert runtime._state is _RuntimeState.CLOSED


def test_close_commit_failure_is_terminal_and_retains_singleton():
    runtime, _, _, events = make_runtime(enforce_singleton=True)
    (pp_group,) = create_groups(runtime, [(0, 0)])
    original_groups = runtime._pp_groups

    class FailClearDict(dict):
        def clear(self) -> None:
            events.append(("runtime_groups_clear_error",))
            raise RuntimeError("injected runtime close commit failure")

    runtime._pp_groups = FailClearDict(original_groups)
    try:
        with pytest.raises(M2nCohortRestartRequired) as exc_info:
            runtime.close()
        assert exc_info.value.phase == "runtime-close-commit"
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "close commit failure" in str(exc_info.value.__cause__)

        assert runtime._state is _RuntimeState.POISONED
        assert runtime._restart_required
        assert runtime._close_abandoned
        assert runtime._teardown_failure_phase == "runtime-close-commit"
        assert runtime._teardown_failure_key is None
        assert pp_group.state is _PPGroupState.CLOSED
        assert runtime.pp_groups == (pp_group,)
        assert _M2nRuntime._live_runtime is runtime
        assert not any(event[0] == "comm_abort_start" for event in events)

        events_before_retry = tuple(events)
        with pytest.raises(M2nCohortRestartRequired):
            runtime.close()
        assert tuple(events) == events_before_retry
    finally:
        runtime._clear_singleton()


def test_healthy_close_clears_singleton_only_after_closed_bookkeeping():
    runtime, _, _, events = make_runtime(enforce_singleton=True)
    create_groups(runtime, [(0, 0)])
    original_clear_singleton = runtime._clear_singleton

    def checked_clear_singleton() -> None:
        assert runtime._state is _RuntimeState.CLOSED
        assert not runtime._pp_groups
        events.append(("singleton_clear",))
        original_clear_singleton()

    runtime._clear_singleton = checked_clear_singleton
    runtime.close()

    assert ("singleton_clear",) in events
    assert _M2nRuntime._live_runtime is None


def test_closed_retry_clears_singleton_without_replaying_native_teardown():
    runtime, _, _, events = make_runtime(enforce_singleton=True)
    create_groups(runtime, [(0, 0)])
    original_clear_singleton = runtime._clear_singleton
    clear_attempts = 0

    def fail_first_clear() -> None:
        nonlocal clear_attempts
        clear_attempts += 1
        if clear_attempts == 1:
            raise RuntimeError("injected singleton clear failure")
        original_clear_singleton()

    runtime._clear_singleton = fail_first_clear
    with pytest.raises(RuntimeError, match="singleton clear failure"):
        runtime.close()

    assert runtime._state is _RuntimeState.CLOSED
    assert not runtime.pp_groups
    assert _M2nRuntime._live_runtime is runtime
    native_events = tuple(
        event
        for event in events
        if event[0]
        in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
    )

    runtime.close()

    assert clear_attempts == 2
    assert _M2nRuntime._live_runtime is None
    assert (
        tuple(
            event
            for event in events
            if event[0]
            in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
        )
        == native_events
    )


def test_close_waits_for_create_and_snapshots_complete_topology():
    runtime, _, nccl, events = make_runtime()
    init_entered = threading.Event()
    release_init = threading.Event()
    original_initialize = nccl.Communicator.initialize
    init_count = 0

    def blocking_initialize(self, nranks, rank, unique_id, config):
        nonlocal init_count
        init_count += 1
        if init_count == 2:
            init_entered.set()
            assert release_init.wait(timeout=10)
        return original_initialize(self, nranks, rank, unique_id, config)

    nccl.Communicator.initialize = blocking_initialize
    create_thread, create_results, create_errors = start_thread(
        lambda: runtime.create_pp_groups([make_spec((1, 0)), make_spec((0, 0))])
    )
    assert init_entered.wait(timeout=10)

    close_thread, _, close_errors = start_thread(runtime.close)
    wait_for_runtime_state(runtime, _RuntimeState.CLOSING)
    assert_active_operations(runtime, 1)

    assert runtime.pp_groups == ()
    assert not any(
        event[0]
        in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
        for event in events
    )
    with pytest.raises(RuntimeError, match="closing"):
        runtime.new_unique_id_bytes()

    release_init.set()
    join_thread(create_thread)
    join_thread(close_thread)

    assert not create_errors
    assert not close_errors
    assert [group.key for group in create_results[0]] == [(0, 0), (1, 0)]
    assert [event[1] for event in events if event[0] == "comm_destroy"] == [
        "0-0",
        "1-0",
    ]


def test_close_timeout_retains_resources_and_requires_process_restart():
    runtime, _, nccl, events = make_runtime(finalize_timeout_s=0.01)
    init_entered = threading.Event()
    release_init = threading.Event()
    original_initialize = nccl.Communicator.initialize

    def blocking_initialize(self, nranks, rank, unique_id, config):
        init_entered.set()
        assert release_init.wait(timeout=10)
        return original_initialize(self, nranks, rank, unique_id, config)

    nccl.Communicator.initialize = blocking_initialize
    create_thread, _, create_errors = start_thread(
        lambda: runtime.create_pp_groups([make_spec((0, 0))])
    )
    assert init_entered.wait(timeout=10)

    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        runtime.close()
    assert exc_info.value.operation == "close"
    assert exc_info.value.phase == "active_operation_drain"
    assert isinstance(exc_info.value.__cause__, TimeoutError)
    assert "active M2N operations" in str(exc_info.value.__cause__)

    assert runtime._state is _RuntimeState.POISONED
    assert runtime._close_abandoned
    assert not any(
        event[0]
        in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
        for event in events
    )

    release_init.set()
    join_thread(create_thread)
    assert len(create_errors) == 1
    assert isinstance(create_errors[0], M2nCohortRestartRequired)
    assert create_errors[0].operation == "create_pp_groups"
    assert create_errors[0].phase == "shutdown_race"
    assert create_errors[0].group_ids == ("0-0",)
    assert create_errors[0].pp_group_keys == ((0, 0),)
    assert len(runtime.pp_groups) == 1
    assert not any(event[0] == "comm_abort_start" for event in events)

    with pytest.raises(M2nCohortRestartRequired):
        runtime.close()
    with pytest.raises(M2nCohortRestartRequired):
        runtime.create_pp_groups([])
    assert not any(
        event[0]
        in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
        for event in events
    )


def test_close_timeout_preserves_failure_recorded_while_draining_active_op():
    runtime, _, _, _ = make_runtime(finalize_timeout_s=0.05)
    operation_entered = threading.Event()
    record_failure = threading.Event()
    failure_recorded = threading.Event()
    release_operation = threading.Event()
    original_failure = RuntimeError("original native operation failure")

    def active_operation() -> None:
        with runtime._active_operation():
            operation_entered.set()
            assert record_failure.wait(timeout=10)
            runtime._enter_fail_stop((), (), original_failure)
            failure_recorded.set()
            assert release_operation.wait(timeout=10)

    operation_thread, _, operation_errors = start_thread(active_operation)
    assert operation_entered.wait(timeout=10)
    close_thread, _, close_errors = start_thread(runtime.close)
    wait_for_runtime_state(runtime, _RuntimeState.CLOSING)

    record_failure.set()
    assert failure_recorded.wait(timeout=10)
    join_thread(close_thread)

    assert len(close_errors) == 1
    error = close_errors[0]
    assert isinstance(error, M2nCohortRestartRequired)
    assert error.operation == "close"
    assert error.phase == "active_operation_drain"
    assert error.reason == "RuntimeError: original native operation failure"
    assert error.__cause__ is original_failure
    assert runtime._fail_stop_cause is original_failure
    assert runtime._fail_stop_reason == error.reason
    assert runtime._close_abandoned
    assert runtime._handle_quarantined

    release_operation.set()
    join_thread(operation_thread)
    assert not operation_errors


def test_close_from_active_operation_fails_without_changing_state():
    runtime, _, _, _ = make_runtime()

    with runtime._active_operation():
        with pytest.raises(RuntimeError, match="active runtime operation"):
            runtime.close()
        assert runtime._state is _RuntimeState.OPEN

    runtime.close()


def test_empty_operations_cannot_bypass_closed_state():
    runtime, _, _, _ = make_runtime()
    runtime.close()

    with pytest.raises(RuntimeError, match="closed"):
        runtime.create_pp_groups([])
    with pytest.raises(RuntimeError, match="closed"):
        runtime.submit_model_update([])


@pytest.mark.parametrize("trainer_pp", [2, 4, 8])
def test_pp_to_pp1_ownership_patterns_cannot_form_ordering_cycle(trainer_pp: int):
    keys = tuple((stage, 0) for stage in range(trainer_pp))
    owner_sequences = [keys, *((key,) for key in keys)]
    edges = {
        (left, right)
        for sequence in owner_sequences
        for left, right in pairwise(sequence)
    }
    assert all(left < right for left, right in edges)


def test_new_unique_id_and_official_comm_api():
    runtime, _, nccl, events = make_runtime()
    assert runtime.new_unique_id_bytes() == nccl.next_uid
    (pp_group,) = runtime.create_pp_groups([make_spec((0, 0), rank=1)])
    assert pp_group.comm_rank == 1
    assert ("comm_init", "0-0", 2, 1, False) in events
    runtime.close()


def test_rejects_old_nccl_version():
    events: list[tuple] = []
    nccl = FakeNccl(events)
    # Regression: old text parser extracted the "4" in the nccl4py field name
    # and treated this VersionInfo as NCCL 4.0.4, allowing libnccl 2.27.
    nccl.get_version = lambda: FakeVersionInfo(
        nccl4py=FakeVersion((0, 4, 1)),
        nccl_bindings=FakeVersion((2, 27, 0)),
        libnccl=FakeLibraryInfo(FakeVersion((2, 27, 0))),
    )
    with pytest.raises(RuntimeError, match="NCCL >= 2.30.5"):
        _M2nRuntime(
            0,
            _m2n_module=FakeM2n(events),
            _nccl_module=nccl,
            _torch_module=SimpleNamespace(cuda=FakeCuda(events)),
            _enforce_singleton=False,
        )


def test_rejects_version_info_without_loaded_libnccl():
    events: list[tuple] = []
    nccl = FakeNccl(events)
    nccl.get_version = lambda: FakeVersionInfo(
        nccl4py=FakeVersion((0, 4, 1)),
        nccl_bindings=FakeVersion((2, 30, 5)),
        libnccl=None,
    )

    with pytest.raises(RuntimeError, match="could not identify a loaded libnccl"):
        _M2nRuntime(
            0,
            _m2n_module=FakeM2n(events),
            _nccl_module=nccl,
            _torch_module=SimpleNamespace(cuda=FakeCuda(events)),
            _enforce_singleton=False,
        )


def test_accepts_encoded_current_nccl_version():
    events: list[tuple] = []
    nccl = FakeNccl(events)
    nccl.get_version = lambda: 23005

    runtime = _M2nRuntime(
        0,
        _m2n_module=FakeM2n(events),
        _nccl_module=nccl,
        _torch_module=SimpleNamespace(cuda=FakeCuda(events)),
        _enforce_singleton=False,
    )
    runtime.close()


def test_runtime_requires_fault_tolerant_nccl4py_api():
    for missing in ("config", "abort"):
        events: list[tuple] = []
        nccl = FakeNccl(events)
        if missing == "config":
            nccl.NCCLConfig = None
        else:
            nccl.Communicator.abort = None
        with pytest.raises(RuntimeError, match="fault-tolerant mode"):
            _M2nRuntime(
                0,
                _m2n_module=FakeM2n(events),
                _nccl_module=nccl,
                _torch_module=SimpleNamespace(cuda=FakeCuda(events)),
                _enforce_singleton=False,
            )
        assert ("m2n_init", None) not in events


def test_nonblocking_init_polls_all_pp_groups_round_robin():
    runtime, _, nccl, events = make_runtime(poll_interval_s=0.0001)
    nccl.async_scripts = {
        "0-0": [7, 0],
        "1-0": [7, 0],
    }

    groups = create_groups(runtime, [(1, 0), (0, 0)])

    assert [group.key for group in groups] == [(0, 0), (1, 0)]
    assert [event for event in events if event[0] == "comm_init"] == [
        ("comm_init", "0-0", 2, 0, False),
        ("comm_init", "1-0", 2, 0, False),
    ]
    assert [(event[1], event[2]) for event in events if event[0] == "comm_async"][
        :4
    ] == [
        ("0-0", 7),
        ("1-0", 7),
        ("0-0", 0),
        ("1-0", 0),
    ]
    runtime.close()


def test_init_timeout_aborts_all_created_groups_canonically():
    runtime, _, nccl, events = make_runtime(
        comm_init_timeout_s=0.005,
        poll_interval_s=0.001,
    )
    nccl.async_scripts = {
        "0-0": [7],
        "1-0": [7],
    }

    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        runtime.create_pp_groups([make_spec((1, 0)), make_spec((0, 0))])
    assert exc_info.value.operation == "create_pp_groups"
    assert exc_info.value.phase == "communicator_init"
    assert exc_info.value.group_ids == ("0-0", "1-0")
    assert exc_info.value.pp_group_keys == ((0, 0), (1, 0))
    assert isinstance(exc_info.value.__cause__, TimeoutError)
    assert "PP-group initialization" in str(exc_info.value.__cause__)

    assert runtime._abort_done.wait(timeout=10)
    assert runtime._restart_required
    assert [group.key for group in runtime.pp_groups] == [(0, 0), (1, 0)]
    assert [event[1] for event in events if event[0] == "comm_abort_start"] == [
        "0-0",
        "1-0",
    ]
    assert not any(
        event[0]
        in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
        for event in events
    )
    with pytest.raises(M2nCohortRestartRequired):
        runtime.close()


def test_completion_requires_nccl_success_and_stream_readiness():
    runtime, _, _, events = make_runtime(poll_interval_s=0.0001)
    (pp_group,) = create_groups(runtime, [(0, 0)])

    events.clear()
    pp_group.communicator.set_async_states(7, 0)
    pp_group.stream.set_query_results(True)
    runtime.submit_model_update([make_batch(runtime, pp_group)])
    assert [event[2] for event in events if event[0] == "comm_async"][:2] == [7, 0]
    assert [event for event in events if event[0] == "stream_query"] == [
        ("stream_query", "owned-0", True)
    ]

    events.clear()
    pp_group.communicator.set_async_states(0)
    pp_group.stream.set_query_results(False, True)
    runtime.submit_model_update([make_batch(runtime, pp_group)])
    assert [event[2] for event in events if event[0] == "stream_query"][:2] == [
        False,
        True,
    ]
    runtime.close()


def test_multi_group_completion_detects_later_error_round_robin():
    runtime, _, _, events = make_runtime(
        transfer_timeout_s=0.05,
        poll_interval_s=0.001,
    )
    early, late = create_groups(runtime, [(0, 0), (1, 0)])
    early.communicator.set_async_states(0)
    early.stream.set_query_results(False)
    late.communicator.set_async_states(2)
    events.clear()

    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        runtime.submit_model_update(
            [make_batch(runtime, late), make_batch(runtime, early)]
        )
    assert exc_info.value.operation == "stage"
    assert exc_info.value.phase == "completion"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "status=2" in str(exc_info.value.__cause__)

    assert runtime._abort_done.wait(timeout=10)
    observed = [event for event in events if event[0] in {"comm_async", "stream_query"}]
    assert observed[:3] == [
        ("comm_async", "0-0", 0),
        ("stream_query", "owned-0", False),
        ("comm_async", "1-0", 2),
    ]
    assert [event[1] for event in events if event[0] == "comm_abort_start"] == [
        "0-0",
        "1-0",
    ]
    with pytest.raises(M2nCohortRestartRequired):
        runtime.close()


def test_hanging_abort_worker_does_not_block_transfer_failure():
    runtime, _, _, events = make_runtime(
        transfer_timeout_s=0.005,
        poll_interval_s=0.001,
    )
    (pp_group,) = create_groups(runtime, [(0, 0)])
    pp_group.stream.set_query_results(False)
    abort_entered = threading.Event()
    abort_release = threading.Event()
    pp_group.communicator.abort_entered = abort_entered
    pp_group.communicator.abort_release = abort_release

    try:
        with pytest.raises(M2nCohortRestartRequired) as exc_info:
            runtime.submit_model_update([make_batch(runtime, pp_group)])
        assert exc_info.value.operation == "stage"
        assert exc_info.value.phase == "completion"
        assert isinstance(exc_info.value.__cause__, TimeoutError)
        assert "model-version staging" in str(exc_info.value.__cause__)
        assert abort_entered.wait(timeout=10)
        assert runtime._abort_thread is not None
        assert runtime._abort_thread.daemon
        assert runtime._abort_thread.is_alive()
        assert runtime._quarantined_batches
        assert runtime._quarantined_batches[0].calls[0].src.buffer == 1
        assert pp_group.stream is not None
        assert ("handle_destroy",) not in events
    finally:
        abort_release.set()

    assert runtime._abort_done.wait(timeout=10)
    with pytest.raises(M2nCohortRestartRequired):
        runtime.close()


def test_healthy_close_drain_timeout_quarantines_without_finalizing():
    runtime, _, _, events = make_runtime(
        finalize_timeout_s=0.005,
        poll_interval_s=0.001,
    )
    (pp_group,) = create_groups(runtime, [(0, 0)])
    pp_group.stream.set_query_results(False)

    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        runtime.close()
    assert exc_info.value.operation == "close"
    assert exc_info.value.phase == "stream_drain"
    assert isinstance(exc_info.value.__cause__, TimeoutError)
    assert "runtime shutdown drain" in str(exc_info.value.__cause__)

    assert runtime._abort_done.wait(timeout=10)
    assert runtime._state is _RuntimeState.POISONED
    assert runtime._restart_required
    assert not runtime._close_abandoned
    assert runtime._teardown_failure_phase is None
    assert ("comm_abort_start", "0-0", True) in events
    assert not any(
        event[0]
        in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
        for event in events
    )


def test_healthy_drain_waits_for_dispatcher_before_poll_or_abort():
    runtime, _, _, events = make_runtime()
    (pp_group,) = create_groups(runtime, [(0, 0)])
    host_entered = threading.Event()
    host_release = threading.Event()
    original_reshard = runtime.handle.reshard

    def blocking_reshard(comm, src, dst, *, stream):
        result = original_reshard(comm, src, dst, stream=stream)
        events.append(("m2n_host_enter",))
        host_entered.set()
        assert host_release.wait(timeout=10)
        events.append(("m2n_host_exit",))
        return result

    original_async_state = pp_group.communicator.get_async_error

    def thread_aware_async_state():
        if threading.current_thread().name == "drain":
            events.append(("drain_status_query",))
            raise RuntimeError("injected drain status failure")
        return original_async_state()

    runtime.handle.reshard = blocking_reshard
    pp_group.communicator.get_async_error = thread_aware_async_state

    drain_poll_entered = threading.Event()
    original_poll_completion = runtime._poll_pp_groups_completion

    def recording_poll_completion(pp_groups, *, operation, deadline):
        if operation == "test drain":
            drain_poll_entered.set()
        return original_poll_completion(
            pp_groups,
            operation=operation,
            deadline=deadline,
        )

    runtime._poll_pp_groups_completion = recording_poll_completion

    submit_thread, _, submit_errors = start_thread(
        lambda: runtime.submit_model_update([make_batch(runtime, pp_group)]),
        name="submit",
    )
    assert host_entered.wait(timeout=10)

    drain_started = threading.Event()

    def drain():
        drain_started.set()
        runtime.wait_for_pp_groups((pp_group,), operation="test drain")

    drain_thread, _, drain_errors = start_thread(drain, name="drain")
    assert drain_started.wait(timeout=10)
    try:
        assert not drain_poll_entered.wait(timeout=0.05)
        assert runtime._dispatcher_lock.locked()
        assert ("drain_status_query",) not in events
        assert not any(event[0] == "comm_abort_start" for event in events)
    finally:
        host_release.set()

    join_thread(submit_thread)
    join_thread(drain_thread)
    assert drain_poll_entered.is_set()
    assert not submit_errors
    assert len(drain_errors) == 1
    assert isinstance(drain_errors[0], M2nCohortRestartRequired)
    assert drain_errors[0].operation == "test drain"
    assert drain_errors[0].phase == "completion"
    assert isinstance(drain_errors[0].__cause__, RuntimeError)
    assert "injected drain status failure" in str(drain_errors[0].__cause__)
    assert runtime._abort_done.wait(timeout=10)
    assert events.index(("m2n_host_exit",)) < events.index(("drain_status_query",))
    abort_index = next(
        index for index, event in enumerate(events) if event[0] == "comm_abort_start"
    )
    assert events.index(("drain_status_query",)) < abort_index


def test_abort_failure_continues_to_later_pp_groups():
    runtime, _, _, events = make_runtime()
    early, late = create_groups(runtime, [(0, 0), (1, 0)])
    early.communicator.abort_error = RuntimeError("injected abort failure")
    late.communicator.set_async_states(RuntimeError("injected status failure"))

    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        runtime.submit_model_update(
            [make_batch(runtime, late), make_batch(runtime, early)]
        )

    assert exc_info.value.operation == "stage"
    assert exc_info.value.phase == "completion"
    assert "injected status failure" in str(exc_info.value.__cause__)
    assert runtime._abort_done.wait(timeout=10)
    assert [event[1] for event in events if event[0] == "comm_abort_start"] == [
        "0-0",
        "1-0",
    ]
    assert [event[1] for event in events if event[0] == "comm_abort_done"] == ["1-0"]
    assert runtime._restart_required
    assert ("handle_destroy",) not in events


def test_quarantine_precedes_abort_and_retains_submitted_batches():
    runtime, _, _, _events = make_runtime()
    (pp_group,) = create_groups(runtime, [(0, 0)])
    pp_group.communicator.set_async_states(2)
    abort_snapshots = []
    original_abort = pp_group.communicator.abort

    def inspect_abort():
        abort_snapshots.append(
            (
                runtime._restart_required,
                runtime._handle_quarantined,
                bool(runtime._quarantined_batches),
                tuple(group.state.name for group in runtime.pp_groups),
            )
        )
        original_abort()

    pp_group.communicator.abort = inspect_abort
    batch = make_batch(runtime, pp_group)

    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        runtime.submit_model_update([batch])
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "status=2" in runtime._fail_stop_reason

    assert runtime._abort_done.wait(timeout=10)
    assert abort_snapshots == [(True, True, True, ("POISONED",))]
    assert runtime._quarantined_batches == (batch,)


def test_staging_uses_one_bounded_transfer_deadline(monkeypatch):
    now = [0.0]

    def monotonic():
        return now[0]

    def advance(duration):
        now[0] += duration

    monkeypatch.setattr(
        runtime_module,
        "time",
        SimpleNamespace(monotonic=monotonic, sleep=advance),
    )
    runtime, _, _, _ = make_runtime(
        transfer_timeout_s=1.0,
        poll_interval_s=0.6,
    )
    (pp_group,) = create_groups(runtime, [(0, 0)])
    pp_group.stream.set_query_results(False)
    batch = make_batch(runtime, pp_group)

    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        runtime.submit_model_update([batch])

    assert exc_info.value.operation == "stage"
    assert exc_info.value.phase == "completion"
    assert isinstance(exc_info.value.__cause__, TimeoutError)
    assert "model-version staging" in str(exc_info.value.__cause__)
    assert now[0] == pytest.approx(1.0)
    assert runtime._restart_required
    assert runtime._abort_done.wait(timeout=10)


def test_poisoned_state_rejects_new_runtime_operations():
    runtime, _, _, _ = make_runtime()
    (pp_group,) = create_groups(runtime, [(0, 0)])

    with runtime._state_cv:
        runtime._state = _RuntimeState.POISONED
        assert not runtime._restart_required

    with pytest.raises(RuntimeError, match="poisoned"):
        runtime.wait_for_pp_groups(
            (pp_group,),
            operation="poisoned wait",
        )

    with runtime._state_cv:
        runtime._state = _RuntimeState.OPEN
    runtime.close()


def test_quarantined_teardown_fails_before_waiting_for_dispatcher():
    runtime, _, _, events = make_runtime()
    (pp_group,) = create_groups(runtime, [(0, 0)])
    events.clear()
    errors = []
    thread = None

    def teardown_like():
        try:
            runtime.wait_for_pp_groups(
                (pp_group,),
                operation="quarantined teardown",
            )
        except BaseException as exc:  # noqa: BLE001 - thread failures are assertions.
            errors.append(exc)

    original_failure = RuntimeError("original transfer failure")
    runtime._enter_fail_stop((pp_group,), (), original_failure)
    runtime._dispatcher_lock.acquire()
    try:
        thread = threading.Thread(
            target=teardown_like,
            daemon=True,
            name="teardown",
        )
        thread.start()
        thread.join(timeout=0.5)
        assert not thread.is_alive()
    finally:
        runtime._dispatcher_lock.release()
        if thread is not None:
            thread.join(timeout=10)

    assert len(errors) == 1
    assert isinstance(errors[0], M2nCohortRestartRequired)
    assert errors[0].phase == "admission"
    assert errors[0].__cause__ is original_failure
    assert runtime._fail_stop_reason == "RuntimeError: original transfer failure"
    assert not any(event[0] == "comm_async" for event in events)
