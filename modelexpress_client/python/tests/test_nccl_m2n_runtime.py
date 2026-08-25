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

from modelexpress.refit.reshard.transport.nccl_m2n.mesh import (
    REPLICATE,
    build_tp_meshes,
    tile_shape,
)
from modelexpress.refit.reshard.transport.nccl_m2n.runtime import (
    _M2nCall,
    _M2nPPGroupBatch,
    _M2nPPGroupSpec,
    _M2nRuntime,
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

    def wait_event(self, event: FakeEvent) -> None:
        self.events.append(("stream_wait_event", self.name, event.name))

    def synchronize(self) -> None:
        self.events.append(("stream_sync", self.name))

    def close(self) -> None:
        self.events.append(("stream_destroy", self.name))


class FakeCuda:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events
        self.stream_count = 0
        self.producer_stream = FakeStream("producer", events)

    def set_device(self, device: int) -> None:
        self.events.append(("set_device", device))

    def Stream(self, *, device: int) -> FakeStream:
        stream = FakeStream(f"owned-{self.stream_count}", self.events)
        self.stream_count += 1
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

    def get_async_error(self) -> int:
        return 0

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


class FakeNccl:
    UniqueId = FakeUniqueId

    def __init__(self, events: list[tuple]) -> None:
        self.events = events
        self.next_uid = b"new-uid"
        owner = self

        class Communicator:
            @staticmethod
            def init(nranks: int, rank: int, unique_id: FakeUniqueId) -> FakeComm:
                name = unique_id.value.decode()
                owner.events.append(("comm_init", name, nranks, rank))
                return FakeComm(name, owner.events)

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
    finalize_timeout_s: float = 300.0,
):
    events: list[tuple] = []
    m2n = FakeM2n(events, host_delay=host_delay, fail_at=fail_at)
    nccl = FakeNccl(events)
    runtime = _M2nRuntime(
        0,
        max_cta=8,
        finalize_timeout_s=finalize_timeout_s,
        _m2n_module=m2n,
        _nccl_module=nccl,
        _torch_module=SimpleNamespace(cuda=FakeCuda(events)),
        _enforce_singleton=False,
    )
    return runtime, m2n, nccl, events


def start_thread(target):
    results = []
    errors = []

    def run() -> None:
        try:
            results.append(target())
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
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


def test_submit_uses_one_group_and_canonical_pp_group_order():
    runtime, _, _, events = make_runtime()
    early, late = create_groups(runtime, [(1, 0), (0, 0)])

    runtime.submit_model_update([make_batch(runtime, late), make_batch(runtime, early)])

    pipeline = [
        event
        for event in events
        if event[0] in {"group_start", "reshard", "group_end", "stream_sync"}
    ]
    assert pipeline[:6] == [
        ("group_start",),
        ("reshard", "0-0", "owned-0"),
        ("reshard", "1-0", "owned-1"),
        ("group_end",),
        ("stream_sync", "owned-0"),
        ("stream_sync", "owned-1"),
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

    with pytest.raises(RuntimeError, match="injected reshard failure"):
        runtime.submit_model_update([make_batch(runtime, pp_group)])
    assert ("group_abort",) in events
    with pytest.raises(RuntimeError, match="poisoned"):
        runtime.submit_model_update([make_batch(runtime, pp_group)])
    runtime.close()


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


def test_close_waits_for_create_and_snapshots_complete_topology():
    runtime, _, nccl, events = make_runtime()
    init_entered = threading.Event()
    release_init = threading.Event()
    original_init = nccl.Communicator.init
    init_count = 0

    def blocking_init(nranks, rank, unique_id):
        nonlocal init_count
        init_count += 1
        if init_count == 2:
            init_entered.set()
            assert release_init.wait(timeout=10)
        return original_init(nranks, rank, unique_id)

    nccl.Communicator.init = staticmethod(blocking_init)
    create_thread, create_results, create_errors = start_thread(
        lambda: runtime.create_pp_groups(
            [make_spec((1, 0)), make_spec((0, 0))]
        )
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
    original_init = nccl.Communicator.init

    def blocking_init(nranks, rank, unique_id):
        init_entered.set()
        assert release_init.wait(timeout=10)
        return original_init(nranks, rank, unique_id)

    nccl.Communicator.init = staticmethod(blocking_init)
    create_thread, _, create_errors = start_thread(
        lambda: runtime.create_pp_groups([make_spec((0, 0))])
    )
    assert init_entered.wait(timeout=10)

    with pytest.raises(TimeoutError, match="process restart is required"):
        runtime.close()

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
    assert "poisoned" in str(create_errors[0])
    assert len(runtime.pp_groups) == 1

    with pytest.raises(RuntimeError, match="process restart is required"):
        runtime.close()
    with pytest.raises(RuntimeError, match="poisoned"):
        runtime.create_pp_groups([])
    assert not any(
        event[0]
        in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
        for event in events
    )


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
    assert ("comm_init", "0-0", 2, 1) in events
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
