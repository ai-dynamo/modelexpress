# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-GPU staging tests for process-level NCCL M2N execution."""

from __future__ import annotations

import gc
import inspect
import threading
import time
import weakref
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from types import SimpleNamespace

import modelexpress.refit.reshard.transport.nccl_m2n as nccl_m2n_public
import modelexpress.refit.reshard.transport.nccl_m2n.executor as executor_module
import modelexpress.refit.reshard.transport.nccl_m2n.runtime as runtime_module
import pytest
import torch
from modelexpress.refit.reshard.megatron_aliases import MegatronTensorSpec
from modelexpress.refit.reshard.transport.nccl_m2n.executor import (
    M2nPPGroupBootstrap,
    NcclM2nExecutor,
    ReshardParam,
    build_reshard_params,
)
from modelexpress.refit.reshard.transport.nccl_m2n.mesh import REPLICATE
from modelexpress.refit.reshard.transport.nccl_m2n.runtime import (
    M2nCohortRestartRequired,
    _M2nPPGroupSpec,
    _M2nRuntime,
    _RuntimeState,
)


class FakeEvent:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events
        self.name = "weights-ready"

    def record(self, stream: FakeStream) -> None:
        self.events.append(("event_record", stream.name))


class FakeStream:
    def __init__(
        self,
        name: str,
        events: list[tuple],
        *,
        fail_sync: bool = False,
    ) -> None:
        self.events = events
        self.fail_sync = fail_sync
        self.name = name
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
        if self.fail_sync:
            raise RuntimeError("injected stream-sync failure")

    def close(self) -> None:
        self.events.append(("stream_destroy", self.name))


class FakeCuda:
    def __init__(
        self,
        events: list[tuple],
        *,
        fail_stream_sync: bool = False,
    ) -> None:
        self.events = events
        self.fail_stream_sync = fail_stream_sync
        self.stream_count = 0
        self.streams: list[FakeStream] = []
        self.producer_stream = FakeStream("producer", events)

    def set_device(self, device: int) -> None:
        self.events.append(("set_device", device))

    def Stream(self, *, device: int) -> FakeStream:
        stream = FakeStream(
            f"pp-stream-{self.stream_count}",
            self.events,
            fail_sync=self.fail_stream_sync,
        )
        self.stream_count += 1
        self.streams.append(stream)
        if self.fail_stream_sync:
            stream.set_query_results(RuntimeError("injected stream-query failure"))
        self.events.append(("stream_create", stream.name, device))
        return stream

    def Event(self) -> FakeEvent:
        return FakeEvent(self.events)

    def current_stream(self, device: int) -> FakeStream:
        return self.producer_stream

    def stream(self, stream: FakeStream):
        self.events.append(("stream_context", stream.name))
        return nullcontext()


class FakeComm:
    def __init__(self, name: str, events: list[tuple]) -> None:
        self.name = name
        self.events = events
        self.finalized = False
        self.async_states: list[int | BaseException] = [0]

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
        self.events.append(("comm_abort_done", self.name))

    def finalize(self) -> None:
        self.events.append(("comm_finalize", self.name))
        self.finalized = True

    def destroy(self) -> None:
        assert self.finalized
        self.events.append(("comm_destroy", self.name))


class FakeUniqueId:
    def __init__(self, value: bytes) -> None:
        self.value = value

    @staticmethod
    def from_bytes(value: bytes) -> FakeUniqueId:
        return FakeUniqueId(value)


@dataclass(frozen=True)
class FakeNCCLConfig:
    blocking: bool | None = None


class FakeNccl:
    UniqueId = FakeUniqueId
    NCCLConfig = FakeNCCLConfig

    def __init__(self, events: list[tuple]) -> None:
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
                del nranks, rank
                self.name = unique_id.value.decode()
                owner.events.append(("comm_init", self.name, config.blocking))

        self.Communicator = Communicator
        self.events = events

    def get_version(self) -> SimpleNamespace:
        return SimpleNamespace(
            libnccl=SimpleNamespace(
                version=SimpleNamespace(release=(2, 30, 5)),
            )
        )

    def get_unique_id(self) -> FakeUniqueId:
        return FakeUniqueId(b"uid")


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
        payloads: list[torch.Tensor],
        *,
        fail_at: int | None = None,
    ) -> None:
        self.events = events
        self.payloads = payloads
        self.fail_at = fail_at
        self.calls: list[dict] = []

    def reshard(self, comm, src, dst, *, stream) -> None:
        index = len(self.calls)
        self.calls.append({"comm": comm, "src": src, "dst": dst, "stream": stream})
        self.events.append(("reshard", comm.name, index, stream.name))
        if index == self.fail_at:
            raise RuntimeError("injected reshard failure")
        if dst.buffer is not None:
            dst.buffer.copy_(self.payloads[index])

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
        payloads: list[torch.Tensor],
        *,
        fail_at: int | None = None,
    ) -> None:
        self.events = events
        self.handle = FakeHandle(events, payloads, fail_at=fail_at)

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


def make_spec(key: tuple[int, int], rank: int) -> _M2nPPGroupSpec:
    name = f"{key[0]}-{key[1]}"
    return _M2nPPGroupSpec(
        group_id=name,
        key=key,
        unique_id=name.encode(),
        source_size=1,
        destination_size=1,
        comm_rank=rank,
        device_id=0,
    )


def make_bootstrap(
    key: tuple[int, int] = (0, 0),
    *,
    rank: int = 0,
    unique_id: bytes | None = None,
) -> M2nPPGroupBootstrap:
    name = f"{key[0]}-{key[1]}"
    return M2nPPGroupBootstrap(
        group_id=name,
        key=key,
        unique_id=name.encode() if unique_id is None else unique_id,
        source_size=1,
        destination_size=1,
        comm_rank=rank,
    )


class FakeCudaProtocolTensor:
    shape = (4,)
    dtype = torch.uint8
    is_cuda = True

    def __init__(self, device_index: int) -> None:
        self.device = SimpleNamespace(type="cuda", index=device_index)

    def numel(self) -> int:
        return 4

    def element_size(self) -> int:
        return 1

    def is_contiguous(self) -> bool:
        return True

    def data_ptr(self) -> int:
        return 4096

    def copy_(self, _other) -> FakeCudaProtocolTensor:
        return self


def make_executor(
    payloads: list[torch.Tensor],
    *,
    rank: int = 1,
    keys: tuple[tuple[int, int], ...] = ((0, 0),),
    fail_at: int | None = None,
    fail_stream_sync: bool = False,
    transfer_timeout_s: float = 900.0,
    finalize_timeout_s: float = 300.0,
    poll_interval_s: float = 0.002,
):
    runtime, m2n, events = make_runtime(
        payloads,
        fail_at=fail_at,
        fail_stream_sync=fail_stream_sync,
        transfer_timeout_s=transfer_timeout_s,
        finalize_timeout_s=finalize_timeout_s,
        poll_interval_s=poll_interval_s,
    )
    groups = runtime.create_pp_groups([make_spec(key, rank) for key in keys])
    executor = NcclM2nExecutor._create_for_tests(runtime)
    return executor, runtime, groups, m2n, events


def make_runtime(
    payloads: list[torch.Tensor],
    *,
    fail_at: int | None = None,
    fail_stream_sync: bool = False,
    transfer_timeout_s: float = 900.0,
    finalize_timeout_s: float = 300.0,
    poll_interval_s: float = 0.002,
):
    events: list[tuple] = []
    m2n = FakeM2n(events, payloads, fail_at=fail_at)
    runtime = _M2nRuntime(
        0,
        transfer_timeout_s=transfer_timeout_s,
        finalize_timeout_s=finalize_timeout_s,
        _poll_interval_s=poll_interval_s,
        _m2n_module=m2n,
        _nccl_module=FakeNccl(events),
        _torch_module=SimpleNamespace(
            cuda=FakeCuda(events, fail_stream_sync=fail_stream_sync)
        ),
        _enforce_singleton=False,
    )
    return runtime, m2n, events


def complete_update(executor, updates):
    """Run the explicit serving-boundary lifecycle used by production callers."""
    update = executor.stage(updates)
    results = executor.apply(update)
    executor.release(update)
    return results


def start_thread(target):
    results = []
    errors = []

    def run() -> None:
        try:
            results.append(target())
        except BaseException as exc:  # noqa: BLE001 - thread failures are assertions.
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


def make_params(prefix: str = "p") -> list[ReshardParam]:
    return [
        ReshardParam(
            name=f"{prefix}0",
            global_shape=(4,),
            shard_dim=REPLICATE,
            local_tensor=torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
        ),
        ReshardParam(
            name=f"{prefix}1",
            global_shape=(3,),
            shard_dim=REPLICATE,
            local_tensor=torch.tensor([5, 6, 7], dtype=torch.uint8),
        ),
    ]


def test_public_exports_exclude_runtime_and_removed_one_shot_apis():
    assert nccl_m2n_public.__all__ == [
        "M2nCohortRestartRequired",
        "M2nPPGroupBootstrap",
        "M2nStagedUpdate",
        "NcclM2nExecutor",
        "ReshardParam",
        "build_reshard_params",
    ]
    assert not hasattr(nccl_m2n_public, "_M2nRuntime")
    assert not hasattr(NcclM2nExecutor, "execute")
    assert not hasattr(NcclM2nExecutor, "execute_batch")
    assert not hasattr(NcclM2nExecutor, "teardown")


def test_current_data_plane_source_excludes_old_binding_and_window_apis():
    source = inspect.getsource(executor_module) + inspect.getsource(runtime_module)
    for forbidden in (
        "_nccl_m2n_bind",
        "reshard_with_window",
        "window_register",
        "def run_reshard(",
        "def execute(",
        "def execute_batch(",
        "def teardown(",
    ):
        assert forbidden not in source


def test_public_factory_creates_and_owns_runtime(monkeypatch):
    runtime, _, events = make_runtime([])
    created: list[tuple[int, dict[str, object]]] = []

    def runtime_factory(device_id: int, **kwargs):
        created.append((device_id, kwargs))
        return runtime

    monkeypatch.setattr(executor_module, "_M2nRuntime", runtime_factory)
    executor = NcclM2nExecutor.create(0, [make_bootstrap()])

    assert created[0][0] == 0
    assert executor.pp_group_keys == ((0, 0),)
    assert executor._enforce_cuda_tensors
    assert runtime._attached_executors == {executor}

    executor.close()
    assert runtime._state is _RuntimeState.CLOSED
    assert ("handle_destroy",) in events


@pytest.mark.parametrize(
    ("failure", "expected_phase"),
    [
        ("missing-destroy", "handle_validation"),
        ("failing-destroy", "handle_cleanup"),
    ],
)
def test_public_factory_handle_failure_reports_full_intended_scope(
    monkeypatch,
    failure: str,
    expected_phase: str,
):
    events: list[tuple] = []
    m2n = FakeM2n(events, [])
    cleanup_error = RuntimeError("injected invalid-handle cleanup failure")
    if failure == "missing-destroy":
        m2n.handle = SimpleNamespace(reshard=lambda *_args, **_kwargs: None)
    else:

        def fail_destroy() -> None:
            raise cleanup_error

        m2n.handle = SimpleNamespace(destroy=fail_destroy)

    def runtime_factory(device_id: int, **kwargs):
        return _M2nRuntime(
            device_id,
            **kwargs,
            _m2n_module=m2n,
            _nccl_module=FakeNccl(events),
            _torch_module=SimpleNamespace(cuda=FakeCuda(events)),
            _enforce_singleton=False,
        )

    monkeypatch.setattr(executor_module, "_M2nRuntime", runtime_factory)
    bootstraps = [
        make_bootstrap((1, 0), unique_id=b"late"),
        make_bootstrap((0, 0), unique_id=b"early"),
    ]

    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        NcclM2nExecutor.create(0, bootstraps)

    error = exc_info.value
    assert error.operation == "create"
    assert error.phase == expected_phase
    assert error.group_ids == ("0-0", "1-0")
    assert error.pp_group_keys == ((0, 0), (1, 0))
    if failure == "missing-destroy":
        assert isinstance(error.__cause__, RuntimeError)
        assert "lacks current Handle.reshard()/destroy()" in str(error.__cause__)
    else:
        assert error.__cause__ is cleanup_error


def test_public_factory_recovers_when_invalid_handle_cleanup_succeeds(monkeypatch):
    events: list[tuple] = []
    invalid_m2n = FakeM2n(events, [])
    valid_m2n = FakeM2n(events, [])

    def destroy_invalid_handle() -> None:
        events.append(("invalid_handle_destroy",))

    invalid_m2n.handle = SimpleNamespace(destroy=destroy_invalid_handle)
    backends = iter((invalid_m2n, valid_m2n))

    def runtime_factory(device_id: int, **kwargs):
        return _M2nRuntime(
            device_id,
            **kwargs,
            _m2n_module=next(backends),
            _nccl_module=FakeNccl(events),
            _torch_module=SimpleNamespace(cuda=FakeCuda(events)),
            _enforce_singleton=True,
        )

    monkeypatch.setattr(executor_module, "_M2nRuntime", runtime_factory)

    with pytest.raises(RuntimeError, match="lacks current Handle.reshard"):
        NcclM2nExecutor.create(0, [make_bootstrap()])

    assert ("invalid_handle_destroy",) in events
    assert _M2nRuntime._live_runtime is None

    executor = NcclM2nExecutor.create(0, [make_bootstrap()])
    executor.close()
    assert _M2nRuntime._live_runtime is None


@pytest.mark.parametrize(
    ("device_id", "bootstraps", "error", "message"),
    [
        (False, [make_bootstrap()], TypeError, "device_id"),
        (-1, [make_bootstrap()], ValueError, "non-negative"),
        (0, [], ValueError, "at least one M2N PP group"),
        (0, [object()], TypeError, "M2nPPGroupBootstrap"),
        (0, [replace(make_bootstrap(), group_id="")], ValueError, "group_id"),
        (0, [replace(make_bootstrap(), unique_id=b"")], ValueError, "unique_id"),
        (0, [replace(make_bootstrap(), key=[0, 0])], ValueError, "key"),
        (0, [replace(make_bootstrap(), key=(0, -1))], ValueError, "key"),
        (
            0,
            [make_bootstrap(), make_bootstrap((0, 0), unique_id=b"other")],
            ValueError,
            "duplicate M2N PP group ID",
        ),
        (
            0,
            [
                make_bootstrap(),
                replace(
                    make_bootstrap((1, 0), unique_id=b"other"),
                    key=(0, 0),
                ),
            ],
            ValueError,
            "duplicate M2N PP group key",
        ),
        (
            0,
            [make_bootstrap(), make_bootstrap((1, 0), unique_id=b"0-0")],
            ValueError,
            "duplicate M2N PP group unique_id",
        ),
        (
            0,
            [replace(make_bootstrap(), source_size=True)],
            TypeError,
            "source_size",
        ),
        (
            0,
            [replace(make_bootstrap(), destination_size=0)],
            ValueError,
            "must be positive",
        ),
        (
            0,
            [replace(make_bootstrap(), comm_rank=2)],
            ValueError,
            "communicator rank",
        ),
    ],
)
def test_public_factory_rejects_invalid_bootstrap_before_native_create(
    device_id,
    bootstraps,
    error,
    message,
):
    with pytest.raises(error, match=message):
        NcclM2nExecutor.create(device_id, bootstraps)


def test_public_factory_snapshots_bootstraps_exactly_once(monkeypatch):
    runtime, _, _ = make_runtime([])

    class SnapshotOnce(Sequence):
        def __init__(self, values) -> None:
            self.values = tuple(values)
            self.iterations = 0

        def __len__(self) -> int:
            return len(self.values)

        def __getitem__(self, index):
            return self.values[index]

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("bootstrap sequence was iterated twice")
            return iter(self.values)

    bootstraps = SnapshotOnce([make_bootstrap()])
    monkeypatch.setattr(
        executor_module,
        "_M2nRuntime",
        lambda _device_id, **_kwargs: runtime,
    )

    executor = NcclM2nExecutor.create(0, bootstraps)
    assert bootstraps.iterations == 1
    executor.close()


def test_public_factory_rejects_wrong_cuda_device_before_submission(monkeypatch):
    runtime, m2n, _ = make_runtime([])
    monkeypatch.setattr(
        executor_module,
        "_M2nRuntime",
        lambda _device_id, **_kwargs: runtime,
    )
    executor = NcclM2nExecutor.create(0, [make_bootstrap()])
    param = ReshardParam(
        name="wrong-device",
        global_shape=(4,),
        shard_dim=REPLICATE,
        local_tensor=FakeCudaProtocolTensor(device_index=1),
    )

    with pytest.raises(ValueError, match="on CUDA device 1, expected 0"):
        executor.stage({(0, 0): [param]})
    assert not m2n.handle.calls
    executor.close()


def test_pending_update_requires_apply_or_discard_before_next_stage_and_close():
    executor, runtime, _, _, events = make_executor([], rank=0)
    update = executor.stage({(0, 0): make_params()})

    with pytest.raises(RuntimeError, match="release the current.*before staging"):
        executor.stage({(0, 0): make_params("next")})
    with pytest.raises(RuntimeError, match="release the current.*before closing"):
        executor.close()

    assert runtime._state is _RuntimeState.OPEN
    executor.release(update)
    executor.close()
    assert ("handle_destroy",) in events


def test_two_concurrent_stage_callers_publish_exactly_one_pending_update():
    executor, _, _, m2n, _ = make_executor([], rank=0)
    first_prepare_entered = threading.Event()
    release_first_prepare = threading.Event()
    original_prepare = executor._prepare_pp_group_batch

    def block_first_prepare(state, params):
        if not first_prepare_entered.is_set():
            first_prepare_entered.set()
            assert release_first_prepare.wait(timeout=10)
        return original_prepare(state, params)

    executor._prepare_pp_group_batch = block_first_prepare
    first_thread, first_results, first_errors = start_thread(
        lambda: executor.stage({(0, 0): make_params("first")})
    )
    assert first_prepare_entered.wait(timeout=10)
    second_thread, second_results, second_errors = start_thread(
        lambda: executor.stage({(0, 0): make_params("second")})
    )

    release_first_prepare.set()
    join_thread(first_thread)
    join_thread(second_thread)

    assert len(first_results) == 1
    assert not first_errors
    assert not second_results
    assert len(second_errors) == 1
    assert "release the current" in str(second_errors[0])
    assert executor._pending_update is first_results[0]
    assert len(m2n.handle.calls) == 2

    executor.release(first_results[0])
    executor.close()


def test_all_empty_update_uses_no_native_group_or_completion_poll():
    executor, _, _, m2n, events = make_executor([], rank=0)

    update = executor.stage({(0, 0): []})
    assert update.results[(0, 0)][0] == 0
    assert ("group_start",) not in events
    assert not m2n.handle.calls
    assert not any(event[0] == "stream_query" for event in events)

    assert executor.apply(update) == update.results
    executor.release(update)
    executor.close()


def test_destination_applies_only_after_complete_version_is_staged():
    payloads = [
        torch.tensor([10, 11, 12, 13], dtype=torch.uint8),
        torch.tensor([20, 21, 22], dtype=torch.uint8),
    ]
    executor, _runtime, _, m2n, events = make_executor(payloads)
    params = make_params()
    original_copy = executor._copy_into_live

    def logged_copy(param, staged) -> None:
        events.append(("live_copy", param.name))
        original_copy(param, staged)

    executor._copy_into_live = logged_copy
    update = executor.stage({(0, 0): params})

    # Staging is deliberately invisible at the serving boundary.
    assert torch.equal(
        params[0].local_tensor,
        torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
    )
    assert torch.equal(
        params[1].local_tensor,
        torch.tensor([5, 6, 7], dtype=torch.uint8),
    )

    results = executor.apply(update)

    assert results[(0, 0)][0] == 7
    assert torch.equal(params[0].local_tensor, payloads[0])
    assert torch.equal(params[1].local_tensor, payloads[1])
    group_end = events.index(("group_end",))
    first_live_copy = min(
        index for index, event in enumerate(events) if event[0] == "live_copy"
    )
    assert group_end < first_live_copy
    assert all(isinstance(call["src"], FakeDistTensor) for call in m2n.handle.calls)
    assert all(isinstance(call["dst"], FakeDistTensor) for call in m2n.handle.calls)
    assert not hasattr(executor, "execute_batch")
    with pytest.raises(ValueError, match="already applied"):
        executor.apply(update)
    executor.release(update)
    executor.close()


def test_two_versions_reset_tokens_and_reuse_destination_staging():
    first_payloads = [
        torch.tensor([10, 11, 12, 13], dtype=torch.uint8),
        torch.tensor([20, 21, 22], dtype=torch.uint8),
    ]
    second_payloads = [
        torch.tensor([30, 31, 32, 33], dtype=torch.uint8),
        torch.tensor([40, 41, 42], dtype=torch.uint8),
    ]
    executor, _runtime, _, _, _ = make_executor([*first_payloads, *second_payloads])
    params = make_params()

    first = executor.stage({(0, 0): params})
    first_staging = tuple(executor._states[(0, 0)].staged)
    executor.apply(first)
    assert all(
        torch.equal(param.local_tensor, expected)
        for param, expected in zip(params, first_payloads, strict=True)
    )
    executor.release(first)
    assert first._state.name == "RELEASED"
    assert first._ordered_updates == ()
    assert first._batches == ()
    assert executor._pending_update is None

    second = executor.stage({(0, 0): params})
    second_staging = tuple(executor._states[(0, 0)].staged)
    assert second is not first
    assert all(
        before is after
        for before, after in zip(first_staging, second_staging, strict=True)
    )
    assert all(
        torch.equal(param.local_tensor, expected)
        for param, expected in zip(params, first_payloads, strict=True)
    )
    executor.apply(second)
    assert all(
        torch.equal(param.local_tensor, expected)
        for param, expected in zip(params, second_payloads, strict=True)
    )
    executor.release(second)
    assert second._state.name == "RELEASED"
    assert second._ordered_updates == ()
    assert second._batches == ()
    assert executor._pending_update is None
    executor.close()


def test_reshard_failure_leaves_live_version_unchanged_and_poisons_executor():
    payloads = [
        torch.tensor([10, 11, 12, 13], dtype=torch.uint8),
        torch.tensor([20, 21, 22], dtype=torch.uint8),
    ]
    executor, runtime, _, m2n, events = make_executor(payloads, fail_at=1)
    params = make_params()
    originals = [param.local_tensor.clone() for param in params]

    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        complete_update(executor, {(0, 0): params})
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "injected reshard failure" in str(exc_info.value.__cause__)
    assert exc_info.value.pp_group_keys == ((0, 0),)

    assert all(
        torch.equal(param.local_tensor, original)
        for param, original in zip(params, originals, strict=True)
    )
    with pytest.raises(M2nCohortRestartRequired):
        complete_update(executor, {(0, 0): params})
    assert len(m2n.handle.calls) == 2
    assert ("group_abort",) in events
    assert runtime._abort_done.wait(timeout=10)
    assert ("comm_abort_start", "0-0", True) in events
    with pytest.raises(M2nCohortRestartRequired):
        executor.close()
    assert executor._states[(0, 0)].staged
    with pytest.raises(M2nCohortRestartRequired):
        runtime.close()
    assert ("handle_destroy",) not in events


def test_apply_failure_poisons_executor_until_reinitialized():
    payloads = [
        torch.tensor([10, 11, 12, 13], dtype=torch.uint8),
        torch.tensor([20, 21, 22], dtype=torch.uint8),
    ]
    executor, runtime, _, m2n, events = make_executor(payloads)
    params = make_params()
    copies = 0
    original_copy = executor._copy_into_live

    def fail_second_copy(param, staged) -> None:
        nonlocal copies
        copies += 1
        if copies == 2:
            raise RuntimeError("injected live-copy failure")
        original_copy(param, staged)

    executor._copy_into_live = fail_second_copy
    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        complete_update(executor, {(0, 0): params})
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "injected live-copy failure" in str(exc_info.value.__cause__)

    call_count = len(m2n.handle.calls)
    with pytest.raises(M2nCohortRestartRequired):
        complete_update(executor, {(0, 0): params})
    assert len(m2n.handle.calls) == call_count
    assert runtime._abort_done.wait(timeout=10)
    with pytest.raises(M2nCohortRestartRequired):
        runtime.close()
    assert ("handle_destroy",) not in events


def test_fatal_apply_rejects_release_and_retains_update_references():
    payloads = [
        torch.tensor([10, 11, 12, 13], dtype=torch.uint8),
        torch.tensor([20, 21, 22], dtype=torch.uint8),
    ]
    executor, runtime, _, _, _ = make_executor(payloads)
    params = make_params()
    update = executor.stage({(0, 0): params})
    retained_updates = update._ordered_updates
    retained_batches = update._batches

    def fail_copy(_param, _staged) -> None:
        raise ValueError("non-RuntimeError copy failure")

    executor._copy_into_live = fail_copy
    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        executor.apply(update)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "non-RuntimeError copy failure" in str(exc_info.value.__cause__)

    with pytest.raises(M2nCohortRestartRequired):
        executor.release(update)
    assert executor._pending_update is update
    assert update._ordered_updates is retained_updates
    assert update._batches is retained_batches
    assert runtime._abort_done.wait(timeout=10)


def test_source_uses_live_tensors_and_mx_owned_stream():
    executor, _, groups, m2n, events = make_executor([], rank=0)
    params = make_params()

    complete_update(executor, {(0, 0): params})

    assert all(
        call["src"].buffer is param.local_tensor
        for call, param in zip(m2n.handle.calls, params, strict=True)
    )
    assert all(call["dst"].buffer is None for call in m2n.handle.calls)
    assert all(call["stream"] is groups[0].stream for call in m2n.handle.calls)
    wait_index = next(
        index for index, event in enumerate(events) if event[0] == "stream_wait_event"
    )
    reshard_index = next(
        index for index, event in enumerate(events) if event[0] == "reshard"
    )
    assert wait_index < reshard_index
    executor.close()


def test_runtime_close_waits_for_source_stage_then_requires_executor_owner_close():
    executor, runtime, _, m2n, events = make_executor([], rank=0)
    params = make_params()
    prepare_entered = threading.Event()
    release_prepare = threading.Event()
    original_prepare = executor._prepare_pp_group_batch

    def blocking_prepare(state, update_params):
        prepare_entered.set()
        assert release_prepare.wait(timeout=10)
        return original_prepare(state, update_params)

    executor._prepare_pp_group_batch = blocking_prepare
    stage_thread, stage_results, stage_errors = start_thread(
        lambda: executor.stage({(0, 0): params})
    )
    assert prepare_entered.wait(timeout=10)

    close_thread, _, close_errors = start_thread(runtime.close)
    wait_for_runtime_state(runtime, _RuntimeState.CLOSING)
    assert_active_operations(runtime, 1)
    assert ("handle_destroy",) not in events

    release_prepare.set()
    join_thread(stage_thread)
    join_thread(close_thread)

    assert not stage_errors
    assert len(close_errors) == 1
    assert "owner mismatch" in str(close_errors[0])
    update = stage_results[0]
    assert update.results[(0, 0)][0] == 7
    assert len(m2n.handle.calls) == 2
    assert runtime._state is _RuntimeState.OPEN
    assert ("handle_destroy",) not in events

    executor.apply(update)
    executor.release(update)
    executor.close()
    assert events.index(("reshard", "0-0", 0, "pp-stream-0")) < events.index(
        ("handle_destroy",)
    )


def test_runtime_close_waits_for_destination_stage_then_requires_executor_owner_close():
    payloads = [
        torch.tensor([10, 11, 12, 13], dtype=torch.uint8),
        torch.tensor([20, 21, 22], dtype=torch.uint8),
    ]
    executor, runtime, groups, _, events = make_executor(payloads, rank=1)
    params = make_params()
    staging_ready = threading.Event()
    release_preparation = threading.Event()
    original_ensure_staging = executor._ensure_staging

    def blocking_ensure_staging(state, update_params):
        original_ensure_staging(state, update_params)
        staging_ready.set()
        assert release_preparation.wait(timeout=10)

    executor._ensure_staging = blocking_ensure_staging
    stage_thread, stage_results, stage_errors = start_thread(
        lambda: executor.stage({(0, 0): params})
    )
    assert staging_ready.wait(timeout=10)

    close_thread, _, close_errors = start_thread(runtime.close)
    wait_for_runtime_state(runtime, _RuntimeState.CLOSING)
    assert_active_operations(runtime, 1)
    assert groups[0].stream is not None
    assert executor._states[(0, 0)].staged
    assert ("handle_destroy",) not in events

    release_preparation.set()
    join_thread(stage_thread)
    join_thread(close_thread)

    assert not stage_errors
    assert len(close_errors) == 1
    assert "owner mismatch" in str(close_errors[0])
    assert runtime._state is _RuntimeState.OPEN
    assert ("handle_destroy",) not in events

    update = stage_results[0]
    executor.apply(update)
    executor.release(update)
    executor.close()
    assert ("handle_destroy",) in events


def test_concurrent_executor_close_fails_fast_while_first_close_finishes():
    executor, runtime, groups, _, events = make_executor([], rank=1)
    query_entered = threading.Event()
    release_query = threading.Event()
    original_query = groups[0].stream.query

    def blocking_query() -> bool:
        query_entered.set()
        assert release_query.wait(timeout=10)
        return original_query()

    groups[0].stream.query = blocking_query
    close_thread, _, close_errors = start_thread(executor.close)
    assert query_entered.wait(timeout=10)

    start = time.monotonic()
    with pytest.raises(RuntimeError, match="already closing"):
        executor.close()
    assert time.monotonic() - start < 1.0

    release_query.set()
    join_thread(close_thread)

    assert not close_errors
    assert executor._torn_down
    assert executor not in runtime._attached_executors
    assert ("handle_destroy",) in events
    assert events.index(("stream_query", "pp-stream-0", True)) < events.index(
        ("handle_destroy",)
    )


def test_close_rejects_attached_executor_before_native_mutation():
    executor, runtime, groups, _, events = make_executor([], rank=0)
    handle = runtime.handle
    stream = groups[0].stream

    with pytest.raises(RuntimeError, match="owner mismatch"):
        runtime.close()

    assert runtime._state is _RuntimeState.OPEN
    assert runtime.handle is handle
    assert groups[0].stream is stream
    assert runtime._attached_executors == {executor}
    assert not any(
        event[0]
        in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
        for event in events
    )

    executor.close()


def test_runtime_strong_reference_prevents_executor_gc_bypass():
    executor, runtime, _, _, events = make_executor([], rank=0)
    executor_ref = weakref.ref(executor)
    del executor
    gc.collect()

    retained = executor_ref()
    assert retained is not None
    assert retained in runtime._attached_executors
    with pytest.raises(RuntimeError, match="owner mismatch"):
        runtime.close()
    assert ("handle_destroy",) not in events

    retained.close()


def test_executor_close_starting_after_runtime_close_is_rejected_then_retry_succeeds():
    executor, runtime, _, _, events = make_executor([], rank=0)
    blocker_entered = threading.Event()
    release_blocker = threading.Event()

    def hold_admitted_operation() -> None:
        with runtime._active_operation():
            blocker_entered.set()
            assert release_blocker.wait(timeout=10)

    blocker_thread, _, blocker_errors = start_thread(hold_admitted_operation)
    assert blocker_entered.wait(timeout=10)

    close_thread, _, close_errors = start_thread(runtime.close)
    wait_for_runtime_state(runtime, _RuntimeState.CLOSING)
    with pytest.raises(RuntimeError, match="closing"):
        executor.close()
    assert executor in runtime._attached_executors
    assert ("handle_destroy",) not in events

    release_blocker.set()
    join_thread(blocker_thread)
    join_thread(close_thread)
    assert not blocker_errors
    assert len(close_errors) == 1
    assert "owner mismatch" in str(close_errors[0])
    assert runtime._state is _RuntimeState.OPEN

    executor.close()
    assert ("handle_destroy",) in events


def test_executor_close_is_idempotent_and_stage_is_rejected_afterward():
    executor, runtime, _, m2n, events = make_executor([], rank=0)

    executor.close()
    events_after_close = tuple(events)
    executor.close()
    assert tuple(events) == events_after_close
    assert executor._torn_down
    assert executor not in runtime._attached_executors

    with pytest.raises(RuntimeError, match="closed"):
        complete_update(executor, {(0, 0): make_params()})
    assert not m2n.handle.calls


def test_failed_executor_close_retains_attachment_and_staging():
    payloads = [
        torch.tensor([10, 11, 12, 13], dtype=torch.uint8),
        torch.tensor([20, 21, 22], dtype=torch.uint8),
    ]
    executor, runtime, groups, _, events = make_executor(payloads)
    complete_update(executor, {(0, 0): make_params()})
    staged = tuple(executor._states[(0, 0)].staged)
    groups[0].stream.set_query_results(RuntimeError("injected close stream failure"))

    with pytest.raises(M2nCohortRestartRequired):
        executor.close()

    assert runtime._abort_done.wait(timeout=10)
    assert tuple(executor._states[(0, 0)].staged) == staged
    assert not executor._torn_down
    assert executor in runtime._attached_executors
    assert ("comm_abort_start", "0-0", True) in events


def test_direct_executor_constructor_is_rejected_without_changing_attachment():
    executor, runtime, _, _, _ = make_executor([], rank=0)
    with pytest.raises(TypeError, match=r"use NcclM2nExecutor\.create\(\)"):
        NcclM2nExecutor(runtime)

    assert runtime._attached_executors == {executor}
    executor.close()


def test_executor_detach_failure_can_retry_without_false_torn_down(monkeypatch):
    payloads = [
        torch.tensor([10, 11, 12, 13], dtype=torch.uint8),
        torch.tensor([20, 21, 22], dtype=torch.uint8),
    ]
    executor, runtime, _, _, _ = make_executor(payloads)
    complete_update(executor, {(0, 0): make_params()})
    original_detach = runtime._detach_executor
    calls = 0

    def fail_once(candidate):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected executor detach failure")
        original_detach(candidate)

    monkeypatch.setattr(runtime, "_detach_executor", fail_once)
    with pytest.raises(RuntimeError, match="detach failure"):
        executor.close()

    assert not executor._torn_down
    assert executor in runtime._attached_executors
    assert executor._states[(0, 0)].staged

    executor.close()
    assert executor._torn_down
    assert executor not in runtime._attached_executors


def test_executor_close_retries_singleton_clear_without_native_replay(monkeypatch):
    executor, runtime, _, _, events = make_executor([], rank=0)
    original_clear = runtime._clear_singleton
    clear_calls = 0

    def fail_once() -> None:
        nonlocal clear_calls
        clear_calls += 1
        if clear_calls == 1:
            raise RuntimeError("injected singleton clear failure")
        original_clear()

    monkeypatch.setattr(runtime, "_clear_singleton", fail_once)
    with pytest.raises(RuntimeError, match="singleton clear failure"):
        executor.close()

    native_events = tuple(
        event
        for event in events
        if event[0]
        in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
    )
    assert runtime._state is _RuntimeState.CLOSED
    assert not executor._torn_down
    assert runtime._attached_executors == {executor}

    executor.close()
    assert clear_calls == 2
    assert executor._torn_down
    assert not runtime._attached_executors
    assert (
        tuple(
            event
            for event in events
            if event[0]
            in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
        )
        == native_events
    )


def test_close_timeout_during_executor_preparation_requires_restart():
    executor, runtime, groups, m2n, events = make_executor(
        [],
        rank=0,
        finalize_timeout_s=0.01,
    )
    prepare_entered = threading.Event()
    release_prepare = threading.Event()
    original_prepare = executor._prepare_pp_group_batch

    def blocking_prepare(state, update_params):
        prepare_entered.set()
        assert release_prepare.wait(timeout=10)
        return original_prepare(state, update_params)

    executor._prepare_pp_group_batch = blocking_prepare
    stage_thread, _, stage_errors = start_thread(
        lambda: complete_update(executor, {(0, 0): make_params()})
    )
    assert prepare_entered.wait(timeout=10)

    with pytest.raises(M2nCohortRestartRequired):
        executor.close()

    assert runtime._state is _RuntimeState.POISONED
    assert runtime._restart_required
    assert runtime._handle_quarantined
    assert runtime.handle is m2n.handle
    assert groups[0].stream is not None
    assert groups[0].communicator is runtime.pp_groups[0].communicator
    assert runtime._attached_executors == {executor}
    assert not m2n.handle.calls
    assert not any(
        event[0]
        in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
        for event in events
    )

    release_prepare.set()
    join_thread(stage_thread)
    assert len(stage_errors) == 1
    assert isinstance(stage_errors[0], M2nCohortRestartRequired)
    with pytest.raises(M2nCohortRestartRequired):
        executor.close()
    with pytest.raises(M2nCohortRestartRequired):
        runtime.close()


def test_executor_close_timeout_preserves_failure_recorded_while_waiting():
    executor, runtime, groups, _, _ = make_executor(
        [],
        rank=0,
        finalize_timeout_s=0.05,
    )
    preparation_entered = threading.Event()
    close_waiting = threading.Event()
    record_failure = threading.Event()
    failure_recorded = threading.Event()
    release_preparation = threading.Event()
    original_failure = RuntimeError("original native executor-operation failure")
    original_prepare = executor._prepare_pp_group_batch

    class ObservedLock:
        def __init__(self) -> None:
            self._lock = threading.Lock()

        def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
            if timeout != -1:
                close_waiting.set()
            return self._lock.acquire(blocking, timeout)

        def release(self) -> None:
            self._lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            del exc_type, exc_value, traceback
            self.release()

    executor._execute_lock = ObservedLock()

    def blocking_prepare(state, update_params):
        preparation_entered.set()
        assert record_failure.wait(timeout=10)
        runtime._enter_fail_stop(groups, (), original_failure)
        failure_recorded.set()
        assert release_preparation.wait(timeout=10)
        return original_prepare(state, update_params)

    executor._prepare_pp_group_batch = blocking_prepare
    stage_thread, _, stage_errors = start_thread(
        lambda: complete_update(executor, {(0, 0): make_params()})
    )
    assert preparation_entered.wait(timeout=10)
    close_thread, _, close_errors = start_thread(executor.close)
    assert close_waiting.wait(timeout=10)

    record_failure.set()
    assert failure_recorded.wait(timeout=10)
    join_thread(close_thread)

    assert len(close_errors) == 1
    error = close_errors[0]
    assert isinstance(error, M2nCohortRestartRequired)
    assert error.operation == "close"
    assert error.phase == "executor_operation_drain"
    assert error.reason == ("RuntimeError: original native executor-operation failure")
    assert error.__cause__ is original_failure
    assert runtime._fail_stop_cause is original_failure
    assert runtime._fail_stop_reason == error.reason
    assert runtime._handle_quarantined

    release_preparation.set()
    join_thread(stage_thread)
    assert len(stage_errors) == 1
    assert isinstance(stage_errors[0], M2nCohortRestartRequired)


def test_stage_preparation_exception_releases_runtime_operation():
    executor, _, _, _, events = make_executor([], rank=0)

    def fail_prepare(state, update_params):
        del state, update_params
        raise RuntimeError("injected preparation failure")

    executor._prepare_pp_group_batch = fail_prepare
    with pytest.raises(RuntimeError, match="injected preparation failure"):
        complete_update(executor, {(0, 0): make_params()})

    executor.close()
    assert ("handle_destroy",) in events


def test_multi_pp_group_submission_uses_one_group_and_sorted_first_occurrence():
    executor, _, _, m2n, events = make_executor(
        [],
        rank=0,
        keys=((1, 0), (0, 0)),
    )
    # Same read-only source tensors may feed distinct M2N buckets by approved
    # contract. Executor retains them through every PP-stream completion.
    shared = make_params()

    complete_update(executor, {(1, 0): shared, (0, 0): shared})
    first_bucket = [call for call in m2n.handle.calls if call["comm"].name == "0-0"]
    second_bucket = [call for call in m2n.handle.calls if call["comm"].name == "1-0"]
    assert len(first_bucket) == len(second_bucket)
    assert all(
        left["src"].buffer is right["src"].buffer
        for left, right in zip(first_bucket, second_bucket, strict=True)
    )

    assert [call["comm"].name for call in m2n.handle.calls] == [
        "0-0",
        "0-0",
        "1-0",
        "1-0",
    ]
    assert events.count(("group_start",)) == 1
    assert events.count(("group_end",)) == 1
    last_reshard = max(
        index for index, event in enumerate(events) if event[0] == "reshard"
    )
    first_query = min(
        index for index, event in enumerate(events) if event[0] == "stream_query"
    )
    assert last_reshard < first_query
    executor.close()


def test_destination_overlap_across_pp_groups_is_rejected():
    executor, _, _, m2n, _ = make_executor(
        [],
        rank=1,
        keys=((0, 0), (1, 0)),
    )
    shared = make_params()

    with pytest.raises(ValueError, match="destination storage overlap"):
        complete_update(executor, {(0, 0): shared, (1, 0): shared})
    assert not m2n.handle.calls
    executor.close()


def test_overlap_within_one_pp_group_is_rejected():
    executor, _, _, m2n, _ = make_executor([], rank=0)
    storage = torch.arange(6, dtype=torch.uint8)
    params = [
        ReshardParam("left", (4,), REPLICATE, storage[:4]),
        ReshardParam("right", (4,), REPLICATE, storage[2:]),
    ]

    with pytest.raises(ValueError, match="within PP group"):
        complete_update(executor, {(0, 0): params})
    assert not m2n.handle.calls
    executor.close()


def test_shard_index_must_match_pp_group_communicator_rank():
    executor, _, _, m2n, _ = make_executor([], rank=0)
    params = [
        ReshardParam(
            name="sharded",
            global_shape=(4,),
            shard_dim=0,
            local_tensor=torch.ones(4),
            local_shard_index=1,
        )
    ]

    with pytest.raises(ValueError, match="communicator shard index 0"):
        complete_update(executor, {(0, 0): params})
    assert not m2n.handle.calls
    executor.close()


def test_failed_stream_drain_retains_staging_and_disables_executor():
    payloads = [
        torch.tensor([10, 11, 12, 13], dtype=torch.uint8),
        torch.tensor([20, 21, 22], dtype=torch.uint8),
    ]
    executor, runtime, _, _, events = make_executor(
        payloads,
        fail_stream_sync=True,
    )
    params = make_params()

    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        complete_update(executor, {(0, 0): params})
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "injected stream-query failure" in str(exc_info.value.__cause__)
    assert executor._states[(0, 0)].staged
    assert runtime._abort_done.wait(timeout=10)
    assert ("comm_abort_start", "0-0", True) in events
    with pytest.raises(M2nCohortRestartRequired):
        complete_update(executor, {(0, 0): params})


def test_build_reshard_params_translates_megatron_specs():
    replicated = MegatronTensorSpec(
        name="norm",
        tensor=torch.ones(4),
        role="replicated",
        hf_names=("norm",),
        global_shape=(4,),
        placement_kind="REPLICATE",
        shard_axis=None,
        local_shard_range=None,
    )
    sharded = MegatronTensorSpec(
        name="linear",
        tensor=torch.ones(2, 3),
        role="column",
        hf_names=("linear",),
        global_shape=(4, 3),
        placement_kind="SHARD",
        shard_axis=0,
        local_shard_range=(0, 2),
    )

    params = build_reshard_params([replicated, sharded])

    assert [(param.name, param.shard_dim) for param in params] == [
        ("norm", REPLICATE),
        ("linear", 0),
    ]
    assert params[0].local_tensor is replicated.tensor
    assert params[1].global_shape == (4, 3)
    assert params[1].local_shard_index == 0


def test_async_nccl_error_preserves_live_version_and_quarantines_staging():
    payloads = [
        torch.tensor([10, 11, 12, 13], dtype=torch.uint8),
        torch.tensor([20, 21, 22], dtype=torch.uint8),
    ]
    executor, runtime, groups, _, events = make_executor(payloads)
    params = make_params()
    originals = [param.local_tensor.clone() for param in params]
    groups[0].communicator.set_async_states(2)

    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        complete_update(executor, {(0, 0): params})
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "status=2" in str(exc_info.value.__cause__)

    assert all(
        torch.equal(param.local_tensor, original)
        for param, original in zip(params, originals, strict=True)
    )
    assert runtime._abort_done.wait(timeout=10)
    assert runtime._quarantined_batches
    batch = runtime._quarantined_batches[0]
    staged = executor._states[(0, 0)].staged
    assert [call.dst.buffer for call in batch.calls] == staged
    assert ("comm_abort_start", "0-0", True) in events
    with pytest.raises(M2nCohortRestartRequired):
        executor.close()
    assert executor._states[(0, 0)].staged
    with pytest.raises(M2nCohortRestartRequired):
        runtime.close()


def test_transfer_timeout_preserves_live_version_and_returns_before_abort_wait():
    payloads = [
        torch.tensor([10, 11, 12, 13], dtype=torch.uint8),
        torch.tensor([20, 21, 22], dtype=torch.uint8),
    ]
    executor, runtime, groups, _, events = make_executor(
        payloads,
        transfer_timeout_s=0.005,
        poll_interval_s=0.001,
    )
    params = make_params()
    originals = [param.local_tensor.clone() for param in params]
    groups[0].stream.set_query_results(False)

    with pytest.raises(M2nCohortRestartRequired) as exc_info:
        complete_update(executor, {(0, 0): params})
    assert isinstance(exc_info.value.__cause__, TimeoutError)
    assert exc_info.value.phase == "completion"

    assert all(
        torch.equal(param.local_tensor, original)
        for param, original in zip(params, originals, strict=True)
    )
    assert runtime._abort_done.wait(timeout=10)
    assert runtime._quarantined_batches
    assert executor._states[(0, 0)].staged
    assert ("handle_destroy",) not in events
    with pytest.raises(M2nCohortRestartRequired):
        runtime.close()
