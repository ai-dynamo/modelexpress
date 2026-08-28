# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-GPU staging tests for process-level NCCL M2N execution."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from modelexpress.refit.reshard.megatron_aliases import MegatronTensorSpec
from modelexpress.refit.reshard.transport.nccl_m2n.executor import (
    NcclM2nExecutor,
    ReshardParam,
    build_reshard_params,
)
from modelexpress.refit.reshard.transport.nccl_m2n.mesh import REPLICATE
from modelexpress.refit.reshard.transport.nccl_m2n.runtime import (
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
    groups = runtime.create_pp_groups([make_spec(key, rank) for key in keys])
    executor = NcclM2nExecutor(runtime)
    return executor, runtime, groups, m2n, events


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


def test_destination_commits_only_after_complete_version_is_staged():
    payloads = [
        torch.tensor([10, 11, 12, 13], dtype=torch.uint8),
        torch.tensor([20, 21, 22], dtype=torch.uint8),
    ]
    executor, runtime, _, m2n, events = make_executor(payloads)
    params = make_params()
    original_copy = executor._copy_into_live

    def logged_copy(param, staged) -> None:
        events.append(("live_copy", param.name))
        original_copy(param, staged)

    executor._copy_into_live = logged_copy
    results = executor.execute({(0, 0): params})

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
    executor.teardown()
    runtime.close()


def test_reshard_failure_leaves_live_version_unchanged_and_poisons_executor():
    payloads = [
        torch.tensor([10, 11, 12, 13], dtype=torch.uint8),
        torch.tensor([20, 21, 22], dtype=torch.uint8),
    ]
    executor, runtime, _, m2n, events = make_executor(payloads, fail_at=1)
    params = make_params()
    originals = [param.local_tensor.clone() for param in params]

    with pytest.raises(RuntimeError, match="injected reshard failure"):
        executor.execute({(0, 0): params})

    assert all(
        torch.equal(param.local_tensor, original)
        for param, original in zip(params, originals, strict=True)
    )
    with pytest.raises(RuntimeError, match="unusable"):
        executor.execute({(0, 0): params})
    assert len(m2n.handle.calls) == 2
    assert ("group_abort",) in events
    assert runtime._abort_done.wait(timeout=10)
    assert ("comm_abort_start", "0-0", True) in events
    with pytest.raises(RuntimeError, match="process restart is required"):
        executor.teardown()
    assert executor._states[(0, 0)].staged
    with pytest.raises(RuntimeError, match="process restart is required"):
        runtime.close()
    assert ("handle_destroy",) not in events


def test_commit_failure_poison_executor_until_reinitialized():
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
    with pytest.raises(RuntimeError, match="serving must remain stopped"):
        executor.execute({(0, 0): params})

    call_count = len(m2n.handle.calls)
    with pytest.raises(RuntimeError, match="unusable"):
        executor.execute({(0, 0): params})
    assert len(m2n.handle.calls) == call_count
    assert runtime._abort_done.wait(timeout=10)
    with pytest.raises(RuntimeError, match="process restart is required"):
        runtime.close()
    assert ("handle_destroy",) not in events


def test_source_uses_live_tensors_and_mx_owned_stream():
    executor, runtime, groups, m2n, events = make_executor([], rank=0)
    params = make_params()

    executor.execute({(0, 0): params})

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
    executor.teardown()
    runtime.close()


def test_close_waits_for_whole_source_execute_and_allows_nested_submit():
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
    execute_thread, execute_results, execute_errors = start_thread(
        lambda: executor.execute({(0, 0): params})
    )
    assert prepare_entered.wait(timeout=10)

    close_thread, _, close_errors = start_thread(runtime.close)
    wait_for_runtime_state(runtime, _RuntimeState.CLOSING)
    assert_active_operations(runtime, 1)
    assert ("handle_destroy",) not in events

    release_prepare.set()
    join_thread(execute_thread)
    join_thread(close_thread)

    assert not execute_errors
    assert not close_errors
    assert execute_results[0][(0, 0)][0] == 7
    assert len(m2n.handle.calls) == 2
    assert events.index(("reshard", "0-0", 0, "pp-stream-0")) < events.index(
        ("handle_destroy",)
    )


def test_close_waits_after_destination_stream_context_exits():
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
    execute_thread, _, execute_errors = start_thread(
        lambda: executor.execute({(0, 0): params})
    )
    assert staging_ready.wait(timeout=10)

    close_thread, _, close_errors = start_thread(runtime.close)
    wait_for_runtime_state(runtime, _RuntimeState.CLOSING)
    assert_active_operations(runtime, 1)
    assert groups[0].stream is not None
    assert executor._states[(0, 0)].staged
    assert ("handle_destroy",) not in events

    release_preparation.set()
    join_thread(execute_thread)
    join_thread(close_thread)

    assert not execute_errors
    assert not close_errors
    assert ("handle_destroy",) in events


def test_close_waits_for_executor_teardown():
    executor, runtime, groups, _, events = make_executor([], rank=1)
    query_entered = threading.Event()
    release_query = threading.Event()
    original_query = groups[0].stream.query
    query_count = 0

    def blocking_query() -> bool:
        nonlocal query_count
        query_count += 1
        if query_count == 1:
            query_entered.set()
            assert release_query.wait(timeout=10)
        return original_query()

    groups[0].stream.query = blocking_query
    teardown_thread, _, teardown_errors = start_thread(executor.teardown)
    assert query_entered.wait(timeout=10)

    close_thread, _, close_errors = start_thread(runtime.close)
    wait_for_runtime_state(runtime, _RuntimeState.CLOSING)
    assert_active_operations(runtime, 1)
    assert ("handle_destroy",) not in events

    release_query.set()
    join_thread(teardown_thread)
    join_thread(close_thread)

    assert not teardown_errors
    assert not close_errors
    assert query_count == 2
    assert ("handle_destroy",) in events
    assert max(
        index for index, event in enumerate(events) if event[0] == "stream_query"
    ) < events.index(("handle_destroy",))


def test_close_timeout_during_executor_preparation_requires_restart():
    executor, runtime, _, m2n, events = make_executor(
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
    execute_thread, _, execute_errors = start_thread(
        lambda: executor.execute({(0, 0): make_params()})
    )
    assert prepare_entered.wait(timeout=10)

    with pytest.raises(TimeoutError, match="process restart is required"):
        runtime.close()

    assert runtime._state is _RuntimeState.POISONED
    assert runtime._close_abandoned
    assert not m2n.handle.calls
    assert not any(
        event[0]
        in {"handle_destroy", "stream_destroy", "comm_finalize", "comm_destroy"}
        for event in events
    )

    release_prepare.set()
    join_thread(execute_thread)
    assert len(execute_errors) == 1
    assert "poisoned" in str(execute_errors[0])
    with pytest.raises(RuntimeError, match="process restart is required"):
        executor.teardown()
    with pytest.raises(RuntimeError, match="process restart is required"):
        runtime.close()


def test_execute_preparation_exception_releases_runtime_operation():
    executor, runtime, _, _, events = make_executor([], rank=0)

    def fail_prepare(state, update_params):
        del state, update_params
        raise RuntimeError("injected preparation failure")

    executor._prepare_pp_group_batch = fail_prepare
    with pytest.raises(RuntimeError, match="injected preparation failure"):
        executor.execute({(0, 0): make_params()})

    runtime.close()
    assert ("handle_destroy",) in events


def test_multi_pp_group_submission_uses_one_group_and_sorted_first_occurrence():
    executor, runtime, _, m2n, events = make_executor(
        [],
        rank=0,
        keys=((1, 0), (0, 0)),
    )
    # Same read-only source tensors may feed distinct M2N buckets by approved
    # contract. Executor retains them through every PP-stream completion.
    shared = make_params()

    executor.execute({(1, 0): shared, (0, 0): shared})

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
    executor.teardown()
    runtime.close()


def test_destination_overlap_across_pp_groups_is_rejected():
    executor, runtime, _, m2n, _ = make_executor(
        [],
        rank=1,
        keys=((0, 0), (1, 0)),
    )
    shared = make_params()

    with pytest.raises(ValueError, match="destination storage overlap"):
        executor.execute({(0, 0): shared, (1, 0): shared})
    assert not m2n.handle.calls
    runtime.close()


def test_overlap_within_one_pp_group_is_rejected():
    executor, runtime, _, m2n, _ = make_executor([], rank=0)
    storage = torch.arange(6, dtype=torch.uint8)
    params = [
        ReshardParam("left", (4,), REPLICATE, storage[:4]),
        ReshardParam("right", (4,), REPLICATE, storage[2:]),
    ]

    with pytest.raises(ValueError, match="within PP group"):
        executor.execute({(0, 0): params})
    assert not m2n.handle.calls
    runtime.close()


def test_shard_index_must_match_pp_group_communicator_rank():
    executor, runtime, _, m2n, _ = make_executor([], rank=0)
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
        executor.execute({(0, 0): params})
    assert not m2n.handle.calls
    runtime.close()


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

    with pytest.raises(RuntimeError, match="CUDA stream query failed"):
        executor.execute({(0, 0): params})
    assert executor._states[(0, 0)].staged
    assert runtime._abort_done.wait(timeout=10)
    assert ("comm_abort_start", "0-0", True) in events
    with pytest.raises(RuntimeError, match="unusable"):
        executor.execute({(0, 0): params})


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

    with pytest.raises(RuntimeError, match="status=2"):
        executor.execute({(0, 0): params})

    assert all(
        torch.equal(param.local_tensor, original)
        for param, original in zip(params, originals, strict=True)
    )
    assert runtime._abort_done.wait(timeout=10)
    assert runtime._quarantined_batches
    batch = runtime._quarantined_batches[0]
    staged = executor._states[(0, 0)].staged
    assert [call.dst.buffer for call in batch.calls] == staged
    assert batch.commit is not None
    assert ("comm_abort_start", "0-0", True) in events
    with pytest.raises(RuntimeError, match="process restart is required"):
        executor.teardown()
    assert executor._states[(0, 0)].staged
    with pytest.raises(RuntimeError, match="process restart is required"):
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

    with pytest.raises(TimeoutError, match="model-version staging"):
        executor.execute({(0, 0): params})

    assert all(
        torch.equal(param.local_tensor, original)
        for param, original in zip(params, originals, strict=True)
    )
    assert runtime._abort_done.wait(timeout=10)
    assert runtime._quarantined_batches
    assert executor._states[(0, 0)].staged
    assert ("handle_destroy",) not in events
    with pytest.raises(RuntimeError, match="process restart is required"):
        runtime.close()
