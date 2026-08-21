# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-GPU staging tests for process-level NCCL M2N execution."""

from __future__ import annotations

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

    def wait_event(self, event: FakeEvent) -> None:
        self.events.append(("stream_wait_event", self.name, event.name))

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
        self.value = value

    @staticmethod
    def from_bytes(value: bytes) -> FakeUniqueId:
        return FakeUniqueId(value)


class FakeNccl:
    UniqueId = FakeUniqueId

    def __init__(self, events: list[tuple]) -> None:
        owner = self

        class Communicator:
            @staticmethod
            def init(nranks: int, rank: int, unique_id: FakeUniqueId) -> FakeComm:
                del nranks, rank
                return FakeComm(unique_id.value.decode(), owner.events)

        self.Communicator = Communicator
        self.events = events

    def get_version(self) -> str:
        return "2.30.5"

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
):
    events: list[tuple] = []
    m2n = FakeM2n(events, payloads, fail_at=fail_at)
    runtime = _M2nRuntime(
        0,
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
    runtime.close()


def test_commit_failure_poison_executor_until_reinitialized():
    payloads = [
        torch.tensor([10, 11, 12, 13], dtype=torch.uint8),
        torch.tensor([20, 21, 22], dtype=torch.uint8),
    ]
    executor, runtime, _, m2n, _ = make_executor(payloads)
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
    runtime.close()


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


def test_multi_pp_group_submission_uses_one_group_and_sorted_first_occurrence():
    executor, runtime, _, m2n, events = make_executor(
        [],
        rank=0,
        keys=((1, 0), (0, 0)),
    )
    # Same read-only source tensors may feed distinct M2N buckets by approved
    # contract. Executor retains them through every PP-stream synchronization.
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
    first_sync = min(
        index for index, event in enumerate(events) if event[0] == "stream_sync"
    )
    assert last_reshard < first_sync
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
    executor, _, _, _, _ = make_executor(
        payloads,
        fail_stream_sync=True,
    )
    params = make_params()

    with pytest.raises(RuntimeError, match="injected stream-sync failure"):
        executor.execute({(0, 0): params})
    assert executor._states[(0, 0)].staged
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
