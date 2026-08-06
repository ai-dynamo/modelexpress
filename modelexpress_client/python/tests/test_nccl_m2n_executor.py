# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-GPU version-staging tests for the current NCCL M2N executor."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from modelexpress.weight_transfer.planner.mesh import REPLICATE
from modelexpress.weight_transfer.transport.nccl_m2n_executor import (
    NcclM2nExecutor,
    ReshardParam,
)
from modelexpress.weight_transfer.transport.nccl_m2n_runtime import _M2nRuntime


class FakeStream:
    def __init__(self, events: list[tuple], *, fail_sync: bool = False) -> None:
        self.events = events
        self.fail_sync = fail_sync
        self.name = "lane-stream"

    def synchronize(self) -> None:
        self.events.append(("stream_sync",))
        if self.fail_sync:
            raise RuntimeError("injected stream-sync failure")


class FakeCuda:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def set_device(self, device: int) -> None:
        self.events.append(("set_device", device))

    def stream(self, stream: FakeStream):
        self.events.append(("stream_context", stream.name))
        return nullcontext()


class FakeComm:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def get_async_error(self) -> int:
        return 0

    def finalize(self) -> None:
        self.events.append(("comm_finalize",))

    def destroy(self) -> None:
        self.events.append(("comm_destroy",))


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


class FakeHandle:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    def destroy(self) -> None:
        self.events.append(("handle_destroy",))


class FakeM2n:
    Config = FakeConfig
    Mesh = FakeMesh
    Shard = FakeShard
    Replicate = FakeReplicate

    def __init__(
        self,
        events: list[tuple],
        payloads: list[torch.Tensor],
        *,
        fail_reshard_at: int | None = None,
    ) -> None:
        self.events = events
        self.payloads = payloads
        self.fail_reshard_at = fail_reshard_at
        self.calls: list[dict] = []
        self.handle = FakeHandle(events)

    def init(self, config: FakeConfig) -> FakeHandle:
        self.events.append(("m2n_init", config.max_cta))
        return self.handle

    def reshard(self, **kwargs) -> None:
        index = len(self.calls)
        self.calls.append(kwargs)
        self.events.append(("reshard", index, kwargs["stream"]))
        if index == self.fail_reshard_at:
            raise RuntimeError("injected reshard failure")
        if kwargs["dst"] is not None:
            kwargs["dst"].copy_(self.payloads[index])


class FakeNccl:
    def get_version(self) -> str:
        return "2.30.5"


def make_executor(
    payloads: list[torch.Tensor],
    *,
    rank: int = 1,
    fail_reshard_at: int | None = None,
    fail_stream_sync: bool = False,
):
    events: list[tuple] = []
    m2n = FakeM2n(events, payloads, fail_reshard_at=fail_reshard_at)
    runtime = _M2nRuntime(
        0,
        _m2n_module=m2n,
        _nccl_module=FakeNccl(),
        _torch_module=SimpleNamespace(cuda=FakeCuda(events)),
        _enforce_singleton=False,
    )
    stream = FakeStream(events, fail_sync=fail_stream_sync)
    lane = runtime.register_lane(
        lane_id="weights",
        key=(0, 0),
        communicator=FakeComm(events),
        nranks=2,
        comm_rank=rank,
        stream=stream,
    )
    executor = NcclM2nExecutor(runtime, lane, tp_src=1, tp_dst=1)
    return executor, runtime, lane, m2n, events


def make_params() -> list[ReshardParam]:
    return [
        ReshardParam(
            name="p0",
            global_shape=(4,),
            shard_dim=REPLICATE,
            local_tensor=torch.tensor([1, 2, 3, 4], dtype=torch.uint8),
        ),
        ReshardParam(
            name="p1",
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
    total_bytes, _ = executor.execute(params)

    assert total_bytes == 7
    assert torch.equal(params[0].local_tensor, payloads[0])
    assert torch.equal(params[1].local_tensor, payloads[1])
    second_reshard = max(i for i, event in enumerate(events) if event[0] == "reshard")
    first_live_copy = min(i for i, event in enumerate(events) if event[0] == "live_copy")
    assert second_reshard < first_live_copy
    assert all(call["handle"] is runtime.handle for call in m2n.calls)
    assert all("window" not in call for call in m2n.calls)
    executor.teardown()
    runtime.close()


def test_reshard_failure_leaves_live_version_unchanged_and_poisons_lane():
    payloads = [
        torch.tensor([10, 11, 12, 13], dtype=torch.uint8),
        torch.tensor([20, 21, 22], dtype=torch.uint8),
    ]
    executor, runtime, _, m2n, _ = make_executor(payloads, fail_reshard_at=1)
    params = make_params()
    originals = [param.local_tensor.clone() for param in params]

    with pytest.raises(RuntimeError, match="injected reshard failure"):
        executor.execute(params)

    assert all(
        torch.equal(param.local_tensor, original)
        for param, original in zip(params, originals, strict=True)
    )
    with pytest.raises(RuntimeError, match="poisoned"):
        executor.execute(params)
    assert len(m2n.calls) == 2
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
        executor.execute(params)

    calls = len(m2n.calls)
    with pytest.raises(RuntimeError, match="unusable after a failed model commit"):
        executor.execute(params)
    assert len(m2n.calls) == calls
    runtime.close()


def test_source_passes_live_tensors_on_one_explicit_lane_stream():
    executor, runtime, lane, m2n, _ = make_executor([], rank=0)
    params = make_params()

    executor.execute(params)

    assert all(
        call["src"] is param.local_tensor
        for call, param in zip(m2n.calls, params, strict=True)
    )
    assert all(call["dst"] is None for call in m2n.calls)
    assert all(call["stream"] is lane.stream for call in m2n.calls)
    executor.teardown()
    runtime.close()


def test_failed_stream_drain_retains_staging_and_disables_lane():
    payloads = [
        torch.tensor([10, 11, 12, 13], dtype=torch.uint8),
        torch.tensor([20, 21, 22], dtype=torch.uint8),
    ]
    executor, _, _, _, _ = make_executor(payloads, fail_stream_sync=True)
    params = make_params()

    with pytest.raises(RuntimeError, match="injected stream-sync failure"):
        executor.execute(params)
    assert executor._staged
    with pytest.raises(RuntimeError, match="stream could not be drained"):
        executor.execute(params)
