# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-GPU tests for deterministic process-level NCCL M2N dispatch."""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from modelexpress.weight_transfer.planner.mesh import REPLICATE, build_tp_meshes, tile_shape
from modelexpress.weight_transfer.transport.nccl_m2n_runtime import (
    _M2nCall,
    _M2nLaneBatch,
    _M2nLaneSpec,
    _M2nRuntime,
)


class FakeStream:
    def __init__(self, name: str, events: list[tuple]) -> None:
        self.name = name
        self.events = events

    def synchronize(self) -> None:
        self.events.append(("stream_sync", self.name))

    def close(self) -> None:
        self.events.append(("stream_destroy", self.name))


class FakeCuda:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events
        self.stream_count = 0

    def set_device(self, device: int) -> None:
        self.events.append(("set_device", device))

    def Stream(self, *, device: int) -> FakeStream:
        stream = FakeStream(f"owned-{self.stream_count}", self.events)
        self.stream_count += 1
        self.events.append(("stream_create", stream.name, device))
        return stream

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
    def from_bytes(value: bytes) -> "FakeUniqueId":
        return FakeUniqueId(value)

    def __bytes__(self) -> bytes:
        return self.value


class FakeNccl:
    UniqueId = FakeUniqueId

    def __init__(self, events: list[tuple]) -> None:
        self.events = events
        self.next_uid = b"u" * 128
        owner = self

        class Communicator:
            @staticmethod
            def init(nranks: int, rank: int, unique_id: FakeUniqueId) -> FakeComm:
                owner.events.append(("comm_init", nranks, rank, unique_id.value))
                return FakeComm(f"created-{rank}", owner.events)

        self.Communicator = Communicator

    def get_version(self) -> str:
        return "2.30.5"

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

    def __init__(self, events: list[tuple], *, host_delay: float = 0.0) -> None:
        self.events = events
        self.host_delay = host_delay
        self.handle = FakeHandle(events)
        self.calls: list[dict] = []

    def init(self, config: FakeConfig) -> FakeHandle:
        self.events.append(("m2n_init", config.max_cta))
        return self.handle

    def reshard(self, **kwargs) -> None:
        self.calls.append(kwargs)
        self.events.append(
            ("reshard", kwargs["comm"].name, kwargs["stream"].name)
        )
        if self.host_delay:
            time.sleep(self.host_delay)


def make_runtime(*, host_delay: float = 0.0):
    events: list[tuple] = []
    m2n = FakeM2n(events, host_delay=host_delay)
    nccl = FakeNccl(events)
    runtime = _M2nRuntime(
        0,
        max_cta=8,
        _m2n_module=m2n,
        _nccl_module=nccl,
        _torch_module=SimpleNamespace(cuda=FakeCuda(events)),
        _enforce_singleton=False,
    )
    return runtime, m2n, nccl, events


def register_lane(
    runtime: _M2nRuntime,
    events: list[tuple],
    key: tuple[int, int],
    *,
    rank: int = 0,
    runtime_owned_stream: bool = False,
):
    name = f"{key[0]}-{key[1]}"
    lane = runtime.register_lane(
        lane_id=name,
        key=key,
        communicator=FakeComm(name, events),
        nranks=2,
        comm_rank=rank,
        stream=(
            None if runtime_owned_stream else FakeStream(f"stream-{name}", events)
        ),
    )
    return lane


def make_call(shard_dim: int = 0) -> _M2nCall:
    src_mesh, dst_mesh = build_tp_meshes(shard_dim, 1, 1)
    return _M2nCall(
        src=1,
        dst=None,
        src_mesh=src_mesh,
        dst_mesh=dst_mesh,
        src_local_shape=tile_shape((4,), src_mesh),
        dst_local_shape=tile_shape((4,), dst_mesh),
        dtype="float32",
    )


def make_batch(lane) -> _M2nLaneBatch:
    return _M2nLaneBatch(lane=lane, calls=(make_call(),), total_bytes=16)


def test_dispatch_sorts_lanes_and_enqueues_all_before_any_stream_wait():
    runtime, _, _, events = make_runtime()
    late = register_lane(runtime, events, (1, 0))
    early = register_lane(runtime, events, (0, 0))

    runtime.dispatch_batch([make_batch(late), make_batch(early)])

    pipeline = [event for event in events if event[0] in ("reshard", "stream_sync")]
    assert pipeline[:4] == [
        ("reshard", "0-0", "stream-0-0"),
        ("reshard", "1-0", "stream-1-0"),
        ("stream_sync", "stream-0-0"),
        ("stream_sync", "stream-1-0"),
    ]
    runtime.close()


def test_two_caller_threads_cannot_change_recorded_lane_order():
    runtime, _, _, events = make_runtime(host_delay=0.01)
    late = register_lane(runtime, events, (1, 0))
    early = register_lane(runtime, events, (0, 0))
    barrier = threading.Barrier(3)

    def submit() -> None:
        barrier.wait()
        runtime.dispatch_batch([make_batch(late), make_batch(early)])

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


def test_independent_partial_lane_submission_is_rejected():
    runtime, _, _, events = make_runtime()
    early = register_lane(runtime, events, (0, 0))
    register_lane(runtime, events, (1, 0))

    with pytest.raises(RuntimeError, match="every locally registered lane"):
        runtime.dispatch_batch([make_batch(early)])
    runtime.close()


def test_two_multi_lane_processes_record_the_same_sorted_sequence():
    sequences = []
    for insertion_order in (((2, 0), (0, 0), (1, 0)), ((1, 0), (2, 0), (0, 0))):
        runtime, _, _, events = make_runtime()
        lanes = [register_lane(runtime, events, key) for key in insertion_order]
        runtime.dispatch_batch([make_batch(lane) for lane in reversed(lanes)])
        sequences.append([event[1] for event in events if event[0] == "reshard"])
        runtime.close()
    assert sequences == [["0-0", "1-0", "2-0"], ["0-0", "1-0", "2-0"]]


def test_fully_replicated_layout_uses_size_one_shard_workaround():
    runtime, m2n, _, events = make_runtime()
    lane = register_lane(runtime, events, (0, 0))

    runtime.dispatch_batch(
        [_M2nLaneBatch(lane=lane, calls=(make_call(REPLICATE),), total_bytes=16)]
    )

    call = m2n.calls[0]
    assert call["src_placements"] == (FakeShard(0), FakeReplicate())
    assert call["dst_placements"] == (FakeShard(0), FakeReplicate())
    runtime.close()


def test_shutdown_finalizes_m2n_before_communicators_in_canonical_order():
    runtime, _, _, events = make_runtime()
    register_lane(runtime, events, (1, 0), runtime_owned_stream=True)
    register_lane(runtime, events, (0, 0), runtime_owned_stream=True)

    runtime.close()

    resource_events = {
        "handle_destroy",
        "stream_destroy",
        "comm_finalize",
        "comm_destroy",
    }
    lifecycle = [
        event
        for event in events
        if event[0] in resource_events
    ]
    assert lifecycle == [
        ("handle_destroy",),
        ("stream_destroy", "owned-1"),
        ("comm_finalize", "0-0"),
        ("comm_destroy", "0-0"),
        ("stream_destroy", "owned-0"),
        ("comm_finalize", "1-0"),
        ("comm_destroy", "1-0"),
    ]


@pytest.mark.parametrize("trainer_pp", [2, 4, 8])
def test_pp_to_pp1_ownership_patterns_cannot_form_ordering_cycle(trainer_pp: int):
    keys = tuple((stage, 0) for stage in range(trainer_pp))
    owner_sequences = [keys, *((key,) for key in keys)]
    edges = {
        (left, right)
        for sequence in owner_sequences
        for left, right in zip(sequence, sequence[1:])
    }

    # Every dependency follows the same strict total order. Trainer processes
    # own singleton subsequences, while the generator owns the complete order.
    assert all(left < right for left, right in edges)


def test_create_lane_uses_official_unique_id_and_communicator_api():
    runtime, _, nccl, events = make_runtime()
    assert runtime.new_unique_id_bytes() == nccl.next_uid
    lane = runtime.create_lane(
        _M2nLaneSpec(
            lane_id="bootstrap",
            key=(0, 0),
            unique_id=b"x" * 128,
            nranks=2,
            comm_rank=1,
            device_id=0,
        )
    )
    assert lane.comm_rank == 1
    assert ("comm_init", 2, 1, b"x" * 128) in events
    runtime.close()


def test_rejects_old_nccl_version():
    events: list[tuple] = []
    nccl = FakeNccl(events)
    nccl.get_version = lambda: "2.30.4"
    with pytest.raises(RuntimeError, match="NCCL >= 2.30.5"):
        _M2nRuntime(
            0,
            _m2n_module=FakeM2n(events),
            _nccl_module=nccl,
            _torch_module=SimpleNamespace(cuda=FakeCuda(events)),
            _enforce_singleton=False,
        )
