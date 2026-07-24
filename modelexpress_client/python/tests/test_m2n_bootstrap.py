# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import threading

import grpc
import pytest

from modelexpress import weight_sync_pb2, weight_sync_pb2_grpc
from modelexpress.weight_transfer.transport import nccl_m2n_bootstrap as bootstrap_module
from modelexpress.weight_transfer.transport.nccl_m2n_bootstrap import (
    M2nBootstrapError,
    M2nBootstrapTimeout,
    bootstrap_comm_from_mx,
)


ATTEMPT_ID = "123e4567-e89b-42d3-a456-426614174000"
OTHER_ATTEMPT_ID = "123e4567-e89b-42d3-a456-426614174001"
UID = bytes(range(128))
ROSTER_DIGEST = bytes([1] * 32)


def record(**overrides) -> weight_sync_pb2.NcclBootstrapRecord:
    fields = {
        "cohort_id": "pp-0",
        "attempt_id": ATTEMPT_ID,
        "nccl_unique_id": UID,
        "source_world_size": 1,
        "destination_world_size": 1,
        "world_size": 2,
        "roster_digest": ROSTER_DIGEST,
    }
    fields.update(overrides)
    return weight_sync_pb2.NcclBootstrapRecord(**fields)


def call_args(rank: int = 0, *, timeout_s: float = 1.0) -> dict:
    return {
        "cohort_id": "pp-0",
        "attempt_id": ATTEMPT_ID,
        "assigned_nccl_rank": rank,
        "source_world_size": 1,
        "destination_world_size": 1,
        "roster_digest": ROSTER_DIGEST,
        "device_id": rank,
        "timeout_s": timeout_s,
        "poll_interval_s": 0.001,
    }


class FakeStub:
    def __init__(
        self,
        gets=(),
        *,
        publish_ok: bool = True,
        publish_error: BaseException | None = None,
        published_record: weight_sync_pb2.NcclBootstrapRecord | None = None,
    ):
        self.gets = deque(gets)
        self.publish_ok = publish_ok
        self.publish_error = publish_error
        self.published_record = published_record
        self.records = {}
        self.publish_calls = []
        self.get_calls = []

    def PublishNcclBootstrap(self, request, *, timeout):
        self.publish_calls.append((request, timeout))
        if self.publish_error is not None:
            raise self.publish_error
        if self.publish_ok:
            stored = self.published_record or request.record
            self.records[request.record.attempt_id] = stored
        return weight_sync_pb2.PublishNcclBootstrapResponse(ok=self.publish_ok)

    def GetNcclBootstrap(self, request, *, timeout):
        self.get_calls.append((request, timeout))
        if self.gets:
            value = self.gets.popleft()
        else:
            value = self.records.get(request.attempt_id)
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, weight_sync_pb2.GetNcclBootstrapResponse):
            return value
        if value is None:
            return weight_sync_pb2.GetNcclBootstrapResponse(found=False)
        return weight_sync_pb2.GetNcclBootstrapResponse(found=True, record=value)


class FakeM2n:
    def __init__(
        self,
        statuses=(bootstrap_module.NCCL_SUCCESS,),
        *,
        init_status=bootstrap_module.NCCL_IN_PROGRESS,
        comm=123,
        uid=UID,
    ):
        self.statuses = deque(statuses)
        self.last_status = bootstrap_module.NCCL_IN_PROGRESS
        self.init_status = init_status
        self.comm = comm
        self.uid = uid
        self.devices = []
        self.generated_uids = 0
        self.init_calls = []
        self.async_calls = []
        self.aborted = []

    def set_device(self, device_id):
        self.devices.append(device_id)

    def get_unique_id_bytes(self):
        self.generated_uids += 1
        return self.uid

    def comm_init_rank_nonblocking(self, nranks, uid, rank):
        self.init_calls.append((nranks, uid, rank))
        return self.comm, self.init_status

    def comm_get_async_error(self, comm):
        self.async_calls.append(comm)
        if self.statuses:
            self.last_status = self.statuses.popleft()
        return self.last_status

    def comm_abort(self, comm):
        self.aborted.append(comm)


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class InProcessWeightSyncServicer(
    weight_sync_pb2_grpc.WeightSyncServiceServicer
):
    """Minimal real-gRPC store matching the Rust WeightSync bootstrap contract."""

    def __init__(self):
        self.records = {}
        self.lock = threading.Lock()
        self.first_missing_get = threading.Event()
        self.publish_count = 0
        self.get_count = 0

    def PublishNcclBootstrap(self, request, context):
        stored = weight_sync_pb2.NcclBootstrapRecord()
        stored.CopyFrom(request.record)
        with self.lock:
            self.publish_count += 1
            self.records[stored.attempt_id] = stored
        return weight_sync_pb2.PublishNcclBootstrapResponse(ok=True)

    def GetNcclBootstrap(self, request, context):
        with self.lock:
            self.get_count += 1
            stored = self.records.get(request.attempt_id)
        if stored is None:
            self.first_missing_get.set()
            return weight_sync_pb2.GetNcclBootstrapResponse(found=False)
        response_record = weight_sync_pb2.NcclBootstrapRecord()
        response_record.CopyFrom(stored)
        return weight_sync_pb2.GetNcclBootstrapResponse(
            found=True, record=response_record
        )


@pytest.fixture
def real_weight_sync_stub():
    servicer = InProcessWeightSyncServicer()
    server = grpc.server(ThreadPoolExecutor(max_workers=8))
    weight_sync_pb2_grpc.add_WeightSyncServiceServicer_to_server(
        servicer, server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    assert port > 0
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    grpc.channel_ready_future(channel).result(timeout=2.0)
    try:
        yield servicer, weight_sync_pb2_grpc.WeightSyncServiceStub(channel)
    finally:
        channel.close()
        server.stop(grace=0).wait(timeout=2.0)


@pytest.mark.parametrize(
    "attempt_id",
    [
        "not-a-uuid",
        "123E4567-E89B-42D3-A456-426614174000",
        "123e4567-e89b-12d3-a456-426614174000",
    ],
)
def test_rejects_noncanonical_uuid_v4(attempt_id):
    args = call_args()
    args["attempt_id"] = attempt_id
    with pytest.raises(ValueError, match="UUIDv4"):
        bootstrap_comm_from_mx(FakeM2n(), FakeStub(), **args)


def test_four_ranks_bootstrap_one_communicator_through_real_grpc_stub(
    real_weight_sync_stub,
):
    servicer, stub = real_weight_sync_stub
    m2n_by_rank = {
        rank: FakeM2n(
            statuses=(
                bootstrap_module.NCCL_IN_PROGRESS,
                bootstrap_module.NCCL_SUCCESS,
            ),
            comm=100 + rank,
        )
        for rank in range(4)
    }

    def bootstrap(rank):
        args = call_args(rank=rank, timeout_s=2.0)
        args.update(
            source_world_size=2,
            destination_world_size=2,
            poll_interval_s=0.005,
        )
        return bootstrap_comm_from_mx(m2n_by_rank[rank], stub, **args)

    with ThreadPoolExecutor(max_workers=4) as pool:
        peer_futures = [pool.submit(bootstrap, rank) for rank in range(1, 4)]
        assert servicer.first_missing_get.wait(timeout=1.0)
        publisher_future = pool.submit(bootstrap, 0)

        assert publisher_future.result(timeout=2.0) == 100
        assert [future.result(timeout=2.0) for future in peer_futures] == [
            101,
            102,
            103,
        ]

    assert servicer.publish_count == 1
    assert servicer.records[ATTEMPT_ID] == record(
        source_world_size=2,
        destination_world_size=2,
        world_size=4,
    )
    for rank, m2n in m2n_by_rank.items():
        assert m2n.generated_uids == (1 if rank == 0 else 0)
        assert m2n.init_calls == [(4, UID, rank)]
        assert len(m2n.async_calls) == 2
        assert not m2n.aborted


def test_rank_zero_generates_once_publishes_once_and_uses_common_get_path():
    stub = FakeStub()
    m2n = FakeM2n(
        statuses=(bootstrap_module.NCCL_IN_PROGRESS, bootstrap_module.NCCL_SUCCESS)
    )

    comm = bootstrap_comm_from_mx(m2n, stub, **call_args())

    assert comm == 123
    assert m2n.devices == [0]
    assert m2n.generated_uids == 1
    assert len(stub.publish_calls) == 1
    assert len(stub.get_calls) == 1
    assert stub.publish_calls[0][0].record == record()
    assert m2n.init_calls == [(2, UID, 0)]
    assert not m2n.aborted


def test_nonzero_rank_only_fetches():
    stub = FakeStub([None, record()])
    m2n = FakeM2n()

    comm = bootstrap_comm_from_mx(m2n, stub, **call_args(rank=1))

    assert comm == 123
    assert m2n.generated_uids == 0
    assert not stub.publish_calls
    assert len(stub.get_calls) == 2
    assert m2n.init_calls == [(2, UID, 1)]


def test_publish_failure_is_not_retried_or_regenerated():
    stub = FakeStub(publish_error=TimeoutError("publish deadline"))
    m2n = FakeM2n()

    with pytest.raises(TimeoutError, match="publish deadline"):
        bootstrap_comm_from_mx(m2n, stub, **call_args())

    assert m2n.generated_uids == 1
    assert len(stub.publish_calls) == 1
    assert not stub.get_calls
    assert not m2n.init_calls


def test_publish_rejection_stops_before_get_or_nccl_init():
    stub = FakeStub(publish_ok=False)
    m2n = FakeM2n()

    with pytest.raises(M2nBootstrapError, match="rejected"):
        bootstrap_comm_from_mx(m2n, stub, **call_args())

    assert m2n.generated_uids == 1
    assert len(stub.publish_calls) == 1
    assert not stub.get_calls
    assert not m2n.init_calls


def test_found_without_record_stops_before_nccl_init():
    stub = FakeStub(
        [weight_sync_pb2.GetNcclBootstrapResponse(found=True)]
    )
    m2n = FakeM2n()

    with pytest.raises(M2nBootstrapError, match="without a record"):
        bootstrap_comm_from_mx(m2n, stub, **call_args(rank=1))

    assert len(stub.get_calls) == 1
    assert not m2n.init_calls


def test_get_rpc_failure_is_not_retried_and_stops_before_nccl_init():
    stub = FakeStub([RuntimeError("get failed")])
    m2n = FakeM2n()

    with pytest.raises(RuntimeError, match="get failed"):
        bootstrap_comm_from_mx(m2n, stub, **call_args(rank=1))

    assert len(stub.get_calls) == 1
    assert not m2n.init_calls


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"cohort_id": "pp-1"}, "cohort_id"),
        ({"attempt_id": OTHER_ATTEMPT_ID}, "attempt_id"),
        ({"source_world_size": 2, "world_size": 3}, "world sizes"),
        ({"destination_world_size": 2, "world_size": 3}, "world sizes"),
        ({"world_size": 3}, "world sizes"),
        ({"roster_digest": bytes([9] * 32)}, "roster digest"),
        ({"nccl_unique_id": UID[:-1]}, "UID"),
    ],
)
def test_record_mismatch_fails_before_nccl_initialization(changed, message):
    stub = FakeStub([record(**changed)])
    m2n = FakeM2n()

    with pytest.raises(M2nBootstrapError, match=message):
        bootstrap_comm_from_mx(m2n, stub, **call_args(rank=1))

    assert not m2n.init_calls
    assert not m2n.aborted


def test_rank_zero_rejects_uid_different_from_the_one_it_published():
    stub = FakeStub(published_record=record(nccl_unique_id=bytes([9] * 128)))
    m2n = FakeM2n()

    with pytest.raises(M2nBootstrapError, match="published UID"):
        bootstrap_comm_from_mx(m2n, stub, **call_args())

    assert not m2n.init_calls


@pytest.mark.parametrize("rank", [-1, 2])
def test_assigned_rank_mismatch_fails_before_any_side_effect(rank):
    m2n = FakeM2n()
    with pytest.raises(ValueError, match="assigned_nccl_rank"):
        bootstrap_comm_from_mx(m2n, FakeStub(), **call_args(rank=rank))
    assert not m2n.devices
    assert not m2n.init_calls


def test_generated_uid_length_is_validated_before_publish():
    stub = FakeStub()
    m2n = FakeM2n(uid=UID[:-1])
    with pytest.raises(M2nBootstrapError, match="generated NCCL UID"):
        bootstrap_comm_from_mx(m2n, stub, **call_args())
    assert not stub.publish_calls


def test_immediate_nccl_success():
    m2n = FakeM2n(init_status=bootstrap_module.NCCL_SUCCESS)
    assert bootstrap_comm_from_mx(m2n, FakeStub(), **call_args()) == 123
    assert m2n.async_calls == [123]


def test_async_nccl_success_and_no_mx_reads_after_init_starts():
    stub = FakeStub([record()])
    m2n = FakeM2n(
        statuses=(
            bootstrap_module.NCCL_IN_PROGRESS,
            bootstrap_module.NCCL_IN_PROGRESS,
            bootstrap_module.NCCL_SUCCESS,
        )
    )

    assert bootstrap_comm_from_mx(m2n, stub, **call_args(rank=1)) == 123
    assert len(stub.get_calls) == 1
    assert len(m2n.async_calls) == 3


def test_immediate_init_failure_aborts_returned_communicator():
    m2n = FakeM2n(init_status=4)
    with pytest.raises(M2nBootstrapError, match="status 4"):
        bootstrap_comm_from_mx(m2n, FakeStub(), **call_args())
    assert m2n.aborted == [123]


def test_null_communicator_is_rejected():
    m2n = FakeM2n(comm=0)
    with pytest.raises(M2nBootstrapError, match="null communicator"):
        bootstrap_comm_from_mx(m2n, FakeStub(), **call_args())
    assert not m2n.aborted


def test_async_nccl_failure_aborts_local_communicator():
    m2n = FakeM2n(statuses=(4,))
    with pytest.raises(M2nBootstrapError, match="status 4"):
        bootstrap_comm_from_mx(m2n, FakeStub(), **call_args())
    assert m2n.aborted == [123]


def test_missing_record_polling_uses_remaining_deadline(monkeypatch):
    fake_time = FakeTime()
    monkeypatch.setattr(bootstrap_module, "time", fake_time)
    stub = FakeStub()

    with pytest.raises(M2nBootstrapTimeout, match="publication"):
        bootstrap_comm_from_mx(
            FakeM2n(), stub, **call_args(rank=1, timeout_s=0.02)
        )

    assert len(stub.get_calls) == 20
    assert all(0 < timeout <= 0.02 for _, timeout in stub.get_calls)


def test_async_nccl_deadline_aborts_local_communicator(monkeypatch):
    fake_time = FakeTime()
    monkeypatch.setattr(bootstrap_module, "time", fake_time)
    m2n = FakeM2n(statuses=(bootstrap_module.NCCL_IN_PROGRESS,))

    with pytest.raises(M2nBootstrapTimeout, match="initializing"):
        bootstrap_comm_from_mx(
            m2n, FakeStub(), **call_args(timeout_s=0.02)
        )

    assert m2n.aborted == [123]


def test_every_rpc_receives_a_positive_remaining_deadline():
    stub = FakeStub()
    bootstrap_comm_from_mx(FakeM2n(), stub, **call_args(timeout_s=0.5))
    timeouts = [timeout for _, timeout in stub.publish_calls + stub.get_calls]
    assert len(timeouts) == 2
    assert all(0 < timeout <= 0.5 for timeout in timeouts)


def test_abort_failure_never_replaces_original_error():
    class AbortFailingM2n(FakeM2n):
        def comm_abort(self, comm):
            raise RuntimeError("abort failed")

    m2n = AbortFailingM2n(statuses=(4,))
    with pytest.raises(M2nBootstrapError, match="status 4") as caught:
        bootstrap_comm_from_mx(m2n, FakeStub(), **call_args())
    assert any("abort failed" in note for note in caught.value.__notes__)
