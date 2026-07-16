# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque

import pytest

from modelexpress import m2n_bootstrap_pb2
from modelexpress.m2n_bootstrap.client import MxM2nBootstrapClient
from modelexpress.m2n_bootstrap.types import M2nBootstrapAssignment
from modelexpress.weight_transfer.transport import nccl_m2n_bootstrap as bootstrap_module
from modelexpress.weight_transfer.transport.nccl_m2n_bootstrap import (
    M2nBootstrapAborted,
    M2nBootstrapError,
    M2nBootstrapTimeout,
    bootstrap_comm_from_mx,
)


ATTEMPT_ID = "123e4567-e89b-42d3-a456-426614174000"
UID = bytes(range(128))
ROSTER_DIGEST = bytes([1] * 32)
CONFIG_DIGEST = bytes([2] * 32)


def assignment(rank: int = 0, *, timeout_s: float = 1.0) -> M2nBootstrapAssignment:
    return M2nBootstrapAssignment(
        job_id="job",
        attempt_id=ATTEMPT_ID,
        cohort_id="cohort",
        participant_id=f"participant-{rank}",
        uid_publisher_participant_id="participant-0",
        assigned_nccl_rank=rank,
        source_world_size=1,
        destination_world_size=1,
        roster_digest=ROSTER_DIGEST,
        config_digest=CONFIG_DIGEST,
        timeout_s=timeout_s,
    )


def record(
    *,
    state: int = m2n_bootstrap_pb2.M2N_BOOTSTRAP_STATE_PUBLISHED,
    roster_digest: bytes = ROSTER_DIGEST,
    reason: str = "",
) -> m2n_bootstrap_pb2.M2nBootstrapRecord:
    return m2n_bootstrap_pb2.M2nBootstrapRecord(
        key=m2n_bootstrap_pb2.M2nBootstrapKey(
            job_id="job", attempt_id=ATTEMPT_ID, cohort_id="cohort"
        ),
        nccl_unique_id=(UID if state == m2n_bootstrap_pb2.M2N_BOOTSTRAP_STATE_PUBLISHED else b""),
        source_world_size=1,
        destination_world_size=1,
        world_size=2,
        roster_digest=roster_digest,
        config_digest=CONFIG_DIGEST,
        publisher_participant_id="participant-0",
        state=state,
        expires_at_ms=10_000,
        reason=reason,
        revision=1,
    )


class FakeClient:
    def __init__(self, records=()):
        self.records = deque(records)
        self.last_record = None
        self.publish_requests = []
        self.get_timeouts = []
        self.abort_requests = []

    def publish_bootstrap(self, request, *, timeout_s=None):
        self.publish_requests.append(request)
        self.last_record = record()
        return self.last_record

    def get_bootstrap(self, key, *, timeout_s=None):
        self.get_timeouts.append(timeout_s)
        if self.records:
            value = self.records.popleft()
            if value is not None:
                self.last_record = value
            return value
        return self.last_record

    def abort_bootstrap(
        self, key, *, requested_by, reason, timeout_s=None
    ):
        self.abort_requests.append((key, requested_by, reason, timeout_s))
        self.last_record = record(
            state=m2n_bootstrap_pb2.M2N_BOOTSTRAP_STATE_ABORTED,
            reason=reason,
        )
        return self.last_record


class FakeM2n:
    def __init__(
        self,
        statuses=(bootstrap_module.NCCL_SUCCESS,),
        *,
        init_status=bootstrap_module.NCCL_IN_PROGRESS,
        comm=123,
    ):
        self.statuses = deque(statuses)
        self.last_status = bootstrap_module.NCCL_IN_PROGRESS
        self.init_status = init_status
        self.comm = comm
        self.devices = []
        self.generated_uids = 0
        self.init_calls = []
        self.aborted = []

    def set_device(self, device_id):
        self.devices.append(device_id)

    def get_unique_id_bytes(self):
        self.generated_uids += 1
        return UID

    def comm_init_rank_nonblocking(self, nranks, uid, rank):
        self.init_calls.append((nranks, uid, rank))
        return self.comm, self.init_status

    def comm_get_async_error(self, comm):
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


def test_assignment_rejects_non_uuid_attempt_id():
    value = assignment()
    invalid = M2nBootstrapAssignment(
        **{**value.__dict__, "attempt_id": "reused-name"}
    )
    with pytest.raises(ValueError, match="UUID"):
        invalid.validate()


def test_assignment_rejects_noncanonical_or_non_v4_attempt_id():
    value = assignment()
    for attempt_id in (
        "123E4567-E89B-42D3-A456-426614174000",
        "123e4567-e89b-12d3-a456-426614174000",
    ):
        invalid = M2nBootstrapAssignment(
            **{**value.__dict__, "attempt_id": attempt_id}
        )
        with pytest.raises(ValueError, match="UUIDv4"):
            invalid.validate()


def test_assignment_requires_rank_zero_to_be_uid_publisher():
    value = assignment()
    invalid = M2nBootstrapAssignment(
        **{**value.__dict__, "uid_publisher_participant_id": "participant-1"}
    )
    with pytest.raises(ValueError, match="rank zero"):
        invalid.validate()


def test_source_rank_publishes_uid_and_uses_nonblocking_init():
    client = FakeClient()
    m2n = FakeM2n(statuses=(bootstrap_module.NCCL_IN_PROGRESS, bootstrap_module.NCCL_SUCCESS))

    comm = bootstrap_comm_from_mx(
        m2n, client, assignment(), device_id=3, poll_interval_s=0.001
    )

    assert comm == 123
    assert m2n.devices == [3]
    assert m2n.generated_uids == 1
    assert m2n.init_calls == [(2, UID, 0)]
    assert client.publish_requests[0].ttl_ms == 1_000
    assert not m2n.aborted


def test_non_publisher_fetches_uid_without_generating_one():
    client = FakeClient([None, record()])
    m2n = FakeM2n()

    comm = bootstrap_comm_from_mx(
        m2n, client, assignment(rank=1), device_id=1, poll_interval_s=0.001
    )

    assert comm == 123
    assert m2n.generated_uids == 0
    assert m2n.init_calls == [(2, UID, 1)]


def test_digest_mismatch_is_rejected_before_nccl_init():
    client = FakeClient([record(roster_digest=bytes([9] * 32))])
    m2n = FakeM2n()

    with pytest.raises(M2nBootstrapError, match="roster digest"):
        bootstrap_comm_from_mx(
            m2n, client, assignment(rank=1), device_id=1, poll_interval_s=0.001
        )

    assert not m2n.init_calls
    assert not m2n.aborted


def test_publisher_mismatch_is_rejected_before_nccl_init():
    mismatched = record()
    mismatched.publisher_participant_id = "unexpected-publisher"
    client = FakeClient([mismatched])
    m2n = FakeM2n()

    with pytest.raises(M2nBootstrapError, match="publisher"):
        bootstrap_comm_from_mx(
            m2n, client, assignment(rank=1), device_id=1, poll_interval_s=0.001
        )

    assert not m2n.init_calls
    assert len(client.abort_requests) == 1


def test_mx_abort_during_init_aborts_local_communicator():
    aborted = record(
        state=m2n_bootstrap_pb2.M2N_BOOTSTRAP_STATE_ABORTED,
        reason="peer failed",
    )
    client = FakeClient([record(), aborted])
    m2n = FakeM2n(statuses=(bootstrap_module.NCCL_IN_PROGRESS,))

    with pytest.raises(M2nBootstrapAborted, match="peer failed"):
        bootstrap_comm_from_mx(
            m2n, client, assignment(rank=1), device_id=1, poll_interval_s=0.001
        )

    assert m2n.aborted == [123]


def test_async_nccl_failure_aborts_local_communicator():
    client = FakeClient([record()])
    m2n = FakeM2n(statuses=(4,))

    with pytest.raises(M2nBootstrapError, match="status 4"):
        bootstrap_comm_from_mx(
            m2n, client, assignment(rank=1), device_id=1, poll_interval_s=0.001
        )

    assert m2n.aborted == [123]


def test_immediate_init_failure_aborts_returned_communicator():
    client = FakeClient([record()])
    m2n = FakeM2n(init_status=4)

    with pytest.raises(M2nBootstrapError, match="status 4"):
        bootstrap_comm_from_mx(
            m2n, client, assignment(rank=1), device_id=1, poll_interval_s=0.001
        )

    assert m2n.aborted == [123]


def test_abort_failure_does_not_mask_bootstrap_failure():
    class AbortFailingM2n(FakeM2n):
        def comm_abort(self, comm):
            raise RuntimeError("abort failed")

    client = FakeClient([record()])
    m2n = AbortFailingM2n(statuses=(4,))

    with pytest.raises(M2nBootstrapError, match="status 4") as caught:
        bootstrap_comm_from_mx(
            m2n, client, assignment(rank=1), device_id=1, poll_interval_s=0.001
        )

    assert any("abort failed" in note for note in caught.value.__notes__)


def test_null_communicator_is_rejected():
    client = FakeClient([record()])
    m2n = FakeM2n(comm=0)

    with pytest.raises(M2nBootstrapError, match="null communicator"):
        bootstrap_comm_from_mx(
            m2n, client, assignment(rank=1), device_id=1, poll_interval_s=0.001
        )

    assert not m2n.aborted


def test_client_rejects_nonpositive_rpc_timeout():
    with pytest.raises(ValueError, match="positive"):
        MxM2nBootstrapClient(rpc_timeout_s=0)


def test_uid_wait_obeys_local_deadline(monkeypatch):
    fake_time = FakeTime()
    monkeypatch.setattr(bootstrap_module, "time", fake_time)
    client = FakeClient([None, None, None])
    m2n = FakeM2n()

    with pytest.raises(M2nBootstrapTimeout, match="publication"):
        bootstrap_comm_from_mx(
            m2n,
            client,
            assignment(rank=1, timeout_s=0.02),
            device_id=1,
            poll_interval_s=0.01,
        )

    assert not m2n.init_calls
    assert all(timeout is not None and timeout <= 0.02 for timeout in client.get_timeouts)


def test_async_poll_obeys_local_deadline(monkeypatch):
    fake_time = FakeTime()
    monkeypatch.setattr(bootstrap_module, "time", fake_time)
    client = FakeClient([record()])
    m2n = FakeM2n(statuses=(bootstrap_module.NCCL_IN_PROGRESS,))

    with pytest.raises(M2nBootstrapTimeout, match="deadline"):
        bootstrap_comm_from_mx(
            m2n,
            client,
            assignment(rank=1, timeout_s=0.02),
            device_id=1,
            poll_interval_s=0.01,
        )

    assert m2n.aborted == [123]
