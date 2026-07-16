# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MX-brokered NCCL communicator bootstrap with cooperative deadlines."""

from __future__ import annotations

import time
from typing import Protocol

from ... import m2n_bootstrap_pb2
from ...m2n_bootstrap.client import MxM2nBootstrapClient
from ...m2n_bootstrap.types import (
    NCCL_UNIQUE_ID_BYTES,
    M2nBootstrapAssignment,
)
NCCL_SUCCESS = 0
NCCL_IN_PROGRESS = 7


class M2nBootstrapError(RuntimeError):
    """Base error for communicator bootstrap failures."""


class M2nBootstrapAborted(M2nBootstrapError):
    """The coordinator or a peer aborted the attempt."""


class M2nBootstrapTimeout(M2nBootstrapError):
    """The UID or communicator did not become ready before its deadline."""


class _M2nBinding(Protocol):
    def set_device(self, device_id: int) -> None: ...

    def get_unique_id_bytes(self) -> bytes: ...

    def comm_init_rank_nonblocking(
        self, nranks: int, uid: bytes, rank: int
    ) -> tuple[int, int]: ...

    def comm_get_async_error(self, comm: int) -> int: ...

    def comm_abort(self, comm: int) -> None: ...


def _validate_record(
    assignment: M2nBootstrapAssignment,
    record: m2n_bootstrap_pb2.M2nBootstrapRecord,
) -> bytes:
    key = record.key
    if (
        key.job_id != assignment.job_id
        or key.attempt_id != assignment.attempt_id
        or key.cohort_id != assignment.cohort_id
    ):
        raise M2nBootstrapError("MX returned a record for a different bootstrap attempt")
    try:
        state = m2n_bootstrap_pb2.M2nBootstrapState.Name(record.state)
    except ValueError as error:
        raise M2nBootstrapError(f"MX returned invalid bootstrap state {record.state}") from error
    if record.state == m2n_bootstrap_pb2.M2N_BOOTSTRAP_STATE_ABORTED:
        raise M2nBootstrapAborted(record.reason or "bootstrap attempt aborted")
    if record.state == m2n_bootstrap_pb2.M2N_BOOTSTRAP_STATE_EXPIRED:
        raise M2nBootstrapAborted(record.reason or "bootstrap attempt expired")
    if record.state != m2n_bootstrap_pb2.M2N_BOOTSTRAP_STATE_PUBLISHED:
        raise M2nBootstrapError(f"bootstrap record is not published: {state}")
    if (
        record.source_world_size != assignment.source_world_size
        or record.destination_world_size != assignment.destination_world_size
        or record.world_size != assignment.world_size
    ):
        raise M2nBootstrapError("MX bootstrap world sizes do not match the assignment")
    if record.roster_digest != assignment.roster_digest:
        raise M2nBootstrapError("MX bootstrap roster digest does not match the assignment")
    if record.config_digest != assignment.config_digest:
        raise M2nBootstrapError("MX bootstrap config digest does not match the assignment")
    if record.publisher_participant_id != assignment.uid_publisher_participant_id:
        raise M2nBootstrapError("MX bootstrap publisher does not match the assignment")
    if len(record.nccl_unique_id) != NCCL_UNIQUE_ID_BYTES:
        raise M2nBootstrapError(
            f"MX bootstrap UID must be exactly {NCCL_UNIQUE_ID_BYTES} bytes"
        )
    return bytes(record.nccl_unique_id)


def _wait_for_record(
    client: MxM2nBootstrapClient,
    assignment: M2nBootstrapAssignment,
    deadline: float,
    poll_interval_s: float,
) -> m2n_bootstrap_pb2.M2nBootstrapRecord:
    key = assignment.key_proto()
    while time.monotonic() < deadline:
        remaining_s = deadline - time.monotonic()
        record = client.get_bootstrap(key, timeout_s=remaining_s)
        if record is not None:
            return record
        time.sleep(
            min(poll_interval_s, max(0.0, deadline - time.monotonic()))
        )
    raise M2nBootstrapTimeout("timed out waiting for MX bootstrap publication")


def _bootstrap_comm_from_mx_local(
    m2n: _M2nBinding,
    client: MxM2nBootstrapClient,
    assignment: M2nBootstrapAssignment,
    *,
    device_id: int,
    poll_interval_s: float = 0.05,
) -> int:
    """Initialize one local NCCL communicator from a coordinator assignment.

    Returning means local NCCL initialization succeeded. The caller must report
    that result to NeMo RL/Dynamo and must not run M2N until the coordinator has
    confirmed that every participant succeeded.
    """

    assignment.validate()
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")

    deadline = time.monotonic() + assignment.timeout_s
    m2n.set_device(device_id)

    if assignment.is_uid_publisher:
        uid = m2n.get_unique_id_bytes()
        record = client.publish_bootstrap(
            m2n_bootstrap_pb2.PublishM2nBootstrapRequest(
                key=assignment.key_proto(),
                nccl_unique_id=uid,
                source_world_size=assignment.source_world_size,
                destination_world_size=assignment.destination_world_size,
                world_size=assignment.world_size,
                roster_digest=assignment.roster_digest,
                config_digest=assignment.config_digest,
                publisher_participant_id=assignment.participant_id,
                ttl_ms=assignment.ttl_ms,
            ),
            timeout_s=max(1e-6, deadline - time.monotonic()),
        )
    else:
        record = _wait_for_record(client, assignment, deadline, poll_interval_s)

    uid = _validate_record(assignment, record)
    comm, init_status = m2n.comm_init_rank_nonblocking(
        assignment.world_size,
        uid,
        assignment.assigned_nccl_rank,
    )
    if init_status not in (NCCL_SUCCESS, NCCL_IN_PROGRESS):
        error = M2nBootstrapError(
            f"ncclCommInitRankConfig failed with status {init_status}"
        )
        _abort_with_note(m2n, comm, error)
        raise error
    if not comm:
        raise M2nBootstrapError(
            "ncclCommInitRankConfig returned a null communicator"
        )

    try:
        while time.monotonic() < deadline:
            current = client.get_bootstrap(
                assignment.key_proto(),
                timeout_s=max(1e-6, deadline - time.monotonic()),
            )
            if current is None:
                raise M2nBootstrapAborted("MX bootstrap record disappeared")
            _validate_record(assignment, current)

            async_status = m2n.comm_get_async_error(comm)
            if async_status == NCCL_SUCCESS:
                return comm
            if async_status != NCCL_IN_PROGRESS:
                raise M2nBootstrapError(
                    f"NCCL communicator initialization failed with status {async_status}"
                )
            time.sleep(
                min(poll_interval_s, max(0.0, deadline - time.monotonic()))
            )
        raise M2nBootstrapTimeout(
            "NCCL communicator initialization exceeded its deadline"
        )
    except BaseException as error:
        _abort_with_note(m2n, comm, error)
        raise


def _abort_with_note(m2n: _M2nBinding, comm: int, error: BaseException) -> None:
    """Abort best-effort without replacing the original bootstrap failure."""

    if not comm:
        return
    try:
        m2n.comm_abort(comm)
    except BaseException as abort_error:
        if hasattr(error, "add_note"):
            error.add_note(f"ncclCommAbort also failed: {abort_error}")


def _bootstrap_reason(error: BaseException) -> str:
    reason = str(error) or type(error).__name__
    return reason.encode("utf-8")[:1024].decode("utf-8", errors="ignore")


def _abort_bootstrap_with_note(
    client: MxM2nBootstrapClient,
    assignment: M2nBootstrapAssignment,
    error: BaseException,
) -> None:
    try:
        client.abort_bootstrap(
            assignment.key_proto(),
            requested_by=assignment.participant_id,
            reason=_bootstrap_reason(error),
            timeout_s=min(5.0, assignment.timeout_s),
        )
    except BaseException as abort_error:
        if hasattr(error, "add_note"):
            error.add_note(f"AbortBootstrap also failed: {abort_error}")


def bootstrap_comm_from_mx(
    m2n: _M2nBinding,
    client: MxM2nBootstrapClient,
    assignment: M2nBootstrapAssignment,
    *,
    device_id: int,
    poll_interval_s: float = 0.05,
) -> int:
    """Initialize locally and fence the MX attempt on any bootstrap failure."""

    assignment.validate()
    try:
        if poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        return _bootstrap_comm_from_mx_local(
            m2n,
            client,
            assignment,
            device_id=device_id,
            poll_interval_s=poll_interval_s,
        )
    except BaseException as error:
        _abort_bootstrap_with_note(client, assignment, error)
        raise
