# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap one NCCL M2N communicator through WeightSyncService."""

from __future__ import annotations

import time
from typing import Protocol
from uuid import UUID

from ... import weight_sync_pb2, weight_sync_pb2_grpc

NCCL_SUCCESS = 0
NCCL_IN_PROGRESS = 7
NCCL_UNIQUE_ID_BYTES = 128
SHA256_BYTES = 32
UINT32_MAX = (1 << 32) - 1


class M2nBootstrapError(RuntimeError):
    """NCCL communicator bootstrap failed."""


class M2nBootstrapTimeout(M2nBootstrapError):
    """Bootstrap did not finish before its local deadline."""


class _M2nBinding(Protocol):
    def set_device(self, device_id: int) -> None: ...

    def get_unique_id_bytes(self) -> bytes: ...

    def comm_init_rank_nonblocking(
        self, nranks: int, uid: bytes, rank: int
    ) -> tuple[int, int]: ...

    def comm_get_async_error(self, comm: int) -> int: ...

    def comm_abort(self, comm: int) -> None: ...


def _validate_inputs(
    *,
    cohort_id: str,
    attempt_id: str,
    assigned_nccl_rank: int,
    source_world_size: int,
    destination_world_size: int,
    roster_digest: bytes,
    device_id: int,
    timeout_s: float,
    poll_interval_s: float,
) -> int:
    if not cohort_id:
        raise ValueError("cohort_id is required")
    try:
        parsed_attempt = UUID(attempt_id)
    except (ValueError, AttributeError) as error:
        raise ValueError("attempt_id must be a canonical UUIDv4") from error
    if parsed_attempt.version != 4 or str(parsed_attempt) != attempt_id:
        raise ValueError("attempt_id must be a canonical UUIDv4")
    if not 0 < source_world_size <= UINT32_MAX:
        raise ValueError("source_world_size must be a positive uint32")
    if not 0 < destination_world_size <= UINT32_MAX:
        raise ValueError("destination_world_size must be a positive uint32")
    world_size = source_world_size + destination_world_size
    if world_size > UINT32_MAX:
        raise ValueError("NCCL world size exceeds uint32")
    if not 0 <= assigned_nccl_rank < world_size:
        raise ValueError("assigned_nccl_rank is outside the NCCL world")
    if len(roster_digest) != SHA256_BYTES:
        raise ValueError("roster_digest must be a 32-byte SHA-256 digest")
    if device_id < 0:
        raise ValueError("device_id must be non-negative")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")
    return world_size


def _remaining(deadline: float, operation: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise M2nBootstrapTimeout(f"timed out {operation}")
    return max(1e-6, remaining)


def _validate_record(
    record: weight_sync_pb2.NcclBootstrapRecord,
    *,
    cohort_id: str,
    attempt_id: str,
    source_world_size: int,
    destination_world_size: int,
    world_size: int,
    roster_digest: bytes,
    expected_uid: bytes | None,
) -> bytes:
    if record.cohort_id != cohort_id:
        raise M2nBootstrapError("MX bootstrap cohort_id does not match")
    if record.attempt_id != attempt_id:
        raise M2nBootstrapError("MX bootstrap attempt_id does not match")
    if (
        record.source_world_size != source_world_size
        or record.destination_world_size != destination_world_size
        or record.world_size != world_size
    ):
        raise M2nBootstrapError("MX bootstrap world sizes do not match")
    if bytes(record.roster_digest) != roster_digest:
        raise M2nBootstrapError("MX bootstrap roster digest does not match")
    if len(record.nccl_unique_id) != NCCL_UNIQUE_ID_BYTES:
        raise M2nBootstrapError(
            f"MX bootstrap UID must be exactly {NCCL_UNIQUE_ID_BYTES} bytes"
        )
    uid = bytes(record.nccl_unique_id)
    if expected_uid is not None and uid != expected_uid:
        raise M2nBootstrapError("MX bootstrap UID does not match the published UID")
    return uid


def _abort_with_note(m2n: _M2nBinding, comm: int, error: BaseException) -> None:
    """Abort locally without replacing the original failure.

    ncclCommAbort is synchronous and may itself hang. A process supervisor must
    enforce the hard whole-attempt termination deadline.
    """

    if not comm:
        return
    try:
        m2n.comm_abort(comm)
    except BaseException as abort_error:
        if hasattr(error, "add_note"):
            error.add_note(f"ncclCommAbort also failed: {abort_error}")


def bootstrap_comm_from_mx(
    m2n: _M2nBinding,
    weight_sync_stub: weight_sync_pb2_grpc.WeightSyncServiceStub,
    *,
    cohort_id: str,
    attempt_id: str,
    assigned_nccl_rank: int,
    source_world_size: int,
    destination_world_size: int,
    roster_digest: bytes,
    device_id: int,
    timeout_s: float,
    poll_interval_s: float = 0.05,
) -> int:
    """Initialize a local nonblocking NCCL communicator.

    The coordinator owns membership, result collection, release, and failure of
    the whole attempt. A return value only means this local rank initialized.
    """

    world_size = _validate_inputs(
        cohort_id=cohort_id,
        attempt_id=attempt_id,
        assigned_nccl_rank=assigned_nccl_rank,
        source_world_size=source_world_size,
        destination_world_size=destination_world_size,
        roster_digest=roster_digest,
        device_id=device_id,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    )
    deadline = time.monotonic() + timeout_s
    m2n.set_device(device_id)

    published_uid = None
    if assigned_nccl_rank == 0:
        published_uid = m2n.get_unique_id_bytes()
        if len(published_uid) != NCCL_UNIQUE_ID_BYTES:
            raise M2nBootstrapError(
                f"generated NCCL UID must be exactly {NCCL_UNIQUE_ID_BYTES} bytes"
            )
        response = weight_sync_stub.PublishNcclBootstrap(
            weight_sync_pb2.PublishNcclBootstrapRequest(
                record=weight_sync_pb2.NcclBootstrapRecord(
                    cohort_id=cohort_id,
                    attempt_id=attempt_id,
                    nccl_unique_id=published_uid,
                    source_world_size=source_world_size,
                    destination_world_size=destination_world_size,
                    world_size=world_size,
                    roster_digest=roster_digest,
                )
            ),
            timeout=_remaining(deadline, "publishing NCCL bootstrap record"),
        )
        if not response.ok:
            raise M2nBootstrapError("MX rejected NCCL bootstrap publication")

    record = None
    while record is None:
        response = weight_sync_stub.GetNcclBootstrap(
            weight_sync_pb2.GetNcclBootstrapRequest(attempt_id=attempt_id),
            timeout=_remaining(deadline, "waiting for NCCL bootstrap publication"),
        )
        if response.found:
            if not response.HasField("record"):
                raise M2nBootstrapError("MX returned found=true without a record")
            record = response.record
            break
        time.sleep(
            min(
                poll_interval_s,
                _remaining(deadline, "waiting for NCCL bootstrap publication"),
            )
        )

    uid = _validate_record(
        record,
        cohort_id=cohort_id,
        attempt_id=attempt_id,
        source_world_size=source_world_size,
        destination_world_size=destination_world_size,
        world_size=world_size,
        roster_digest=roster_digest,
        expected_uid=published_uid,
    )
    comm, init_status = m2n.comm_init_rank_nonblocking(
        world_size, uid, assigned_nccl_rank
    )
    if init_status not in (NCCL_SUCCESS, NCCL_IN_PROGRESS):
        error = M2nBootstrapError(
            f"ncclCommInitRankConfig failed with status {init_status}"
        )
        _abort_with_note(m2n, comm, error)
        raise error
    if not comm:
        raise M2nBootstrapError("ncclCommInitRankConfig returned a null communicator")

    try:
        while True:
            _remaining(deadline, "initializing the NCCL communicator")
            async_status = m2n.comm_get_async_error(comm)
            if async_status == NCCL_SUCCESS:
                return comm
            if async_status != NCCL_IN_PROGRESS:
                raise M2nBootstrapError(
                    "NCCL communicator initialization failed with "
                    f"status {async_status}"
                )
            time.sleep(
                min(
                    poll_interval_s,
                    _remaining(deadline, "initializing the NCCL communicator"),
                )
            )
    except BaseException as error:
        _abort_with_note(m2n, comm, error)
        raise
