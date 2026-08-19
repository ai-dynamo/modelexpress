# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MX-brokered rendezvous for the NCCL M2N collective refit path.

This replaces the raw ``TCPStore`` a collective normally bootstraps through.
The store only ever had to move 128 bytes of ``ncclUniqueId`` from rank 0 to
everyone else, but doing it through MX buys three things a store structurally
cannot: admission against an explicit expected set, fencing of a stale worker
generation, and a readiness state that a *third* party can observe -- which
matters because the trainer must not enter the collective until the generators
it is pushing into have joined and prepared their destinations.

Torch-free and NCCL-free by construction. The identifier is opaque bytes here;
only the backend that owns the communicator interprets it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import grpc

from .. import refit_collective_pb2 as pb
from .. import refit_collective_pb2_grpc as pb_grpc
from . import envs
from .types import Role

_ROLE_TO_PROTO = {
    Role.TRAINER: pb.COLLECTIVE_ROLE_TRAINER,
    Role.GENERATOR: pb.COLLECTIVE_ROLE_GENERATOR,
}

NCCL_UNIQUE_ID_BYTES = 128


class RendezvousError(RuntimeError):
    """The group could not be formed, or was superseded while forming."""


class GroupNotReadyError(RendezvousError):
    """The group did not reach READY before the deadline.

    Carries the participants MX is still waiting on, because "the collective
    hung" is not actionable and "trainer slot t3 never joined" is.
    """

    def __init__(self, group_id: str, missing: list[str], waited_s: float) -> None:
        self.group_id = group_id
        self.missing = missing
        detail = ", ".join(missing[:8]) if missing else "no slot detail available"
        super().__init__(
            f"collective group {group_id} did not reach READY within {waited_s:.0f}s; "
            f"still waiting on: {detail}"
        )


class EpochChangedError(RendezvousError):
    """The group's membership or plan changed under this worker.

    The caller must rebuild: a communicator created against the old epoch, or a
    plan fetched for it, no longer describes this group.
    """

    def __init__(self, group_id: str, expected: int, actual: int) -> None:
        self.group_id = group_id
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"collective group {group_id} moved from epoch {expected} to {actual}; "
            "the cached communicator and plan must be rebuilt"
        )


@dataclass(frozen=True)
class LaneMembership:
    """This worker's placement in one lane."""

    lane_id: int
    kind: str
    rank_in_lane: int
    world_size: int


@dataclass(frozen=True)
class Membership:
    """What this worker needs in order to build its communicators."""

    group_id: str
    epoch: int
    lanes: tuple[LaneMembership, ...]
    is_bootstrap_leader: bool

    def lane(self, lane_id: int) -> LaneMembership:
        for lane in self.lanes:
            if lane.lane_id == lane_id:
                return lane
        raise KeyError(f"this worker has no assignment in lane {lane_id}")

    @property
    def reshard_lanes(self) -> tuple[LaneMembership, ...]:
        return tuple(lane for lane in self.lanes if lane.kind == "RESHARD")

    @property
    def broadcast_lane(self) -> LaneMembership:
        for lane in self.lanes:
            if lane.kind == "BROADCAST":
                return lane
        raise KeyError("this worker has no broadcast lane assignment")


def _lane_kind(value: int) -> str:
    if value == pb.LANE_KIND_RESHARD:
        return "RESHARD"
    if value == pb.LANE_KIND_BROADCAST:
        return "BROADCAST"
    return "UNSPECIFIED"


class CollectiveRendezvous:
    """Client for ``RefitCollectiveService``.

    One instance per worker process. It holds no communicator and allocates no
    buffers; it only resolves *where this rank sits* and *when it is safe to
    enter*.
    """

    def __init__(
        self,
        channel: grpc.Channel,
        *,
        rpc_timeout_s: float = 30.0,
    ) -> None:
        self._stub = pb_grpc.RefitCollectiveServiceStub(channel)
        self._rpc_timeout_s = rpc_timeout_s

    def join(
        self,
        *,
        model_name: str,
        trainer_slots: list[str],
        generator_slots: list[str],
        source_partition_count: int,
        slot_id: str,
        worker_id: str,
        role: Role,
        index_in_role: int,
        plan_digest: str,
        source_partition: int | None = None,
        plan_endpoint: str | None = None,
    ) -> Membership:
        """Ask MX to admit this worker, and take the rank it assigns.

        The membership declaration must be byte-identical across every
        participant of one operation: MX hashes it into the group identity, so
        a worker that declares a different set resolves a *different* group and
        waits there alone rather than corrupting the real one.
        """
        spec = pb.CollectiveGroupSpec(
            model_name=model_name,
            expected_trainer_slots=trainer_slots,
            expected_generator_slots=generator_slots,
            source_partition_count=source_partition_count,
        )
        request = pb.JoinCollectiveGroupRequest(
            spec=spec,
            slot_id=slot_id,
            worker_id=worker_id,
            role=_ROLE_TO_PROTO[role],
            index_in_role=index_in_role,
            plan_digest=plan_digest,
        )
        if source_partition is not None:
            request.source_partition = source_partition
        if plan_endpoint is not None:
            request.plan_source.CopyFrom(
                pb.PlanSource(
                    worker_id=worker_id,
                    endpoint=plan_endpoint,
                    digest=plan_digest,
                )
            )

        response = self._stub.JoinCollectiveGroup(request, timeout=self._rpc_timeout_s)
        return Membership(
            group_id=response.group_id,
            epoch=response.epoch,
            lanes=tuple(
                LaneMembership(
                    lane_id=a.lane_id,
                    kind=_lane_kind(a.kind),
                    rank_in_lane=a.rank_in_lane,
                    world_size=a.world_size,
                )
                for a in response.assignments
            ),
            is_bootstrap_leader=response.is_bootstrap_leader,
        )

    def publish_bootstrap(
        self,
        *,
        group_id: str,
        epoch: int,
        lane_id: int,
        worker_id: str,
        nccl_unique_id: bytes,
    ) -> None:
        """Post one lane's identifier, stamped with the epoch it was made for.

        A publication naming a superseded epoch is rejected by MX rather than
        applied, so a slow leader cannot overwrite the identifier a newer
        membership is already initializing against.
        """
        if len(nccl_unique_id) != NCCL_UNIQUE_ID_BYTES:
            raise ValueError(
                f"nccl_unique_id must be {NCCL_UNIQUE_ID_BYTES} bytes, "
                f"got {len(nccl_unique_id)}"
            )
        try:
            self._stub.PublishGroupBootstrap(
                pb.PublishGroupBootstrapRequest(
                    group_id=group_id,
                    epoch=epoch,
                    lane_id=lane_id,
                    worker_id=worker_id,
                    nccl_unique_id=nccl_unique_id,
                ),
                timeout=self._rpc_timeout_s,
            )
        except grpc.RpcError as error:
            if error.code() is grpc.StatusCode.FAILED_PRECONDITION:
                raise EpochChangedError(group_id, epoch, -1) from error
            raise

    def await_ready(
        self,
        *,
        group_id: str,
        epoch: int,
        timeout_s: float | None = None,
        poll_interval_s: float | None = None,
    ) -> pb.CollectiveGroup:
        """Block until MX reports the group READY at ``epoch``.

        Polling rather than a server stream is deliberate: a stalled peer then
        surfaces as a client-side deadline carrying the group's own participant
        list, instead of as a stream that never yields.

        Raises :class:`EpochChangedError` if the epoch moves while waiting --
        that is not a retryable condition, it means the caller's plan and
        communicator are stale.
        """
        timeout_s = timeout_s if timeout_s is not None else envs.MX_NCCL_REFIT_GROUP_TIMEOUT_S
        poll_interval_s = (
            poll_interval_s
            if poll_interval_s is not None
            else envs.MX_NCCL_REFIT_POLL_INTERVAL_S
        )
        deadline = time.monotonic() + timeout_s
        group = None

        while True:
            group = self._stub.GetCollectiveGroup(
                pb.GetCollectiveGroupRequest(group_id=group_id),
                timeout=self._rpc_timeout_s,
            )
            if group.epoch != epoch:
                raise EpochChangedError(group_id, epoch, group.epoch)
            if group.state == pb.COLLECTIVE_GROUP_STATE_READY:
                return group
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GroupNotReadyError(group_id, _missing_slots(group), timeout_s)
            time.sleep(min(poll_interval_s, remaining))

    def report(
        self,
        *,
        operation_id: str,
        group_id: str,
        epoch: int,
        worker_id: str,
        succeeded: bool,
        message: str = "",
    ) -> pb.CollectiveTransfer:
        """Record this worker's terminal result for one refit."""
        if not succeeded and not message:
            raise ValueError("a failed report must carry a message")
        return self._stub.ReportCollectiveTransfer(
            pb.ReportCollectiveTransferRequest(
                operation_id=operation_id,
                group_id=group_id,
                epoch=epoch,
                worker_id=worker_id,
                succeeded=succeeded,
                message=message,
            ),
            timeout=self._rpc_timeout_s,
        )


def _missing_slots(group: pb.CollectiveGroup) -> list[str]:
    """Which expected slots have not been admitted yet.

    Read off the broadcast lane, which is the only one every participant joins,
    so it is the single place the full admitted set is visible.
    """
    admitted: set[str] = set()
    for lane in group.lanes:
        if lane.kind == pb.LANE_KIND_BROADCAST:
            admitted = {p.slot_id for p in lane.participants}
            break

    expected = list(group.expected_trainer_slots) + list(group.expected_generator_slots)
    missing = [slot for slot in expected if slot not in admitted]
    if missing:
        return missing

    # Every slot is present, so readiness is waiting on a lane whose bootstrap
    # identifier has not been posted for this epoch yet.
    return [
        f"lane {lane.lane_id} bootstrap"
        for lane in group.lanes
        if lane.bootstrap_epoch != group.epoch
    ]
