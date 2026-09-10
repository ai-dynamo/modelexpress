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

import math
import threading
import time
from dataclasses import dataclass

import grpc

from .. import refit_collective_pb2 as pb
from .. import refit_collective_pb2_grpc as pb_grpc
from .. import refit_pb2, refit_pb2_grpc
from . import envs
from .types import Role

_ROLE_TO_PROTO = {
    Role.TRAINER: pb.COLLECTIVE_ROLE_TRAINER,
    Role.GENERATOR: pb.COLLECTIVE_ROLE_GENERATOR,
}

_ROLE_TO_WORKER_PROTO = {
    Role.TRAINER: refit_pb2.WORKER_ROLE_TRAINER,
    Role.GENERATOR: refit_pb2.WORKER_ROLE_GENERATOR,
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

    ``actual`` is ``-1`` when the current epoch could not be read back.
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


@dataclass(frozen=True)
class _WorkerRegistrationSpec:
    worker_id: str
    role: Role
    model_name: str
    endpoint: str


def _lane_kind(value: int) -> str:
    if value == pb.LANE_KIND_RESHARD:
        return "RESHARD"
    if value == pb.LANE_KIND_BROADCAST:
        return "BROADCAST"
    return "UNSPECIFIED"


def _positive_finite(value: float, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite number") from error
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value}")
    return parsed


def _expected_assignments(
    *,
    trainer_count: int,
    generator_count: int,
    source_partition_count: int,
    role: Role,
    index_in_role: int,
    source_partition: int | None,
) -> tuple[tuple[LaneMembership, ...], bool]:
    """Mirror the server's lane arithmetic and fail before communicator init."""
    if source_partition_count <= 0:
        raise ValueError("source_partition_count must be positive")
    if trainer_count <= 0 or generator_count <= 0:
        raise ValueError("trainer_slots and generator_slots must not be empty")
    if trainer_count % source_partition_count != 0:
        raise ValueError(
            f"trainer count {trainer_count} is not divisible by "
            f"source_partition_count {source_partition_count}"
        )

    trainers_per_partition = trainer_count // source_partition_count
    reshard_world_size = trainers_per_partition + generator_count
    broadcast_world_size = trainer_count + generator_count
    broadcast_lane_id = source_partition_count

    if role is Role.TRAINER:
        if not 0 <= index_in_role < trainer_count:
            raise ValueError(f"trainer index_in_role {index_in_role} is out of range")
        if source_partition is None:
            raise ValueError("a trainer must declare its source_partition")
        implied_partition = index_in_role // trainers_per_partition
        if source_partition != implied_partition:
            raise ValueError(
                f"trainer index_in_role {index_in_role} implies source partition "
                f"{implied_partition}, not {source_partition}"
            )
        reshard_rank = index_in_role % trainers_per_partition
        return (
            (
                LaneMembership(
                    source_partition,
                    "RESHARD",
                    reshard_rank,
                    reshard_world_size,
                ),
                LaneMembership(
                    broadcast_lane_id,
                    "BROADCAST",
                    index_in_role,
                    broadcast_world_size,
                ),
            ),
            reshard_rank == 0,
        )

    if role is Role.GENERATOR:
        if not 0 <= index_in_role < generator_count:
            raise ValueError(f"generator index_in_role {index_in_role} is out of range")
        if source_partition is not None:
            raise ValueError("a generator must not declare a source_partition")
        lanes = tuple(
            LaneMembership(
                lane_id,
                "RESHARD",
                trainers_per_partition + index_in_role,
                reshard_world_size,
            )
            for lane_id in range(source_partition_count)
        )
        return (
            lanes
            + (
                LaneMembership(
                    broadcast_lane_id,
                    "BROADCAST",
                    trainer_count + index_in_role,
                    broadcast_world_size,
                ),
            ),
            False,
        )

    raise ValueError(f"unsupported collective role {role!r}")


def _validate_assignments(
    response: pb.CollectiveGroupMembership,
    expected: tuple[LaneMembership, ...],
    expected_leader: bool,
) -> tuple[LaneMembership, ...]:
    if not response.group_id or response.epoch <= 0:
        raise RendezvousError("MX returned an invalid collective group identity or epoch")

    actual = tuple(
        LaneMembership(
            lane_id=assignment.lane_id,
            kind=_lane_kind(assignment.kind),
            rank_in_lane=assignment.rank_in_lane,
            world_size=assignment.world_size,
        )
        for assignment in response.assignments
    )
    actual_by_lane = {lane.lane_id: lane for lane in actual}
    expected_by_lane = {lane.lane_id: lane for lane in expected}
    if len(actual_by_lane) != len(actual) or actual_by_lane != expected_by_lane:
        raise RendezvousError(
            "MX returned lane assignments that disagree with the client-side "
            f"rank mirror; expected {expected}, got {actual}"
        )
    if response.is_bootstrap_leader != expected_leader:
        raise RendezvousError(
            "MX returned a bootstrap-leader flag that disagrees with the "
            "client-side rank mirror"
        )
    return actual


class CollectiveRendezvous:
    """Client for ``RefitCollectiveService``.

    One instance per worker process. It holds no communicator and allocates no
    buffers; it registers and renews that process's liveness lease, then resolves
    *where this rank sits* and *when it is safe to enter*.
    """

    def __init__(
        self,
        channel: grpc.Channel,
        *,
        rpc_timeout_s: float = 30.0,
        registration_ttl_s: int | None = None,
    ) -> None:
        self._stub = pb_grpc.RefitCollectiveServiceStub(channel)
        self._registration_stub = refit_pb2_grpc.RefitServiceStub(channel)
        self._rpc_timeout_s = _positive_finite(rpc_timeout_s, "rpc_timeout_s")
        registration_ttl_s = (
            registration_ttl_s
            if registration_ttl_s is not None
            else envs.MX_NCCL_REFIT_REGISTRATION_TTL_S
        )
        if (
            isinstance(registration_ttl_s, bool)
            or not isinstance(registration_ttl_s, int)
            or not 0 < registration_ttl_s <= 0xFFFFFFFF
        ):
            raise ValueError("registration_ttl_s must be a positive uint32")
        self._registration_ttl_s = registration_ttl_s
        self._registration_lock = threading.Lock()
        self._registration_stop = threading.Event()
        self._registration_thread: threading.Thread | None = None
        self._registration: _WorkerRegistrationSpec | None = None
        self._closed = False

    def _register_worker(self, registration: _WorkerRegistrationSpec) -> None:
        self._registration_stub.RegisterWorker(
            refit_pb2.RegisterWorkerRequest(
                worker=refit_pb2.WorkerRegistration(
                    worker_id=registration.worker_id,
                    role=_ROLE_TO_WORKER_PROTO[registration.role],
                    model_name=registration.model_name,
                    endpoint=registration.endpoint,
                ),
                ttl_seconds=self._registration_ttl_s,
            ),
            timeout=self._rpc_timeout_s,
        )

    def _start_registration_renewal(self) -> None:
        registration = self._registration
        if registration is None:
            raise RuntimeError("worker registration must exist before renewal starts")
        self._registration_thread = threading.Thread(
            target=self._renew_worker_registration,
            name=f"modelexpress-collective-renew-{registration.worker_id}",
            daemon=True,
        )
        self._registration_thread.start()

    def _ensure_worker_registration(self, registration: _WorkerRegistrationSpec) -> None:
        """Synchronously establish liveness before joining the collective group."""
        with self._registration_lock:
            if self._closed:
                raise RendezvousError("the collective rendezvous is closed")
            if self._registration is not None and self._registration != registration:
                raise RendezvousError(
                    "one CollectiveRendezvous cannot register more than one worker identity"
                )
            self._registration = registration
            # Refresh synchronously on every join. READY is allowed to depend on
            # this lease, so joining with only a best-effort background renewal
            # would race the server's liveness gate.
            self._register_worker(registration)
            if self._registration_thread is None:
                self._start_registration_renewal()

    def _renew_worker_registration(self) -> None:
        interval_s = max(self._registration_ttl_s / 3, 0.1)
        while not self._registration_stop.wait(interval_s):
            registration = self._registration
            if registration is None:
                continue
            try:
                self._register_worker(registration)
            except grpc.RpcError:
                # A later renewal retries after a transient control-plane error.
                # If failures persist, the server lets the lease expire and
                # moves the collective out of READY.
                continue

    def close(self) -> None:
        """Stop lease renewal; the server reclaims liveness after the TTL."""
        with self._registration_lock:
            if self._closed:
                return
            self._closed = True
            self._registration_stop.set()
            thread = self._registration_thread
        if thread is not None:
            thread.join()
        with self._registration_lock:
            self._registration_thread = None

    def __enter__(self) -> CollectiveRendezvous:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    def join(
        self,
        *,
        model_name: str,
        trainer_slots: list[str],
        generator_slots: list[str],
        source_partition_count: int,
        slot_id: str,
        worker_id: str,
        worker_endpoint: str | None = None,
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

        ``worker_endpoint`` is registration metadata, not a collective data
        path. Workers without a peer service use the stable opaque
        ``collective://<worker_id>`` marker; a plan-serving trainer advertises
        its reachable address separately through ``plan_endpoint``.
        """
        if not isinstance(role, Role):
            raise ValueError(f"unsupported collective role {role!r}")
        if len(set(trainer_slots)) != len(trainer_slots):
            raise ValueError("trainer_slots must not contain duplicates")
        if len(set(generator_slots)) != len(generator_slots):
            raise ValueError("generator_slots must not contain duplicates")
        if any(not slot for slot in trainer_slots + generator_slots):
            raise ValueError("collective slot ids must not be empty")
        role_slots = trainer_slots if role is Role.TRAINER else generator_slots
        if slot_id not in role_slots:
            raise ValueError(f"slot_id {slot_id!r} is not declared for role {role.value}")
        expected_assignments, expected_leader = _expected_assignments(
            trainer_count=len(trainer_slots),
            generator_count=len(generator_slots),
            source_partition_count=source_partition_count,
            role=role,
            index_in_role=index_in_role,
            source_partition=source_partition,
        )
        if plan_endpoint is not None and not (role is Role.TRAINER and index_in_role == 0):
            raise ValueError("only trainer index 0 may advertise the reshard plan endpoint")
        registration_endpoint = worker_endpoint or plan_endpoint or f"collective://{worker_id}"

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

        self._ensure_worker_registration(
            _WorkerRegistrationSpec(
                worker_id=worker_id,
                role=role,
                model_name=model_name,
                endpoint=registration_endpoint,
            )
        )
        response = self._stub.JoinCollectiveGroup(request, timeout=self._rpc_timeout_s)
        assignments = _validate_assignments(response, expected_assignments, expected_leader)
        return Membership(
            group_id=response.group_id,
            epoch=response.epoch,
            lanes=assignments,
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
                # FAILED_PRECONDITION covers more than a stale epoch here: MX
                # also rejects a publisher that is not the lane's live rank 0.
                # Read the epoch back so a rejection that is not an epoch move
                # keeps the server's own explanation instead of being relabelled
                # -- and so the one that is names the epoch it moved to.
                current = self._current_epoch(group_id)
                if current != epoch:
                    raise EpochChangedError(group_id, epoch, current) from error
            raise

    def _current_epoch(self, group_id: str) -> int:
        """The group's epoch now, or ``-1`` when it cannot be read.

        MX reports the epoch that rejected a publication in the status detail
        rather than in a typed field, so it is read back from the group. A
        failure here must not replace the rejection the caller needs to see, so
        it degrades to ``-1``, which no live group ever carries.
        """
        try:
            return self._stub.GetCollectiveGroup(
                pb.GetCollectiveGroupRequest(group_id=group_id),
                timeout=self._rpc_timeout_s,
            ).epoch
        except grpc.RpcError:
            return -1

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
        timeout_s = _positive_finite(
            timeout_s if timeout_s is not None else envs.MX_NCCL_REFIT_GROUP_TIMEOUT_S,
            "timeout_s",
        )
        poll_interval_s = _positive_finite(
            poll_interval_s
            if poll_interval_s is not None
            else envs.MX_NCCL_REFIT_POLL_INTERVAL_S,
            "poll_interval_s",
        )
        deadline = time.monotonic() + timeout_s
        group = None

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GroupNotReadyError(
                    group_id,
                    _missing_slots(group) if group is not None else [],
                    timeout_s,
                )
            try:
                group = self._stub.GetCollectiveGroup(
                    pb.GetCollectiveGroupRequest(group_id=group_id),
                    timeout=min(self._rpc_timeout_s, remaining),
                )
            except grpc.RpcError as error:
                if (
                    error.code() is grpc.StatusCode.DEADLINE_EXCEEDED
                    and time.monotonic() >= deadline
                ):
                    raise GroupNotReadyError(
                        group_id,
                        _missing_slots(group) if group is not None else [],
                        timeout_s,
                    ) from error
                raise
            if group.epoch != epoch:
                raise EpochChangedError(group_id, epoch, group.epoch)
            if group.state == pb.COLLECTIVE_GROUP_STATE_READY:
                return group
            if group.state == pb.COLLECTIVE_GROUP_STATE_RELEASING:
                raise RendezvousError(
                    f"collective group {group_id} is releasing and cannot become READY"
                )
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
    admitted: set[tuple[int, str]] = set()
    for lane in group.lanes:
        if lane.kind == pb.LANE_KIND_BROADCAST:
            admitted = {(p.role, p.slot_id) for p in lane.participants}
            break

    expected = [
        (pb.COLLECTIVE_ROLE_TRAINER, slot, f"trainer slot {slot}")
        for slot in group.expected_trainer_slots
    ] + [
        (pb.COLLECTIVE_ROLE_GENERATOR, slot, f"generator slot {slot}")
        for slot in group.expected_generator_slots
    ]
    missing = [label for role, slot, label in expected if (role, slot) not in admitted]
    if missing:
        return missing

    # Every slot is present, so readiness is waiting on a lane whose bootstrap
    # identifier has not been posted for this epoch yet.
    return [
        f"lane {lane.lane_id} bootstrap"
        for lane in group.lanes
        if lane.bootstrap_epoch != group.epoch
    ]
