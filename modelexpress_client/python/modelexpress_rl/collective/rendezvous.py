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

import logging
import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

import grpc

from .. import refit_collective_pb2 as pb
from .. import refit_collective_pb2_grpc as pb_grpc
from .. import refit_pb2, refit_pb2_grpc
from . import envs
from .types import Role

logger = logging.getLogger("modelexpress_rl.collective.rendezvous")

#: Poll failures that say nothing about whether the group will become READY,
#: so they are retried until the group deadline rather than failing the wait.
_RETRYABLE_POLL_CODES = frozenset(
    {
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.UNAVAILABLE,
    }
)

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


class BootstrapFenceTimeoutError(RendezvousError):
    """Not every admitted slot reached one bootstrap step before its deadline."""

    def __init__(
        self,
        group_id: str,
        lane_id: int,
        missing: list[str],
        waited_s: float,
    ) -> None:
        self.group_id = group_id
        self.lane_id = lane_id
        self.missing = missing
        detail = ", ".join(missing[:8]) if missing else "no slot detail available"
        super().__init__(
            f"collective group {group_id} bootstrap fence {lane_id} did not release "
            f"within {waited_s:.0f}s; still waiting on: {detail}"
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


_KIND_TO_PROTO = {
    "RESHARD": pb.LANE_KIND_RESHARD,
    "BROADCAST": pb.LANE_KIND_BROADCAST,
}
_FENCE_PHASE_TO_PROTO = {
    "PRE_BARRIER": pb.BOOTSTRAP_FENCE_PHASE_PRE_BARRIER,
    "COMPLETE": pb.BOOTSTRAP_FENCE_PHASE_COMPLETE,
}
_RESERVED_SLOT_DELIMITERS = "\0\n\r|,"


@dataclass(frozen=True)
class LaneDeclaration:
    """One communicator the caller wants, and who is on it in rank order.

    MX brokers a bootstrap for it and hands each slot the rank its position
    here gives it. What the lane MEANS -- a pipeline stage, a replica group,
    anything else -- is the caller's business and is never sent.
    """

    lane_id: int
    kind: str
    trainer_slots: tuple[str, ...]
    generator_slots: tuple[str, ...]

    def slots_in_rank_order(self) -> tuple[str, ...]:
        return tuple(self.trainer_slots) + tuple(self.generator_slots)


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


def _required_identifier(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} is required")
    return value


def _canonical_slot_list(slots: Sequence[str], name: str) -> tuple[str, ...]:
    canonical = tuple(sorted(slots))
    if len(set(canonical)) != len(canonical):
        raise ValueError(f"{name} must not contain duplicates")
    return _validated_slot_list(canonical, name)


def _validated_slot_list(slots: Sequence[str], name: str) -> tuple[str, ...]:
    validated = tuple(slots)
    for slot in validated:
        _required_identifier(slot, name)
        if any(delimiter in slot for delimiter in _RESERVED_SLOT_DELIMITERS):
            raise ValueError(f"{name} must not contain Redis record delimiters")
    return validated


def _collective_group_spec(
    *,
    model_name: str,
    trainer_slots: Sequence[str],
    generator_slots: Sequence[str],
    lanes: Sequence[LaneDeclaration],
) -> pb.CollectiveGroupSpec:
    canonical_trainers = _canonical_slot_list(trainer_slots, "trainer_slots")
    canonical_generators = _canonical_slot_list(generator_slots, "generator_slots")
    return pb.CollectiveGroupSpec(
        model_name=_required_identifier(model_name, "model_name"),
        expected_trainer_slots=canonical_trainers,
        expected_generator_slots=canonical_generators,
        lanes=[
            pb.LaneSpec(
                lane_id=lane.lane_id,
                kind=_KIND_TO_PROTO[lane.kind],
                trainer_slots=_validated_slot_list(
                    lane.trainer_slots, "lane trainer_slots"
                ),
                generator_slots=_validated_slot_list(
                    lane.generator_slots, "lane generator_slots"
                ),
            )
            for lane in lanes
        ],
    )


def _expected_assignments(
    *,
    lanes: Sequence[LaneDeclaration],
    slot_id: str,
) -> tuple[tuple[LaneMembership, ...], bool]:
    """Mirror the server's placement and fail before communicator init.

    The server does the same walk over the same declaration, so a disagreement
    here means the two sides would have entered different communicators.
    """
    if not lanes:
        raise ValueError("at least one lane must be declared")
    seen_ids: set[int] = set()
    broadcast = 0
    memberships: list[LaneMembership] = []
    for lane in lanes:
        if lane.lane_id in seen_ids:
            raise ValueError(f"lane_id {lane.lane_id} is declared more than once")
        seen_ids.add(lane.lane_id)
        if lane.kind not in _KIND_TO_PROTO:
            raise ValueError(f"unsupported lane kind {lane.kind!r}")
        if lane.kind == "BROADCAST":
            broadcast += 1
        slots = lane.slots_in_rank_order()
        if len(set(slots)) != len(slots):
            raise ValueError(f"lane {lane.lane_id} declares a slot more than once")
        if slot_id in slots:
            memberships.append(
                LaneMembership(
                    lane.lane_id,
                    lane.kind,
                    slots.index(slot_id),
                    len(slots),
                )
            )
    if broadcast > 1:
        raise ValueError("at most one broadcast lane may be declared")
    if not memberships:
        raise ValueError(f"slot {slot_id!r} is on none of the declared lanes")
    return tuple(memberships), any(m.rank_in_lane == 0 for m in memberships)


def _validate_assignments(
    response: pb.CollectiveGroupMembership,
    expected: tuple[LaneMembership, ...],
    expected_leader: bool,
) -> tuple[LaneMembership, ...]:
    if not response.group_id or response.epoch <= 0:
        raise RendezvousError(
            "MX returned an invalid collective group identity or epoch"
        )

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

    def _bounded_rpc_timeout(self, deadline: float | None, operation: str) -> float:
        if deadline is None:
            return self._rpc_timeout_s
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise TimeoutError(f"{operation} exhausted its deadline")
        return min(self._rpc_timeout_s, remaining_s)

    def _register_worker(
        self,
        registration: _WorkerRegistrationSpec,
        *,
        deadline: float | None = None,
    ) -> None:
        self._registration_stub.RegisterWorker(
            refit_pb2.RegisterWorkerRequest(
                worker=refit_pb2.WorkerRegistration(
                    worker_id=registration.worker_id,
                    role=_ROLE_TO_WORKER_PROTO[registration.role],
                    model_name=registration.model_name,
                ),
                ttl_seconds=self._registration_ttl_s,
            ),
            timeout=self._bounded_rpc_timeout(
                deadline, "collective worker registration"
            ),
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

    def _ensure_worker_registration(
        self,
        registration: _WorkerRegistrationSpec,
        *,
        deadline: float | None = None,
    ) -> None:
        """Synchronously establish liveness before joining the collective group."""
        if deadline is None:
            acquired = self._registration_lock.acquire()
        else:
            acquired = self._registration_lock.acquire(
                timeout=self._bounded_rpc_timeout(
                    deadline, "collective worker registration lock"
                )
            )
        if not acquired:
            raise TimeoutError(
                "collective worker registration lock exhausted its deadline"
            )
        try:
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
            self._register_worker(registration, deadline=deadline)
            if self._registration_thread is None:
                self._start_registration_renewal()
        finally:
            self._registration_lock.release()

    def _renew_worker_registration(self) -> None:
        interval_s = max(self._registration_ttl_s / 3, 0.1)
        while not self._registration_stop.wait(interval_s):
            registration = self._registration
            if registration is None:
                continue
            try:
                self._register_worker(registration)
            except Exception:  # noqa: BLE001 - the renewal loop must outlive any one failure
                # A later renewal retries after a transient control-plane error.
                # If failures persist, the server lets the lease expire and
                # moves the collective out of READY.
                #
                # Not narrowed to grpc.RpcError: anything else escaping here
                # ends the thread while _registration_thread stays set, so no
                # replacement ever starts and the lease expires with nothing
                # naming the cause. An RPC on a closed channel raises
                # ValueError on several grpcio versions.
                logger.warning(
                    "collective worker registration renewal failed", exc_info=True
                )
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
        lanes: Sequence[LaneDeclaration],
        slot_id: str,
        worker_id: str,
        role: Role,
        index_in_role: int,
        plan_digest: str,
        plan_endpoint: str | None = None,
        timeout_s: float | None = None,
    ) -> Membership:
        """Ask MX to admit this worker, and take the rank it assigns.

        The membership declaration must be byte-identical across every
        participant of one operation: MX hashes it into the group identity, so
        a worker that declares a different set resolves a *different* group and
        waits there alone rather than corrupting the real one.

        A plan-serving trainer advertises its reachable address through
        ``plan_endpoint``. Worker registration carries no endpoint of its own,
        so only trainer index 0 may advertise a plan source and only under its
        own ``worker_id``.
        """
        if not isinstance(role, Role):
            raise ValueError(f"unsupported collective role {role!r}")
        canonical_trainers = _canonical_slot_list(trainer_slots, "trainer_slots")
        canonical_generators = _canonical_slot_list(generator_slots, "generator_slots")
        role_slots = (
            canonical_trainers if role is Role.TRAINER else canonical_generators
        )
        if slot_id not in role_slots:
            raise ValueError(
                f"slot_id {slot_id!r} is not declared for role {role.value}"
            )
        canonical_index = role_slots.index(slot_id)
        expected_assignments, expected_leader = _expected_assignments(
            lanes=lanes,
            slot_id=slot_id,
        )
        if plan_endpoint is not None and not (
            role is Role.TRAINER and canonical_index == 0
        ):
            raise ValueError(
                "only trainer index 0 may advertise the reshard plan endpoint"
            )

        spec = _collective_group_spec(
            model_name=model_name,
            trainer_slots=canonical_trainers,
            generator_slots=canonical_generators,
            lanes=lanes,
        )
        request = pb.JoinCollectiveGroupRequest(
            spec=spec,
            slot_id=slot_id,
            worker_id=worker_id,
            role=_ROLE_TO_PROTO[role],
            index_in_role=canonical_index,
            plan_digest=plan_digest,
        )
        if plan_endpoint is not None:
            request.plan_source.CopyFrom(
                pb.PlanSource(
                    worker_id=worker_id,
                    endpoint=plan_endpoint,
                    digest=plan_digest,
                )
            )

        deadline = (
            None
            if timeout_s is None
            else time.monotonic() + _positive_finite(timeout_s, "timeout_s")
        )
        self._ensure_worker_registration(
            _WorkerRegistrationSpec(
                worker_id=worker_id,
                role=role,
                model_name=model_name,
            ),
            deadline=deadline,
        )
        response = self._stub.JoinCollectiveGroup(
            request,
            timeout=self._bounded_rpc_timeout(deadline, "collective group join"),
        )
        assignments = _validate_assignments(
            response, expected_assignments, expected_leader
        )
        return Membership(
            group_id=response.group_id,
            epoch=response.epoch,
            lanes=assignments,
            is_bootstrap_leader=response.is_bootstrap_leader,
        )

    def create_transfer(
        self,
        *,
        model_name: str,
        trainer_slots: list[str],
        generator_slots: list[str],
        lanes: Sequence[LaneDeclaration],
        version_id: str,
        idempotency_key: str,
    ) -> pb.CollectiveTransfer:
        """Create one idempotent transfer operation for an exact group spec."""
        return self._stub.CreateCollectiveTransfer(
            pb.CreateCollectiveTransferRequest(
                spec=_collective_group_spec(
                    model_name=model_name,
                    trainer_slots=trainer_slots,
                    generator_slots=generator_slots,
                    lanes=lanes,
                ),
                version_id=_required_identifier(version_id, "version_id"),
                idempotency_key=_required_identifier(
                    idempotency_key, "idempotency_key"
                ),
            ),
            timeout=self._rpc_timeout_s,
        )

    def get_transfer(self, operation_id: str) -> pb.CollectiveTransfer:
        """Read the current state of one collective transfer operation."""
        return self._stub.GetCollectiveTransfer(
            pb.GetCollectiveTransferRequest(
                operation_id=_required_identifier(operation_id, "operation_id")
            ),
            timeout=self._rpc_timeout_s,
        )

    def delete_transfer(self, operation_id: str) -> pb.CollectiveTransfer:
        """Delete one terminal collective transfer operation."""
        return self._stub.DeleteCollectiveTransfer(
            pb.DeleteCollectiveTransferRequest(
                operation_id=_required_identifier(operation_id, "operation_id")
            ),
            timeout=self._rpc_timeout_s,
        )

    def publish_bootstrap(
        self,
        *,
        group_id: str,
        epoch: int,
        lane_id: int,
        worker_id: str,
        nccl_unique_id: bytes,
        timeout_s: float | None = None,
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
        deadline = (
            None
            if timeout_s is None
            else time.monotonic() + _positive_finite(timeout_s, "timeout_s")
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
                timeout=self._bounded_rpc_timeout(
                    deadline, "collective bootstrap publication"
                ),
            )
        except grpc.RpcError as error:
            if error.code() is grpc.StatusCode.FAILED_PRECONDITION:
                # FAILED_PRECONDITION covers more than a stale epoch here: MX
                # also rejects a publisher that is not the lane's live rank 0.
                # Read the epoch back so a rejection that is not an epoch move
                # keeps the server's own explanation instead of being relabelled
                # -- and so the one that is names the epoch it moved to.
                current = self._current_epoch(group_id, deadline=deadline)
                if current != epoch:
                    raise EpochChangedError(group_id, epoch, current) from error
            raise

    def _current_epoch(self, group_id: str, *, deadline: float | None = None) -> int:
        """The group's epoch now, or ``-1`` when it cannot be read.

        MX reports the epoch that rejected a publication in the status detail
        rather than in a typed field, so it is read back from the group. A
        failure here must not replace the rejection the caller needs to see, so
        it degrades to ``-1``, which no live group ever carries.
        """
        try:
            return self._stub.GetCollectiveGroup(
                pb.GetCollectiveGroupRequest(group_id=group_id),
                timeout=self._bounded_rpc_timeout(
                    deadline, "collective epoch readback"
                ),
            ).epoch
        except (grpc.RpcError, TimeoutError):
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
                # The per-RPC timeout is much shorter than the group timeout,
                # so one slow GetCollectiveGroup - or a control-plane restart
                # answering UNAVAILABLE - must not fail a rendezvous that has
                # most of its deadline left. Same posture as the registration
                # renewal above.
                if error.code() not in _RETRYABLE_POLL_CODES:
                    raise
                if time.monotonic() >= deadline:
                    raise GroupNotReadyError(
                        group_id,
                        _missing_slots(group) if group is not None else [],
                        timeout_s,
                    ) from error
                time.sleep(max(0.0, min(poll_interval_s, deadline - time.monotonic())))
                continue
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

    def await_bootstrap_fence(
        self,
        *,
        group_id: str,
        epoch: int,
        lane_id: int,
        slot_id: str,
        worker_id: str,
        timeout_s: float,
        phase: str = "PRE_BARRIER",
        poll_interval_s: float | None = None,
    ) -> None:
        """Idempotently arrive and wait for every admitted slot at one step."""
        timeout_s = _positive_finite(timeout_s, "timeout_s")
        poll_interval_s = _positive_finite(
            poll_interval_s
            if poll_interval_s is not None
            else envs.MX_NCCL_REFIT_POLL_INTERVAL_S,
            "poll_interval_s",
        )
        deadline = time.monotonic() + timeout_s
        try:
            phase_proto = _FENCE_PHASE_TO_PROTO[phase]
        except KeyError as error:
            raise ValueError(f"unsupported bootstrap fence phase {phase!r}") from error
        missing: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BootstrapFenceTimeoutError(group_id, lane_id, missing, timeout_s)
            try:
                fence = self._stub.ReachCollectiveBootstrapFence(
                    pb.ReachCollectiveBootstrapFenceRequest(
                        group_id=group_id,
                        epoch=epoch,
                        lane_id=lane_id,
                        slot_id=slot_id,
                        worker_id=worker_id,
                        phase=phase_proto,
                    ),
                    timeout=min(self._rpc_timeout_s, remaining),
                )
            except grpc.RpcError as error:
                if error.code() is grpc.StatusCode.FAILED_PRECONDITION:
                    current = self._current_epoch(group_id, deadline=deadline)
                    if current != epoch:
                        raise EpochChangedError(group_id, epoch, current) from error
                if error.code() not in _RETRYABLE_POLL_CODES:
                    raise
            else:
                if (
                    fence.group_id != group_id
                    or fence.epoch != epoch
                    or fence.lane_id != lane_id
                    or fence.phase != phase_proto
                ):
                    raise RendezvousError(
                        "MX returned a bootstrap fence for a different group, "
                        "epoch, or lane"
                    )
                missing = list(fence.missing_slots)
                if missing != sorted(missing):
                    raise RendezvousError(
                        "MX returned non-deterministically ordered missing fence slots"
                    )
                if fence.released:
                    if missing:
                        raise RendezvousError(
                            "MX released a bootstrap fence with missing slots"
                        )
                    return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise BootstrapFenceTimeoutError(group_id, lane_id, missing, timeout_s)
            time.sleep(min(poll_interval_s, remaining))

    def abort_bootstrap(
        self,
        *,
        group_id: str,
        epoch: int,
        slot_id: str,
        worker_id: str,
        message: str,
        timeout_s: float,
    ) -> pb.CollectiveGroup:
        """Atomically fence a failed READY epoch before the cohort retries."""
        deadline = time.monotonic() + _positive_finite(timeout_s, "timeout_s")
        try:
            return self._stub.AbortCollectiveBootstrap(
                pb.AbortCollectiveBootstrapRequest(
                    group_id=group_id,
                    epoch=epoch,
                    slot_id=slot_id,
                    worker_id=worker_id,
                    message=_required_identifier(message, "message"),
                ),
                timeout=self._bounded_rpc_timeout(
                    deadline, "collective bootstrap abort"
                ),
            )
        except grpc.RpcError as error:
            if error.code() is grpc.StatusCode.FAILED_PRECONDITION:
                current = self._current_epoch(group_id, deadline=deadline)
                if current != epoch:
                    raise EpochChangedError(group_id, epoch, current) from error
            raise

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
    # A digest disagreement is the CAUSE, not a symptom: every lane bootstrap
    # will also look stale, so reporting the lanes would point at the wrong
    # subsystem. Say what actually happened.
    if group.disagreeing_slots:
        return [
            f"plan digest disagreement on slot {slot}"
            for slot in group.disagreeing_slots
        ]

    # The broadcast lane is the one place the full admitted set is visible in
    # a single read, but a caller need not declare one, so fall back to the
    # union across every lane rather than reporting everyone as missing.
    admitted: set[tuple[int, str]] = set()
    broadcast = [lane for lane in group.lanes if lane.kind == pb.LANE_KIND_BROADCAST]
    for lane in broadcast or group.lanes:
        admitted |= {(p.role, p.slot_id) for p in lane.participants}

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
