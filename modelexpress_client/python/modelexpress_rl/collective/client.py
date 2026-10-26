# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-sided refit clients for the NCCL M2N collective path.

``RefitClientTrainer`` and ``RefitClientGenerator`` are mirror images: the same
lifecycle, the same sequencing rules, differing only in which engine boundary
they drive and which end of each transfer they own.

The sequencing is the contract, and two of its rules are load-bearing here:

- ``compute_plan`` must complete on every worker before any ``start_weight_update``
  begins, because it is where the communicators come into existence;
- the trainer must not enter the collective before MX reports the group READY,
  because a push into a generator that has not prepared its destinations has
  nowhere to land.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from contextlib import nullcontext
from typing import Any

from .. import refit_collective_pb2 as pb
from . import envs
from .backend import (
    _QUARANTINED_REFIT_BUFFERS,
    DEFAULT_LAYER_GROUP,
    NcclM2nReceiver,
    NcclM2nSender,
    installed_m2n_build,
    require_nccl_m2n,
)
from .comm import (
    CommunicatorCache,
    CommunicatorWedgedError,
    LaneCommunicator,
    LaneKey,
    new_unique_id,
    stream_handle,
)
from .plan import (
    DEFAULT_RECEIVER_PROTOCOL,
    plan_digest,
    validate_coverage,
    validate_layer_groups,
)
from .rendezvous import (
    CollectiveRendezvous,
    LaneDeclaration,
    Membership,
    RendezvousError,
)
from .spi import Loader, Publisher, resolve_specs
from .types import ReshardPlan, Role

logger = logging.getLogger("modelexpress_rl.collective.client")

#: Delivery attempts for a round's success report before the binding is kept
#: for ``retry_report``. The report is idempotent server-side (a worker is
#: recorded once and terminal states are immutable), so resending is safe.
_SUCCESS_REPORT_ATTEMPTS = 3
#: Sleep before the second attempt; it doubles for each one after.
_SUCCESS_REPORT_BACKOFF_S = 0.2
#: Operation states a success report can come back with when another
#: participant already ended the round.
_ENDED_TRANSFER_STATES = frozenset(
    {pb.COLLECTIVE_TRANSFER_STATE_FAILED, pb.COLLECTIVE_TRANSFER_STATE_ABORTED}
)
#: gRPC codes that say nothing about whether the report landed.
_TRANSIENT_REPORT_CODES = frozenset({"UNAVAILABLE", "DEADLINE_EXCEEDED"})


def _is_transient_rpc_error(error: BaseException) -> bool:
    code = getattr(error, "code", None)
    if not callable(code):
        return False
    try:
        status = code()
    except Exception:  # noqa: BLE001 - an unreadable code is not transient
        return False
    return getattr(status, "name", None) in _TRANSIENT_REPORT_CODES


def _bootstrap_barrier(
    lane: LaneCommunicator, device: Any, *, timeout_s: float | None = None
) -> None:
    """Full-group barrier used between overlapping communicator initializations.

    Bounded like every other wait past READY. This barrier runs after each
    lane comes up, so a peer that dies between READY and here would otherwise
    block this rank forever: ``MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S`` bounds
    ``Communicator.init`` only, and the transfer deadline does not arm until
    bootstrap is done. The caller's ``except BaseException`` aborts the group,
    so a timeout raised here tears it down and forces a fresh epoch.
    """
    import torch

    device_context = torch.cuda.device(device) if device is not None else nullcontext()
    with device_context:
        barrier = torch.zeros(
            1, dtype=torch.uint8, device=device if device is not None else "cuda"
        )
        stream = lane.stream
        stream_arg = (
            None if stream is None else stream_handle(stream)
        )
        lane.handle.broadcast(
            sendbuf=barrier,
            recvbuf=barrier,
            root=0,
            stream=stream_arg,
        )
        lane.synchronize(
            timeout_s=envs.MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S
            if timeout_s is None
            else timeout_s
        )


class _RefitClientBase:
    def __init__(
        self,
        *,
        rendezvous: CollectiveRendezvous,
        model_name: str,
        trainer_slots: list[str],
        generator_slots: list[str],
        source_partition_count: int,
        slot_id: str,
        worker_id: str,
        index_in_role: int,
        receiver_protocol: str = DEFAULT_RECEIVER_PROTOCOL,
        m2n_abi_version: str = "",
        device: Any = None,
        streams: list[Any] | None = None,
    ) -> None:
        self._rendezvous = rendezvous
        self._model_name = model_name
        self._trainer_slots = list(trainer_slots)
        self._generator_slots = list(generator_slots)
        self._source_partition_count = source_partition_count
        self._slot_id = slot_id
        self._worker_id = worker_id
        self._index_in_role = index_in_role
        self._receiver_protocol = receiver_protocol
        # Peers on different M2N builds must not share a plan digest, so when
        # the caller does not name one, the installed build stands in for it.
        self._m2n_abi_version = m2n_abi_version or installed_m2n_build()
        self._device = device
        self._streams = list(streams) if streams else [None]

        # Both modes change what a rank does inside native collectives, so
        # they are read once, hashed into the digest, and handed to the code
        # that runs them; a later env change cannot split a cohort.
        self._blocking_data_plane = envs.MX_NCCL_REFIT_BLOCKING_DATA_PLANE
        self._grouped_submission = envs.MX_NCCL_REFIT_GROUPED_SUBMISSION
        self._cache = CommunicatorCache(blocking_data_plane=self._blocking_data_plane)
        self._publisher: Publisher | None = None
        self._loader: Loader | None = None
        self._expected_parameters: list[str] | None = None
        self._plan: ReshardPlan | None = None
        self._digest: str | None = None
        self._membership: Membership | None = None
        self._half: NcclM2nSender | NcclM2nReceiver | None = None
        self._groupings: list[list[str]] | None = None
        self._round_started = False
        self._report_pending = False
        self._version: str | None = None
        self._operation_id: str | None = None
        self._round_membership: Membership | None = None
        # Set once a communicator abort wedged. Fatal for the process: nothing
        # the wedged kernel may still use is released after it.
        self._wedged: CommunicatorWedgedError | None = None

    @property
    def membership(self) -> Membership:
        if self._membership is None:
            raise RuntimeError("compute_plan has not run on this worker yet")
        return self._membership

    @property
    def plan(self) -> ReshardPlan:
        if self._plan is None:
            raise RuntimeError("initialize has not run on this worker yet")
        return self._plan

    def _require_round(
        self, version: str, method: str
    ) -> NcclM2nSender | NcclM2nReceiver:
        """Reject a call that does not belong to the round now in flight.

        Every per-round call names the version ``start_weight_update`` opened,
        so a mismatch is refused here rather than moving one round's tensors
        under another version's label on every rank.
        """
        if self._half is None:
            raise RuntimeError(f"compute_plan must run before {method}")
        if not self._round_started:
            raise RuntimeError(f"start_weight_update must run before {method}")
        if self._report_pending:
            raise RuntimeError(
                f"the round for version {self._version!r} already finished and "
                f"only its success report is outstanding; call retry_report() "
                f"instead of {method}"
            )
        if version != self._version:
            raise ValueError(
                f"{method} names version {version!r}, but the round in flight is "
                f"{self._version!r}"
            )
        return self._half

    def _fail_round(self, error: BaseException) -> None:
        """The one exit for a bound round that failed anywhere after binding.

        Order matters. Peers may already be inside the collective waiting on
        this rank, so the lanes are aborted first: that is local and cannot
        hang on the network. The failure is then reported against the bound
        operation, best effort, which moves the group to a fresh epoch. The
        round state is reset last and unconditionally, so a later call cannot
        continue it.

        Normally never raises; the caller re-raises ``error``. The one
        exception is :class:`CommunicatorWedgedError`: if this half's abort
        cannot tear a communicator down (``ncclCommAbort`` stuck on a running
        M2N kernel), the failure is still reported so peers re-form without this
        worker, the round state is still reset, and then the wedge is re-raised.
        It is fatal and the runner must exit the process on it.
        """
        wedged: CommunicatorWedgedError | None = None
        try:
            if self._half is not None:
                self._half.abort()
        except CommunicatorWedgedError as error_wedged:
            wedged = error_wedged
            logger.error(
                "the local half's communicator abort wedged (%s); it cannot be "
                "torn down in this process. Reporting the round FAILED so peers "
                "re-form without this worker, then the process must exit.",
                error_wedged,
            )
        except Exception:  # noqa: BLE001 - the report and reset must still run
            logger.warning(
                "aborting the local half after a failed round raised; reporting "
                "and resetting anyway",
                exc_info=True,
            )
        if self._operation_id is not None:
            try:
                self._report(succeeded=False, message=repr(error))
            except Exception:  # noqa: BLE001 - the original failure is re-raised
                logger.warning(
                    "reporting the failed round to the control plane raised; "
                    "the local half is already aborted so peers are released, "
                    "and the original failure is re-raised",
                    exc_info=True,
                )
        self._reset_round()
        if wedged is None and isinstance(error, CommunicatorWedgedError):
            wedged = error
        if wedged is not None:
            self._wedged = wedged
            raise wedged

    def _abort_group(self, group_id: str) -> None:
        """Abort a group's lanes, keeping a wedge for ``cleanup`` to re-raise.

        A wedged abort still drops every lane, so a later abort finds nothing
        left to wedge on; without recording it here, cleanup would succeed and
        release engine buffers the stuck kernel may still use.
        """
        try:
            self._cache.abort_group(group_id)
        except CommunicatorWedgedError as error:
            self._wedged = error
            raise

    def _close_half(self) -> None:
        """Destroy the current half's M2N handle; its lanes are torn down after."""
        half, self._half = self._half, None
        if half is None:
            return
        try:
            half.close()
        except Exception:  # noqa: BLE001 - teardown of the lanes must proceed
            logger.warning(
                "closing the collective half raised; tearing its lanes down anyway",
                exc_info=True,
            )

    def _release_engine(self, engine: Publisher | Loader) -> None:
        """Let the engine free its buffers, unless a communicator abort wedged.

        After a wedge the M2N kernel may still read or write those buffers, so
        the engine is quarantined for the life of the process instead; the
        process must exit on the wedge. Every other failure frees them.
        """
        if self._wedged is None:
            engine.cleanup()
            return
        _QUARANTINED_REFIT_BUFFERS.append(engine)
        logger.error(
            "a communicator abort wedged; the engine's buffers are kept, not "
            "freed, until the process exits"
        )

    def _reset_round(self) -> None:
        self._round_started = False
        self._version = None
        self._operation_id = None
        self._round_membership = None
        self._report_pending = False

    def _bind_operation(self, version: str, operation_id: str) -> None:
        """Bind the round about to open to its control-plane operation.

        Runs before the engine is prepared and before any NCCL work, so an
        operation created for another version, group or epoch is refused while
        every peer is still outside the collective. The bound operation is the
        one the round's outcome is reported against.
        """
        if self._report_pending:
            raise RuntimeError(
                f"the success report for version {self._version!r} was not "
                "delivered; call retry_report() before starting another round"
            )
        if self._round_started:
            raise RuntimeError(
                f"the round for version {self._version!r} is still in flight; "
                "finish it before starting another"
            )
        self._check_still_current()
        membership = self.membership
        self._rendezvous.bind_transfer(membership, operation_id, version)
        self._operation_id = operation_id
        self._round_membership = membership

    def create_operation(self, version: str, *, key_prefix: str) -> str:
        """Create, or replay, the operation for ``version`` at the current epoch.

        The create is idempotent on ``(key_prefix, group, epoch, version)``, so
        every participant may call this and gets the same operation id back; a
        coordinator that already created the operation can hand its id to
        ``start_weight_update`` instead. Call it before ``start_weight_update``:
        once a peer has reported, the replay no longer returns a PENDING
        operation and is refused.
        """
        return self._rendezvous.create_transfer(
            self.membership, version, key_prefix=key_prefix
        ).operation_id

    def _check_operation_alias(self, operation_id: str | None) -> None:
        if operation_id is not None and operation_id != self._operation_id:
            raise ValueError(
                f"finish_weight_update names operation {operation_id!r}, but the "
                f"round in flight is bound to {self._operation_id!r}"
            )

    def _report(self, *, succeeded: bool, message: str = "") -> Any:
        """Report the bound round's outcome against the identity it was admitted with."""
        if self._operation_id is None or self._round_membership is None:
            raise RuntimeError("no operation is bound to this round")
        return self._rendezvous.report(
            operation_id=self._operation_id,
            group_id=self._round_membership.group_id,
            epoch=self._round_membership.epoch,
            worker_id=self._worker_id,
            succeeded=succeeded,
            message=message,
        )

    def _report_success(self) -> None:
        """Report the bound round as succeeded, then release the binding.

        Transient gRPC errors (UNAVAILABLE, DEADLINE_EXCEEDED) are retried a
        bounded number of times. If they persist, the operation binding and
        round identity are kept and the error is raised, so ``retry_report``
        can deliver the report later instead of leaving the operation RUNNING
        with nothing left to report it. Any other error is the control plane's
        answer, so the binding is released and the error is raised.

        A report against an operation another participant already ended comes
        back FAILED or ABORTED. That is raised too, with the server's failure
        message, after the binding is released: the operation is terminal, so
        there is nothing to retry. RUNNING and COMPLETE are both a normal finish,
        since only the last reporter sees COMPLETE.
        """
        for attempt in range(1, _SUCCESS_REPORT_ATTEMPTS + 1):
            try:
                transfer = self._report(succeeded=True)
            except BaseException as error:
                if not _is_transient_rpc_error(error):
                    self._reset_round()
                    raise
                if attempt == _SUCCESS_REPORT_ATTEMPTS:
                    self._report_pending = True
                    logger.warning(
                        "the success report for operation %s was not delivered "
                        "after %d attempts; the binding is kept for retry_report()",
                        self._operation_id,
                        attempt,
                    )
                    raise
                time.sleep(_SUCCESS_REPORT_BACKOFF_S * 2 ** (attempt - 1))
            else:
                operation_id = self._operation_id
                self._reset_round()
                state = getattr(transfer, "state", None)
                if state in _ENDED_TRANSFER_STATES:
                    raise RendezvousError(
                        f"operation {operation_id} was already "
                        f"{pb.CollectiveTransferState.Name(state)} when this worker "
                        f"reported success: {transfer.failure_message or 'no message'}"
                    )
                return

    def retry_report(self) -> None:
        """Deliver a success report that ``finish_weight_update`` could not.

        Only valid after ``finish_weight_update`` raised a transient error from
        the report. The round's data plane already completed; this resends the
        report against the same operation and releases the binding once it is
        acknowledged.
        """
        if not self._report_pending:
            raise RuntimeError("no undelivered success report is pending")
        self._report_success()

    def _check_still_current(self) -> None:
        """One freshness read before a round prepares the engine.

        A membership change between refits moves the epoch, so this catches a
        cohort that already moved before anything is staged. It is a single
        read, not a lease: a change that lands after it still surfaces at the
        transfer deadline or in the report.
        """
        membership = self.membership
        self._rendezvous.check_current(
            group_id=membership.group_id, epoch=membership.epoch
        )

    def _capture(self, engine: Publisher | Loader, expected: list[str] | None) -> None:
        plan = engine.capture()
        if plan.source_partition_count != self._source_partition_count:
            raise ValueError(
                "captured plan source_partition_count does not match the group spec: "
                f"{plan.source_partition_count} != {self._source_partition_count}"
            )
        if expected is None:
            parameter_names = getattr(engine, "parameter_names", None)
            if not callable(parameter_names):
                raise ValueError(
                    "expected_parameters is required unless the Publisher/Loader "
                    "implements parameter_names(); plan coverage cannot be optional"
                )
            expected = list(parameter_names())
        validate_coverage(plan, list(expected))
        if self._groupings is not None:
            # Before the join, so a grouping this plan cannot execute never
            # reaches READY and bootstraps lanes it would only reject after.
            validate_layer_groups(plan, self._groupings)
        self._plan = plan
        self._digest = plan_digest(
            plan,
            receiver_protocol=self._receiver_protocol,
            m2n_abi_version=self._m2n_abi_version,
            layer_groups=self._groupings,
            blocking_data_plane=self._blocking_data_plane,
            grouped_submission=self._grouped_submission,
        )

    def setup_layer_groups(self, groupings: list[list[str]] | None) -> None:
        """Optional. Without it every bulk parameter is in layer group 0.

        The grouping is part of the readiness digest, so it is fixed before
        ``compute_plan`` and peers that split bulk differently are not admitted
        together. Invoking each group exactly once per round, in the same order
        on every participant, is the caller's contract; the backend refuses a
        repeated group and a finish that skipped one.
        """
        if self._membership is not None:
            raise RuntimeError(
                "setup_layer_groups must run before compute_plan; the grouping "
                "is part of the digest this worker was admitted with"
            )
        self._groupings = (
            [list(group) for group in groupings] if groupings is not None else None
        )

    def _stream_for(self, lane_id: int) -> Any:
        """Spread reshard lanes over the configured streams.

        Per-partition lanes are independent communicators, so giving them
        different streams is what lets them actually overlap rather than
        serialize behind one another.
        """
        return self._streams[lane_id % len(self._streams)]

    def _declared_lanes(self) -> list[LaneDeclaration]:
        """The communicators this client wants, from ITS OWN partitioning.

        The split lives here and is never sent as a partition count: MX is
        handed the resulting membership and nothing about what produced it.
        """
        if len(self._trainer_slots) % self._source_partition_count:
            raise ValueError(
                "source_partition_count must divide the trainer slot count: "
                f"{len(self._trainer_slots)} trainer slots, "
                f"{self._source_partition_count} source partitions"
            )
        per_lane = len(self._trainer_slots) // self._source_partition_count
        lanes = [
            LaneDeclaration(
                partition,
                "RESHARD",
                tuple(
                    self._trainer_slots[
                        partition * per_lane : (partition + 1) * per_lane
                    ]
                ),
                tuple(self._generator_slots),
            )
            for partition in range(self._source_partition_count)
        ]
        lanes.append(
            LaneDeclaration(
                self._source_partition_count,
                "BROADCAST",
                tuple(self._trainer_slots),
                tuple(self._generator_slots),
            )
        )
        return lanes

    def _join_and_bootstrap(self, role: Role) -> Membership:
        if self._digest is None:
            raise RuntimeError("initialize must run before compute_plan")
        require_nccl_m2n()
        if self._operation_id is not None:
            self._fail_round(
                RuntimeError(
                    f"compute_plan re-ran while the round for version "
                    f"{self._version!r} was in flight"
                )
            )

        declared = self._declared_lanes()
        membership = self._rendezvous.join(
            model_name=self._model_name,
            trainer_slots=self._trainer_slots,
            generator_slots=self._generator_slots,
            lanes=declared,
            slot_id=self._slot_id,
            worker_id=self._worker_id,
            role=role,
            index_in_role=self._index_in_role,
            plan_digest=self._digest,
        )
        # Which reshard lanes this worker belongs on is a fact about what it
        # just declared, not a re-derivation of a rule the server also applies.
        expected_reshard = {
            lane.lane_id
            for lane in declared
            if lane.kind == "RESHARD" and self._slot_id in lane.slots_in_rank_order()
        }
        actual_reshard = {lane.lane_id for lane in membership.reshard_lanes}
        if actual_reshard != expected_reshard:
            raise RuntimeError(
                "MX returned unexpected reshard-lane membership: "
                f"expected {sorted(expected_reshard)}, got {sorted(actual_reshard)}"
            )
        declared_broadcast = next(
            lane.lane_id for lane in declared if lane.kind == "BROADCAST"
        )
        if membership.broadcast_lane.lane_id != declared_broadcast:
            raise RuntimeError(
                "MX returned an unexpected broadcast lane id: "
                f"expected {declared_broadcast}, got "
                f"{membership.broadcast_lane.lane_id}"
            )

        # The previous half's M2N handle holds resources keyed on the lanes
        # about to be dropped, so it is destroyed before any of them go.
        self._close_half()
        previous = self._membership
        if previous is not None and previous.group_id != membership.group_id:
            self._abort_group(previous.group_id)
        # An epoch move invalidates every cached communicator. The plan was
        # freshly captured before this join and is guarded by its digest.
        dropped = self._cache.invalidate_epoch(membership.group_id, membership.epoch)
        if dropped:
            logger.info(
                "epoch moved to %s; dropped %s stale lane(s)", membership.epoch, dropped
            )

        if membership.is_bootstrap_leader:
            for lane in membership.reshard_lanes:
                if lane.rank_in_lane == 0:
                    self._rendezvous.publish_bootstrap(
                        group_id=membership.group_id,
                        epoch=membership.epoch,
                        lane_id=lane.lane_id,
                        worker_id=self._worker_id,
                        nccl_unique_id=new_unique_id(),
                    )
            broadcast = membership.broadcast_lane
            if broadcast.rank_in_lane == 0:
                self._rendezvous.publish_bootstrap(
                    group_id=membership.group_id,
                    epoch=membership.epoch,
                    lane_id=broadcast.lane_id,
                    worker_id=self._worker_id,
                    nccl_unique_id=new_unique_id(),
                )

        group = self._rendezvous.await_ready(
            group_id=membership.group_id, epoch=membership.epoch
        )
        # A same-slot replacement admitted in a clean FORMING epoch does not
        # bump the epoch, so the displaced generation can still see READY at
        # its own epoch. Confirm this process still holds its slot before it
        # enters a communicator at the replacement's rank.
        holders = {
            participant.worker_id
            for lane in group.lanes
            for participant in lane.participants
            if participant.slot_id == self._slot_id
        }
        if holders != {self._worker_id}:
            raise RendezvousError(
                f"slot {self._slot_id!r} in collective group {membership.group_id} "
                f"is held by {sorted(holders)} rather than worker "
                f"{self._worker_id!r}; this worker was superseded"
            )

        by_lane_id = {lane.lane_id: lane for lane in group.lanes}
        declared = self._declared_lanes()
        all_lane_ids = {lane.lane_id for lane in declared}
        missing = sorted(all_lane_ids - set(by_lane_id))
        if missing:
            raise RuntimeError(
                f"READY group omitted lane(s) assigned to this worker: {missing}"
            )

        # All ranks create the broadcast communicator first. It then provides a
        # full-group barrier after every reshard-lane init. Without those
        # barriers, a PP-stage trainer can enter lane N+1 while generators are
        # still initializing lane N; overlapping communicator creation can hang.
        # Every rank walks the FULL declared lane set, including lanes it is
        # not on, so the barriers line up; membership.lane raises for those.
        lane_order = [membership.broadcast_lane.lane_id] + [
            lane.lane_id for lane in declared if lane.kind == "RESHARD"
        ]
        try:
            for lane_id in lane_order:
                lane_record = by_lane_id[lane_id]
                try:
                    mine = membership.lane(lane_id)
                except KeyError:
                    mine = None
                if mine is not None:
                    self._cache.create(
                        LaneKey(
                            group_id=membership.group_id,
                            epoch=membership.epoch,
                            lane_id=lane_id,
                        ),
                        rank=mine.rank_in_lane,
                        world_size=mine.world_size,
                        unique_id=bytes(lane_record.nccl_unique_id),
                        device=self._device,
                        stream=self._stream_for(lane_id),
                    )

                # A fresh non-blocking communicator can report Success at
                # creation and still fail moments later in its scalable-init
                # trailing phase. Polling every local lane here turns that
                # into a clean init error naming the lane instead of a
                # rejected barrier launch on a peer.
                self._cache.settle_group(membership.group_id, membership.epoch)

                broadcast = self._cache.get(
                    LaneKey(
                        group_id=membership.group_id,
                        epoch=membership.epoch,
                        lane_id=membership.broadcast_lane.lane_id,
                    )
                )
                if broadcast is None:
                    raise RuntimeError(
                        "broadcast communicator was not initialized first"
                    )
                _bootstrap_barrier(broadcast, self._device)
        except BaseException:
            # Fatal for this worker. Its lease would otherwise keep the group
            # READY at this epoch with the bootstrap ids just aborted, and a
            # retry in this process would re-enter them. Closing the
            # rendezvous stops the lease renewal so its TTL moves the epoch,
            # and refuses any further join here: the worker must restart.
            try:
                self._abort_group(membership.group_id)
            finally:
                self._membership = None
                self._close_rendezvous_after_failed_bootstrap(membership)
            raise

        self._membership = membership
        return membership

    def _close_rendezvous_after_failed_bootstrap(self, membership: Membership) -> None:
        logger.error(
            "communicator bootstrap of group %s at epoch %s failed; closing the "
            "rendezvous so this worker's lease expires and the epoch moves. "
            "This worker must be restarted to rejoin.",
            membership.group_id,
            membership.epoch,
        )
        try:
            self._rendezvous.close()
        except Exception:  # noqa: BLE001 - the bootstrap failure is re-raised
            logger.warning(
                "closing the rendezvous after a failed bootstrap raised",
                exc_info=True,
            )

    def cleanup(self) -> None:
        if self._wedged is not None:
            # A wedged abort already tore down what could be torn down; what
            # is left may still be in use by the kernel it could not stop.
            raise self._wedged
        if self._operation_id is not None:
            # Tearing down mid-round is a failure of that round: peers are
            # released and the bound operation is told, rather than left
            # PENDING until its key expires.
            self._fail_round(
                RuntimeError(
                    f"cleanup ran while the round for version {self._version!r} "
                    "was in flight"
                )
            )
        self._close_half()
        if self._membership is not None:
            self._abort_group(self._membership.group_id)
        self._membership = None
        self._half = None
        self._reset_round()


class RefitClientTrainer(_RefitClientBase):
    """Trainer-side lifecycle."""

    def initialize(
        self,
        publisher: Publisher,
        *,
        source_partition: int,
        expected_parameters: list[str] | None = None,
    ) -> None:
        if not 0 <= source_partition < self._source_partition_count:
            raise ValueError(
                f"source_partition must be in [0, {self._source_partition_count}), "
                f"got {source_partition}"
            )
        # The lanes are derived from this slot's position in trainer_slots, so a
        # partition that disagrees would pass READY and then fail on the first
        # publish with peers already inside the collective.
        own_lane = next(
            (
                lane.lane_id
                for lane in self._declared_lanes()
                if lane.kind == "RESHARD" and self._slot_id in lane.trainer_slots
            ),
            None,
        )
        if own_lane is None:
            raise ValueError(
                f"slot {self._slot_id!r} is not on any reshard lane; it must be "
                "one of the trainer slots"
            )
        if own_lane != source_partition:
            raise ValueError(
                f"source_partition {source_partition} does not match the reshard "
                f"lane of slot {self._slot_id!r}, which is {own_lane}"
            )
        self._publisher = publisher
        self._source_partition = source_partition
        self._expected_parameters = (
            list(expected_parameters) if expected_parameters is not None else None
        )
        self._capture(publisher, self._expected_parameters)

    def compute_plan(self) -> Membership:
        if self._publisher is None:
            raise RuntimeError("initialize must run before compute_plan")
        self._capture(self._publisher, self._expected_parameters)
        specs = self._publisher.local_params()
        required = [
            entry.name
            for entry in self.plan.bulk
            if entry.partition_id == self._source_partition
        ] + [entry.name for entry in self.plan.misc]
        # Resolve storage before joining. READY must not include a worker that
        # will discover only afterward that it cannot issue the agreed ops.
        resolve_specs(self.plan, specs, required)
        membership = self._join_and_bootstrap(Role.TRAINER)
        try:
            self._half = NcclM2nSender(
                plan=self.plan,
                specs=specs,
                group_id=membership.group_id,
                epoch=membership.epoch,
                cache=self._cache,
                source_partition=self._source_partition,
                grouped_submission=self._grouped_submission,
            )
            self._half.setup_layer_groups(self._groupings)
        except BaseException:
            self._abort_group(membership.group_id)
            self._membership = None
            self._half = None
            raise
        return membership

    def start_weight_update(self, version: str, operation_id: str) -> None:
        """Open the round for ``version``, bound to ``operation_id``.

        ``operation_id`` is the ``CollectiveTransfer`` created for ``version``
        against this group's current epoch. It is fetched and checked before
        the publisher is prepared or any NCCL work is issued, and the round's
        outcome is reported against it.
        """
        if self._half is None:
            raise RuntimeError("compute_plan must run before start_weight_update")
        if self._publisher is None:
            raise RuntimeError("initialize must run before start_weight_update")
        self._bind_operation(version, operation_id)
        try:
            self._publisher.start_new_round(version)
            self._half.start_weight_update(version)
        except BaseException as error:
            self._fail_round(error)
            raise
        self._round_started = True
        self._version = version

    def publish_weights(
        self, version: str, layer_group_id: int = DEFAULT_LAYER_GROUP
    ) -> None:
        half = self._require_round(version, "publish_weights")
        try:
            half.publish_weights(layer_group_id)
        except BaseException as error:
            self._fail_round(error)
            raise

    def finish_weight_update(
        self, version: str, *, operation_id: str | None = None
    ) -> None:
        """Drain the round and report it against the operation it was bound to.

        ``operation_id`` is optional and only checked: the operation is the one
        ``start_weight_update`` bound.
        """
        half = self._require_round(version, "finish_weight_update")
        self._check_operation_alias(operation_id)
        try:
            half.finish_weight_update(self.membership.broadcast_lane.lane_id)
        except BaseException as error:
            self._fail_round(error)
            raise
        self._report_success()

    def report_failure(
        self,
        operation_id: str,
        error: BaseException,
        *,
        membership: Membership | None = None,
    ) -> None:
        """Report using the admitted round identity, including after transport cleanup."""
        admitted = self._membership if membership is None else membership
        if admitted is None:
            raise RendezvousError(
                "a transfer can only be reported against a group this client "
                "joined; compute_plan has not admitted this worker yet"
            )
        self._rendezvous.report(
            operation_id=operation_id,
            group_id=admitted.group_id,
            epoch=admitted.epoch,
            worker_id=self._worker_id,
            succeeded=False,
            message=repr(error),
        )

    def cleanup(self) -> None:
        # The lanes and the M2N handle may still reference the engine's
        # buffers, so they are released first; the engine frees its buffers
        # after, even if that teardown raised, unless a communicator abort
        # wedged: then they are kept and the process must exit.
        publisher, self._publisher = self._publisher, None
        try:
            super().cleanup()
        finally:
            if publisher is not None:
                self._release_engine(publisher)


class RefitClientGenerator(_RefitClientBase):
    """Generator-side lifecycle.

    Layer groups are pipelined: ``update_weights`` enqueues a group's receives
    and installs the OLDEST group still waiting once
    ``MX_NCCL_REFIT_PIPELINE_DEPTH`` groups are outstanding, so one group is
    installed while the next is on the wire. Every group is installed before
    ``finish_weight_update`` returns. Depth 1 installs each group before
    ``update_weights`` returns, at the cost of serializing install and
    transfer. A depth of N keeps up to N groups' receive buffers live.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._unfinished: deque[int] = deque()
        self._pipeline_depth = 1

    def initialize(
        self,
        loader: Loader,
        *,
        expected_parameters: list[str] | None = None,
    ) -> None:
        self._loader = loader
        self._expected_parameters = (
            list(expected_parameters) if expected_parameters is not None else None
        )
        self._capture(loader, self._expected_parameters)

    def compute_plan(self) -> Membership:
        if self._loader is None:
            raise RuntimeError("initialize must run before compute_plan")
        self._capture(self._loader, self._expected_parameters)
        specs = self._loader.local_params()
        resolve_specs(self.plan, specs)
        membership = self._join_and_bootstrap(Role.GENERATOR)
        try:
            self._half = NcclM2nReceiver(
                plan=self.plan,
                specs=specs,
                group_id=membership.group_id,
                epoch=membership.epoch,
                cache=self._cache,
                grouped_submission=self._grouped_submission,
            )
            self._half.setup_layer_groups(self._groupings)
        except BaseException:
            self._abort_group(membership.group_id)
            self._membership = None
            self._half = None
            raise
        return membership

    def start_weight_update(self, version: str, operation_id: str) -> None:
        """Open the round for ``version``, bound to ``operation_id``.

        Same contract as the trainer: the operation is fetched and checked
        before the loader is prepared or any receive is enqueued.
        """
        if self._half is None:
            raise RuntimeError("compute_plan must run before start_weight_update")
        if self._loader is None:
            raise RuntimeError("initialize must run before start_weight_update")
        self._bind_operation(version, operation_id)
        try:
            self._loader.start_new_round(version)
            self._half.start_weight_update(version)
        except BaseException as error:
            self._fail_round(error)
            raise
        self._unfinished = deque()
        self._pipeline_depth = envs.MX_NCCL_REFIT_PIPELINE_DEPTH
        self._round_started = True
        self._version = version

    def _reset_round(self) -> None:
        self._unfinished = deque()
        super()._reset_round()

    def _install_oldest(self, half: NcclM2nReceiver) -> None:
        layer_group_id = self._unfinished.popleft()
        half.wait_weights(layer_group_id)
        self._loader.install(layer_group_id)

    def update_weights(
        self, version: str, layer_group_id: int = DEFAULT_LAYER_GROUP
    ) -> None:
        half = self._require_round(version, "update_weights")
        if self._loader is None:
            raise RuntimeError("initialize must run before update_weights")
        try:
            half.issue_weights(layer_group_id)
            self._unfinished.append(layer_group_id)
            while len(self._unfinished) >= self._pipeline_depth:
                self._install_oldest(half)
        except BaseException as error:
            self._fail_round(error)
            raise

    def finish_weight_update(
        self, version: str, *, operation_id: str | None = None
    ) -> None:
        """Install what is outstanding, then report against the bound operation.

        ``operation_id`` is optional and only checked: the operation is the one
        ``start_weight_update`` bound.
        """
        half = self._require_round(version, "finish_weight_update")
        if self._loader is None:
            raise RuntimeError("initialize must run before finish_weight_update")
        self._check_operation_alias(operation_id)
        try:
            while self._unfinished:
                self._install_oldest(half)
            half.finish_weight_update(self.membership.broadcast_lane.lane_id)
            self._loader.finish()
        except BaseException as error:
            self._fail_round(error)
            raise
        self._report_success()

    def cleanup(self) -> None:
        # The lanes and the M2N handle may still reference the engine's
        # buffers, so they are released first; the engine frees its buffers
        # after, even if that teardown raised, unless a communicator abort
        # wedged: then they are kept and the process must exit.
        loader, self._loader = self._loader, None
        try:
            super().cleanup()
        finally:
            if loader is not None:
                self._release_engine(loader)
