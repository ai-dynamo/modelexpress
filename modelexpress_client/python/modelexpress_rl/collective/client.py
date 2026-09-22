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
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

from . import envs
from .backend import (
    DEFAULT_LAYER_GROUP,
    NcclM2nReceiver,
    NcclM2nSender,
    require_nccl_m2n,
)
from .comm import CommunicatorCache, LaneCommunicator, LaneKey, new_unique_id
from .plan import DEFAULT_RECEIVER_PROTOCOL, plan_digest, validate_coverage
from .rendezvous import (
    CollectiveRendezvous,
    EpochChangedError,
    GroupNotReadyError,
    LaneDeclaration,
    Membership,
)
from .spi import Loader, Publisher, resolve_specs
from .types import ReshardPlan, Role

logger = logging.getLogger("modelexpress_rl.collective.client")

# Epoch invalidation is cleanup, not another formation attempt. It retains a
# short independent bound so expiry of the shared formation deadline cannot
# skip the server-side reset and leave released fence arrivals reusable.
_BOOTSTRAP_ABORT_TIMEOUT_S = 1.0


def build_partitioned_lanes(
    trainer_slots: list[str],
    generator_slots: list[str],
    source_partition_count: int,
) -> list[LaneDeclaration]:
    """Build contiguous reshard lanes plus the all-participant broadcast lane."""
    if (
        isinstance(source_partition_count, bool)
        or not isinstance(source_partition_count, int)
        or source_partition_count <= 0
    ):
        raise ValueError("source_partition_count must be a positive integer")
    if len(trainer_slots) % source_partition_count:
        raise ValueError(
            "source_partition_count must divide the trainer slot count: "
            f"{len(trainer_slots)} trainer slots, "
            f"{source_partition_count} source partitions"
        )
    trainers = tuple(trainer_slots)
    generators = tuple(generator_slots)
    per_lane = len(trainers) // source_partition_count
    lanes = [
        LaneDeclaration(
            partition,
            "RESHARD",
            trainers[partition * per_lane : (partition + 1) * per_lane],
            generators,
        )
        for partition in range(source_partition_count)
    ]
    lanes.append(
        LaneDeclaration(
            source_partition_count,
            "BROADCAST",
            trainers,
            generators,
        )
    )
    return lanes


def _broadcast_barrier(
    lane: LaneCommunicator, barrier: Any, timeout_s: float | None
) -> None:
    """Broadcast one device byte over the lane and wait for it, bounded."""
    stream = lane.stream
    stream_arg = None if stream is None else int(getattr(stream, "cuda_stream", stream))
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


def _bootstrap_barrier(
    lane: LaneCommunicator,
    device: Any,
    *,
    timeout_s: float | None = None,
    alloc: Callable[[Any], Any] | None = None,
) -> None:
    """Full-group barrier used between overlapping communicator initializations.

    Bounded like every other wait past READY. This barrier runs after each
    lane comes up, so a peer that dies between READY and here would otherwise
    block this rank forever: ``MX_NCCL_REFIT_COMM_INIT_TIMEOUT_S`` bounds
    ``Communicator.init`` only, and the transfer deadline does not arm until
    bootstrap is done. The caller's ``except BaseException`` aborts the group,
    so a timeout raised here tears it down and forces a fresh epoch.

    ``alloc`` supplies the byte for a worker whose framework is not torch. The
    buffer is opaque to this path -- it is written and never read, so only the
    device it sits on matters -- and nccl4py resolves a buffer through the CUDA
    Array Interface or DLPack, which a ``jax.Array`` satisfies. Leaving it None
    keeps the torch allocation this path has always used.
    """
    if alloc is not None:
        _broadcast_barrier(lane, alloc(device), timeout_s)
        return

    import torch

    device_context = torch.cuda.device(device) if device is not None else nullcontext()
    with device_context:
        barrier = torch.zeros(
            1, dtype=torch.uint8, device=device if device is not None else "cuda"
        )
        _broadcast_barrier(lane, barrier, timeout_s)


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
        barrier_alloc: Callable[[Any], Any] | None = None,
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
        self._m2n_abi_version = m2n_abi_version
        self._device = device
        self._streams = list(streams) if streams else [None]
        self._barrier_alloc = barrier_alloc

        self._cache = CommunicatorCache()
        self._publisher: Publisher | None = None
        self._loader: Loader | None = None
        self._expected_parameters: list[str] | None = None
        self._plan: ReshardPlan | None = None
        self._digest: str | None = None
        self._membership: Membership | None = None
        self._half: NcclM2nSender | NcclM2nReceiver | None = None
        self._groupings: list[list[str]] | None = None
        self._round_started = False
        self._version: str | None = None
        self._bootstrap_poisoned: BaseException | None = None

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

        ``version`` was accepted and ignored, so a call naming a version other
        than the one ``start_weight_update`` opened moved that round's tensors
        under the label of a different one -- silently, and on every rank.
        """
        if self._half is None:
            raise RuntimeError(f"compute_plan must run before {method}")
        if not self._round_started:
            raise RuntimeError(f"start_weight_update must run before {method}")
        if version != self._version:
            raise ValueError(
                f"{method} names version {version!r}, but the round in flight is "
                f"{self._version!r}"
            )
        return self._half

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
        self._plan = plan
        self._digest = plan_digest(
            plan,
            receiver_protocol=self._receiver_protocol,
            m2n_abi_version=self._m2n_abi_version,
        )

    def setup_layer_groups(self, groupings: list[list[str]] | None) -> None:
        """Optional. Without it every bulk parameter is in layer group 0."""
        self._groupings = groupings
        if self._half is not None:
            self._half.setup_layer_groups(groupings)

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
        return build_partitioned_lanes(
            self._trainer_slots,
            self._generator_slots,
            self._source_partition_count,
        )

    def _join_and_bootstrap(
        self, role: Role, source_partition: int | None
    ) -> Membership:
        if self._bootstrap_poisoned is not None:
            raise RuntimeError(
                "this refit client cannot bootstrap again because the previous "
                "failed epoch could not be invalidated; construct a new client"
            ) from self._bootstrap_poisoned
        if self._digest is None:
            raise RuntimeError("initialize must run before compute_plan")
        require_nccl_m2n()

        declared = self._declared_lanes()
        expected_reshard = {
            lane.lane_id
            for lane in declared
            if lane.kind == "RESHARD" and self._slot_id in lane.slots_in_rank_order()
        }
        declared_broadcast = next(
            lane.lane_id for lane in declared if lane.kind == "BROADCAST"
        )
        timeout_s = envs.MX_NCCL_REFIT_GROUP_TIMEOUT_S
        deadline = time.monotonic() + timeout_s
        # One replacement per expected slot can advance the epoch while a full
        # cohort restarts. The initial attempt plus that many bounded retries
        # lets all fresh workers converge without masking unbounded churn.
        max_attempts = len(self._trainer_slots) + len(self._generator_slots) + 1

        def remaining_budget(group_id: str) -> float:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                raise GroupNotReadyError(group_id, [], timeout_s)
            return remaining_s

        for attempt in range(max_attempts):
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
                timeout_s=remaining_budget(self._model_name),
            )

            # Which reshard lanes this worker belongs on is a fact about what it
            # just declared, not a re-derivation of a rule the server also applies.
            actual_reshard = {lane.lane_id for lane in membership.reshard_lanes}
            if actual_reshard != expected_reshard:
                raise RuntimeError(
                    "MX returned unexpected reshard-lane membership: "
                    f"expected {sorted(expected_reshard)}, got "
                    f"{sorted(actual_reshard)}"
                )
            if membership.broadcast_lane.lane_id != declared_broadcast:
                raise RuntimeError(
                    "MX returned an unexpected broadcast lane id: "
                    f"expected {declared_broadcast}, got "
                    f"{membership.broadcast_lane.lane_id}"
                )

            previous = self._membership
            if previous is not None and previous.group_id != membership.group_id:
                self._cache.abort_group(previous.group_id)
            # An epoch move invalidates every cached communicator. The plan was
            # freshly captured before this join and is guarded by its digest.
            dropped = self._cache.invalidate_epoch(
                membership.group_id, membership.epoch
            )
            if dropped:
                logger.info(
                    "epoch moved to %s; dropped %s stale lane(s)",
                    membership.epoch,
                    dropped,
                )

            try:
                if membership.is_bootstrap_leader:
                    for lane in membership.reshard_lanes:
                        if lane.rank_in_lane == 0:
                            self._rendezvous.publish_bootstrap(
                                group_id=membership.group_id,
                                epoch=membership.epoch,
                                lane_id=lane.lane_id,
                                worker_id=self._worker_id,
                                nccl_unique_id=new_unique_id(),
                                timeout_s=remaining_budget(membership.group_id),
                            )
                    broadcast = membership.broadcast_lane
                    if broadcast.rank_in_lane == 0:
                        self._rendezvous.publish_bootstrap(
                            group_id=membership.group_id,
                            epoch=membership.epoch,
                            lane_id=broadcast.lane_id,
                            worker_id=self._worker_id,
                            nccl_unique_id=new_unique_id(),
                            timeout_s=remaining_budget(membership.group_id),
                        )

                group = self._rendezvous.await_ready(
                    group_id=membership.group_id,
                    epoch=membership.epoch,
                    timeout_s=remaining_budget(membership.group_id),
                )
            except EpochChangedError:
                self._cache.abort_group(membership.group_id)
                self._membership = None
                if attempt + 1 >= max_attempts or time.monotonic() >= deadline:
                    raise
                logger.info(
                    "collective group %s moved past epoch %s during formation; "
                    "rejoining with worker %s (%s/%s)",
                    membership.group_id,
                    membership.epoch,
                    self._worker_id,
                    attempt + 2,
                    max_attempts,
                )
                continue
            # Every rank walks the full declared lane order. After its optional
            # communicator creation it settles every local communicator, reaches
            # a server-backed full-cohort fence, and only then enters the NCCL
            # broadcast barrier. The control-plane fence closes the race where
            # a fast nonmember could reuse broadcast while a lane member was
            # still initializing. The broadcast step itself follows the same
            # protocol so no rank can start the first reshard step early.
            try:
                by_lane_id = {lane.lane_id: lane for lane in group.lanes}
                all_lane_ids = {lane.lane_id for lane in declared}
                missing = sorted(all_lane_ids - set(by_lane_id))
                if missing:
                    raise RuntimeError(
                        f"READY group omitted lane(s) assigned to this worker: {missing}"
                    )
                lane_order = [membership.broadcast_lane.lane_id] + [
                    lane.lane_id for lane in declared if lane.kind == "RESHARD"
                ]
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
                            timeout_s=remaining_budget(membership.group_id),
                        )

                    self._cache.settle_group(
                        membership.group_id,
                        membership.epoch,
                        timeout_s=remaining_budget(membership.group_id),
                    )
                    self._rendezvous.await_bootstrap_fence(
                        group_id=membership.group_id,
                        epoch=membership.epoch,
                        lane_id=lane_id,
                        slot_id=self._slot_id,
                        worker_id=self._worker_id,
                        timeout_s=remaining_budget(membership.group_id),
                        phase="PRE_BARRIER",
                    )

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
                    _bootstrap_barrier(
                        broadcast,
                        self._device,
                        timeout_s=remaining_budget(membership.group_id),
                        alloc=self._barrier_alloc,
                    )
                self._rendezvous.await_bootstrap_fence(
                    group_id=membership.group_id,
                    epoch=membership.epoch,
                    lane_id=lane_order[-1],
                    slot_id=self._slot_id,
                    worker_id=self._worker_id,
                    timeout_s=remaining_budget(membership.group_id),
                    phase="COMPLETE",
                )
            except BaseException as bootstrap_error:
                self._cache.abort_group(membership.group_id)
                self._membership = None
                try:
                    self._rendezvous.abort_bootstrap(
                        group_id=membership.group_id,
                        epoch=membership.epoch,
                        slot_id=self._slot_id,
                        worker_id=self._worker_id,
                        message=f"{type(bootstrap_error).__name__}: {bootstrap_error}",
                        timeout_s=_BOOTSTRAP_ABORT_TIMEOUT_S,
                    )
                except EpochChangedError:
                    # Another rank fenced this same failed epoch first.
                    pass
                except BaseException as abort_error:
                    self._bootstrap_poisoned = abort_error
                    raise bootstrap_error from abort_error
                if not isinstance(bootstrap_error, Exception):
                    raise
                if attempt + 1 >= max_attempts or time.monotonic() >= deadline:
                    raise
                logger.info(
                    "collective group %s epoch %s failed during bootstrap; "
                    "rejoining with worker %s (%s/%s)",
                    membership.group_id,
                    membership.epoch,
                    self._worker_id,
                    attempt + 2,
                    max_attempts,
                )
                continue

            self._membership = membership
            return membership
        raise AssertionError("collective formation loop exhausted")  # pragma: no cover

    def cleanup(self) -> None:
        if self._membership is not None:
            self._cache.abort_group(self._membership.group_id)
        self._membership = None
        self._half = None
        self._round_started = False
        self._version = None


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
        membership = self._join_and_bootstrap(Role.TRAINER, self._source_partition)
        try:
            self._half = NcclM2nSender(
                plan=self.plan,
                specs=specs,
                group_id=membership.group_id,
                epoch=membership.epoch,
                cache=self._cache,
                source_partition=self._source_partition,
            )
            self._half.setup_layer_groups(self._groupings)
        except BaseException:
            self._cache.abort_group(membership.group_id)
            self._membership = None
            self._half = None
            raise
        return membership

    def start_weight_update(self, version: str) -> None:
        if self._half is None:
            raise RuntimeError("compute_plan must run before start_weight_update")
        if self._publisher is None:
            raise RuntimeError("initialize must run before start_weight_update")
        self._publisher.start_new_round(version)
        self._half.start_weight_update(version)
        self._round_started = True
        self._version = version

    def publish_weights(
        self, version: str, layer_group_id: int = DEFAULT_LAYER_GROUP
    ) -> None:
        half = self._require_round(version, "publish_weights")
        half.publish_weights(layer_group_id)

    def finish_weight_update(
        self, version: str, operation_id: str | None = None
    ) -> None:
        half = self._require_round(version, "finish_weight_update")
        try:
            half.finish_weight_update(self.membership.broadcast_lane.lane_id)
        except Exception as error:
            half.abort()
            try:
                self._report(operation_id, succeeded=False, message=repr(error))
            except Exception:
                logger.warning(
                    "reporting the failed round to the control plane raised; "
                    "the local half is already aborted so peers are released, "
                    "and the original failure is re-raised below",
                    exc_info=True,
                )
            raise
        finally:
            self._round_started = False
            self._version = None
        self._report(operation_id, succeeded=True)

    def _report(
        self, operation_id: str | None, *, succeeded: bool, message: str = ""
    ) -> None:
        if operation_id is None:
            return
        self._rendezvous.report(
            operation_id=operation_id,
            group_id=self.membership.group_id,
            epoch=self.membership.epoch,
            worker_id=self._worker_id,
            succeeded=succeeded,
            message=message,
        )

    def cleanup(self) -> None:
        if self._publisher is not None:
            self._publisher.cleanup()
        self._publisher = None
        super().cleanup()


class RefitClientGenerator(_RefitClientBase):
    """Generator-side lifecycle."""

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
        membership = self._join_and_bootstrap(Role.GENERATOR, None)
        try:
            self._half = NcclM2nReceiver(
                plan=self.plan,
                specs=specs,
                group_id=membership.group_id,
                epoch=membership.epoch,
                cache=self._cache,
            )
            self._half.setup_layer_groups(self._groupings)
        except BaseException:
            self._cache.abort_group(membership.group_id)
            self._membership = None
            self._half = None
            raise
        return membership

    def start_weight_update(self, version: str) -> None:
        if self._half is None:
            raise RuntimeError("compute_plan must run before start_weight_update")
        if self._loader is None:
            raise RuntimeError("initialize must run before start_weight_update")
        self._loader.start_new_round(version)
        self._half.start_weight_update(version)
        self._round_started = True
        self._version = version

    def update_weights(
        self, version: str, layer_group_id: int = DEFAULT_LAYER_GROUP
    ) -> None:
        half = self._require_round(version, "update_weights")
        if self._loader is None:
            raise RuntimeError("initialize must run before update_weights")
        half.update_weights(layer_group_id)
        self._loader.install(layer_group_id)

    def finish_weight_update(
        self, version: str, operation_id: str | None = None
    ) -> None:
        half = self._require_round(version, "finish_weight_update")
        if self._loader is None:
            raise RuntimeError("initialize must run before finish_weight_update")
        try:
            half.finish_weight_update(self.membership.broadcast_lane.lane_id)
            self._loader.finish()
        except Exception as error:
            half.abort()
            try:
                self._report(operation_id, succeeded=False, message=repr(error))
            except Exception:
                logger.warning(
                    "reporting the failed round to the control plane raised; "
                    "the local half is already aborted so peers are released, "
                    "and the original failure is re-raised below",
                    exc_info=True,
                )
            raise
        finally:
            self._round_started = False
            self._version = None
        self._report(operation_id, succeeded=True)

    def _report(
        self, operation_id: str | None, *, succeeded: bool, message: str = ""
    ) -> None:
        if operation_id is None:
            return
        self._rendezvous.report(
            operation_id=operation_id,
            group_id=self.membership.group_id,
            epoch=self.membership.epoch,
            worker_id=self._worker_id,
            succeeded=succeeded,
            message=message,
        )

    def cleanup(self) -> None:
        if self._loader is not None:
            self._loader.cleanup()
        self._loader = None
        super().cleanup()


def num_streams() -> int:
    return envs.MX_NCCL_REFIT_NUM_STREAMS
