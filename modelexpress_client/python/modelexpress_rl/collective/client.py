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
from .plan import plan_digest, validate_coverage
from .rendezvous import CollectiveRendezvous, Membership
from .spi import Loader, Publisher, resolve_specs
from .types import ReshardPlan, Role

logger = logging.getLogger("modelexpress_rl.collective.client")


def _bootstrap_barrier(lane: LaneCommunicator, device: Any) -> None:
    """Full-group barrier used between overlapping communicator initializations."""
    import torch

    device_context = torch.cuda.device(device) if device is not None else nullcontext()
    with device_context:
        barrier = torch.zeros(
            1, dtype=torch.uint8, device=device if device is not None else "cuda"
        )
        stream = lane.stream
        stream_arg = (
            None if stream is None else int(getattr(stream, "cuda_stream", stream))
        )
        lane.handle.broadcast(
            sendbuf=barrier,
            recvbuf=barrier,
            root=0,
            stream=stream_arg,
        )
        lane.synchronize()


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
        self._device = device
        self._streams = list(streams) if streams else [None]

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
        self._digest = plan_digest(plan)

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

    def _join_and_bootstrap(
        self, role: Role, source_partition: int | None
    ) -> Membership:
        if self._digest is None:
            raise RuntimeError("initialize must run before compute_plan")
        require_nccl_m2n()

        membership = self._rendezvous.join(
            model_name=self._model_name,
            trainer_slots=self._trainer_slots,
            generator_slots=self._generator_slots,
            source_partition_count=self._source_partition_count,
            slot_id=self._slot_id,
            worker_id=self._worker_id,
            role=role,
            index_in_role=self._index_in_role,
            plan_digest=self._digest,
            source_partition=source_partition,
        )
        expected_reshard = (
            {source_partition}
            if role is Role.TRAINER and source_partition is not None
            else set(range(self._source_partition_count))
        )
        actual_reshard = {lane.lane_id for lane in membership.reshard_lanes}
        if actual_reshard != expected_reshard:
            raise RuntimeError(
                "MX returned unexpected reshard-lane membership: "
                f"expected {sorted(expected_reshard)}, got {sorted(actual_reshard)}"
            )
        if membership.broadcast_lane.lane_id != self._source_partition_count:
            raise RuntimeError(
                "MX returned an unexpected broadcast lane id: "
                f"expected {self._source_partition_count}, got "
                f"{membership.broadcast_lane.lane_id}"
            )

        previous = self._membership
        if previous is not None and previous.group_id != membership.group_id:
            self._cache.abort_group(previous.group_id)
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

        by_lane_id = {lane.lane_id: lane for lane in group.lanes}
        all_lane_ids = set(range(self._source_partition_count)) | {
            membership.broadcast_lane.lane_id
        }
        missing = sorted(all_lane_ids - set(by_lane_id))
        if missing:
            raise RuntimeError(
                f"READY group omitted lane(s) assigned to this worker: {missing}"
            )

        # All ranks create the broadcast communicator first. It then provides a
        # full-group barrier after every reshard-lane init. Without those
        # barriers, a PP-stage trainer can enter lane N+1 while generators are
        # still initializing lane N; overlapping communicator creation can hang.
        lane_order = [membership.broadcast_lane.lane_id] + list(
            range(self._source_partition_count)
        )
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
            self._cache.abort_group(membership.group_id)
            self._membership = None
            raise

        self._membership = membership
        return membership

    def cleanup(self) -> None:
        if self._membership is not None:
            self._cache.abort_group(self._membership.group_id)
        self._membership = None
        self._half = None
        self._round_started = False


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

    def publish_weights(
        self, version: str, layer_group_id: int = DEFAULT_LAYER_GROUP
    ) -> None:
        if self._half is None:
            raise RuntimeError("compute_plan must run before publish_weights")
        if not self._round_started:
            raise RuntimeError("start_weight_update must run before publish_weights")
        self._half.publish_weights(layer_group_id)

    def finish_weight_update(
        self, version: str, operation_id: str | None = None
    ) -> None:
        if self._half is None:
            raise RuntimeError("compute_plan must run before finish_weight_update")
        if not self._round_started:
            raise RuntimeError(
                "start_weight_update must run before finish_weight_update"
            )
        try:
            self._half.finish_weight_update(self.membership.broadcast_lane.lane_id)
        except Exception as error:
            self._report(operation_id, succeeded=False, message=repr(error))
            self._half.abort()
            raise
        finally:
            self._round_started = False
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

    def update_weights(
        self, version: str, layer_group_id: int = DEFAULT_LAYER_GROUP
    ) -> None:
        if self._half is None:
            raise RuntimeError("compute_plan must run before update_weights")
        if not self._round_started:
            raise RuntimeError("start_weight_update must run before update_weights")
        if self._loader is None:
            raise RuntimeError("initialize must run before update_weights")
        self._half.update_weights(layer_group_id)
        self._loader.install(layer_group_id)

    def finish_weight_update(
        self, version: str, operation_id: str | None = None
    ) -> None:
        if self._half is None:
            raise RuntimeError("compute_plan must run before finish_weight_update")
        if not self._round_started:
            raise RuntimeError(
                "start_weight_update must run before finish_weight_update"
            )
        if self._loader is None:
            raise RuntimeError("initialize must run before finish_weight_update")
        try:
            self._half.finish_weight_update(self.membership.broadcast_lane.lane_id)
            self._loader.finish()
        except Exception as error:
            self._report(operation_id, succeeded=False, message=repr(error))
            self._half.abort()
            raise
        finally:
            self._round_started = False
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
