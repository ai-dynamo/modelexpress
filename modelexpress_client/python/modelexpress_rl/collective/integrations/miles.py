# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MILES-side bindings for the NCCL M2N collective SPI."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..client import RefitClientTrainer
from ..plan import DEFAULT_RECEIVER_PROTOCOL
from ..rendezvous import CollectiveRendezvous
from ..spi import LocalParamSpec
from .miles_topology import MilesReshardTopologyPlan
from ._common import (
    _FrozenPlan,
    _check_stable,
    _client_device,
    _collective_streams,
    _layer_groups,
    _local_shape,
    _order_current_cuda_stream_before,
    _single_device,
    _tensor_signature,
    _text,
)

logger = logging.getLogger("modelexpress_rl.collective.integrations.miles")


def _validate_miles_route_alignment(
    miles_plan: MilesReshardTopologyPlan,
    trainer_lanes: tuple[tuple[int, ...], ...],
) -> None:
    bulk_name_counts = {}
    for entry in miles_plan.plan.bulk:
        bulk_name_counts[entry.name] = bulk_name_counts.get(entry.name, 0) + 1
    duplicate_bulk_names = sorted(
        name for name, count in bulk_name_counts.items() if count != 1
    )
    if duplicate_bulk_names:
        raise ValueError(
            f"MILES plan bulk parameter names must be unique: {duplicate_bulk_names}"
        )

    routes_by_name = {}
    for route in miles_plan.routes:
        routes_by_name.setdefault(route.canonical_name, []).append(route)

    plan_names = set(bulk_name_counts)
    unexpected_routes = sorted(set(routes_by_name) - plan_names)
    if unexpected_routes:
        raise ValueError(
            f"MILES source routes have no matching plan bulk entry: {unexpected_routes}"
        )

    for entry in miles_plan.plan.bulk:
        matching_routes = routes_by_name.get(entry.name, ())
        if len(matching_routes) != 1:
            raise ValueError(
                f"{entry.name}: expected exactly one source route, "
                f"got {len(matching_routes)}"
            )
        route = matching_routes[0]
        if route.partition_id != entry.partition_id:
            raise ValueError(
                f"{entry.name}: route partition {route.partition_id} does not "
                f"match plan partition {entry.partition_id}"
            )
        if not 0 <= entry.partition_id < len(trainer_lanes):
            raise ValueError(
                f"{entry.name}: plan partition {entry.partition_id} has no trainer lane"
            )

        lane = trainer_lanes[entry.partition_id]
        source_mesh_ranks = tuple(entry.src_mesh.ranks())
        if any(rank >= len(lane) for rank in source_mesh_ranks):
            raise ValueError(
                f"{entry.name}: source mesh ranks {source_mesh_ranks} exceed "
                f"trainer lane width {len(lane)}"
            )
        expected_owners = tuple(lane[rank] for rank in source_mesh_ranks)
        route_owners = tuple(route.source_world_ranks)
        if any(
            isinstance(world_rank, bool) or not isinstance(world_rank, int)
            for world_rank in route_owners
        ):
            raise ValueError(f"{entry.name}: route source world ranks must be integers")
        if route_owners != expected_owners:
            raise ValueError(
                f"{entry.name}: source owners do not match source mesh ranks; "
                f"expected {expected_owners}, got {route.source_world_ranks}"
            )

        named_owners = tuple(
            world_rank for world_rank, _ in route.source_names_by_world
        )
        if any(
            isinstance(world_rank, bool) or not isinstance(world_rank, int)
            for world_rank in named_owners
        ):
            raise ValueError(
                f"{entry.name}: source_names_by_world owners must be integers"
            )
        if named_owners != expected_owners:
            raise ValueError(
                f"{entry.name}: source_names_by_world owners do not match "
                f"source mesh ranks; expected {expected_owners}, got {named_owners}"
            )
        if any(
            not source_names or any(not source_name for source_name in source_names)
            for _, source_names in route.source_names_by_world
        ):
            raise ValueError(
                f"{entry.name}: every source owner must provide source names"
            )


@dataclass(frozen=True)
class CollectiveTopology:
    """The immutable participant and ABI identity shared by every rank."""

    model_name: str
    trainer_slots: tuple[str, ...]
    generator_slots: tuple[str, ...]
    source_partition_count: int
    m2n_abi_version: str
    receiver_protocol: str = DEFAULT_RECEIVER_PROTOCOL

    @classmethod
    def from_miles_reshard_plan(
        cls,
        *,
        model_name: str,
        miles_plan: MilesReshardTopologyPlan,
        trainer_slots_by_world_rank: Mapping[int, str],
        generator_slots: Sequence[str],
        m2n_abi_version: str,
        receiver_protocol: str = DEFAULT_RECEIVER_PROTOCOL,
    ) -> CollectiveTopology:
        """Bind MILES world ranks to PR 795 lane-local collective ranks."""
        trainer_lanes = tuple(
            tuple(world_rank for world_rank in lane)
            for lane in miles_plan.trainer_lanes
        )
        source_partition_count = miles_plan.plan.source_partition_count
        if len(trainer_lanes) != source_partition_count:
            raise ValueError(
                "MILES trainer lanes must match the plan source partitions: "
                f"{len(trainer_lanes)} != {source_partition_count}"
            )
        if not trainer_lanes or any(not lane for lane in trainer_lanes):
            raise ValueError("MILES trainer lanes must not be empty")
        lane_sizes = {len(lane) for lane in trainer_lanes}
        if len(lane_sizes) != 1:
            raise ValueError(
                "MILES trainer lanes must contain the same number of trainers"
            )

        ordered_world_ranks = tuple(
            world_rank for lane in trainer_lanes for world_rank in lane
        )
        if any(
            isinstance(world_rank, bool) or not isinstance(world_rank, int)
            for world_rank in ordered_world_ranks
        ):
            raise ValueError("MILES trainer lane world ranks must be integers")
        if len(ordered_world_ranks) != len(set(ordered_world_ranks)):
            raise ValueError("MILES trainer lanes contain duplicate world ranks")
        _validate_miles_route_alignment(miles_plan, trainer_lanes)

        slots_by_world_rank = dict(trainer_slots_by_world_rank)
        if any(
            isinstance(world_rank, bool) or not isinstance(world_rank, int)
            for world_rank in slots_by_world_rank
        ):
            raise ValueError("trainer slot projection world ranks must be integers")
        expected_world_ranks = set(ordered_world_ranks)
        actual_world_ranks = set(slots_by_world_rank)
        missing = sorted(expected_world_ranks - actual_world_ranks)
        if missing:
            raise ValueError(
                f"trainer slot projection is missing world ranks {missing}"
            )
        unexpected = sorted(
            actual_world_ranks - expected_world_ranks,
            key=repr,
        )
        if unexpected:
            raise ValueError(
                f"trainer slot projection contains unexpected world ranks {unexpected}"
            )

        trainer_slots = tuple(
            str(slots_by_world_rank[world_rank]) for world_rank in ordered_world_ranks
        )
        if len(trainer_slots) != len(set(trainer_slots)):
            raise ValueError("trainer slot projection contains duplicate trainer slots")
        return cls(
            model_name=model_name,
            trainer_slots=trainer_slots,
            generator_slots=tuple(generator_slots),
            source_partition_count=source_partition_count,
            m2n_abi_version=m2n_abi_version,
            receiver_protocol=receiver_protocol,
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "trainer_slots",
            tuple(str(slot) for slot in self.trainer_slots),
        )
        object.__setattr__(
            self,
            "generator_slots",
            tuple(str(slot) for slot in self.generator_slots),
        )
        object.__setattr__(self, "model_name", _text(self.model_name, "model_name"))
        object.__setattr__(
            self,
            "m2n_abi_version",
            _text(self.m2n_abi_version, "m2n_abi_version"),
        )
        object.__setattr__(
            self,
            "receiver_protocol",
            _text(self.receiver_protocol, "receiver_protocol"),
        )
        if self.source_partition_count <= 0:
            raise ValueError("source_partition_count must be positive")
        if not self.trainer_slots or not self.generator_slots:
            raise ValueError("trainer_slots and generator_slots must not be empty")
        all_slots = self.trainer_slots + self.generator_slots
        if any(not str(slot).strip() for slot in all_slots):
            raise ValueError("collective slot ids must not be empty")
        if len(all_slots) != len(set(all_slots)):
            raise ValueError("collective slot ids must be unique across both roles")
        if len(self.trainer_slots) % self.source_partition_count:
            raise ValueError(
                "source_partition_count must divide the trainer slot count"
            )


class MilesPublisher:
    """Bind explicit MILES aliases to stable partition-local trainer tensors."""

    def __init__(
        self,
        *,
        plan,
        source_partition: int,
        tensors: dict[str, Any],
        aliases: dict[str, str],
    ) -> None:
        self._plan = _FrozenPlan(plan)
        if not 0 <= source_partition < self._plan.source_partition_count:
            raise ValueError(
                f"source_partition must be in [0, "
                f"{self._plan.source_partition_count}), got {source_partition}"
            )
        self._source_partition = source_partition
        self._tensors = dict(tensors)
        self._aliases = dict(aliases)
        if set(self._tensors) != set(self._aliases):
            raise ValueError("every MILES tensor must have exactly one explicit alias")
        if len(set(self._aliases.values())) != len(self._aliases):
            raise ValueError("MILES canonical aliases must not contain duplicates")

        required = [
            name
            for name in self._plan.names()
            if self._plan.entry(name).partition_id == source_partition
        ]
        if set(self._aliases.values()) != set(required):
            raise ValueError(
                f"MILES aliases must exactly cover partition {source_partition} "
                f"; expected {required}, got {list(self._aliases.values())}"
            )
        native_by_canonical = {
            canonical_name: native_name
            for native_name, canonical_name in self._aliases.items()
        }
        self._ordered_aliases = tuple(
            (native_by_canonical[canonical_name], canonical_name)
            for canonical_name in required
        )

        self._signatures = {}
        for native_name, canonical_name in self._ordered_aliases:
            entry = self._plan.entry(canonical_name)
            self._signatures[native_name] = _tensor_signature(
                canonical_name,
                self._tensors[native_name],
                expected_shape=_local_shape(
                    entry.global_shape,
                    entry.src_mesh,
                    entry.src_placements,
                ),
                expected_dtype=entry.dtype,
            )
        self._device = _single_device(self._signatures, "MILES publisher")

    @property
    def source_partition_count(self) -> int:
        return self._plan.source_partition_count

    @property
    def device(self) -> str:
        return self._device

    def validate_topology(self, topology: CollectiveTopology) -> None:
        self._plan.validate_topology(topology)

    def _validate_stable(self) -> None:
        for native_name, canonical_name in self._ordered_aliases:
            entry = self._plan.entry(canonical_name)
            _check_stable(
                canonical_name,
                self._tensors[native_name],
                self._signatures[native_name],
                expected_shape=_local_shape(
                    entry.global_shape,
                    entry.src_mesh,
                    entry.src_placements,
                ),
                expected_dtype=entry.dtype,
            )

    def capture(self):
        self._validate_stable()
        return self._plan.capture()

    def parameter_names(self) -> list[str]:
        return self._plan.names()

    def local_params(self) -> dict[str, LocalParamSpec]:
        self._validate_stable()
        return {
            canonical_name: LocalParamSpec(base=self._tensors[native_name])
            for native_name, canonical_name in self._ordered_aliases
        }

    def start_new_round(self, version: str) -> None:
        _text(version, "version")
        self._validate_stable()

    def cleanup(self) -> None:
        return None


class MilesTrainerSession:
    """Own one trainer rank's reusable collective membership and lifecycle."""

    def __init__(
        self,
        *,
        client: RefitClientTrainer,
        rendezvous: CollectiveRendezvous,
        publisher: MilesPublisher,
        source_partition: int,
        layer_groups: tuple[tuple[str, ...], ...] = (),
        device: Any = None,
        streams: list[Any] | None = None,
    ) -> None:
        self._client = client
        self._rendezvous = rendezvous
        self._publisher = publisher
        self._source_partition = source_partition
        self._groups = _layer_groups(layer_groups, publisher.parameter_names())
        self._device = _client_device(device, publisher.device, "MILES trainer")
        self._streams = list(streams) if streams else [None]
        self._membership = None
        self._prepared = False
        self._closed = False
        self._round_version: str | None = None

    @classmethod
    def create(
        cls,
        *,
        rendezvous: CollectiveRendezvous,
        topology: CollectiveTopology,
        publisher: MilesPublisher,
        source_partition: int,
        slot_id: str,
        worker_id: str,
        index_in_role: int,
        layer_groups: tuple[tuple[str, ...], ...] = (),
        device: Any = None,
        streams: list[Any] | None = None,
    ) -> MilesTrainerSession:
        publisher.validate_topology(topology)
        device = _client_device(device, publisher.device, "MILES trainer")
        streams = _collective_streams(streams, device=device)
        client = RefitClientTrainer(
            rendezvous=rendezvous,
            model_name=topology.model_name,
            trainer_slots=list(topology.trainer_slots),
            generator_slots=list(topology.generator_slots),
            source_partition_count=topology.source_partition_count,
            slot_id=_text(slot_id, "slot_id"),
            worker_id=_text(worker_id, "worker_id"),
            index_in_role=index_in_role,
            receiver_protocol=topology.receiver_protocol,
            m2n_abi_version=topology.m2n_abi_version,
            device=device,
            streams=streams,
        )
        return cls(
            client=client,
            rendezvous=rendezvous,
            publisher=publisher,
            source_partition=source_partition,
            layer_groups=layer_groups,
            device=device,
            streams=streams,
        )

    def prepare(self):
        if self._closed:
            raise RuntimeError("the MILES trainer session is closed")
        if self._prepared:
            return self._membership
        try:
            self._client.initialize(
                self._publisher,
                source_partition=self._source_partition,
            )
            self._client.setup_layer_groups(self._groups)
            self._membership = self._client.compute_plan()
            self._prepared = True
            return self._membership
        except BaseException:
            self.close()
            raise

    @property
    def membership(self):
        if self._membership is None:
            raise RuntimeError("prepare must complete before reading membership")
        return self._membership

    @property
    def group_count(self) -> int:
        return len(self._groups)

    def _require_open_round(self, version: str, method: str) -> None:
        if not self._prepared or self._membership is None:
            raise RuntimeError(f"prepare must complete before {method}")
        if self._closed:
            raise RuntimeError("the MILES trainer session is closed")
        if self._round_version is None:
            raise RuntimeError(f"begin_round must run before {method}")
        if version != self._round_version:
            raise ValueError(
                f"{method} names version {version!r}, but the round in flight is "
                f"{self._round_version!r}"
            )

    def begin_round(self, *, version: str) -> None:
        if not self._prepared or self._membership is None:
            raise RuntimeError("prepare must complete before a trainer round")
        if self._closed:
            raise RuntimeError("the MILES trainer session is closed")
        if self._round_version is not None:
            raise RuntimeError(
                f"a round for version {self._round_version!r} is already in flight"
            )
        version = _text(version, "version")
        try:
            _order_current_cuda_stream_before(self._streams, device=self._device)
            self._client.start_weight_update(version)
        except BaseException as error:
            self._fail_round()
            raise
        self._round_version = version

    def publish_group(self, *, version: str, layer_group_id: int) -> None:
        self._require_open_round(version, "publish_group")
        if not 0 <= layer_group_id < len(self._groups):
            raise ValueError(
                f"layer_group_id {layer_group_id} is outside the "
                f"{len(self._groups)} declared publish groups"
            )
        try:
            self._client.publish_weights(version, layer_group_id)
        except BaseException as error:
            self._fail_round()
            raise

    def finish_round(self, *, version: str) -> None:
        self._require_open_round(version, "finish_round")
        try:
            self._client.finish_weight_update(version)
        except BaseException as error:
            self._fail_round()
            raise
        finally:
            self._round_version = None

    def _fail_round(self) -> None:
        # The round failure itself propagates; this is the teardown path.
        try:
            self._client.cleanup()
        except BaseException:
            logger.warning(
                "trainer cleanup failed after the round error", exc_info=True
            )
        try:
            self._rendezvous.close()
        except BaseException:
            logger.warning(
                "closing trainer rendezvous after the round error failed",
                exc_info=True,
            )
        self._closed = True
        self._round_version = None

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._client.cleanup()
        finally:
            self._rendezvous.close()
            self._closed = True


__all__ = [
    "CollectiveTopology",
    "MilesPublisher",
    "MilesTrainerSession",
]
