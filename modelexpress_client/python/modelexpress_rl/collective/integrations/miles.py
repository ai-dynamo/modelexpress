# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MILES-side bindings for the NCCL M2N collective SPI."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..client import RefitClientTrainer
from ..plan import DEFAULT_RECEIVER_PROTOCOL
from ..rendezvous import CollectiveRendezvous, LaneDeclaration
from ..spi import LocalParamSpec
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


@dataclass(frozen=True)
class CollectiveTopology:
    """The immutable participant and ABI identity shared by every rank."""

    model_name: str
    trainer_slots: tuple[str, ...]
    generator_slots: tuple[str, ...]
    source_partition_count: int
    m2n_abi_version: str
    receiver_protocol: str = DEFAULT_RECEIVER_PROTOCOL

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

    def lanes(self) -> list[LaneDeclaration]:
        from ..client import build_partitioned_lanes

        return build_partitioned_lanes(
            list(self.trainer_slots),
            list(self.generator_slots),
            self.source_partition_count,
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


class MilesTransferCoordinator:
    """Create and retire versioned operations from one frozen group spec."""

    def __init__(
        self,
        rendezvous: CollectiveRendezvous,
        topology: CollectiveTopology,
    ) -> None:
        self._rendezvous = rendezvous
        self._topology = topology
        self._lanes = topology.lanes()

    def create(self, version_id: str, *, idempotency_key: str):
        return self._rendezvous.create_transfer(
            model_name=self._topology.model_name,
            trainer_slots=list(self._topology.trainer_slots),
            generator_slots=list(self._topology.generator_slots),
            lanes=self._lanes,
            version_id=_text(version_id, "version_id"),
            idempotency_key=_text(idempotency_key, "idempotency_key"),
        )

    def get(self, operation_id: str):
        return self._rendezvous.get_transfer(_text(operation_id, "operation_id"))

    def delete(self, operation_id: str):
        return self._rendezvous.delete_transfer(_text(operation_id, "operation_id"))


class MilesTrainerSession:
    """Own one trainer rank's reusable collective membership and lifecycle."""

    def __init__(
        self,
        *,
        client: RefitClientTrainer,
        rendezvous: CollectiveRendezvous,
        publisher: MilesPublisher,
        source_partition: int,
        worker_id: str,
        layer_groups: tuple[tuple[str, ...], ...] = (),
        device: Any = None,
        streams: list[Any] | None = None,
    ) -> None:
        self._client = client
        self._rendezvous = rendezvous
        self._publisher = publisher
        self._source_partition = source_partition
        self._worker_id = _text(worker_id, "worker_id")
        self._groups = _layer_groups(layer_groups, publisher.parameter_names())
        self._device = _client_device(device, publisher.device, "MILES trainer")
        self._streams = list(streams) if streams else [None]
        self._membership = None
        self._prepared = False
        self._closed = False

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
            worker_id=worker_id,
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

    def run_round(self, *, version: str, operation_id: str) -> None:
        if not self._prepared or self._membership is None:
            raise RuntimeError("prepare must complete before a trainer round")
        if self._closed:
            raise RuntimeError("the MILES trainer session is closed")
        version = _text(version, "version")
        operation_id = _text(operation_id, "operation_id")
        try:
            _order_current_cuda_stream_before(self._streams, device=self._device)
            self._client.start_weight_update(version)
            for layer_group_id in range(len(self._groups)):
                self._client.publish_weights(version, layer_group_id)
            self._client.finish_weight_update(
                version,
                operation_id=operation_id,
            )
        except BaseException as error:
            self._fail_round(
                error,
                operation_id=operation_id,
                report=True,
            )
            raise

    def _fail_round(
        self,
        error: BaseException,
        *,
        operation_id: str,
        report: bool,
    ) -> None:
        membership = self._membership
        try:
            self._client.cleanup()
        except BaseException:
            logger.warning(
                "trainer cleanup failed after the round error", exc_info=True
            )
        if report and membership is not None:
            try:
                self._rendezvous.report(
                    operation_id=operation_id,
                    group_id=membership.group_id,
                    epoch=membership.epoch,
                    worker_id=self._worker_id,
                    succeeded=False,
                    message=repr(error),
                )
            except BaseException:
                logger.warning(
                    "reporting the trainer round failure also failed",
                    exc_info=True,
                )
        try:
            self._rendezvous.close()
        except BaseException:
            logger.warning(
                "closing trainer rendezvous after the round error failed",
                exc_info=True,
            )
        self._closed = True

    def report_failure(self, *, operation_id: str, error: BaseException) -> None:
        membership = self.membership
        self._rendezvous.report(
            operation_id=_text(operation_id, "operation_id"),
            group_id=membership.group_id,
            epoch=membership.epoch,
            worker_id=self._worker_id,
            succeeded=False,
            message=repr(error),
        )

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
    "MilesTransferCoordinator",
]
