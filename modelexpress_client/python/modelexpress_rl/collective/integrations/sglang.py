# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang generator-side engine boundary for the MILES collective bridge."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..client import RefitClientGenerator
from ..rendezvous import CollectiveRendezvous, Membership
from ..spi import LocalParamSpec
from ..types import ReshardPlan
from ._common import (
    _check_stable,
    _client_device,
    _collective_streams,
    _FrozenPlan,
    _layer_groups,
    _order_current_cuda_stream_before,
    _single_device,
    _tensor_signature,
    _text,
)
from .miles import CollectiveTopology

logger = logging.getLogger("modelexpress_rl.collective.integrations.sglang")


class SglangLoader:
    """``Loader`` over a live SGLang model, one layer group at a time.

    Every canonical parameter lands in a persistent scratch buffer handed to
    the model's own ``load_weights``, which owns fusion and TP layout. A
    failed ``install`` can leave a partially written model, so it poisons the
    loader.
    """

    def __init__(
        self,
        *,
        plan: ReshardPlan,
        model: Any,
        device: Any,
        layer_groups: tuple[tuple[str, ...], ...] = (),
    ) -> None:
        import torch

        self._plan = _FrozenPlan(plan)
        self._model = model
        self._groups = _layer_groups(layer_groups, self._plan.names())
        self._buffers: dict[str, Any] = {}
        self._signatures = {}
        for name in self._plan.names():
            entry = self._plan.entry(name)
            buffer = torch.empty(
                entry.global_shape, dtype=torch.bfloat16, device=device
            )
            self._buffers[name] = buffer
            self._signatures[name] = _tensor_signature(
                name,
                buffer,
                expected_shape=entry.global_shape,
                expected_dtype=entry.dtype,
            )
        self._device = _single_device(self._signatures, "SGLang loader")
        self._round_version: str | None = None
        self._poisoned = False

    @property
    def source_partition_count(self) -> int:
        return self._plan.source_partition_count

    @property
    def device(self) -> str:
        return self._device

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def layer_groups(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(group) for group in self._groups)

    def validate_topology(self, topology: CollectiveTopology) -> None:
        self._plan.validate_topology(topology)

    def _require_healthy(self) -> None:
        if self._poisoned:
            raise RuntimeError(
                "the SGLang loader is poisoned by a failed install; the live "
                "model may hold a partial weight set"
            )

    def _validate_stable(self) -> None:
        for name, buffer in self._buffers.items():
            entry = self._plan.entry(name)
            _check_stable(
                name,
                buffer,
                self._signatures[name],
                expected_shape=entry.global_shape,
                expected_dtype=entry.dtype,
            )

    # --- Loader protocol -------------------------------------------------

    def capture(self) -> ReshardPlan:
        self._validate_stable()
        return self._plan.capture()

    def parameter_names(self) -> list[str]:
        return self._plan.names()

    def local_params(self) -> dict[str, LocalParamSpec]:
        self._require_healthy()
        self._validate_stable()
        return {
            name: LocalParamSpec(base=buffer) for name, buffer in self._buffers.items()
        }

    def start_new_round(self, version: str) -> None:
        self._require_healthy()
        version = _text(version, "version")
        if self._round_version is not None:
            raise RuntimeError(
                f"a round for version {self._round_version!r} is already in flight"
            )
        self._validate_stable()
        self._round_version = version

    def install(self, layer_group_id: int) -> None:
        """Hand one layer group's received tensors to SGLang's own loader."""
        self._require_healthy()
        if self._round_version is None:
            raise RuntimeError("start_new_round must run before install")
        if not 0 <= layer_group_id < len(self._groups):
            raise ValueError(
                f"layer_group_id {layer_group_id} is outside the "
                f"{len(self._groups)} declared layer groups"
            )
        names = self._groups[layer_group_id]
        try:
            self._model.load_weights([(name, self._buffers[name]) for name in names])
        except BaseException:
            self._poisoned = True
            raise

    def finish(self) -> None:
        self._require_healthy()
        if self._round_version is None:
            raise RuntimeError("start_new_round must run before finish")
        self._round_version = None

    def fail_round(self, *, possibly_mutated: bool) -> None:
        """Retire a failed round; a round that may have written the model poisons."""
        if possibly_mutated:
            self._poisoned = True
        self._round_version = None

    def cleanup(self) -> None:
        self._round_version = None
        self._buffers.clear()


class SglangGeneratorSession:
    """Own one generator rank's collective membership and round lifecycle.

    The session closes its rendezvous on every teardown path; the receiver's
    teardown (``close_generator_resources``) closes it again as the final
    owner, which is safe because ``CollectiveRendezvous.close()`` is
    idempotent.
    """

    def __init__(
        self,
        *,
        client: RefitClientGenerator,
        rendezvous: CollectiveRendezvous,
        loader: SglangLoader,
        worker_id: str,
        device: Any = None,
        streams: list[Any] | None = None,
    ) -> None:
        self._client = client
        self._rendezvous = rendezvous
        self._loader = loader
        self._worker_id = _text(worker_id, "worker_id")
        self._groups = [list(group) for group in loader.layer_groups]
        self._device = device
        self._streams = streams
        self._membership: Membership | None = None
        self._prepared = False
        self._closed = False

    @classmethod
    def create(
        cls,
        *,
        rendezvous: CollectiveRendezvous,
        topology: CollectiveTopology,
        loader: SglangLoader,
        slot_id: str,
        worker_id: str,
        index_in_role: int,
        device: Any = None,
        streams: list[Any] | None = None,
    ) -> SglangGeneratorSession:
        loader.validate_topology(topology)
        device = _client_device(device, loader.device, "SGLang generator")
        streams = _collective_streams(streams, device=device)
        client = RefitClientGenerator(
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
            loader=loader,
            worker_id=worker_id,
            device=device,
            streams=streams,
        )

    @property
    def membership(self) -> Membership:
        if self._membership is None:
            raise RuntimeError("prepare must complete before reading membership")
        return self._membership

    def prepare(self) -> Membership:
        if self._closed:
            raise RuntimeError("the SGLang generator session is closed")
        if self._prepared and self._membership is not None:
            return self._membership
        try:
            self._client.initialize(self._loader)
            self._client.setup_layer_groups(self._groups)
            self._membership = self._client.compute_plan()
            self._prepared = True
            return self._membership
        except BaseException:
            self.close()
            raise

    def run_round(self, *, version: str, operation_id: str) -> None:
        if not self._prepared or self._membership is None:
            raise RuntimeError("prepare must complete before a generator round")
        if self._closed:
            raise RuntimeError("the SGLang generator session is closed")
        version = _text(version, "version")
        operation_id = _text(operation_id, "operation_id")
        update_attempted = False
        try:
            _order_current_cuda_stream_before(self._streams, device=self._device)
            self._client.start_weight_update(version)
            for layer_group_id in range(len(self._groups)):
                update_attempted = True
                self._client.update_weights(version, layer_group_id)
            self._client.finish_weight_update(version, operation_id=operation_id)
        except BaseException as error:
            self._loader.fail_round(possibly_mutated=update_attempted)
            self._fail_round(error, operation_id=operation_id)
            raise

    def _fail_round(self, error: BaseException, *, operation_id: str) -> None:
        # The round failure itself propagates; this is the teardown path. The
        # failure report is best effort: when finish_weight_update already
        # reported it, the server keeps the first terminal result.
        membership = self._membership
        try:
            self._client.cleanup()
        except BaseException:
            logger.warning(
                "generator cleanup failed after the round error", exc_info=True
            )
        if membership is not None:
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
                    "reporting the generator round failure also failed",
                    exc_info=True,
                )
        try:
            self._rendezvous.close()
        except BaseException:
            logger.warning(
                "closing generator rendezvous after the round error failed",
                exc_info=True,
            )
        self._closed = True

    def close(self) -> None:
        if self._closed:
            return
        # One-shot, like the trainer session: a failed teardown must not rerun.
        self._closed = True
        try:
            self._client.cleanup()
        finally:
            self._rendezvous.close()


__all__ = [
    "GeneratorLoader",
    "SglangGeneratorSession",
    "SglangLoader",
    "build_generator_loader",
    "close_generator_resources",
]


@dataclass(frozen=True)
class GeneratorLoader:
    """A generator rank's loader with the session inputs derived from its plan."""

    loader: SglangLoader
    slot_id: str
    local_index: int


def build_generator_loader(
    *,
    plan: ReshardPlan,
    topology: CollectiveTopology,
    model: Any,
    device: Any,
    generator_slot_offset: int,
    tp_rank: int,
) -> GeneratorLoader:
    """Build one SGLang scheduler's loader and its session inputs from a plan.

    Used by the public receiver factory; it owns the slot and the layer
    groups.
    """
    local_index = generator_slot_offset + tp_rank
    try:
        slot_id = topology.generator_slots[local_index]
    except IndexError as error:
        raise RuntimeError(
            "the explicit generator topology does not contain generator index "
            f"{local_index}"
        ) from error

    # Both boundaries require canonical plan order; contiguous trainer groups
    # and receiver singletons therefore traverse the same per-lane sequence.
    loader = SglangLoader(
        plan=plan,
        model=model,
        device=device,
        layer_groups=tuple((entry.name,) for entry in plan.bulk),
    )
    return GeneratorLoader(
        loader=loader,
        slot_id=slot_id,
        local_index=local_index,
    )


def close_generator_resources(
    session: SglangGeneratorSession | None,
    rendezvous: CollectiveRendezvous | None,
    channel: Any,
    *,
    suppress_errors: bool,
) -> None:
    """Close a generator's session, rendezvous and channel, in that order.

    Every resource is attempted. The first failure propagates after the rest
    settle unless ``suppress_errors`` is set; each failure is logged.
    """
    first_error = None
    for name, resource in (
        ("session", session),
        ("rendezvous", rendezvous),
        ("channel", channel),
    ):
        if resource is None:
            continue
        try:
            resource.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
            logger.warning(
                "closing ModelExpress collective %s failed", name, exc_info=True
            )
    if first_error is not None and not suppress_errors:
        raise first_error
