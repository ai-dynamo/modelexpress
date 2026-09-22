# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang-side bindings for the NCCL M2N collective SPI."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ContextManager

from .. import envs
from ..client import RefitClientGenerator
from ..rendezvous import CollectiveRendezvous
from ..spi import LocalParamSpec
from ._common import (
    _FrozenPlan,
    _check_stable,
    _client_device,
    _collective_streams,
    _exact_tensor_sha256,
    _layer_groups,
    _local_shape,
    _single_device,
    _tensor_signature,
    _text,
)
from .wire import TensorDigestMap, tensor_equality_receipt, validate_tensor_digests

logger = logging.getLogger("modelexpress_rl.collective.integrations.sglang")


@dataclass(frozen=True)
class SglangParameterBinding:
    """Live SGLang storage and, when needed, an explicit staging adapter."""

    live: Any
    receive: Any | None = None
    install: Callable[[Any, Any], None] | None = None

    def __post_init__(self) -> None:
        if (self.receive is None) != (self.install is None):
            raise ValueError(
                "staged SGLang bindings require both receive storage and an "
                "install callback"
            )

    @property
    def direct(self) -> bool:
        return self.receive is None

    @property
    def wire_tensor(self) -> Any:
        return self.live if self.direct else self.receive


class SglangLoader:
    """Bind explicit canonical parameters to stable SGLang-owned buffers."""

    def __init__(
        self,
        *,
        plan,
        bindings: dict[str, SglangParameterBinding],
        layer_groups: tuple[tuple[str, ...], ...] = (),
        finish: Callable[[], None] | None = None,
        cleanup: Callable[[], None] | None = None,
        verify_tensor_equality: bool | None = None,
    ) -> None:
        self._plan = _FrozenPlan(plan)
        self._bindings = dict(bindings)
        expected = self._plan.names()
        if list(self._bindings) != expected:
            raise ValueError(
                "SGLang bindings must exactly cover the collective plan in plan "
                f"order; expected {expected}, got {list(self._bindings)}"
            )
        self._groups = _layer_groups(layer_groups, expected)
        self._finish_callback = finish
        self._cleanup_callback = cleanup
        self._live_signatures = {}
        self._wire_signatures = {}
        for name, binding in self._bindings.items():
            entry = self._plan.entry(name)
            local_shape = _local_shape(
                entry.global_shape,
                entry.dst_mesh,
                entry.dst_placements,
            )
            self._live_signatures[name] = _tensor_signature(
                name,
                binding.live,
                expected_shape=local_shape,
                expected_dtype=entry.dtype,
            )
            self._wire_signatures[name] = _tensor_signature(
                name,
                binding.wire_tensor,
                expected_shape=local_shape,
                expected_dtype=entry.dtype,
            )
            if self._live_signatures[name].device != self._wire_signatures[name].device:
                raise ValueError(
                    f"{name}: live and receive storage must share one CUDA device"
                )
        self._device = _single_device(self._live_signatures, "SGLang loader")
        self._poisoned = False
        self._round_started = False
        self._mutation_possible = False
        self._installed_groups: set[int] = set()
        self._verified_names: set[str] = set()
        self._expected_digests: dict[str, str] | None = None
        self._round_version: str | None = None
        self._operation_id: str | None = None
        self._verify_tensor_equality = (
            envs.MX_MILES_VERIFY_TENSOR_EQUALITY
            if verify_tensor_equality is None
            else verify_tensor_equality
        )

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def uses_direct_buffers(self) -> bool:
        return any(binding.direct for binding in self._bindings.values())

    def _require_healthy(self) -> None:
        if self._poisoned:
            raise RuntimeError(
                "the SGLang loader is poisoned by a possibly mutating failure; "
                "restart the worker before serving or refitting"
            )

    def _validate_names(self, names: list[str]) -> None:
        for name in names:
            entry = self._plan.entry(name)
            binding = self._bindings[name]
            local_shape = _local_shape(
                entry.global_shape,
                entry.dst_mesh,
                entry.dst_placements,
            )
            _check_stable(
                name,
                binding.live,
                self._live_signatures[name],
                expected_shape=local_shape,
                expected_dtype=entry.dtype,
            )
            _check_stable(
                name,
                binding.wire_tensor,
                self._wire_signatures[name],
                expected_shape=local_shape,
                expected_dtype=entry.dtype,
            )

    def capture(self):
        self._require_healthy()
        self._validate_names(self._plan.names())
        return self._plan.capture()

    def parameter_names(self) -> list[str]:
        return self._plan.names()

    @property
    def source_partition_count(self) -> int:
        return self._plan.source_partition_count

    @property
    def device(self) -> str:
        return self._device

    def validate_topology(self, topology: Any) -> None:
        self._plan.validate_topology(topology)

    def local_params(self) -> dict[str, LocalParamSpec]:
        self._require_healthy()
        self._validate_names(self._plan.names())
        return {
            name: LocalParamSpec(base=binding.wire_tensor)
            for name, binding in self._bindings.items()
        }

    def expect_round(
        self,
        *,
        version: str,
        operation_id: str,
        tensor_digests: TensorDigestMap | None,
    ) -> None:
        self._require_healthy()
        try:
            version = _text(version, "version")
            operation_id = _text(operation_id, "operation_id")
            if not self._verify_tensor_equality:
                if tensor_digests is not None:
                    raise ValueError(
                        "round supplied tensor digests while exact equality "
                        "verification is disabled"
                    )
                return
            tensor_digests = validate_tensor_digests(tensor_digests)
            expected_names = sorted(self._plan.names())
            digest_names = [name for name, _digest in tensor_digests]
            if digest_names != expected_names:
                missing = sorted(set(expected_names) - set(digest_names))
                extra = sorted(set(digest_names) - set(expected_names))
                raise ValueError(
                    "round tensor digests must exactly match the frozen plan "
                    f"(missing={missing[:5]}, extra={extra[:5]})"
                )
        except BaseException:
            self._poisoned = True
            raise
        self._expected_digests = dict(tensor_digests)
        self._round_version = version
        self._operation_id = operation_id

    def start_new_round(self, version: str) -> None:
        self._require_healthy()
        version = _text(version, "version")
        if self._verify_tensor_equality and (
            self._expected_digests is None
            or self._round_version is None
            or self._operation_id is None
        ):
            raise RuntimeError("round tensor digests must be installed before start")
        if self._verify_tensor_equality and version != self._round_version:
            raise ValueError(
                f"round digest version {self._round_version!r} does not match "
                f"start version {version!r}"
            )
        self._validate_names(self._plan.names())
        self._round_started = True
        self._mutation_possible = False
        self._installed_groups.clear()
        self._verified_names.clear()

    def install(self, layer_group_id: int) -> None:
        self._require_healthy()
        if not self._round_started:
            raise RuntimeError("start_new_round must run before SGLang install")
        if not 0 <= layer_group_id < len(self._groups):
            raise ValueError(f"unknown layer_group_id {layer_group_id}")
        if layer_group_id in self._installed_groups:
            raise RuntimeError(f"layer group {layer_group_id} was already installed")

        names = self._groups[layer_group_id]
        self._validate_names(names)
        if any(self._bindings[name].direct for name in names):
            self._mutation_possible = True
        try:
            if self._verify_tensor_equality:
                if self._expected_digests is None:
                    raise RuntimeError("round tensor digests are unavailable")
                actual_digests = {
                    name: _exact_tensor_sha256(self._bindings[name].wire_tensor)
                    for name in names
                }
                mismatches = [
                    (name, self._expected_digests[name], actual_digests[name])
                    for name in names
                    if actual_digests[name] != self._expected_digests[name]
                ]
                if mismatches:
                    self._poisoned = True
                    name, expected, actual = mismatches[0]
                    raise RuntimeError(
                        f"{name}: tensor digest mismatch; expected {expected}, "
                        f"got {actual}"
                    )
            for name in names:
                binding = self._bindings[name]
                if binding.direct:
                    continue
                self._mutation_possible = True
                if binding.install is None:
                    raise RuntimeError(f"{name}: staged install callback is missing")
                binding.install(binding.receive, binding.live)
        except BaseException:
            if self._mutation_possible:
                self._poisoned = True
            raise
        if self._verify_tensor_equality:
            self._verified_names.update(names)
        self._installed_groups.add(layer_group_id)

    def finish(self) -> None:
        self._require_healthy()
        if self._installed_groups != set(range(len(self._groups))):
            missing = sorted(set(range(len(self._groups))) - self._installed_groups)
            raise RuntimeError(f"SGLang finish is missing layer groups {missing}")
        if self._verify_tensor_equality:
            expected_names = set(self._plan.names())
            if self._verified_names != expected_names:
                missing = sorted(expected_names - self._verified_names)
                raise RuntimeError(
                    f"SGLang finish is missing tensor digest verification for {missing}"
                )
        try:
            if self._finish_callback is not None:
                self._mutation_possible = True
                self._finish_callback()
            if self._verify_tensor_equality:
                if (
                    self._expected_digests is None
                    or self._round_version is None
                    or self._operation_id is None
                ):
                    raise RuntimeError(
                        "round tensor equality receipt state is unavailable"
                    )
                digest_map = tuple(sorted(self._expected_digests.items()))
                receipt = tensor_equality_receipt(
                    version=self._round_version,
                    operation_id=self._operation_id,
                    tensor_digests=digest_map,
                )
                logger.info(
                    "MILES tensor equality receipt role=generator version=%s "
                    "operation_id=%s tensor_count=%d sha256=%s",
                    self._round_version,
                    self._operation_id,
                    len(digest_map),
                    receipt,
                )
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._round_started = False
            self._expected_digests = None
            self._round_version = None
            self._operation_id = None

    def fail_round(self, *, possibly_mutated: bool) -> None:
        if possibly_mutated or self._mutation_possible:
            self._poisoned = True
        self._round_started = False

    def cleanup(self) -> None:
        if self._cleanup_callback is not None:
            self._cleanup_callback()
        self._round_started = False


class SglangGeneratorSession:
    """Run one generator rank's full collective inside an engine safe point."""

    def __init__(
        self,
        *,
        client: RefitClientGenerator,
        rendezvous: CollectiveRendezvous,
        loader: SglangLoader,
        worker_id: str,
        safe_point: Callable[[], ContextManager[None]],
        layer_groups: tuple[tuple[str, ...], ...] = (),
    ) -> None:
        self._client = client
        self._rendezvous = rendezvous
        self._loader = loader
        self._worker_id = _text(worker_id, "worker_id")
        self._safe_point = safe_point
        self._groups = _layer_groups(layer_groups, loader.parameter_names())
        self._membership = None
        self._prepared = False
        self._closed = False

    @classmethod
    def create(
        cls,
        *,
        rendezvous: CollectiveRendezvous,
        topology: Any,
        loader: SglangLoader,
        slot_id: str,
        worker_id: str,
        index_in_role: int,
        safe_point: Callable[[], ContextManager[None]],
        layer_groups: tuple[tuple[str, ...], ...] = (),
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
            safe_point=safe_point,
            layer_groups=layer_groups,
        )

    def prepare(self):
        if self._closed:
            raise RuntimeError("the SGLang generator session is closed")
        if self._prepared:
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

    def run_round(
        self,
        *,
        version: str,
        operation_id: str,
        tensor_digests: TensorDigestMap | None,
    ) -> None:
        if not self._prepared or self._membership is None:
            raise RuntimeError("prepare must complete before a generator round")
        if self._closed:
            raise RuntimeError("the SGLang generator session is closed")
        version = _text(version, "version")
        operation_id = _text(operation_id, "operation_id")
        update_attempted = False
        try:
            self._loader.expect_round(
                version=version,
                operation_id=operation_id,
                tensor_digests=tensor_digests,
            )
            with self._safe_point():
                self._client.start_weight_update(version)
                for layer_group_id in range(len(self._groups)):
                    update_attempted = True
                    self._client.update_weights(version, layer_group_id)
                self._client.finish_weight_update(
                    version,
                    operation_id=operation_id,
                )
        except BaseException as error:
            self._loader.fail_round(
                possibly_mutated=update_attempted and self._loader.uses_direct_buffers
            )
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
                "generator cleanup failed after the round error", exc_info=True
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
        try:
            self._client.cleanup()
        finally:
            self._rendezvous.close()
            self._closed = True


__all__ = [
    "SglangGeneratorSession",
    "SglangLoader",
    "SglangParameterBinding",
]
