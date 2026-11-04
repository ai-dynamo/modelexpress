# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle for one rank-local generator weight update."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import grpc
from modelexpress_rl.refit.timing import refit_span
from modelexpress.adapter import StrategyRecoveryError
from modelexpress.types import ManifestMismatchError

from ..control import WeightVersion
from ..train import WeightPayloadFormat
from .plan import (
    ResolvedSource,
    EngineInstaller,
    SourceResolver,
    UpdateMethod,
    WeightSource,
    WeightUpdatePlan,
)

logger = logging.getLogger("modelexpress_rl.inference.session")


@dataclass
class SessionUpdate:
    """An opaque prepared payload and its protected version lease."""

    plan: WeightUpdatePlan
    # Only methods and installers interpret the prepared payload.
    prepared: Any
    lease: Any
    applied: bool = False
    apply_result: Any = None
    released: bool = False
    installation_started: bool = False

    @property
    def metrics(self) -> dict[str, float]:
        return self.plan.method.prepared_metrics(self.prepared)


class _LeaseGroup:
    """Close all revision leases held by one replay operation."""

    def __init__(self, leases: list[Any]) -> None:
        self._leases = leases

    def close(self) -> None:
        first_error: BaseException | None = None
        for lease in reversed(self._leases):
            try:
                lease.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


class WeightUpdateSession:
    """Select sources, prepare updates, and protect their version leases.

    Own source compatibility, retry budgets, fallback and replay-chain preparation
    for one generator rank. Coordinate installation at the caller's safe point and
    retain leases until the method permits release. Keep prepared payloads opaque;
    methods own preparation recovery and transfer resources, while installers own
    engine mutation.
    """

    def __init__(
        self,
        *,
        resolvers: tuple[SourceResolver, ...],
        methods: tuple[UpdateMethod, ...],
        installer: EngineInstaller,
        max_transfer_attempts: int,
        start_lease: Callable[[str], Any],
        resolve_replay_chain: Callable[[WeightVersion], tuple[WeightVersion, ...]]
        | None = None,
    ) -> None:
        self._resolvers = resolvers
        self._methods = methods
        self._installer = installer
        self._max_transfer_attempts = max_transfer_attempts
        self._start_lease = start_lease
        self._resolve_replay_chain = resolve_replay_chain

    def stage(
        self,
        version: WeightVersion,
    ) -> SessionUpdate:
        """Prepare one target and retain its leases until apply or release."""
        if not self._resolvers:
            raise RuntimeError("no refit source is configured")
        for index, resolver in enumerate(self._resolvers):
            try:
                return self._stage_source(version, resolver=resolver)
            except StrategyRecoveryError:
                raise
            except (
                grpc.RpcError,
                RuntimeError,
                ManifestMismatchError,
            ) as error:
                if index == len(self._resolvers) - 1:
                    raise
                logger.info(
                    "ModelExpress active refit version=%s source=%s unavailable: %s",
                    version.version_id,
                    resolver.kind.value,
                    error,
                )

    def _stage_source(
        self,
        version: WeightVersion,
        *,
        resolver: SourceResolver,
    ) -> SessionUpdate:
        logger.info(
            "ModelExpress active refit version=%s trying source=%s",
            version.version_id,
            resolver.kind.value,
        )
        if resolver.kind is not WeightSource.OBJECT_STORAGE:
            return self._stage_one(version, resolver=resolver)
        if self._resolve_replay_chain is None:
            raise RuntimeError("object-storage replay-chain resolver is unavailable")
        versions = self._resolve_replay_chain(version)
        for replay_version in versions:
            self._validate(replay_version, resolver)
        return self._stage_replay_chain(
            versions,
            resolver,
        )

    def _stage_one(
        self,
        version: WeightVersion,
        *,
        resolver: SourceResolver,
    ) -> SessionUpdate:
        with refit_span("setup_registration"):
            lease = self._start_lease(version.version_id)
        try:
            plan, prepared = self._prepare_candidates(version, resolver=resolver)
            return SessionUpdate(plan=plan, prepared=prepared, lease=lease)
        except BaseException as primary_error:
            self._close_lease(lease, version.version_id, primary_error)
            raise

    @property
    def source_order(self) -> tuple[WeightSource, ...]:
        return tuple(resolver.kind for resolver in self._resolvers)

    def _select_method(
        self,
        version: WeightVersion,
        kind: WeightSource,
        payload_format: WeightPayloadFormat,
    ) -> UpdateMethod | None:
        for method in self._methods:
            capabilities = method.capabilities
            if (
                payload_format in capabilities.payload_formats
                and kind in capabilities.sources
                and capabilities.artifact_type
                in self._installer.capabilities.artifact_types
                and (
                    not capabilities.requires_base_version
                    or version.base_version_id is not None
                )
            ):
                return method
        return None

    def _validate(self, version: WeightVersion, resolver: SourceResolver) -> None:
        if (
            resolver.supports(version)
            and self._select_method(
                version, resolver.kind, resolver.payload_format(version)
            )
            is not None
        ):
            return
        raise RuntimeError(
            "no legal source, update method, and engine installer composition for "
            f"payload={version.payload_format.value}"
        )

    def _plan(
        self, version: WeightVersion, source: ResolvedSource
    ) -> WeightUpdatePlan | None:
        method = self._select_method(version, source.kind, source.payload_format)
        if method is not None:
            return WeightUpdatePlan(version, source, method, self._installer)
        return None

    def _prepare_candidates(
        self,
        version: WeightVersion,
        *,
        resolver: SourceResolver,
    ) -> tuple[WeightUpdatePlan, Any]:
        last_error = None
        if resolver.supports(version):
            candidates = iter(resolver.candidates(version))
            seen = 0
            only_source = None
            for _ in range(self._max_transfer_attempts):
                try:
                    source = next(candidates)
                    seen += 1
                    only_source = source if seen == 1 else None
                except StopIteration:
                    if seen != 1 or resolver.kind is WeightSource.GENERATOR:
                        break
                    source = only_source
                plan = self._plan(version, source)
                if plan is None:
                    continue
                logger.info(
                    "ModelExpress weight update version=%s trying source=%s method=%s installer=%s",
                    version.version_id,
                    source.kind.value,
                    type(plan.method).__name__,
                    type(plan.installer).__name__,
                )
                try:
                    prepared = plan.method.prepare(version=version, source=source)
                except (grpc.RpcError, RuntimeError, ManifestMismatchError) as error:
                    plan.method.recover_preparation(error)
                    last_error = error
                    logger.warning(
                        "ModelExpress weight update version=%s preparation failed source=%s method=%s error=%s",
                        version.version_id,
                        source.kind.value,
                        type(plan.method).__name__,
                        error,
                    )
                    continue
                logger.info(
                    "ModelExpress weight update version=%s prepared source=%s method=%s",
                    version.version_id,
                    source.kind.value,
                    type(plan.method).__name__,
                )
                return plan, prepared
        if last_error is not None:
            raise last_error
        raise RuntimeError(
            f"no usable refit source for weight version {version.version_id!r}"
        )

    def _stage_replay_chain(
        self,
        versions: tuple[WeightVersion, ...],
        resolver: SourceResolver,
    ) -> SessionUpdate:
        """Prepare an already-resolved base-to-target chain atomically."""
        if not versions:
            raise ValueError("version chain is empty")
        if len(versions) == 1:
            return self._stage_one(
                versions[0],
                resolver=resolver,
            )
        leases = []
        try:
            with refit_span("setup_registration"):
                for version in versions:
                    leases.append(self._start_lease(version.version_id))
        except BaseException as primary_error:
            self._close_lease(
                _LeaseGroup(leases), versions[-1].version_id, primary_error
            )
            raise
        lease_group = _LeaseGroup(leases)
        try:
            plans = []
            for version in versions:
                source = next(iter(resolver.candidates(version)), None)
                plan = None if source is None else self._plan(version, source)
                if plan is None:
                    raise RuntimeError(
                        f"no usable refit source for replay revision {version.version_id!r}"
                    )
                plans.append(plan)
            target_plan = plans[-1]
            if any(
                plan.method is not target_plan.method
                or plan.installer is not target_plan.installer
                for plan in plans
            ):
                raise RuntimeError("version replay chain requires one update method")
            try:
                prepared = target_plan.method.prepare_chain(
                    tuple((plan.version, plan.source) for plan in plans)
                )
            except BaseException as error:
                target_plan.method.recover_preparation(error)
                raise
            return SessionUpdate(
                plan=target_plan,
                prepared=prepared,
                lease=lease_group,
            )
        except BaseException as primary_error:
            self._close_lease(lease_group, versions[-1].version_id, primary_error)
            raise

    def apply(self, update: SessionUpdate) -> Any:
        if update.released:
            raise RuntimeError("staged weight has already been released")
        primary_error: BaseException | None = None
        try:
            if update.applied:
                return update.apply_result
            logger.info(
                "ModelExpress weight update version=%s installing "
                "source=%s method=%s installer=%s",
                update.plan.version.version_id,
                update.plan.source.kind.value,
                type(update.plan.method).__name__,
                type(update.plan.installer).__name__,
            )
            with update.plan.method.installation_context(update.prepared):
                update.installation_started = True
                update.apply_result = update.plan.installer.install(update.prepared)
            update.applied = True
            logger.info(
                "ModelExpress weight update version=%s installed "
                "source=%s method=%s installer=%s",
                update.plan.version.version_id,
                update.plan.source.kind.value,
                type(update.plan.method).__name__,
                type(update.plan.installer).__name__,
            )
            return update.apply_result
        except BaseException as error:
            primary_error = error
            if not update.installation_started:
                update.installation_started = (
                    update.plan.method.mutated_during_installation_context(
                        update.prepared
                    )
                )
            if update.installation_started:
                try:
                    update.plan.method.installation_failed(update.prepared)
                except Exception:
                    logger.exception(
                        "failed to fence state after installation failure for %s",
                        update.plan.version.version_id,
                    )
            raise
        finally:
            if update.plan.method.can_release(update.prepared):
                self._close_lease(
                    update.lease,
                    update.plan.version.version_id,
                    primary_error,
                )

    def release(self, update: SessionUpdate) -> None:
        if update.released:
            return
        update.plan.method.validate_release(update.prepared)
        primary_error: BaseException | None = None
        try:
            update.plan.method.release(update.prepared)
            logger.info(
                "ModelExpress weight update version=%s released",
                update.plan.version.version_id,
            )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            update.released = True
            self._close_lease(
                update.lease,
                update.plan.version.version_id,
                primary_error,
            )

    @staticmethod
    def _close_lease(
        lease,
        version_id: str,
        primary_error: BaseException | None,
    ) -> None:
        try:
            lease.close()
        except grpc.RpcError:
            if primary_error is None:
                raise
            logger.warning(
                "failed to release version %s lease while handling %s",
                version_id,
                type(primary_error).__name__,
                exc_info=True,
            )


__all__ = ["SessionUpdate", "WeightUpdateSession"]
