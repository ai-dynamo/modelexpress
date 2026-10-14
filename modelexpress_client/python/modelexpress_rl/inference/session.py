# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle for one rank-local generator weight update."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import grpc
from modelexpress.refit.timing import refit_span
from modelexpress.adapter import StrategyRecoveryError
from modelexpress.types import ManifestMismatchError

from ..control import TrainerMesh, WeightVersion
from .plan import (
    PreparedArtifact,
    PreparedStreamingTensors,
    WeightSource,
    WeightUpdatePlan,
    WeightUpdatePlanner,
)

logger = logging.getLogger("modelexpress_rl.inference.session")

_RECOVERABLE_ERRORS = (grpc.RpcError, RuntimeError, ManifestMismatchError)


@dataclass
class StagedUpdate:
    """One staged update and its protected version leases."""

    plan: WeightUpdatePlan
    prepared: PreparedArtifact
    lease: Any
    applied: bool = False
    apply_result: Any = None
    released: bool = False
    installation_started: bool = False


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
    """Coordinate one rank-local generator update across composed modules.

    The session owns reusable preparation, version leases, and the update
    lifecycle, but no engine-specific behavior. ``stage`` stages one target from a selected source.
    Object storage targets are first expanded into an ordered replay chain;
    other sources use candidate retries and source fallback.
    During ``apply`` the session installs the prepared artifact at the caller's
    safe point. ``release`` ends update ownership and closes its lease. If
    streaming cleanup is unproven, resources and the lease remain retained
    until process reset. Reusable buffers remain owned by the method.
    Publication of installed runtime tensors is owned by the generator
    runtime, independently of the source method selected for the update.

    For canonical object storage, the selected plan is an
    ``ObjectStorageSourceResolver`` feeding ``CanonicalDeltaUpdateMethod``,
    which produces a ``PreparedCheckpointArtifact`` for the engine installer.
    The client validates the exact serving base before this session acquires a
    lease.
    """

    def __init__(
        self,
        *,
        planner: WeightUpdatePlanner,
        start_lease: Callable[[str], Any],
        resolve_replay_chain: Callable[[WeightVersion], tuple[WeightVersion, ...]]
        | None = None,
    ) -> None:
        self._planner = planner
        self._start_lease = start_lease
        self._resolve_replay_chain = resolve_replay_chain

    def prepare(
        self,
        mesh: TrainerMesh | None,
        validate_mesh: Callable[[], object] | None = None,
    ) -> dict[WeightSource, Exception]:
        """Return recoverable source failures; unsafe engine state remains fatal."""
        errors = {}
        for source_kind in self._planner.source_order:
            try:
                self._planner.prepare(
                    mesh, source_kind=source_kind, validate_mesh=validate_mesh,
                )
            except StrategyRecoveryError:
                raise
            except _RECOVERABLE_ERRORS as error:
                errors[source_kind] = error
        return errors

    def stage(
        self,
        version: WeightVersion,
        *,
        preparation_errors: dict[WeightSource, Exception] | None = None,
    ) -> StagedUpdate:
        """Stage one target and retain its leases until installation or release."""
        source_order = self._planner.source_order
        if not source_order:
            raise RuntimeError("no refit source is configured")
        for index, source_kind in enumerate(source_order):
            try:
                if preparation_errors and source_kind in preparation_errors:
                    raise preparation_errors[source_kind]
                return self._stage_source(version, source_kind=source_kind)
            except StrategyRecoveryError:
                raise
            except _RECOVERABLE_ERRORS as error:
                if index == len(source_order) - 1:
                    raise
                logger.info(
                    "ModelExpress active refit version=%s source=%s unavailable: %s",
                    version.version_id,
                    source_kind.value,
                    error,
                )

    def _stage_source(
        self,
        version: WeightVersion,
        *,
        source_kind: WeightSource,
    ) -> StagedUpdate:
        logger.info(
            "ModelExpress active refit version=%s trying source=%s",
            version.version_id,
            source_kind.value,
        )
        if source_kind is not WeightSource.OBJECT_STORAGE:
            return self._stage_one(version, source_kind=source_kind)
        if self._resolve_replay_chain is None:
            raise RuntimeError("object-storage replay-chain resolver is unavailable")
        versions = self._resolve_replay_chain(version)
        for replay_version in versions:
            self._planner.validate(
                replay_version,
                source_kind=source_kind,
            )
        return self._stage_replay_chain(
            versions,
        )

    def _stage_one(
        self,
        version: WeightVersion,
        *,
        source_kind: WeightSource,
    ) -> StagedUpdate:
        with self._lease_versions((version,)) as lease:
            plan, prepared = self._stage_candidates(version, source_kind=source_kind)
            return StagedUpdate(plan=plan, prepared=prepared, lease=lease)

    @contextmanager
    def _lease_versions(self, versions: tuple[WeightVersion, ...]) -> Iterator[_LeaseGroup]:
        leases = []
        group = _LeaseGroup(leases)
        try:
            with refit_span("setup_registration"):
                for version in versions:
                    leases.append(self._start_lease(version.version_id))
            yield group
        except BaseException as primary_error:
            self._close_lease(group, versions[-1].version_id, primary_error)
            raise

    def _stage_candidates(
        self,
        version: WeightVersion,
        *,
        source_kind: WeightSource,
    ) -> tuple[WeightUpdatePlan, PreparedArtifact]:
        last_error: Exception | None = None
        for plan in self._planner.plans(version, source_kind=source_kind):
            logger.info(
                "ModelExpress weight update version=%s trying source=%s method=%s installer=%s",
                version.version_id,
                plan.source.kind.value,
                type(plan.method).__name__,
                type(plan.installer).__name__,
            )
            try:
                prepared = plan.method.stage(version=version, source=plan.source)
            except StrategyRecoveryError:
                raise
            except _RECOVERABLE_ERRORS as error:
                last_error = error
                logger.warning(
                    "ModelExpress weight update version=%s staging failed source=%s method=%s error=%s",
                    version.version_id,
                    plan.source.kind.value,
                    type(plan.method).__name__,
                    error,
                )
                continue
            logger.info(
                "ModelExpress weight update version=%s staged source=%s method=%s",
                version.version_id,
                plan.source.kind.value,
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
    ) -> StagedUpdate:
        """Stage an already-resolved base-to-target chain atomically."""
        if not versions:
            raise ValueError("version chain is empty")
        if len(versions) == 1:
            return self._stage_one(
                versions[0],
                source_kind=WeightSource.OBJECT_STORAGE,
            )
        with self._lease_versions(versions) as lease_group:
            plans = []
            for version in versions:
                try:
                    plan = next(
                        self._planner.plans(
                            version,
                            source_kind=WeightSource.OBJECT_STORAGE,
                        )
                    )
                except StopIteration as error:
                    raise RuntimeError(
                        f"no usable refit source for replay revision "
                        f"{version.version_id!r}"
                    ) from error
                plans.append(plan)
            target_plan = plans[-1]
            if any(
                plan.method is not target_plan.method
                or plan.installer is not target_plan.installer
                for plan in plans
            ):
                raise RuntimeError("version replay chain requires one update method")
            prepared = target_plan.method.stage_chain(
                tuple((plan.version, plan.source) for plan in plans)
            )
            return StagedUpdate(
                plan=target_plan,
                prepared=prepared,
                lease=lease_group,
            )

    def apply(self, update: StagedUpdate) -> Any:
        if update.released:
            raise RuntimeError("staged weight has already been released")
        if update.applied:
            return update.apply_result
        primary_error: BaseException | None = None
        try:
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
            if not (
                isinstance(update.prepared, PreparedStreamingTensors)
                and update.prepared.ownership.release_blocked
            ):
                self._close_lease(
                    update.lease,
                    update.plan.version.version_id,
                    primary_error,
                )

    def release(self, update: StagedUpdate) -> None:
        if update.released:
            return
        if (
            isinstance(update.prepared, PreparedStreamingTensors)
            and update.prepared.ownership.release_blocked
        ):
            raise RuntimeError(
                "streaming cleanup is unproven; retain resources and reset the process"
            )
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


__all__ = ["StagedUpdate", "WeightUpdateSession"]
