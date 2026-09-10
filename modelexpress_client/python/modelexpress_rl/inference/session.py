# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle for one rank-local generator weight update."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import grpc
from modelexpress.adapter import StrategyRecoveryError
from modelexpress.types import ManifestMismatchError

from ..control import WeightVersion
from .plan import (
    PreparedArtifact,
    UpdateMethod,
    WeightSource,
    WeightUpdatePlan,
    WeightUpdatePlanner,
)

logger = logging.getLogger("modelexpress_rl.inference.session")


@dataclass
class SessionUpdate:
    """Active prepared update and its protected version lease."""

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

    The session owns the version lease and update lifecycle, but no engine-
    specific behavior. ``stage`` prepares one target from a selected source.
    Object storage targets are first expanded into an ordered replay chain;
    other sources retain the planner's candidate retry and fallback behavior.
    During ``apply`` the session installs the prepared artifact at the caller's
    safe point and lets the selected method advertise the applied version when
    that method supports peer publication. ``release`` always returns method-
    owned staging and the version lease.

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
        resolve_replay_chain: Callable[
            [WeightVersion], tuple[WeightVersion, ...]
        ]
        | None = None,
    ) -> None:
        self._planner = planner
        self._start_lease = start_lease
        self._resolve_replay_chain = resolve_replay_chain

    def stage(
        self,
        version: WeightVersion,
        *,
        source_kind: WeightSource | None = None,
    ) -> SessionUpdate:
        """Prepare one target version from the selected source."""
        if source_kind is not WeightSource.OBJECT_STORAGE:
            self._planner.validate(version, source_kind=source_kind)
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
            source_kind=source_kind,
        )

    def _stage_one(
        self,
        version: WeightVersion,
        *,
        source_kind: WeightSource | None = None,
    ) -> SessionUpdate:
        lease = self._start_lease(version.version_id)
        try:
            last_error: BaseException | None = None
            found_plan = False
            try:
                for plan in self._planner.plans(
                    version,
                    source_kind=source_kind,
                ):
                    found_plan = True
                    source = plan.source.kind.value
                    method = type(plan.method).__name__
                    installer = type(plan.installer).__name__
                    logger.info(
                        "ModelExpress weight update version=%s trying "
                        "source=%s method=%s installer=%s",
                        version.version_id,
                        source,
                        method,
                        installer,
                    )
                    try:
                        prepared = plan.method.prepare(
                            version=version,
                            source=plan.source,
                        )
                    except StrategyRecoveryError:
                        raise
                    except (
                        grpc.RpcError,
                        RuntimeError,
                        ManifestMismatchError,
                    ) as error:
                        self._recover_preparation(plan.method, error)
                        last_error = error
                        logger.warning(
                            "ModelExpress weight update version=%s preparation "
                            "failed source=%s method=%s error=%s",
                            version.version_id,
                            source,
                            method,
                            error,
                        )
                        continue
                    logger.info(
                        "ModelExpress weight update version=%s prepared "
                        "source=%s method=%s",
                        version.version_id,
                        source,
                        method,
                    )
                    return SessionUpdate(
                        plan=plan,
                        prepared=prepared,
                        lease=lease,
                    )
            except (grpc.RpcError, RuntimeError) as error:
                last_error = error
            if not found_plan and last_error is None:
                last_error = RuntimeError(
                    f"no usable refit source for weight version {version.version_id!r}"
                )
            assert last_error is not None
            raise last_error
        except BaseException as primary_error:
            self._close_lease(lease, version.version_id, primary_error)
            raise

    def _stage_replay_chain(
        self,
        versions: tuple[WeightVersion, ...],
        *,
        source_kind: WeightSource | None = None,
    ) -> SessionUpdate:
        """Prepare an already-resolved base-to-target chain atomically."""
        if not versions:
            raise ValueError("version chain is empty")
        if len(versions) == 1:
            return self._stage_one(
                versions[0],
                source_kind=source_kind,
            )
        leases = []
        try:
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
                try:
                    plan = next(
                        self._planner.plans(
                            version,
                            source_kind=source_kind,
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
            try:
                prepared = target_plan.method.prepare_chain(
                    tuple((plan.version, plan.source) for plan in plans)
                )
            except BaseException as error:
                self._recover_preparation(target_plan.method, error)
                raise
            return SessionUpdate(
                plan=target_plan,
                prepared=prepared,
                lease=lease_group,
            )
        except BaseException as primary_error:
            self._close_lease(lease_group, versions[-1].version_id, primary_error)
            raise

    @staticmethod
    def _recover_preparation(
        method: UpdateMethod,
        primary_error: BaseException,
    ) -> None:
        try:
            method.preparation_failed()
        except Exception as recovery_error:
            raise StrategyRecoveryError(
                f"{type(method).__name__} could not recover after preparation "
                f"failed: {primary_error}"
            ) from recovery_error

    def apply(self, update: SessionUpdate) -> Any:
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
            try:
                update.plan.method.publish_applied(
                    version_id=update.plan.version.version_id,
                    prepared=update.prepared,
                )
            except Exception:
                logger.exception(
                    "failed to publish applied version %s as a P2P source",
                    update.plan.version.version_id,
                )
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
            self._close_lease(
                update.lease,
                update.plan.version.version_id,
                primary_error,
            )

    def release(self, update: SessionUpdate) -> None:
        if update.released:
            return
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
