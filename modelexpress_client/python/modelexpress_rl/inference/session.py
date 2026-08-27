# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle for one rank-local generator weight update."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import grpc
from modelexpress.types import ManifestMismatchError

from ..control import WeightVersion
from .plan import PreparedArtifact, WeightUpdatePlan, WeightUpdatePlanner

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


class WeightUpdateSession:
    """Own planning, staging, installation, publication, and cleanup."""

    def __init__(
        self,
        *,
        planner: WeightUpdatePlanner,
        start_lease: Callable[[str], Any],
    ) -> None:
        self._planner = planner
        self._start_lease = start_lease

    def validate(self, version: WeightVersion) -> None:
        self._planner.validate(version)

    def stage(self, version: WeightVersion) -> SessionUpdate:
        lease = self._start_lease(version.version_id)
        try:
            last_error: BaseException | None = None
            found_plan = False
            try:
                for plan in self._planner.plans(version):
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
                    except (
                        grpc.RpcError,
                        RuntimeError,
                        ManifestMismatchError,
                    ) as error:
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
