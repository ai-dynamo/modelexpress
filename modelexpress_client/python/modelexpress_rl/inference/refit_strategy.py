# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered active-refit staging strategies."""

from __future__ import annotations

import logging

import grpc
from modelexpress.adapter import StrategyRecoveryError
from modelexpress.types import ManifestMismatchError

from ..control import WeightVersion
from .plan import WeightSource
from .session import SessionUpdate, WeightUpdateSession

logger = logging.getLogger("modelexpress_rl.inference.refit_strategy")


class RefitStageChain:
    """Prepare an exact version using the configured source order.

    Staging transfers or reconstructs and validates the target, but does not
    install it into the live engine. Source fallback is safe only in this phase;
    ``WeightUpdateSession.apply`` performs the later engine mutation at its safe
    point.
    """

    def __init__(
        self,
        *,
        source_order: tuple[WeightSource, ...],
        session: WeightUpdateSession,
    ) -> None:
        if not source_order:
            raise ValueError("refit stage chain must not be empty")
        self._source_order: tuple[WeightSource | None, ...] = (
            source_order
            if WeightSource.OBJECT_STORAGE in source_order
            else (None,)
        )
        self._session = session

    def stage(self, version: WeightVersion) -> SessionUpdate:
        for source_kind in self._source_order[:-1]:
            try:
                return self._stage_source(version, source_kind=source_kind)
            except StrategyRecoveryError:
                raise
            except (
                grpc.RpcError,
                RuntimeError,
                ManifestMismatchError,
            ) as error:
                logger.info(
                    "ModelExpress active refit version=%s source=%s unavailable: %s",
                    version.version_id,
                    self._source_name(source_kind),
                    error,
                )
        return self._stage_source(
            version,
            source_kind=self._source_order[-1],
        )

    def _stage_source(
        self,
        version: WeightVersion,
        *,
        source_kind: WeightSource | None,
    ) -> SessionUpdate:
        logger.info(
            "ModelExpress active refit version=%s trying source=%s",
            version.version_id,
            self._source_name(source_kind),
        )
        return self._session.stage(version, source_kind=source_kind)

    @staticmethod
    def _source_name(source_kind: WeightSource | None) -> str:
        return source_kind.value if source_kind is not None else "configured_sources"


__all__ = ["RefitStageChain"]
