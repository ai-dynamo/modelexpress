# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prioritized strategies for restoring GMS weight snapshots."""

from __future__ import annotations

import logging

from .base import RestoreStrategy, RestoreStrategyFailed
from .context import GmsRestoreContext

__all__ = [
    "GdsRestoreStrategy",
    "GmsRestoreContext",
    "MxGmsRestoreStrategyChain",
    "PosixRestoreStrategy",
    "RdmaRestoreStrategy",
    "RestoreStrategy",
    "RestoreStrategyFailed",
    "run_gms_restore",
]

logger = logging.getLogger("modelexpress.restore_strategy")

_STRATEGY_CHAIN = "rdma->gds->posix"


def _build_restore_strategies() -> list[RestoreStrategy]:
    from .gds_strategy import GdsRestoreStrategy
    from .posix_strategy import PosixRestoreStrategy
    from .rdma_strategy import RdmaRestoreStrategy

    return [RdmaRestoreStrategy(), GdsRestoreStrategy(), PosixRestoreStrategy()]


class MxGmsRestoreStrategyChain:
    """Run the fixed ``rdma -> gds -> posix`` GMS restore policy."""

    @staticmethod
    def run(ctx: GmsRestoreContext) -> dict[str, object]:
        failures: list[str] = []
        strategies = _build_restore_strategies()
        eligible = [strategy.name for strategy in strategies if strategy.is_available(ctx)]
        logger.info("Eligible GMS restore strategies: %s", eligible)

        for strategy in strategies:
            if strategy.name not in eligible:
                failures.append(f"{strategy.name}: unavailable")
                continue

            logger.info("Trying GMS restore strategy: %s", strategy.name)
            try:
                stats = strategy.restore(ctx)
            except RestoreStrategyFailed as exc:
                failures.append(
                    f"{strategy.name}: {exc} (mutated={exc.mutated})"
                )
                logger.warning(
                    "GMS restore strategy %s failed; trying durable fallback: %s",
                    strategy.name,
                    exc,
                )
                strategy.rollback(ctx)
                continue

            result = dict(stats)
            result["selected_strategy"] = strategy.name
            return result

        raise RuntimeError(
            "no MX restore strategy succeeded: "
            f"chain={_STRATEGY_CHAIN} failures={'; '.join(failures)}"
        )


def run_gms_restore(ctx: GmsRestoreContext) -> dict[str, object]:
    """Run the ModelExpress GMS restore policy."""
    return MxGmsRestoreStrategyChain.run(ctx)


def __getattr__(name: str):
    if name == "GdsRestoreStrategy":
        from .gds_strategy import GdsRestoreStrategy

        return GdsRestoreStrategy
    if name == "RdmaRestoreStrategy":
        from .rdma_strategy import RdmaRestoreStrategy

        return RdmaRestoreStrategy
    if name == "PosixRestoreStrategy":
        from .posix_strategy import PosixRestoreStrategy

        return PosixRestoreStrategy
    raise AttributeError(name)
