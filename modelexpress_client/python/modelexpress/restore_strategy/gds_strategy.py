# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GDS restore strategy for GMS snapshot file ranges."""

from __future__ import annotations

import logging
import sys

from ..gds_loader import MxGdsLoader
from ..gds_transfer import is_gds_available
from .base import RestoreStrategy, RestoreStrategyFailed
from .context import GmsRestoreContext

logger = logging.getLogger("modelexpress.restore_strategy_gds")


class GdsRestoreStrategy(RestoreStrategy):
    """Restore checkpoint extents directly into GMS VAs through GDS."""

    name = "gds"

    def is_available(self, ctx: GmsRestoreContext) -> bool:
        try:
            return bool(is_gds_available())
        except Exception:
            logger.warning("GDS capability check failed", exc_info=True)
            return False

    def restore(self, ctx: GmsRestoreContext) -> dict[str, object]:
        try:
            loader = MxGdsLoader(ctx.accelerator_backend)
        except Exception as exc:
            raise RestoreStrategyFailed(str(exc), mutated=True) from exc

        try:
            try:
                return loader.restore_gms_snapshot(
                    grouped_sources=ctx.grouped_sources,
                    device=ctx.device,
                    max_workers=ctx.max_workers,
                    chunk_size_bytes=ctx.gds_chunk_size,
                    max_inflight_batches=ctx.gds_max_inflight,
                )
            except Exception as exc:
                raise RestoreStrategyFailed(str(exc), mutated=True) from exc
        finally:
            failure_in_flight = sys.exc_info()[0] is not None
            try:
                loader.shutdown()
            except Exception as exc:
                if not failure_in_flight:
                    raise RestoreStrategyFailed(str(exc), mutated=True) from exc
                logger.warning("GDS loader shutdown failed", exc_info=True)
