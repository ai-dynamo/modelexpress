# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LocalPlanner: resolve op chains and route regions entirely on the client.

Use this when no MX server is available or for offline testing.  Each worker
builds its own plan independently; there is no caching across workers.
"""

from __future__ import annotations

import logging

from .base import AbstractPlanner
from .router import route_regions
from ..protocol.types import RdmaDescriptor, ResolvedRegion, TrainerTable

logger = logging.getLogger("modelexpress.weight_transfer.local_planner")


class LocalPlanner(AbstractPlanner):
    """Build and cache the RDMA plan entirely on the client."""

    def __init__(self) -> None:
        self._cache: dict[str, list[RdmaDescriptor]] = {}

    def build(
        self,
        regions: list[ResolvedRegion],
        table: TrainerTable,
        plan_key: str,
    ) -> list[RdmaDescriptor]:
        if plan_key in self._cache:
            logger.debug("LocalPlanner: returning cached plan for %s", plan_key)
            return self._cache[plan_key]

        descriptors = route_regions(regions, table)
        self._cache[plan_key] = descriptors
        total_bytes = sum(d.nbytes for d in descriptors)
        logger.info(
            "LocalPlanner: built plan %s: %d descriptors, %.2f GB",
            plan_key,
            len(descriptors),
            total_bytes / 1e9,
        )
        return descriptors

    def invalidate(self, plan_key: str) -> None:
        """Remove a cached plan (call when trainer reshards)."""
        self._cache.pop(plan_key, None)
