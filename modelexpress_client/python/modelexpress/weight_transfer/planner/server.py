# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ServerPlanner: offload region routing to the MX WeightSyncService.

The client resolves op chains locally (torch-dependent) then sends the
resulting ResolvedRegions to the MX server.  The server performs the routing
math in Rust and caches the plan by plan_key so all workers sharing the same
model fetch the cached plan without duplicate routing work.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from .base import AbstractPlanner
from .local import LocalPlanner
from ..protocol.serialization import (
    decode_rdma_descriptors,
    encode_resolved_regions,
)
from ..protocol.types import RdmaDescriptor, ResolvedRegion, TrainerTable

logger = logging.getLogger("modelexpress.weight_transfer.server_planner")


class ServerPlanner(AbstractPlanner):
    """Offload region routing to the MX server; fall back to LocalPlanner."""

    def __init__(
        self,
        mx_client: object,
        fallback: LocalPlanner | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._client = mx_client
        self._fallback = fallback or LocalPlanner()
        self._timeout = timeout
        # Local cache avoids a round-trip after the plan is built once
        self._local_cache: dict[str, list[RdmaDescriptor]] = {}

    def build(
        self,
        regions: list[ResolvedRegion],
        table: TrainerTable,
        plan_key: str,
    ) -> list[RdmaDescriptor]:
        if plan_key in self._local_cache:
            return self._local_cache[plan_key]

        try:
            descriptors = self._build_via_server(regions, plan_key)
        except Exception as e:
            logger.warning(
                "ServerPlanner: server build failed (%s), falling back to LocalPlanner",
                e,
            )
            descriptors = self._fallback.build(regions, table, plan_key)

        self._local_cache[plan_key] = descriptors
        return descriptors

    def invalidate(self, plan_key: str) -> None:
        """Evict the local cache entry so the next build triggers a fresh server round-trip."""
        self._local_cache.pop(plan_key, None)
        self._fallback.invalidate(plan_key)

    def _build_via_server(
        self,
        regions: list[ResolvedRegion],
        plan_key: str,
    ) -> list[RdmaDescriptor]:
        """Submit regions to server, poll for the built plan, return descriptors."""
        encoded = encode_resolved_regions(regions)

        build_resp = self._client.build_plan(
            plan_key=plan_key,
            regions_payload=encoded,
        )
        plan_id = build_resp.plan_id

        logger.info(
            "ServerPlanner: submitted %d regions for plan %s (id=%s)",
            len(regions),
            plan_key,
            plan_id,
        )

        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            get_resp = self._client.get_plan(plan_id=plan_id)
            if get_resp.ready:
                descs = decode_rdma_descriptors(get_resp.descriptors_payload)
                logger.info(
                    "ServerPlanner: received plan %s: %d descriptors",
                    plan_id,
                    len(descs),
                )
                return descs
            time.sleep(0.1)

        raise TimeoutError(
            f"Server did not build plan {plan_id!r} within {self._timeout}s"
        )
