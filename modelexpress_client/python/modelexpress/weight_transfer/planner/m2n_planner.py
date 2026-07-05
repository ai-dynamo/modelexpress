# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""M2nPlanner: globally-coordinated M2N plan built by the MX server.

All inference workers register their resolved regions with the MX server.
The server holds a barrier until total_workers have registered, then routes
all workers' regions in one pass and stores per-worker descriptor slices.
Workers poll GetM2nPlan until the plan is ready.

Protocol:
  RegisterM2nWorker(model_key, worker_rank, total_workers, regions, nixl_metadata)
  GetM2nPlan(m2n_plan_id, worker_rank)
  InvalidateM2nPlan(model_key)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from .base import AbstractPlanner
from .local import LocalPlanner
from ..protocol.types import M2nDescriptor, RdmaDescriptor, ResolvedRegion, TrainerTable

if TYPE_CHECKING:
    pass

logger = logging.getLogger("modelexpress.weight_transfer.m2n_planner")


class M2nPlanner(AbstractPlanner):
    """Coordinate plan building across all workers via the MX server."""

    def __init__(
        self,
        mx_client: object,
        model_key: str,
        worker_rank: int,
        total_workers: int,
        nixl_metadata: bytes,
        fallback: LocalPlanner | None = None,
        timeout: float = 120.0,
        poll_interval: float = 0.1,
    ) -> None:
        self._client = mx_client
        self._model_key = model_key
        self._worker_rank = worker_rank
        self._total_workers = total_workers
        self._nixl_metadata = nixl_metadata
        self._fallback = fallback or LocalPlanner()
        self._timeout = timeout
        self._poll_interval = poll_interval

        self._local_cache: dict[str, list[RdmaDescriptor]] = {}
        self._m2n_cache: dict[str, list[M2nDescriptor]] = {}

    def build(
        self,
        regions: list[ResolvedRegion],
        table: TrainerTable,
        plan_key: str,
    ) -> list[RdmaDescriptor]:
        """Build RDMA descriptors for this worker's PULL path via M2N coordination."""
        if plan_key in self._local_cache:
            return self._local_cache[plan_key]

        try:
            m2n_descs = self.build_m2n(regions, table, plan_key)
        except Exception as e:
            logger.warning(
                "M2nPlanner: server coordination failed (%s), falling back to LocalPlanner",
                e,
            )
            descriptors = self._fallback.build(regions, table, plan_key)
            self._local_cache[plan_key] = descriptors
            return descriptors

        descriptors = [d.to_rdma_descriptor() for d in m2n_descs]
        self._local_cache[plan_key] = descriptors
        return descriptors

    def build_m2n(
        self,
        regions: list[ResolvedRegion],
        table: TrainerTable,
        plan_key: str,
    ) -> list[M2nDescriptor]:
        """Register with the MX server and poll until the global plan is ready."""
        if plan_key in self._m2n_cache:
            return self._m2n_cache[plan_key]

        m2n_descs = self._coordinate_via_server(regions, plan_key)
        self._m2n_cache[plan_key] = m2n_descs
        total_bytes = sum(d.nbytes for d in m2n_descs)
        logger.info(
            "M2nPlanner: worker %d plan ready: %d descriptors, %.2f GB",
            self._worker_rank,
            len(m2n_descs),
            total_bytes / 1e9,
        )
        return m2n_descs

    def invalidate(self, plan_key: str) -> None:
        """Evict caches and invalidate the server-side M2N plan."""
        self._local_cache.pop(plan_key, None)
        self._m2n_cache.pop(plan_key, None)
        self._fallback.invalidate(plan_key)
        try:
            self._client.invalidate_m2n_plan(model_key=self._model_key)
        except Exception as e:
            logger.warning("M2nPlanner: failed to invalidate server plan: %s", e)

    def _coordinate_via_server(
        self,
        regions: list[ResolvedRegion],
        plan_key: str,
    ) -> list[M2nDescriptor]:
        """Register with the server and poll until the global plan is ready."""
        reg_resp = self._client.register_m2n_worker(
            model_key=self._model_key,
            worker_rank=self._worker_rank,
            total_workers=self._total_workers,
            regions=regions,
            nixl_metadata=self._nixl_metadata,
        )
        m2n_plan_id: str = reg_resp.m2n_plan_id

        logger.info(
            "M2nPlanner: worker %d registered (plan_id=%r, total_workers=%d)",
            self._worker_rank,
            m2n_plan_id,
            self._total_workers,
        )

        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            if m2n_plan_id:
                get_resp = self._client.get_m2n_plan(
                    m2n_plan_id=m2n_plan_id,
                    worker_rank=self._worker_rank,
                )
                if get_resp.ready:
                    return _decode_m2n_proto_descriptors(get_resp.descriptors)
            else:
                # Barrier not yet satisfied; re-register to get the plan_id
                # once the last worker fires.
                reg_resp = self._client.register_m2n_worker(
                    model_key=self._model_key,
                    worker_rank=self._worker_rank,
                    total_workers=self._total_workers,
                    regions=regions,
                    nixl_metadata=self._nixl_metadata,
                )
                m2n_plan_id = reg_resp.m2n_plan_id

            time.sleep(self._poll_interval)

        raise TimeoutError(
            f"M2nPlanner: worker {self._worker_rank} did not receive a plan "
            f"within {self._timeout}s (plan_id={m2n_plan_id!r})"
        )


def _decode_m2n_proto_descriptors(protos: list) -> list[M2nDescriptor]:
    return [
        M2nDescriptor(
            src_agent_index=p.src_agent_index,
            dst_agent_index=p.dst_agent_index,
            src_addr=p.src_addr,
            dst_addr=p.dst_addr,
            nbytes=p.nbytes,
        )
        for p in protos
    ]
