# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract planner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..protocol.types import RdmaDescriptor, ResolvedRegion, TrainerTable


class AbstractPlanner(ABC):
    """Converts resolved element-run regions into NIXL RDMA descriptors.

    The bake+resolve step (driving the weight loader with LazyWeights and
    replaying op chains on meta tensors) always runs on the client because
    it requires torch.  The routing step (mapping element offsets to trainer
    shard GPU addresses) can run locally OR be offloaded to the MX server.

    LocalPlanner does both steps on the client.
    ServerPlanner sends ResolvedRegions to the MX WeightSyncService, which
    routes them in Rust and caches the plan for all workers sharing the model.
    """

    @abstractmethod
    def build(
        self,
        regions: list[ResolvedRegion],
        table: TrainerTable,
        plan_key: str,
    ) -> list[RdmaDescriptor]:
        """Build RDMA descriptors from resolved regions.

        Args:
            regions: Element-run pairs already resolved from op chains.
            table: Trainer shard layout (needed by LocalPlanner; ignored by
                ServerPlanner which reads it from the server cache).
            plan_key: Stable identifier for this (model, worker) combination.
                ServerPlanner uses it to retrieve a cached plan on subsequent
                calls so repeated syncs skip the routing work entirely.

        Returns:
            Flat list of RdmaDescriptor ready for NIXL execution.
        """
