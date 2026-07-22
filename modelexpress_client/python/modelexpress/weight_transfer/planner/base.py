# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract planner interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..protocol.types import RdmaDescriptor, ResolvedRegion, TrainerTable


class AbstractPlanner(ABC):
    """Converts resolved element-run regions into NIXL RDMA descriptors.

    The routing step (mapping element offsets to trainer shard GPU addresses)
    can run locally (LocalPlanner) or be offloaded to the MX server (ServerPlanner).
    """

    @abstractmethod
    def build(
        self,
        regions: list[ResolvedRegion],
        table: TrainerTable,
        plan_key: str,
    ) -> list[RdmaDescriptor]:
        """Build RDMA descriptors from resolved regions."""
