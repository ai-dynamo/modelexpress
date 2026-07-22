# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract base for sync roles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..protocol.types import TrainerTable, InferenceTable


class WeightSyncRole(ABC):
    """One side of a trainer-inference weight sync operation.

    PullRole -- inference worker; pulls weights from trainer via NIXL RDMA READ.
    PushRole  -- trainer; pushes weights to inference workers via NIXL RDMA WRITE.
    """

    @abstractmethod
    def initialize(self, model: Any, table: Any) -> None:
        """Bake the plan and set up NIXL agents (called once at startup)."""

    @abstractmethod
    def sync(self) -> None:
        """Execute one weight sync step using the pre-built plan."""

    @abstractmethod
    def teardown(self) -> None:
        """Release NIXL resources."""
