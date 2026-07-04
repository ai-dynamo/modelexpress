# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base contract for GMS snapshot restore strategies."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import GmsRestoreContext

logger = logging.getLogger("modelexpress.restore_strategy")


class RestoreStrategyFailed(Exception):
    """An expected strategy failure that permits durable fallback."""

    def __init__(self, message: str, *, mutated: bool = False) -> None:
        super().__init__(message)
        self.mutated = mutated


class RestoreStrategy(ABC):
    """One implementation in the ordered GMS restore policy."""

    name: str

    def is_available(self, ctx: GmsRestoreContext) -> bool:
        """Return whether this strategy can run in the current process."""
        return True

    @abstractmethod
    def restore(self, ctx: GmsRestoreContext) -> dict[str, object]:
        """Restore every source into its allocation-matched target."""

    def rollback(self, ctx: GmsRestoreContext) -> None:
        """Release strategy-owned resources after a failed attempt."""
        return None
