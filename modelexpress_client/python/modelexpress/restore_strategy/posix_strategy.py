# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""POSIX staging restore strategy."""

from __future__ import annotations

import logging

from .base import RestoreStrategy
from .context import GmsRestoreContext

logger = logging.getLogger("modelexpress.restore_strategy_posix")


class PosixRestoreStrategy(RestoreStrategy):
    """Phase-4 placeholder for ModelExpress POSIX staging restore."""

    name = "posix"

    def restore(self, ctx: GmsRestoreContext) -> dict[str, object]:
        raise NotImplementedError(
            "MX POSIX restore is not implemented yet; "
            f"device={ctx.device} gds_chunk_size={ctx.gds_chunk_size}"
        )
