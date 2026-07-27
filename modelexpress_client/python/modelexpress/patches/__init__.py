# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight runtime compatibility patch support."""

from collections.abc import Callable, Iterable
import logging

from .. import envs

logger = logging.getLogger(__name__)

Patch = Callable[[], bool]


def apply_patches(patches: Iterable[Patch]) -> None:
    """Apply patches in order, logging only patches that changed the runtime."""
    if envs.MX_DISABLE_PATCHES:
        logger.info("Compatibility patches disabled by MX_DISABLE_PATCHES")
        return

    for patch in patches:
        if patch():
            logger.info("Applied compatibility patch: %s", patch.__name__)


__all__ = ["Patch", "apply_patches"]
