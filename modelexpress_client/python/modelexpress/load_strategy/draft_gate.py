# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publication gate shared by engine loaders that run a speculative draft pass."""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger("modelexpress.load_strategy.draft_gate")

# Upper bound on how long a main load's publication waits for its draft pass.
# Engines load the drafter right after the target, so this only matters when
# the expected draft never comes through the ModelExpress loader (e.g. a
# draft-specific load format); the target must still publish.
DRAFT_PUBLICATION_GRACE_SECS = 600.0


class DraftPublicationGate:
    """Keeps a main load undiscoverable until its draft has joined the manifest.

    Opens when the draft pass on the same device finishes, or after
    ``grace_secs`` from the end of the main pass.
    """

    def __init__(self, grace_secs: float = DRAFT_PUBLICATION_GRACE_SECS):
        self._released = threading.Event()
        self._grace_secs = grace_secs
        self._deadline: float | None = None
        self._expired_logged = False

    def arm(self) -> None:
        self._deadline = time.monotonic() + self._grace_secs

    def release(self) -> None:
        self._released.set()

    def is_open(self) -> bool:
        if self._released.is_set():
            return True
        if self._deadline is None or time.monotonic() < self._deadline:
            return False
        if not self._expired_logged:
            self._expired_logged = True
            logger.warning(
                "Speculative draft pass did not reach the ModelExpress loader "
                "within %.0fs of the target load; publishing the target without "
                "draft tensors",
                self._grace_secs,
            )
        return True
