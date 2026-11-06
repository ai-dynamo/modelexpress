# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-local artifact publication gate for engine startup hooks."""

import logging
from collections.abc import Callable
from threading import Event

logger = logging.getLogger(__name__)


class StartupGate:
    """Release publishers only after every engine rank finishes startup."""

    def __init__(self) -> None:
        self.available = False
        self._ready = Event()

    def prepare(self) -> None:
        self._ready.clear()
        self.available = True

    def complete(self, synchronize: Callable[[], None]) -> None:
        try:
            synchronize()
        except Exception:
            logger.warning(
                "Engine startup synchronization failed; keeping artifact "
                "publication gated",
                exc_info=True,
            )
            return
        self._ready.set()

    def is_ready(self) -> bool:
        return self._ready.is_set()
