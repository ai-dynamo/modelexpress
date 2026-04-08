# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Client-side heartbeat and metadata publisher.

The HeartbeatThread manages the full source lifecycle:
1. Publish phase: repeatedly attempts to publish metadata via a caller-
   supplied callback, retrying each tick until success or a configurable
   timeout (default 30 min).
2. Heartbeat phase: once published, periodically checks NIXL agent health
   and calls UpdateStatus(READY) to refresh the updated_at timestamp.
   The server-side reaper uses this timestamp to detect dead sources.
3. Shutdown: atexit handler sends UpdateStatus(STALE) for immediate
   detection without waiting for the reaper timeout.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .client import MxClient
    from .nixl_transfer import NixlTransferManager

logger = logging.getLogger("modelexpress.heartbeat")

PUBLISH_TIMEOUT_SECS_DEFAULT = 30 * 60  # 30 minutes


class HeartbeatThread:
    """Background thread that publishes metadata then signals source liveness.

    The thread starts in *publish* mode: each tick calls ``publish_fn`` to
    attempt metadata registration.  On success ``publish_fn`` returns the
    ``mx_source_id`` and the thread transitions to *heartbeat* mode where it
    sends periodic ``UpdateStatus(READY)`` RPCs.

    If publishing does not succeed within ``publish_timeout_secs`` the thread
    logs a warning and stops retrying, but keeps running so that a future
    manual publish could still transition it.

    Args:
        mx_client: gRPC client for UpdateStatus calls.
        worker_id: Unique worker identifier.
        worker_rank: Global rank of this worker.
        nixl_manager: NIXL transfer manager for agent health checks.
        publish_fn: Callable that performs the metadata publish and returns
            the ``mx_source_id`` on success.  Raising signals a transient
            failure that will be retried on the next tick.
        publish_timeout_secs: Seconds to keep retrying publish before giving
            up.  Defaults to ``MX_PUBLISH_TIMEOUT_SECS`` env var or 30 min.
    """

    def __init__(
        self,
        mx_client: MxClient,
        worker_id: str,
        worker_rank: int,
        nixl_manager: NixlTransferManager,
        publish_fn: Callable[[], str],
        publish_timeout_secs: int | None = None,
    ):
        self._mx_client = mx_client
        self._worker_id = worker_id
        self._worker_rank = worker_rank
        self._nixl_manager = nixl_manager
        self._publish_fn = publish_fn

        if publish_timeout_secs is not None:
            self._publish_timeout = publish_timeout_secs
        else:
            self._publish_timeout = int(
                os.environ.get(
                    "MX_PUBLISH_TIMEOUT_SECS",
                    str(PUBLISH_TIMEOUT_SECS_DEFAULT),
                )
            )

        self._mx_source_id: str | None = None
        self._publish_started_at: float | None = None
        self._publish_given_up = False

        self._interval = int(
            os.environ.get("MX_HEARTBEAT_INTERVAL_SECS", "30")
        )
        self._stop_event = threading.Event()
        self._started = False
        self._thread: threading.Thread | None = None

    @property
    def mx_source_id(self) -> str | None:
        return self._mx_source_id

    def start(self) -> None:
        """Start the background thread."""
        self._thread = threading.Thread(
            target=self._run,
            name=f"mx-heartbeat-{self._worker_rank}",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self._on_exit)
        logger.info(
            f"[Worker {self._worker_rank}] Heartbeat started "
            f"(interval={self._interval}s, "
            f"publish_timeout={self._publish_timeout}s)"
        )

    def stop(self) -> None:
        """Signal the thread to stop and mark STALE."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5)
        self._mark_stale()
        logger.info(f"[Worker {self._worker_rank}] Heartbeat stopped")

    def _on_exit(self) -> None:
        """atexit handler: mark STALE on clean shutdown (SIGTERM)."""
        self._stop_event.set()
        self._mark_stale()

    def _mark_stale(self) -> None:
        """Best-effort UpdateStatus(STALE). Swallows all errors."""
        if not self._started:
            return
        try:
            from . import p2p_pb2
            self._update_status(p2p_pb2.SOURCE_STATUS_STALE)
            logger.info(f"[Worker {self._worker_rank}] Marked STALE on shutdown")
            self._started = False
        except Exception:
            logger.warning(
                f"[Worker {self._worker_rank}] Failed to mark STALE on shutdown",
                exc_info=True,
            )
            self._started = False

    def _update_status(self, status: int) -> None:
        """Send UpdateStatus RPC."""
        self._mx_client.update_status(
            mx_source_id=self._mx_source_id,
            worker_id=self._worker_id,
            worker_rank=self._worker_rank,
            status=status,
        )

    def _try_publish(self) -> None:
        """Attempt metadata publish. On success, store mx_source_id."""
        if self._publish_started_at is None:
            self._publish_started_at = time.monotonic()

        elapsed = time.monotonic() - self._publish_started_at
        if elapsed > self._publish_timeout:
            if not self._publish_given_up:
                logger.warning(
                    f"[Worker {self._worker_rank}] Giving up on metadata publish "
                    f"after {elapsed:.0f}s (timeout={self._publish_timeout}s). "
                    f"Worker will continue without P2P serving."
                )
                self._publish_given_up = True
            return

        try:
            self._mx_source_id = self._publish_fn()
            self._started = True
            logger.info(
                f"[Worker {self._worker_rank}] Metadata published successfully "
                f"(mx_source_id={self._mx_source_id})"
            )
        except Exception as e:
            logger.warning(
                f"[Worker {self._worker_rank}] Metadata publish attempt failed "
                f"({elapsed:.0f}s elapsed, timeout={self._publish_timeout}s), "
                f"will retry next tick: {e}"
            )

    def _tick(self) -> None:
        """Single tick: publish if needed, otherwise heartbeat."""
        if self._mx_source_id is None:
            self._try_publish()
            return

        from . import p2p_pb2

        if not self._nixl_manager.is_healthy():
            return

        self._update_status(p2p_pb2.SOURCE_STATUS_READY)
        if not self._started:
            logger.info(
                f"[Worker {self._worker_rank}] Status -> READY"
            )
            self._started = True

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception(
                    f"[Worker {self._worker_rank}] Heartbeat tick failed"
                )
            self._stop_event.wait(timeout=self._interval)
