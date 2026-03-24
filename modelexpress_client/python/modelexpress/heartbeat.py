# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Client-side heartbeat for source liveness signaling.

The HeartbeatThread periodically checks NIXL agent health and calls
UpdateStatus(READY) to refresh the updated_at timestamp. The server-side
reaper uses this timestamp to detect dead sources.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import MxClient
    from .nixl_transfer import NixlTransferManager

logger = logging.getLogger("modelexpress.heartbeat")


class HeartbeatThread:
    """Background thread that signals source liveness via UpdateStatus RPCs.

    After PublishMetadata(status=INITIALIZING), the source spawns this thread.
    Each tick: if NIXL agent is healthy, send UpdateStatus(READY) to refresh
    updated_at. If not healthy, skip — the server-side reaper detects the
    stale updated_at and marks the worker STALE.

    On clean shutdown (SIGTERM), atexit handler sends UpdateStatus(STALE)
    for immediate detection without waiting for the reaper timeout.

    Args:
        mx_client: gRPC client for UpdateStatus calls.
        mx_source_id: Source identity hash returned by PublishMetadata.
        worker_id: Unique worker identifier.
        worker_rank: Global rank of this worker.
        nixl_manager: NIXL transfer manager for agent health checks.
    """

    def __init__(
        self,
        mx_client: MxClient,
        mx_source_id: str,
        worker_id: str,
        worker_rank: int,
        nixl_manager: NixlTransferManager,
        
    ):
        self._mx_client = mx_client
        self._mx_source_id = mx_source_id
        self._worker_id = worker_id
        self._worker_rank = worker_rank
        self._nixl_manager = nixl_manager

        self._interval = int(
            os.environ.get("MX_HEARTBEAT_INTERVAL_SECS", "30")
        )
        self._stop_event = threading.Event()
        self._started = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the heartbeat background thread."""
        self._thread = threading.Thread(
            target=self._run,
            name=f"mx-heartbeat-{self._worker_rank}",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self._on_exit)
        logger.info(
            f"[Worker {self._worker_rank}] Heartbeat started "
            f"(interval={self._interval}s)"
        )

    def stop(self) -> None:
        """Signal the heartbeat thread to stop and mark STALE."""
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
            pass

    def _update_status(self, status: int) -> None:
        """Send UpdateStatus RPC."""
        self._mx_client.update_status(
            mx_source_id=self._mx_source_id,
            worker_id=self._worker_id,
            worker_rank=self._worker_rank,
            status=status,
        )

    def _tick(self) -> None:
        """Single heartbeat tick: check health and send READY if healthy."""
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
