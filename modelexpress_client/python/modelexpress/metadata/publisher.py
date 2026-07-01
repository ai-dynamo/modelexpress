# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client-side source publication and heartbeat signaling."""

from __future__ import annotations

import atexit
import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from .. import envs

if TYPE_CHECKING:
    from ..client import MxClient
    from ..nixl_transfer import NixlTransferManager

logger = logging.getLogger("modelexpress.metadata.publisher")

PUBLISH_TIMEOUT_SECS_DEFAULT = 30 * 60


class PublisherThread:
    """Background thread that publishes a source and keeps it READY.

    Callers may provide an already-known ``mx_source_id`` for pure heartbeat
    behavior, or a ``publish_fn`` that is retried on ticks until it returns an
    ``mx_source_id``. A ``ready_fn`` can gate publication until the engine has
    finished work that must happen before the source is advertised.

    On clean shutdown (SIGTERM), atexit handler sends UpdateStatus(STALE)
    for immediate detection without waiting for the reaper timeout.

    Args:
        mx_client: gRPC client for UpdateStatus calls.
        mx_source_id: Source identity hash returned by PublishMetadata.
        worker_id: Unique worker identifier.
        worker_rank: Model-shard rank used for metadata/status keying.
        nixl_manager: Optional NIXL transfer manager for agent health checks.
            Non-NIXL transports pass None and heartbeat unconditionally.
        publish_fn: Optional callback that publishes the source and returns
            its mx_source_id.
        ready_fn: Optional callback that must return True before publish_fn
            is called.
        cleanup_fn: Optional best-effort callback invoked on stop/exit after
            stale marking.
        publish_timeout_secs: Seconds to keep waiting/publishing before giving
            up and stopping the thread.
        interval_secs: Optional tick interval override. Defaults to
            ``MX_HEARTBEAT_INTERVAL_SECS``.
        heartbeat_after_publish: If False, the thread exits after publish_fn
            succeeds instead of sending READY heartbeats.
    """

    def __init__(
        self,
        mx_client: MxClient,
        mx_source_id: str | None = None,
        worker_id: str | None = None,
        worker_rank: int | None = None,
        nixl_manager: NixlTransferManager | None = None,
        publish_fn: Callable[[], str] | None = None,
        ready_fn: Callable[[], bool] | None = None,
        cleanup_fn: Callable[[], None] | None = None,
        publish_timeout_secs: int | None = None,
        interval_secs: int | None = None,
        heartbeat_after_publish: bool = True,
    ):
        if mx_source_id is None and publish_fn is None:
            raise ValueError("PublisherThread requires mx_source_id or publish_fn")
        if worker_id is None or worker_rank is None:
            raise ValueError("PublisherThread requires worker_id and worker_rank")

        self._mx_client = mx_client
        self._mx_source_id = mx_source_id
        self._worker_id = worker_id
        self._worker_rank = worker_rank
        self._nixl_manager = nixl_manager
        self._publish_fn = publish_fn
        self._ready_fn = ready_fn
        self._cleanup_fn = cleanup_fn
        self._heartbeat_after_publish = heartbeat_after_publish

        self._publish_timeout = (
            publish_timeout_secs
            if publish_timeout_secs is not None
            else envs.MX_PUBLISH_TIMEOUT_SECS
        )
        self._publish_started_at: float | None = None
        self._publish_given_up = False
        self._cleaned_up = False

        self._interval = (
            interval_secs
            if interval_secs is not None
            else envs.MX_HEARTBEAT_INTERVAL_SECS
        )
        self._stop_event = threading.Event()
        self._status_lock = threading.Lock()
        self._started = False
        self._thread: threading.Thread | None = None

    @property
    def mx_source_id(self) -> str | None:
        return self._mx_source_id

    @property
    def _one_shot_publisher(self) -> bool:
        return self._publish_fn is not None and not self._heartbeat_after_publish

    def start(self) -> None:
        """Start the publisher background thread."""
        self._thread = threading.Thread(
            target=self._run,
            name=f"mx-publisher-{self._worker_rank}",
            daemon=True,
        )
        self._thread.start()
        atexit.register(self._on_exit)
        log = logger.debug if self._one_shot_publisher else logger.info
        log(
            f"[Worker {self._worker_rank}] Publisher thread started "
            f"(interval={self._interval}s, "
            f"publish_timeout={self._publish_timeout}s)"
        )

    def stop(self) -> None:
        """Signal the publisher thread to stop, mark STALE, and clean up."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5)
        self._mark_stale()
        self._cleanup()
        logger.info(f"[Worker {self._worker_rank}] Publisher thread stopped")

    def _on_exit(self) -> None:
        """atexit handler: mark STALE on clean shutdown (SIGTERM)."""
        self._stop_event.set()
        self._mark_stale()
        self._cleanup()

    def _mark_stale(self) -> None:
        """Best-effort UpdateStatus(STALE). Swallows all errors."""
        with self._status_lock:
            if not self._started:
                return
            try:
                from .. import p2p_pb2
                self._update_status(p2p_pb2.SOURCE_STATUS_STALE)
                logger.info(f"[Worker {self._worker_rank}] Marked STALE on shutdown")
                self._started = False
            except Exception:
                logger.debug(
                    f"[Worker {self._worker_rank}] Failed to mark STALE on shutdown",
                    exc_info=True,
                )
                self._started = False

    def _update_status(self, status: int) -> None:
        """Send UpdateStatus RPC."""
        if self._mx_source_id is None:
            return
        self._mx_client.update_status(
            mx_source_id=self._mx_source_id,
            worker_id=self._worker_id,
            worker_rank=self._worker_rank,
            status=status,
        )

    def _cleanup(self) -> None:
        if self._cleanup_fn is None or self._cleaned_up:
            return
        self._cleaned_up = True
        try:
            self._cleanup_fn()
        except Exception:
            logger.debug(
                f"[Worker {self._worker_rank}] Publisher cleanup failed",
                exc_info=True,
            )

    def _publish_elapsed(self) -> float:
        if self._publish_started_at is None:
            self._publish_started_at = time.monotonic()
        return time.monotonic() - self._publish_started_at

    def _publish_timed_out(self, elapsed: float) -> bool:
        if elapsed <= self._publish_timeout:
            return False
        if not self._publish_given_up:
            logger.warning(
                f"[Worker {self._worker_rank}] Giving up on source publish "
                f"after {elapsed:.0f}s (timeout={self._publish_timeout}s). "
                f"Worker will continue without this source."
            )
            self._publish_given_up = True
        self._stop_event.set()
        self._cleanup()
        return True

    def _try_publish(self) -> bool:
        """Attempt readiness-gated publication."""
        if self._stop_event.is_set():
            return False
        if self._publish_fn is None:
            return self._mx_source_id is not None

        elapsed = self._publish_elapsed()
        if self._publish_timed_out(elapsed):
            return False

        if self._ready_fn is not None:
            try:
                if not self._ready_fn():
                    return False
            except Exception as exc:
                logger.warning(
                    f"[Worker {self._worker_rank}] Source readiness check "
                    f"failed ({elapsed:.0f}s elapsed, "
                    f"timeout={self._publish_timeout}s), will retry: {exc}"
                )
                return False

        if self._stop_event.is_set():
            return False
        try:
            self._mx_source_id = self._publish_fn()
            log = logger.debug if self._one_shot_publisher else logger.info
            log(
                f"[Worker {self._worker_rank}] Source published successfully "
                f"(mx_source_id={self._mx_source_id})"
            )
            return True
        except Exception as exc:
            logger.warning(
                f"[Worker {self._worker_rank}] Source publish attempt failed "
                f"({elapsed:.0f}s elapsed, timeout={self._publish_timeout}s), "
                f"will retry next tick: {exc}"
            )
            return False

    def _tick(self) -> None:
        """Single tick: publish if needed, then send READY if healthy."""
        from .. import p2p_pb2

        if self._stop_event.is_set():
            return
        if self._mx_source_id is None:
            if not self._try_publish():
                return
            if not self._heartbeat_after_publish:
                self._stop_event.set()
                return

        if self._nixl_manager is not None and not self._nixl_manager.is_healthy():
            return

        with self._status_lock:
            if self._stop_event.is_set():
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
                    f"[Worker {self._worker_rank}] Publisher tick failed"
                )
            self._stop_event.wait(timeout=self._interval)
