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

READY_POLL_SECS = 5

# How many heartbeats in a row the server must reject (success=false) before
# the worker assumes the server lost its registration and publishes again.
# Errors reaching the server do not count: publishing would fail too.
REREGISTER_AFTER_CONSECUTIVE_REJECTIONS = 2


class PublisherThread:
    """Background thread that publishes a source and keeps it READY.

    Callers may provide an already-known ``mx_source_id`` for pure heartbeat
    behavior, or a ``publish_fn`` that is retried on ticks until it returns an
    ``mx_source_id``. A ``ready_fn`` can gate publication until the engine has
    finished work that must happen before the source is advertised.

    On normal interpreter exit, an atexit handler sends UpdateStatus(STALE)
    for immediate detection without waiting for the reaper timeout. The default
    SIGTERM disposition does not run atexit handlers.

    If the server rejects a heartbeat (``success=false``), it has most likely
    lost this worker's registration (for example after its metadata store
    restarted empty). The server never notifies workers of that, so after
    ``REREGISTER_AFTER_CONSECUTIVE_REJECTIONS`` rejections in a row the thread
    drops its cached ``mx_source_id`` and publishes again via ``publish_fn``.
    Without a ``publish_fn`` it cannot recover; it logs one warning and keeps
    heartbeating in case the registration comes back.

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
        # True once we have demoted this worker for an unhealthy data plane, so the
        # demotion and its log line happen once rather than every interval.
        self._unhealthy = False
        # How many heartbeats in a row the server has rejected.
        self._heartbeat_rejections = 0
        # True once we warned that there is no publish_fn to re-register with,
        # so the warning is logged only once.
        self._reregister_unavailable_logged = False

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
        """Stop the publisher before marking it STALE."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5)
        self._mark_stale()
        self._cleanup()

    def _mark_stale(self) -> None:
        """Best-effort UpdateStatus(STALE). Swallows all errors."""
        with self._status_lock:
            if not self._started:
                return
            try:
                from .. import p2p_pb2
                if self._update_status(p2p_pb2.SOURCE_STATUS_STALE):
                    logger.info(
                        f"[Worker {self._worker_rank}] Marked STALE on shutdown"
                    )
                self._started = False
            except Exception:
                logger.debug(
                    f"[Worker {self._worker_rank}] Failed to mark STALE on shutdown",
                    exc_info=True,
                )
                self._started = False

    def _mark_unhealthy(self) -> None:
        """Best-effort UpdateStatus(STALE) for a worker whose data plane is down.

        Distinct from ``_mark_stale``, which is a shutdown path and clears
        ``_started`` permanently. Here the worker is still running and may recover,
        so ``_started`` is left alone: the next healthy tick sends READY again, and
        this method stays quiet after the first demotion so a persistently broken
        agent does not log once per interval forever.
        """
        with self._status_lock:
            if self._stop_event.is_set() or self._unhealthy:
                return
            if not self._started:
                # Never advertised READY, so no target can have selected us and
                # there is nothing to demote. Deliberately does not latch
                # ``_unhealthy``: if this worker later goes READY and then breaks,
                # that is the case worth demoting.
                return
            reason = getattr(self._nixl_manager, "data_plane_error", None)
            try:
                from .. import p2p_pb2
                self._update_status(p2p_pb2.SOURCE_STATUS_STALE)
                self._unhealthy = True
                logger.warning(
                    f"[Worker {self._worker_rank}] NIXL agent unhealthy, marked "
                    f"STALE so targets stop selecting it: {reason}"
                )
            except Exception:
                logger.debug(
                    f"[Worker {self._worker_rank}] Failed to mark STALE for "
                    f"unhealthy agent",
                    exc_info=True,
                )

    def _update_status(self, status: int) -> bool:
        """Send UpdateStatus RPC. Returns True when the server accepted it."""
        if self._mx_source_id is None:
            return False
        return self._mx_client.update_status(
            mx_source_id=self._mx_source_id,
            worker_id=self._worker_id,
            worker_rank=self._worker_rank,
            status=status,
        )

    def _on_heartbeat_rejected(self) -> None:
        """Handle a heartbeat the server rejected (``success=false``).

        Called with ``_status_lock`` held. The server has most likely lost
        this worker's registration, so heartbeats alone can never succeed
        again: after enough rejections in a row, drop the cached
        ``mx_source_id`` so the next tick publishes again.
        """
        self._heartbeat_rejections += 1
        if self._heartbeat_rejections < REREGISTER_AFTER_CONSECUTIVE_REJECTIONS:
            return
        self._heartbeat_rejections = 0

        if self._publish_fn is None:
            if not self._reregister_unavailable_logged:
                self._reregister_unavailable_logged = True
                logger.warning(
                    f"[Worker {self._worker_rank}] Server keeps rejecting "
                    f"heartbeats for mx_source_id={self._mx_source_id} and "
                    f"there is no publish_fn to re-register with; heartbeats "
                    f"continue in case the registration comes back"
                )
            return

        logger.warning(
            f"[Worker {self._worker_rank}] Server rejected "
            f"{REREGISTER_AFTER_CONSECUTIVE_REJECTIONS} heartbeats in a row "
            f"for mx_source_id={self._mx_source_id}; re-registering the source"
        )
        # Reset publish state so the next tick publishes again, with a fresh
        # publish timeout.
        self._mx_source_id = None
        self._publish_started_at = None
        self._publish_given_up = False

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
            # Demote rather than just skip the update. Skipping only stops
            # refreshing updated_at, so the entry keeps its last status and stays
            # selectable until the server's reaper times it out - 90s by default,
            # during which every target that picks this worker pays a full
            # transfer timeout. Saying STALE now removes it from the candidate set
            # immediately. It is not permanent: if the agent recovers, the next
            # tick puts it back to READY.
            self._mark_unhealthy()
            return

        with self._status_lock:
            if self._stop_event.is_set():
                return
            if not self._update_status(p2p_pb2.SOURCE_STATUS_READY):
                # The server answered but said no - it probably lost this
                # worker's registration. (Errors reaching the server raise
                # instead and are handled in _run.)
                self._on_heartbeat_rejected()
                return
            self._heartbeat_rejections = 0
            if self._unhealthy:
                # Recovered: allow a future failure to demote us again.
                self._unhealthy = False
                logger.info(
                    f"[Worker {self._worker_rank}] NIXL agent healthy again, "
                    f"status -> READY"
                )
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
            interval = self._interval
            if self._mx_source_id is None and self._ready_fn is not None:
                interval = min(interval, READY_POLL_SECS)
            self._stop_event.wait(timeout=interval)
