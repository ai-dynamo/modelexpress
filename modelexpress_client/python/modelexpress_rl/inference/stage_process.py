# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a generator client's staging step in a child process.

``StageProcess`` keeps one worker, ``modelexpress_rl.inference.stage_worker``,
and passes it stage requests. The worker is a second generator client, built
from the same configuration but without an engine. It runs the client's whole
staging step, MX server calls included, and returns the server's answers so
that the client needs to ask for nothing. Nothing here knows how a version is
staged.

Only a worker stages. A worker that exits is replaced at once, by a thread that
waits for its exit. A request that still finds no worker starts one, and a
request whose worker is gone before it replies starts another and is sent again.

The two processes exchange Python objects over one socket pair:

    client -> worker   {"op": "init", "config": ModelExpressGeneratorConfig}
    worker -> client   {"event": "ready", "pid", "package"}
                       {"event": "fatal", "error"}
    client -> worker   {"op": "stage", "version_id", "serving_version_id"}
    worker -> client   {"ok": True, "versions", "metrics", "seconds"}
                       {"ok": False, "error"}
    client -> worker   {"op": "release"}    drop what the last stage holds
    client -> worker   {"op": "shutdown"}   or end of file

Either side sees end of file once the other process is gone.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from .. import envs as rl_envs

if TYPE_CHECKING:
    from .client import ModelExpressGeneratorConfig

logger = logging.getLogger("modelexpress_rl.inference.stage_process")

_T = TypeVar("_T")

_WORKER_MODULE = "modelexpress_rl.inference.stage_worker"
# Where this process imported modelexpress_rl from. The worker must run the
# same code.
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# How often a wait re-checks for close(), an exited worker and its deadline.
_TICK_SECONDS = 0.1
# Bounds the wait for a worker that closed its connection without exiting.
_REAP_SECONDS = 5.0
# How many workers one call may try before it gives up: each attempt starts a
# worker if there is none and, for a stage request, sends the request. Also how
# many workers in a row are replaced on exit with no reply to a request between.
_ATTEMPTS = 3


class StageProcessError(RuntimeError):
    """A version was not staged, or no worker started; the message says why."""


class _WorkerGone(Exception):
    """There is no worker any more; a new one may do better."""


class _DeadlineExceeded(Exception):
    pass


class _Closed(Exception):
    pass


class StageProcess:
    """A stage worker, replaced when it is gone, and this process's end of the
    connection to it.

    Requests are serialized. ``close`` may be called from any thread while a
    request waits; that request then fails within one tick.
    """

    def __init__(
        self,
        config: ModelExpressGeneratorConfig,
        *,
        timeout_seconds: float | None = None,
        start_timeout_seconds: float | None = None,
        close_grace_seconds: float | None = None,
    ) -> None:
        """Start a worker and wait until it is ready.

        ``config`` is the worker's own generator client configuration. The
        timeouts default to the MX_REFIT_STAGE_PROCESS_* settings. Raise
        StageProcessError if no worker starts.
        """
        if timeout_seconds is None:
            timeout_seconds = rl_envs.MX_REFIT_STAGE_PROCESS_TIMEOUT_SECONDS
        if start_timeout_seconds is None:
            start_timeout_seconds = rl_envs.MX_REFIT_STAGE_PROCESS_START_TIMEOUT_SECONDS
        if close_grace_seconds is None:
            close_grace_seconds = rl_envs.MX_REFIT_STAGE_PROCESS_CLOSE_GRACE_SECONDS
        self._config = config
        self._timeout_seconds = timeout_seconds
        self._start_timeout_seconds = start_timeout_seconds
        self._close_grace_seconds = close_grace_seconds
        self._lock = threading.Lock()
        self._closed = False
        self._proc: subprocess.Popen | None = None
        self._conn: Any = None
        # Workers replaced on exit since the last reply to a stage request.
        self._replaced = 0
        with self._lock:
            self._with_worker("starting the stage process", lambda: None)

    @property
    def pid(self) -> int | None:
        proc = self._proc
        return None if proc is None else proc.pid

    def stage(
        self, version_id: str, *, serving_version_id: str | None
    ) -> tuple[dict[str, Any], dict[str, float]]:
        """Have a worker run the staging step for ``version_id``.

        Return the version records the worker read from the MX server, by id,
        and what its staging step measured. The worker holds the staged
        version's leases until ``release``.

        ``serving_version_id`` is the version the caller's engine serves. A
        worker that is gone is replaced and the request sent again. Raise
        StageProcessError when the staging step failed or ran out of time, in
        which case it is not tried again, and when every attempt lost its
        worker.
        """

        def request() -> dict[str, Any]:
            try:
                self._conn.send(
                    {
                        "op": "stage",
                        "version_id": version_id,
                        "serving_version_id": serving_version_id,
                    }
                )
                return self._receive(time.monotonic() + self._timeout_seconds)
            except _DeadlineExceeded:
                # More time is unlikely to help: what it waits on is stuck.
                self._drop(kill=True)
                raise StageProcessError(
                    f"stage process did not stage version {version_id!r} within "
                    f"{self._timeout_seconds:g} s and was killed"
                ) from None
            except (_WorkerGone, OSError, ValueError):
                code = self._drop(kill=False)
                raise _WorkerGone(f"it exited with code {code}") from None

        with self._lock:
            try:
                reply, started = self._with_worker(
                    f"staging version {version_id!r}", request
                )
            except _Closed:
                raise StageProcessError(
                    f"stage process was closed while staging version {version_id!r}"
                ) from None
            self._replaced = 0
        if not reply.get("ok"):
            raise StageProcessError(
                f"stage process failed to stage version {version_id!r}: "
                f"{reply.get('error')}"
            )
        metrics = {**reply["metrics"], "perf/mx_stage_process_time": reply["seconds"]}
        if started:
            # Workers this request had to start before one replied.
            metrics["perf/mx_stage_process_restarts"] = float(started)
        return reply["versions"], metrics

    def release(self) -> None:
        """Tell the worker to drop what its last stage holds; best effort."""
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.send({"op": "release"})
            except (OSError, ValueError):
                logger.debug("stage process did not take a release", exc_info=True)

    def close(self) -> None:
        """Stop the worker without interrupting a staging step; never raises."""
        # A waiting stage() notices this and gives up the lock.
        self._closed = True
        with self._lock:
            proc, conn = self._proc, self._conn
            self._proc = self._conn = None
        if proc is None or conn is None:
            return
        try:
            try:
                # Closing the connection alone does not reach a worker when
                # another process, such as a fork of this one, holds this end.
                conn.send({"op": "shutdown"})
            except (OSError, ValueError):
                pass
            conn.close()
            try:
                proc.wait(timeout=self._close_grace_seconds)
            except subprocess.TimeoutExpired:
                # Killing it during a staging step would leave that step's
                # state half written. It exits once the step ends.
                logger.info(
                    "stage process pid=%d is still staging; it exits on its own",
                    proc.pid,
                )
                threading.Thread(
                    target=proc.wait,
                    name="modelexpress-refit-stage-process-reaper",
                    daemon=True,
                ).start()
        except Exception:
            logger.warning("failed to stop the stage process", exc_info=True)

    def _with_worker(self, what: str, request: Callable[[], _T]) -> tuple[_T, int]:
        """Run ``request`` with a ready worker and return its result.

        Start a worker first when there is none. When the worker is gone, or
        does not start, try again with a new one, ``_ATTEMPTS`` times in all.
        Also return how many workers this call started. The caller holds the
        lock.
        """
        started = 0
        reason: Exception | None = None
        for attempt in range(1, _ATTEMPTS + 1):
            if self._closed:
                raise _Closed
            try:
                if self._conn is None:
                    started += 1
                    self._start()
                return request(), started
            except _WorkerGone as error:
                reason = error
                logger.warning(
                    "%s: no stage process (%s), attempt %d of %d",
                    what,
                    reason,
                    attempt,
                    _ATTEMPTS,
                )
        raise StageProcessError(f"{what} failed in {_ATTEMPTS} attempts: {reason}")

    def _start(self) -> None:
        """Start a worker and wait until it is ready; _WorkerGone if it is not."""
        # One socket pair, private to the two processes.
        conn, worker_end = multiprocessing.Pipe()
        try:
            # The configuration waits in the socket until the worker reads it.
            conn.send({"op": "init", "config": self._config})
            proc = subprocess.Popen(
                [sys.executable, "-m", _WORKER_MODULE, str(worker_end.fileno())],
                pass_fds=[worker_end.fileno()],
                stdin=subprocess.DEVNULL,
                # The worker never uses a GPU.
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            )
        except Exception as error:
            conn.close()
            raise _WorkerGone(f"it could not be started: {error}") from error
        finally:
            worker_end.close()
        self._proc, self._conn = proc, conn
        try:
            message = self._receive(time.monotonic() + self._start_timeout_seconds)
        except _WorkerGone:
            code = self._drop(kill=False)
            raise _WorkerGone(
                f"it exited with code {code} before it was ready"
            ) from None
        except _DeadlineExceeded:
            self._drop(kill=True)
            raise _WorkerGone(
                f"it was not ready within {self._start_timeout_seconds:g} s"
            ) from None
        package = str(message.get("package"))
        if message.get("event") != "ready":
            reason = f"it failed to start: {message.get('error')}"
        elif os.path.realpath(package) != os.path.realpath(_PACKAGE_DIR):
            reason = (
                f"it imported ModelExpress from {package}, but this process "
                f"uses {_PACKAGE_DIR}"
            )
        else:
            logger.info("started stage process pid=%d", proc.pid)
            threading.Thread(
                target=self._replace_on_exit,
                args=(proc,),
                name="modelexpress-refit-stage-process-watch",
                daemon=True,
            ).start()
            return
        self._drop(kill=True)
        raise _WorkerGone(reason)

    def _replace_on_exit(self, proc: subprocess.Popen) -> None:
        """Wait for a ready worker to exit, then start its replacement.

        The next request then finds a worker ready. A request that was waiting
        on this worker replaces it itself, and nothing is left to do here.
        """
        code = proc.wait()
        with self._lock:
            if self._closed:
                return
            if self._proc is proc:
                self._drop(kill=False)
            if self._conn is not None:
                return
            if self._replaced >= _ATTEMPTS:
                logger.warning(
                    "stage process pid=%d exited with code %s; %d in a row exited "
                    "without a reply, so the next request starts one",
                    proc.pid,
                    code,
                    _ATTEMPTS + 1,
                )
                return
            self._replaced += 1
            logger.warning(
                "stage process pid=%d exited with code %s; starting another",
                proc.pid,
                code,
            )
            try:
                self._with_worker("replacing the stage process", lambda: None)
            except _Closed:
                pass
            except StageProcessError as error:
                logger.warning("%s; the next request tries again", error)

    def _receive(self, deadline: float) -> dict[str, Any]:
        proc, conn = self._proc, self._conn
        assert proc is not None
        while True:
            if self._closed:
                raise _Closed
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _DeadlineExceeded
            try:
                if conn.poll(min(_TICK_SECONDS, remaining)):
                    return conn.recv()
                # End of file does not arrive while a process that the worker
                # started still holds the worker's end.
                if proc.poll() is not None and not conn.poll():
                    raise _WorkerGone
            except (EOFError, OSError):
                raise _WorkerGone from None

    def _drop(self, *, kill: bool) -> int | None:
        """Forget the worker and return its exit code; ``kill`` does not wait."""
        proc, conn = self._proc, self._conn
        self._proc = self._conn = None
        if proc is None or conn is None:
            return None
        conn.close()
        if not kill:
            try:
                return proc.wait(timeout=_REAP_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        proc.kill()
        return proc.wait()


__all__ = ["StageProcess", "StageProcessError"]
