# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The stage worker: a generator client without an engine, in its own process.

``StageProcess`` starts it as ``python -m modelexpress_rl.inference.stage_worker
<fd>``, where ``<fd>`` is its end of the connection. It builds a
``ModelExpressGeneratorClient`` from the configuration the client sends and
answers each stage request by running that client's staging step,
``_stage_weight``. It never installs.
``stage_process.py`` lists the messages.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from multiprocessing.connection import Connection
from typing import Any

from ..version import WeightVersionRef
from .client import ModelExpressGeneratorClient, StagedWeightHandle, _EngineState

logger = logging.getLogger("modelexpress_rl.inference.stage_worker")

# How often an idle worker checks that its client is still there.
_IDLE_TICK_SECONDS = 1.0
# How long a started worker waits for its configuration.
_INIT_WAIT_SECONDS = 30.0


def _send(conn: Connection, message: dict[str, Any]) -> bool:
    """Send one message; return False once the client no longer reads."""
    try:
        conn.send(message)
    except (OSError, ValueError):
        return False
    return True


def _release(handle: StagedWeightHandle | None) -> None:
    if handle is None:
        return
    try:
        handle.release()
    except Exception:
        logger.warning(
            "failed to release staged version %s", handle.version_id, exc_info=True
        )


def _serve(
    client: ModelExpressGeneratorClient,
    conn: Connection,
    *,
    parent_pid: int,
    draining: threading.Event,
) -> None:
    """Answer requests until the client shuts this worker down or is gone."""
    handle: StagedWeightHandle | None = None
    try:
        while True:
            if draining.is_set():
                logger.info("terminating on SIGTERM")
                return
            # A process forked from the client can hold the client's end open
            # after the client is gone. This worker's parent then changes.
            if os.getppid() != parent_pid:
                logger.info("parent process %d is gone; exiting", parent_pid)
                return
            try:
                if not conn.poll(_IDLE_TICK_SECONDS):
                    continue
                request = conn.recv()
            except (EOFError, OSError):
                logger.info("the client closed the connection; exiting")
                return
            if draining.is_set():
                # SIGTERM came while this waited: the request is not started.
                logger.info("terminating on SIGTERM")
                return
            op = request.get("op")
            if op == "shutdown":
                return
            if op == "release":
                _release(handle)
                handle = None
            elif op == "stage":
                # What an earlier request staged is given up for a newer one.
                _release(handle)
                handle = None
                started = time.perf_counter()
                try:
                    # This client never installs, so each request tells it
                    # which version the engine serves and whether the engine is
                    # known to hold it. After a failed install it is not, and
                    # staging that version again is a real update.
                    client._serving_version_id = request["serving_version_id"]
                    client._engine_state = _EngineState(request["engine_state"])
                    handle = client._stage_weight(
                        WeightVersionRef(request["version_id"])
                    )
                    reply = {
                        "ok": True,
                        # What the MX server said, so the engine's client
                        # does not have to ask it again.
                        "versions": dict(client._fetched_versions),
                        "metrics": dict(handle.metrics),
                        "seconds": time.perf_counter() - started,
                    }
                except Exception as error:
                    logger.exception(
                        "failed to stage version %s", request.get("version_id")
                    )
                    reply = {"ok": False, "error": f"{type(error).__name__}: {error}"}
                if not _send(conn, reply):
                    logger.info("the client no longer reads replies; exiting")
                    return
            else:
                logger.warning("ignoring a request with unknown op %r", op)
    finally:
        _release(handle)


def main() -> int:
    # Only the ModelExpress loggers log below WARNING. Object-storage and HTTP
    # libraries log every request below it, and at DEBUG its signed headers.
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING,
        format=(
            "%(asctime)s %(levelname)s "
            f"[mx-stage-worker pid={os.getpid()} parent={os.getppid()}] "
            "%(name)s: %(message)s"
        ),
    )
    for name in ("modelexpress", "modelexpress_rl"):
        logging.getLogger(name).setLevel(logging.INFO)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # Termination waits for the staging step in progress.
    draining = threading.Event()
    signal.signal(signal.SIGTERM, lambda _signum, _frame: draining.set())

    fd = int(sys.argv[1])
    # A process this worker starts must not hold the connection open after it.
    os.set_inheritable(fd, False)
    conn = Connection(fd)
    parent_pid = os.getppid()
    try:
        if not conn.poll(_INIT_WAIT_SECONDS):
            logger.error("no configuration arrived within %g s", _INIT_WAIT_SECONDS)
            return 1
        init = conn.recv()
    except (EOFError, OSError):
        # The client is gone.
        return 0
    try:
        client = ModelExpressGeneratorClient.initialize(init["config"])
    except Exception as error:
        logger.exception("stage worker failed to start")
        _send(conn, {"event": "fatal", "error": f"{type(error).__name__}: {error}"})
        return 1
    try:
        ready = {
            "event": "ready",
            "pid": os.getpid(),
            # Where this worker imported ModelExpress from.
            "package": os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        }
        if _send(conn, ready):
            _serve(client, conn, parent_pid=parent_pid, draining=draining)
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
