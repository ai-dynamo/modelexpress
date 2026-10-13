# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared kubectl helpers used by every pytest test in this directory.

Kept deliberately small: the only things in here are operations that were
copy-pasted across `test_p2p_k8s.py`, `test_dynamo_p2p.py`, and
`test_stale_metadata.py`. If you find yourself adding a function used by
only one test, it belongs in that test file, not here.
"""

import errno
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path


def kubectl(*args: str, namespace: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run `kubectl -n <namespace> <args...>` and return the completed process.

    `check=True` (default) raises on non-zero exit, matching the original
    per-test implementations. Pass `check=False` for "best-effort" queries
    where a missing resource shouldn't be treated as a test failure (e.g.
    log dumps during cleanup).
    """
    command = ["kubectl", "-n", namespace, *args]
    if args and args[0] == "get":
        command.insert(1, "--request-timeout=10s")
    for attempt in range(4):
        result = subprocess.run(command, capture_output=True, text=True)
        transient = any(error in result.stderr.lower() for error in (
            "connection refused", "connection reset by peer", "unexpected eof",
            "context deadline exceeded", "client.timeout exceeded", "i/o timeout", "tls handshake timeout",
        ))
        if result.returncode == 0 or not transient or not args or args[0] != "get" or attempt == 3:
            if check:
                result.check_returncode()
            return result
        print(f"kubectl API read failed; retrying: {result.stderr.strip()}", file=sys.stderr)
        time.sleep(2)


@contextmanager
def port_forward(namespace: str, target: str, local_port: int, remote_port: int):
    """Background `kubectl port-forward` to a Service or Pod, yield the local port.

    `target` accepts the kubectl-native forms `svc/<name>` or `pod/<name>`
    (or a bare pod name).

    Readiness is checked with a raw TCP `socket.connect` rather than an HTTP
    probe — works uniformly for HTTP servers (vLLM, SGLang, Dynamo Frontend)
    and pure-gRPC servers (mx-server). The HTTP probe used by earlier copies
    accepted any HTTP response (including 4xx/5xx) as "ready", so it was
    really just verifying the socket accepted — same signal as this probe,
    without the BadStatusLine failure mode when speaking HTTP/1.x at a
    gRPC server.

    On exit, SIGTERM the supervisor and reap; if it doesn't die within
    5s, SIGKILL its process group. Without this fallback a stuck
    kubectl could outlive the test process and hold the local port,
    breaking later tests on the same runner.
    """
    log = tempfile.TemporaryFile(mode="w+")
    supervisor = Path(__file__).resolve().parents[3] / ".github/scripts/keep-port-forward.sh"
    proc = subprocess.Popen(
        ["bash", str(supervisor), "-n", namespace, "port-forward", target, f"{local_port}:{remote_port}"],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        deadline = time.perf_counter() + 60
        ready = False
        while time.perf_counter() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2)
                try:
                    sock.connect(("localhost", local_port))
                    ready = True
                    break
                except (ConnectionRefusedError, socket.timeout, OSError):
                    time.sleep(1)
        if not ready:
            # Without this raise, the with-block proceeds with a port that
            # nothing's listening on, and the caller's first operation
            # surfaces an opaque "connection refused" several frames removed
            # from the real cause. Failing loudly here keeps the blame on
            # the port-forward setup rather than on the test logic.
            raise TimeoutError(
                f"kubectl port-forward in namespace {namespace} to {target} "
                f"(local:{local_port} -> remote:{remote_port}) "
                f"not ready after 60s"
            )
        yield local_port
    finally:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        finally:
            try:
                log.seek(0)
                diagnostics = "\n".join(
                    line for line in log.read().splitlines()
                    if not line.startswith(("Forwarding from ", "Handling connection for "))
                    and not line.endswith("Starting kubectl port-forward")
                )
                if diagnostics:
                    print(f"Port-forward log ({namespace}/{target}):\n{diagnostics}", file=sys.stderr)
            finally:
                log.close()


def completion_response(request: urllib.request.Request) -> dict:
    """Retry connection resets/refusals while the port-forward reconnects."""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.loads(response.read())
        except (ConnectionError, urllib.error.URLError) as error:
            reason = error.reason if isinstance(error, urllib.error.URLError) else error
            if not isinstance(reason, OSError) or reason.errno not in (errno.ECONNRESET, errno.ECONNREFUSED) or attempt == 2:
                raise
            print(f"Inference connection interrupted; retrying: {error}")
            time.sleep(3)
