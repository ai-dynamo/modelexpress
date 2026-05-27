# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared kubectl helpers used by every pytest test in this directory.

Kept deliberately small: the only things in here are operations that were
copy-pasted across `test_p2p_k8s.py`, `test_dynamo_p2p.py`, and
`test_stale_metadata.py`. If you find yourself adding a function used by
only one test, it belongs in that test file, not here.
"""

import socket
import subprocess
import time
from contextlib import contextmanager


def kubectl(*args: str, namespace: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run `kubectl -n <namespace> <args...>` and return the completed process.

    `check=True` (default) raises on non-zero exit, matching the original
    per-test implementations. Pass `check=False` for "best-effort" queries
    where a missing resource shouldn't be treated as a test failure (e.g.
    log dumps during cleanup).
    """
    return subprocess.run(
        ["kubectl", "-n", namespace, *args],
        capture_output=True,
        text=True,
        check=check,
    )


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

    On exit, SIGTERM the port-forward and reap; if it doesn't die within
    5s, escalate to SIGKILL. Without the SIGKILL fallback a stuck
    kubectl could outlive the test process and hold the local port,
    breaking later tests on the same runner.
    """
    proc = subprocess.Popen(
        ["kubectl", "-n", namespace, "port-forward", target, f"{local_port}:{remote_port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.perf_counter() + 60
        while time.perf_counter() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(2)
                try:
                    sock.connect(("localhost", local_port))
                    break
                except (ConnectionRefusedError, socket.timeout, OSError):
                    time.sleep(1)
        yield local_port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
