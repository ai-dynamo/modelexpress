# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import errno
import io
import subprocess
import urllib.error
import urllib.request
from unittest.mock import patch

import pytest

from kube_utils import completion_response, kubectl


@pytest.mark.parametrize("error", ["connection refused", "connection reset by peer", "unexpected EOF",
                                 "context deadline exceeded", "Client.Timeout exceeded", "i/o timeout", "TLS handshake timeout"])
def test_api_read_recovers(error):
    failed = subprocess.CompletedProcess(["kubectl"], 1, "", error)
    ready = subprocess.CompletedProcess(["kubectl"], 0, "pod", "")
    with patch("kube_utils.subprocess.run", side_effect=[failed, ready]) as run, patch("kube_utils.time.sleep"):
        assert kubectl("get", "pods", namespace="test").stdout == "pod"
        assert run.call_count == 2


@pytest.mark.parametrize("command,error,calls", [
    ("get", "connection refused", 4),
    ("get", "Forbidden", 1),
    ("delete", "connection refused", 1),
])
def test_api_retry_boundaries(command, error, calls):
    failed = subprocess.CompletedProcess(["kubectl"], 1, "", error)
    with patch("kube_utils.subprocess.run", return_value=failed) as run, patch("kube_utils.time.sleep"):
        with pytest.raises(subprocess.CalledProcessError):
            kubectl(command, "pods", namespace="test")
        assert run.call_count == calls


@pytest.mark.parametrize("error", [ConnectionResetError(errno.ECONNRESET, "reset"), urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "refused"))])
def test_completion_recovers(error):
    with patch("kube_utils.urllib.request.urlopen", side_effect=[error, io.BytesIO(b'{"choices":[{"text":"Paris"}]}')]) as request, patch("kube_utils.time.sleep"):
        assert completion_response(urllib.request.Request("http://localhost"))["choices"][0]["text"] == "Paris"
        assert request.call_count == 2


@pytest.mark.parametrize("error,calls", [
    (ConnectionResetError(errno.ECONNRESET, "reset"), 3),
    (urllib.error.HTTPError("http://localhost", 500, "server error", {}, None), 1),
    (TimeoutError("timeout"), 1),
])
def test_completion_retry_boundaries(error, calls):
    with patch("kube_utils.urllib.request.urlopen", side_effect=error) as request, patch("kube_utils.time.sleep"):
        with pytest.raises(type(error)):
            completion_response(urllib.request.Request("http://localhost"))
        assert request.call_count == calls


def test_tunnel_restarts_on_same_port_and_cleans_up(tmp_path, monkeypatch):
    import os
    import time
    from pathlib import Path

    fake = tmp_path / "kubectl"
    fake.write_text('''#!/usr/bin/env python3
import os, sys, time
from pathlib import Path
p = Path(os.environ["TUNNEL_FIXTURE"])
with (p / "calls").open("a") as out:
    out.write(" ".join(sys.argv[1:]) + "\\n")
if len((p / "calls").read_text().splitlines()) == 1:
    print("error: lost connection to pod", flush=True)
    sys.exit(1)
(p / "child").write_text(str(os.getpid()))
while True:
    time.sleep(0.1)
''')
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("TUNNEL_FIXTURE", str(tmp_path))
    supervisor = Path(__file__).resolve().parents[3] / ".github/scripts/keep-port-forward.sh"
    with (tmp_path / "log").open("w") as log:
        proc = subprocess.Popen(
            ["bash", str(supervisor), "--kubeconfig=host", "port-forward", "svc/vcluster", "51783:443"],
            stdout=log, stderr=log,
        )
        try:
            deadline = time.monotonic() + 10
            while not (tmp_path / "child").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            assert (tmp_path / "child").exists(), (tmp_path / "log").read_text()
            calls = (tmp_path / "calls").read_text().splitlines()
            assert len(calls) == 2 and calls[0] == calls[1]
            child = int((tmp_path / "child").read_text())
            proc.terminate()
            proc.wait(timeout=5)
            with pytest.raises(ProcessLookupError):
                os.kill(child, 0)
            assert "lost connection to pod" in (tmp_path / "log").read_text()
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)


def test_tunnel_diagnostics_keep_stdout_parseable(capsys):
    from kube_utils import port_forward

    with patch("kube_utils.subprocess.Popen"), patch("kube_utils.socket.socket"):
        with port_forward("fixture", "svc/mx-server", 18000, 8000):
            print("2 1")
    captured = capsys.readouterr()
    assert captured.out == "2 1\n"
    assert captured.err == ""


def test_tunnel_exit_during_kill_preserves_original_failure(capsys):
    from kube_utils import port_forward

    with patch("kube_utils.subprocess.Popen") as popen, patch("kube_utils.socket.socket"), \
            patch("kube_utils.os.killpg", side_effect=ProcessLookupError):
        popen.side_effect = lambda *args, **kwargs: (
            kwargs["stdout"].write("lost connection to pod\n"), popen.return_value
        )[1]
        popen.return_value.wait.side_effect = [subprocess.TimeoutExpired("supervisor", 5), 0]
        with pytest.raises(ValueError, match="original test failure"):
            with port_forward("fixture", "svc/mx-server", 18000, 8000):
                raise ValueError("original test failure")
    assert "Port-forward log" in capsys.readouterr().err


@pytest.mark.parametrize("error", ["", "lost connection to pod\n"])
def test_port_forward_suppresses_routine_output_and_retains_errors(capsys, error):
    from kube_utils import port_forward

    output = (
        "2026-10-13T00:00:00Z Starting kubectl port-forward\n"
        "Forwarding from 127.0.0.1:18000 -> 8000\n"
        "Forwarding from [::1]:18000 -> 8000\n"
        "Handling connection for 18000\n" + error
    )
    with patch("kube_utils.subprocess.Popen") as popen, patch("kube_utils.socket.socket"):
        popen.side_effect = lambda *args, **kwargs: (
            kwargs["stdout"].write(output), popen.return_value
        )[1]
        with port_forward("fixture", "svc/mx-server", 18000, 8000):
            pass
    captured = capsys.readouterr()
    assert "Forwarding from" not in captured.err
    assert "Handling connection" not in captured.err
    if error:
        assert error in captured.err
    else:
        assert captured.err == ""
