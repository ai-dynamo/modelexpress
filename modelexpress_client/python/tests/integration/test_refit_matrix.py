# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral cleanup checks for the matrix runner, without NIXL dependencies."""

import os
from pathlib import Path
import signal
import sys
import threading
import time

import pytest

from .run_refit_matrix import run_suite


@pytest.mark.parametrize("interrupt", [False, True], ids=["timeout", "interrupt"])
def test_suite_cleans_process_tree(tmp_path, interrupt):
    executable = tmp_path / "synthetic-python"
    pids = tmp_path / "pids"
    child_code = (
        "import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"open({str(pids)!r}, 'w').write(str(os.getppid()) + ' ' + str(os.getpid())); "
        "time.sleep(60)"
    )
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os, signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        f"{child_code!r}])\n"
        "time.sleep(60)\n"
    )
    executable.chmod(0o755)
    timer = threading.Timer(1, lambda: os.kill(os.getpid(), signal.SIGINT)) if interrupt else None
    started = time.monotonic()
    try:
        if timer:
            timer.start()
        if interrupt:
            with pytest.raises(KeyboardInterrupt):
                run_suite(executable, tmp_path, tmp_path, "synthetic", ["unused"], os.environ, timeout=10)
        else:
            result = run_suite(executable, tmp_path, tmp_path, "synthetic", ["unused"], os.environ, timeout=1)
            assert result["timed_out"]
            assert result["tests"][0]["status"] == "blocked"
        parent, child = map(int, pids.read_text().split())
        assert time.monotonic() - started < 5
        with pytest.raises(ChildProcessError):
            os.waitpid(parent, os.WNOHANG)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            state = Path(f"/proc/{child}/stat")
            if not state.exists() or state.read_text().split()[2] == "Z":
                break
            time.sleep(0.01)
        else:
            pytest.fail("SIGTERM-ignoring child survived suite cleanup")
    finally:
        if timer:
            timer.cancel()
            timer.join()
        if pids.exists():
            for pid in map(int, pids.read_text().split()):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
