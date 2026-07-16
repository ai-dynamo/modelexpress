# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4


def test_external_coordinator_publishes_one_immutable_release(tmp_path):
    python_root = Path(__file__).parents[1]
    script = Path(__file__).parent / "gpu" / "m2n_test_coordinator.py"
    attempt_id = str(uuid4())
    attempt_directory = tmp_path / attempt_id
    env = os.environ.copy()
    env.update(
        {
            "MX_M2N_ATTEMPT_ID": attempt_id,
            "MX_M2N_COORDINATOR_DIR": str(tmp_path),
            "MX_M2N_TIMEOUT_S": "5",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": os.pathsep.join(
                filter(None, [str(python_root), env.get("PYTHONPATH", "")])
            ),
            "WORLD_SIZE": "2",
        }
    )

    process = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=python_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 2
    while not attempt_directory.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert attempt_directory.exists()

    (attempt_directory / "ready-0").write_text("rank-0", encoding="utf-8")
    time.sleep(0.1)
    assert process.poll() is None
    assert not (attempt_directory / "decision").exists()

    (attempt_directory / "ready-1").write_text("rank-1", encoding="utf-8")
    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 0, (stdout, stderr)
    decision = attempt_directory / "decision"
    assert decision.read_text(encoding="utf-8") == "release"

    duplicate = subprocess.run(
        [sys.executable, str(script)],
        cwd=python_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert duplicate.returncode != 0
    assert decision.read_text(encoding="utf-8") == "release"
