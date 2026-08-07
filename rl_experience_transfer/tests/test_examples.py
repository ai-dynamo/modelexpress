# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep every dependency-light example executable."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.e2e
@pytest.mark.multi_process
@pytest.mark.parametrize(
    "script",
    ["basic.py", "cross_framework.py", "filesystem_process.py", "reliable_transfer.py"],
)
def test_dependency_light_example(script: str) -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "examples" / script)],
        cwd=root,
        capture_output=True,
        check=False,
        text=True,
        timeout=20.0,
    )
    assert result.returncode == 0, result.stdout + result.stderr
