# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The TYPE_CHECKING re-exports in modelexpress_rl must match the lazy table."""

import ast
import subprocess
import sys
from pathlib import Path

import modelexpress_rl


def _type_checking_imports() -> dict[str, str]:
    tree = ast.parse(Path(modelexpress_rl.__file__).read_text())
    found: dict[str, str] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for stmt in node.body:
                assert isinstance(stmt, ast.ImportFrom)
                for alias in stmt.names:
                    found[alias.name] = "." + (stmt.module or "")
    return found


def test_every_lazy_export_has_a_static_type():
    assert _type_checking_imports() == modelexpress_rl._LAZY


def test_importing_the_package_stays_lazy():
    code = (
        "import sys, modelexpress_rl; "
        "print(sorted(m for m in sys.modules if m.startswith('modelexpress_rl.')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
