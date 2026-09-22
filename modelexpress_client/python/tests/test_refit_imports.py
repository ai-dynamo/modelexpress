# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys


def test_full_tensor_package_import_does_not_require_delta_compression():
    code = """
import builtins

original_import = builtins.__import__

def import_without_zstandard(name, *args, **kwargs):
    if name == "zstandard":
        raise ModuleNotFoundError("blocked optional delta dependency")
    return original_import(name, *args, **kwargs)

builtins.__import__ = import_without_zstandard
import modelexpress_rl
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_package_import_does_not_load_optional_engine_implementations():
    code = """
import sys
import modelexpress_rl

assert "modelexpress.engines.vllm.adapter" not in sys.modules
assert "modelexpress_rl.inference.engines.vllm.installer" not in sys.modules
assert "modelexpress_rl.inference.engines.sglang.installer" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
