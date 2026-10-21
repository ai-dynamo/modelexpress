# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "suffix",
    [
        "timing", "reshard", "reshard.geometry", "reshard.types",
        "reshard.slice_plan", "reshard.transfer_plan", "reshard.receiver",
        "reshard.rendezvous", "reshard.verify", "reshard.throughput",
        "reshard.cuda_pool", "reshard.transport", "reshard.transport.base",
        "reshard.transport.nixl",
    ],
)
def test_legacy_refit_modules_share_rl_implementation(suffix):
    legacy = importlib.import_module(f"modelexpress.refit.{suffix}")
    current = importlib.import_module(f"modelexpress_rl.refit.{suffix}")
    assert legacy is current


def test_legacy_refit_exports_share_timing_state():
    import modelexpress.refit as legacy
    import modelexpress.refit_timing as old_timing
    from modelexpress_rl.refit import timing

    assert old_timing is timing
    assert legacy.RefitTimingRecorder is timing.RefitTimingRecorder
    recorder = legacy.RefitTimingRecorder(backend="test", version="version-a")
    with legacy.use_refit_timing(recorder):
        assert timing.current_refit_timing() is recorder


@pytest.mark.parametrize("name", ["aliases", "publisher"])
def test_legacy_megatron_modules_share_rl_implementation(name):
    legacy = importlib.import_module(f"modelexpress.refit.reshard.megatron_{name}")
    current = importlib.import_module(f"modelexpress_rl.train.engines.megatron.{name}")
    assert legacy is current


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


def test_package_import_does_not_load_optional_engine_or_framework_implementations():
    code = """
import sys
import modelexpress_rl
from modelexpress_rl.inference.engines.sglang import get_modelexpress_generator

assert callable(get_modelexpress_generator)
assert "miles" not in sys.modules
assert "sglang" not in sys.modules
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


def test_storage_readers_import_without_azure_or_native_streaming_dependencies():
    code = """
import builtins

original_import = builtins.__import__

def import_without_optional_dependencies(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split('.')[0] in {"azure", "runai_model_streamer", "vllm", "nixl"}:
        raise ModuleNotFoundError("blocked optional dependency", name=name.split('.')[0])
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_optional_dependencies
from modelexpress_rl.s3 import S3Client
from modelexpress_rl.azure import AzureBlobReader

try:
    AzureBlobReader()
except ImportError as error:
    assert "modelexpress[azure]" in str(error)
    assert isinstance(error.__cause__, ModuleNotFoundError)
    assert error.__cause__.name == "azure"
else:
    raise AssertionError("expected a missing Azure dependency error")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
