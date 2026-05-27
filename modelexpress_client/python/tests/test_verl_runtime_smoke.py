# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
from pathlib import Path
import sys

import pytest


_LIVE_SERVER_ENV = "MX_LIVE_SERVER_URL"
_VERL_REPO_ENV = "MX_VERL_REPO_PATH"
_HARNESS = "_mx_verl_runtime_harness"

pytestmark = pytest.mark.skipif(
    not os.environ.get(_LIVE_SERVER_ENV) or not os.environ.get(_VERL_REPO_ENV),
    reason=f"{_LIVE_SERVER_ENV} and {_VERL_REPO_ENV} must be set",
)


def test_live_verl_checkpoint_manager_updates_weights_with_modelexpress(tmp_path):
    """Exercise veRL's real CheckpointEngineManager with the MX backend."""
    run_verl_checkpoint_manager_update = _load_harness().run_verl_checkpoint_manager_update
    result = run_verl_checkpoint_manager_update(
        backend="modelexpress",
        tmp_path=tmp_path,
        verl_repo=Path(os.environ[_VERL_REPO_ENV]).resolve(),
        mx_python_path=Path(__file__).resolve().parents[1],
        server_url=os.environ[_LIVE_SERVER_ENV],
    )

    assert result.backend == "modelexpress"
    assert result.update_seconds > 0.0
    assert result.check_seconds > 0.0


def test_live_verl_checkpoint_manager_compares_nccl_and_modelexpress(tmp_path):
    """Run CE/NCCL and MX/NIXL through the same tiny veRL manager harness."""
    run_verl_checkpoint_manager_update = _load_harness().run_verl_checkpoint_manager_update
    verl_repo = Path(os.environ[_VERL_REPO_ENV]).resolve()
    mx_python_path = Path(__file__).resolve().parents[1]
    nccl_result = run_verl_checkpoint_manager_update(
        backend="nccl",
        tmp_path=tmp_path,
        verl_repo=verl_repo,
        mx_python_path=mx_python_path,
    )
    mx_result = run_verl_checkpoint_manager_update(
        backend="modelexpress",
        tmp_path=tmp_path,
        verl_repo=verl_repo,
        mx_python_path=mx_python_path,
        server_url=os.environ[_LIVE_SERVER_ENV],
    )

    assert nccl_result.update_seconds > 0.0
    assert mx_result.update_seconds > 0.0
    print(
        "veRL tiny-Qwen2 update timings: "
        f"nccl={nccl_result.update_seconds:.6f}s, "
        f"modelexpress={mx_result.update_seconds:.6f}s"
    )


def _load_harness():
    module_path = Path(__file__).with_name("_verl_runtime_harness.py")
    spec = importlib.util.spec_from_file_location(_HARNESS, module_path)
    if spec is None or spec.loader is None:
        pytest.skip(f"could not load veRL runtime harness from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_HARNESS] = module
    spec.loader.exec_module(module)
    return module
