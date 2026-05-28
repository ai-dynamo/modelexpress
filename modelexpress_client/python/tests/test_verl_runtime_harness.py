# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
import sys

import pytest

_HARNESS = "_mx_verl_runtime_harness_unit"


def test_verl_runtime_result_serializes_json_ready_values():
    harness = _load_harness()
    result = harness.VerlRuntimeSmokeResult(
        backend="modelexpress",
        model_name="tiny-qwen2",
        update_seconds=1.5,
        check_seconds=0.25,
        total_seconds=2.0,
        source_roles=("trainer",),
        attempt_successes=(True,),
        lease_summary_statuses=None,
    )

    output = result.to_dict()

    assert output["backend"] == "modelexpress"
    assert output["source_roles"] == ["trainer"]
    assert output["attempt_successes"] == [True]
    assert output["lease_summary_statuses"] is None


def test_verl_runtime_comparison_serializes_summary():
    harness = _load_harness()
    nccl_result = harness.VerlRuntimeSmokeResult(
        backend="nccl",
        model_name="tiny-qwen2",
        update_seconds=2.0,
        check_seconds=0.25,
        total_seconds=2.5,
    )
    mx_result = harness.VerlRuntimeSmokeResult(
        backend="modelexpress",
        model_name="tiny-qwen2",
        update_seconds=3.0,
        check_seconds=0.5,
        total_seconds=4.0,
        source_roles=("trainer",),
        bytes_transferred=1024,
        tensor_count=15,
        attempt_lease_ids=("lease-a",),
    )

    output = harness.verl_runtime_comparison_to_dict(nccl_result, mx_result)

    assert output["schema_version"] == 1
    assert output["benchmark"] == "verl_tiny_qwen2_checkpoint_manager"
    assert output["summary"] == {
        "nccl_update_seconds": 2.0,
        "modelexpress_update_seconds": 3.0,
        "update_seconds_delta": 1.0,
        "modelexpress_to_nccl_update_ratio": 1.5,
        "modelexpress_bytes_transferred": 1024,
        "modelexpress_tensor_count": 15,
        "modelexpress_retry_count": 0,
        "modelexpress_source_roles": ["trainer"],
        "modelexpress_attempt_lease_ids": ["lease-a"],
    }
    assert output["results"]["nccl"]["backend"] == "nccl"
    assert output["results"]["modelexpress"]["attempt_lease_ids"] == ["lease-a"]


def test_verl_runtime_comparison_ratio_handles_zero_baseline():
    harness = _load_harness()
    nccl_result = harness.VerlRuntimeSmokeResult(
        backend="nccl",
        model_name="tiny-qwen2",
        update_seconds=0.0,
        check_seconds=0.0,
        total_seconds=0.0,
    )
    mx_result = harness.VerlRuntimeSmokeResult(
        backend="modelexpress",
        model_name="tiny-qwen2",
        update_seconds=3.0,
        check_seconds=0.0,
        total_seconds=3.0,
    )

    output = harness.verl_runtime_comparison_to_dict(nccl_result, mx_result)

    assert output["summary"]["modelexpress_to_nccl_update_ratio"] is None


def _load_harness():
    module_path = Path(__file__).with_name("_verl_runtime_harness.py")
    spec = importlib.util.spec_from_file_location(_HARNESS, module_path)
    if spec is None or spec.loader is None:
        pytest.skip(f"could not load veRL runtime harness from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_HARNESS] = module
    spec.loader.exec_module(module)
    return module
