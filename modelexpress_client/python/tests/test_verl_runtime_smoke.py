# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from modelexpress import p2p_pb2
from modelexpress.rl_metadata import RlSourceRole

_LIVE_SERVER_ENV = "MX_LIVE_SERVER_URL"
_VERL_REPO_ENV = "MX_VERL_REPO_PATH"
_RUNTIME_COMPARISON_OUTPUT_ENV = "MX_VERL_RUNTIME_OUTPUT_JSON"
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
    assert result.receive_success
    assert result.requested_model_version == 17
    assert result.resolved_model_version == 17
    assert result.source_roles == (RlSourceRole.TRAINER.value,)
    assert result.attempt_roles == (RlSourceRole.TRAINER.value,)
    assert len(result.attempt_worker_ids) == 1
    assert result.attempt_successes == (True,)
    assert result.attempt_source_statuses == (p2p_pb2.SOURCE_STATUS_READY,)
    assert all(updated_at > 0 for updated_at in result.attempt_source_updated_at)
    assert result.retry_count == 0
    assert result.tensor_count == 15
    assert result.bytes_transferred > 0
    assert result.transfer_duration_seconds > 0.0
    assert len(result.attempt_duration_seconds) == 1
    assert result.attempt_duration_seconds[0] > 0.0
    assert result.attempt_lease_ids
    assert result.lease_summary_target_worker_id
    assert result.lease_summary_model_version == result.resolved_model_version
    assert result.lease_summary_source_worker_id == result.attempt_worker_ids[0]
    assert result.lease_summary_statuses is None


def test_live_verl_checkpoint_manager_compares_nccl_and_modelexpress(tmp_path):
    """Run CE/NCCL and MX/NIXL through the same tiny veRL manager harness."""
    harness = _load_harness()
    run_verl_checkpoint_manager_update = harness.run_verl_checkpoint_manager_update
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
    assert mx_result.source_roles == (RlSourceRole.TRAINER.value,)
    assert mx_result.attempt_successes == (True,)
    assert mx_result.tensor_count == 15
    assert mx_result.bytes_transferred > 0
    comparison = harness.verl_runtime_comparison_to_dict(nccl_result, mx_result)
    print(
        "veRL tiny-Qwen2 update timings: "
        f"nccl={nccl_result.update_seconds:.6f}s, "
        f"modelexpress={mx_result.update_seconds:.6f}s"
    )
    if output_json := os.environ.get(_RUNTIME_COMPARISON_OUTPUT_ENV):
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def test_live_verl_modelexpress_refit_failure_exposes_lease_summary(tmp_path):
    """Surface MX transfer lease state after a veRL-side refit failure."""
    run_verl_checkpoint_manager_update = _load_harness().run_verl_checkpoint_manager_update
    result = run_verl_checkpoint_manager_update(
        backend="modelexpress",
        tmp_path=tmp_path,
        verl_repo=Path(os.environ[_VERL_REPO_ENV]).resolve(),
        mx_python_path=Path(__file__).resolve().parents[1],
        server_url=os.environ[_LIVE_SERVER_ENV],
        fail_refit_after_tensors=1,
        expect_update_failure=True,
    )

    assert result.failed
    assert "synthetic veRL refit failure" in result.error_message
    assert result.receive_success
    assert result.source_roles == (RlSourceRole.TRAINER.value,)
    assert len(result.attempt_worker_ids) == 1
    assert result.attempt_successes == (True,)
    assert result.attempt_source_statuses == (p2p_pb2.SOURCE_STATUS_READY,)
    assert all(updated_at > 0 for updated_at in result.attempt_source_updated_at)
    assert result.retry_count == 0
    assert result.tensor_count == 15
    assert result.bytes_transferred > 0
    assert result.transfer_duration_seconds > 0.0
    assert len(result.attempt_duration_seconds) == 1
    assert result.attempt_duration_seconds[0] > 0.0
    assert result.report_lease_ids
    assert result.attempt_lease_ids == result.report_lease_ids
    assert result.transfer_lease_discovery_supported
    assert result.lease_summary_target_worker_id
    assert result.lease_summary_model_version == result.resolved_model_version
    assert result.lease_summary_source_worker_id == result.attempt_worker_ids[0]
    assert result.lease_summary_statuses is None
    assert result.matching_lease_statuses == (
        p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
    )
    assert result.missing_lease_ids == ()
    assert result.non_completed_lease_statuses == ()
    expected_events = (
        "abort_all_requests",
        "release_kv_cache",
    )
    if result.manager_cleanup_supported:
        expected_events += ("resume_kv_cache", "resume_generation")
    assert result.replica_events == expected_events


def test_live_verl_modelexpress_restarted_rollout_recovers_latest_from_replica(
    tmp_path,
):
    """Recover a fresh veRL rollout worker from an MX inference replica source."""
    run_verl_checkpoint_manager_update = _load_harness().run_verl_checkpoint_manager_update
    result = run_verl_checkpoint_manager_update(
        backend="modelexpress",
        tmp_path=tmp_path,
        verl_repo=Path(os.environ[_VERL_REPO_ENV]).resolve(),
        mx_python_path=Path(__file__).resolve().parents[1],
        server_url=os.environ[_LIVE_SERVER_ENV],
        republish_received=True,
        recover_latest_from_replica=True,
    )

    assert result.recovery_success
    assert result.recovery_update_seconds > 0.0
    assert result.recovery_check_seconds > 0.0
    assert result.recovery_requested_model_version is None
    assert result.recovery_resolved_model_version == 17
    assert result.recovery_source_roles == (RlSourceRole.INFERENCE_REPLICA.value,)
    assert result.recovery_attempt_roles == (RlSourceRole.INFERENCE_REPLICA.value,)
    assert len(result.recovery_attempt_worker_ids) == 1
    assert result.recovery_attempt_successes == (True,)
    assert result.recovery_attempt_source_statuses == (
        p2p_pb2.SOURCE_STATUS_READY,
    )
    assert all(
        updated_at > 0 for updated_at in result.recovery_attempt_source_updated_at
    )
    assert result.recovery_retry_count == 0
    assert result.recovery_tensor_count == 15
    assert result.recovery_bytes_transferred > 0
    assert result.recovery_transfer_duration_seconds > 0.0
    assert len(result.recovery_attempt_duration_seconds) == 1
    assert result.recovery_attempt_duration_seconds[0] > 0.0
    assert result.recovery_attempt_lease_ids
    assert result.recovery_attempt_lease_ids == result.recovery_report_lease_ids
    assert result.recovery_transfer_lease_discovery_supported
    assert result.recovery_lease_summary_target_worker_id
    assert (
        result.recovery_lease_summary_model_version
        == result.recovery_resolved_model_version
    )
    assert (
        result.recovery_lease_summary_source_worker_id
        == result.recovery_attempt_worker_ids[0]
    )
    assert result.recovery_lease_summary_statuses is None
    assert result.recovery_matching_lease_statuses == (
        p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
    )
    assert result.recovery_missing_lease_ids == ()
    assert result.recovery_non_completed_lease_statuses == ()


def test_live_verl_modelexpress_trainer_failure_falls_back_to_replica(
    tmp_path,
):
    """Retry a failed trainer transfer and recover from a replica source."""
    run_verl_checkpoint_manager_update = _load_harness().run_verl_checkpoint_manager_update
    result = run_verl_checkpoint_manager_update(
        backend="modelexpress",
        tmp_path=tmp_path,
        verl_repo=Path(os.environ[_VERL_REPO_ENV]).resolve(),
        mx_python_path=Path(__file__).resolve().parents[1],
        server_url=os.environ[_LIVE_SERVER_ENV],
        republish_received=True,
        recover_latest_from_replica=True,
        fail_trainer_transfer_before_recovery=True,
    )

    assert result.recovery_success
    assert result.recovery_requested_model_version is None
    assert result.recovery_resolved_model_version == 17
    assert result.recovery_source_roles == (RlSourceRole.INFERENCE_REPLICA.value,)
    assert result.recovery_attempt_roles == (
        RlSourceRole.TRAINER.value,
        RlSourceRole.INFERENCE_REPLICA.value,
    )
    assert len(result.recovery_attempt_worker_ids) == 2
    assert result.recovery_attempt_successes == (False, True)
    assert result.recovery_attempt_source_statuses == (
        p2p_pb2.SOURCE_STATUS_READY,
        p2p_pb2.SOURCE_STATUS_READY,
    )
    assert all(
        updated_at > 0 for updated_at in result.recovery_attempt_source_updated_at
    )
    assert result.recovery_retry_count == 1
    assert result.recovery_tensor_count == 15
    assert result.recovery_bytes_transferred > 0
    assert result.recovery_transfer_duration_seconds > 0.0
    assert len(result.recovery_attempt_duration_seconds) == 2
    assert result.recovery_attempt_duration_seconds[0] >= 0.0
    assert result.recovery_attempt_duration_seconds[1] > 0.0
    assert len(result.recovery_attempt_lease_ids) == 2
    assert result.recovery_attempt_lease_ids == result.recovery_report_lease_ids
    assert result.recovery_transfer_lease_discovery_supported
    assert result.recovery_lease_summary_target_worker_id
    assert (
        result.recovery_lease_summary_model_version
        == result.recovery_resolved_model_version
    )
    assert result.recovery_lease_summary_source_worker_id == ""
    assert result.recovery_lease_summary_statuses is None
    assert result.recovery_matching_lease_statuses == (
        p2p_pb2.TRANSFER_LEASE_STATUS_FAILED,
        p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
    )
    assert result.recovery_missing_lease_ids == ()
    assert result.recovery_non_completed_lease_statuses == (
        p2p_pb2.TRANSFER_LEASE_STATUS_FAILED,
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
