# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import uuid

import pytest

from modelexpress import p2p_pb2
from modelexpress.rl_benchmark import (
    RlTransferBenchmarkConfig,
    run_mx_rl_transfer_benchmark,
)

_LIVE_SERVER_ENV = "MX_LIVE_SERVER_URL"

pytestmark = pytest.mark.skipif(
    not os.environ.get(_LIVE_SERVER_ENV),
    reason=f"{_LIVE_SERVER_ENV} is not set",
)


def test_live_rl_benchmark_reports_transfer_lease_summary():
    torch = pytest.importorskip("torch")
    pytest.importorskip("nixl._api")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < 2:
        pytest.skip("live RL benchmark lease summary needs at least 2 CUDA devices")

    result = run_mx_rl_transfer_benchmark(
        RlTransferBenchmarkConfig(
            server_url=os.environ[_LIVE_SERVER_ENV],
            model_name=f"mx-live-rl-benchmark-{uuid.uuid4().hex[:8]}",
            tensor_count=2,
            tensor_shape=(256, 256),
            iterations=1,
            warmup_iterations=0,
            source_device_id=0,
            target_device_id=1,
        )
    )

    output = result.to_dict()
    summary = output["summary"]
    if not summary["lease_discovery_supported"]:
        pytest.skip("server does not expose transfer lease discovery")

    assert summary["iterations"] == 1
    assert summary["total_attempts"] == 1
    assert summary["successful_attempts"] == 1
    assert summary["failed_attempts"] == 0
    assert summary["attempts_with_lease_ids"] == 1
    assert summary["iterations_with_missing_lease_ids"] == 0
    assert summary["non_completed_matching_leases"] == 0
    assert summary["mean_transfer_duration_seconds"] > 0.0
    assert summary["mean_transfer_bandwidth_gbps"] > 0.0

    iteration = output["iterations"][0]
    assert iteration["transfer_lease_discovery_supported"] is True
    assert iteration["transfer_duration_seconds"] > 0.0
    assert iteration["transfer_bandwidth_gbps"] > 0.0
    assert iteration["matching_lease_status_names"] == [
        "TRANSFER_LEASE_STATUS_COMPLETED",
    ]
    assert iteration["non_completed_lease_status_names"] == []
    assert iteration["attempt_lease_ids"]
    assert iteration["report_lease_ids"] == iteration["attempt_lease_ids"]
    assert iteration["lease_summary_target_worker_id"]
    assert iteration["lease_summary_model_version"] == iteration["model_version"]
    assert iteration["lease_summary_source_worker_id"] == iteration["source_worker_id"]
    assert iteration["lease_summary_statuses"] is None
    assert iteration["missing_lease_ids"] == []
    assert iteration["matching_lease_statuses"] == [
        p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED
    ]
    assert iteration["non_completed_lease_statuses"] == []
    assert iteration["attempts"][0]["lease_id"] == iteration["attempt_lease_ids"][0]


def test_live_rl_benchmark_reports_dense_fanin_transfer():
    torch = pytest.importorskip("torch")
    pytest.importorskip("nixl._api")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < 3:
        pytest.skip("live RL dense fan-in benchmark needs at least 3 CUDA devices")

    result = run_mx_rl_transfer_benchmark(
        RlTransferBenchmarkConfig(
            server_url=os.environ[_LIVE_SERVER_ENV],
            model_name=f"mx-live-rl-benchmark-fanin-{uuid.uuid4().hex[:8]}",
            tensor_count=2,
            tensor_shape=(256, 128),
            iterations=1,
            warmup_iterations=0,
            source_device_id=0,
            target_device_id=2,
            dense_fanin_sources=2,
        )
    )

    output = result.to_dict()
    summary = output["summary"]
    iteration = output["iterations"][0]

    assert output["config"]["dense_fanin_sources"] == 2
    assert output["config"]["source_device_ids"] == [0, 1]
    assert summary["iterations"] == 1
    assert summary["total_attempts"] == 2
    assert summary["successful_attempts"] == 2
    assert summary["failed_attempts"] == 0
    assert summary["total_transferred_bytes"] == iteration["expected_bytes"]
    assert iteration["transferred_bytes"] == iteration["expected_bytes"]
    assert iteration["tensor_count"] == 4
    assert iteration["transfer_duration_seconds"] > 0.0
    assert iteration["transfer_bandwidth_gbps"] > 0.0
    assert [attempt["worker_rank"] for attempt in iteration["attempts"]] == [0, 1]
    assert [attempt["role"] for attempt in iteration["attempts"]] == [
        "trainer",
        "trainer",
    ]
    assert [attempt["success"] for attempt in iteration["attempts"]] == [True, True]
    if iteration["transfer_lease_discovery_supported"]:
        assert iteration["matching_lease_status_names"] == [
            "TRANSFER_LEASE_STATUS_COMPLETED",
            "TRANSFER_LEASE_STATUS_COMPLETED",
        ]
        assert iteration["missing_lease_ids"] == []


def test_live_rl_benchmark_recovers_latest_from_replica():
    torch = pytest.importorskip("torch")
    pytest.importorskip("nixl._api")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < 2:
        pytest.skip("live RL benchmark replica recovery needs at least 2 CUDA devices")

    result = run_mx_rl_transfer_benchmark(
        RlTransferBenchmarkConfig(
            server_url=os.environ[_LIVE_SERVER_ENV],
            model_name=f"mx-live-rl-benchmark-replica-{uuid.uuid4().hex[:8]}",
            tensor_count=1,
            tensor_shape=(128, 128),
            iterations=1,
            warmup_iterations=0,
            source_device_id=0,
            target_device_id=1,
            republish_received=True,
            recover_latest_from_replica=True,
        )
    )

    output = result.to_dict()
    summary = output["summary"]
    iteration = output["iterations"][0]
    recovery = iteration["recovery_receive"]

    assert summary["recovery_iterations"] == 1
    assert summary["recovery_source_roles"] == ["inference_replica"]
    assert summary["mean_recovery_transfer_duration_seconds"] > 0.0
    assert summary["mean_recovery_transfer_bandwidth_gbps"] > 0.0
    assert recovery["source_role"] == "inference_replica"
    assert recovery["transfer_duration_seconds"] > 0.0
    assert recovery["transfer_bandwidth_gbps"] > 0.0
    assert recovery["attempts"][0]["role"] == "inference_replica"
    if recovery["transfer_lease_discovery_supported"]:
        assert recovery["matching_lease_status_names"] == [
            "TRANSFER_LEASE_STATUS_COMPLETED",
        ]
        assert recovery["missing_lease_ids"] == []


def test_live_rl_benchmark_falls_back_from_failed_trainer_to_replica():
    torch = pytest.importorskip("torch")
    pytest.importorskip("nixl._api")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.device_count() < 2:
        pytest.skip("live RL benchmark trainer fallback needs at least 2 CUDA devices")

    result = run_mx_rl_transfer_benchmark(
        RlTransferBenchmarkConfig(
            server_url=os.environ[_LIVE_SERVER_ENV],
            model_name=f"mx-live-rl-benchmark-fallback-{uuid.uuid4().hex[:8]}",
            tensor_count=1,
            tensor_shape=(128, 128),
            iterations=1,
            warmup_iterations=0,
            source_device_id=0,
            target_device_id=1,
            timeout_seconds=5.0,
            republish_received=True,
            recover_latest_from_replica=True,
            fail_trainer_transfer_before_recovery=True,
        )
    )

    output = result.to_dict()
    summary = output["summary"]
    recovery = output["iterations"][0]["recovery_receive"]

    assert summary["recovery_iterations"] == 1
    assert summary["recovery_total_retries"] == 1
    assert summary["recovery_source_roles"] == ["inference_replica"]
    assert [attempt["role"] for attempt in recovery["attempts"]] == [
        "trainer",
        "inference_replica",
    ]
    assert [attempt["success"] for attempt in recovery["attempts"]] == [False, True]
    assert recovery["source_role"] == "inference_replica"
    assert recovery["retry_count"] == 1
    assert recovery["transfer_duration_seconds"] > 0.0
    assert recovery["transfer_bandwidth_gbps"] > 0.0
    if recovery["transfer_lease_discovery_supported"]:
        assert recovery["matching_lease_status_names"] == [
            "TRANSFER_LEASE_STATUS_FAILED",
            "TRANSFER_LEASE_STATUS_COMPLETED",
        ]
        assert recovery["missing_lease_ids"] == []
