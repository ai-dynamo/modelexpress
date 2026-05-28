# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from modelexpress import p2p_pb2
from modelexpress.rl_benchmark import (
    RlTransferBenchmarkConfig,
    RlTransferBenchmarkResult,
    _build_parser,
    _benchmark_iteration_from_report,
    _benchmark_receive_from_report,
    _lease_summary_for_report,
    _parse_tensor_shape,
)
from modelexpress.rl_metadata import RlSourceRole
from modelexpress.rl_transfer import RlTransferAttempt, RlTransferReport
from modelexpress.rl_transfer_lease import (
    RlTransferLeaseInventory,
    summarize_report_leases,
)


def _lease(lease_id: str, *, status: int):
    return p2p_pb2.TransferLease(
        lease_id=lease_id,
        status=status,
        model_version=7,
    )


def test_benchmark_config_normalizes_dtype_and_counts_bytes():
    config = RlTransferBenchmarkConfig(
        server_url="localhost:8001",
        model_name="bench",
        tensor_count=3,
        tensor_shape=(2, 4),
        dtype="float16",
    )

    assert config.dtype == "torch.float16"
    assert config.tensor_bytes == 3 * 2 * 4 * 2
    assert config.to_dict()["tensor_shape"] == [2, 4]


def test_benchmark_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="tensor_count"):
        RlTransferBenchmarkConfig(
            server_url="localhost:8001",
            model_name="bench",
            tensor_count=0,
        )

    with pytest.raises(ValueError, match="iterations"):
        RlTransferBenchmarkConfig(
            server_url="localhost:8001",
            model_name="bench",
            iterations=0,
        )

    with pytest.raises(ValueError, match="device ids"):
        RlTransferBenchmarkConfig(
            server_url="localhost:8001",
            model_name="bench",
            source_device_id=-1,
        )

    with pytest.raises(ValueError, match="requires republish_received"):
        RlTransferBenchmarkConfig(
            server_url="localhost:8001",
            model_name="bench",
            recover_latest_from_replica=True,
        )

    with pytest.raises(ValueError, match="requires recover_latest_from_replica"):
        RlTransferBenchmarkConfig(
            server_url="localhost:8001",
            model_name="bench",
            republish_received=True,
            fail_trainer_transfer_before_recovery=True,
        )


def test_benchmark_iteration_serializes_report_attempts():
    report = RlTransferReport(
        requested_model_version=7,
        resolved_model_version=7,
        receiver_rank=0,
        attempts=(
            RlTransferAttempt(
                mx_source_id="source-a",
                worker_id="worker-a",
                worker_rank=0,
                role=RlSourceRole.TRAINER,
                model_version=7,
                success=False,
                error="first failed",
                lease_id="lease-source-a",
            ),
            RlTransferAttempt(
                mx_source_id="source-b",
                worker_id="worker-b",
                worker_rank=0,
                role=RlSourceRole.INFERENCE_REPLICA,
                model_version=7,
                success=True,
                lease_id="lease-source-b",
                source_status=p2p_pb2.SOURCE_STATUS_READY,
                source_updated_at=1234567890000,
                bytes_transferred=1024,
                tensor_count=2,
                duration_seconds=0.01,
            ),
        ),
    )
    lease_summary = summarize_report_leases(
        report,
        RlTransferLeaseInventory(
            target_worker_id="target-worker",
            statuses=(
                p2p_pb2.TRANSFER_LEASE_STATUS_FAILED,
                p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
            ),
            model_version=7,
            source_worker_id="worker-b",
            leases=(
                _lease(
                    "lease-source-a",
                    status=p2p_pb2.TRANSFER_LEASE_STATUS_FAILED,
                ),
                _lease(
                    "lease-source-b",
                    status=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
                ),
            ),
        ),
    )

    item = _benchmark_iteration_from_report(
        index=0,
        warmup=False,
        model_version=7,
        expected_bytes=1024,
        publish_seconds=0.2,
        receive_seconds=0.5,
        report=report,
        lease_summary=lease_summary,
    )

    assert item.retry_count == 1
    assert item.source_role == "inference_replica"
    assert item.source_worker_id == "worker-b"
    assert item.transferred_bytes == 1024
    assert item.tensor_count == 2
    assert item.attempt_lease_ids == ("lease-source-a", "lease-source-b")
    assert item.transfer_lease_discovery_supported
    assert item.report_lease_ids == ("lease-source-a", "lease-source-b")
    assert item.matching_lease_statuses == (
        p2p_pb2.TRANSFER_LEASE_STATUS_FAILED,
        p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
    )
    assert item.matching_lease_status_names == (
        "TRANSFER_LEASE_STATUS_FAILED",
        "TRANSFER_LEASE_STATUS_COMPLETED",
    )
    assert item.missing_lease_ids == ()
    assert item.non_completed_lease_statuses == (
        p2p_pb2.TRANSFER_LEASE_STATUS_FAILED,
    )
    assert item.non_completed_lease_status_names == (
        "TRANSFER_LEASE_STATUS_FAILED",
    )
    assert item.lease_summary_target_worker_id == "target-worker"
    assert item.lease_summary_model_version == 7
    assert item.lease_summary_source_worker_id == "worker-b"
    assert item.lease_summary_statuses == (
        p2p_pb2.TRANSFER_LEASE_STATUS_FAILED,
        p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
    )
    assert item.lease_summary_status_names == (
        "TRANSFER_LEASE_STATUS_FAILED",
        "TRANSFER_LEASE_STATUS_COMPLETED",
    )
    assert item.effective_bandwidth_gbps == pytest.approx(0.000016384)
    assert item.transfer_bandwidth_gbps == pytest.approx(0.0008192)
    assert item.to_dict()["lease_summary_target_worker_id"] == "target-worker"
    assert item.to_dict()["lease_summary_model_version"] == 7
    assert item.to_dict()["lease_summary_source_worker_id"] == "worker-b"
    assert item.to_dict()["lease_summary_statuses"] == [
        p2p_pb2.TRANSFER_LEASE_STATUS_FAILED,
        p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
    ]
    assert item.to_dict()["lease_summary_status_names"] == [
        "TRANSFER_LEASE_STATUS_FAILED",
        "TRANSFER_LEASE_STATUS_COMPLETED",
    ]
    assert item.to_dict()["attempts"][0]["error"] == "first failed"
    assert item.to_dict()["attempts"][0]["lease_id"] == "lease-source-a"
    assert item.to_dict()["matching_lease_statuses"] == [
        p2p_pb2.TRANSFER_LEASE_STATUS_FAILED,
        p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
    ]
    assert item.to_dict()["matching_lease_status_names"] == [
        "TRANSFER_LEASE_STATUS_FAILED",
        "TRANSFER_LEASE_STATUS_COMPLETED",
    ]
    assert item.to_dict()["non_completed_lease_status_names"] == [
        "TRANSFER_LEASE_STATUS_FAILED",
    ]
    assert item.to_dict()["attempts"][1]["source_status"] == (
        p2p_pb2.SOURCE_STATUS_READY
    )
    assert item.to_dict()["attempts"][1]["source_status_name"] == "SOURCE_STATUS_READY"
    assert item.to_dict()["attempts"][1]["source_updated_at"] == 1234567890000
    assert item.to_dict()["attempt_lease_ids"] == [
        "lease-source-a",
        "lease-source-b",
    ]
    assert item.to_dict()["transfer_bandwidth_gbps"] == pytest.approx(0.0008192)
    assert item.to_dict()["recovery_receive"] is None


def test_benchmark_lease_summary_scopes_to_report_version_and_source_worker():
    report = RlTransferReport(
        requested_model_version=None,
        resolved_model_version=7,
        receiver_rank=0,
        attempts=(
            RlTransferAttempt(
                mx_source_id="source-a",
                worker_id="worker-a",
                worker_rank=0,
                role=RlSourceRole.TRAINER,
                model_version=7,
                success=True,
                lease_id="lease-a",
            ),
        ),
    )

    class _Receiver:
        def __init__(self):
            self.kwargs = None

        def list_target_transfer_leases(self, **kwargs):
            self.kwargs = kwargs
            return RlTransferLeaseInventory(
                target_worker_id="target-worker",
                model_version=kwargs["model_version"],
                source_worker_id=kwargs["source_worker_id"],
                leases=(
                    _lease(
                        "lease-a",
                        status=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
                    ),
                ),
            )

    receiver = _Receiver()
    summary = _lease_summary_for_report(receiver, report)

    assert receiver.kwargs == {
        "model_version": 7,
        "source_worker_id": "worker-a",
    }
    assert summary.inventory.model_version == 7
    assert summary.inventory.source_worker_id == "worker-a"
    assert [lease.lease_id for lease in summary.matching_leases] == ["lease-a"]


def test_benchmark_lease_summary_keeps_multi_source_reports_unscoped():
    report = RlTransferReport(
        requested_model_version=None,
        resolved_model_version=7,
        receiver_rank=0,
        attempts=(
            RlTransferAttempt(
                mx_source_id="source-a",
                worker_id="worker-a",
                worker_rank=0,
                role=RlSourceRole.TRAINER,
                model_version=7,
                success=False,
                lease_id="lease-a",
            ),
            RlTransferAttempt(
                mx_source_id="source-b",
                worker_id="worker-b",
                worker_rank=0,
                role=RlSourceRole.INFERENCE_REPLICA,
                model_version=7,
                success=True,
                lease_id="lease-b",
            ),
        ),
    )

    class _Receiver:
        def __init__(self):
            self.kwargs = None

        def list_target_transfer_leases(self, **kwargs):
            self.kwargs = kwargs
            return RlTransferLeaseInventory(
                target_worker_id="target-worker",
                model_version=kwargs["model_version"],
                source_worker_id=kwargs["source_worker_id"],
                leases=(
                    _lease(
                        "lease-a",
                        status=p2p_pb2.TRANSFER_LEASE_STATUS_FAILED,
                    ),
                    _lease(
                        "lease-b",
                        status=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
                    ),
                ),
            )

    receiver = _Receiver()
    summary = _lease_summary_for_report(receiver, report)

    assert receiver.kwargs == {
        "model_version": 7,
        "source_worker_id": "",
    }
    assert summary.inventory.model_version == 7
    assert summary.inventory.source_worker_id == ""
    assert [lease.lease_id for lease in summary.matching_leases] == [
        "lease-a",
        "lease-b",
    ]


def test_benchmark_result_summary_ignores_warmups():
    report = RlTransferReport(
        requested_model_version=1,
        resolved_model_version=1,
        receiver_rank=0,
        attempts=(
            RlTransferAttempt(
                mx_source_id="source-a",
                worker_id="worker-a",
                worker_rank=0,
                role=RlSourceRole.TRAINER,
                model_version=1,
                success=True,
                lease_id="lease-source-a",
                bytes_transferred=100,
                tensor_count=1,
                duration_seconds=0.01,
            ),
        ),
    )
    warmup = _benchmark_iteration_from_report(
        index=0,
        warmup=True,
        model_version=1,
        expected_bytes=100,
        publish_seconds=0.1,
        receive_seconds=10.0,
        report=report,
    )
    measured = _benchmark_iteration_from_report(
        index=1,
        warmup=False,
        model_version=2,
        expected_bytes=100,
        publish_seconds=0.1,
        receive_seconds=1.0,
        report=report,
    )
    result = RlTransferBenchmarkResult(
        backend="modelexpress",
        config=RlTransferBenchmarkConfig(
            server_url="localhost:8001",
            model_name="bench",
        ),
        iterations=(warmup, measured),
    )

    summary = result.summary()

    assert summary["iterations"] == 1
    assert summary["total_transferred_bytes"] == 100
    assert summary["mean_receive_seconds"] == 1.0
    assert summary["mean_transfer_duration_seconds"] == 0.01
    assert summary["mean_transfer_bandwidth_gbps"] == pytest.approx(0.00008)
    assert summary["total_attempts"] == 1
    assert summary["successful_attempts"] == 1
    assert summary["failed_attempts"] == 0
    assert summary["attempts_with_lease_ids"] == 1
    assert not summary["lease_discovery_supported"]
    assert summary["iterations_with_missing_lease_ids"] == 0
    assert summary["non_completed_matching_leases"] == 0
    assert summary["recovery_iterations"] == 0


def test_benchmark_result_summary_reports_replica_recovery_receive():
    primary_report = RlTransferReport(
        requested_model_version=3,
        resolved_model_version=3,
        receiver_rank=0,
        attempts=(
            RlTransferAttempt(
                mx_source_id="trainer-source",
                worker_id="trainer-worker",
                worker_rank=0,
                role=RlSourceRole.TRAINER,
                model_version=3,
                success=True,
                bytes_transferred=100,
                tensor_count=1,
                duration_seconds=0.01,
            ),
        ),
    )
    recovery_report = RlTransferReport(
        requested_model_version=None,
        resolved_model_version=3,
        receiver_rank=0,
        attempts=(
            RlTransferAttempt(
                mx_source_id="trainer-source",
                worker_id="trainer-worker",
                worker_rank=0,
                role=RlSourceRole.TRAINER,
                model_version=3,
                success=False,
                error="trainer unavailable",
                lease_id="lease-trainer",
            ),
            RlTransferAttempt(
                mx_source_id="replica-source",
                worker_id="replica-worker",
                worker_rank=0,
                role=RlSourceRole.INFERENCE_REPLICA,
                model_version=3,
                success=True,
                bytes_transferred=100,
                tensor_count=1,
                duration_seconds=0.02,
            ),
        ),
    )
    recovery_receive = _benchmark_receive_from_report(
        receive_seconds=2.0,
        report=recovery_report,
    )
    iteration = _benchmark_iteration_from_report(
        index=0,
        warmup=False,
        model_version=3,
        expected_bytes=100,
        publish_seconds=0.1,
        receive_seconds=1.0,
        report=primary_report,
        recovery_receive=recovery_receive,
    )
    result = RlTransferBenchmarkResult(
        backend="modelexpress",
        config=RlTransferBenchmarkConfig(
            server_url="localhost:8001",
            model_name="bench",
            republish_received=True,
            recover_latest_from_replica=True,
        ),
        iterations=(iteration,),
    )

    output = result.to_dict()
    summary = output["summary"]
    recovery_output = output["iterations"][0]["recovery_receive"]

    assert summary["recovery_iterations"] == 1
    assert summary["mean_recovery_receive_seconds"] == 2.0
    assert summary["mean_recovery_transfer_duration_seconds"] == 0.02
    assert summary["mean_recovery_transfer_bandwidth_gbps"] == pytest.approx(0.00004)
    assert summary["recovery_total_retries"] == 1
    assert summary["recovery_source_roles"] == ["inference_replica"]
    assert recovery_output["source_role"] == "inference_replica"
    assert recovery_output["transferred_bytes"] == 100
    assert [attempt["role"] for attempt in recovery_output["attempts"]] == [
        "trainer",
        "inference_replica",
    ]
    assert [attempt["success"] for attempt in recovery_output["attempts"]] == [
        False,
        True,
    ]


def test_parse_tensor_shape_accepts_comma_or_x_separators():
    assert _parse_tensor_shape("2,3,4") == (2, 3, 4)
    assert _parse_tensor_shape("2x3") == (2, 3)

    with pytest.raises(argparse.ArgumentTypeError, match="invalid tensor shape"):
        _parse_tensor_shape("2,bad")


def test_parser_accepts_output_json_path(tmp_path):
    args = _build_parser().parse_args(
        [
            "--republish-received",
            "--recover-latest-from-replica",
            "--fail-trainer-transfer-before-recovery",
            "--output-json",
            str(tmp_path / "result.json"),
        ]
    )

    assert args.output_json == tmp_path / "result.json"
    assert args.republish_received is True
    assert args.recover_latest_from_replica is True
    assert args.fail_trainer_transfer_before_recovery is True
