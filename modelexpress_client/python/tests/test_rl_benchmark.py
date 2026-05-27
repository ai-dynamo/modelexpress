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
    _parse_tensor_shape,
)
from modelexpress.rl_metadata import RlSourceRole
from modelexpress.rl_transfer import RlTransferAttempt, RlTransferReport


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
            ),
            RlTransferAttempt(
                mx_source_id="source-b",
                worker_id="worker-b",
                worker_rank=0,
                role=RlSourceRole.INFERENCE_REPLICA,
                model_version=7,
                success=True,
                source_status=p2p_pb2.SOURCE_STATUS_READY,
                source_updated_at=1234567890000,
                bytes_transferred=1024,
                tensor_count=2,
                duration_seconds=0.01,
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
    )

    assert item.retry_count == 1
    assert item.source_role == "inference_replica"
    assert item.source_worker_id == "worker-b"
    assert item.transferred_bytes == 1024
    assert item.tensor_count == 2
    assert item.effective_bandwidth_gbps == pytest.approx(0.000016384)
    assert item.to_dict()["attempts"][0]["error"] == "first failed"
    assert item.to_dict()["attempts"][1]["source_status"] == (
        p2p_pb2.SOURCE_STATUS_READY
    )
    assert item.to_dict()["attempts"][1]["source_updated_at"] == 1234567890000


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


def test_parse_tensor_shape_accepts_comma_or_x_separators():
    assert _parse_tensor_shape("2,3,4") == (2, 3, 4)
    assert _parse_tensor_shape("2x3") == (2, 3)

    with pytest.raises(argparse.ArgumentTypeError, match="invalid tensor shape"):
        _parse_tensor_shape("2,bad")


def test_parser_accepts_output_json_path(tmp_path):
    args = _build_parser().parse_args(["--output-json", str(tmp_path / "result.json")])

    assert args.output_json == tmp_path / "result.json"
