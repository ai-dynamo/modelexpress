# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live RL transfer benchmark harness for ModelExpress."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from modelexpress.rl_metadata import RlSourceRole
from modelexpress.rl_shape_registry import torch_dtype_from_string
from modelexpress.rl_transfer import (
    RlNixlWeightTransfer,
    RlTransferAttempt,
    RlTransferReport,
    build_rl_base_identity,
)
from modelexpress.rl_transfer_lease import (
    RlTransferLeaseInventory,
    RlTransferLeaseReportSummary,
    summarize_report_leases,
)


@dataclass(frozen=True)
class RlTransferBenchmarkConfig:
    """Inputs for one live ModelExpress RL transfer benchmark run."""

    server_url: str
    model_name: str
    tensor_count: int = 4
    tensor_shape: tuple[int, ...] = (1024, 1024)
    dtype: str = "torch.float32"
    iterations: int = 3
    warmup_iterations: int = 1
    source_device_id: int = 0
    target_device_id: int = 0
    timeout_seconds: float = 60.0
    retain_latest_k: int = 1
    verify: bool = True

    def __post_init__(self) -> None:
        tensor_shape = tuple(int(dim) for dim in self.tensor_shape)
        if self.tensor_count <= 0:
            raise ValueError("tensor_count must be positive")
        if not tensor_shape or any(dim <= 0 for dim in tensor_shape):
            raise ValueError("tensor_shape dimensions must be positive")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if self.retain_latest_k <= 0:
            raise ValueError("retain_latest_k must be positive")
        if self.source_device_id < 0 or self.target_device_id < 0:
            raise ValueError("CUDA device ids must be non-negative")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive")
        object.__setattr__(self, "tensor_shape", tensor_shape)
        object.__setattr__(self, "dtype", str(torch_dtype_from_string(self.dtype)))

    @property
    def tensor_bytes(self) -> int:
        numel = 1
        for dim in self.tensor_shape:
            numel *= dim
        dtype = torch_dtype_from_string(self.dtype)
        return self.tensor_count * numel * torch.empty((), dtype=dtype).element_size()

    def to_dict(self) -> dict[str, Any]:
        return {
            "server_url": self.server_url,
            "model_name": self.model_name,
            "tensor_count": self.tensor_count,
            "tensor_shape": list(self.tensor_shape),
            "dtype": self.dtype,
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "source_device_id": self.source_device_id,
            "target_device_id": self.target_device_id,
            "timeout_seconds": self.timeout_seconds,
            "retain_latest_k": self.retain_latest_k,
            "verify": self.verify,
            "tensor_bytes": self.tensor_bytes,
        }


@dataclass(frozen=True)
class RlTransferBenchmarkIteration:
    """Measurements for one publish plus receive benchmark iteration."""

    index: int
    warmup: bool
    model_version: int
    tensor_count: int
    expected_bytes: int
    transferred_bytes: int
    publish_seconds: float
    receive_seconds: float
    transfer_duration_seconds: float
    effective_bandwidth_gbps: float
    transfer_bandwidth_gbps: float
    retry_count: int
    attempt_lease_ids: tuple[str, ...]
    transfer_lease_discovery_supported: bool
    report_lease_ids: tuple[str, ...]
    matching_lease_statuses: tuple[int, ...]
    matching_lease_status_names: tuple[str, ...]
    missing_lease_ids: tuple[str, ...]
    non_completed_lease_statuses: tuple[int, ...]
    non_completed_lease_status_names: tuple[str, ...]
    lease_summary_target_worker_id: str
    lease_summary_model_version: int | None
    lease_summary_source_worker_id: str
    lease_summary_statuses: tuple[int, ...] | None
    lease_summary_status_names: tuple[str, ...] | None
    source_role: str | None
    source_worker_id: str | None
    attempts: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "warmup": self.warmup,
            "model_version": self.model_version,
            "tensor_count": self.tensor_count,
            "expected_bytes": self.expected_bytes,
            "transferred_bytes": self.transferred_bytes,
            "publish_seconds": self.publish_seconds,
            "receive_seconds": self.receive_seconds,
            "transfer_duration_seconds": self.transfer_duration_seconds,
            "effective_bandwidth_gbps": self.effective_bandwidth_gbps,
            "transfer_bandwidth_gbps": self.transfer_bandwidth_gbps,
            "retry_count": self.retry_count,
            "attempt_lease_ids": list(self.attempt_lease_ids),
            "transfer_lease_discovery_supported": (
                self.transfer_lease_discovery_supported
            ),
            "report_lease_ids": list(self.report_lease_ids),
            "matching_lease_statuses": list(self.matching_lease_statuses),
            "matching_lease_status_names": list(self.matching_lease_status_names),
            "missing_lease_ids": list(self.missing_lease_ids),
            "non_completed_lease_statuses": list(self.non_completed_lease_statuses),
            "non_completed_lease_status_names": list(
                self.non_completed_lease_status_names
            ),
            "lease_summary_target_worker_id": self.lease_summary_target_worker_id,
            "lease_summary_model_version": self.lease_summary_model_version,
            "lease_summary_source_worker_id": self.lease_summary_source_worker_id,
            "lease_summary_statuses": (
                list(self.lease_summary_statuses)
                if self.lease_summary_statuses is not None
                else None
            ),
            "lease_summary_status_names": (
                list(self.lease_summary_status_names)
                if self.lease_summary_status_names is not None
                else None
            ),
            "source_role": self.source_role,
            "source_worker_id": self.source_worker_id,
            "attempts": list(self.attempts),
        }


@dataclass(frozen=True)
class RlTransferBenchmarkResult:
    """Serializable output for MX-vs-baseline RL transfer comparisons."""

    backend: str
    config: RlTransferBenchmarkConfig
    iterations: tuple[RlTransferBenchmarkIteration, ...]

    @property
    def measured_iterations(self) -> tuple[RlTransferBenchmarkIteration, ...]:
        return tuple(item for item in self.iterations if not item.warmup)

    def summary(self) -> dict[str, Any]:
        measured = self.measured_iterations
        receive_seconds = [item.receive_seconds for item in measured]
        effective_bandwidth = [item.effective_bandwidth_gbps for item in measured]
        transfer_bandwidth = [item.transfer_bandwidth_gbps for item in measured]
        transfer_seconds = [item.transfer_duration_seconds for item in measured]
        attempts = [
            attempt
            for item in measured
            for attempt in item.attempts
        ]
        return {
            "iterations": len(measured),
            "total_transferred_bytes": sum(item.transferred_bytes for item in measured),
            "mean_receive_seconds": statistics.fmean(receive_seconds),
            "median_receive_seconds": statistics.median(receive_seconds),
            "min_receive_seconds": min(receive_seconds),
            "max_receive_seconds": max(receive_seconds),
            "mean_transfer_duration_seconds": statistics.fmean(transfer_seconds),
            "median_transfer_duration_seconds": statistics.median(transfer_seconds),
            "min_transfer_duration_seconds": min(transfer_seconds),
            "max_transfer_duration_seconds": max(transfer_seconds),
            "mean_effective_bandwidth_gbps": statistics.fmean(effective_bandwidth),
            "median_effective_bandwidth_gbps": statistics.median(effective_bandwidth),
            "max_effective_bandwidth_gbps": max(effective_bandwidth),
            "mean_transfer_bandwidth_gbps": statistics.fmean(transfer_bandwidth),
            "median_transfer_bandwidth_gbps": statistics.median(transfer_bandwidth),
            "max_transfer_bandwidth_gbps": max(transfer_bandwidth),
            "total_retries": sum(item.retry_count for item in measured),
            "total_attempts": len(attempts),
            "successful_attempts": sum(1 for attempt in attempts if attempt["success"]),
            "failed_attempts": sum(1 for attempt in attempts if not attempt["success"]),
            "attempts_with_lease_ids": sum(
                1 for attempt in attempts if attempt["lease_id"]
            ),
            "lease_discovery_supported": all(
                item.transfer_lease_discovery_supported for item in measured
            ),
            "iterations_with_missing_lease_ids": sum(
                1 for item in measured if item.missing_lease_ids
            ),
            "non_completed_matching_leases": sum(
                len(item.non_completed_lease_statuses) for item in measured
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "config": self.config.to_dict(),
            "summary": self.summary(),
            "iterations": [item.to_dict() for item in self.iterations],
        }


def run_mx_rl_transfer_benchmark(
    config: RlTransferBenchmarkConfig,
) -> RlTransferBenchmarkResult:
    """Run a live synthetic trainer-to-receiver MX RL transfer benchmark."""
    _require_live_dependencies(config)
    client = MxClient(server_url=config.server_url)
    base_identity = build_rl_base_identity(
        model_name=config.model_name,
        mx_version="0.3.0",
        backend_framework="vllm",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=0,
        dtype=config.dtype.removeprefix("torch."),
        quantization="",
        revision="",
    )
    publisher = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"mx-rl-bench-publisher-{uuid.uuid4().hex[:8]}",
        retain_latest_k=config.retain_latest_k,
        device_id=config.source_device_id,
        timeout_seconds=config.timeout_seconds,
    )
    receiver = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"mx-rl-bench-receiver-{uuid.uuid4().hex[:8]}",
        device_id=config.target_device_id,
        timeout_seconds=config.timeout_seconds,
    )

    iterations: list[RlTransferBenchmarkIteration] = []
    try:
        total_iterations = config.warmup_iterations + config.iterations
        for index in range(total_iterations):
            model_version = index + 1
            warmup = index < config.warmup_iterations
            source_tensors = _make_tensors(
                config,
                device=f"cuda:{config.source_device_id}",
                value=float(model_version),
            )
            torch.cuda.synchronize(config.source_device_id)
            publish_start = time.perf_counter()
            publisher.publish_tensors(source_tensors, model_version=model_version)
            torch.cuda.synchronize(config.source_device_id)
            publish_seconds = time.perf_counter() - publish_start

            target_tensors = {
                name: torch.empty_like(tensor, device=f"cuda:{config.target_device_id}")
                for name, tensor in source_tensors.items()
            }
            torch.cuda.synchronize(config.target_device_id)
            receive_start = time.perf_counter()
            received = asyncio.run(
                receiver.receive_into_tensors(
                    target_tensors,
                    model_version=model_version,
                    receiver_rank=0,
                )
            )
            torch.cuda.synchronize(config.target_device_id)
            receive_seconds = time.perf_counter() - receive_start

            report = receiver.last_receive_report
            if report is None or not report.success:
                raise RuntimeError("ModelExpress RL benchmark receive did not succeed")
            if config.verify:
                _verify_received(source_tensors, dict(received))
            lease_summary = _lease_summary_for_report(receiver, report)
            iterations.append(
                _benchmark_iteration_from_report(
                    index=index,
                    warmup=warmup,
                    model_version=model_version,
                    expected_bytes=config.tensor_bytes,
                    publish_seconds=publish_seconds,
                    receive_seconds=receive_seconds,
                    report=report,
                    lease_summary=lease_summary,
                )
            )
    finally:
        receiver.finalize()
        publisher.finalize()
        client.close()

    return RlTransferBenchmarkResult(
        backend="modelexpress",
        config=config,
        iterations=tuple(iterations),
    )


def _lease_summary_for_report(
    receiver: RlNixlWeightTransfer,
    report: RlTransferReport,
) -> RlTransferLeaseReportSummary:
    return summarize_report_leases(
        report,
        receiver.list_target_transfer_leases(
            model_version=report.resolved_model_version,
            source_worker_id=report.single_lease_source_worker_id or "",
        ),
    )


def _benchmark_iteration_from_report(
    *,
    index: int,
    warmup: bool,
    model_version: int,
    expected_bytes: int,
    publish_seconds: float,
    receive_seconds: float,
    report: RlTransferReport,
    lease_summary: RlTransferLeaseReportSummary | None = None,
) -> RlTransferBenchmarkIteration:
    successful_attempts = [attempt for attempt in report.attempts if attempt.success]
    transferred_bytes = sum(attempt.bytes_transferred for attempt in successful_attempts)
    tensor_count = sum(attempt.tensor_count for attempt in successful_attempts)
    transfer_duration_seconds = sum(
        attempt.duration_seconds for attempt in successful_attempts
    )
    if lease_summary is None:
        lease_summary = summarize_report_leases(
            report,
            RlTransferLeaseInventory(
                target_worker_id="",
                discovery_supported=False,
            ),
        )
    lease_discovery_supported = lease_summary.inventory.discovery_supported
    return RlTransferBenchmarkIteration(
        index=index,
        warmup=warmup,
        model_version=model_version,
        tensor_count=tensor_count,
        expected_bytes=expected_bytes,
        transferred_bytes=transferred_bytes,
        publish_seconds=publish_seconds,
        receive_seconds=receive_seconds,
        transfer_duration_seconds=transfer_duration_seconds,
        effective_bandwidth_gbps=_bandwidth_gbps(transferred_bytes, receive_seconds),
        transfer_bandwidth_gbps=_bandwidth_gbps(
            transferred_bytes,
            transfer_duration_seconds,
        ),
        retry_count=report.retry_count,
        attempt_lease_ids=tuple(
            attempt.lease_id for attempt in report.attempts if attempt.lease_id
        ),
        transfer_lease_discovery_supported=lease_discovery_supported,
        report_lease_ids=lease_summary.report_lease_ids,
        matching_lease_statuses=tuple(
            int(lease.status) for lease in lease_summary.matching_leases
        ),
        matching_lease_status_names=tuple(
            _transfer_lease_status_name(lease.status)
            for lease in lease_summary.matching_leases
        ),
        missing_lease_ids=lease_summary.missing_lease_ids,
        non_completed_lease_statuses=tuple(
            int(lease.status) for lease in lease_summary.non_completed_matching_leases
        ),
        non_completed_lease_status_names=tuple(
            _transfer_lease_status_name(lease.status)
            for lease in lease_summary.non_completed_matching_leases
        ),
        lease_summary_target_worker_id=lease_summary.inventory.target_worker_id,
        lease_summary_model_version=lease_summary.inventory.model_version,
        lease_summary_source_worker_id=lease_summary.inventory.source_worker_id,
        lease_summary_statuses=lease_summary.inventory.statuses,
        lease_summary_status_names=(
            tuple(
                _transfer_lease_status_name(status)
                for status in lease_summary.inventory.statuses
            )
            if lease_summary.inventory.statuses is not None
            else None
        ),
        source_role=_role_value(report.source_role),
        source_worker_id=report.source_worker_id,
        attempts=tuple(_attempt_to_dict(attempt) for attempt in report.attempts),
    )


def _attempt_to_dict(attempt: RlTransferAttempt) -> dict[str, Any]:
    return {
        "mx_source_id": attempt.mx_source_id,
        "worker_id": attempt.worker_id,
        "worker_rank": attempt.worker_rank,
        "role": _role_value(attempt.role),
        "model_version": attempt.model_version,
        "success": attempt.success,
        "error": attempt.error,
        "lease_id": attempt.lease_id,
        "source_status": attempt.source_status,
        "source_status_name": _source_status_name(attempt.source_status),
        "source_updated_at": attempt.source_updated_at,
        "bytes_transferred": attempt.bytes_transferred,
        "tensor_count": attempt.tensor_count,
        "duration_seconds": attempt.duration_seconds,
    }


def _role_value(role: RlSourceRole | None) -> str | None:
    return role.value if role is not None else None


def _source_status_name(status: int) -> str:
    return _enum_name(p2p_pb2.SourceStatus, int(status))


def _transfer_lease_status_name(status: int) -> str:
    return _enum_name(p2p_pb2.TransferLeaseStatus, int(status))


def _enum_name(enum_type: Any, value: int) -> str:
    try:
        return str(enum_type.Name(value))
    except ValueError:
        return f"UNKNOWN_{value}"


def _bandwidth_gbps(bytes_transferred: int, seconds: float) -> float:
    if seconds <= 0.0:
        return 0.0
    return bytes_transferred * 8.0 / seconds / 1e9


def _make_tensors(
    config: RlTransferBenchmarkConfig,
    *,
    device: torch.device | str,
    value: float,
) -> dict[str, torch.Tensor]:
    dtype = torch_dtype_from_string(config.dtype)
    return {
        f"w{index}": torch.full(
            config.tensor_shape,
            value + index,
            dtype=dtype,
            device=device,
        )
        for index in range(config.tensor_count)
    }


def _verify_received(
    source_tensors: dict[str, torch.Tensor],
    target_tensors: dict[str, torch.Tensor],
) -> None:
    if set(source_tensors) != set(target_tensors):
        raise RuntimeError("ModelExpress RL benchmark received unexpected tensors")
    for name, source in source_tensors.items():
        if not torch.equal(target_tensors[name], source.to(target_tensors[name].device)):
            raise RuntimeError(f"ModelExpress RL benchmark tensor {name!r} mismatch")


def _require_live_dependencies(config: RlTransferBenchmarkConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the ModelExpress RL benchmark")
    if config.source_device_id >= torch.cuda.device_count():
        raise RuntimeError(f"source device {config.source_device_id} is not available")
    if config.target_device_id >= torch.cuda.device_count():
        raise RuntimeError(f"target device {config.target_device_id} is not available")
    try:
        import nixl._api  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("NIXL is required for the ModelExpress RL benchmark") from exc


def _parse_tensor_shape(value: str) -> tuple[int, ...]:
    parts = value.replace("x", ",").split(",")
    try:
        shape = tuple(int(part.strip()) for part in parts if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid tensor shape {value!r}; expected comma or x separated integers"
        ) from exc
    if not shape or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError(
            f"invalid tensor shape {value!r}; dimensions must be positive"
        )
    return shape


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-url",
        default=os.environ.get("MX_LIVE_SERVER_URL", "localhost:8001"),
    )
    parser.add_argument(
        "--model-name",
        default=f"mx-rl-benchmark-{uuid.uuid4().hex[:8]}",
    )
    parser.add_argument("--tensor-count", type=int, default=4)
    parser.add_argument("--tensor-shape", type=_parse_tensor_shape, default=(1024, 1024))
    parser.add_argument("--dtype", default="torch.float32")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--source-device-id", type=int, default=0)
    parser.add_argument("--target-device-id", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--retain-latest-k", type=int, default=1)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write the structured benchmark result to this file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_mx_rl_transfer_benchmark(
        RlTransferBenchmarkConfig(
            server_url=args.server_url,
            model_name=args.model_name,
            tensor_count=args.tensor_count,
            tensor_shape=args.tensor_shape,
            dtype=args.dtype,
            iterations=args.iterations,
            warmup_iterations=args.warmup_iterations,
            source_device_id=args.source_device_id,
            target_device_id=args.target_device_id,
            timeout_seconds=args.timeout_seconds,
            retain_latest_k=args.retain_latest_k,
            verify=not args.no_verify,
        )
    )
    output = json.dumps(result.to_dict(), indent=2, sort_keys=True)
    if args.output_json is None:
        print(output)
    else:
        args.output_json.write_text(output + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
