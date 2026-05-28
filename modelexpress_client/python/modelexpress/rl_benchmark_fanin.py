# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dense fan-in runtime for the live RL transfer benchmark."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

import torch

from modelexpress.client import MxClient
from modelexpress.rl_benchmark import (
    RlTransferBenchmarkConfig,
    RlTransferBenchmarkIteration,
    RlTransferBenchmarkResult,
    _benchmark_iteration_from_report,
    _lease_summary_for_report,
)
from modelexpress.rl_reshard import TensorReceiveSpec
from modelexpress.rl_shape_registry import torch_dtype_from_string
from modelexpress.rl_transfer import RlNixlWeightTransfer


def run_mx_dense_fanin_transfer_benchmark(
    *,
    config: RlTransferBenchmarkConfig,
    client: MxClient,
    base_identity: Any,
) -> RlTransferBenchmarkResult:
    """Run a synthetic multi-source dense fan-in MX RL transfer benchmark."""
    publishers = [
        RlNixlWeightTransfer(
            mx_client=client,
            base_identity=base_identity,
            worker_id=f"mx-rl-bench-publisher-r{rank}-{uuid.uuid4().hex[:8]}",
            retain_latest_k=config.retain_latest_k,
            device_id=device_id,
            timeout_seconds=config.timeout_seconds,
        )
        for rank, device_id in enumerate(config.source_device_ids)
    ]
    receiver = RlNixlWeightTransfer(
        mx_client=client,
        base_identity=base_identity,
        worker_id=f"mx-rl-bench-fanin-receiver-{uuid.uuid4().hex[:8]}",
        device_id=config.target_device_id,
        timeout_seconds=config.timeout_seconds,
    )

    iterations: list[RlTransferBenchmarkIteration] = []
    try:
        total_iterations = config.warmup_iterations + config.iterations
        target_specs = _dense_fanin_target_specs(config)
        for index in range(total_iterations):
            model_version = index + 1
            warmup = index < config.warmup_iterations
            source_tensors_by_rank = [
                _make_dense_fanin_source_tensors(
                    config,
                    rank=rank,
                    device=f"cuda:{device_id}",
                    value=float(model_version),
                )
                for rank, device_id in enumerate(config.source_device_ids)
            ]
            for device_id in config.source_device_ids:
                torch.cuda.synchronize(device_id)
            publish_start = time.perf_counter()
            for rank, publisher in enumerate(publishers):
                publisher.publish_tensors(
                    source_tensors_by_rank[rank],
                    model_version=model_version,
                    worker_rank=rank,
                    source_world_size=config.dense_fanin_sources,
                    tensor_metadata=_dense_fanin_tensor_metadata(config, rank=rank),
                )
            for device_id in config.source_device_ids:
                torch.cuda.synchronize(device_id)
            publish_seconds = time.perf_counter() - publish_start

            target_tensors = _make_dense_fanin_target_tensors(config)
            torch.cuda.synchronize(config.target_device_id)
            receive_start = time.perf_counter()
            received = asyncio.run(
                receiver.receive_into_tensors(
                    target_tensors,
                    model_version=model_version,
                    receiver_rank=0,
                    target_specs=target_specs,
                )
            )
            torch.cuda.synchronize(config.target_device_id)
            receive_seconds = time.perf_counter() - receive_start

            report = receiver.last_receive_report
            if report is None or not report.success:
                raise RuntimeError(
                    "ModelExpress RL dense fan-in benchmark receive did not succeed"
                )
            if config.verify:
                _verify_dense_fanin_received(source_tensors_by_rank, dict(received))
            iterations.append(
                _benchmark_iteration_from_report(
                    index=index,
                    warmup=warmup,
                    model_version=model_version,
                    expected_bytes=config.tensor_bytes,
                    publish_seconds=publish_seconds,
                    receive_seconds=receive_seconds,
                    report=report,
                    lease_summary=_lease_summary_for_report(receiver, report),
                )
            )
    finally:
        receiver.finalize()
        for publisher in reversed(publishers):
            publisher.finalize()

    return RlTransferBenchmarkResult(
        backend="modelexpress",
        config=config,
        iterations=tuple(iterations),
    )


def _dense_fanin_shard_shape(config: RlTransferBenchmarkConfig) -> tuple[int, ...]:
    return (
        config.tensor_shape[0] // config.dense_fanin_sources,
        *config.tensor_shape[1:],
    )


def _dense_fanin_shard_offsets(
    config: RlTransferBenchmarkConfig,
    *,
    rank: int,
) -> tuple[int, ...]:
    return (
        rank * _dense_fanin_shard_shape(config)[0],
        *(0 for _dim in config.tensor_shape[1:]),
    )


def _dense_fanin_tensor_metadata(
    config: RlTransferBenchmarkConfig,
    *,
    rank: int,
) -> dict[str, dict[str, Any]]:
    return {
        f"w{index}": {
            "global_shape": list(config.tensor_shape),
            "shard_offsets": list(_dense_fanin_shard_offsets(config, rank=rank)),
        }
        for index in range(config.tensor_count)
    }


def _dense_fanin_target_specs(
    config: RlTransferBenchmarkConfig,
) -> tuple[TensorReceiveSpec, ...]:
    return tuple(
        TensorReceiveSpec(
            name=f"w{index}",
            receiver_rank=0,
            shape=config.tensor_shape,
            dtype=config.dtype,
            global_shape=config.tensor_shape,
            shard_offsets=tuple(0 for _dim in config.tensor_shape),
        )
        for index in range(config.tensor_count)
    )


def _make_dense_fanin_source_tensors(
    config: RlTransferBenchmarkConfig,
    *,
    rank: int,
    device: torch.device | str,
    value: float,
) -> dict[str, torch.Tensor]:
    dtype = torch_dtype_from_string(config.dtype)
    shard_shape = _dense_fanin_shard_shape(config)
    return {
        f"w{index}": torch.full(
            shard_shape,
            value * 100.0 + index * 10.0 + rank,
            dtype=dtype,
            device=device,
        )
        for index in range(config.tensor_count)
    }


def _make_dense_fanin_target_tensors(
    config: RlTransferBenchmarkConfig,
) -> dict[str, torch.Tensor]:
    dtype = torch_dtype_from_string(config.dtype)
    return {
        f"w{index}": torch.empty(
            config.tensor_shape,
            dtype=dtype,
            device=f"cuda:{config.target_device_id}",
        )
        for index in range(config.tensor_count)
    }


def _verify_dense_fanin_received(
    source_tensors_by_rank: list[dict[str, torch.Tensor]],
    target_tensors: dict[str, torch.Tensor],
) -> None:
    expected_names = set(source_tensors_by_rank[0])
    if set(target_tensors) != expected_names:
        raise RuntimeError("ModelExpress RL dense fan-in received unexpected tensors")
    for source_tensors in source_tensors_by_rank:
        if set(source_tensors) != expected_names:
            raise RuntimeError("ModelExpress RL dense fan-in source tensor mismatch")
    for name, target in target_tensors.items():
        offset = 0
        for source_tensors in source_tensors_by_rank:
            source = source_tensors[name].to(target.device)
            next_offset = offset + source.shape[0]
            target_slice = (
                slice(offset, next_offset),
                *[slice(None)] * (target.ndim - 1),
            )
            if not torch.equal(target[target_slice], source):
                raise RuntimeError(
                    f"ModelExpress RL dense fan-in tensor {name!r} mismatch"
                )
            offset = next_offset
