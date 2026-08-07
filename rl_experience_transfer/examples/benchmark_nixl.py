# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one byte-exact NIXL transfer and emit machine-readable measurements."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from rlxfer.model import SCHEMA_VERSION, ExperienceBatch, ExperienceMetadata, TensorPayload
from rlxfer.observability import InMemoryMetrics
from rlxfer.serialization import JsonExperienceSerializer
from rlxfer.transports.nixl import NixlTransport


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unavailable"


def run(device: str) -> dict[str, Any]:
    """Run the measured transfer on ``device`` and return its result."""

    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    tokens = torch.arange(4096, dtype=torch.int64, device=device).reshape(16, 256)
    rewards = torch.linspace(-1, 1, 4096, dtype=torch.float32, device=device).reshape(16, 256)
    batch = ExperienceBatch(
        metadata=ExperienceMetadata("benchmark", "native-torch", torch.__version__),
        tensors={
            "tokens": TensorPayload(tokens),
            "per_token_rewards": TensorPayload(rewards),
        },
    )
    serializer = JsonExperienceSerializer(
        cpu_staging=False,
        preserve_device=device != "cpu",
    )
    producer_metrics = InMemoryMetrics()
    consumer_metrics = InMemoryMetrics()
    with tempfile.TemporaryDirectory(prefix="rlxfer-nixl-") as control_path:
        producer = NixlTransport(
            control_path,
            agent_name=f"benchmark-producer-{device.replace(':', '-')}",
            metrics=producer_metrics,
        )
        consumer = NixlTransport(
            control_path,
            agent_name=f"benchmark-consumer-{device.replace(':', '-')}",
            target_device=device,
            metrics=consumer_metrics,
        )
        end_to_end_started = time.perf_counter()
        serialize_started = time.perf_counter()
        payload = serializer.serialize(batch)
        serialization_seconds = time.perf_counter() - serialize_started
        publish_started = time.perf_counter()
        receipt = producer.publish(
            payload,
            experience_id=batch.experience_id,
            idempotency_key=f"benchmark-{device}",
        )
        publish_seconds = time.perf_counter() - publish_started
        receive_started = time.perf_counter()
        delivery = consumer.receive(10.0)
        receive_seconds = time.perf_counter() - receive_started
        if delivery is None:
            raise RuntimeError("NIXL benchmark receive timed out")
        deserialize_started = time.perf_counter()
        restored = serializer.deserialize(delivery.payload)
        deserialization_seconds = time.perf_counter() - deserialize_started
        if not torch.equal(restored.tensors["tokens"].data, tokens):
            raise RuntimeError("token tensor failed byte-exact validation")
        if not torch.equal(restored.tensors["per_token_rewards"].data, rewards):
            raise RuntimeError("reward tensor failed byte-exact validation")
        ack_started = time.perf_counter()
        consumer.ack(delivery.token)
        receipt_result = receipt.wait(2.0)
        acknowledgement_seconds = time.perf_counter() - ack_started
        end_to_end_seconds = time.perf_counter() - end_to_end_started
        producer.close()
        consumer.close()
    producer_counters, producer_observations = producer_metrics.snapshot()
    consumer_counters, consumer_observations = consumer_metrics.snapshot()
    return {
        "result": "PASSED",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "nixl": _version("nixl"),
        "nixl_backend_wheel": _version("nixl-cu12"),
        "transport": "nixl:ucx",
        "device": device,
        "schema_version": SCHEMA_VERSION,
        "experience_id": batch.experience_id,
        "trajectory_count": 0,
        "tensor_count": len(payload.buffers),
        "tensor_catalog": [
            {"name": segment.name, "shape": segment.shape, "dtype": segment.dtype}
            for segment in payload.buffers
        ],
        "metadata_bytes": len(payload.metadata),
        "tensor_bytes": sum(segment.nbytes for segment in payload.buffers),
        "total_bytes": payload.nbytes,
        "serialization_seconds": serialization_seconds,
        "registration_and_publish_seconds": publish_seconds,
        "receive_and_transfer_seconds": receive_seconds,
        "deserialization_seconds": deserialization_seconds,
        "acknowledgement_seconds": acknowledgement_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "acknowledgement": receipt_result.state.value,
        "byte_exact": True,
        "producer_metrics": producer_counters | producer_observations,
        "consumer_metrics": consumer_counters | consumer_observations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(arguments.device)
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
