# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests against real NIXL Python bindings; no mocked NIXL objects."""

from __future__ import annotations

import json
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from rlxfer.errors import DeliveryError
from rlxfer.model import ExperienceBatch, ExperienceMetadata, TensorPayload
from rlxfer.serialization import AuthenticatedExperienceSerializer, JsonExperienceSerializer
from rlxfer.transport import ReceiptState
from rlxfer.transports.nixl import NixlTransport

pytest.importorskip("nixl")
torch: Any = pytest.importorskip("torch")


def _cpu_batch() -> ExperienceBatch:
    return ExperienceBatch(
        metadata=ExperienceMetadata("nixl-producer", "test", "1"),
        tensors={
            "tokens": TensorPayload(np.arange(40, dtype=np.int64).reshape(5, 8)),
            "advantages": TensorPayload(np.linspace(-1, 1, 40, dtype=np.float32)),
        },
    )


def _consume_nixl(path: str, output: Any) -> None:
    started = time.perf_counter()
    consumer = NixlTransport(path, agent_name="nixl-consumer-process")
    delivery = consumer.receive(10.0)
    if delivery is None:
        output.put({"error": "receive timed out"})
        return
    restored = JsonExperienceSerializer().deserialize(delivery.payload)
    tokens = cast(np.ndarray[Any, Any], restored.tensors["tokens"].data)
    advantages = cast(np.ndarray[Any, Any], restored.tensors["advantages"].data)
    consumer.ack(delivery.token)
    consumer.close()
    output.put(
        {
            "tokens": tokens.tolist(),
            "advantages": advantages.tobytes(),
            "elapsed": time.perf_counter() - started,
        }
    )


@pytest.mark.nixl
@pytest.mark.integration
@pytest.mark.multi_process
def test_real_nixl_multiprocess_cpu_scatter_gather(tmp_path: Path) -> None:
    batch = _cpu_batch()
    serializer = JsonExperienceSerializer()
    producer = NixlTransport(tmp_path, agent_name="nixl-producer-process")
    started = time.perf_counter()
    receipt = producer.publish(
        serializer.serialize(batch),
        experience_id=batch.experience_id,
        idempotency_key="real-nixl-cpu",
    )

    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(target=_consume_nixl, args=(str(tmp_path), output))
    process.start()
    process.join(20.0)
    assert process.exitcode == 0
    result = output.get(timeout=1.0)
    assert "error" not in result
    assert result["tokens"] == np.arange(40, dtype=np.int64).reshape(5, 8).tolist()
    assert result["advantages"] == np.linspace(-1, 1, 40, dtype=np.float32).tobytes()
    assert receipt.wait(2.0).state is ReceiptState.ACKED
    assert time.perf_counter() - started < 20.0
    producer.close()


@pytest.mark.nixl
@pytest.mark.integration
def test_real_nixl_rejects_bad_descriptor_and_cleans_up(tmp_path: Path) -> None:
    batch = _cpu_batch()
    producer = NixlTransport(tmp_path, agent_name="nixl-failure-producer")
    consumer = NixlTransport(tmp_path, agent_name="nixl-failure-consumer")
    receipt = producer.publish(
        JsonExperienceSerializer().serialize(batch),
        experience_id=batch.experience_id,
        idempotency_key="real-nixl-failure",
        max_retries=0,
    )
    pending = next((tmp_path / "control" / "pending").iterdir())
    control_path = pending / "metadata.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    control["buffers"][0]["region_size"] += 1
    control_path.write_text(json.dumps(control), encoding="utf-8")

    with pytest.raises(DeliveryError, match="descriptor"):
        consumer.receive(1.0)
    assert receipt.wait(1.0).state is ReceiptState.NACKED
    assert producer.health().queue_depth == 0
    consumer.close()
    producer.close()


@pytest.mark.nixl
@pytest.mark.integration
def test_real_nixl_cancel_releases_pending_source(tmp_path: Path) -> None:
    batch = _cpu_batch()
    producer = NixlTransport(tmp_path, agent_name="nixl-cancel-producer")
    receipt = producer.publish(
        JsonExperienceSerializer().serialize(batch),
        experience_id=batch.experience_id,
        idempotency_key="real-nixl-cancel",
    )

    producer.cancel(receipt.receipt_id, "caller cancelled")

    result = receipt.wait(1.0)
    assert (result.state, result.reason) == (
        ReceiptState.CANCELLED,
        "caller cancelled",
    )
    assert producer.health().queue_depth == 0
    producer.close()


@pytest.mark.nixl
@pytest.mark.integration
def test_real_nixl_concurrent_duplicate_publish_is_idempotent(tmp_path: Path) -> None:
    batch = _cpu_batch()
    producer = NixlTransport(tmp_path, agent_name="nixl-concurrent-producer")
    consumer = NixlTransport(tmp_path, agent_name="nixl-concurrent-consumer")
    payload = JsonExperienceSerializer().serialize(batch)

    def publish(_: int) -> Any:
        return producer.publish(
            payload,
            experience_id=batch.experience_id,
            idempotency_key="real-nixl-concurrent",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(publish, range(2)))

    assert receipts[0].receipt_id == receipts[1].receipt_id
    delivery = consumer.receive(2.0)
    assert delivery is not None
    consumer.ack(delivery.token)
    assert receipts[0].wait(1.0).state is ReceiptState.ACKED
    assert receipts[1].wait(1.0).state is ReceiptState.ACKED
    assert consumer.receive(0.0) is None
    assert not producer.capabilities.zero_copy
    consumer.close()
    producer.close()


@pytest.mark.nixl
@pytest.mark.integration
def test_real_nixl_transfers_authenticated_payload(tmp_path: Path) -> None:
    batch = _cpu_batch()
    serializer = AuthenticatedExperienceSerializer({"test": b"k" * 32}, signing_key_id="test")
    producer = NixlTransport(tmp_path, agent_name="nixl-auth-producer")
    consumer = NixlTransport(tmp_path, agent_name="nixl-auth-consumer")
    receipt = producer.publish(
        serializer.serialize(batch),
        experience_id=batch.experience_id,
        idempotency_key="real-nixl-auth",
    )

    delivery = consumer.receive(2.0)
    assert delivery is not None
    assert serializer.deserialize(delivery.payload).experience_id == batch.experience_id
    consumer.ack(delivery.token)
    assert receipt.wait(1.0).state is ReceiptState.ACKED
    consumer.close()
    producer.close()


@pytest.mark.nixl
@pytest.mark.integration
@pytest.mark.requires_gpu
def test_real_nixl_cuda_buffer(tmp_path: Path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tokens = torch.arange(64, dtype=torch.int64, device="cuda:0").reshape(8, 8)
    batch = ExperienceBatch(
        metadata=ExperienceMetadata("nixl-cuda", "test", "1"),
        tensors={"tokens": TensorPayload(tokens)},
    )
    serializer = JsonExperienceSerializer(cpu_staging=False, preserve_device=True)
    producer = NixlTransport(tmp_path, agent_name="nixl-cuda-producer")
    consumer = NixlTransport(tmp_path, agent_name="nixl-cuda-consumer")
    receipt = producer.publish(
        serializer.serialize(batch),
        experience_id=batch.experience_id,
        idempotency_key="real-nixl-cuda",
    )
    delivery = consumer.receive(5.0)
    assert delivery is not None
    restored = serializer.deserialize(delivery.payload)
    received = restored.tensors["tokens"].data
    assert isinstance(received, torch.Tensor)
    assert received.device.type == "cuda"
    assert torch.equal(received, tokens)
    consumer.ack(delivery.token)
    assert receipt.wait(1.0).state is ReceiptState.ACKED
    consumer.close()
    producer.close()


@pytest.mark.nixl
@pytest.mark.integration
@pytest.mark.requires_gpu
def test_real_nixl_multi_gpu_batch(tmp_path: Path) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices are unavailable")
    first = torch.arange(32, dtype=torch.int64, device="cuda:0")
    second = torch.linspace(-1, 1, 32, dtype=torch.float32, device="cuda:1")
    batch = ExperienceBatch(
        metadata=ExperienceMetadata("nixl-multi-gpu", "test", "1"),
        tensors={"tokens": TensorPayload(first), "advantages": TensorPayload(second)},
    )
    serializer = JsonExperienceSerializer(cpu_staging=False, preserve_device=True)
    producer = NixlTransport(tmp_path, agent_name="nixl-multi-gpu-producer")
    consumer = NixlTransport(tmp_path, agent_name="nixl-multi-gpu-consumer")
    receipt = producer.publish(
        serializer.serialize(batch),
        experience_id=batch.experience_id,
        idempotency_key="real-nixl-multi-gpu",
    )
    delivery = consumer.receive(5.0)
    assert delivery is not None
    restored = serializer.deserialize(delivery.payload)
    assert torch.equal(restored.tensors["tokens"].data, first)
    assert torch.equal(restored.tensors["advantages"].data, second)
    consumer.ack(delivery.token)
    assert receipt.wait(1.0).state is ReceiptState.ACKED
    consumer.close()
    producer.close()
