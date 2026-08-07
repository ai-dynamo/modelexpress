# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import ClassVar

import numpy as np
import pytest

from rlxfer.api import Delivery, ExperienceConsumer, ExperienceProducer
from rlxfer.errors import CapabilityError, ClosedError, DeliveryError, IntegrityError
from rlxfer.model import ExperienceBatch, ExperienceMetadata, TensorPayload, Trajectory
from rlxfer.observability import InMemoryMetrics
from rlxfer.serialization import JsonExperienceSerializer, SerializedExperience
from rlxfer.transport import (
    DeliveryReceipt,
    HealthStatus,
    ReceiptState,
    TransferPlan,
    TransportCapabilities,
    TransportDelivery,
)
from rlxfer.transports.memory import InMemoryTransport


def _trajectory() -> Trajectory:
    return Trajectory(
        prompt="2+2?",
        response="4",
        tokens=TensorPayload(np.asarray([4], dtype=np.int64)),
        log_probs=TensorPayload(np.asarray([-0.2], dtype=np.float32)),
        rewards={"task": 1.0},
    )


def _batch(*, idempotency_key: str | None = None) -> ExperienceBatch:
    return ExperienceBatch(
        metadata=ExperienceMetadata(
            producer_id="rollout-1",
            producer_framework="test",
            producer_framework_version="1",
            idempotency_key=idempotency_key,
        ),
        trajectories=(_trajectory(),),
    )


@pytest.mark.unit
def test_publish_trajectory_receive_ack_and_metrics() -> None:
    transport = InMemoryTransport()
    producer_metrics = InMemoryMetrics()
    consumer_metrics = InMemoryMetrics()
    producer = ExperienceProducer(
        transport,
        metrics=producer_metrics,
        producer_id="rollout-1",
    )
    consumer = ExperienceConsumer(transport, metrics=consumer_metrics)

    receipt = producer.publish_trajectory(_trajectory())
    delivery = consumer.receive(timeout=0.1)

    assert isinstance(delivery, Delivery)
    assert delivery.attempt == 1
    assert delivery.to_framework() is delivery.batch
    assert delivery.batch.metadata.producer_id == "rollout-1"
    delivery.ack()
    assert delivery.state == "acknowledged"
    assert receipt.wait(0.1).state is ReceiptState.ACKED
    with pytest.raises(DeliveryError, match="already acknowledged"):
        delivery.ack()

    producer_counters, _ = producer_metrics.snapshot()
    consumer_counters, _ = consumer_metrics.snapshot()
    assert producer_counters["produced_batches"] == 1
    assert producer_counters["produced_trajectories"] == 1
    assert producer_counters["bytes_transferred"] > 0
    assert consumer_counters["received_batches"] == 1
    assert consumer_counters["consumed_batches"] == 1
    assert consumer_counters["consumed_trajectories"] == 1


@pytest.mark.unit
def test_nack_retry_and_permanent_rejection() -> None:
    transport = InMemoryTransport()
    producer = ExperienceProducer(transport)
    consumer = ExperienceConsumer(transport)

    retried_receipt = producer.publish_batch(_batch(), max_retries=2)
    first = consumer.receive(0.1)
    assert first is not None
    first.nack("trainer temporarily unavailable")
    assert first.state == "nacked"

    second = consumer.receive(0.1)
    assert second is not None
    assert second.attempt == 2
    second.ack()
    result = retried_receipt.wait(0.1)
    assert result.state is ReceiptState.ACKED
    assert result.attempts == 2

    rejected_receipt = producer.publish_batch(_batch())
    rejected = consumer.receive(0.1)
    assert rejected is not None
    rejected.reject("algorithm is incompatible")
    assert rejected_receipt.wait(0.1).state is ReceiptState.REJECTED


class _NativeAdapter:
    framework_name: ClassVar[str] = "native_test"

    @property
    def adapter_version(self) -> str:
        return "test"

    @property
    def framework_version(self) -> str:
        return "2"

    def from_framework(self, native: object) -> ExperienceBatch:
        if not isinstance(native, dict) or not isinstance(native.get("tokens"), list):
            raise TypeError("native rollout must contain tokens")
        tokens = np.asarray(native["tokens"], dtype=np.int64)
        return ExperienceBatch(
            metadata=ExperienceMetadata("native-worker", self.framework_name, "2"),
            trajectories=(Trajectory(tokens=TensorPayload(tokens)),),
            extensions={self.framework_name: {"kind": "native"}},
        )

    def to_framework(self, batch: ExperienceBatch) -> object:
        tokens = batch.trajectories[0].tokens
        assert tokens is not None
        return {"tokens": np.asarray(tokens.data).tolist()}

    def validate_compatible(self, batch: ExperienceBatch) -> None:
        if self.framework_name not in batch.extensions:
            raise ValueError("native extension is required")


@pytest.mark.unit
def test_optional_adapter_converts_native_rollout_and_training_input() -> None:
    transport = InMemoryTransport()
    adapter = _NativeAdapter()
    producer = ExperienceProducer(transport, adapter=adapter)
    consumer = ExperienceConsumer(transport, adapter=adapter)

    receipt = producer.publish({"tokens": [7, 8]})
    delivery = consumer.receive(0.1)

    assert delivery is not None
    assert delivery.to_framework() == {"tokens": [7, 8]}
    delivery.ack()
    assert receipt.wait(0.1).state is ReceiptState.ACKED


@pytest.mark.unit
def test_producer_checks_transfer_requirements_before_publish() -> None:
    transport = InMemoryTransport()
    producer = ExperienceProducer(
        transport,
        transfer_plan=TransferPlan(require_persistence=True),
    )

    with pytest.raises(CapabilityError, match="persistence"):
        producer.publish(_batch())

    assert transport.health().queue_depth == 0


class _ReplayTransport:
    def __init__(self, deliveries: list[TransportDelivery]) -> None:
        self.deliveries = deque(deliveries)
        self.acked: list[str] = []
        self.nacked: list[str] = []
        self.rejected: list[str] = []
        self.cancelled: list[str] = []
        self.closed = False

    @property
    def capabilities(self) -> TransportCapabilities:
        return TransportCapabilities(name="replay")

    def publish(
        self,
        payload: SerializedExperience,
        *,
        experience_id: str,
        idempotency_key: str,
        timeout: float | None = None,
        max_retries: int = 3,
    ) -> DeliveryReceipt:
        del payload, experience_id, idempotency_key, timeout, max_retries
        raise AssertionError("replay transport is receive-only")

    def receive(self, timeout: float | None = None) -> TransportDelivery | None:
        del timeout
        return self.deliveries.popleft() if self.deliveries else None

    def ack(self, token: str) -> None:
        self.acked.append(token)

    def nack(self, token: str, reason: str, *, retry: bool = True) -> None:
        del reason, retry
        self.nacked.append(token)

    def reject(self, token: str, reason: str) -> None:
        del reason
        self.rejected.append(token)

    def cancel(self, receipt_id: str, reason: str = "cancelled") -> None:
        del reason
        self.cancelled.append(receipt_id)

    def health(self) -> HealthStatus:
        return HealthStatus(not self.closed)

    def close(self, timeout: float | None = None) -> None:
        del timeout
        self.closed = True


def _raw_delivery(
    batch: ExperienceBatch,
    token: str,
    *,
    experience_id: str | None = None,
) -> TransportDelivery:
    return TransportDelivery(
        token=token,
        experience_id=experience_id or batch.experience_id,
        idempotency_key=batch.metadata.idempotency_key or batch.experience_id,
        payload=JsonExperienceSerializer().serialize(batch),
        attempt=1,
        published_at=time.time(),
    )


@pytest.mark.unit
def test_consumer_suppresses_already_acknowledged_duplicate() -> None:
    batch = _batch(idempotency_key="same-request")
    transport = _ReplayTransport([_raw_delivery(batch, "first"), _raw_delivery(batch, "duplicate")])
    metrics = InMemoryMetrics()
    consumer = ExperienceConsumer(transport, metrics=metrics)

    first = consumer.receive(0.0)
    assert first is not None
    first.ack()
    assert consumer.receive(0.0) is None

    assert transport.acked == ["first", "duplicate"]
    counters, _ = metrics.snapshot()
    assert counters["received_batches"] == 1
    assert counters["duplicate_deliveries"] == 1


@pytest.mark.unit
def test_invalid_envelope_is_rejected_before_delivery() -> None:
    batch = _batch(idempotency_key="invalid-request")
    transport = _ReplayTransport([_raw_delivery(batch, "invalid", experience_id="different-id")])
    consumer = ExperienceConsumer(transport)

    with pytest.raises(IntegrityError, match="experience IDs differ"):
        consumer.receive(0.0)

    assert transport.rejected == ["invalid"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_wrappers_and_context_shutdown() -> None:
    transport = InMemoryTransport()
    producer = ExperienceProducer(transport)
    consumer = ExperienceConsumer(transport)

    async with producer, consumer:
        receipt = await producer.publish_async(_batch())
        delivery = await consumer.receive_async(0.1)
        assert delivery is not None
        await delivery.ack_async()
        assert receipt.wait(0.1).state is ReceiptState.ACKED

    assert not transport.health().healthy
    with pytest.raises(ClosedError):
        producer.publish(_batch())


@pytest.mark.unit
@pytest.mark.asyncio
async def test_async_receive_cancellation_and_cleanup() -> None:
    transport = InMemoryTransport()
    consumer = ExperienceConsumer(transport)
    receive = asyncio.create_task(consumer.receive_async(10.0))
    await asyncio.sleep(0)
    receive.cancel()
    with pytest.raises(asyncio.CancelledError):
        await receive
    await consumer.close_async()
    assert not transport.health().healthy


@pytest.mark.unit
@pytest.mark.asyncio
async def test_producer_can_cancel_pending_publish() -> None:
    transport = InMemoryTransport()
    producer = ExperienceProducer(transport)
    receipt = producer.publish(_batch())

    await producer.cancel_async(receipt, "caller cancelled")

    result = receipt.wait(0.1)
    assert (result.state, result.reason) == (
        ReceiptState.CANCELLED,
        "caller cancelled",
    )
    assert transport.health().queue_depth == 0
