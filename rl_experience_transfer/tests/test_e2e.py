# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end matrices for public API reliability guarantees."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rlxfer import (
    AuthenticatedExperienceSerializer,
    CompatibilityRequirements,
    ConsumerContract,
    ExperienceBatch,
    ExperienceConsumer,
    ExperienceMetadata,
    ExperienceProducer,
    ExperienceSerializer,
    JsonExperienceSerializer,
    PolicyVersion,
    ReceiptState,
    SqliteDeliveryState,
    TensorPayload,
    TraceContext,
    Trajectory,
    trace_context_from,
    with_trace_context,
)
from rlxfer.transport import ExperienceTransport
from rlxfer.transports import FileSystemTransport, InMemoryTransport

_TRACE = TraceContext("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")


def _serializer(name: str) -> ExperienceSerializer:
    if name == "authenticated":
        return AuthenticatedExperienceSerializer(
            {"test-key": b"k" * 32},
            signing_key_id="test-key",
        )
    return JsonExperienceSerializer(checksum=True)


def _transport(name: str, path: Path) -> ExperienceTransport:
    return FileSystemTransport(path) if name == "filesystem" else InMemoryTransport()


def _batch(key: str) -> ExperienceBatch:
    policy = PolicyVersion(4, policy_id="actor", model_id="tiny")
    return with_trace_context(
        ExperienceBatch(
            metadata=ExperienceMetadata(
                "rollout",
                "test",
                "1",
                idempotency_key=key,
                policy_version=policy,
            ),
            trajectories=(
                Trajectory(
                    policy_version=policy,
                    tokens=TensorPayload(np.asarray([3, 4], dtype=np.int64)),
                    rewards={"task": 1.0},
                ),
            ),
        ),
        _TRACE,
    )


@pytest.mark.e2e
@pytest.mark.parametrize("transport_name", ["memory", "filesystem"])
@pytest.mark.parametrize("serializer_name", ["json", "authenticated"])
@pytest.mark.parametrize("outcome", ["ack", "retry_then_ack", "reject"])
def test_delivery_lifecycle_matrix(
    tmp_path: Path,
    transport_name: str,
    serializer_name: str,
    outcome: str,
) -> None:
    """Exercise 2 transports x 2 security modes x 3 settlement paths."""

    key = f"{transport_name}:{serializer_name}:{outcome}"
    serializer = _serializer(serializer_name)
    transport = _transport(transport_name, tmp_path / "queue")
    state = SqliteDeliveryState(tmp_path / "delivery.sqlite")
    contract = ConsumerContract(
        CompatibilityRequirements(
            "trainer",
            "1",
            policy_version=PolicyVersion(5, policy_id="actor", model_id="tiny"),
            max_policy_lag=1,
        ),
        required_fields=frozenset({"extensions.w3c.trace_context.traceparent"}),
    )
    producer = ExperienceProducer(
        transport,
        serializer=serializer,
        consumer_contract=contract,
    )
    consumer = ExperienceConsumer(
        transport,
        serializer=serializer,
        consumer_contract=contract,
        state_store=state,
    )
    try:
        receipt = producer.publish(_batch(key), max_retries=1)
        delivery = consumer.receive(1.0)
        assert delivery is not None
        assert trace_context_from(delivery.batch) == _TRACE
        tokens = delivery.batch.trajectories[0].tokens
        assert tokens is not None
        np.testing.assert_array_equal(tokens.data, [3, 4])

        if outcome == "retry_then_ack":
            delivery.nack("temporary trainer pressure")
            delivery = consumer.receive(1.0)
            assert delivery is not None and delivery.attempt == 2
            delivery.ack()
        elif outcome == "reject":
            delivery.reject("application rejected the experience")
        else:
            delivery.ack()
        result = receipt.wait(1.0)
    finally:
        transport.close()

    if outcome == "reject":
        assert result.state is ReceiptState.REJECTED
        assert not state.was_consumed(key)
        assert state.dead_letters()[0].idempotency_key == key
    else:
        assert result.state is ReceiptState.ACKED
        assert result.attempts == (2 if outcome == "retry_then_ack" else 1)
        assert state.was_consumed(key)
        assert state.dead_letters() == ()
