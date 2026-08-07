# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rlxfer.api import ExperienceConsumer, ExperienceProducer
from rlxfer.compatibility import CompatibilityRequirements
from rlxfer.contracts import ConsumerContract
from rlxfer.errors import IntegrityError, SerializationError, TransportError
from rlxfer.model import (
    ExperienceBatch,
    ExperienceMetadata,
    PolicyVersion,
    TensorPayload,
    Trajectory,
)
from rlxfer.serialization import (
    AuthenticatedExperienceSerializer,
    JsonExperienceSerializer,
    validate_metadata,
)
from rlxfer.state import SqliteDeliveryState
from rlxfer.tracing import TraceContext, trace_context_from, with_trace_context
from rlxfer.transport import ReceiptState
from rlxfer.transports.fallback import FallbackTransport
from rlxfer.transports.memory import InMemoryTransport


def _batch(*, key: str = "stable-key") -> ExperienceBatch:
    return ExperienceBatch(
        metadata=ExperienceMetadata("rollout", "test", "1", idempotency_key=key),
        trajectories=(
            Trajectory(
                prompt="sensitive prompt",
                tokens=TensorPayload(np.arange(4, dtype=np.int64)),
            ),
        ),
    )


@pytest.mark.unit
def test_authenticated_serializer_covers_metadata_and_external_buffers() -> None:
    serializer = AuthenticatedExperienceSerializer({"current": b"k" * 32}, signing_key_id="current")
    batch = _batch()
    encoded = serializer.serialize(batch)

    assert validate_metadata(encoded.metadata)
    assert serializer.deserialize(encoded).experience_id == batch.experience_id
    with pytest.raises(SerializationError, match="unknown or missing fields"):
        JsonExperienceSerializer().deserialize(encoded)

    segment = encoded.buffers[0]
    raw = segment.materialize()
    tampered = replace(segment, data=bytes([raw[0] ^ 1]) + raw[1:], owner=None)
    with pytest.raises(IntegrityError, match="authentication failed"):
        serializer.deserialize(replace(encoded, buffers=(tampered,)))

    document = json.loads(encoded.metadata)
    document["authentication"]["key_id"] = "rotated"
    changed_key_id = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
    rotating = AuthenticatedExperienceSerializer(
        {"current": b"k" * 32, "rotated": b"k" * 32}, signing_key_id="current"
    )
    with pytest.raises(IntegrityError, match="authentication failed"):
        rotating.deserialize(replace(encoded, metadata=changed_key_id))

    unsigned = JsonExperienceSerializer().serialize(_batch())
    with pytest.raises(IntegrityError, match="requires a signature"):
        serializer.deserialize(unsigned)


@pytest.mark.unit
def test_trace_context_survives_safe_serialization() -> None:
    context = TraceContext(
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        "vendor=value",
    )
    batch = _batch()
    traced = with_trace_context(batch, context)
    restored = JsonExperienceSerializer().deserialize(JsonExperienceSerializer().serialize(traced))

    assert trace_context_from(batch) is None
    assert trace_context_from(restored) == context
    with pytest.raises(ValueError, match="traceparent"):
        TraceContext("00-00000000000000000000000000000000-0000000000000000-00")
    with pytest.raises(ValueError, match="tracestate"):
        TraceContext(context.traceparent, "vendor=one,vendor=two")


@pytest.mark.integration
def test_sqlite_state_suppresses_restart_duplicates_and_records_dead_letters(
    tmp_path: Path,
) -> None:
    path = tmp_path / "delivery-state.sqlite"
    first_transport = InMemoryTransport()
    first_receipt = ExperienceProducer(first_transport).publish(_batch())
    first = ExperienceConsumer(first_transport, state_store=SqliteDeliveryState(path)).receive(0.1)
    assert first is not None
    first.ack()
    assert first_receipt.wait(0.1).state is ReceiptState.ACKED

    replay_transport = InMemoryTransport()
    replay_receipt = ExperienceProducer(replay_transport).publish(_batch())
    restarted = ExperienceConsumer(replay_transport, state_store=SqliteDeliveryState(path))
    assert restarted.receive(0.01) is None
    assert replay_receipt.wait(0.1).state is ReceiptState.ACKED

    rejected_transport = InMemoryTransport()
    rejected_receipt = ExperienceProducer(rejected_transport).publish(_batch(key="rejected"))
    state = SqliteDeliveryState(path)
    rejected = ExperienceConsumer(rejected_transport, state_store=state).receive(0.1)
    assert rejected is not None
    rejected.reject("incompatible policy")

    assert rejected_receipt.wait(0.1).state is ReceiptState.REJECTED
    assert [(letter.idempotency_key, letter.reason) for letter in state.dead_letters()] == [
        ("rejected", "incompatible policy")
    ]
    assert b"sensitive prompt" not in path.read_bytes()

    exhausted_transport = InMemoryTransport()
    exhausted_receipt = ExperienceProducer(exhausted_transport).publish(
        _batch(key="exhausted"), max_retries=0
    )
    exhausted = ExperienceConsumer(exhausted_transport, state_store=state).receive(0.1)
    assert exhausted is not None
    exhausted.nack("retry budget exhausted")
    assert exhausted_receipt.wait(0.1).state is ReceiptState.NACKED
    assert state.dead_letters()[0].idempotency_key == "exhausted"


@pytest.mark.integration
def test_reliability_features_compose_end_to_end(tmp_path: Path) -> None:
    context = TraceContext("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
    batch = _batch(key="composed")
    batch.metadata.policy_version = PolicyVersion(4, policy_id="actor", model_id="tiny")
    batch = with_trace_context(batch, context)
    contract = ConsumerContract(
        CompatibilityRequirements(
            "trainer",
            "1",
            policy_version=PolicyVersion(5, policy_id="actor", model_id="tiny"),
            max_policy_lag=1,
        ),
        required_fields=frozenset({"extensions.w3c.trace_context.traceparent"}),
    )
    transport = FallbackTransport(
        (InMemoryTransport(failure_at_publish=1), InMemoryTransport()),
        fallback_exceptions=(TransportError,),
    )
    serializer = AuthenticatedExperienceSerializer({"test": b"k" * 32}, signing_key_id="test")
    state = SqliteDeliveryState(tmp_path / "composed.sqlite")

    receipt = ExperienceProducer(
        transport, serializer=serializer, consumer_contract=contract
    ).publish(batch)
    delivery = ExperienceConsumer(
        transport,
        serializer=serializer,
        consumer_contract=contract,
        state_store=state,
    ).receive(0.1)

    assert delivery is not None and trace_context_from(delivery.batch) == context
    delivery.ack()
    assert receipt.wait(0.1).state is ReceiptState.ACKED
    assert state.was_consumed("composed")
