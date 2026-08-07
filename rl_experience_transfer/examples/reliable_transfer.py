# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an authenticated, traced, retrying transfer with durable consumer state."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from rlxfer import (
    AuthenticatedExperienceSerializer,
    CompatibilityRequirements,
    ConsumerContract,
    ExperienceBatch,
    ExperienceConsumer,
    ExperienceMetadata,
    ExperienceProducer,
    PolicyVersion,
    ReceiptState,
    SqliteDeliveryState,
    TensorPayload,
    TraceContext,
    Trajectory,
    trace_context_from,
    with_trace_context,
)
from rlxfer.transports import FileSystemTransport

_TRACE = TraceContext("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")


def run(queue_path: Path, state_path: Path) -> dict[str, object]:
    """Transfer one batch, retry once, acknowledge it, and verify durable state."""

    policy = PolicyVersion(7, policy_id="actor", model_id="tiny-policy")
    batch = with_trace_context(
        ExperienceBatch(
            metadata=ExperienceMetadata(
                "rollout-worker",
                "canonical",
                "1.0",
                idempotency_key="reliable-example",
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
    contract = ConsumerContract(
        CompatibilityRequirements(
            "trainer",
            "1.0",
            policy_version=PolicyVersion(8, policy_id="actor", model_id="tiny-policy"),
            max_policy_lag=1,
        ),
        required_fields=frozenset({"extensions.w3c.trace_context.traceparent"}),
    )
    serializer = AuthenticatedExperienceSerializer(
        {"example-key": b"rlxfer-example-signing-key-32byt"},
        signing_key_id="example-key",
    )
    state = SqliteDeliveryState(state_path)
    transport = FileSystemTransport(queue_path)
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
        receipt = producer.publish(batch, max_retries=1)
        first = consumer.receive(1.0)
        if first is None:
            raise RuntimeError("first receive timed out")
        first.nack("transient trainer backpressure")
        retried = consumer.receive(1.0)
        if retried is None:
            raise RuntimeError("retry receive timed out")
        if retried.attempt != 2 or trace_context_from(retried.batch) != _TRACE:
            raise RuntimeError("retry or trace context was not preserved")
        retried.ack()
        result = receipt.wait(1.0)
    finally:
        transport.close()
    if result.state is not ReceiptState.ACKED or not state.was_consumed("reliable-example"):
        raise RuntimeError("terminal receipt or durable idempotency state is incorrect")
    return {
        "attempts": result.attempts,
        "authenticated": True,
        "durably_consumed": True,
        "result": "PASSED",
        "state": result.state.value,
        "traceparent": _TRACE.traceparent,
        "transport": "filesystem",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-path", type=Path)
    parser.add_argument("--state-path", type=Path)
    arguments = parser.parse_args()
    try:
        if arguments.queue_path is None and arguments.state_path is None:
            with tempfile.TemporaryDirectory(prefix="rlxfer-reliable-") as temporary:
                root = Path(temporary)
                result = run(root / "queue", root / "delivery.sqlite")
        elif arguments.queue_path is not None and arguments.state_path is not None:
            result = run(arguments.queue_path, arguments.state_path)
        else:
            raise ValueError("--queue-path and --state-path must be supplied together")
    except Exception as error:
        result = {"error": f"{type(error).__name__}: {error}", "result": "FAILED"}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["result"] == "PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
