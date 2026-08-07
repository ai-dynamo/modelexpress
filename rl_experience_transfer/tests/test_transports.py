# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference transport behavior and multiprocess persistence tests."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from rlxfer.errors import (
    BackpressureError,
    CapabilityError,
    DeliveryError,
    SerializationError,
    TransportError,
)
from rlxfer.model import ExperienceBatch, ExperienceMetadata, TensorPayload
from rlxfer.serialization import JsonExperienceSerializer, SerializationLimits
from rlxfer.transport import (
    ExperienceTransport,
    ReceiptState,
    TransferPlan,
    TransportConfig,
    create_transport,
)
from rlxfer.transports.fallback import FallbackTransport
from rlxfer.transports.filesystem import FileSystemTransport
from rlxfer.transports.memory import InMemoryTransport


def _batch() -> ExperienceBatch:
    return ExperienceBatch(
        metadata=ExperienceMetadata("producer", "test", "1"),
        tensors={
            "tokens": TensorPayload(np.arange(12, dtype=np.int64).reshape(3, 4)),
            "rewards": TensorPayload(np.asarray([0.5, -1.0, 2.0], dtype=np.float32)),
        },
        extensions={"test": {"nested": [1, {"preserved": True}]}},
    )


def _consume_filesystem(path: str, output: Any) -> None:
    transport = FileSystemTransport(path)
    delivery = transport.receive(5.0)
    if delivery is None:
        output.put({"error": "receive timed out"})
        return
    batch = JsonExperienceSerializer().deserialize(delivery.payload)
    output.put(
        {
            "tokens": cast(np.ndarray[Any, Any], batch.tensors["tokens"].data).tolist(),
            "attempt": delivery.attempt,
        }
    )
    transport.ack(delivery.token)
    transport.close()


@pytest.mark.unit
def test_memory_round_trip_ack_and_duplicate() -> None:
    serializer = JsonExperienceSerializer()
    batch = _batch()
    payload = serializer.serialize(batch)
    transport = InMemoryTransport(max_queue=2)

    receipt = transport.publish(
        payload,
        experience_id=batch.experience_id,
        idempotency_key="stable-key",
    )
    duplicate = transport.publish(
        payload,
        experience_id=batch.experience_id,
        idempotency_key="stable-key",
    )
    assert duplicate.receipt_id == receipt.receipt_id

    delivery = transport.receive(0.1)
    assert delivery is not None
    restored = serializer.deserialize(delivery.payload)
    np.testing.assert_array_equal(restored.tensors["tokens"].data, batch.tensors["tokens"].data)
    transport.ack(delivery.token)
    assert receipt.wait(0.1).state is ReceiptState.ACKED


@pytest.mark.unit
def test_memory_retry_rejection_backpressure_and_capability() -> None:
    serializer = JsonExperienceSerializer()
    batch = _batch()
    payload = serializer.serialize(batch)
    transport = InMemoryTransport(max_queue=1)
    receipt = transport.publish(
        payload,
        experience_id=batch.experience_id,
        idempotency_key="retry-key",
        max_retries=1,
    )
    with pytest.raises(BackpressureError):
        transport.publish(
            payload,
            experience_id=ExperienceMetadata("other", "test", "1").experience_id,
            idempotency_key="blocked-key",
            timeout=0.0,
        )
    delivery = transport.receive(0.1)
    assert delivery is not None
    transport.nack(delivery.token, "injected", retry=True)
    retried = transport.receive(0.1)
    assert retried is not None and retried.attempt == 2
    transport.reject(retried.token, "malformed")
    result = receipt.wait(0.1)
    assert (result.state, result.reason, result.attempts) == (
        ReceiptState.REJECTED,
        "malformed",
        2,
    )
    with pytest.raises(DeliveryError):
        transport.ack(retried.token)
    with pytest.raises(CapabilityError):
        TransferPlan(require_persistence=True).check(transport.capabilities)


@pytest.mark.unit
def test_memory_failure_injection_leaves_queue_recoverable() -> None:
    serializer = JsonExperienceSerializer()
    batch = _batch()
    payload = serializer.serialize(batch)
    publish_failure = InMemoryTransport(failure_at_publish=1)
    with pytest.raises(TransportError, match="injected publish"):
        publish_failure.publish(
            payload,
            experience_id=batch.experience_id,
            idempotency_key="publish-failure",
        )
    assert publish_failure.health().queue_depth == 0

    receive_failure = InMemoryTransport(failure_at_receive=1)
    receipt = receive_failure.publish(
        payload,
        experience_id=batch.experience_id,
        idempotency_key="receive-failure",
    )
    with pytest.raises(TransportError, match="injected receive"):
        receive_failure.receive(0.1)
    delivery = receive_failure.receive(0.1)
    assert delivery is not None
    receive_failure.ack(delivery.token)
    assert receipt.wait(0.1).state is ReceiptState.ACKED


@pytest.mark.unit
def test_fallback_routes_delivery_and_does_not_retry_ambiguous_errors_by_default() -> None:
    serializer = JsonExperienceSerializer()
    batch = _batch()
    payload = serializer.serialize(batch)
    primary = InMemoryTransport(failure_at_publish=1)
    secondary = InMemoryTransport()

    safe = FallbackTransport((primary, secondary))
    with pytest.raises(TransportError, match="injected publish"):
        safe.publish(payload, experience_id=batch.experience_id, idempotency_key="safe")
    assert secondary.health().queue_depth == 0

    enabled = create_transport(
        TransportConfig(
            "fallback",
            {
                "transports": (InMemoryTransport(failure_at_publish=1), secondary),
                "fallback_exceptions": [TransportError],
            },
        )
    )
    assert isinstance(enabled, FallbackTransport)
    receipt = enabled.publish(
        payload,
        experience_id=batch.experience_id,
        idempotency_key="fallback",
    )
    delivery = enabled.receive(0.1)
    assert delivery is not None
    assert receipt.receipt_id.startswith("1:") and delivery.token.startswith("1:")
    enabled.ack(delivery.token)
    assert receipt.wait(0.1).state is ReceiptState.ACKED


@pytest.mark.unit
def test_reference_transports_enforce_transfer_size_limits(tmp_path: Path) -> None:
    payload = JsonExperienceSerializer().serialize(_batch())
    limits = replace(
        SerializationLimits(),
        max_tensor_bytes=64,
        max_total_tensor_bytes=128,
    )
    transports: list[ExperienceTransport] = [
        InMemoryTransport(limits=limits),
        FileSystemTransport(tmp_path, limits=limits),
    ]

    for index, transport in enumerate(transports):
        assert transport.capabilities.max_transfer_size == (
            limits.max_metadata_bytes + limits.max_total_tensor_bytes
        )
        with pytest.raises(SerializationError, match="per-tensor byte limit"):
            transport.publish(
                payload,
                experience_id=_batch().experience_id,
                idempotency_key=f"limited-{index}",
            )
        assert transport.health().queue_depth == 0


@pytest.mark.unit
def test_memory_cancel_only_applies_before_delivery() -> None:
    serializer = JsonExperienceSerializer()
    batch = _batch()
    payload = serializer.serialize(batch)
    transport = InMemoryTransport()
    pending = transport.publish(
        payload,
        experience_id=batch.experience_id,
        idempotency_key="cancel-pending",
    )
    transport.cancel(pending.receipt_id, "not needed")
    assert pending.wait(0.1).state is ReceiptState.CANCELLED
    assert transport.receive(0.0) is None

    inflight = transport.publish(
        payload,
        experience_id=batch.experience_id,
        idempotency_key="cancel-inflight",
    )
    assert transport.receive(0.1) is not None
    with pytest.raises(DeliveryError, match="inflight"):
        transport.cancel(inflight.receipt_id)


@pytest.mark.integration
@pytest.mark.multi_process
def test_filesystem_multiprocess_byte_exact(tmp_path: Path) -> None:
    serializer = JsonExperienceSerializer()
    batch = _batch()
    producer = FileSystemTransport(tmp_path)
    receipt = producer.publish(
        serializer.serialize(batch),
        experience_id=batch.experience_id,
        idempotency_key="multiprocess-key",
    )

    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    process = context.Process(target=_consume_filesystem, args=(str(tmp_path), output))
    process.start()
    process.join(10.0)
    assert process.exitcode == 0
    assert output.get(timeout=1.0) == {
        "tokens": np.arange(12, dtype=np.int64).reshape(3, 4).tolist(),
        "attempt": 1,
    }
    assert receipt.wait(1.0).state is ReceiptState.ACKED
    assert producer.health().queue_depth == 0


@pytest.mark.integration
def test_filesystem_recovers_stale_inflight(tmp_path: Path) -> None:
    serializer = JsonExperienceSerializer()
    batch = _batch()
    transport = FileSystemTransport(tmp_path, lease_timeout=0.01)
    transport.publish(
        serializer.serialize(batch),
        experience_id=batch.experience_id,
        idempotency_key="recovery-key",
    )
    abandoned = transport.receive(0.1)
    assert abandoned is not None
    inflight = next((tmp_path / "inflight").iterdir())
    old = time.time() - 10.0
    os.utime(inflight, (old, old))

    recovered = FileSystemTransport(tmp_path, lease_timeout=0.01).receive(0.1)
    assert recovered is not None
    assert recovered.experience_id == batch.experience_id
    FileSystemTransport(tmp_path).ack(recovered.token)


@pytest.mark.integration
def test_filesystem_claim_refreshes_an_old_pending_lease(tmp_path: Path) -> None:
    serializer = JsonExperienceSerializer()
    batch = _batch()
    owner = FileSystemTransport(tmp_path, lease_timeout=1.0)
    owner.publish(
        serializer.serialize(batch),
        experience_id=batch.experience_id,
        idempotency_key="old-pending",
    )
    pending = next((tmp_path / "pending").iterdir())
    old = time.time() - 10.0
    os.utime(pending, (old, old))

    active = owner.receive(0.1)
    assert active is not None
    contender = FileSystemTransport(tmp_path, lease_timeout=1.0)

    assert contender.receive(0.0) is None
    owner.ack(active.token)


@pytest.mark.integration
def test_filesystem_publish_rolls_back_failed_index_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serializer = JsonExperienceSerializer()
    batch = _batch()
    transport = FileSystemTransport(tmp_path)
    original = transport._write_json

    def fail_index(target: Path, value: Mapping[str, Any]) -> None:
        if target.parent.name == "idempotency":
            raise OSError("injected index failure")
        original(target, value)

    monkeypatch.setattr(transport, "_write_json", fail_index)
    with pytest.raises(OSError, match="injected index failure"):
        transport.publish(
            serializer.serialize(batch),
            experience_id=batch.experience_id,
            idempotency_key="index-failure",
        )

    assert not any((tmp_path / "pending").iterdir())
    assert transport.health().queue_depth == 0


@pytest.mark.integration
def test_filesystem_repairs_missing_idempotency_index(tmp_path: Path) -> None:
    serializer = JsonExperienceSerializer()
    batch = _batch()
    first = FileSystemTransport(tmp_path)
    receipt = first.publish(
        serializer.serialize(batch),
        experience_id=batch.experience_id,
        idempotency_key="repair-index",
    )
    next((tmp_path / "idempotency").iterdir()).unlink()

    recovered = FileSystemTransport(tmp_path)
    duplicate = recovered.publish(
        serializer.serialize(batch),
        experience_id=batch.experience_id,
        idempotency_key="repair-index",
    )

    assert duplicate.receipt_id == receipt.receipt_id
    assert recovered.health().queue_depth == 1


@pytest.mark.integration
def test_filesystem_multiproducer_bound_is_atomic(tmp_path: Path) -> None:
    serializer = JsonExperienceSerializer()

    def publish(index: int) -> str:
        batch = _batch()
        transport = FileSystemTransport(tmp_path, max_queue=1, poll_interval=0.001)
        try:
            transport.publish(
                serializer.serialize(batch),
                experience_id=batch.experience_id,
                idempotency_key=f"concurrent-{index}",
                timeout=0.02,
            )
        except BackpressureError:
            return "blocked"
        return "accepted"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(publish, range(8)))

    assert outcomes.count("accepted") == 1
    assert outcomes.count("blocked") == 7
    assert FileSystemTransport(tmp_path, max_queue=1).health().queue_depth == 1


@pytest.mark.integration
def test_filesystem_corrupt_partial_delivery_is_rejected_and_removed(tmp_path: Path) -> None:
    serializer = JsonExperienceSerializer()
    batch = _batch()
    transport = FileSystemTransport(tmp_path)
    receipt = transport.publish(
        serializer.serialize(batch),
        experience_id=batch.experience_id,
        idempotency_key="corrupt-key",
    )
    pending = next((tmp_path / "pending").iterdir())
    next((pending / "buffers").iterdir()).unlink()

    with pytest.raises(DeliveryError, match="corrupt"):
        transport.receive(0.1)
    assert receipt.wait(0.1).state is ReceiptState.REJECTED
    assert transport.health().queue_depth == 0


@pytest.mark.integration
def test_filesystem_rejects_manifest_path_escape(tmp_path: Path) -> None:
    serializer = JsonExperienceSerializer()
    batch = _batch()
    transport = FileSystemTransport(tmp_path)
    receipt = transport.publish(
        serializer.serialize(batch),
        experience_id=batch.experience_id,
        idempotency_key="path-escape",
    )
    pending = next((tmp_path / "pending").iterdir())
    manifest_path = pending / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    manifest["buffers"][0]["filename"] = "../../outside"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(DeliveryError, match="corrupt"):
        transport.receive(0.1)

    assert receipt.wait(0.1).state is ReceiptState.REJECTED
    assert not (tmp_path / "outside").exists()


@pytest.mark.integration
def test_filesystem_cancel_pending_delivery(tmp_path: Path) -> None:
    serializer = JsonExperienceSerializer()
    batch = _batch()
    transport = FileSystemTransport(tmp_path)
    receipt = transport.publish(
        serializer.serialize(batch),
        experience_id=batch.experience_id,
        idempotency_key="cancel-filesystem",
    )

    transport.cancel(receipt.receipt_id, "shutdown")

    result = receipt.wait(0.1)
    assert (result.state, result.reason) == (ReceiptState.CANCELLED, "shutdown")
    assert transport.health().queue_depth == 0
