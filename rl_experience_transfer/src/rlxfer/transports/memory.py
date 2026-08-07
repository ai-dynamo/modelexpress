# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded deterministic in-process transport."""

from __future__ import annotations

import time
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from threading import Condition
from typing import Any

from rlxfer.errors import BackpressureError, ClosedError, DeliveryError, TransportError
from rlxfer.serialization import (
    SerializationLimits,
    SerializedExperience,
    validate_transfer_limits,
)
from rlxfer.transport import (
    DeliveryReceipt,
    HealthStatus,
    ReceiptResult,
    ReceiptState,
    TransportCapabilities,
    TransportDelivery,
)


@dataclass(slots=True)
class _Record:
    payload: SerializedExperience
    experience_id: str
    idempotency_key: str
    published_at: float
    max_retries: int
    attempt: int = 1
    state: ReceiptState = ReceiptState.ACCEPTED
    reason: str | None = None
    token: str | None = None


class InMemoryTransport:
    """Thread-safe at-least-once queue with explicit acknowledgement."""

    def __init__(
        self,
        *,
        max_queue: int = 128,
        failure_at_publish: int | None = None,
        failure_at_receive: int | None = None,
        limits: SerializationLimits | None = None,
    ) -> None:
        if max_queue < 1:
            raise ValueError("max_queue must be positive")
        self._max_queue = max_queue
        self._failure_at_publish = failure_at_publish
        self._failure_at_receive = failure_at_receive
        self._publish_count = 0
        self._receive_count = 0
        self._limits = limits or SerializationLimits()
        self._records: dict[str, _Record] = {}
        self._idempotency: dict[str, str] = {}
        self._pending: deque[str] = deque()
        self._inflight: dict[str, str] = {}
        self._closed = False
        self._condition = Condition()

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> InMemoryTransport:
        return cls(**dict(options))

    @property
    def capabilities(self) -> TransportCapabilities:
        return TransportCapabilities(
            name="memory",
            asynchronous=True,
            acknowledgements=True,
            max_transfer_size=(
                self._limits.max_metadata_bytes + self._limits.max_total_tensor_bytes
            ),
            delivery_guarantee="at-least-once",
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClosedError("in-memory transport is closed")

    def _depth(self) -> int:
        return len(self._pending) + len(self._inflight)

    def publish(
        self,
        payload: SerializedExperience,
        *,
        experience_id: str,
        idempotency_key: str,
        timeout: float | None = None,
        max_retries: int = 3,
    ) -> DeliveryReceipt:
        if not experience_id or not idempotency_key:
            raise ValueError("experience_id and idempotency_key are required")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        validate_transfer_limits(
            metadata_bytes=len(payload.metadata),
            tensor_sizes=(segment.nbytes for segment in payload.buffers),
            limits=self._limits,
        )
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            self._ensure_open()
            existing_id = self._idempotency.get(idempotency_key)
            if existing_id is not None:
                if self._records[existing_id].experience_id != experience_id:
                    raise DeliveryError("idempotency key reused for another experience")
                return self._receipt(existing_id)
            while self._depth() >= self._max_queue:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise BackpressureError("in-memory queue is full")
                self._condition.wait(remaining)
                self._ensure_open()
            self._publish_count += 1
            if self._failure_at_publish == self._publish_count:
                raise TransportError("injected publish failure")
            record_id = uuid.uuid4().hex
            copied = SerializedExperience(
                metadata=bytes(payload.metadata),
                buffers=tuple(
                    replace(segment, data=segment.materialize(), owner=None)
                    for segment in payload.buffers
                ),
            )
            self._records[record_id] = _Record(
                payload=copied,
                experience_id=experience_id,
                idempotency_key=idempotency_key,
                published_at=time.time(),
                max_retries=max_retries,
            )
            self._idempotency[idempotency_key] = record_id
            self._pending.append(record_id)
            self._condition.notify_all()
            return self._receipt(record_id)

    def _receipt(self, record_id: str) -> DeliveryReceipt:
        record = self._records[record_id]
        return DeliveryReceipt(
            receipt_id=record_id,
            experience_id=record.experience_id,
            idempotency_key=record.idempotency_key,
            accepted_at=record.published_at,
            _wait=lambda timeout: self._wait_receipt(record_id, timeout),
        )

    def _wait_receipt(self, record_id: str, timeout: float | None) -> ReceiptResult:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._records[record_id].state.terminal:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return ReceiptResult(
                        ReceiptState.EXPIRED,
                        "receipt wait timed out",
                        self._records[record_id].attempt,
                    )
                self._condition.wait(remaining)
            record = self._records[record_id]
            return ReceiptResult(record.state, record.reason, record.attempt)

    def receive(self, timeout: float | None = None) -> TransportDelivery | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            self._ensure_open()
            while not self._pending:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
                self._ensure_open()
            self._receive_count += 1
            if self._failure_at_receive == self._receive_count:
                raise TransportError("injected receive failure")
            record_id = self._pending.popleft()
            record = self._records[record_id]
            token = uuid.uuid4().hex
            record.token = token
            self._inflight[token] = record_id
            return TransportDelivery(
                token=token,
                experience_id=record.experience_id,
                idempotency_key=record.idempotency_key,
                payload=record.payload,
                attempt=record.attempt,
                published_at=record.published_at,
                max_retries=record.max_retries,
            )

    def _pop_inflight(self, token: str) -> tuple[str, _Record]:
        try:
            record_id = self._inflight.pop(token)
        except KeyError as error:
            raise DeliveryError("unknown or already-settled delivery token") from error
        return record_id, self._records[record_id]

    def ack(self, token: str) -> None:
        with self._condition:
            _, record = self._pop_inflight(token)
            record.state = ReceiptState.ACKED
            record.payload = SerializedExperience(metadata=b"{}", buffers=())
            self._condition.notify_all()

    def nack(self, token: str, reason: str, *, retry: bool = True) -> None:
        with self._condition:
            record_id, record = self._pop_inflight(token)
            record.reason = reason
            record.token = None
            if retry and record.attempt <= record.max_retries:
                record.attempt += 1
                record.state = ReceiptState.ACCEPTED
                self._pending.append(record_id)
            else:
                record.state = ReceiptState.NACKED
                record.payload = SerializedExperience(metadata=b"{}", buffers=())
            self._condition.notify_all()

    def reject(self, token: str, reason: str) -> None:
        with self._condition:
            _, record = self._pop_inflight(token)
            record.reason = reason
            record.state = ReceiptState.REJECTED
            record.payload = SerializedExperience(metadata=b"{}", buffers=())
            self._condition.notify_all()

    def cancel(self, receipt_id: str, reason: str = "cancelled") -> None:
        """Cancel a delivery that has not been handed to a consumer."""

        with self._condition:
            try:
                record = self._records[receipt_id]
            except KeyError as error:
                raise DeliveryError("unknown delivery receipt") from error
            if record.state.terminal:
                return
            if record.token is not None:
                raise DeliveryError("cannot cancel an inflight delivery")
            try:
                self._pending.remove(receipt_id)
            except ValueError as error:
                raise DeliveryError("delivery is not pending") from error
            record.reason = reason
            record.state = ReceiptState.CANCELLED
            record.payload = SerializedExperience(metadata=b"{}", buffers=())
            self._condition.notify_all()

    def health(self) -> HealthStatus:
        with self._condition:
            return HealthStatus(not self._closed, "closed" if self._closed else "ok", self._depth())

    def close(self, timeout: float | None = None) -> None:
        del timeout
        with self._condition:
            self._closed = True
            self._condition.notify_all()
