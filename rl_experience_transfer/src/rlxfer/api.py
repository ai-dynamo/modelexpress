# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport-independent producer, consumer, and delivery APIs."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import TypeVar
from uuid import uuid4

from .adapters.base import ExperienceAdapter
from .contracts import ConsumerContract, SchemaMigrationRegistry
from .errors import ClosedError, DeliveryError, IntegrityError
from .model import SCHEMA_VERSION, ExperienceBatch, ExperienceMetadata, Trajectory
from .observability import Metrics, NullMetrics, structured_log
from .serialization import ExperienceSerializer, JsonExperienceSerializer
from .state import DeadLetter, DeliveryStateStore, InMemoryDeliveryState
from .transport import (
    DeliveryReceipt,
    ExperienceTransport,
    HealthStatus,
    TransferPlan,
    TransportCapabilities,
    TransportDelivery,
)

_EndpointT = TypeVar("_EndpointT", bound="_Endpoint")


class _Endpoint:
    """Shared transport lifecycle for producers and consumers."""

    _role = "endpoint"

    def __init__(
        self,
        transport: ExperienceTransport,
        serializer: ExperienceSerializer | None,
        metrics: Metrics | None,
        logger: logging.Logger | None,
    ) -> None:
        self.transport = transport
        self.serializer = serializer or JsonExperienceSerializer()
        self.metrics = metrics or NullMetrics()
        self.logger = logger
        self._closed = False
        self._close_lock = Lock()

    @property
    def capabilities(self) -> TransportCapabilities:
        """Return the selected transport's advertised capabilities."""

        return self.transport.capabilities

    def health(self) -> HealthStatus:
        """Return transport health."""

        return self.transport.health()

    def close(self, timeout: float | None = None) -> None:
        """Gracefully close the selected transport."""

        with self._close_lock:
            if self._closed:
                return
            self.transport.close(timeout)
            self._closed = True

    async def close_async(self, timeout: float | None = None) -> None:
        """Asynchronously close using a worker thread."""

        await asyncio.to_thread(self.close, timeout)

    def __enter__(self: _EndpointT) -> _EndpointT:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        self.close()

    async def __aenter__(self: _EndpointT) -> _EndpointT:
        self._ensure_open()
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        del exc_type, exc_value, traceback
        await self.close_async()

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClosedError(f"experience {self._role} is closed")

    def _log(self, event: str, **fields: object) -> None:
        if self.logger is not None:
            structured_log(self.logger, event, **fields)


@dataclass(slots=True)
class Delivery:
    """One received experience with explicit, single-use settlement methods."""

    experience: ExperienceBatch
    attempt: int
    published_at: float
    idempotency_key: str
    _adapter: ExperienceAdapter | None = field(default=None, repr=False)
    _ack_callback: Callable[[], None] = field(repr=False, default=lambda: None)
    _nack_callback: Callable[[str, bool], None] = field(
        repr=False, default=lambda _reason, _retry: None
    )
    _reject_callback: Callable[[str], None] = field(repr=False, default=lambda _reason: None)
    _state: str | None = field(init=False, default=None, repr=False)
    _lock: Lock = field(init=False, default_factory=Lock, repr=False)

    @property
    def batch(self) -> ExperienceBatch:
        """Alias for :attr:`experience`."""

        return self.experience

    @property
    def experience_id(self) -> str:
        """Return the canonical experience identifier."""

        return self.experience.experience_id

    @property
    def settled(self) -> bool:
        """Whether this delivery was acknowledged, nacked, or rejected."""

        return self._state is not None

    @property
    def state(self) -> str | None:
        """Return ``acknowledged``, ``nacked``, ``rejected``, or ``None``."""

        return self._state

    def to_framework(self) -> object:
        """Convert to native training input, or return the canonical batch unchanged."""

        if self._adapter is None:
            return self.experience
        return self._adapter.to_framework(self.experience)

    def ack(self) -> None:
        """Acknowledge successful, idempotent consumption."""

        self._settle("acknowledged", self._ack_callback)

    def nack(self, reason: str, *, retry: bool = True) -> None:
        """Negatively acknowledge, optionally requesting redelivery."""

        if not reason:
            raise ValueError("nack reason must be non-empty")
        self._settle("nacked", lambda: self._nack_callback(reason, retry))

    def reject(self, reason: str) -> None:
        """Permanently reject an incompatible or malformed experience."""

        if not reason:
            raise ValueError("rejection reason must be non-empty")
        self._settle("rejected", lambda: self._reject_callback(reason))

    async def ack_async(self) -> None:
        """Asynchronously acknowledge using a worker thread."""

        await asyncio.to_thread(self.ack)

    async def nack_async(self, reason: str, *, retry: bool = True) -> None:
        """Asynchronously nack using a worker thread."""

        await asyncio.to_thread(self.nack, reason, retry=retry)

    async def reject_async(self, reason: str) -> None:
        """Asynchronously reject using a worker thread."""

        await asyncio.to_thread(self.reject, reason)

    def _settle(self, state: str, callback: Callable[[], None]) -> None:
        with self._lock:
            if self._state is not None:
                raise DeliveryError(f"delivery is already {self._state}")
            callback()
            self._state = state


class ExperienceProducer(_Endpoint):
    """Validate, serialize, and publish framework-native or canonical experience."""

    _role = "producer"

    def __init__(
        self,
        transport: ExperienceTransport,
        adapter: ExperienceAdapter | None = None,
        serializer: ExperienceSerializer | None = None,
        metrics: Metrics | None = None,
        logger: logging.Logger | None = None,
        *,
        producer_id: str | None = None,
        producer_framework: str | None = None,
        producer_framework_version: str | None = None,
        transfer_plan: TransferPlan | None = None,
        consumer_contract: ConsumerContract | None = None,
        migration_registry: SchemaMigrationRegistry | None = None,
    ) -> None:
        super().__init__(transport, serializer, metrics, logger)
        self.adapter = adapter
        self.producer_id = producer_id or f"producer-{uuid4()}"
        self.producer_framework = producer_framework or (
            adapter.framework_name if adapter is not None else "canonical"
        )
        self.producer_framework_version = producer_framework_version or (
            adapter.framework_version if adapter is not None else SCHEMA_VERSION
        )
        self.transfer_plan = transfer_plan
        self.consumer_contract = consumer_contract
        self.migration_registry = migration_registry

    def publish(
        self,
        value: object,
        *,
        timeout: float | None = None,
        max_retries: int = 3,
        idempotency_key: str | None = None,
    ) -> DeliveryReceipt:
        """Publish a native rollout, canonical batch, or one trajectory."""

        self._ensure_open()
        batch = self._canonicalize(value)
        key = idempotency_key or batch.metadata.idempotency_key or batch.experience_id
        if not key:
            raise ValueError("idempotency key must be non-empty")
        if batch.metadata.idempotency_key != key:
            batch = replace(batch, metadata=replace(batch.metadata, idempotency_key=key))
        if self.consumer_contract is not None:
            batch = self.consumer_contract.negotiate(batch, self.migration_registry)
            if batch.metadata.idempotency_key != key:
                batch = replace(batch, metadata=replace(batch.metadata, idempotency_key=key))
        serialization_started = time.monotonic()
        try:
            serialized = self.serializer.serialize(batch)
        except Exception:
            self.metrics.increment("serialization_failures")
            raise
        self.metrics.observe(
            "serialization_latency_seconds", time.monotonic() - serialization_started
        )
        self.metrics.observe("metadata_bytes", float(len(serialized.metadata)))
        actual_devices = frozenset(
            segment.wire_device.split(":", 1)[0] for segment in serialized.buffers
        ) or frozenset({"cpu"})
        requirements = self.transfer_plan or TransferPlan(device_types=frozenset())
        replace(
            requirements,
            total_bytes=serialized.nbytes,
            buffer_count=len(serialized.buffers),
            device_types=requirements.device_types | actual_devices,
        ).check(self.transport.capabilities)
        transfer_started = time.monotonic()
        try:
            receipt = self.transport.publish(
                serialized,
                experience_id=batch.experience_id,
                idempotency_key=key,
                timeout=timeout,
                max_retries=max_retries,
            )
        except Exception:
            self.metrics.increment("transfer_failures")
            raise
        self.metrics.observe("transfer_latency_seconds", time.monotonic() - transfer_started)
        self.metrics.increment("produced_batches")
        self.metrics.increment("produced_trajectories", _trajectory_count(batch))
        self.metrics.increment("bytes_transferred", serialized.nbytes)
        self._log(
            "experience_published",
            experience_id=batch.experience_id,
            producer_id=batch.metadata.producer_id,
            producer_framework=batch.metadata.producer_framework,
            transport=self.transport.capabilities.name,
            trajectories=_trajectory_count(batch),
            bytes=serialized.nbytes,
        )
        return receipt

    def publish_batch(
        self,
        batch: ExperienceBatch,
        *,
        timeout: float | None = None,
        max_retries: int = 3,
        idempotency_key: str | None = None,
    ) -> DeliveryReceipt:
        """Typed convenience wrapper around :meth:`publish`."""

        return self.publish(
            batch,
            timeout=timeout,
            max_retries=max_retries,
            idempotency_key=idempotency_key,
        )

    def publish_trajectory(
        self,
        trajectory: Trajectory,
        *,
        timeout: float | None = None,
        max_retries: int = 3,
        idempotency_key: str | None = None,
    ) -> DeliveryReceipt:
        """Wrap and publish one canonical trajectory."""

        return self.publish(
            trajectory,
            timeout=timeout,
            max_retries=max_retries,
            idempotency_key=idempotency_key,
        )

    async def publish_async(
        self,
        value: object,
        *,
        timeout: float | None = None,
        max_retries: int = 3,
        idempotency_key: str | None = None,
    ) -> DeliveryReceipt:
        """Asynchronously publish using a worker thread."""

        return await asyncio.to_thread(
            self.publish,
            value,
            timeout=timeout,
            max_retries=max_retries,
            idempotency_key=idempotency_key,
        )

    def cancel(self, receipt: DeliveryReceipt, reason: str = "cancelled") -> None:
        """Cancel a publish that has not become inflight."""

        if not reason:
            raise ValueError("cancellation reason must be non-empty")
        self.transport.cancel(receipt.receipt_id, reason)

    async def cancel_async(self, receipt: DeliveryReceipt, reason: str = "cancelled") -> None:
        """Asynchronously cancel a pending publish."""

        await asyncio.to_thread(self.cancel, receipt, reason)

    def _canonicalize(self, value: object) -> ExperienceBatch:
        if isinstance(value, ExperienceBatch):
            return value
        if isinstance(value, Trajectory):
            metadata = ExperienceMetadata(
                producer_id=self.producer_id,
                producer_framework=self.producer_framework,
                producer_framework_version=self.producer_framework_version,
                policy_version=value.policy_version,
            )
            return ExperienceBatch(metadata=metadata, trajectories=(value,))
        if self.adapter is None:
            raise TypeError("publishing framework-native experience requires an ExperienceAdapter")
        return self.adapter.from_framework(value)


class ExperienceConsumer(_Endpoint):
    """Receive canonical experience with explicit settlement and duplicate suppression."""

    _role = "consumer"

    def __init__(
        self,
        transport: ExperienceTransport,
        adapter: ExperienceAdapter | None = None,
        serializer: ExperienceSerializer | None = None,
        metrics: Metrics | None = None,
        logger: logging.Logger | None = None,
        *,
        duplicate_cache_size: int = 4096,
        state_store: DeliveryStateStore | None = None,
        consumer_contract: ConsumerContract | None = None,
        migration_registry: SchemaMigrationRegistry | None = None,
    ) -> None:
        if isinstance(duplicate_cache_size, bool) or duplicate_cache_size < 1:
            raise ValueError("duplicate_cache_size must be positive")
        super().__init__(transport, serializer, metrics, logger)
        self.adapter = adapter
        self.consumer_contract = consumer_contract
        self.migration_registry = migration_registry
        self.state_store = state_store or InMemoryDeliveryState(duplicate_cache_size)

    def receive(self, timeout: float | None = None) -> Delivery | None:
        """Receive the next non-duplicate delivery, blocking up to ``timeout`` seconds."""

        self._ensure_open()
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            raw = self.transport.receive(remaining)
            if raw is None:
                return None
            if self._was_seen(raw.idempotency_key):
                self.transport.ack(raw.token)
                self.metrics.increment("duplicate_deliveries")
                self._log(
                    "duplicate_delivery_acknowledged",
                    experience_id=raw.experience_id,
                    transport=self.transport.capabilities.name,
                )
                continue
            return self._prepare(raw)

    async def receive_async(self, timeout: float | None = None) -> Delivery | None:
        """Asynchronously receive using a worker thread."""

        return await asyncio.to_thread(self.receive, timeout)

    def _prepare(self, raw: TransportDelivery) -> Delivery:
        started = time.monotonic()
        try:
            batch = self.serializer.deserialize(raw.payload)
            if batch.experience_id != raw.experience_id:
                raise IntegrityError("transport and payload experience IDs differ")
            if (
                batch.metadata.idempotency_key is not None
                and batch.metadata.idempotency_key != raw.idempotency_key
            ):
                raise IntegrityError("transport and payload idempotency keys differ")
            if self.consumer_contract is not None:
                batch = self.consumer_contract.negotiate(batch, self.migration_registry)
            if self.adapter is not None:
                batch.validate(
                    consumer_framework=self.adapter.framework_name,
                    consumer_framework_version=self.adapter.framework_version,
                )
                self.adapter.validate_compatible(batch)
        except Exception as error:
            self.metrics.increment("rejected_deliveries")
            reason = _safe_reason(error)
            self._log(
                "experience_rejected",
                experience_id=raw.experience_id,
                error_type=type(error).__name__,
                transport=self.transport.capabilities.name,
            )
            self._record_dead_letter(raw, reason)
            try:
                self.transport.reject(raw.token, reason)
            except Exception:
                self.metrics.increment("cleanup_failures")
            raise
        self.metrics.observe("deserialization_latency_seconds", time.monotonic() - started)
        self.metrics.increment("received_batches")
        self._log(
            "experience_received",
            experience_id=batch.experience_id,
            producer_framework=batch.metadata.producer_framework,
            transport=self.transport.capabilities.name,
            attempt=raw.attempt,
        )

        def ack() -> None:
            ack_started = time.monotonic()
            self._remember(raw.idempotency_key)
            self.transport.ack(raw.token)
            self.metrics.increment("consumed_batches")
            self.metrics.increment("consumed_trajectories", _trajectory_count(batch))
            self.metrics.observe("acknowledgement_latency_seconds", time.monotonic() - ack_started)
            self.metrics.observe(
                "end_to_end_latency_seconds", max(0.0, time.time() - raw.published_at)
            )
            self._log(
                "experience_acknowledged",
                experience_id=batch.experience_id,
                transport=self.transport.capabilities.name,
            )

        def nack(reason: str, retry: bool) -> None:
            if not retry or raw.attempt > raw.max_retries:
                self._record_dead_letter(raw, reason)
            self.transport.nack(raw.token, reason, retry=retry)
            self.metrics.increment("nacks")
            if retry:
                self.metrics.increment("retries")
            self._log(
                "experience_nacked",
                experience_id=batch.experience_id,
                retry=retry,
                transport=self.transport.capabilities.name,
            )

        def reject(reason: str) -> None:
            self._record_dead_letter(raw, reason)
            self.transport.reject(raw.token, reason)
            self.metrics.increment("rejected_deliveries")
            self._log(
                "experience_rejected",
                experience_id=batch.experience_id,
                transport=self.transport.capabilities.name,
            )

        return Delivery(
            experience=batch,
            attempt=raw.attempt,
            published_at=raw.published_at,
            idempotency_key=raw.idempotency_key,
            _adapter=self.adapter,
            _ack_callback=ack,
            _nack_callback=nack,
            _reject_callback=reject,
        )

    def _was_seen(self, key: str) -> bool:
        return self.state_store.was_consumed(key)

    def _remember(self, key: str) -> None:
        self.state_store.mark_consumed(key)

    def _record_dead_letter(self, raw: TransportDelivery, reason: str) -> None:
        try:
            self.state_store.record_dead_letter(
                DeadLetter(
                    experience_id=raw.experience_id,
                    idempotency_key=raw.idempotency_key,
                    reason=reason[:2048],
                    attempt=raw.attempt,
                )
            )
        except Exception:
            self.metrics.increment("state_store_failures")


def _trajectory_count(batch: ExperienceBatch) -> int:
    return len(batch.trajectories) + sum(len(episode.trajectories) for episode in batch.episodes)


def _safe_reason(error: Exception) -> str:
    return f"{type(error).__name__}: delivery validation failed"
