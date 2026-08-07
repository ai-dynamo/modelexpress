# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordered transport fallback with routed receipts and delivery tokens."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from rlxfer.errors import (
    BackpressureError,
    CapabilityError,
    ClosedError,
    DeliveryError,
    TransportError,
)
from rlxfer.serialization import SerializedExperience
from rlxfer.transport import (
    DeliveryReceipt,
    ExperienceTransport,
    HealthStatus,
    TransportCapabilities,
    TransportDelivery,
)


class FallbackTransport:
    """Try healthy transports in order while preserving settlement routing."""

    def __init__(
        self,
        transports: Sequence[ExperienceTransport],
        *,
        poll_interval: float = 0.05,
        fallback_exceptions: Sequence[type[Exception]] = (
            BackpressureError,
            CapabilityError,
            ClosedError,
        ),
    ) -> None:
        if not transports or len({id(transport) for transport in transports}) != len(transports):
            raise ValueError("fallback transports must be non-empty and unique")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if not fallback_exceptions or any(
            not isinstance(error, type) or not issubclass(error, Exception)
            for error in fallback_exceptions
        ):
            raise ValueError("fallback_exceptions must contain exception types")
        self._transports = tuple(transports)
        self._poll_interval = poll_interval
        self._fallback_exceptions = tuple(fallback_exceptions)
        self._next_receive = 0

    @classmethod
    def from_options(cls, options: Mapping[str, Any]) -> FallbackTransport:
        return cls(**dict(options))

    @property
    def capabilities(self) -> TransportCapabilities:
        """Report only capabilities guaranteed by every fallback path."""

        capabilities = tuple(transport.capabilities for transport in self._transports)
        finite_sizes = [
            item.max_transfer_size for item in capabilities if item.max_transfer_size is not None
        ]
        accelerators = set(capabilities[0].accelerator_buffers)
        for item in capabilities[1:]:
            accelerators.intersection_update(item.accelerator_buffers)
        names = ",".join(item.name for item in capabilities)
        return TransportCapabilities(
            name=f"fallback({names})",
            zero_copy=all(item.zero_copy for item in capabilities),
            cpu_buffers=all(item.cpu_buffers for item in capabilities),
            accelerator_buffers=frozenset(accelerators),
            remote=all(item.remote for item in capabilities),
            scatter_gather=all(item.scatter_gather for item in capabilities),
            asynchronous=all(item.asynchronous for item in capabilities),
            acknowledgements=all(item.acknowledgements for item in capabilities),
            persistence=all(item.persistence for item in capabilities),
            max_transfer_size=min(finite_sizes) if finite_sizes else None,
            requires_registration=any(item.requires_registration for item in capabilities),
            delivery_guarantee="at-least-once",
        )

    def publish(
        self,
        payload: SerializedExperience,
        *,
        experience_id: str,
        idempotency_key: str,
        timeout: float | None = None,
        max_retries: int = 3,
    ) -> DeliveryReceipt:
        failures: list[Exception] = []
        deadline = None if timeout is None else time.monotonic() + timeout
        for index, transport in enumerate(self._transports):
            if not _healthy(transport):
                continue
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                receipt = transport.publish(
                    payload,
                    experience_id=experience_id,
                    idempotency_key=idempotency_key,
                    timeout=remaining,
                    max_retries=max_retries,
                )
            except self._fallback_exceptions as error:
                failures.append(error)
                continue
            return replace(receipt, receipt_id=_routed(index, receipt.receipt_id))
        detail = ", ".join(type(error).__name__ for error in failures) or "no healthy transport"
        raise TransportError(f"all fallback transports failed before acceptance: {detail}") from (
            failures[-1] if failures else None
        )

    def receive(self, timeout: float | None = None) -> TransportDelivery | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            succeeded = False
            failures: list[Exception] = []
            for _ in self._transports:
                index = self._next_receive
                self._next_receive = (index + 1) % len(self._transports)
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                wait = (
                    self._poll_interval
                    if remaining is None
                    else min(self._poll_interval, remaining)
                )
                try:
                    delivery = self._transports[index].receive(wait)
                    succeeded = True
                except (TransportError, ClosedError) as error:
                    failures.append(error)
                    continue
                if delivery is not None:
                    return replace(delivery, token=_routed(index, delivery.token))
            if not succeeded:
                raise TransportError(
                    "all fallback transports failed while receiving"
                ) from failures[-1]
            if deadline is not None and time.monotonic() >= deadline:
                return None

    def ack(self, token: str) -> None:
        transport, inner = self._route(token)
        transport.ack(inner)

    def nack(self, token: str, reason: str, *, retry: bool = True) -> None:
        transport, inner = self._route(token)
        transport.nack(inner, reason, retry=retry)

    def reject(self, token: str, reason: str) -> None:
        transport, inner = self._route(token)
        transport.reject(inner, reason)

    def cancel(self, receipt_id: str, reason: str = "cancelled") -> None:
        transport, inner = self._route(receipt_id)
        transport.cancel(inner, reason)

    def health(self) -> HealthStatus:
        statuses = tuple(_health(transport) for transport in self._transports)
        unhealthy = [
            self._transports[index].capabilities.name
            for index, status in enumerate(statuses)
            if not status.healthy
        ]
        return HealthStatus(
            any(status.healthy for status in statuses),
            "ok" if not unhealthy else f"unhealthy: {', '.join(unhealthy)}",
            sum(status.queue_depth for status in statuses),
        )

    def close(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        failure: Exception | None = None
        for transport in self._transports:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                transport.close(remaining)
            except Exception as error:
                failure = failure or error
        if failure is not None:
            raise TransportError("one or more fallback transports failed to close") from failure

    def _route(self, value: str) -> tuple[ExperienceTransport, str]:
        prefix, separator, inner = value.partition(":")
        if not separator or not inner or not prefix.isdigit():
            raise DeliveryError("fallback route is malformed")
        try:
            index = int(prefix)
            if not 0 <= index < len(self._transports):
                raise IndexError(index)
            return self._transports[index], inner
        except (ValueError, IndexError) as error:
            raise DeliveryError("fallback route is unknown") from error


def _routed(index: int, value: str) -> str:
    return f"{index}:{value}"


def _healthy(transport: ExperienceTransport) -> bool:
    return _health(transport).healthy


def _health(transport: ExperienceTransport) -> HealthStatus:
    try:
        return transport.health()
    except Exception as error:
        return HealthStatus(False, type(error).__name__)
