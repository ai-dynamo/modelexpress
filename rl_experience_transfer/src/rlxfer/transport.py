# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport contracts and dependency-injected construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from importlib import metadata as importlib_metadata
from types import MappingProxyType
from typing import Any, Protocol

from rlxfer.errors import CapabilityError
from rlxfer.serialization import SerializedExperience


class ReceiptState(str, Enum):
    """Application delivery state, distinct from byte-transfer completion."""

    ACCEPTED = "accepted"
    ACKED = "acked"
    NACKED = "nacked"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"

    @property
    def terminal(self) -> bool:
        """Whether no further delivery-state transition is possible."""

        return self not in {ReceiptState.ACCEPTED, ReceiptState.EXPIRED}


@dataclass(frozen=True, slots=True)
class TransportCapabilities:
    """Features and limits advertised by a transport instance."""

    name: str
    zero_copy: bool = False
    cpu_buffers: bool = True
    accelerator_buffers: frozenset[str] = frozenset()
    remote: bool = False
    scatter_gather: bool = False
    asynchronous: bool = False
    acknowledgements: bool = True
    persistence: bool = False
    max_transfer_size: int | None = None
    requires_registration: bool = False
    delivery_guarantee: str = "at-least-once"


@dataclass(frozen=True, slots=True)
class TransferPlan:
    """Transport-independent requirements decided before data movement."""

    strategy: str = "metadata+buffers"
    total_bytes: int = 0
    buffer_count: int = 0
    require_zero_copy: bool = False
    require_remote: bool = False
    require_async: bool = False
    require_persistence: bool = False
    device_types: frozenset[str] = frozenset({"cpu"})

    def check(self, capabilities: TransportCapabilities) -> None:
        """Fail before transfer when required capabilities are unavailable."""

        missing: list[str] = []
        if self.require_zero_copy and not capabilities.zero_copy:
            missing.append("zero_copy")
        if self.require_remote and not capabilities.remote:
            missing.append("remote")
        if self.require_async and not capabilities.asynchronous:
            missing.append("asynchronous")
        if self.require_persistence and not capabilities.persistence:
            missing.append("persistence")
        if "cpu" in self.device_types and not capabilities.cpu_buffers:
            missing.append("cpu_buffers")
        unsupported_devices = self.device_types - {"cpu"} - capabilities.accelerator_buffers
        missing.extend(f"device:{device}" for device in sorted(unsupported_devices))
        if (
            capabilities.max_transfer_size is not None
            and self.total_bytes > capabilities.max_transfer_size
        ):
            missing.append(f"max_transfer_size:{capabilities.max_transfer_size}")
        if missing:
            raise CapabilityError(
                f"transport {capabilities.name!r} cannot satisfy: {', '.join(missing)}"
            )


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """Small health response suitable for readiness checks."""

    healthy: bool
    detail: str = "ok"
    queue_depth: int = 0


@dataclass(frozen=True, slots=True)
class ReceiptResult:
    """Terminal or current state returned by receipt polling."""

    state: ReceiptState
    reason: str | None = None
    attempts: int = 0


WaitReceipt = Callable[[float | None], ReceiptResult]


@dataclass(frozen=True, slots=True)
class DeliveryReceipt:
    """Receipt returned after a transport accepts an experience."""

    receipt_id: str
    experience_id: str
    idempotency_key: str
    accepted_at: float
    _wait: WaitReceipt = field(repr=False, compare=False)

    def wait(self, timeout: float | None = None) -> ReceiptResult:
        """Wait for acknowledgement, rejection, or retry exhaustion."""

        return self._wait(timeout)


@dataclass(frozen=True, slots=True)
class TransportDelivery:
    """Opaque transport delivery passed to the consumer API."""

    token: str
    experience_id: str
    idempotency_key: str
    payload: SerializedExperience
    attempt: int
    published_at: float
    max_retries: int = 3


class ExperienceTransport(Protocol):
    """Minimal control/data-plane contract implemented by every backend."""

    @property
    def capabilities(self) -> TransportCapabilities: ...

    def publish(
        self,
        payload: SerializedExperience,
        *,
        experience_id: str,
        idempotency_key: str,
        timeout: float | None = None,
        max_retries: int = 3,
    ) -> DeliveryReceipt: ...

    def receive(self, timeout: float | None = None) -> TransportDelivery | None: ...

    def ack(self, token: str) -> None: ...

    def nack(self, token: str, reason: str, *, retry: bool = True) -> None: ...

    def reject(self, token: str, reason: str) -> None: ...

    def cancel(self, receipt_id: str, reason: str = "cancelled") -> None: ...

    def health(self) -> HealthStatus: ...

    def close(self, timeout: float | None = None) -> None: ...


TransportFactory = Callable[[Mapping[str, Any]], ExperienceTransport]


@dataclass(frozen=True, slots=True)
class TransportConfig:
    """Name plus backend-owned options; the core never branches on the name."""

    name: str
    options: Mapping[str, Any] = field(default_factory=dict)


class TransportRegistry:
    """Instance-scoped registry with optional Python entry-point discovery."""

    def __init__(self, factories: Mapping[str, TransportFactory] | None = None) -> None:
        self._factories = dict(factories or {})

    def register(self, name: str, factory: TransportFactory) -> None:
        if not name or name in self._factories:
            raise ValueError(f"transport {name!r} is already registered or invalid")
        self._factories[name] = factory

    def discover(self, group: str = "rlxfer.transports") -> None:
        for entry_point in importlib_metadata.entry_points(group=group):
            self.register(entry_point.name, entry_point.load())

    def create(self, config: TransportConfig) -> ExperienceTransport:
        try:
            factory = self._factories[config.name]
        except KeyError as error:
            available = ", ".join(sorted(self._factories)) or "none"
            raise ValueError(
                f"unknown transport {config.name!r}; registered transports: {available}"
            ) from error
        return factory(config.options)


def default_registry() -> TransportRegistry:
    """Return a new registry, avoiding process-global mutable registration state."""

    factories: Mapping[str, TransportFactory] = MappingProxyType(
        {
            "fallback": _fallback_factory,
            "filesystem": _filesystem_factory,
            "memory": _memory_factory,
            "nixl": _nixl_factory,
        }
    )
    return TransportRegistry(factories)


def _memory_factory(options: Mapping[str, Any]) -> ExperienceTransport:
    from rlxfer.transports.memory import InMemoryTransport

    return InMemoryTransport.from_options(options)


def _fallback_factory(options: Mapping[str, Any]) -> ExperienceTransport:
    from rlxfer.transports.fallback import FallbackTransport

    return FallbackTransport.from_options(options)


def _filesystem_factory(options: Mapping[str, Any]) -> ExperienceTransport:
    from rlxfer.transports.filesystem import FileSystemTransport

    return FileSystemTransport.from_options(options)


def _nixl_factory(options: Mapping[str, Any]) -> ExperienceTransport:
    from rlxfer.transports.nixl import NixlTransport

    return NixlTransport.from_options(options)


def create_transport(
    config: TransportConfig, registry: TransportRegistry | None = None
) -> ExperienceTransport:
    """Construct a transport without exposing backend conditionals to callers."""

    return (registry or default_registry()).create(config)
