# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free metric hooks and content-safe structured logging."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Lock
from typing import Protocol

MetricAttributes = Mapping[str, str | int | float | bool]
_SENSITIVE_NAMES = (
    "prompt",
    "response",
    "completion",
    "content",
    "message",
    "text",
    "token",
    "input_ids",
    "output_ids",
)


class Metrics(Protocol):
    """Small vendor-neutral metrics interface used by the core library."""

    def increment(
        self,
        name: str,
        value: int = 1,
        attributes: MetricAttributes | None = None,
    ) -> None:
        """Increase a counter."""

    def observe(
        self,
        name: str,
        value: float,
        attributes: MetricAttributes | None = None,
    ) -> None:
        """Record a measurement."""


@dataclass(slots=True)
class NullMetrics:
    """No-op default that avoids global state and optional dependencies."""

    def increment(
        self,
        name: str,
        value: int = 1,
        attributes: MetricAttributes | None = None,
    ) -> None:
        del name, value, attributes

    def observe(
        self,
        name: str,
        value: float,
        attributes: MetricAttributes | None = None,
    ) -> None:
        del name, value, attributes


@dataclass(slots=True)
class InMemoryMetrics:
    """Thread-safe metric recorder for deterministic tests and examples."""

    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    observations: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _lock: Lock = field(default_factory=Lock, repr=False)

    def increment(
        self,
        name: str,
        value: int = 1,
        attributes: MetricAttributes | None = None,
    ) -> None:
        key = _metric_key(name, attributes)
        with self._lock:
            self.counters[key] += value

    def observe(
        self,
        name: str,
        value: float,
        attributes: MetricAttributes | None = None,
    ) -> None:
        key = _metric_key(name, attributes)
        with self._lock:
            self.observations[key].append(value)

    def snapshot(self) -> tuple[dict[str, int], dict[str, tuple[float, ...]]]:
        """Return an immutable-value copy of all recorded metrics."""

        with self._lock:
            return dict(self.counters), {
                name: tuple(values) for name, values in self.observations.items()
            }


def structured_log(logger: logging.Logger, event: str, **fields: object) -> None:
    """Log one event after recursively removing prompt/response-like content."""

    logger.info(event, extra={"rlxfer": _redact(fields)})


def _metric_key(name: str, attributes: MetricAttributes | None) -> str:
    if not attributes:
        return name
    suffix = ",".join(f"{key}={attributes[key]}" for key in sorted(attributes))
    return f"{name}{{{suffix}}}"


def _redact(value: object) -> object:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            name = str(key)
            lowered = name.lower()
            if any(secret in lowered for secret in _SENSITIVE_NAMES):
                result[name] = "<redacted>"
            else:
                result[name] = _redact(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value
