# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exceptions raised by :mod:`rlxfer`."""

from __future__ import annotations


class RlxferError(Exception):
    """Base class for all library errors."""


class SchemaValidationError(RlxferError, ValueError):
    """A schema error with producer, consumer, and experience context."""

    def __init__(
        self,
        message: str | None = None,
        *,
        field: str = "<unknown>",
        expected: object = "valid value",
        actual: object = None,
        producer_framework: str | None = None,
        producer_framework_version: str | None = None,
        consumer_framework: str | None = None,
        consumer_framework_version: str | None = None,
        experience_id: str | None = None,
    ) -> None:
        self.field = field
        self.expected = expected
        self.actual = actual
        self.producer_framework = producer_framework
        self.producer_framework_version = producer_framework_version
        self.consumer_framework = consumer_framework
        self.consumer_framework_version = consumer_framework_version
        self.experience_id = experience_id
        detail = message or f"invalid {field}: expected {expected!r}, got {_short_repr(actual)}"
        context = (
            f"experience_id={experience_id or '<unknown>'}, "
            f"producer={_framework(producer_framework, producer_framework_version)}, "
            f"consumer={_framework(consumer_framework, consumer_framework_version)}"
        )
        super().__init__(f"{detail} ({context})")


class CompatibilityError(RlxferError, ValueError):
    """Experience cannot safely be consumed by the requested consumer."""


class MigrationError(CompatibilityError):
    """An experience cannot be migrated to the requested schema version."""


class SerializationError(RlxferError, ValueError):
    """Metadata or tensor serialization failed."""


class IntegrityError(SerializationError):
    """Transferred data failed an integrity check."""


class TransportError(RlxferError):
    """A transport operation failed."""


class CapabilityError(TransportError):
    """A transport does not provide a required capability."""


class BackpressureError(TransportError):
    """A bounded transport cannot currently accept more data."""


class DeliveryError(TransportError):
    """A delivery could not be completed or acknowledged."""


class MissingDependencyError(RlxferError, ImportError):
    """An optional integration dependency is unavailable."""


class ClosedError(RlxferError):
    """An operation was attempted on a closed component."""


class TransferTimeoutError(TransportError, TimeoutError):
    """A transfer operation exceeded its deadline."""


def _framework(name: str | None, version: str | None) -> str:
    if name is None:
        return "<unknown>"
    return f"{name}@{version or '<unknown>'}"


def _short_repr(value: object, limit: int = 160) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."
