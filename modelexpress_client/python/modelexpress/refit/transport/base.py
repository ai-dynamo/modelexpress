# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common immutable CANONICAL transport contract."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol, runtime_checkable

from ..codec import crc32c_hex
from ..manifest import DeltaLocation

_CRC32C = re.compile(r"[0-9a-f]{8}")


class TransportError(RuntimeError):
    """Canonical object transport failed."""


class TransportClosedError(TransportError):
    """The transport has already released its client resources."""


class ImmutableObjectConflict(TransportError):
    """A create-only key already names different physical bytes."""


class ObjectVerificationError(TransportError):
    """Published or fetched bytes failed physical integrity verification."""


@dataclass(frozen=True)
class StoredObject:
    location: DeltaLocation
    checksum: str
    size: int


@dataclass(frozen=True)
class CanonicalTransportIdentity:
    """Stable kind and namespace owned by one transport instance."""

    kind: str
    namespace: str


@runtime_checkable
class CanonicalTransport(Protocol):
    @property
    def identity(self) -> CanonicalTransportIdentity: ...

    def publish(self, key: str, data: bytes, checksum: str) -> StoredObject: ...

    def resolve(
        self,
        location: DeltaLocation,
        checksum: str,
        maximum_size: int,
    ) -> StoredObject: ...

    def fetch(self, stored: StoredObject) -> bytes: ...

    def verify(self, stored: StoredObject) -> None: ...

    def close(self) -> None: ...


def validate_checksum(checksum: str) -> None:
    if not isinstance(checksum, str) or not _CRC32C.fullmatch(checksum):
        raise ObjectVerificationError(
            "checksum must be bare eight-character lowercase CRC32C"
        )


def validate_maximum_size(maximum_size: int) -> None:
    if (
        not isinstance(maximum_size, int)
        or isinstance(maximum_size, bool)
        or maximum_size < 0
    ):
        raise ObjectVerificationError("maximum_size must be a non-negative integer")


def verify_payload(data: bytes, checksum: str, size: int, *, context: str) -> None:
    validate_checksum(checksum)
    if len(data) != size or crc32c_hex(data) != checksum:
        raise ObjectVerificationError(
            f"{context} verification failed: expected size={size} checksum={checksum}, "
            f"got size={len(data)} checksum={crc32c_hex(data)}"
        )


def validate_relative_key(key: str) -> PurePosixPath:
    if not isinstance(key, str) or not key or "\\" in key:
        raise ValueError("canonical transport needs a non-empty relative object key")
    segments = key.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ValueError(
            f"canonical transport needs a safe relative object key, got {key!r}"
        )
    path = PurePosixPath(key)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(
            f"canonical transport needs a safe relative object key, got {key!r}"
        )
    return path


def canonical_object_key(
    model_id: str,
    base_version: str,
    target_version: str,
    object_name: str,
) -> str:
    """Bind an immutable object namespace to the complete logical revision identity."""
    try:
        name = validate_relative_key(object_name)
    except ValueError as exc:
        raise ValueError(f"canonical object name is invalid: {object_name!r}") from exc
    if len(name.parts) != 1:
        raise ValueError("canonical object name must be one safe path segment")
    identity = json.dumps(
        [model_id, base_version, target_version],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    revision_key = hashlib.sha256(b"mx.canonical.object.v1\0" + identity).hexdigest()
    return f"canonical/{revision_key}/{name}"
