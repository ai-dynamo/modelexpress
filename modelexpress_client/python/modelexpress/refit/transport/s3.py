# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create-only checksum-verified S3 CANONICAL transport."""

from __future__ import annotations

import base64
import math
from typing import Any

from ..codec import crc32c_hex
from ..manifest import DeltaLocation, S3Location
from .base import (
    CanonicalTransportIdentity,
    ImmutableObjectConflict,
    ObjectVerificationError,
    StoredObject,
    TransportClosedError,
    TransportError,
    validate_checksum,
    validate_maximum_size,
    validate_relative_key,
    verify_payload,
)


def _s3_crc32c(checksum: str) -> str:
    return base64.b64encode(int(checksum, 16).to_bytes(4, "big")).decode("ascii")


def _error_code(error: Exception) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return None
    detail = response.get("Error")
    return detail.get("Code") if isinstance(detail, dict) else None


def _object_version(response: dict[str, Any]) -> str | None:
    version = response.get("VersionId")
    if version is None:
        return None
    if not isinstance(version, str) or not version:
        raise ObjectVerificationError("S3 object returned a malformed version")
    return version


class S3CanonicalTransport:
    """Immutable S3 keys with optional object-version pinning."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        *,
        client: Any = None,
        request_timeout_seconds: float = 5.0,
    ) -> None:
        if not bucket:
            raise ValueError("S3 bucket must be non-empty")
        if (
            isinstance(request_timeout_seconds, bool)
            or not isinstance(request_timeout_seconds, (int, float))
            or not math.isfinite(request_timeout_seconds)
            or request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be finite and positive")
        if prefix:
            prefix = str(validate_relative_key(prefix))
        self._bucket = bucket
        self._prefix = prefix
        self._owns_client = client is None
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "s3",
                config=Config(
                    connect_timeout=float(request_timeout_seconds),
                    read_timeout=float(request_timeout_seconds),
                    retries={"mode": "standard", "total_max_attempts": 3},
                    tcp_keepalive=True,
                ),
            )
        self._client = client
        self._closed = False
        self._published: dict[str, StoredObject] = {}

    @property
    def identity(self) -> CanonicalTransportIdentity:
        namespace = f"s3://{self._bucket}"
        if self._prefix:
            namespace = f"{namespace}/{self._prefix}"
        return CanonicalTransportIdentity("s3", namespace)

    def publish(self, key: str, data: bytes, checksum: str) -> StoredObject:
        self._ensure_open()
        validate_checksum(checksum)
        if crc32c_hex(data) != checksum:
            raise ObjectVerificationError(
                "payload checksum does not match bytes before publish"
            )
        full_key = self._full_key(key)
        encoded_checksum = _s3_crc32c(checksum)
        response = None
        for attempt in range(3):
            try:
                response = self._client.put_object(
                    Bucket=self._bucket,
                    Key=full_key,
                    Body=data,
                    ChecksumAlgorithm="CRC32C",
                    ChecksumCRC32C=encoded_checksum,
                    IfNoneMatch="*",
                )
                break
            except Exception as exc:
                code = _error_code(exc)
                if code in {"412", "PreconditionFailed"}:
                    return self._existing_retry(full_key, data, checksum)
                if code in {"409", "ConditionalRequestConflict"}:
                    try:
                        return self._existing_retry(full_key, data, checksum)
                    except ObjectVerificationError:
                        if attempt < 2:
                            continue
                raise TransportError(f"S3 immutable PUT failed: {exc}") from exc
        if response is None:
            raise TransportError(
                "S3 immutable PUT exhausted conditional-conflict retries"
            )
        if not isinstance(response, dict):
            raise ObjectVerificationError("S3 immutable PUT response is invalid")
        version = _object_version(response)
        stored = StoredObject(
            location=DeltaLocation(
                s3=S3Location(
                    bucket=self._bucket,
                    key=full_key,
                    object_version=version,
                )
            ),
            checksum=checksum,
            size=len(data),
        )
        verified_version = self._verify_metadata(stored)
        if version is None and verified_version is not None:
            stored = StoredObject(
                location=DeltaLocation(
                    s3=S3Location(
                        bucket=self._bucket,
                        key=full_key,
                        object_version=verified_version,
                    )
                ),
                checksum=checksum,
                size=len(data),
            )
        self.verify(stored)
        self._published[full_key] = stored
        return stored

    def resolve(
        self,
        location: DeltaLocation,
        checksum: str,
        maximum_size: int,
    ) -> StoredObject:
        self._ensure_open()
        validate_checksum(checksum)
        validate_maximum_size(maximum_size)
        provisional = StoredObject(location=location, checksum=checksum, size=0)
        s3_location = self._validated_location(provisional)
        response = self._head(s3_location)
        size = response.get("ContentLength")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ObjectVerificationError("S3 object metadata has an invalid size")
        if size > maximum_size:
            raise ObjectVerificationError("S3 object size exceeds maximum_size")
        if response.get("ChecksumCRC32C") != _s3_crc32c(checksum):
            raise ObjectVerificationError(
                "S3 object metadata failed checksum verification"
            )
        version = _object_version(response)
        if (
            s3_location.object_version is not None
            and version != s3_location.object_version
        ):
            raise ObjectVerificationError(
                "S3 object metadata returned a different or missing version"
            )
        resolved_location = s3_location
        if resolved_location.object_version is None and version is not None:
            resolved_location = S3Location(
                bucket=resolved_location.bucket,
                key=resolved_location.key,
                object_version=version,
            )
        stored = StoredObject(
            location=DeltaLocation(s3=resolved_location),
            checksum=checksum,
            size=size,
        )
        self.verify(stored)
        return stored

    def fetch(self, stored: StoredObject) -> bytes:
        self._ensure_open()
        location = self._validated_location(stored)
        request: dict[str, object] = {"Bucket": self._bucket, "Key": location.key}
        if location.object_version is not None:
            request["VersionId"] = location.object_version
        try:
            response = self._client.get_object(**request)
            if not isinstance(response, dict):
                raise ObjectVerificationError("S3 object response is invalid")
            if (
                location.object_version is not None
                and response.get("VersionId") != location.object_version
            ):
                raise ObjectVerificationError(
                    "S3 object returned a different or missing version"
                )
            body = response["Body"]
            try:
                data = body.read(stored.size + 1)
            finally:
                close = getattr(body, "close", None)
                if callable(close):
                    close()
        except Exception as exc:
            raise ObjectVerificationError(f"S3 object is unreadable: {exc}") from exc
        if not isinstance(data, bytes):
            data = bytes(data)
        verify_payload(data, stored.checksum, stored.size, context="S3 object")
        return data

    def verify(self, stored: StoredObject) -> None:
        self._verify_metadata(stored)
        self.fetch(stored)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise TransportClosedError("S3 canonical transport is closed")

    def _full_key(self, key: str) -> str:
        relative = str(validate_relative_key(key))
        return f"{self._prefix}/{relative}" if self._prefix else relative

    def _validated_location(self, stored: StoredObject) -> S3Location:
        if (
            not isinstance(stored, StoredObject)
            or not isinstance(stored.size, int)
            or isinstance(stored.size, bool)
            or stored.size < 0
        ):
            raise ObjectVerificationError("S3 object metadata has an invalid size")
        if not isinstance(stored.location, DeltaLocation):
            raise ObjectVerificationError(
                "S3 transport received a location for another transport"
            )
        location = stored.location.s3
        if location is None or location.bucket != self._bucket:
            raise ObjectVerificationError(
                "S3 transport received a location for another transport"
            )
        expected_prefix = f"{self._prefix}/" if self._prefix else ""
        if not location.key.startswith(expected_prefix):
            raise ObjectVerificationError("S3 object is outside the configured prefix")
        relative = location.key[len(expected_prefix) :]
        try:
            normalized = str(validate_relative_key(relative))
        except ValueError as exc:
            raise ObjectVerificationError("S3 object key is not canonical") from exc
        if self._full_key(normalized) != location.key:
            raise ObjectVerificationError("S3 object is outside the configured prefix")
        if location.object_version is not None and (
            not isinstance(location.object_version, str) or not location.object_version
        ):
            raise ObjectVerificationError("S3 object has a malformed version")
        validate_checksum(stored.checksum)
        return location

    def _head(self, location: S3Location) -> dict[str, Any]:
        request: dict[str, object] = {
            "Bucket": self._bucket,
            "Key": location.key,
            "ChecksumMode": "ENABLED",
        }
        if location.object_version is not None:
            request["VersionId"] = location.object_version
        try:
            response = self._client.head_object(**request)
        except Exception as exc:
            raise ObjectVerificationError(
                f"S3 object metadata is unreadable: {exc}"
            ) from exc
        if not isinstance(response, dict):
            raise ObjectVerificationError("S3 object metadata response is invalid")
        return response

    def _verify_metadata(self, stored: StoredObject) -> str | None:
        location = self._validated_location(stored)
        response = self._head(location)
        if response.get("ContentLength") != stored.size or response.get(
            "ChecksumCRC32C"
        ) != _s3_crc32c(stored.checksum):
            raise ObjectVerificationError(
                "S3 object metadata failed size/checksum verification"
            )
        version = _object_version(response)
        if location.object_version is not None and version != location.object_version:
            raise ObjectVerificationError(
                "S3 object metadata returned a different or missing version"
            )
        return version

    def _existing_retry(self, key: str, data: bytes, checksum: str) -> StoredObject:
        known = self._published.get(key)
        if known is not None:
            existing = self.fetch(known)
            if existing != data or known.checksum != checksum:
                raise ImmutableObjectConflict(
                    f"immutable object conflict for S3 key {key}"
                )
            self._verify_metadata(known)
            return known
        provisional = S3Location(bucket=self._bucket, key=key)
        response = self._head(provisional)
        version = _object_version(response)
        stored = StoredObject(
            location=DeltaLocation(
                s3=S3Location(
                    bucket=self._bucket,
                    key=key,
                    object_version=version,
                )
            ),
            checksum=checksum,
            size=len(data),
        )
        try:
            existing = self.fetch(stored)
        except ObjectVerificationError as exc:
            raise ImmutableObjectConflict(
                f"immutable object conflict for S3 key {key}"
            ) from exc
        if existing != data:
            raise ImmutableObjectConflict(f"immutable object conflict for S3 key {key}")
        self._verify_metadata(stored)
        return stored
