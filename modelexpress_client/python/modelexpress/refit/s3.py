# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct boto3 publication of immutable V0 canonical payload objects."""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import google_crc32c

from .api import S3Config
from .manifest import S3Object


class S3UploadError(RuntimeError):
    """An S3 object was not durably published and verified."""


class ImmutableS3Conflict(S3UploadError):
    """An immutable key already contains different bytes."""


@dataclass(frozen=True)
class UploadedS3Object:
    object: S3Object
    size: int


def _error_code(error: Exception) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return None
    detail = response.get("Error")
    if not isinstance(detail, dict):
        return None
    code = detail.get("Code")
    return str(code) if code is not None else None


def _checksum(data: bytes) -> tuple[str, str]:
    value = google_crc32c.value(data)
    raw = value.to_bytes(4, "big")
    return f"crc32c:{value:08x}", base64.b64encode(raw).decode("ascii")


def _version(response: dict[str, Any]) -> str | None:
    value = response.get("VersionId")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise S3UploadError("S3 returned a malformed object version")
    return value


def _relative(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if not value:
        return ""
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a normalized relative S3 key")
    return path.as_posix()


class S3Uploader:
    """Create-only S3 writer with checksum and byte-for-byte readback proof."""

    def __init__(
        self,
        config: S3Config,
        *,
        client: Any = None,
        request_timeout_seconds: float = 5.0,
    ) -> None:
        if not isinstance(config, S3Config):
            raise TypeError("config must be S3Config")
        if not isinstance(config.bucket, str) or not config.bucket.strip():
            raise ValueError("S3 bucket must be non-empty")
        if (
            isinstance(request_timeout_seconds, bool)
            or not isinstance(request_timeout_seconds, (int, float))
            or not math.isfinite(request_timeout_seconds)
            or request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be finite and positive")
        self._config = config
        self._prefix = _relative(config.prefix.strip("/"), "S3 prefix")
        self._owns_client = client is None
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "s3",
                endpoint_url=config.endpoint_url,
                region_name=config.region_name,
                config=Config(
                    connect_timeout=float(request_timeout_seconds),
                    read_timeout=float(request_timeout_seconds),
                    retries={"mode": "standard", "total_max_attempts": 3},
                    tcp_keepalive=True,
                ),
            )
        self._client = client
        self._closed = False

    def put(self, key: str, data: bytes) -> UploadedS3Object:
        if self._closed:
            raise S3UploadError("S3 uploader is closed")
        if not isinstance(data, bytes):
            raise TypeError("S3 payload must be bytes")
        relative = _relative(key, "S3 object key")
        if not relative:
            raise ValueError("S3 object key must be non-empty")
        full_key = f"{self._prefix}/{relative}" if self._prefix else relative
        checksum, encoded_checksum = _checksum(data)
        try:
            response = self._client.put_object(
                Bucket=self._config.bucket,
                Key=full_key,
                Body=data,
                ChecksumAlgorithm="CRC32C",
                ChecksumCRC32C=encoded_checksum,
                IfNoneMatch="*",
            )
        except Exception as exc:
            if _error_code(exc) not in {
                "409",
                "412",
                "ConditionalRequestConflict",
                "PreconditionFailed",
            }:
                raise S3UploadError(f"immutable S3 PUT failed: {exc}") from exc
            return self._verify_existing(full_key, data, checksum, encoded_checksum)
        if not isinstance(response, dict):
            raise S3UploadError("immutable S3 PUT returned an invalid response")
        stored = UploadedS3Object(
            S3Object(
                bucket=self._config.bucket,
                key=full_key,
                checksum=checksum,
                object_version=_version(response),
            ),
            len(data),
        )
        return self._verify(stored, data, encoded_checksum)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()

    def _verify_existing(
        self,
        key: str,
        data: bytes,
        checksum: str,
        encoded_checksum: str,
    ) -> UploadedS3Object:
        try:
            head = self._client.head_object(
                Bucket=self._config.bucket,
                Key=key,
                ChecksumMode="ENABLED",
            )
            if not isinstance(head, dict):
                raise S3UploadError("S3 HEAD returned an invalid response")
            stored = UploadedS3Object(
                S3Object(
                    bucket=self._config.bucket,
                    key=key,
                    checksum=checksum,
                    object_version=_version(head),
                ),
                len(data),
            )
            return self._verify(stored, data, encoded_checksum)
        except ImmutableS3Conflict:
            raise
        except Exception as exc:
            raise ImmutableS3Conflict(
                f"immutable S3 object conflict for {self._config.bucket}/{key}"
            ) from exc

    def _verify(
        self,
        stored: UploadedS3Object,
        expected: bytes,
        encoded_checksum: str,
    ) -> UploadedS3Object:
        location = stored.object
        head_request: dict[str, object] = {
            "Bucket": location.bucket,
            "Key": location.key,
            "ChecksumMode": "ENABLED",
        }
        get_request: dict[str, object] = {
            "Bucket": location.bucket,
            "Key": location.key,
            "ChecksumMode": "ENABLED",
        }
        if location.object_version is not None:
            head_request["VersionId"] = location.object_version
            get_request["VersionId"] = location.object_version
        try:
            head = self._client.head_object(**head_request)
            if not isinstance(head, dict):
                raise S3UploadError("S3 HEAD returned an invalid response")
            if head.get("ContentLength") != len(expected):
                raise ImmutableS3Conflict("immutable S3 object size differs")
            if head.get("ChecksumCRC32C") != encoded_checksum:
                raise ImmutableS3Conflict("immutable S3 object checksum differs")
            head_version = _version(head)
            if (
                location.object_version is not None
                and head_version != location.object_version
            ):
                raise ImmutableS3Conflict("immutable S3 object version differs")
            response = self._client.get_object(**get_request)
            if not isinstance(response, dict):
                raise S3UploadError("S3 GET returned an invalid response")
            body = response["Body"]
            try:
                actual = body.read(len(expected) + 1)
            finally:
                close = getattr(body, "close", None)
                if callable(close):
                    close()
            if bytes(actual) != expected:
                raise ImmutableS3Conflict("immutable S3 object bytes differ")
            resolved_version = location.object_version or head_version
            if resolved_version != location.object_version:
                stored = UploadedS3Object(
                    S3Object(
                        bucket=location.bucket,
                        key=location.key,
                        checksum=location.checksum,
                        object_version=resolved_version,
                    ),
                    stored.size,
                )
            return stored
        except (ImmutableS3Conflict, S3UploadError):
            raise
        except Exception as exc:
            raise S3UploadError(f"published S3 object is unreadable: {exc}") from exc
