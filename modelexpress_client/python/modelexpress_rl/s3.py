# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct immutable S3 writes for canonical refit artifacts."""

from __future__ import annotations

import base64
from dataclasses import dataclass

import google_crc32c


@dataclass(frozen=True)
class S3Object:
    """Location and transport checksum of one S3 object."""

    bucket: str
    key: str
    checksum: str
    object_version: str | None = None


class ImmutableS3Conflict(RuntimeError):
    """An immutable key already contains different bytes."""


def _error_code(error: Exception) -> str | None:
    try:
        return str(error.response["Error"]["Code"])  # type: ignore[attr-defined]
    except (AttributeError, KeyError, TypeError):
        return None


def _checksum(data: bytes) -> tuple[str, str]:
    value = google_crc32c.value(data)
    encoded = base64.b64encode(value.to_bytes(4, "big")).decode()
    return f"crc32c:{value:08x}", encoded


class S3Client:
    """Small immutable PUT client used by the trainer publisher."""

    def __init__(
        self,
        *,
        endpoint_url: str | None = None,
        region_name: str | None = None,
    ) -> None:
        import boto3
        from botocore.config import Config as BotoConfig

        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            config=BotoConfig(max_pool_connections=32),
        )

    def put(self, *, bucket: str, key: str, data: bytes) -> S3Object:
        """Create an immutable object, accepting an identical retry."""
        checksum, encoded_checksum = _checksum(data)
        try:
            response = self._client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                ChecksumAlgorithm="CRC32C",
                ChecksumCRC32C=encoded_checksum,
                IfNoneMatch="*",
            )
            object_version = response.get("VersionId")
        except Exception as error:
            if _error_code(error) not in {
                "409",
                "412",
                "ConditionalRequestConflict",
                "PreconditionFailed",
            }:
                raise
            existing, object_version = self._read(bucket=bucket, key=key)
            if existing != data:
                raise ImmutableS3Conflict(
                    f"immutable S3 object conflict for {bucket}/{key}"
                ) from error
        return S3Object(
            bucket=bucket,
            key=key,
            checksum=checksum,
            object_version=object_version,
        )

    def _read(self, *, bucket: str, key: str) -> tuple[bytes, str | None]:
        response = self._client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        try:
            return body.read(), response.get("VersionId")
        finally:
            close = getattr(body, "close", None)
            if close is not None:
                close()

    def close(self) -> None:
        """Close the underlying SDK client when supported."""
        close = getattr(self._client, "close", None)
        if close is not None:
            close()


__all__ = ["ImmutableS3Conflict", "S3Client", "S3Object"]
