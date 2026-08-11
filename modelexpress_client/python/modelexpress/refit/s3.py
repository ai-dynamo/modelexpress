# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct immutable S3 transport for canonical payload objects."""

from __future__ import annotations

import base64

import google_crc32c

from .. import envs
from .manifest import S3Object


class ImmutableS3Conflict(RuntimeError):
    pass


def _error_code(error: Exception) -> str | None:
    try:
        return str(error.response["Error"]["Code"])
    except (AttributeError, KeyError, TypeError):
        return None


def _checksum(data: bytes) -> tuple[str, str]:
    value = google_crc32c.value(data)
    encoded = base64.b64encode(value.to_bytes(4, "big")).decode()
    return f"crc32c:{value:08x}", encoded


class S3Client:
    def __init__(
        self,
        endpoint_url: str | None = None,
        region_name: str | None = None,
    ) -> None:
        import boto3
        from botocore.config import Config as BotoConfig

        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            config=BotoConfig(
                max_pool_connections=max(1, envs.MX_REFIT_S3_MAX_POOL_CONNECTIONS)
            ),
        )

    def put(self, bucket: str, key: str, data: bytes) -> S3Object:
        checksum, encoded_checksum = _checksum(data)
        try:
            response = self.client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                ChecksumAlgorithm="CRC32C",
                ChecksumCRC32C=encoded_checksum,
                IfNoneMatch="*",
            )
            version = response.get("VersionId")
        except Exception as error:
            if _error_code(error) not in {
                "409",
                "412",
                "ConditionalRequestConflict",
                "PreconditionFailed",
            }:
                raise
            existing, version = self._read(bucket=bucket, key=key)
            if existing != data:
                raise ImmutableS3Conflict(
                    f"immutable S3 object conflict for {bucket}/{key}"
                ) from error

        return S3Object(
            bucket=bucket,
            key=key,
            checksum=checksum,
            object_version=version,
        )

    def get(self, location: S3Object) -> bytes:
        data, _version = self._read(
            bucket=location.bucket,
            key=location.key,
            version=location.object_version,
        )
        checksum, _encoded = _checksum(data)
        if checksum != location.checksum:
            raise ValueError(f"S3 checksum differs for {location.key}")
        return data

    def _read(
        self,
        bucket: str,
        key: str,
        version: str | None = None,
    ) -> tuple[bytes, str | None]:
        request = {"Bucket": bucket, "Key": key}
        if version is not None:
            request["VersionId"] = version
        response = self.client.get_object(**request)
        body = response["Body"]
        try:
            return body.read(), response.get("VersionId")
        finally:
            close = getattr(body, "close", None)
            if close is not None:
                close()

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if close is not None:
            close()
