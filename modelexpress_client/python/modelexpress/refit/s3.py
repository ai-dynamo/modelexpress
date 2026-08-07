# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct immutable S3 publication for canonical payload objects."""

from __future__ import annotations

import base64
from typing import Any

import google_crc32c

from .. import envs
from .api import S3Config
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


class S3Uploader:
    def __init__(self, config: S3Config, *, client: Any = None) -> None:
        self.config = config
        self.owns_client = client is None
        if client is None:
            import boto3
            from botocore.config import Config as BotoConfig

            client = boto3.client(
                "s3",
                endpoint_url=config.endpoint_url,
                region_name=config.region_name,
                config=BotoConfig(
                    max_pool_connections=max(1, envs.MX_REFIT_S3_MAX_POOL_CONNECTIONS)
                ),
            )
        self.client = client

    def put(self, key: str, data: bytes) -> S3Object:
        key = "/".join(part for part in (self.config.prefix.strip("/"), key) if part)
        checksum, encoded_checksum = _checksum(data)
        try:
            response = self.client.put_object(
                Bucket=self.config.bucket,
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
            response = self.client.get_object(Bucket=self.config.bucket, Key=key)
            body = response["Body"]
            try:
                existing = body.read()
            finally:
                close = getattr(body, "close", None)
                if close is not None:
                    close()
            if existing != data:
                raise ImmutableS3Conflict(
                    f"immutable S3 object conflict for {self.config.bucket}/{key}"
                ) from error
            version = response.get("VersionId")

        return S3Object(
            bucket=self.config.bucket,
            key=key,
            checksum=checksum,
            object_version=version,
        )

    def close(self) -> None:
        if self.owns_client:
            close = getattr(self.client, "close", None)
            if close is not None:
                close()
