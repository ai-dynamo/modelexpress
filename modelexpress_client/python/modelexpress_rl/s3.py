# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct immutable S3 writes for canonical refit artifacts."""

from __future__ import annotations

from urllib.parse import urlsplit


class ImmutableS3Conflict(RuntimeError):
    """An immutable key already contains different bytes."""


def _error_code(error: Exception) -> str | None:
    try:
        return str(error.response["Error"]["Code"])  # type: ignore[attr-defined]
    except (AttributeError, KeyError, TypeError):
        return None


def _parse_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not parsed.path.startswith("/")
        or parsed.path.startswith("//")
        or len(parsed.path) == 1
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"invalid S3 URI: {uri!r}")
    return parsed.netloc, parsed.path[1:]


class S3Client:
    """Small immutable S3 client for canonical refit artifacts."""

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

    def put(self, *, uri: str, data: bytes) -> None:
        """Create an immutable object, accepting an identical retry."""
        bucket, key = _parse_uri(uri)
        try:
            self._client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                IfNoneMatch="*",
            )
        except Exception as error:
            if _error_code(error) not in {
                "409",
                "412",
                "ConditionalRequestConflict",
                "PreconditionFailed",
            }:
                raise
            existing = self.get(uri)
            if existing != data:
                raise ImmutableS3Conflict(
                    f"immutable S3 object conflict for {bucket}/{key}"
                ) from error

    def get(self, uri: str) -> bytes:
        """Read one S3 object."""
        bucket, key = _parse_uri(uri)
        request = {"Bucket": bucket, "Key": key}
        response = self._client.get_object(**request)
        body = response["Body"]
        try:
            return body.read()
        finally:
            close = getattr(body, "close", None)
            if close is not None:
                close()

    def close(self) -> None:
        """Close the underlying SDK client when supported."""
        close = getattr(self._client, "close", None)
        if close is not None:
            close()


__all__ = ["ImmutableS3Conflict", "S3Client"]
