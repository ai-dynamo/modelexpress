# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct immutable S3 writes for canonical refit artifacts."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, wait
from io import BytesIO
from urllib.parse import urlsplit

from modelexpress_rl import envs as rl_envs


_MIN_MULTIPART_PART_BYTES = 5 * 1024**2
_MAX_MULTIPART_PART_BYTES = 5 * 1024**3
_MAX_MULTIPART_PARTS = 10_000
logger = logging.getLogger(__name__)


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
        from s3transfer.manager import TransferConfig, TransferManager

        self._multipart_threshold_bytes = rl_envs.MX_S3_MULTIPART_THRESHOLD_BYTES
        self._upload_part_bytes = rl_envs.MX_S3_UPLOAD_PART_BYTES
        if not (
            _MIN_MULTIPART_PART_BYTES
            <= self._upload_part_bytes
            <= _MAX_MULTIPART_PART_BYTES
        ):
            raise ValueError("MX_S3_UPLOAD_PART_BYTES must be between 5 MiB and 5 GiB")
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            region_name=region_name,
            config=BotoConfig(
                max_pool_connections=rl_envs.MX_S3_MAX_POOL_CONNECTIONS,
                retries={
                    "total_max_attempts": rl_envs.MX_S3_MAX_ATTEMPTS,
                    "mode": "standard",
                },
                tcp_keepalive=rl_envs.MX_S3_TCP_KEEPALIVE,
            ),
        )
        self._upload_pool = ThreadPoolExecutor(
            max_workers=rl_envs.MX_S3_UPLOAD_WORKERS,
            thread_name_prefix="modelexpress-s3-upload",
        )
        self._download_manager = TransferManager(
            self._client,
            config=TransferConfig(
                multipart_threshold=rl_envs.MX_S3_DOWNLOAD_RANGE_THRESHOLD_BYTES,
                multipart_chunksize=rl_envs.MX_S3_DOWNLOAD_RANGE_BYTES,
                max_request_concurrency=rl_envs.MX_S3_DOWNLOAD_WORKERS,
                io_chunksize=rl_envs.MX_S3_DOWNLOAD_IO_CHUNK_BYTES,
                max_io_queue_size=(rl_envs.MX_S3_DOWNLOAD_MAX_IN_MEMORY_CHUNKS),
                max_in_memory_download_chunks=(
                    rl_envs.MX_S3_DOWNLOAD_MAX_IN_MEMORY_CHUNKS
                ),
                num_download_attempts=rl_envs.MX_S3_MAX_ATTEMPTS,
            ),
        )

    def put(self, *, uri: str, data: bytes) -> None:
        """Create an immutable object, accepting an identical retry."""
        bucket, key = _parse_uri(uri)
        if len(data) >= self._multipart_threshold_bytes:
            self._put_multipart(bucket=bucket, key=key, uri=uri, data=data)
            return
        try:
            self._client.put_object(
                Bucket=bucket,
                Key=key,
                Body=data,
                IfNoneMatch="*",
            )
        except Exception as error:
            if _error_code(error) not in {"412", "PreconditionFailed"}:
                raise
            existing = self.get(uri)
            if existing != data:
                raise ImmutableS3Conflict(
                    f"immutable S3 object conflict for {bucket}/{key}"
                ) from error

    def _put_multipart(
        self,
        *,
        bucket: str,
        key: str,
        uri: str,
        data: bytes,
    ) -> None:
        part_count = (
            len(data) + self._upload_part_bytes - 1
        ) // self._upload_part_bytes
        if part_count > _MAX_MULTIPART_PARTS:
            raise ValueError("multipart upload exceeds 10,000 parts")
        upload_id = self._client.create_multipart_upload(
            Bucket=bucket,
            Key=key,
        )["UploadId"]

        def upload_part(index: int) -> dict[str, int | str]:
            start = index * self._upload_part_bytes
            response = self._client.upload_part(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=index + 1,
                Body=data[start : start + self._upload_part_bytes],
            )
            return {"ETag": response["ETag"], "PartNumber": index + 1}

        try:
            futures = [
                self._upload_pool.submit(upload_part, index)
                for index in range(part_count)
            ]
            try:
                parts = [future.result() for future in futures]
            except Exception:
                wait(futures)
                raise
            self._client.complete_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
                IfNoneMatch="*",
            )
        except Exception as error:
            try:
                self._client.abort_multipart_upload(
                    Bucket=bucket,
                    Key=key,
                    UploadId=upload_id,
                )
            except Exception:
                logger.warning("Failed to abort multipart upload", exc_info=True)
            if _error_code(error) not in {"412", "PreconditionFailed"}:
                raise
            existing = self.get(uri)
            if existing != data:
                raise ImmutableS3Conflict(
                    f"immutable S3 object conflict for {bucket}/{key}"
                ) from error

    def get(self, uri: str) -> bytes:
        """Read one S3 object."""
        bucket, key = _parse_uri(uri)
        target = BytesIO()
        self._download_manager.download(bucket, key, target).result()
        return target.getvalue()

    def size(self, uri: str) -> int:
        """Return one S3 object's byte size without downloading its payload."""
        bucket, key = _parse_uri(uri)
        return int(self._client.head_object(Bucket=bucket, Key=key)["ContentLength"])

    def close(self) -> None:
        """Close the underlying SDK client when supported."""
        self._upload_pool.shutdown()
        self._download_manager.shutdown()
        close = getattr(self._client, "close", None)
        if close is not None:
            close()


__all__ = ["ImmutableS3Conflict", "S3Client"]
