# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""S3-compatible storage helpers for delta weight directories."""

from __future__ import annotations

import logging
import os
import re
import shutil
import tempfile
import time

logger = logging.getLogger("modelexpress.rl.s3_delta")


def _mb(num_bytes: int) -> float:
    """Bytes as decimal megabytes (MB = 1e6 bytes)."""
    return num_bytes / 1e6


def _bandwidth_mb_s(num_bytes: int, elapsed_s: float) -> float:
    """MB/s over a measured interval; 0.0 when no time or no bytes elapsed."""
    if elapsed_s <= 0 or num_bytes <= 0:
        return 0.0
    return _mb(num_bytes) / elapsed_s


def s3_client():
    """Create the path-style S3 client used by the delta hooks."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ["MX_S3_ENDPOINT"],
        config=Config(
            s3={"addressing_style": "path"},
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )


def _key_prefix(*parts: str) -> str:
    return "/".join(part.strip("/") for part in parts if part.strip("/"))


def upload_version_dir(s3, bucket: str, prefix: str, version_dir: str) -> int:
    """Upload every completed file in a version directory and return the count.

    Logs an ``mx_upload`` line with the elapsed wall-clock time, total bytes, and
    the derived upload bandwidth (MB/s) for the version. This is the S3-client
    transport time on rank 0, not a per-rank or global figure.
    """
    version_name = os.path.basename(os.path.normpath(version_dir))
    uploaded = 0
    uploaded_bytes = 0
    start = time.perf_counter()
    for root, _, files in os.walk(version_dir):
        for name in files:
            if name.endswith(".tmp"):
                continue
            local_path = os.path.join(root, name)
            relative_path = os.path.relpath(local_path, version_dir).replace(os.sep, "/")
            key = _key_prefix(prefix, version_name, relative_path)
            uploaded_bytes += os.path.getsize(local_path)
            s3.upload_file(local_path, bucket, key)
            uploaded += 1
    elapsed = time.perf_counter() - start
    logger.info(
        "mx_upload: %s files=%d bytes=%d elapsed_s=%.3f bandwidth_mb_s=%.1f "
        "to s3://%s/%s",
        version_name,
        uploaded,
        uploaded_bytes,
        elapsed,
        _bandwidth_mb_s(uploaded_bytes, elapsed),
        bucket,
        _key_prefix(prefix, version_name),
    )
    return uploaded


def _list_objects(s3, bucket: str, object_prefix: str):
    continuation_token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": object_prefix}
        if continuation_token is not None:
            kwargs["ContinuationToken"] = continuation_token
        response = s3.list_objects_v2(**kwargs)
        yield from response.get("Contents", [])
        if not response.get("IsTruncated", False):
            return
        continuation_token = response["NextContinuationToken"]


def download_version_dir(
    s3,
    bucket: str,
    prefix: str,
    version: int,
    dest_parent: str,
) -> str:
    """Download a version into a private temporary directory without installing it."""
    version_name = f"weight_v{version:06d}"
    object_prefix = _key_prefix(prefix, version_name) + "/"
    os.makedirs(dest_parent, exist_ok=True)
    temp_dir = tempfile.mkdtemp(
        prefix=f"{version_name}.tmp-{os.getpid()}-",
        dir=dest_parent,
    )
    try:
        downloaded_files = 0
        downloaded_bytes = 0
        start = time.perf_counter()
        for obj in _list_objects(s3, bucket, object_prefix):
            relative_path = obj["Key"][len(object_prefix) :]
            if not relative_path:
                continue
            local_path = os.path.join(temp_dir, *relative_path.split("/"))
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, obj["Key"], local_path)
            downloaded_files += 1
            downloaded_bytes += int(obj.get("Size", 0))
        elapsed = time.perf_counter() - start
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    logger.info(
        "mx_download: version=%d files=%d bytes=%d elapsed_s=%.3f "
        "bandwidth_mb_s=%.1f from s3://%s/%s",
        version,
        downloaded_files,
        downloaded_bytes,
        elapsed,
        _bandwidth_mb_s(downloaded_bytes, elapsed),
        bucket,
        object_prefix.rstrip("/"),
    )
    return temp_dir


def list_versions(s3, bucket: str, prefix: str) -> list[int]:
    """List sorted delta versions present under a stream prefix."""
    object_prefix = _key_prefix(prefix) + "/"
    pattern = re.compile(rf"^{re.escape(object_prefix)}weight_v(\d{{6,}})/")
    versions = set()
    for obj in _list_objects(s3, bucket, object_prefix):
        match = pattern.match(obj["Key"])
        if match:
            versions.add(int(match.group(1)))
    return sorted(versions)
