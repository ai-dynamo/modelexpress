# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang pre-read hook for atomically staging complete delta versions."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil

from .s3_delta import download_version_dir, list_versions, s3_client

logger = logging.getLogger("modelexpress.rl.delta_visibility")

_INDEX_NAME = "model.safetensors.index.json"


def _object_key(prefix: str, version: int, name: str) -> str:
    return "/".join(
        part.strip("/")
        for part in (prefix, f"weight_v{version:06d}", name)
        if part.strip("/")
    )


def _validate_version_dir(
    s3,
    bucket: str,
    prefix: str,
    version: int,
    version_dir: str,
) -> int:
    index_path = os.path.join(version_dir, _INDEX_NAME)
    if not os.path.isfile(index_path):
        raise RuntimeError(f"delta version {version} missing index.json")
    try:
        with open(index_path) as file:
            weight_map = json.load(file)["weight_map"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"delta version {version} has an invalid index.json") from exc

    shard_names = sorted(set(weight_map.values()))
    for shard_name in shard_names:
        shard_path = os.path.join(version_dir, shard_name)
        if not os.path.isfile(shard_path):
            raise RuntimeError(
                f"delta version {version} missing shard {shard_name}"
            )
        try:
            expected_size = s3.head_object(
                Bucket=bucket,
                Key=_object_key(prefix, version, shard_name),
            )["ContentLength"]
        except Exception as exc:
            raise RuntimeError(
                f"unable to verify delta version {version} shard {shard_name}"
            ) from exc
        actual_size = os.path.getsize(shard_path)
        if actual_size != expected_size:
            raise RuntimeError(
                f"delta version {version} shard {shard_name} has size "
                f"{actual_size}, expected {expected_size}"
            )
    return len(shard_names)


def ensure_visible(source_dir: str, target_version: int) -> None:
    """Ensure every S3 delta from v1 through target is atomically visible."""
    os.makedirs(source_dir, exist_ok=True)
    lock_path = os.path.join(source_dir, ".mx_visibility.lock")
    with open(lock_path, "a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        s3 = s3_client()
        bucket = os.environ["MX_S3_BUCKET"]
        prefix = os.environ["MX_DELTA_STREAM_PREFIX"]
        available = set(list_versions(s3, bucket, prefix))
        missing = [
            version
            for version in range(1, target_version + 1)
            if version not in available
        ]
        if missing:
            raise RuntimeError(f"delta stream has a gap at version {missing[0]}")

        for version in range(1, target_version + 1):
            version_name = f"weight_v{version:06d}"
            final_dir = os.path.join(source_dir, version_name)
            installed = False
            try:
                shard_count = _validate_version_dir(
                    s3, bucket, prefix, version, final_dir
                )
                logger.info(
                    "mx_ensure_visible: version=%d cache=hit shards=%d "
                    "(already complete, no download)",
                    version,
                    shard_count,
                )
            except RuntimeError:
                logger.info(
                    "mx_ensure_visible: version=%d cache=miss downloading from S3",
                    version,
                )
                temp_dir = download_version_dir(
                    s3,
                    bucket,
                    prefix,
                    version,
                    source_dir,
                )
                try:
                    shard_count = _validate_version_dir(
                        s3, bucket, prefix, version, temp_dir
                    )
                    if os.path.isdir(final_dir):
                        shutil.rmtree(final_dir)
                    os.replace(temp_dir, final_dir)
                    installed = True
                finally:
                    if os.path.isdir(temp_dir):
                        shutil.rmtree(temp_dir, ignore_errors=True)

            if installed:
                logger.info(
                    "mx_ensure_visible: version=%d complete shards=%d (installed from S3)",
                    version,
                    shard_count,
                )
