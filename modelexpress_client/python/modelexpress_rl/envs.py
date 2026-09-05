# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RL-specific deployment policy.

Model identity, server connectivity, worker endpoints, and heartbeat timing
use the shared :mod:`modelexpress.envs` configuration. Rank-local identity and
endpoints are derived from the initialized engine.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    MX_REFIT_CHECKSUM_FORMAT: str
    MX_REFIT_DELTA_BUCKET_BYTES: int
    MX_REFIT_DELTA_WORKERS: int
    MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES: int
    MX_REFIT_METADATA_PORT: int
    MX_S3_DOWNLOAD_RANGE_BYTES: int
    MX_S3_DOWNLOAD_RANGE_THRESHOLD_BYTES: int
    MX_S3_DOWNLOAD_IO_CHUNK_BYTES: int
    MX_S3_DOWNLOAD_MAX_IN_MEMORY_CHUNKS: int
    MX_S3_DOWNLOAD_WORKERS: int
    MX_S3_MAX_ATTEMPTS: int
    MX_S3_MAX_POOL_CONNECTIONS: int
    MX_S3_MULTIPART_THRESHOLD_BYTES: int
    MX_S3_TCP_KEEPALIVE: bool
    MX_S3_UPLOAD_PART_BYTES: int
    MX_S3_UPLOAD_WORKERS: int
    MX_TRAINER_STAGING_MODE: str
    MX_WEIGHT_PAYLOAD_FORMAT: str
    LOCAL_RANK: int | None
    RANK: str | None


environment_variables: dict[str, Callable[[], Any]] = {
    "LOCAL_RANK": lambda: (
        int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else None
    ),
    "MX_REFIT_METADATA_PORT": lambda: require_positive_int(
        int(os.environ.get("MX_REFIT_METADATA_PORT", "7555")),
        "MX_REFIT_METADATA_PORT",
    ),
    "MX_TRAINER_STAGING_MODE": lambda: (
        os.environ.get("MX_TRAINER_STAGING_MODE", "IN_PLACE").strip().upper()
    ),
    "MX_WEIGHT_PAYLOAD_FORMAT": lambda: (
        os.environ.get("MX_WEIGHT_PAYLOAD_FORMAT", "FULL_TENSOR").strip().upper()
    ),
    "MX_REFIT_CHECKSUM_FORMAT": lambda: (
        os.environ.get("MX_REFIT_CHECKSUM_FORMAT", "adler32").strip().lower()
    ),
    # Framework integrations read this while constructing ``hf_tensor_iter``;
    # ModelExpress preserves the supplied framework bucket boundaries.
    "MX_REFIT_DELTA_BUCKET_BYTES": lambda: require_positive_int(
        int(os.environ.get("MX_REFIT_DELTA_BUCKET_BYTES", 512 * 1024**2)),
        "MX_REFIT_DELTA_BUCKET_BYTES",
    ),
    "MX_REFIT_DELTA_WORKERS": lambda: require_positive_int(
        int(
            os.environ.get(
                "MX_REFIT_DELTA_WORKERS",
                min(32, os.cpu_count() or 8),
            )
        ),
        "MX_REFIT_DELTA_WORKERS",
    ),
    "MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES": lambda: require_positive_int(
        int(
            os.environ.get(
                "MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES",
                str(4 * 1024**3),
            )
        ),
        "MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES",
    ),
    "MX_S3_MULTIPART_THRESHOLD_BYTES": lambda: require_positive_int(
        int(os.environ.get("MX_S3_MULTIPART_THRESHOLD_BYTES", 100 * 1024**2)),
        "MX_S3_MULTIPART_THRESHOLD_BYTES",
    ),
    "MX_S3_UPLOAD_PART_BYTES": lambda: require_positive_int(
        int(os.environ.get("MX_S3_UPLOAD_PART_BYTES", 16 * 1024**2)),
        "MX_S3_UPLOAD_PART_BYTES",
    ),
    "MX_S3_UPLOAD_WORKERS": lambda: require_positive_int(
        int(os.environ.get("MX_S3_UPLOAD_WORKERS", 8)),
        "MX_S3_UPLOAD_WORKERS",
    ),
    "MX_S3_DOWNLOAD_RANGE_THRESHOLD_BYTES": lambda: require_positive_int(
        int(
            os.environ.get(
                "MX_S3_DOWNLOAD_RANGE_THRESHOLD_BYTES",
                100 * 1024**2,
            )
        ),
        "MX_S3_DOWNLOAD_RANGE_THRESHOLD_BYTES",
    ),
    "MX_S3_DOWNLOAD_RANGE_BYTES": lambda: require_positive_int(
        int(os.environ.get("MX_S3_DOWNLOAD_RANGE_BYTES", 8 * 1024**2)),
        "MX_S3_DOWNLOAD_RANGE_BYTES",
    ),
    "MX_S3_DOWNLOAD_IO_CHUNK_BYTES": lambda: require_positive_int(
        int(os.environ.get("MX_S3_DOWNLOAD_IO_CHUNK_BYTES", 1024**2)),
        "MX_S3_DOWNLOAD_IO_CHUNK_BYTES",
    ),
    "MX_S3_DOWNLOAD_MAX_IN_MEMORY_CHUNKS": lambda: require_positive_int(
        int(os.environ.get("MX_S3_DOWNLOAD_MAX_IN_MEMORY_CHUNKS", 16)),
        "MX_S3_DOWNLOAD_MAX_IN_MEMORY_CHUNKS",
    ),
    "MX_S3_DOWNLOAD_WORKERS": lambda: require_positive_int(
        int(os.environ.get("MX_S3_DOWNLOAD_WORKERS", 16)),
        "MX_S3_DOWNLOAD_WORKERS",
    ),
    "MX_S3_MAX_POOL_CONNECTIONS": lambda: require_positive_int(
        int(os.environ.get("MX_S3_MAX_POOL_CONNECTIONS", 32)),
        "MX_S3_MAX_POOL_CONNECTIONS",
    ),
    "MX_S3_MAX_ATTEMPTS": lambda: require_positive_int(
        int(os.environ.get("MX_S3_MAX_ATTEMPTS", 5)),
        "MX_S3_MAX_ATTEMPTS",
    ),
    "MX_S3_TCP_KEEPALIVE": lambda: parse_bool(
        os.environ.get("MX_S3_TCP_KEEPALIVE", "true"),
        "MX_S3_TCP_KEEPALIVE",
    ),
    "RANK": lambda: os.environ.get("RANK"),
}


def require_positive_int(value: int, name: str) -> int:
    """Return ``value`` or raise when it is not positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def require_positive_float(value: float, name: str) -> float:
    """Return ``value`` or raise when it is not finite and positive."""
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def parse_bool(value: str, name: str) -> bool:
    """Parse one boolean environment setting."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(environment_variables)
