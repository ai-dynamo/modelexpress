# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RL-specific deployment policy.

Model identity, server connectivity, worker endpoints, and heartbeat timing
use the shared :mod:`modelexpress.envs` configuration. Rank-local identity and
endpoints are derived from the initialized engine.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from modelexpress.envs import _TRUTHY, _env_bool, _env_float, _env_positive_float

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    MX_RESHARD_FUSED_WIRE: bool
    MX_RESHARD_BATCH_INSTALL: bool
    MX_RESHARD_CACHE_DESCRIPTORS: bool
    MX_RESHARD_REQUIRE_FULL_COVERAGE: bool
    MX_RESHARD_COVERAGE_FLOOR: float
    MX_RESHARD_HANDSHAKE_TIMEOUT_S: float
    MX_RESHARD_HANDSHAKE_ATTEMPT_S: float
    MX_RESHARD_HANDSHAKE_BACKOFF_S: float
    MX_REFIT_STAGE_RECORD: bool
    MX_REFIT_PACK_MODULES: bool
    MX_REFIT_PREFETCH_REPLICA_METADATA: bool
    MX_REFIT_DEBUG_VALIDATE_REPLICAS: bool
    MX_RESHARD_MAX_SEGMENTS_PER_COPY: int
    MX_RESHARD_MAX_GBPS: float
    MX_RESHARD_MIN_GBPS: float
    MX_RESHARD_PUBLISH_DIGEST: bool
    MX_REDIS_URL: str
    MX_GENERATOR_SOURCE_ORDER: str | None
    AZURE_STORAGE_ACCOUNT_NAME: str | None
    AZURE_STORAGE_ACCOUNT_URL: str | None
    AZURE_STORAGE_CONNECTION_STRING: str | None
    MX_AZURE_DOWNLOAD_CONCURRENCY: int
    MX_AZURE_DOWNLOAD_RANGE_BYTES: int
    MX_AZURE_DOWNLOAD_RANGE_THRESHOLD_BYTES: int
    MX_AZURE_MAX_POOL_CONNECTIONS: int
    MX_AZURE_MAX_ATTEMPTS: int
    MX_AZURE_CONNECTION_TIMEOUT_SECONDS: int
    MX_AZURE_READ_TIMEOUT_SECONDS: int
    MX_REFIT_CHECKSUM_FORMAT: str
    MX_REFIT_DESIRED_VERSION_UID: str | None
    MX_REFIT_CHECKPOINT_DIR: str | None
    MX_REFIT_DELTA_BUCKET_BYTES: int
    MX_REFIT_DELTA_SURGICAL: bool
    MX_REFIT_DEBUG_VALIDATE_PLAN: bool
    MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT: bool
    MX_REFIT_DELTA_WORKERS: int
    MX_REFIT_FULL_CHECKPOINT_BATCH_BYTES: int
    MX_REFIT_METADATA_PORT: int
    MX_REFIT_FULL_STREAMING: bool
    MX_REFIT_STREAM_WINDOW_LAYERS: int
    MX_REFIT_TIMING: bool
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
    "MX_RESHARD_FUSED_WIRE": lambda: _env_bool("MX_RESHARD_FUSED_WIRE", True),
    # Issue the per-view re-slice copies of full-pulled sources as one batched
    # _foreach_copy_ instead of a copy_() per view. On by default: it is the same
    # set of copies, and one launch per view means thousands of launches whose
    # Python and launch overhead can rival the RDMA itself. Set to 0 to fall back
    # to the per-view loop. See modelexpress.rl.refit.reshard.receiver.
    "MX_RESHARD_BATCH_INSTALL": lambda: _env_bool("MX_RESHARD_BATCH_INSTALL", True),
    # Build the RDMA read descriptor lists once per plan instead of once per step.
    # The descriptors are a pure function of the cached plan and the registered
    # buffer addresses, neither of which changes between steps, so rebuilding them
    # every refit re-derives an identical list of hundreds of thousands of objects
    # in Python. On by default; set to 0 to rebuild per step for an A/B. The cache
    # is dropped whenever the plan is rebuilt. See
    # modelexpress.rl.refit.reshard.receiver.
    "MX_RESHARD_CACHE_DESCRIPTORS": lambda: _env_bool(
        "MX_RESHARD_CACHE_DESCRIPTORS", True
    ),
    # Refit coverage gate. The floor is a fraction of the engine's parameter
    # bytes; ReshardReceiver validates its range at the point of use. What a
    # complete refit scores is engine- and model-specific, so the default is set
    # loose enough to pass any complete refit and still catch a gross hole; see
    # modelexpress.rl.refit.reshard.receiver._coverage_floor.
    "MX_RESHARD_REQUIRE_FULL_COVERAGE": lambda: (
        os.environ.get("MX_RESHARD_REQUIRE_FULL_COVERAGE", "").strip().lower()
        in _TRUTHY
    ),
    "MX_RESHARD_COVERAGE_FLOOR": lambda: _env_float("MX_RESHARD_COVERAGE_FLOOR", 0.995),
    # Bounds for the reshard peer handshake; see
    # modelexpress.rl.refit.reshard.receiver.handshake_with_peers.
    #
    # TIMEOUT_S is the budget across every peer and every retry. A refit timeout is
    # the wrong bound, since it lets one unreachable peer consume the whole refit,
    # but it must stay generous: a peer registering tens of GB with the fabric
    # provider blocks its listen thread for minutes, and waiting is the correct
    # response to that. ATTEMPT_S caps a single dial, so an unreachable peer frees
    # the budget for a different one instead of blocking on it. BACKOFF_S is the
    # pause after a full pass over the pending peers makes no progress.
    "MX_RESHARD_HANDSHAKE_TIMEOUT_S": lambda: _env_positive_float(
        "MX_RESHARD_HANDSHAKE_TIMEOUT_S", 900.0
    ),
    "MX_RESHARD_HANDSHAKE_ATTEMPT_S": lambda: _env_positive_float(
        "MX_RESHARD_HANDSHAKE_ATTEMPT_S", 20.0
    ),
    "MX_RESHARD_HANDSHAKE_BACKOFF_S": lambda: _env_positive_float(
        "MX_RESHARD_HANDSHAKE_BACKOFF_S", 2.0
    ),
    # One JSON stage record per refit. On by default: the timings are already
    # computed, and at INFO they were never captured by a benchmark run.
    "MX_REFIT_STAGE_RECORD": lambda: _env_bool("MX_REFIT_STAGE_RECORD", True),
    "MX_REFIT_PREFETCH_REPLICA_METADATA": lambda: _env_bool(
        "MX_REFIT_PREFETCH_REPLICA_METADATA", False
    ),
    "MX_REFIT_DEBUG_VALIDATE_REPLICAS": lambda: _env_bool(
        "MX_REFIT_DEBUG_VALIDATE_REPLICAS", False
    ),
    "MX_RESHARD_MAX_SEGMENTS_PER_COPY": lambda: int(
        os.environ.get("MX_RESHARD_MAX_SEGMENTS_PER_COPY", "64")
    ),
    # Coalesce consecutive owning-module batches up to the staging budget.
    "MX_REFIT_PACK_MODULES": lambda: _env_bool("MX_REFIT_PACK_MODULES", True),
    # Per-rank fabric ceiling in Gbps used to reject impossible wire rates. Zero
    # disables the check, and is the default because only the operator knows the
    # real per-rank limit for their fabric.
    "MX_RESHARD_MAX_GBPS": lambda: _env_float("MX_RESHARD_MAX_GBPS", 0.0),
    # Per-rank floor in Gbps below which a refit is reported as degraded. Zero
    # disables it, and is the default for the same reason as the ceiling.
    #
    # The ceiling catches transfers that did not happen. Nothing caught a transfer
    # that happened 20x too slowly: four concurrent receivers collapsed from
    # ~26 GB/s to ~1.5 GB/s with no error, no warning, byte counts exact,
    # descriptor counts exact, coverage 100% and fallback 0. Every signal we had
    # said the refit was healthy. A rate is only interpretable against a bound,
    # and we only ever had the upper one.
    "MX_RESHARD_MIN_GBPS": lambda: _env_float("MX_RESHARD_MIN_GBPS", 0.0),
    # Have publishers digest each shard they advertise, so a receiver-side check has
    # something to compare against. Off by default: it costs a reduction over every
    # published tensor, which is a large relative cost against a ~1.5 s wire, so it
    # belongs in a qualification run rather than a throughput measurement. See
    # modelexpress.rl.refit.reshard.verify.
    "MX_RESHARD_PUBLISH_DIGEST": lambda: _env_bool("MX_RESHARD_PUBLISH_DIGEST", False),
    # ── Trainer pull (live weight sync from a running trainer) ─────────────
    "MX_REDIS_URL": lambda: os.environ.get("MX_REDIS_URL", "redis://localhost:6379"),
    # ── Generator weight sync ──────────────────────────────────────────────
    "MX_GENERATOR_SOURCE_ORDER": lambda: os.environ.get("MX_GENERATOR_SOURCE_ORDER"),
    "AZURE_STORAGE_ACCOUNT_NAME": lambda: (
        os.environ.get("AZURE_STORAGE_ACCOUNT_NAME", "").strip() or None
    ),
    "AZURE_STORAGE_ACCOUNT_URL": lambda: (
        os.environ.get("AZURE_STORAGE_ACCOUNT_URL", "").strip() or None
    ),
    "AZURE_STORAGE_CONNECTION_STRING": lambda: (
        os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip() or None
    ),
    "MX_AZURE_DOWNLOAD_CONCURRENCY": lambda: positive_int_env(
        "MX_AZURE_DOWNLOAD_CONCURRENCY", 1
    ),
    "MX_AZURE_DOWNLOAD_RANGE_BYTES": lambda: positive_int_env(
        "MX_AZURE_DOWNLOAD_RANGE_BYTES", 4 * 1024**2
    ),
    "MX_AZURE_DOWNLOAD_RANGE_THRESHOLD_BYTES": lambda: positive_int_env(
        "MX_AZURE_DOWNLOAD_RANGE_THRESHOLD_BYTES", 32 * 1024**2
    ),
    "MX_AZURE_MAX_POOL_CONNECTIONS": lambda: positive_int_env(
        "MX_AZURE_MAX_POOL_CONNECTIONS", 32
    ),
    "MX_AZURE_MAX_ATTEMPTS": lambda: positive_int_env("MX_AZURE_MAX_ATTEMPTS", 4),
    "MX_AZURE_CONNECTION_TIMEOUT_SECONDS": lambda: positive_int_env(
        "MX_AZURE_CONNECTION_TIMEOUT_SECONDS", 20
    ),
    "MX_AZURE_READ_TIMEOUT_SECONDS": lambda: positive_int_env(
        "MX_AZURE_READ_TIMEOUT_SECONDS", 60
    ),
    "LOCAL_RANK": lambda: (
        int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else None
    ),
    "MX_REFIT_DEBUG_VALIDATE_PLAN": lambda: parse_bool(
        os.environ.get("MX_REFIT_DEBUG_VALIDATE_PLAN", "false"),
        "MX_REFIT_DEBUG_VALIDATE_PLAN",
    ),
    "MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT": lambda: parse_bool(
        os.environ.get("MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT", "false"),
        "MX_REFIT_DEBUG_VALIDATE_GENERATOR_LAYOUT",
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
    "MX_REFIT_DESIRED_VERSION_UID": lambda: (
        os.environ.get("MX_REFIT_DESIRED_VERSION_UID", "").strip() or None
    ),
    "MX_REFIT_CHECKPOINT_DIR": lambda: (
        os.environ.get("MX_REFIT_CHECKPOINT_DIR", "").strip() or None
    ),
    # Framework integrations read this while constructing ``tensor_iter``;
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
    # Install a delta by reloading only the modules it changes.
    "MX_REFIT_DELTA_SURGICAL": lambda: parse_bool(
        os.environ.get("MX_REFIT_DELTA_SURGICAL", "false"),
        "MX_REFIT_DELTA_SURGICAL",
    ),
    "MX_REFIT_FULL_STREAMING": lambda: parse_bool(
        os.environ.get("MX_REFIT_FULL_STREAMING", "false"),
        "MX_REFIT_FULL_STREAMING",
    ),
    # Decoder layers per streamed full-refit request; 0 streams all at once.
    "MX_REFIT_STREAM_WINDOW_LAYERS": lambda: require_non_negative_int(
        int(os.environ.get("MX_REFIT_STREAM_WINDOW_LAYERS", 2)),
        "MX_REFIT_STREAM_WINDOW_LAYERS",
    ),
    # One normalized timing record per generator refit. On by default: the
    # durations are already being measured on the staging path, so recording
    # them costs a few perf_counter calls and one log line, against a refit
    # measured in seconds. Off is for callers that drive their own recorder and
    # do not want a second one nested inside it.
    "MX_REFIT_TIMING": lambda: parse_bool(
        os.environ.get("MX_REFIT_TIMING", "true"),
        "MX_REFIT_TIMING",
    ),
    "RANK": lambda: os.environ.get("RANK"),
}


def parse_int(value: str, name: str) -> int:
    """Parse one integer environment setting."""
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer") from None


def require_positive_int(value: int, name: str) -> int:
    """Return ``value`` or raise when it is not positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def positive_int_env(name: str, default: int) -> int:
    """Read a positive integer while retaining its setting name in errors."""
    return require_positive_int(parse_int(os.environ.get(name, str(default)), name), name)


def require_non_negative_int(value: int, name: str) -> int:
    """Return ``value`` or raise when it is negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
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
