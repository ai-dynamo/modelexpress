# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized registry of environment variables the ModelExpress client reads.

Modeled on vLLM's ``envs.py``. ``environment_variables`` maps each variable
name to a zero-argument callable that reads ``os.environ`` and applies the
variable's parsing and default. Access is by attribute, so values are computed
live on every read (tests can ``monkeypatch`` and callers see fresh values
without re-importing)::

    from modelexpress import envs
    if envs.MX_NIXL_BACKEND == "LIBFABRIC":
        ...

This is a leaf module: it imports only the standard library, so any package
module can import it without creating a cycle.

Reads are centralized here; **writes stay at their call sites** (vLLM's
registry is read-only). ``UCX_TLS`` and ``UCX_NET_DEVICES`` are registered for
reading, but the code that sets them does so inline.

Not covered here (intentional exceptions):
- ``MX_SKIP_EXT`` and ``CXX`` are read by ``setup.py`` before the package is
  importable, so they cannot route through this module.
- The deprecated ``MX_VMM_ARENA_BYTES`` / ``MX_VMM_ARENA_CHUNK_BYTES`` are
  presence-only deprecation warnings; check them with :func:`is_set`.
"""

from __future__ import annotations

import logging
import math
import os
from typing import TYPE_CHECKING, Any, Callable, Optional

logger = logging.getLogger("modelexpress.envs")

# Static type hints for editors/type-checkers. The actual values are produced
# by ``__getattr__`` below; these annotations never execute at runtime.
if TYPE_CHECKING:
    # ModelExpress server address / logging
    MODEL_EXPRESS_URL: Optional[str]
    MX_SERVER_ADDRESS: Optional[str]
    MODEL_EXPRESS_CACHE_DIRECTORY: Optional[str]
    MODEL_EXPRESS_NO_SHARED_STORAGE: bool
    MODEL_EXPRESS_TRANSFER_CHUNK_SIZE: Optional[str]
    MODEL_EXPRESS_LOG_LEVEL: str
    MODEL_NAME: Optional[str]
    # Auth (client)
    MX_AUTH_TOKEN_PATH: Optional[str]
    MX_AUTH_TOKEN_TTL_SECONDS: Optional[str]
    # Runtime compatibility
    MX_DISABLE_PATCHES: bool
    # Metadata / worker
    MX_METADATA_BACKEND: str
    MX_METADATA_PORT: int
    MX_WORKER_GRPC_PORT: int
    MX_WORKER_HOST: str
    MX_HEARTBEAT_INTERVAL_SECS: int
    MX_PUBLISH_TIMEOUT_SECS: int
    MX_MODEL_REVISION: str
    MX_MODEL_URI: Optional[str]
    MX_P2P_METADATA: str
    MX_RESHARD_FUSED_WIRE: bool
    MX_RESHARD_BATCH_INSTALL: bool
    MX_RESHARD_CACHE_DESCRIPTORS: bool
    MX_RESHARD_REQUIRE_FULL_COVERAGE: bool
    MX_RESHARD_COVERAGE_FLOOR: float
    MX_RESHARD_HANDSHAKE_TIMEOUT_S: float
    MX_RESHARD_HANDSHAKE_ATTEMPT_S: float
    MX_RESHARD_HANDSHAKE_BACKOFF_S: float
    MX_REFIT_STAGE_RECORD: bool
    MX_RESHARD_MAX_GBPS: float
    MX_RESHARD_PUBLISH_DIGEST: bool
    # Kubernetes service backend
    MX_K8S_SERVICE_PATTERN: str
    MX_K8S_SOURCE_RETRIES: str
    MX_K8S_SOURCE_BACKOFF_SECONDS: str
    # NIXL / transport
    MX_NIXL_BACKEND: str
    MX_POOL_REG: bool
    NIXL_UCX_TLS: Optional[str]
    UCX_TLS: Optional[str]
    UCX_NET_DEVICES: Optional[str]
    MX_RDMA_NIC_PIN: str
    MX_RDMA_NIC_PIN_MIN_RATE_GBPS: Optional[str]
    # GPUDirect Storage
    MX_GDS_MAX_CHUNK_KB: Optional[str]
    MX_GDS_THREADS: int
    MX_GDS_TIMEOUT: float
    # Model streamer
    MX_MS_DISTRIBUTED: bool
    # InstantTensor loader
    MX_INSTANT_TENSOR: bool
    # TRT-LLM live transfer
    MX_SOURCE_QUERY_TIMEOUT: int
    MX_TRANSFER_TIMEOUT: int
    MX_TRANSFER_LOG_DIR: str
    # VMM arena
    MX_VMM_ARENA: bool
    MX_ARENA_SINGLE_MR: bool
    # Framework artifact (JIT cache) transfer
    MX_ARTIFACT_TRANSFER: bool
    MX_ARTIFACT_BUNDLE_ROOT: Optional[str]
    MX_ARTIFACT_COMPILE_CONFIG_DIGEST: str
    MX_ARTIFACT_READY_URL: str
    MX_ARTIFACT_READY_TIMEOUT_SECS: int
    MX_ARTIFACT_TRANSFER_CHUNK_SIZE: Optional[str]
    # Trainer weight sync
    MX_REDIS_URL: str
    # P2P source selection
    MX_P2P_SOURCE_SELECTOR: Optional[str]
    # Opt-in metrics collector
    MX_METRICS_ENABLED: bool
    MX_METRICS_PORT: Optional[str]
    MX_METRICS_PUSHGATEWAY: Optional[str]
    MX_METRICS_SCHEME: str
    MX_METRICS_BIND_RETRY_SECS: float
    MX_METRICS_SOURCE_ID_LABEL: bool
    PROMETHEUS_MULTIPROC_DIR: Optional[str]
    # Third-party JIT/compile cache locations read for artifact transfer
    TRITON_CACHE_DIR: Optional[str]
    TVM_FFI_CACHE_DIR: Optional[str]
    DG_JIT_CACHE_DIR: Optional[str]
    DEEP_GEMM_CACHE_DIR: Optional[str]
    SGLANG_DG_CACHE_DIR: Optional[str]
    SGLANG_CACHE_DIR: Optional[str]
    TILELANG_CACHE_DIR: Optional[str]
    CUTE_DSL_CACHE_DIR: Optional[str]
    FLASHINFER_WORKSPACE_BASE: Optional[str]
    TORCHINDUCTOR_CACHE_DIR: Optional[str]
    VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR: Optional[str]
    VLLM_CACHE_ROOT: Optional[str]
    # Other third-party / system
    VLLM_ATTENTION_BACKEND: str
    HOSTNAME: str
    # Injected by LeaderWorkerSet into every pod of a group, as
    # <leader-pod>.<headless-service>.<namespace>. Fallback head address when
    # the engine does not expose its distributed-init address.
    LWS_LEADER_ADDRESS: str
    POD_NAMESPACE: str
    POD_NAME: str
    POD_UID: str

_TRUTHY = {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Parse an int env var, falling back to ``default`` (and warning) on error."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    """Parse a float env var, falling back to ``default`` (and warning) on error."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Parse a bool env var, falling back to ``default`` (and warning) on error."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid %s=%r; using default %s", name, raw, default)
    return default


def _env_positive_float(name: str, default: float) -> float:
    """As :func:`_env_float`, for a variable that has to be a finite positive bound.

    Timeouts, per-attempt ceilings and backoff pauses are all bounds, and anything
    that is not a finite positive number is not a different bound but an absent
    one: it turns the loop it governs into an unbounded one, which is the failure
    the variable exists to prevent. Zero and negatives are the obvious cases;
    ``inf`` and ``nan`` are the ones worth naming, since both parse happily and
    both make a ``now >= deadline`` test that never fires.

    Falling back to the documented default keeps the bound in place, and the
    warning says which value was ignored.
    """
    value = _env_float(name, default)
    if not math.isfinite(value) or value <= 0.0:
        logger.warning(
            "Invalid %s=%r; a bound must be a finite positive number, using "
            "default %s",
            name,
            os.environ.get(name),
            default,
        )
        return default
    return value


# One entry per variable. The lambda owns the default and parsing; callers that
# need a site-specific default receive the raw value (``None`` when unset) and
# apply their own fallback.
environment_variables: dict[str, Callable[[], Any]] = {
    # ── ModelExpress server address / logging ──────────────────────────────
    # Site-varying defaults: return raw (None when unset), callers add defaults.
    "MODEL_EXPRESS_URL": lambda: os.environ.get("MODEL_EXPRESS_URL"),
    "MX_SERVER_ADDRESS": lambda: os.environ.get("MX_SERVER_ADDRESS"),
    "MODEL_EXPRESS_CACHE_DIRECTORY": lambda: os.environ.get("MODEL_EXPRESS_CACHE_DIRECTORY"),
    "MODEL_EXPRESS_NO_SHARED_STORAGE": lambda: _env_bool("MODEL_EXPRESS_NO_SHARED_STORAGE", False),
    "MODEL_EXPRESS_TRANSFER_CHUNK_SIZE": lambda: os.environ.get(
        "MODEL_EXPRESS_TRANSFER_CHUNK_SIZE"
    ),
    "MODEL_EXPRESS_LOG_LEVEL": lambda: os.environ.get("MODEL_EXPRESS_LOG_LEVEL", "").upper(),
    "MODEL_NAME": lambda: os.environ.get("MODEL_NAME"),
    # ── Auth (client) ──────────────────────────────────────────────────────
    "MX_AUTH_TOKEN_PATH": lambda: os.environ.get("MX_AUTH_TOKEN_PATH"),
    "MX_AUTH_TOKEN_TTL_SECONDS": lambda: os.environ.get("MX_AUTH_TOKEN_TTL_SECONDS"),
    # ── Runtime compatibility ──────────────────────────────────────────────
    "MX_DISABLE_PATCHES": lambda: os.environ.get("MX_DISABLE_PATCHES", "").strip().lower()
    in _TRUTHY,
    # ── Metadata / worker ──────────────────────────────────────────────────
    "MX_METADATA_BACKEND": lambda: os.environ.get("MX_METADATA_BACKEND", "").lower().strip(),
    "MX_METADATA_PORT": lambda: _env_int("MX_METADATA_PORT", 5555),
    "MX_WORKER_GRPC_PORT": lambda: _env_int("MX_WORKER_GRPC_PORT", 6555),
    "MX_WORKER_HOST": lambda: os.environ.get("MX_WORKER_HOST", ""),
    "MX_HEARTBEAT_INTERVAL_SECS": lambda: _env_int("MX_HEARTBEAT_INTERVAL_SECS", 30),
    "MX_PUBLISH_TIMEOUT_SECS": lambda: _env_int("MX_PUBLISH_TIMEOUT_SECS", 30 * 60),
    "MX_MODEL_REVISION": lambda: os.environ.get("MX_MODEL_REVISION", ""),
    "MX_MODEL_URI": lambda: os.environ.get("MX_MODEL_URI"),
    "MX_P2P_METADATA": lambda: os.environ.get("MX_P2P_METADATA", "1"),
    "MX_RESHARD_FUSED_WIRE": lambda: _env_bool("MX_RESHARD_FUSED_WIRE", True),
    # Issue the per-view re-slice copies of full-pulled sources as one batched
    # _foreach_copy_ instead of a copy_() per view. On by default: it is the same
    # set of copies, and one launch per view means thousands of launches whose
    # Python and launch overhead can rival the RDMA itself. Set to 0 to fall back
    # to the per-view loop. See modelexpress.refit.reshard.receiver.
    "MX_RESHARD_BATCH_INSTALL": lambda: _env_bool("MX_RESHARD_BATCH_INSTALL", True),
    # Build the RDMA read descriptor lists once per plan instead of once per step.
    # The descriptors are a pure function of the cached plan and the registered
    # buffer addresses, neither of which changes between steps, so rebuilding them
    # every refit re-derives an identical list of hundreds of thousands of objects
    # in Python. On by default; set to 0 to rebuild per step for an A/B. The cache
    # is dropped whenever the plan is rebuilt. See
    # modelexpress.refit.reshard.receiver.
    "MX_RESHARD_CACHE_DESCRIPTORS": lambda: _env_bool(
        "MX_RESHARD_CACHE_DESCRIPTORS", True
    ),
    # Refit coverage gate. The floor is a fraction of the engine's parameter
    # bytes; ReshardReceiver validates its range at the point of use. What a
    # complete refit scores is engine- and model-specific, so the default is set
    # loose enough to pass any complete refit and still catch a gross hole; see
    # modelexpress.refit.reshard.receiver._coverage_floor.
    "MX_RESHARD_REQUIRE_FULL_COVERAGE": lambda: os.environ.get(
        "MX_RESHARD_REQUIRE_FULL_COVERAGE", ""
    )
    .strip()
    .lower()
    in _TRUTHY,
    "MX_RESHARD_COVERAGE_FLOOR": lambda: _env_float("MX_RESHARD_COVERAGE_FLOOR", 0.995),
    # Bounds for the reshard peer handshake; see
    # modelexpress.refit.reshard.receiver.handshake_with_peers.
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
    # Per-rank fabric ceiling in Gbps used to reject impossible wire rates. Zero
    # disables the check, and is the default because only the operator knows the
    # real per-rank limit for their fabric.
    "MX_RESHARD_MAX_GBPS": lambda: _env_float("MX_RESHARD_MAX_GBPS", 0.0),
    # Have publishers digest each shard they advertise, so a receiver-side check has
    # something to compare against. Off by default: it costs a reduction over every
    # published tensor, which is a large relative cost against a ~1.5 s wire, so it
    # belongs in a qualification run rather than a throughput measurement. See
    # modelexpress.refit.reshard.verify.
    "MX_RESHARD_PUBLISH_DIGEST": lambda: _env_bool("MX_RESHARD_PUBLISH_DIGEST", False),
    # ── Kubernetes service backend ─────────────────────────────────────────
    "MX_K8S_SERVICE_PATTERN": lambda: os.environ.get("MX_K8S_SERVICE_PATTERN", "mx-sources"),
    "MX_K8S_SOURCE_RETRIES": lambda: os.environ.get("MX_K8S_SOURCE_RETRIES", ""),
    "MX_K8S_SOURCE_BACKOFF_SECONDS": lambda: os.environ.get("MX_K8S_SOURCE_BACKOFF_SECONDS", ""),
    # ── NIXL / transport ───────────────────────────────────────────────────
    "MX_NIXL_BACKEND": lambda: os.environ.get("MX_NIXL_BACKEND", "UCX").strip().upper(),
    "MX_POOL_REG": lambda: os.environ.get("MX_POOL_REG", "0") == "1",
    "NIXL_UCX_TLS": lambda: os.environ.get("NIXL_UCX_TLS"),
    "UCX_TLS": lambda: os.environ.get("UCX_TLS"),
    "UCX_NET_DEVICES": lambda: os.environ.get("UCX_NET_DEVICES"),
    "MX_RDMA_NIC_PIN": lambda: os.environ.get("MX_RDMA_NIC_PIN", "").strip(),
    "MX_RDMA_NIC_PIN_MIN_RATE_GBPS": lambda: os.environ.get("MX_RDMA_NIC_PIN_MIN_RATE_GBPS"),
    # ── GPUDirect Storage ──────────────────────────────────────────────────
    "MX_GDS_MAX_CHUNK_KB": lambda: os.environ.get("MX_GDS_MAX_CHUNK_KB"),
    "MX_GDS_THREADS": lambda: _env_int("MX_GDS_THREADS", 8),
    "MX_GDS_TIMEOUT": lambda: _env_float("MX_GDS_TIMEOUT", 120.0),
    # ── Model streamer ─────────────────────────────────────────────────────
    "MX_MS_DISTRIBUTED": lambda: os.environ.get("MX_MS_DISTRIBUTED", "1").lower() in ("1", "true"),
    # ── InstantTensor loader ───────────────────────────────────────────────
    # Enabled by default; the strategy is still gated on the instanttensor
    # package and a CUDA device, so opting out is only needed to force a
    # different local-load path. Set MX_INSTANT_TENSOR=0 to disable.
    "MX_INSTANT_TENSOR": lambda: os.environ.get("MX_INSTANT_TENSOR", "1").lower() in ("1", "true"),
    # ── TRT-LLM live transfer ──────────────────────────────────────────────
    "MX_SOURCE_QUERY_TIMEOUT": lambda: _env_int("MX_SOURCE_QUERY_TIMEOUT", 3600),
    "MX_TRANSFER_TIMEOUT": lambda: _env_int("MX_TRANSFER_TIMEOUT", 900),
    "MX_TRANSFER_LOG_DIR": lambda: os.environ.get("MX_TRANSFER_LOG_DIR", "/tmp/mx_logs"),
    # ── VMM arena ──────────────────────────────────────────────────────────
    "MX_VMM_ARENA": lambda: os.environ.get("MX_VMM_ARENA") == "1",
    "MX_ARENA_SINGLE_MR": lambda: os.environ.get("MX_ARENA_SINGLE_MR") == "1",
    # ── Framework artifact (JIT cache) transfer ────────────────────────────
    "MX_ARTIFACT_TRANSFER": lambda: os.environ.get("MX_ARTIFACT_TRANSFER", "").strip().lower()
    in _TRUTHY,
    "MX_ARTIFACT_BUNDLE_ROOT": lambda: os.environ.get("MX_ARTIFACT_BUNDLE_ROOT"),
    "MX_ARTIFACT_COMPILE_CONFIG_DIGEST": lambda: os.environ.get(
        "MX_ARTIFACT_COMPILE_CONFIG_DIGEST", ""
    ),
    "MX_ARTIFACT_READY_URL": lambda: os.environ.get("MX_ARTIFACT_READY_URL", ""),
    "MX_ARTIFACT_READY_TIMEOUT_SECS": lambda: _env_int("MX_ARTIFACT_READY_TIMEOUT_SECS", 1800),
    # Raw string: artifact_manifest.artifact_transfer_chunk_size() owns the
    # int parse plus its non-positive/max-bound validation and default param.
    "MX_ARTIFACT_TRANSFER_CHUNK_SIZE": lambda: os.environ.get("MX_ARTIFACT_TRANSFER_CHUNK_SIZE"),
    # ── Trainer pull (live weight sync from a running trainer) ─────────────
    "MX_REDIS_URL": lambda: os.environ.get("MX_REDIS_URL", "redis://localhost:6379"),
    # ── P2P source selection ───────────────────────────────────────────────
    # Raw (None when unset); source_selection applies its DEFAULT_SELECTOR fallback.
    "MX_P2P_SOURCE_SELECTOR": lambda: os.environ.get("MX_P2P_SOURCE_SELECTOR"),
    # ── Opt-in metrics collector ───────────────────────────────────────────
    "MX_METRICS_ENABLED": lambda: os.environ.get("MX_METRICS_ENABLED", "0").strip().lower()
    in _TRUTHY,
    "MX_METRICS_PORT": lambda: os.environ.get("MX_METRICS_PORT"),
    "MX_METRICS_PUSHGATEWAY": lambda: os.environ.get("MX_METRICS_PUSHGATEWAY"),
    "MX_METRICS_SCHEME": lambda: os.environ.get("MX_METRICS_SCHEME", ""),
    # Interval at which a rank that lost the /metrics bind re-attempts it, so
    # endpoint ownership migrates when the winning rank exits.
    "MX_METRICS_BIND_RETRY_SECS": lambda: _env_float("MX_METRICS_BIND_RETRY_SECS", 15.0),
    # Restore the per-peer source_worker_id label on
    # mx_p2p_source_selections_total. Benchmark runs only: the id is a
    # per-process uuid, so its label domain grows with process count over time
    # rather than with cluster size.
    "MX_METRICS_SOURCE_ID_LABEL": lambda: os.environ.get("MX_METRICS_SOURCE_ID_LABEL", "0")
    .strip()
    .lower()
    in _TRUTHY,
    # prometheus_client multiprocess directory. Read-only here: it MUST be set
    # in the pod manifest, never assigned in Python. get_value_class() latches at
    # prometheus_client import time, so an in-process assignment lands after the
    # engine has already imported it and produces zero .db files with no error.
    "PROMETHEUS_MULTIPROC_DIR": lambda: os.environ.get("PROMETHEUS_MULTIPROC_DIR"),
    # ── Third-party JIT/compile cache locations (raw; caller builds path) ──
    "TRITON_CACHE_DIR": lambda: os.environ.get("TRITON_CACHE_DIR"),
    "TVM_FFI_CACHE_DIR": lambda: os.environ.get("TVM_FFI_CACHE_DIR"),
    "DG_JIT_CACHE_DIR": lambda: os.environ.get("DG_JIT_CACHE_DIR"),
    "DEEP_GEMM_CACHE_DIR": lambda: os.environ.get("DEEP_GEMM_CACHE_DIR"),
    "SGLANG_DG_CACHE_DIR": lambda: os.environ.get("SGLANG_DG_CACHE_DIR"),
    "SGLANG_CACHE_DIR": lambda: os.environ.get("SGLANG_CACHE_DIR"),
    "TILELANG_CACHE_DIR": lambda: os.environ.get("TILELANG_CACHE_DIR"),
    "CUTE_DSL_CACHE_DIR": lambda: os.environ.get("CUTE_DSL_CACHE_DIR"),
    "FLASHINFER_WORKSPACE_BASE": lambda: os.environ.get("FLASHINFER_WORKSPACE_BASE"),
    "TORCHINDUCTOR_CACHE_DIR": lambda: os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": lambda: os.environ.get(
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"
    ),
    "VLLM_CACHE_ROOT": lambda: os.environ.get("VLLM_CACHE_ROOT"),
    # ── Other third-party / system ─────────────────────────────────────────
    "VLLM_ATTENTION_BACKEND": lambda: os.environ.get("VLLM_ATTENTION_BACKEND", "auto"),
    "HOSTNAME": lambda: os.environ.get("HOSTNAME", ""),
    "LWS_LEADER_ADDRESS": lambda: os.environ.get("LWS_LEADER_ADDRESS", ""),
    "POD_NAMESPACE": lambda: os.environ.get("POD_NAMESPACE", ""),
    "POD_NAME": lambda: os.environ.get("POD_NAME", ""),
    "POD_UID": lambda: os.environ.get("POD_UID", "")
}


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(environment_variables)


def is_set(name: str) -> bool:
    """Return ``True`` if the environment variable is present, regardless of value."""
    return name in os.environ
