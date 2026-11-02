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
registry is read-only). ``UCX_TLS``, ``UCX_NET_DEVICES``, and
``UCX_MEM_EVENTS`` are registered for reading, but the code that sets them
does so inline.

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
    MX_MODEL_NAME_OVERRIDE: Optional[str]
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
    MX_PLUGIN_DIR: str
    MX_LOAD_PLUGIN: Optional[str]
    MX_LOAD_PLUGIN_BEFORE: str
    MX_LOAD_STRATEGY_CHAIN: str
    MX_P2P_METADATA: str
    MX_DISABLE_DRAFT_P2P: bool
    MX_DRAFT_MODEL_URI: Optional[str]
    MX_DRAFT_MODEL_REVISION: str
    MX_DRAFT_PORT_OFFSET: int
    # Kubernetes service backend
    MX_K8S_SERVICE_PATTERN: str
    MX_K8S_SOURCE_RETRIES: str
    MX_K8S_SOURCE_BACKOFF_SECONDS: str
    # NIXL / transport
    MX_NIXL_BACKEND: str
    MX_NIXL_METADATA_TIMEOUT: float
    MX_POOL_REG: bool
    NIXL_UCX_TLS: Optional[str]
    UCX_TLS: Optional[str]
    UCX_NET_DEVICES: Optional[str]
    UCX_MEM_EVENTS: Optional[str]
    MX_UCX_DISABLE_MEM_EVENTS: bool
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
    MX_ARTIFACT_BACKEND: str
    MX_ARTIFACT_MOONCAKE_NAMESPACE: str
    MX_ARTIFACT_MOONCAKE_POOL_BYTES: int
    MX_ARTIFACT_MOONCAKE_DELETE_RETRIES: int
    MX_ARTIFACT_MOONCAKE_DELETE_RETRY_DELAY_SECS: float
    MX_ARTIFACT_READY_URL: str
    MX_ARTIFACT_READY_TIMEOUT_SECS: int
    MX_ARTIFACT_TRANSFER_CHUNK_SIZE: Optional[str]
    # P2P source selection
    MX_P2P_SOURCE_SELECTOR: Optional[str]
    # Weight of the NIC-utilization penalty in the load_aware selector.
    MX_P2P_LOAD_WEIGHT: float
    # Optional runtime /metrics URL (vLLM/SGLang) for the source_load signal.
    MX_P2P_RUNTIME_METRICS_URL: Optional[str]
    # Topology-aware selection: ordered levels (broad->narrow), this node's
    # {level: value} JSON map, and the optional within-tier load-blend weight.
    MX_P2P_TOPOLOGY_LEVELS: Optional[str]
    MX_P2P_TOPOLOGY: Optional[str]
    MX_P2P_TOPOLOGY_LOAD_WEIGHT: float
    # Transfer-domain policy: the topology domain sources must share with this
    # node, and whether that is required (filter) or preferred (ordering).
    MX_P2P_TOPOLOGY_DOMAIN: Optional[str]
    MX_P2P_TOPOLOGY_ENFORCEMENT: Optional[str]
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
    """Parse a float env var, falling back to ``default`` (and warning) on error.

    Rejects non-finite values (``inf``/``nan``): they are always a misconfig and
    can poison arithmetic downstream (e.g. ``inf * 0.0 -> nan`` in a selector
    score, which would break deterministic ordering).
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default
    if not math.isfinite(value):
        logger.warning("Non-finite %s=%r; using default %s", name, raw, default)
        return default
    return value


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
            "Invalid %s=%r; a bound must be a finite positive number, using default %s",
            name,
            os.environ.get(name),
            default,
        )
        return default
    return value


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


# One entry per variable. The lambda owns the default and parsing; callers that
# need a site-specific default receive the raw value (``None`` when unset) and
# apply their own fallback.
environment_variables: dict[str, Callable[[], Any]] = {
    # ── ModelExpress server address / logging ──────────────────────────────
    # Site-varying defaults: return raw (None when unset), callers add defaults.
    "MODEL_EXPRESS_URL": lambda: os.environ.get("MODEL_EXPRESS_URL"),
    "MX_SERVER_ADDRESS": lambda: os.environ.get("MX_SERVER_ADDRESS"),
    "MODEL_EXPRESS_CACHE_DIRECTORY": lambda: os.environ.get(
        "MODEL_EXPRESS_CACHE_DIRECTORY"
    ),
    "MODEL_EXPRESS_NO_SHARED_STORAGE": lambda: _env_bool(
        "MODEL_EXPRESS_NO_SHARED_STORAGE", False
    ),
    "MODEL_EXPRESS_TRANSFER_CHUNK_SIZE": lambda: os.environ.get(
        "MODEL_EXPRESS_TRANSFER_CHUNK_SIZE"
    ),
    "MODEL_EXPRESS_LOG_LEVEL": lambda: os.environ.get("MODEL_EXPRESS_LOG_LEVEL", "").upper(),
    "MX_MODEL_NAME_OVERRIDE": lambda: os.environ.get("MX_MODEL_NAME_OVERRIDE"),
    # ── Auth (client) ──────────────────────────────────────────────────────
    "MX_AUTH_TOKEN_PATH": lambda: os.environ.get("MX_AUTH_TOKEN_PATH"),
    "MX_AUTH_TOKEN_TTL_SECONDS": lambda: os.environ.get("MX_AUTH_TOKEN_TTL_SECONDS"),
    # ── Runtime compatibility ──────────────────────────────────────────────
    "MX_DISABLE_PATCHES": lambda: (
        os.environ.get("MX_DISABLE_PATCHES", "").strip().lower() in _TRUTHY
    ),
    # ── Metadata / worker ──────────────────────────────────────────────────
    "MX_METADATA_BACKEND": lambda: (
        os.environ.get("MX_METADATA_BACKEND", "").lower().strip()
    ),
    "MX_METADATA_PORT": lambda: _env_int("MX_METADATA_PORT", 5555),
    "MX_WORKER_GRPC_PORT": lambda: _env_int("MX_WORKER_GRPC_PORT", 6555),
    "MX_WORKER_HOST": lambda: os.environ.get("MX_WORKER_HOST", ""),
    "MX_HEARTBEAT_INTERVAL_SECS": lambda: _env_int("MX_HEARTBEAT_INTERVAL_SECS", 30),
    "MX_PUBLISH_TIMEOUT_SECS": lambda: _env_int("MX_PUBLISH_TIMEOUT_SECS", 30 * 60),
    "MX_MODEL_REVISION": lambda: os.environ.get("MX_MODEL_REVISION", ""),
    "MX_MODEL_URI": lambda: os.environ.get("MX_MODEL_URI"),
    "MX_PLUGIN_DIR": lambda: os.environ.get("MX_PLUGIN_DIR", "/opt/modelexpress/plugins"),
    "MX_LOAD_PLUGIN": lambda: os.environ.get("MX_LOAD_PLUGIN"),
    "MX_LOAD_PLUGIN_BEFORE": lambda: os.environ.get("MX_LOAD_PLUGIN_BEFORE", "server-cache"),
    "MX_LOAD_STRATEGY_CHAIN": lambda: (
        os.environ.get("MX_LOAD_STRATEGY_CHAIN", "INFERENCE").strip().upper()
    ),
    "MX_P2P_METADATA": lambda: os.environ.get("MX_P2P_METADATA", "1"),
    "MX_DISABLE_DRAFT_P2P": lambda: _env_bool("MX_DISABLE_DRAFT_P2P", False),
    "MX_DRAFT_MODEL_URI": lambda: os.environ.get("MX_DRAFT_MODEL_URI"),
    "MX_DRAFT_MODEL_REVISION": lambda: os.environ.get("MX_DRAFT_MODEL_REVISION", ""),
    "MX_DRAFT_PORT_OFFSET": lambda: _env_int("MX_DRAFT_PORT_OFFSET", 10000),
    # ── Kubernetes service backend ─────────────────────────────────────────
    "MX_K8S_SERVICE_PATTERN": lambda: os.environ.get(
        "MX_K8S_SERVICE_PATTERN", "mx-sources"
    ),
    "MX_K8S_SOURCE_RETRIES": lambda: os.environ.get("MX_K8S_SOURCE_RETRIES", ""),
    "MX_K8S_SOURCE_BACKOFF_SECONDS": lambda: os.environ.get(
        "MX_K8S_SOURCE_BACKOFF_SECONDS", ""
    ),
    # ── NIXL / transport ───────────────────────────────────────────────────
    "MX_NIXL_BACKEND": lambda: os.environ.get("MX_NIXL_BACKEND", "UCX").strip().upper(),
    "MX_NIXL_METADATA_TIMEOUT": lambda: _env_positive_float(
        "MX_NIXL_METADATA_TIMEOUT", 120.0
    ),
    "MX_POOL_REG": lambda: os.environ.get("MX_POOL_REG", "0") == "1",
    "NIXL_UCX_TLS": lambda: os.environ.get("NIXL_UCX_TLS"),
    "UCX_TLS": lambda: os.environ.get("UCX_TLS"),
    "UCX_NET_DEVICES": lambda: os.environ.get("UCX_NET_DEVICES"),
    "UCX_MEM_EVENTS": lambda: os.environ.get("UCX_MEM_EVENTS"),
    "MX_UCX_DISABLE_MEM_EVENTS": lambda: _env_bool("MX_UCX_DISABLE_MEM_EVENTS", False),
    "MX_RDMA_NIC_PIN": lambda: os.environ.get("MX_RDMA_NIC_PIN", "").strip(),
    "MX_RDMA_NIC_PIN_MIN_RATE_GBPS": lambda: os.environ.get(
        "MX_RDMA_NIC_PIN_MIN_RATE_GBPS"
    ),
    # ── GPUDirect Storage ──────────────────────────────────────────────────
    "MX_GDS_MAX_CHUNK_KB": lambda: os.environ.get("MX_GDS_MAX_CHUNK_KB"),
    "MX_GDS_THREADS": lambda: _env_int("MX_GDS_THREADS", 8),
    "MX_GDS_TIMEOUT": lambda: _env_float("MX_GDS_TIMEOUT", 120.0),
    # ── Model streamer ─────────────────────────────────────────────────────
    "MX_MS_DISTRIBUTED": lambda: (
        os.environ.get("MX_MS_DISTRIBUTED", "1").lower() in ("1", "true")
    ),
    # ── InstantTensor loader ───────────────────────────────────────────────
    # Enabled by default; the strategy is still gated on the instanttensor
    # package and a CUDA device, so opting out is only needed to force a
    # different local-load path. Set MX_INSTANT_TENSOR=0 to disable.
    "MX_INSTANT_TENSOR": lambda: (
        os.environ.get("MX_INSTANT_TENSOR", "1").lower() in ("1", "true")
    ),
    # ── TRT-LLM live transfer ──────────────────────────────────────────────
    "MX_SOURCE_QUERY_TIMEOUT": lambda: _env_int("MX_SOURCE_QUERY_TIMEOUT", 3600),
    "MX_TRANSFER_TIMEOUT": lambda: _env_int("MX_TRANSFER_TIMEOUT", 900),
    "MX_TRANSFER_LOG_DIR": lambda: os.environ.get(
        "MX_TRANSFER_LOG_DIR", "/tmp/mx_logs"
    ),
    # ── VMM arena ──────────────────────────────────────────────────────────
    "MX_VMM_ARENA": lambda: os.environ.get("MX_VMM_ARENA") == "1",
    "MX_ARENA_SINGLE_MR": lambda: os.environ.get("MX_ARENA_SINGLE_MR") == "1",
    # ── Framework artifact (JIT cache) transfer ────────────────────────────
    "MX_ARTIFACT_TRANSFER": lambda: (
        os.environ.get("MX_ARTIFACT_TRANSFER", "").strip().lower() in _TRUTHY
    ),
    "MX_ARTIFACT_BUNDLE_ROOT": lambda: os.environ.get("MX_ARTIFACT_BUNDLE_ROOT"),
    "MX_ARTIFACT_COMPILE_CONFIG_DIGEST": lambda: os.environ.get(
        "MX_ARTIFACT_COMPILE_CONFIG_DIGEST", ""
    ),
    "MX_ARTIFACT_BACKEND": lambda: os.environ.get("MX_ARTIFACT_BACKEND", "nixl")
    .strip()
    .lower(),
    "MX_ARTIFACT_MOONCAKE_NAMESPACE": lambda: os.environ.get(
        "MX_ARTIFACT_MOONCAKE_NAMESPACE", "modelexpress/artifacts"
    ).strip(),
    "MX_ARTIFACT_MOONCAKE_POOL_BYTES": lambda: _env_int(
        "MX_ARTIFACT_MOONCAKE_POOL_BYTES", 512 * 1024 * 1024
    ),
    "MX_ARTIFACT_MOONCAKE_DELETE_RETRIES": lambda: _env_int(
        "MX_ARTIFACT_MOONCAKE_DELETE_RETRIES", 10
    ),
    "MX_ARTIFACT_MOONCAKE_DELETE_RETRY_DELAY_SECS": lambda: _env_float(
        "MX_ARTIFACT_MOONCAKE_DELETE_RETRY_DELAY_SECS", 0.7
    ),
    "MX_ARTIFACT_READY_URL": lambda: os.environ.get("MX_ARTIFACT_READY_URL", ""),
    "MX_ARTIFACT_READY_TIMEOUT_SECS": lambda: _env_int(
        "MX_ARTIFACT_READY_TIMEOUT_SECS", 1800
    ),
    # Raw string: artifact_manifest.artifact_transfer_chunk_size() owns the
    # int parse plus its non-positive/max-bound validation and default param.
    "MX_ARTIFACT_TRANSFER_CHUNK_SIZE": lambda: os.environ.get(
        "MX_ARTIFACT_TRANSFER_CHUNK_SIZE"
    ),
    # ── P2P source selection ───────────────────────────────────────────────
    # Raw (None when unset); source_selection applies its DEFAULT_SELECTOR fallback.
    "MX_P2P_SOURCE_SELECTOR": lambda: os.environ.get("MX_P2P_SOURCE_SELECTOR"),
    # Clamp to >= 0: a negative weight would invert LoadAwareSelector into
    # preferring busier sources. 0 disables the load term (== rendezvous_hash).
    "MX_P2P_LOAD_WEIGHT": lambda: max(0.0, _env_float("MX_P2P_LOAD_WEIGHT", 1.0)),
    "MX_P2P_RUNTIME_METRICS_URL": lambda: os.environ.get("MX_P2P_RUNTIME_METRICS_URL"),
    "MX_P2P_TOPOLOGY_LEVELS": lambda: os.environ.get("MX_P2P_TOPOLOGY_LEVELS"),
    "MX_P2P_TOPOLOGY": lambda: os.environ.get("MX_P2P_TOPOLOGY"),
    # Clamp to >= 0: a negative weight would invert the within-tier load blend
    # into preferring busy sources. 0 (default) keeps the pure rendezvous jitter.
    "MX_P2P_TOPOLOGY_LOAD_WEIGHT": lambda: max(
        0.0, _env_float("MX_P2P_TOPOLOGY_LOAD_WEIGHT", 0.0)
    ),
    # Raw; topology.resolve_policy validates and applies the "required" default.
    "MX_P2P_TOPOLOGY_DOMAIN": lambda: os.environ.get("MX_P2P_TOPOLOGY_DOMAIN"),
    "MX_P2P_TOPOLOGY_ENFORCEMENT": lambda: os.environ.get("MX_P2P_TOPOLOGY_ENFORCEMENT"),
    # ── Opt-in metrics collector ───────────────────────────────────────────
    "MX_METRICS_ENABLED": lambda: (
        os.environ.get("MX_METRICS_ENABLED", "0").strip().lower() in _TRUTHY
    ),
    "MX_METRICS_PORT": lambda: os.environ.get("MX_METRICS_PORT"),
    "MX_METRICS_PUSHGATEWAY": lambda: os.environ.get("MX_METRICS_PUSHGATEWAY"),
    "MX_METRICS_SCHEME": lambda: os.environ.get("MX_METRICS_SCHEME", ""),
    # Interval at which a rank that lost the /metrics bind re-attempts it, so
    # endpoint ownership migrates when the winning rank exits.
    "MX_METRICS_BIND_RETRY_SECS": lambda: _env_float(
        "MX_METRICS_BIND_RETRY_SECS", 15.0
    ),
    # Restore the per-peer source_worker_id label on
    # mx_p2p_source_selections_total. Benchmark runs only: the id is a
    # per-process uuid, so its label domain grows with process count over time
    # rather than with cluster size.
    "MX_METRICS_SOURCE_ID_LABEL": lambda: (
        os.environ.get("MX_METRICS_SOURCE_ID_LABEL", "0").strip().lower() in _TRUTHY
    ),
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
    "POD_UID": lambda: os.environ.get("POD_UID", ""),
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
