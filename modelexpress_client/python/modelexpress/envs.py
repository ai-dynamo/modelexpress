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
- ``MODEL_EXPRESS_SOURCE`` only appears in a docstring example, not live code.
- The deprecated ``MX_VMM_ARENA_BYTES`` / ``MX_VMM_ARENA_CHUNK_BYTES`` are
  presence-only deprecation warnings; check them with :func:`is_set`.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Callable, Optional

logger = logging.getLogger("modelexpress.envs")

# Static type hints for editors/type-checkers. The actual values are produced
# by ``__getattr__`` below; these annotations never execute at runtime.
if TYPE_CHECKING:
    # ModelExpress server address / logging
    MODEL_EXPRESS_URL: Optional[str]
    MX_SERVER_ADDRESS: Optional[str]
    MODEL_EXPRESS_LOG_LEVEL: str
    MODEL_NAME: Optional[str]
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
    # TRT-LLM live transfer
    MX_SOURCE_QUERY_TIMEOUT: int
    MX_TRANSFER_TIMEOUT: int
    MX_TRANSFER_LOG_DIR: str
    # VMM arena
    MX_VMM_ARENA: bool
    # vLLM artifact (JIT cache) transfer
    MX_ARTIFACT_TRANSFER: bool
    MX_ARTIFACT_BUNDLE_ROOT: Optional[str]
    MX_ARTIFACT_COMPILE_CONFIG_DIGEST: str
    MX_ARTIFACT_READY_URL: str
    MX_ARTIFACT_READY_TIMEOUT_SECS: int
    MX_ARTIFACT_TRANSFER_CHUNK_SIZE: Optional[str]
    # P2P source selection
    MX_P2P_SOURCE_SELECTOR: Optional[str]
    MX_P2P_LOAD_WEIGHT: float
    # Opt-in metrics collector
    MX_METRICS_ENABLED: bool
    MX_METRICS_PORT: Optional[str]
    MX_METRICS_PUSHGATEWAY: Optional[str]
    MX_METRICS_SCHEME: str
    # Third-party JIT/compile cache locations read for artifact transfer
    TRITON_CACHE_DIR: Optional[str]
    DG_JIT_CACHE_DIR: Optional[str]
    DEEP_GEMM_CACHE_DIR: Optional[str]
    TILELANG_CACHE_DIR: Optional[str]
    CUTE_DSL_CACHE_DIR: Optional[str]
    FLASHINFER_WORKSPACE_BASE: Optional[str]
    VLLM_CACHE_ROOT: Optional[str]
    # Other third-party / system
    VLLM_ATTENTION_BACKEND: str
    HOSTNAME: str
    POD_NAMESPACE: str

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


# One entry per variable. The lambda owns the default and parsing; callers that
# need a site-specific default receive the raw value (``None`` when unset) and
# apply their own fallback.
environment_variables: dict[str, Callable[[], Any]] = {
    # ── ModelExpress server address / logging ──────────────────────────────
    # Site-varying defaults: return raw (None when unset), callers add defaults.
    "MODEL_EXPRESS_URL": lambda: os.environ.get("MODEL_EXPRESS_URL"),
    "MX_SERVER_ADDRESS": lambda: os.environ.get("MX_SERVER_ADDRESS"),
    "MODEL_EXPRESS_LOG_LEVEL": lambda: os.environ.get(
        "MODEL_EXPRESS_LOG_LEVEL", ""
    ).upper(),
    "MODEL_NAME": lambda: os.environ.get("MODEL_NAME"),
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
    "MX_P2P_METADATA": lambda: os.environ.get("MX_P2P_METADATA", ""),
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
    "MX_POOL_REG": lambda: os.environ.get("MX_POOL_REG", "0") == "1",
    "NIXL_UCX_TLS": lambda: os.environ.get("NIXL_UCX_TLS"),
    "UCX_TLS": lambda: os.environ.get("UCX_TLS"),
    "UCX_NET_DEVICES": lambda: os.environ.get("UCX_NET_DEVICES"),
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
        os.environ.get("MX_MS_DISTRIBUTED", "0").lower() in ("1", "true")
    ),
    # ── TRT-LLM live transfer ──────────────────────────────────────────────
    "MX_SOURCE_QUERY_TIMEOUT": lambda: _env_int("MX_SOURCE_QUERY_TIMEOUT", 3600),
    "MX_TRANSFER_TIMEOUT": lambda: _env_int("MX_TRANSFER_TIMEOUT", 900),
    "MX_TRANSFER_LOG_DIR": lambda: os.environ.get(
        "MX_TRANSFER_LOG_DIR", "/tmp/mx_logs"
    ),
    # ── VMM arena ──────────────────────────────────────────────────────────
    "MX_VMM_ARENA": lambda: os.environ.get("MX_VMM_ARENA") == "1",
    # ── vLLM artifact (JIT cache) transfer ─────────────────────────────────
    "MX_ARTIFACT_TRANSFER": lambda: (
        os.environ.get("MX_ARTIFACT_TRANSFER", "").strip().lower() in _TRUTHY
    ),
    "MX_ARTIFACT_BUNDLE_ROOT": lambda: os.environ.get("MX_ARTIFACT_BUNDLE_ROOT"),
    "MX_ARTIFACT_COMPILE_CONFIG_DIGEST": lambda: os.environ.get(
        "MX_ARTIFACT_COMPILE_CONFIG_DIGEST", ""
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
    # Weight of the load penalty in the load_aware selector (0 disables it,
    # collapsing load_aware to rendezvous_hash). Default 1.0.
    "MX_P2P_LOAD_WEIGHT": lambda: _env_float("MX_P2P_LOAD_WEIGHT", 1.0),
    # ── Opt-in metrics collector ───────────────────────────────────────────
    "MX_METRICS_ENABLED": lambda: (
        os.environ.get("MX_METRICS_ENABLED", "0").strip().lower() in _TRUTHY
    ),
    "MX_METRICS_PORT": lambda: os.environ.get("MX_METRICS_PORT"),
    "MX_METRICS_PUSHGATEWAY": lambda: os.environ.get("MX_METRICS_PUSHGATEWAY"),
    "MX_METRICS_SCHEME": lambda: os.environ.get("MX_METRICS_SCHEME", ""),
    # ── Third-party JIT/compile cache locations (raw; caller builds path) ──
    "TRITON_CACHE_DIR": lambda: os.environ.get("TRITON_CACHE_DIR"),
    "DG_JIT_CACHE_DIR": lambda: os.environ.get("DG_JIT_CACHE_DIR"),
    "DEEP_GEMM_CACHE_DIR": lambda: os.environ.get("DEEP_GEMM_CACHE_DIR"),
    "TILELANG_CACHE_DIR": lambda: os.environ.get("TILELANG_CACHE_DIR"),
    "CUTE_DSL_CACHE_DIR": lambda: os.environ.get("CUTE_DSL_CACHE_DIR"),
    "FLASHINFER_WORKSPACE_BASE": lambda: os.environ.get("FLASHINFER_WORKSPACE_BASE"),
    "VLLM_CACHE_ROOT": lambda: os.environ.get("VLLM_CACHE_ROOT"),
    # ── Other third-party / system ─────────────────────────────────────────
    "VLLM_ATTENTION_BACKEND": lambda: os.environ.get("VLLM_ATTENTION_BACKEND", "auto"),
    "HOSTNAME": lambda: os.environ.get("HOSTNAME", ""),
    "POD_NAMESPACE": lambda: os.environ.get("POD_NAMESPACE", ""),
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
