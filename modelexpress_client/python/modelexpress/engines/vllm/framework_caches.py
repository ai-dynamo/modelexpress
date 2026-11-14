# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-owned file caches published as framework cache artifacts.

Each provider inspects the running vLLM release and returns a
``FrameworkCacheArtifact`` or ``None``. Providers run on the main model's
adapter before ``initialize_model``, so a compatibility key computed here sees
the configuration as the user specified it rather than as model construction
later mutates it. A provider that fails is skipped without affecting other
caches or model loading. To support another vLLM cache, add a provider to
``_PROVIDERS``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import fields, is_dataclass
from enum import Enum
from importlib import import_module
from pathlib import Path, PurePosixPath

import torch

from ... import p2p_pb2
from ...metadata.framework_cache import FrameworkCacheArtifact

logger = logging.getLogger(__name__)

FrameworkCacheProvider = Callable[[object], "FrameworkCacheArtifact | None"]

_MODEL_CONFIG_LOCAL_FIELDS = {
    "hf_token", "model", "model_weights", "hf_config_path", "tokenizer",
    "tokenizer_revision", "served_model_name", "allowed_local_media_path",
    "allowed_media_domains",
}
_PARALLEL_BACKEND_FIELDS = (
    "disable_custom_all_reduce", "worker_cls", "sd_worker_cls",
    "worker_extension_cls", "distributed_executor_backend", "data_parallel_backend",
)
# Plans saved by this boot are newer than the descriptor. The margin absorbs
# file systems that store modification times at one-second granularity.
_MTIME_MARGIN_SECS = 2.0


def build_framework_cache_artifacts(vllm_config) -> tuple[FrameworkCacheArtifact, ...]:
    """Return the framework caches the running vLLM release supports."""
    caches = []
    for provider in _PROVIDERS:
        try:
            cache = provider(vllm_config)
        except Exception as exc:  # noqa: BLE001 - one cache must not break loading
            logger.warning("Skipping vLLM %s artifact: %s", _provider_name(provider), exc)
            continue
        if cache is None:
            continue
        logger.info(
            "vLLM framework cache: name=%s schema=%s compatibility_key=%s root=%s",
            cache.name, cache.schema, cache.compatibility_key, cache.root,
        )
        caches.append(cache)
    return tuple(caches)


def startup_plan_cache(vllm_config) -> FrameworkCacheArtifact | None:
    """Share the KV-cache memory plans of ``VLLM_ENABLE_STARTUP_PLAN``.

    vLLM saves one plan per rank under ``VLLM_CACHE_ROOT/startup_plan`` and
    applies a plan only when its fingerprint matches and enough device memory
    is free. ModelExpress installs fetched plans into that directory and
    leaves selection and the free-memory check to vLLM. No vLLM function is
    replaced.
    """
    runtime_envs = import_module("vllm.envs")
    if getattr(runtime_envs, "VLLM_ENABLE_STARTUP_PLAN", False) is not True:
        return None
    if vllm_config.cache_config.kv_cache_memory_bytes is not None:
        # vLLM never consults a plan when the KV-cache size is explicit.
        return None
    native = import_module("vllm.v1.worker.startup_plan")
    schema = getattr(native, "PLAN_SCHEMA_VERSION", None)
    if type(schema) is not int or schema != 1:
        raise ValueError(f"unsupported native startup plan schema: {schema}")
    for method in (
        "compute_plan_fingerprint", "_plan_path",
        "maybe_apply_startup_plan", "maybe_save_startup_plan",
    ):
        if not callable(getattr(native, method, None)):
            raise ValueError(f"native startup plan API is missing {method}")
    parallel_config = vllm_config.parallel_config
    world_size = parallel_config.world_size
    native_key = native.compute_plan_fingerprint(vllm_config, 0, world_size)
    if not isinstance(native_key, str) or not native_key:
        raise ValueError("native startup plan fingerprint is unavailable")
    # The native fingerprint omits settings that change the profiled KV-cache
    # size, such as gpu_memory_utilization, and draft-model details. The key
    # adds them so that a fetched plan always comes from the same settings.
    factors = {
        "native_fingerprint": native_key,
        "cache_config": _stable_config(vllm_config.cache_config),
        "model_config": _stable_config(
            vllm_config.model_config, excluded=_MODEL_CONFIG_LOCAL_FIELDS
        ),
        "scheduler_config": _stable_config(getattr(vllm_config, "scheduler_config", None)),
        "parallel_backends": _stable_config({
            name: getattr(parallel_config, name, None) for name in _PARALLEL_BACKEND_FIELDS
        }),
        # Each node saves the plans of its own ranks.
        "node_rank": int(getattr(parallel_config, "node_rank", 0) or 0),
        "compile_environment": _stable_config(runtime_envs.compile_factors()),
        "device": _device_signature(),
    }
    speculative = getattr(vllm_config, "speculative_config", None)
    if speculative is not None:
        factors["speculative_config"] = _speculative_config_snapshot(speculative)
    root = Path(native._plan_path(native_key)).parent
    return FrameworkCacheArtifact(
        name="startup_plan",
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        backend_framework_version=import_module("vllm").__version__,
        compatibility_key=hashlib.sha256(
            json.dumps(factors, sort_keys=True).encode()
        ).hexdigest(),
        schema=str(schema),
        root=root,
        include=_StartupPlanSelection(
            native, vllm_config, root, time.time() - _MTIME_MARGIN_SECS
        ),
    )


class _StartupPlanSelection:
    """Select the plans this boot saved for its own ranks.

    vLLM fingerprints the configuration as it stands after model construction,
    which can differ from the configuration this module snapshots beforehand.
    The fingerprints are therefore computed when the bundle is prepared, after
    readiness, which is when vLLM has saved them. Plans saved by an earlier
    boot are left out: their settings may differ in ways the native fingerprint
    does not cover, and installed plans are already published elsewhere.
    """

    def __init__(self, native, vllm_config, root: Path, saved_after: float) -> None:
        self._native = native
        self._vllm_config = vllm_config
        self._root = root
        self._saved_after = saved_after
        self._names: frozenset[str] | None = None

    def __call__(self, relative_path: str) -> bool:
        path = PurePosixPath(relative_path)
        if len(path.parts) != 1 or path.name not in self._plan_names():
            return False
        try:
            return (self._root / path.name).stat().st_mtime >= self._saved_after
        except OSError:
            return False

    def _plan_names(self) -> frozenset[str]:
        if self._names is None:
            world_size = self._vllm_config.parallel_config.world_size
            self._names = frozenset(
                Path(self._native._plan_path(
                    self._native.compute_plan_fingerprint(self._vllm_config, rank, world_size)
                )).name
                for rank in range(world_size)
            )
        return self._names


def _speculative_config_snapshot(speculative) -> dict:
    snapshot = _stable_config(speculative, excluded={
        "target_model_config", "target_parallel_config",
        "draft_model_config", "draft_parallel_config",
    })
    for name in ("draft_model_config", "draft_parallel_config"):
        config = getattr(speculative, name, None)
        if config is None:
            continue
        native_hash = config.compute_hash()
        if not isinstance(native_hash, str) or not native_hash.strip():
            raise ValueError(f"native {name} hash is unavailable")
        if name == "draft_model_config":
            # vLLM injects an HF override callback; its resolved result is hashed above.
            profiling = _stable_config(
                config, excluded=_MODEL_CONFIG_LOCAL_FIELDS | {"hf_overrides"}
            )
        else:
            profiling = _stable_config({
                field: getattr(config, field, None) for field in _PARALLEL_BACKEND_FIELDS
            })
        snapshot[name] = {"native_hash": native_hash, "profiling": profiling}
    return snapshot


def _stable_config(value, *, excluded: set[str] | None = None):
    """Snapshot configuration values without runtime registries or object reprs."""
    if isinstance(value, Enum):
        return _stable_config(value.value)
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _stable_config(getattr(value, field.name))
            for field in fields(value)
            if field.init and field.name not in (excluded or ())
        }
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _stable_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stable_config(item) for item in value]
    raise ValueError(f"unsupported startup profiling setting: {type(value).__name__}")


def _device_signature() -> dict[str, str | int]:
    """Describe the worker's device; vLLM requires identical devices per engine."""
    if not torch.cuda.is_available():
        raise ValueError("no CUDA device available for startup profiling")
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "name": device.name,
        "total_memory": device.total_memory,
        "major": device.major,
        "minor": device.minor,
    }


def _provider_name(provider: FrameworkCacheProvider) -> str:
    name = getattr(provider, "__name__", type(provider).__name__)
    return name.removesuffix("_cache").replace("_", " ")


_PROVIDERS: tuple[FrameworkCacheProvider, ...] = (startup_plan_cache,)
