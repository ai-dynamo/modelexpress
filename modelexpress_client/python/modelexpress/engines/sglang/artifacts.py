# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang cache artifact configuration for the ModelExpress loader."""

from __future__ import annotations

import logging
import tempfile
from getpass import getuser
from importlib.metadata import version as pkg_version
from pathlib import Path

import torch

from ... import envs
from ... import p2p_pb2
from ...load_strategy.context import LoadContext
from ...metadata import artifact_lifecycle as _common_artifacts
from ...metadata.artifact_transfer import (
    ArtifactCacheRoot,
    P2PArtifactTransfer,
    PublishedArtifactSource,
    cute_dsl_cache_artifact_transfer,
    deep_gemm_cache_artifact_transfer,
    flashinfer_cache_artifact_transfer,
    tilelang_cache_artifact_transfer,
    triton_cache_artifact_transfer,
    torch_compile_cache_artifact_transfer,
)
from ...metadata.publisher import PublisherThread

logger = logging.getLogger("modelexpress.engines.sglang.artifacts")

_DEFAULT_READY_URL = "http://127.0.0.1:30000/health"
_published_sources: dict[tuple[int, int], PublishedArtifactSource] = {}
_scheduled_publishers: dict[tuple[int, int], PublisherThread] = {}


def install_sglang_cache_artifacts(ctx: LoadContext) -> None:
    """Best-effort install of compatible SGLang cache artifacts before load."""
    _common_artifacts.install_artifacts(
        ctx,
        lambda: _sglang_artifact_transfers(ctx),
        engine_label="SGLang",
        log=logger,
    )


def schedule_sglang_cache_artifact_publish(ctx: LoadContext) -> None:
    """Schedule publication of local SGLang artifacts after server readiness."""
    _common_artifacts.schedule_artifact_publish(
        ctx,
        lambda: _sglang_artifact_transfers(ctx),
        engine_label="SGLang",
        ready_fn_factory=lambda roots: _sglang_artifact_ready_fn(roots),
        artifact_publish_fn=lambda transfer, identity: (
            _publish_sglang_cache_artifact(ctx, transfer, identity)
        ),
        scheduled_publishers=_scheduled_publishers,
        log=logger,
    )


def _artifact_transfer_enabled() -> bool:
    return envs.MX_ARTIFACT_TRANSFER


def _publish_sglang_cache_artifact(
    ctx: LoadContext,
    transfer: P2PArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
) -> PublishedArtifactSource:
    """Compatibility wrapper for the shared source publication operation."""
    return _common_artifacts.publish_artifact(
        ctx,
        transfer,
        identity,
        engine_label="SGLang",
        accelerator=ctx.accelerator_backend.name,
        published_sources=_published_sources,
        log=logger,
    )


def _sglang_artifact_transfers(
    ctx: LoadContext,
) -> list[tuple[P2PArtifactTransfer, p2p_pb2.SourceIdentity]]:
    bundle_root = _bundle_root(ctx)
    torch_compile_cache_root = _torch_compile_cache_root()
    triton_cache_root = _common_artifacts.triton_cache_root()
    deep_gemm_cache_root = _deep_gemm_cache_root()
    tilelang_cache_root = _common_artifacts.tilelang_cache_root()
    cute_dsl_cache_root = _common_artifacts.cute_dsl_cache_root()
    flashinfer_cache_root = _common_artifacts.flashinfer_cache_root()
    flashinfer_autotune_cache_root = _flashinfer_autotune_cache_root()
    return [
        (
            torch_compile_cache_artifact_transfer(
                torch_compile_cache_root,
                torch_compile_cache_root,
                bundle_root / "torch_compile_cache",
            ),
            _artifact_identity(ctx, p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE),
        ),
        (
            triton_cache_artifact_transfer(
                triton_cache_root,
                triton_cache_root,
                bundle_root / "triton_cache",
            ),
            _artifact_identity(ctx, p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE),
        ),
        (
            deep_gemm_cache_artifact_transfer(
                deep_gemm_cache_root,
                deep_gemm_cache_root,
                bundle_root / "deep_gemm_cache",
            ),
            _artifact_identity(ctx, p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE),
        ),
        (
            tilelang_cache_artifact_transfer(
                tilelang_cache_root,
                tilelang_cache_root,
                bundle_root / "tilelang_cache",
            ),
            _artifact_identity(ctx, p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE),
        ),
        (
            cute_dsl_cache_artifact_transfer(
                cute_dsl_cache_root,
                cute_dsl_cache_root,
                bundle_root / "cute_dsl_cache",
            ),
            _artifact_identity(ctx, p2p_pb2.MX_SOURCE_TYPE_CUTE_DSL_CACHE),
        ),
        (
            flashinfer_cache_artifact_transfer(
                flashinfer_cache_root,
                flashinfer_cache_root,
                bundle_root / "flashinfer_cache",
                additional_roots=(
                    ArtifactCacheRoot(
                        name="autotune",
                        source_root=flashinfer_autotune_cache_root,
                        target_root=flashinfer_autotune_cache_root,
                        optional=True,
                    ),
                ),
            ),
            _artifact_identity(ctx, p2p_pb2.MX_SOURCE_TYPE_FLASHINFER_CACHE),
        ),
    ]


def _torch_compile_cache_root() -> Path:
    configured = envs.TORCHINDUCTOR_CACHE_DIR
    if configured:
        return Path(configured)
    try:
        from torch._inductor.codecache import cache_dir

        return Path(cache_dir())
    except Exception:
        try:
            user = getuser()
        except (KeyError, OSError):
            user = "unknown"
        return Path(tempfile.gettempdir()) / f"torchinductor_{user}"


def _deep_gemm_cache_root() -> Path:
    configured = envs.SGLANG_DG_CACHE_DIR
    return Path(configured) if configured else Path.home() / ".cache" / "deep_gemm"


def _flashinfer_autotune_cache_root() -> Path:
    configured = envs.SGLANG_CACHE_DIR
    return (
        Path(configured) / "flashinfer" / "autotune"
        if configured
        else Path.home() / ".cache" / "sglang" / "flashinfer" / "autotune"
    )


def _bundle_root(ctx: LoadContext) -> Path:
    configured = envs.MX_ARTIFACT_BUNDLE_ROOT
    if configured:
        return Path(configured) / f"rank-{ctx.worker_rank}"
    return (
        Path(tempfile.gettempdir())
        / "modelexpress-artifacts"
        / f"worker-{ctx.worker_id}"
        / f"rank-{ctx.worker_rank}"
    )


def _artifact_identity(
    ctx: LoadContext,
    mx_source_type: int,
) -> p2p_pb2.SourceIdentity:
    builders = {
        p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE: _torch_compile_cache_identity,
        p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE: _triton_cache_identity,
        p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE: _deep_gemm_cache_identity,
        p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE: _tilelang_cache_identity,
        p2p_pb2.MX_SOURCE_TYPE_CUTE_DSL_CACHE: _cute_dsl_cache_identity,
        p2p_pb2.MX_SOURCE_TYPE_FLASHINFER_CACHE: _flashinfer_cache_identity,
    }
    builder = builders.get(mx_source_type)
    if builder is None:
        raise ValueError(
            f"unknown SGLang artifact source type: {mx_source_type}"
        )
    return builder(ctx)


def _torch_compile_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        model_name=ctx.identity.model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
        tensor_parallel_size=ctx.identity.tensor_parallel_size,
        pipeline_parallel_size=ctx.identity.pipeline_parallel_size,
        expert_parallel_size=ctx.identity.expert_parallel_size,
        dtype=ctx.identity.dtype,
        quantization=ctx.identity.quantization,
        revision=ctx.identity.revision,
        backend_framework_version=_sglang_version(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "",
        triton_version=_common_artifacts.triton_version(),
        gpu_arch=_common_artifacts.gpu_arch(ctx.device_id),
        compile_config_digest=envs.MX_ARTIFACT_COMPILE_CONFIG_DIGEST,
    )
    _set_extra_if_present(identity, "triton_key", _common_artifacts.triton_key())
    return identity


def _triton_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        model_name=ctx.identity.model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
        cuda_version=torch.version.cuda or "",
        triton_version=_common_artifacts.triton_version(),
        gpu_arch=_common_artifacts.gpu_arch(ctx.device_id),
    )
    _set_extra_if_present(identity, "triton_key", _common_artifacts.triton_key())
    return identity


def _deep_gemm_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE,
        model_name=ctx.identity.model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
        cuda_version=torch.version.cuda or "",
        gpu_arch=_common_artifacts.gpu_arch(ctx.device_id),
    )
    _set_extra_if_present(
        identity,
        "deep_gemm_jit_key",
        _common_artifacts.deep_gemm_jit_key(),
    )
    return identity


def _tilelang_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE,
        model_name=ctx.identity.model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
        cuda_version=torch.version.cuda or "",
        gpu_arch=_common_artifacts.gpu_arch(ctx.device_id),
    )
    _set_extra_if_present(
        identity,
        "tilelang_version",
        _common_artifacts.tilelang_version(),
    )
    return identity


def _cute_dsl_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_CUTE_DSL_CACHE,
        model_name=ctx.identity.model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
        cuda_version=torch.version.cuda or "",
        gpu_arch=_common_artifacts.gpu_arch(ctx.device_id),
    )
    _set_extra_if_present(
        identity,
        "cutlass_dsl_version",
        _common_artifacts.cutlass_dsl_version(),
    )
    return identity


def _flashinfer_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_FLASHINFER_CACHE,
        model_name=ctx.identity.model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "",
        gpu_arch=_common_artifacts.gpu_arch(ctx.device_id),
    )
    _set_extra_if_present(
        identity,
        "flashinfer_version",
        _common_artifacts.flashinfer_version(),
    )
    return identity


def _set_extra_if_present(
    identity: p2p_pb2.SourceIdentity,
    key: str,
    value: str,
) -> None:
    if value:
        identity.extra_parameters[key] = value


def _sglang_artifact_ready_fn(source_roots: tuple[ArtifactCacheRoot, ...]):
    return _common_artifacts.artifact_ready_fn(source_roots, _sglang_health_ready)


def _sglang_health_ready() -> bool:
    return _common_artifacts.artifact_health_ready(_sglang_health_url())


def _sglang_health_url() -> str:
    configured = envs.MX_ARTIFACT_READY_URL.strip()
    fallback = (
        _common_artifacts.statefulset_head_health_url(port=30000)
        or _DEFAULT_READY_URL
    )
    if not configured or configured == _DEFAULT_READY_URL:
        return fallback
    if _common_artifacts.is_http_url(configured):
        return configured
    logger.warning("Invalid MX_ARTIFACT_READY_URL=%r; using %s", configured, fallback)
    return fallback


def _sglang_version() -> str:
    try:
        import sglang

        version = getattr(sglang, "__version__", "")
        if isinstance(version, str) and version:
            return version
    except Exception:
        logger.debug("Failed to read SGLang package version", exc_info=True)
    try:
        return pkg_version("sglang")
    except Exception:
        return ""
