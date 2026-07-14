# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang cache artifact integration for the ModelExpress loader."""

from __future__ import annotations

import logging
import time
import tempfile
from getpass import getuser
from importlib.metadata import version as pkg_version
from pathlib import Path

import torch

from ... import envs
from ... import p2p_pb2
from ...load_strategy.base import (
    _init_nixl_manager,
    _metadata_publication_configured,
)
from ...load_strategy.context import LoadContext
from ...metadata.artifact_transfer import (
    ArtifactCacheRoot,
    P2PArtifactTransfer,
    PublishedArtifactSource,
    cute_dsl_cache_artifact_transfer,
    deep_gemm_cache_artifact_transfer,
    flashinfer_cache_artifact_transfer,
    publish_artifact_source,
    tilelang_cache_artifact_transfer,
    triton_cache_artifact_transfer,
    torch_compile_cache_artifact_transfer,
)
from ...metadata.publisher import PublisherThread
from ...metadata.publish import _get_worker_server, _is_p2p_metadata_enabled
from ...nixl_transfer import is_nixl_available
from ..vllm import artifacts as _common_artifacts

logger = logging.getLogger("modelexpress.engines.sglang.artifacts")

_DEFAULT_READY_URL = "http://127.0.0.1:30000/health"
_READY_POLL_SECS = 5

_published_sources: dict[tuple[int, int], PublishedArtifactSource] = {}
_scheduled_publishers: dict[tuple[int, int], PublisherThread] = {}


def install_sglang_cache_artifacts(ctx: LoadContext) -> None:
    """Best-effort install of compatible SGLang cache artifacts before load."""
    if not _artifact_transfer_enabled():
        return
    if not _p2p_metadata_enabled_for_artifacts(ctx):
        return
    if not _metadata_publication_configured(ctx):
        logger.info(
            "[Worker %s] No MX metadata path configured, skipping SGLang artifacts",
            ctx.global_rank,
        )
        return
    if not is_nixl_available():
        logger.info(
            "[Worker %s] NIXL not available, skipping SGLang artifact install",
            ctx.global_rank,
        )
        return

    _ensure_nixl_manager(ctx)
    if ctx.nixl_manager is None:
        return

    for transfer, identity in _sglang_artifact_transfers(ctx):
        try:
            start = time.perf_counter()
            header = _common_artifacts._install_vllm_cache_artifact_once(
                ctx,
                transfer,
                identity,
            )
            elapsed = time.perf_counter() - start
            if header is None:
                logger.debug(
                    "[Worker %s] SGLang artifact %s already attempted in this pod",
                    ctx.global_rank,
                    transfer.name,
                )
                continue
            logger.info(
                "[Worker %s] [TIMING] SGLang artifact install complete: "
                "name=%s artifact_id=%s size=%.2f MiB elapsed=%.3fs",
                ctx.global_rank,
                transfer.name,
                header.artifact_id,
                header.total_size / (1024 * 1024),
                elapsed,
            )
        except LookupError:
            logger.debug(
                "[Worker %s] No ready SGLang artifact source for %s",
                ctx.global_rank,
                transfer.name,
            )
        except Exception as exc:
            logger.warning(
                "[Worker %s] Failed to install SGLang artifact %s: %s",
                ctx.global_rank,
                transfer.name,
                exc,
            )


def schedule_sglang_cache_artifact_publish(ctx: LoadContext) -> None:
    """Schedule publication of local SGLang artifacts after server readiness."""
    if not _artifact_transfer_enabled():
        return
    if not _p2p_metadata_enabled_for_artifacts(ctx):
        return
    if not _metadata_publication_configured(ctx):
        logger.info(
            "[Worker %s] No MX metadata path configured, skipping SGLang artifacts",
            ctx.global_rank,
        )
        return
    if ctx.nixl_manager is None:
        logger.info(
            "[Worker %s] No NIXL manager, skipping SGLang artifact publish",
            ctx.global_rank,
        )
        return

    for transfer, identity in _sglang_artifact_transfers(ctx):
        marker_path = _common_artifacts._mark_vllm_cache_artifact_publish_scheduled(
            ctx,
            transfer,
            identity,
        )
        if marker_path is None:
            continue
        key = (ctx.device_id, transfer.mx_source_type)
        previous = _scheduled_publishers.pop(key, None)
        if previous is not None:
            previous.stop()

        source_roots = transfer.roots
        publisher_ref: list[PublisherThread | None] = [None]
        publisher = PublisherThread(
            mx_client=ctx.mx_client,
            worker_id=ctx.worker_id,
            worker_rank=ctx.worker_rank,
            nixl_manager=ctx.nixl_manager,
            publish_fn=lambda transfer=transfer, identity=identity: (
                _publish_sglang_cache_artifact(ctx, transfer, identity).endpoint.mx_source_id
            ),
            ready_fn=_sglang_artifact_ready_fn(source_roots),
            publish_timeout_secs=envs.MX_ARTIFACT_READY_TIMEOUT_SECS,
            interval_secs=_READY_POLL_SECS,
            heartbeat_after_publish=False,
            cleanup_fn=lambda marker_path=marker_path, publisher_ref=publisher_ref: (
                _common_artifacts._clear_vllm_cache_artifact_publish_scheduled(
                    publisher_ref[0],
                    marker_path,
                )
            ),
        )
        publisher_ref[0] = publisher
        _scheduled_publishers[key] = publisher
        publisher.start()
        logger.info(
            "[Worker %s] Scheduled SGLang artifact publisher: name=%s roots=%s",
            ctx.global_rank,
            transfer.name,
            [str(root.source_root) for root in source_roots],
        )


def _publish_sglang_cache_artifact(
    ctx: LoadContext,
    transfer: P2PArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
) -> PublishedArtifactSource:
    if ctx.nixl_manager is None:
        raise RuntimeError("NIXL manager is required for SGLang artifact publish")
    worker_grpc_server = _get_worker_server(ctx.device_id)
    if worker_grpc_server is None:
        raise RuntimeError("P2P worker gRPC server is required for artifact publish")
    required_roots = tuple(
        root.source_root for root in transfer.roots if not root.optional
    )
    if not all(_common_artifacts._has_files(path) for path in required_roots):
        raise LookupError(
            f"Required SGLang artifact sources {transfer.name} are empty or missing: "
            f"{required_roots}"
        )

    start = time.perf_counter()
    bundle = transfer.prepare_source()
    key = (ctx.device_id, transfer.mx_source_type)
    previous = _published_sources.pop(key, None)
    if previous is not None:
        previous.stop()
    published = publish_artifact_source(
        ctx.mx_client,
        transfer,
        bundle,
        identity,
        ctx.nixl_manager,
        worker_id=ctx.worker_id,
        worker_grpc_server=worker_grpc_server,
        worker_rank=ctx.worker_rank,
        node_rank=ctx.node_rank,
    )
    _published_sources[key] = published
    elapsed = time.perf_counter() - start
    total_size = sum(file.size for file in bundle.manifest.files)
    logger.info(
        "[Worker %s] [TIMING] SGLang artifact publish complete: "
        "name=%s artifact_id=%s mx_source_id=%s size=%.2f MiB elapsed=%.3fs",
        ctx.global_rank,
        transfer.name,
        bundle.artifact_id,
        published.endpoint.mx_source_id,
        total_size / (1024 * 1024),
        elapsed,
    )
    return published


def _artifact_transfer_enabled() -> bool:
    return envs.MX_ARTIFACT_TRANSFER


def _p2p_metadata_enabled_for_artifacts(ctx: LoadContext) -> bool:
    if _is_p2p_metadata_enabled(ctx.mx_client):
        return True
    logger.warning(
        "[Worker %s] MX_ARTIFACT_TRANSFER is enabled but "
        "MX_P2P_METADATA is disabled; skipping SGLang artifact transfer",
        ctx.global_rank,
    )
    return False


def _ensure_nixl_manager(ctx: LoadContext) -> None:
    if ctx.nixl_manager is not None:
        return
    base_port = envs.MX_METADATA_PORT
    try:
        ctx.nixl_manager = _init_nixl_manager(
            ctx.global_rank,
            ctx.device_id,
            "artifact",
            base_port + ctx.device_id,
        )
    except Exception as exc:
        logger.warning(
            "[Worker %s] NIXL initialization failed, skipping SGLang artifacts: %s",
            ctx.global_rank,
            exc,
        )


def _sglang_artifact_transfers(
    ctx: LoadContext,
) -> list[tuple[P2PArtifactTransfer, p2p_pb2.SourceIdentity]]:
    bundle_root = _bundle_root(ctx)
    torch_compile_cache_root = _torch_compile_cache_root()
    triton_cache_root = _common_artifacts._triton_cache_root()
    deep_gemm_cache_root = _deep_gemm_cache_root()
    tilelang_cache_root = _common_artifacts._tilelang_cache_root()
    cute_dsl_cache_root = _common_artifacts._cute_dsl_cache_root()
    flashinfer_cache_root = _common_artifacts._flashinfer_cache_root()
    flashinfer_autotune_cache_root = (
        _common_artifacts._flashinfer_autotune_cache_root()
    )
    return [
        (
            torch_compile_cache_artifact_transfer(
                torch_compile_cache_root,
                torch_compile_cache_root,
                bundle_root / "torch_compile_cache",
            ),
            _artifact_identity(
                ctx,
                p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
            ),
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
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "deep_gemm"


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
    if mx_source_type == p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE:
        return _torch_compile_cache_identity(ctx)
    if mx_source_type == p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE:
        return _triton_cache_identity(ctx)
    if mx_source_type == p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE:
        return _deep_gemm_cache_identity(ctx)
    if mx_source_type == p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE:
        return _tilelang_cache_identity(ctx)
    if mx_source_type == p2p_pb2.MX_SOURCE_TYPE_CUTE_DSL_CACHE:
        return _cute_dsl_cache_identity(ctx)
    if mx_source_type == p2p_pb2.MX_SOURCE_TYPE_FLASHINFER_CACHE:
        return _flashinfer_cache_identity(ctx)
    raise ValueError(f"unknown SGLang artifact source type: {mx_source_type}")


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
        triton_version=_common_artifacts._triton_version(),
        gpu_arch=_common_artifacts._gpu_arch(ctx.device_id),
        compile_config_digest=envs.MX_ARTIFACT_COMPILE_CONFIG_DIGEST,
    )
    _set_extra_if_present(identity, "triton_key", _common_artifacts._triton_key())
    return identity


def _triton_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        model_name=ctx.identity.model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
        cuda_version=torch.version.cuda or "",
        triton_version=_common_artifacts._triton_version(),
        gpu_arch=_common_artifacts._gpu_arch(ctx.device_id),
    )
    _set_extra_if_present(identity, "triton_key", _common_artifacts._triton_key())
    return identity


def _deep_gemm_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE,
        model_name=ctx.identity.model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
        cuda_version=torch.version.cuda or "",
        gpu_arch=_common_artifacts._gpu_arch(ctx.device_id),
    )
    _set_extra_if_present(
        identity,
        "deep_gemm_jit_key",
        _common_artifacts._deep_gemm_jit_key(),
    )
    return identity


def _tilelang_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE,
        model_name=ctx.identity.model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
        cuda_version=torch.version.cuda or "",
        gpu_arch=_common_artifacts._gpu_arch(ctx.device_id),
    )
    _set_extra_if_present(
        identity,
        "tilelang_version",
        _common_artifacts._tilelang_version(),
    )
    return identity


def _cute_dsl_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_CUTE_DSL_CACHE,
        model_name=ctx.identity.model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
        cuda_version=torch.version.cuda or "",
        gpu_arch=_common_artifacts._gpu_arch(ctx.device_id),
    )
    _set_extra_if_present(
        identity,
        "cutlass_dsl_version",
        _common_artifacts._cutlass_dsl_version(),
    )
    return identity


def _flashinfer_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_FLASHINFER_CACHE,
        model_name=ctx.identity.model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "",
        gpu_arch=_common_artifacts._gpu_arch(ctx.device_id),
    )
    _set_extra_if_present(
        identity,
        "flashinfer_version",
        _common_artifacts._flashinfer_version(),
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
    return _common_artifacts._vllm_artifact_ready_fn(
        source_roots,
        _sglang_health_ready,
    )


def _sglang_health_ready() -> bool:
    return _common_artifacts._artifact_health_ready(_sglang_health_url())


def _sglang_health_url() -> str:
    configured = envs.MX_ARTIFACT_READY_URL.strip()
    fallback = (
        _common_artifacts._statefulset_head_health_url(port=30000)
        or _DEFAULT_READY_URL
    )
    if not configured or configured == _DEFAULT_READY_URL:
        return fallback
    if _common_artifacts._is_http_url(configured):
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
