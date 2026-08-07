# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM cache artifact configuration for the ModelExpress loader."""

from __future__ import annotations

import logging
import tempfile
from importlib.metadata import version as pkg_version
from pathlib import Path

import torch

from ... import envs
from ... import p2p_pb2
from ...load_strategy.context import LoadContext
from ...metadata import artifact_lifecycle as _artifact_lifecycle
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

logger = logging.getLogger("modelexpress.engines.vllm.artifacts")

_DEFAULT_READY_URL = "http://127.0.0.1:8000/health"
_CACHE_SETTLE_SECS = _artifact_lifecycle.CACHE_SETTLE_SECS

_published_sources: dict[tuple[int, int], PublishedArtifactSource] = {}
_scheduled_publishers: dict[tuple[int, int], PublisherThread] = {}
# torch.compile cache directories created by an install, keyed by device id.
# Values are POSIX-style paths relative to the torch.compile cache root, at every
# depth. Compared against the directory vLLM actually selects once the engine is
# up; see _warn_if_compile_cache_unused.
_installed_compile_cache_dirs: dict[int, frozenset[str]] = {}

# Directory walk depth. vLLM's deepest layout is
# torch_aot_compile/<hash>/rank_<r>_<dp>/<prefix>, so four levels below the root
# covers every published shape without walking an unbounded tree.
_CACHE_SCAN_DEPTH = 4


def install_vllm_cache_artifacts(ctx: LoadContext) -> None:
    """Best-effort install of compatible vLLM cache artifacts before load."""
    # Snapshot only when transfer is on; the default path should not pay for a
    # directory scan it will never consult.
    track = _artifact_transfer_enabled()
    before = _torch_compile_cache_dirs() if track else frozenset()
    _artifact_lifecycle.install_artifacts(
        ctx,
        lambda: _vllm_artifact_transfers(ctx),
        engine_label="vLLM",
        log=logger,
    )
    if not track:
        return
    device_id = getattr(ctx, "device_id", None)
    if device_id is None:
        return
    installed = _torch_compile_cache_dirs() - before
    if installed:
        _installed_compile_cache_dirs[device_id] = installed
    else:
        # A later load that installs nothing must not be judged against the
        # previous load's directories. Weight refit and resume_serving both
        # re-enter this path, so leaving a stale entry would compare the engine's
        # current cache_dir against a set it never had a chance to match.
        _installed_compile_cache_dirs.pop(device_id, None)


def _torch_compile_cache_dirs() -> frozenset[str]:
    """Directories under the torch.compile cache root, as relative POSIX paths.

    Recording every depth rather than only immediate children is what makes the
    AOT layout distinguishable: ``torch_aot_compile`` is a shared container, so a
    name-level snapshot cannot tell ``torch_aot_compile/<installed-hash>`` from
    ``torch_aot_compile/<other-hash>``.
    """
    try:
        root = _torch_compile_cache_root()
    except Exception:  # noqa: BLE001 - env lookup must not break loading
        return frozenset()
    out: set[str] = set()

    def walk(directory: Path, prefix: str, depth: int) -> None:
        if depth > _CACHE_SCAN_DEPTH:
            return
        try:
            entries = list(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir():
                continue
            rel = f"{prefix}/{entry.name}" if prefix else entry.name
            out.add(rel)
            walk(entry, rel, depth + 1)

    walk(root, "", 1)
    return frozenset(out)


def _selected_dir_was_installed(cache_dir: str, installed: frozenset[str]) -> bool:
    """Whether vLLM's chosen directory is one this pod installed.

    Matching is on the exact relative path, never on an ancestor. Ancestor
    matching is what made the AOT layout report a false hit: an install creates
    the shared ``torch_aot_compile`` container alongside its hash directory, so
    any ancestor test would accept ``torch_aot_compile/<some-other-hash>``.

    The tradeoff runs the safe way. If vLLM creates a subdirectory the install
    did not (a rank directory for a rank the source never published, say), this
    returns False and the caller warns about a cache that was in fact partly
    reused. A spurious warning gets investigated; a spurious "reused
    successfully" hides exactly the waste this check exists to surface.
    """
    try:
        root = _torch_compile_cache_root()
        rel = Path(cache_dir).resolve().relative_to(root.resolve()).as_posix()
    except Exception:  # noqa: BLE001 - unresolvable path is simply not a hit
        return False
    return rel in installed


def _warn_if_compile_cache_unused(ctx: LoadContext) -> None:
    """Report whether vLLM selected a torch.compile cache we installed.

    ModelExpress cannot predict vLLM's cache directory at load time: the key
    mixes a code hash derived from ``compilation_config.traced_files``, which is
    populated only while Dynamo traces and cleared immediately afterwards. The
    directory is therefore only observable after the engine has compiled, which
    is why this runs from the publisher path rather than the loader.

    A mismatch means the installed bundle is inert - vLLM rebuilt its cache. It
    is not a correctness problem, but it is silent waste worth surfacing.
    """
    installed = _installed_compile_cache_dirs.get(ctx.device_id)
    if not installed:
        return
    adapter = getattr(ctx, "adapter", None)
    vllm_config = getattr(adapter, "vllm_config", None)
    cache_dir = getattr(
        getattr(vllm_config, "compilation_config", None), "cache_dir", ""
    )
    if not cache_dir:
        # enforce_eager or compilation disabled: nothing selected a cache dir.
        logger.debug(
            "[Worker %s] vLLM reported no torch.compile cache directory; "
            "skipping artifact effectiveness check",
            ctx.global_rank,
        )
        return
    if _selected_dir_was_installed(cache_dir, installed):
        logger.info(
            "[Worker %s] vLLM selected torch.compile cache directory %s, "
            "which ModelExpress installed",
            ctx.global_rank,
            cache_dir,
        )
        return
    logger.warning(
        "[Worker %s] ModelExpress installed torch.compile cache directory/ies %s "
        "but vLLM selected %s, so the transferred cache was not reused and the "
        "engine recompiled. This happens when the source worker's compile "
        "configuration differs from this one (for example max_num_batched_tokens "
        "or max_model_len). Set MX_ARTIFACT_COMPILE_CONFIG_DIGEST to a distinct "
        "value per compile configuration so each group discovers only its own "
        "cache.",
        ctx.global_rank,
        sorted(installed),
        cache_dir,
    )


def schedule_vllm_cache_artifact_publish(ctx: LoadContext) -> None:
    """Schedule publication of local vLLM artifacts after server readiness."""
    _artifact_lifecycle.schedule_artifact_publish(
        ctx,
        lambda: _vllm_artifact_transfers(ctx),
        engine_label="vLLM",
        ready_fn_factory=lambda roots: _vllm_artifact_ready_fn(roots, ctx=ctx),
        artifact_publish_fn=lambda transfer, identity: (
            _publish_vllm_cache_artifact(ctx, transfer, identity)
        ),
        scheduled_publishers=_scheduled_publishers,
        log=logger,
    )


def _install_vllm_cache_artifact_once(
    ctx: LoadContext,
    transfer: P2PArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
):
    """Compatibility wrapper for the shared install-once operation."""
    return _artifact_lifecycle.install_artifact_once(
        ctx,
        transfer,
        identity,
        engine_label="vLLM",
    )


def _publish_vllm_cache_artifact(
    ctx: LoadContext,
    transfer: P2PArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
):
    """Compatibility wrapper for the shared source publication operation."""
    if transfer.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE:
        # Runs on the publisher thread, which is gated on engine readiness, so
        # compilation has finished and vLLM's cache_dir is populated. Never let
        # a diagnostic break publication.
        try:
            _warn_if_compile_cache_unused(ctx)
        except Exception as exc:  # noqa: BLE001 - diagnostic must not propagate
            logger.debug(
                "[Worker %s] torch.compile cache effectiveness check failed: %s",
                ctx.global_rank,
                exc,
            )
    return _artifact_lifecycle.publish_artifact(
        ctx,
        transfer,
        identity,
        engine_label="vLLM",
        accelerator=ctx.accelerator_backend.name,
        published_sources=_published_sources,
        log=logger,
    )


def _artifact_transfer_enabled() -> bool:
    return envs.MX_ARTIFACT_TRANSFER


def _vllm_artifact_transfers(
    ctx: LoadContext,
) -> list[tuple[P2PArtifactTransfer, p2p_pb2.SourceIdentity]]:
    bundle_root = _bundle_root(ctx)
    torch_compile_cache_root = _torch_compile_cache_root()
    triton_cache_root = _triton_cache_root()
    deep_gemm_cache_root = _deep_gemm_cache_root()
    tilelang_cache_root = _tilelang_cache_root()
    cute_dsl_cache_root = _cute_dsl_cache_root()
    flashinfer_cache_root = _flashinfer_cache_root()
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
    return _vllm_cache_root() / "torch_compile_cache"


def _triton_cache_root() -> Path:
    return _artifact_lifecycle.triton_cache_root()


def _deep_gemm_cache_root() -> Path:
    configured = envs.DG_JIT_CACHE_DIR or envs.DEEP_GEMM_CACHE_DIR
    return Path(configured) if configured else _vllm_cache_root() / "deep_gemm"


def _tilelang_cache_root() -> Path:
    return _artifact_lifecycle.tilelang_cache_root()


def _cute_dsl_cache_root() -> Path:
    return _artifact_lifecycle.cute_dsl_cache_root()


def _flashinfer_cache_root() -> Path:
    return _artifact_lifecycle.flashinfer_cache_root()


def _flashinfer_autotune_cache_root() -> Path:
    configured = envs.VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR
    return (
        Path(configured)
        if configured
        else _vllm_cache_root() / "flashinfer_autotune_cache"
    )


def _vllm_cache_root() -> Path:
    configured = envs.VLLM_CACHE_ROOT
    return Path(configured) if configured else Path.home() / ".cache" / "vllm"


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
            f"unknown vLLM artifact source type: {mx_source_type}"
        )
    return builder(ctx)


def _torch_compile_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        model_name=ctx.identity.model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        tensor_parallel_size=ctx.identity.tensor_parallel_size,
        pipeline_parallel_size=ctx.identity.pipeline_parallel_size,
        expert_parallel_size=ctx.identity.expert_parallel_size,
        dtype=ctx.identity.dtype,
        quantization=ctx.identity.quantization,
        revision=ctx.identity.revision,
        backend_framework_version=_vllm_version(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "",
        triton_version=_triton_version(),
        gpu_arch=_gpu_arch(ctx.device_id),
        compile_config_digest=envs.MX_ARTIFACT_COMPILE_CONFIG_DIGEST,
    )
    _set_extra_if_present(identity, "triton_key", _triton_key())
    return identity


def _triton_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        model_name=ctx.identity.model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        cuda_version=torch.version.cuda or "",
        triton_version=_triton_version(),
        gpu_arch=_gpu_arch(ctx.device_id),
    )
    _set_extra_if_present(identity, "triton_key", _triton_key())
    return identity


def _deep_gemm_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE,
        model_name=ctx.identity.model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        cuda_version=torch.version.cuda or "",
        gpu_arch=_gpu_arch(ctx.device_id),
    )
    _set_extra_if_present(identity, "deep_gemm_jit_key", _deep_gemm_jit_key())
    return identity


def _tilelang_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE,
        model_name=ctx.identity.model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        cuda_version=torch.version.cuda or "",
        gpu_arch=_gpu_arch(ctx.device_id),
    )
    _set_extra_if_present(identity, "tilelang_version", _tilelang_version())
    return identity


def _cute_dsl_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_CUTE_DSL_CACHE,
        model_name=ctx.identity.model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        cuda_version=torch.version.cuda or "",
        gpu_arch=_gpu_arch(ctx.device_id),
    )
    _set_extra_if_present(identity, "cutlass_dsl_version", _cutlass_dsl_version())
    return identity


def _flashinfer_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_FLASHINFER_CACHE,
        model_name=ctx.identity.model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda or "",
        gpu_arch=_gpu_arch(ctx.device_id),
    )
    _set_extra_if_present(identity, "flashinfer_version", _flashinfer_version())
    return identity


def _set_extra_if_present(
    identity: p2p_pb2.SourceIdentity,
    key: str,
    value: str,
) -> None:
    if value:
        identity.extra_parameters[key] = value


def _vllm_artifact_ready_fn(
    source_roots: tuple[ArtifactCacheRoot, ...],
    health_ready_fn=None,
    ctx: LoadContext | None = None,
):
    if health_ready_fn is None:
        # Resolve the engine hook at call time so existing tests and diagnostics
        # can replace _vllm_health_ready after constructing the readiness check.
        health_ready_fn = lambda: _vllm_health_ready(ctx)
    return _artifact_lifecycle.artifact_ready_fn(
        source_roots,
        health_ready_fn,
    )


def _vllm_health_ready(ctx: LoadContext | None = None) -> bool:
    return _artifact_lifecycle.artifact_health_ready(_vllm_health_url(ctx))


def _vllm_health_url(ctx: LoadContext | None = None) -> str:
    return _artifact_lifecycle.resolve_health_url(
        envs.MX_ARTIFACT_READY_URL,
        _DEFAULT_READY_URL,
        getattr(ctx, "head_addr", None),
    )


def _is_http_url(url: str) -> bool:
    return _artifact_lifecycle.is_http_url(url)


def _has_files(path: Path) -> bool:
    return _artifact_lifecycle.has_files(path)


def _vllm_version() -> str:
    try:
        import vllm

        version = getattr(vllm, "__version__", "")
        if isinstance(version, str) and version:
            return version
    except Exception:
        pass
    try:
        return pkg_version("vllm")
    except Exception:
        return ""


def _triton_version() -> str:
    return _artifact_lifecycle.triton_version()


def _triton_key() -> str:
    return _artifact_lifecycle.triton_key()


def _deep_gemm_jit_key() -> str:
    return _artifact_lifecycle.deep_gemm_jit_key()


def _tilelang_version() -> str:
    return _artifact_lifecycle.tilelang_version()


def _cutlass_dsl_version() -> str:
    return _artifact_lifecycle.cutlass_dsl_version()


def _flashinfer_version() -> str:
    return _artifact_lifecycle.flashinfer_version()


def _gpu_arch(device_id: int) -> str:
    return _artifact_lifecycle.gpu_arch(device_id)
