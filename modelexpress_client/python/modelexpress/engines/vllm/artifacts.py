# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM cache artifact integration for the ModelExpress loader."""

from __future__ import annotations

import logging
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from fcntl import flock, LOCK_EX, LOCK_UN
from getpass import getuser
from hashlib import sha256
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Iterator

import torch

from ... import p2p_pb2
from ...load_strategy.base import (
    _init_nixl_manager,
    _metadata_publication_configured,
)
from ...load_strategy.context import LoadContext
from ...metadata.artifact_transfer import (
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

logger = logging.getLogger("modelexpress.engines.vllm.artifacts")

_ENABLE_ENV = "MX_ARTIFACT_TRANSFER"
_BUNDLE_ROOT_ENV = "MX_ARTIFACT_BUNDLE_ROOT"
_COMPILE_CONFIG_DIGEST_ENV = "MX_ARTIFACT_COMPILE_CONFIG_DIGEST"
_DEEP_GEMM_CACHE_DIR_ENV = "DEEP_GEMM_CACHE_DIR"
_DG_JIT_CACHE_DIR_ENV = "DG_JIT_CACHE_DIR"
_READY_URL_ENV = "MX_ARTIFACT_READY_URL"
_READY_TIMEOUT_ENV = "MX_ARTIFACT_READY_TIMEOUT_SECS"
_DEFAULT_READY_URL = "http://127.0.0.1:8000/health"
_READY_POLL_SECS = 5
_CACHE_SETTLE_SECS = 5
_TRUTHY = {"1", "true", "yes", "on"}

_published_sources: dict[tuple[int, int], PublishedArtifactSource] = {}
_scheduled_publishers: dict[tuple[int, int], PublisherThread] = {}


def install_vllm_cache_artifacts(ctx: LoadContext) -> None:
    """Best-effort install of compatible vLLM cache artifacts before load."""
    if not _artifact_transfer_enabled():
        return
    if not _p2p_metadata_enabled_for_artifacts(ctx):
        return
    if not _metadata_publication_configured(ctx):
        logger.info(
            "[Worker %s] No MX metadata path configured, skipping vLLM artifacts",
            ctx.global_rank,
        )
        return
    if not is_nixl_available():
        logger.info(
            "[Worker %s] NIXL not available, skipping vLLM artifact install",
            ctx.global_rank,
        )
        return

    _ensure_nixl_manager(ctx)
    if ctx.nixl_manager is None:
        return

    for transfer, identity in _vllm_artifact_transfers(ctx):
        try:
            start = time.perf_counter()
            header = _install_vllm_cache_artifact_once(ctx, transfer, identity)
            elapsed = time.perf_counter() - start
            if header is None:
                logger.debug(
                    "[Worker %s] vLLM artifact %s already attempted in this pod",
                    ctx.global_rank,
                    transfer.name,
                )
                continue
            logger.info(
                "[Worker %s] [TIMING] vLLM artifact install complete: "
                "name=%s artifact_id=%s size=%.2f MiB elapsed=%.3fs",
                ctx.global_rank,
                transfer.name,
                header.artifact_id,
                header.total_size / (1024 * 1024),
                elapsed,
            )
        except LookupError:
            logger.debug(
                "[Worker %s] No ready vLLM artifact source for %s",
                ctx.global_rank,
                transfer.name,
            )
        except Exception as exc:
            logger.warning(
                "[Worker %s] Failed to install vLLM artifact %s: %s",
                ctx.global_rank,
                transfer.name,
                exc,
            )


def schedule_vllm_cache_artifact_publish(ctx: LoadContext) -> None:
    """Schedule publication of local artifacts after the vLLM server is ready.

    Artifact readiness is independent of the primary weight source: each cache
    becomes discoverable only after its own publisher succeeds.
    """
    if not _artifact_transfer_enabled():
        return
    if not _p2p_metadata_enabled_for_artifacts(ctx):
        return
    if not _metadata_publication_configured(ctx):
        logger.info(
            "[Worker %s] No MX metadata path configured, skipping vLLM artifacts",
            ctx.global_rank,
        )
        return
    if ctx.nixl_manager is None:
        logger.info(
            "[Worker %s] No NIXL manager, skipping vLLM artifact publish",
            ctx.global_rank,
        )
        return

    for transfer, identity in _vllm_artifact_transfers(ctx):
        marker_path = _mark_vllm_cache_artifact_publish_scheduled(
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

        publisher_ref: list[PublisherThread | None] = [None]
        publisher = PublisherThread(
            mx_client=ctx.mx_client,
            worker_id=_artifact_source_worker_id(ctx),
            worker_rank=ctx.worker_rank,
            nixl_manager=ctx.nixl_manager,
            publish_fn=lambda transfer=transfer, identity=identity: (
                _publish_vllm_cache_artifact(ctx, transfer, identity).endpoint.mx_source_id
            ),
            ready_fn=_vllm_artifact_ready_fn(transfer.source_root),
            publish_timeout_secs=_ready_timeout_secs(),
            interval_secs=_READY_POLL_SECS,
            heartbeat_after_publish=False,
            cleanup_fn=lambda marker_path=marker_path, publisher_ref=publisher_ref: (
                _clear_vllm_cache_artifact_publish_scheduled(
                    publisher_ref[0],
                    marker_path,
                )
            ),
        )
        publisher_ref[0] = publisher
        _scheduled_publishers[key] = publisher
        publisher.start()
        logger.info(
            "[Worker %s] Scheduled vLLM artifact publisher: name=%s root=%s",
            ctx.global_rank,
            transfer.name,
            transfer.source_root,
        )


def _mark_vllm_cache_artifact_publish_scheduled(
    ctx: LoadContext,
    transfer: P2PArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
) -> Path | None:
    # vLLM JIT cache artifacts are pod-scoped, so only one local process should
    # publish each cache type for the pod.
    marker_path = _artifact_marker_path(transfer, identity, "publish-scheduled")
    with _artifact_lock(marker_path):
        if marker_path.exists():
            return None
        _write_marker(marker_path, str(ctx.global_rank))
        return marker_path


def _clear_vllm_cache_artifact_publish_scheduled(
    publisher: PublisherThread | None,
    marker_path: Path,
) -> None:
    if publisher is None or publisher.mx_source_id is not None:
        return
    with _artifact_lock(marker_path):
        marker_path.unlink(missing_ok=True)


def _install_vllm_cache_artifact_once(
    ctx: LoadContext,
    transfer: P2PArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
) -> p2p_pb2.GetArtifactManifestHeaderResponse | None:
    # Cache artifacts are pod-scoped. Mark before transfer so a failure is not
    # retried serially by every local worker.
    marker_path = _artifact_marker_path(transfer, identity, "install-attempted")
    with _artifact_lock(marker_path):
        if marker_path.exists():
            return None
        if ctx.nixl_manager is None:
            raise RuntimeError("NIXL manager is required for vLLM artifact install")
        _write_marker(marker_path, "attempted")
        header = transfer.discover_and_transfer(
            ctx.mx_client,
            identity,
            ctx.nixl_manager,
            worker_rank=None,
        )
        transfer.install(header)
        _write_marker(marker_path, header.artifact_id)
        return header


def _publish_vllm_cache_artifact(
    ctx: LoadContext,
    transfer: P2PArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
) -> PublishedArtifactSource:
    if ctx.nixl_manager is None:
        raise RuntimeError("NIXL manager is required for vLLM artifact publish")
    worker_grpc_server = _get_worker_server(ctx.device_id)
    if worker_grpc_server is None:
        raise RuntimeError("P2P worker gRPC server is required for artifact publish")
    if not _has_files(transfer.source_root):
        raise LookupError(
            f"vLLM artifact source {transfer.name} is empty or missing: "
            f"{transfer.source_root}"
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
        worker_id=_artifact_source_worker_id(ctx),
        worker_grpc_server=worker_grpc_server,
        worker_rank=ctx.worker_rank,
    )
    _published_sources[key] = published
    elapsed = time.perf_counter() - start
    total_size = sum(file.size for file in bundle.manifest.files)
    logger.info(
        "[Worker %s] [TIMING] vLLM artifact publish complete: "
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
    return os.environ.get(_ENABLE_ENV, "").strip().lower() in _TRUTHY


def _artifact_source_worker_id(
    ctx: LoadContext,
) -> str:
    return ctx.worker_id


def _artifact_marker_path(
    transfer: P2PArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
    action: str,
) -> Path:
    return _artifact_lock_root() / (
        f"{action}-{transfer.name}-{_artifact_marker_key(transfer, identity, action)}"
        ".done"
    )


def _artifact_marker_key(
    transfer: P2PArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
    action: str,
) -> str:
    path = transfer.source_root if action == "publish" else transfer.target_root
    digest = sha256()
    digest.update(identity.SerializeToString())
    digest.update(str(path.resolve()).encode())
    digest.update(transfer.name.encode())
    return digest.hexdigest()[:16]


def _artifact_lock_root() -> Path:
    return Path(tempfile.gettempdir()) / "modelexpress-artifacts" / "locks"


@contextmanager
def _artifact_lock(marker_path: Path) -> Iterator[None]:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = marker_path.with_suffix(".lock")
    with lock_path.open("w") as lock_file:
        flock(lock_file.fileno(), LOCK_EX)
        try:
            yield
        finally:
            flock(lock_file.fileno(), LOCK_UN)


def _read_marker(marker_path: Path) -> str:
    try:
        return marker_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def _write_marker(marker_path: Path, value: str) -> None:
    marker_path.write_text(f"{value}\n", encoding="utf-8")


def _p2p_metadata_enabled_for_artifacts(ctx: LoadContext) -> bool:
    if _is_p2p_metadata_enabled(ctx.mx_client):
        return True
    logger.warning(
        "[Worker %s] MX_ARTIFACT_TRANSFER is enabled but "
        "MX_P2P_METADATA is disabled; skipping vLLM artifact transfer",
        ctx.global_rank,
    )
    return False


def _ensure_nixl_manager(ctx: LoadContext) -> None:
    if ctx.nixl_manager is not None:
        return
    base_port = _env_int("MX_METADATA_PORT", 5555)
    try:
        ctx.nixl_manager = _init_nixl_manager(
            ctx.global_rank,
            ctx.device_id,
            "artifact",
            base_port + ctx.device_id,
        )
    except Exception as exc:
        logger.warning(
            "[Worker %s] NIXL initialization failed, skipping vLLM artifacts: %s",
            ctx.global_rank,
            exc,
        )


def _vllm_artifact_transfers(
    ctx: LoadContext,
) -> list[tuple[P2PArtifactTransfer, p2p_pb2.SourceIdentity]]:
    bundle_root = _bundle_root(ctx)
    transfer_specs = [
        (
            "torch_compile_cache",
            torch_compile_cache_artifact_transfer,
            p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
            _torch_compile_cache_root(),
        ),
        (
            "triton_cache",
            triton_cache_artifact_transfer,
            p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
            _triton_cache_root(),
        ),
        (
            "deep_gemm_cache",
            deep_gemm_cache_artifact_transfer,
            p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE,
            _deep_gemm_cache_root(),
        ),
        (
            "tilelang_cache",
            tilelang_cache_artifact_transfer,
            p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE,
            _tilelang_cache_root(),
        ),
        (
            "cute_dsl_cache",
            cute_dsl_cache_artifact_transfer,
            p2p_pb2.MX_SOURCE_TYPE_CUTE_DSL_CACHE,
            _cute_dsl_cache_root(),
        ),
        (
            "flashinfer_cache",
            flashinfer_cache_artifact_transfer,
            p2p_pb2.MX_SOURCE_TYPE_FLASHINFER_CACHE,
            _flashinfer_cache_root(),
        ),
    ]

    return [
        (
            factory(cache_root, cache_root, bundle_root / name),
            _artifact_identity(ctx, mx_source_type),
        )
        for name, factory, mx_source_type, cache_root in transfer_specs
    ]


def _torch_compile_cache_root() -> Path:
    return _vllm_cache_root() / "torch_compile_cache"


def _triton_cache_root() -> Path:
    configured = os.environ.get("TRITON_CACHE_DIR")
    if configured:
        return Path(configured)
    return Path.home() / ".triton" / "cache"


def _deep_gemm_cache_root() -> Path:
    configured = os.environ.get(_DG_JIT_CACHE_DIR_ENV) or os.environ.get(
        _DEEP_GEMM_CACHE_DIR_ENV
    )
    if configured:
        return Path(configured)
    return _vllm_cache_root() / "deep_gemm"


def _tilelang_cache_root() -> Path:
    configured = os.environ.get("TILELANG_CACHE_DIR")
    if configured:
        return Path(configured)
    return Path.home() / ".tilelang" / "cache"


def _cute_dsl_cache_root() -> Path:
    configured = os.environ.get("CUTE_DSL_CACHE_DIR")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / getuser() / "cutlass_python_cache"


def _flashinfer_cache_root() -> Path:
    workspace_base = os.environ.get("FLASHINFER_WORKSPACE_BASE")
    if workspace_base:
        return Path(workspace_base) / ".cache" / "flashinfer"
    return Path.home() / ".cache" / "flashinfer"


def _vllm_cache_root() -> Path:
    configured = os.environ.get("VLLM_CACHE_ROOT")
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "vllm"


def _bundle_root(ctx: LoadContext) -> Path:
    configured = os.environ.get(_BUNDLE_ROOT_ENV)
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
    raise ValueError(f"unknown vLLM artifact source type: {mx_source_type}")


def _torch_compile_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    # TorchInductor cache entries are tied to the model graph and compiler stack.
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
        compile_config_digest=os.environ.get(_COMPILE_CONFIG_DIGEST_ENV, ""),
    )
    _set_extra_if_present(identity, "triton_key", _triton_key())
    return identity


def _triton_cache_identity(ctx: LoadContext) -> p2p_pb2.SourceIdentity:
    # MX requires model_name; the other fields describe Triton runtime compatibility.
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
    # MX requires model_name; the other fields describe DeepGEMM compatibility.
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
    # MX requires model_name; TileLang cache entries carry their own kernel keys.
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


def _has_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(child.is_file() for child in path.rglob("*"))


def _vllm_artifact_ready_fn(source_root: Path):
    stable_since: float | None = None
    last_signature: tuple[int, int, int] | None = None

    def ready() -> bool:
        nonlocal stable_since, last_signature
        if not _vllm_health_ready():
            stable_since = None
            last_signature = None
            return False

        signature = _cache_signature(source_root)
        if signature is None:
            stable_since = None
            last_signature = None
            return False
        if signature != last_signature:
            last_signature = signature
            stable_since = time.monotonic()
            return False
        if stable_since is None:
            stable_since = time.monotonic()
            return False
        return time.monotonic() - stable_since >= _CACHE_SETTLE_SECS

    return ready


def _cache_signature(path: Path) -> tuple[int, int, int] | None:
    if not path.is_dir():
        return None
    count = 0
    total_size = 0
    max_mtime_ns = 0
    try:
        for child in path.rglob("*"):
            if not child.is_file():
                continue
            stat = child.stat()
            count += 1
            total_size += stat.st_size
            max_mtime_ns = max(max_mtime_ns, stat.st_mtime_ns)
    except OSError:
        return None
    if count == 0:
        return None
    return count, total_size, max_mtime_ns


def _vllm_health_ready() -> bool:
    url = _vllm_health_url()
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            return 200 <= response.status < 400
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def _vllm_health_url() -> str:
    configured = os.environ.get(_READY_URL_ENV, "").strip()
    fallback = _statefulset_head_health_url() or _DEFAULT_READY_URL
    if not configured or configured == _DEFAULT_READY_URL:
        return fallback
    if _is_http_url(configured):
        return configured
    logger.warning("Invalid %s=%r; using %s", _READY_URL_ENV, configured, fallback)
    return fallback


def _is_http_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _statefulset_head_health_url() -> str | None:
    pod_name, _, ordinal = os.environ.get("HOSTNAME", "").rpartition("-")
    if not pod_name or not ordinal.isdigit() or ordinal == "0":
        return None
    namespace = os.environ.get("POD_NAMESPACE", "").strip()
    host = f"{pod_name}-0.{pod_name}"
    if namespace:
        host = f"{host}.{namespace}.svc"
    return f"http://{host}:8000/health"


def _ready_timeout_secs() -> int:
    return _env_int(_READY_TIMEOUT_ENV, 1800)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default


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
    try:
        import triton

        version = getattr(triton, "__version__", "")
        return version if isinstance(version, str) else str(version)
    except Exception:
        return ""


def _triton_key() -> str:
    try:
        from triton.runtime.cache import triton_key

        key = triton_key()
        if isinstance(key, str) and key:
            return key
    except Exception:
        return ""
    return ""


def _deep_gemm_jit_key() -> str:
    try:
        from deep_gemm.jit.compiler import get_deep_gemm_version

        return get_deep_gemm_version()
    except Exception:
        return ""


def _tilelang_version() -> str:
    try:
        return pkg_version("tilelang")
    except Exception:
        return ""


def _cutlass_dsl_version() -> str:
    try:
        return pkg_version("nvidia-cutlass-dsl")
    except Exception:
        return ""


def _flashinfer_version() -> str:
    try:
        return pkg_version("flashinfer-python")
    except Exception:
        return ""


def _gpu_arch(device_id: int) -> str:
    if not torch.cuda.is_available():
        return ""
    major, minor = torch.cuda.get_device_capability(device_id)
    return f"sm{major}{minor}"
