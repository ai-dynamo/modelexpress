# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-agnostic lifecycle helpers for cache artifact transfer."""

from __future__ import annotations

import ipaddress
import logging
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_NB, LOCK_UN, flock
from getpass import getuser
from hashlib import sha256
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Callable, Iterator, TextIO

import torch

from .. import envs
from .. import p2p_pb2
from ..load_strategy.base import _init_nixl_manager, _metadata_publication_configured
from ..load_strategy.context import LoadContext
from ..nixl_transfer import is_nixl_available
from .artifact_transfer import (
    ArtifactCacheRoot,
    P2PArtifactTransfer,
    PublishedArtifactSource,
    publish_artifact_source,
)
from .publisher import PublisherThread
from .publish import _get_worker_server, _is_p2p_metadata_enabled
from .source_id import compute_mx_source_id

logger = logging.getLogger("modelexpress.metadata.artifact_lifecycle")

READY_POLL_SECS = 5
CACHE_SETTLE_SECS = 5

ArtifactEntry = tuple[P2PArtifactTransfer, p2p_pb2.SourceIdentity]
InstallCompleted = Callable[[P2PArtifactTransfer, p2p_pb2.SourceIdentity], None]
_publish_leases: dict[Path, TextIO] = {}


def install_artifacts(
    ctx: LoadContext,
    transfers_factory: Callable[[], list[ArtifactEntry]],
    *,
    engine_label: str,
    on_install_completed: InstallCompleted | None = None,
    log: logging.Logger = logger,
) -> None:
    """Best-effort install of compatible artifacts before model loading."""
    if not _artifact_transfer_enabled():
        return
    if not _p2p_metadata_enabled_for_artifacts(ctx, engine_label, log):
        return
    if not _metadata_publication_configured(ctx):
        log.info(
            "[Worker %s] No MX metadata path configured, skipping %s artifacts",
            ctx.global_rank,
            engine_label,
        )
        return
    if not is_nixl_available():
        log.info(
            "[Worker %s] NIXL not available, skipping %s artifact install",
            ctx.global_rank,
            engine_label,
        )
        return

    _ensure_nixl_manager(ctx, engine_label, log)
    if ctx.nixl_manager is None:
        return

    for transfer, identity in transfers_factory():
        try:
            start = time.perf_counter()
            header = install_artifact_once(
                ctx,
                transfer,
                identity,
                engine_label=engine_label,
                on_install_completed=on_install_completed,
            )
            elapsed = time.perf_counter() - start
            if header is None:
                log.debug(
                    "[Worker %s] %s artifact %s already attempted in this pod",
                    ctx.global_rank,
                    engine_label,
                    transfer.name,
                )
                continue
            log.info(
                "[Worker %s] [TIMING] %s artifact install complete: "
                "name=%s artifact_id=%s mx_source_id=%s size=%.2f MiB elapsed=%.3fs",
                ctx.global_rank,
                engine_label,
                transfer.name,
                header.artifact_id,
                compute_mx_source_id(identity),
                header.total_size / (1024 * 1024),
                elapsed,
            )
        except LookupError:
            # Logged at INFO, not DEBUG: a miss here is the difference between a
            # warm start and a full recompile, and the mx_source_id plus digest
            # are what an operator needs to diff two pods that fail to pair.
            log.info(
                "[Worker %s] No ready %s artifact source for %s "
                "(mx_source_id=%s compile_config_digest=%r); "
                "the engine will rebuild this cache locally",
                ctx.global_rank,
                engine_label,
                transfer.name,
                compute_mx_source_id(identity),
                identity.compile_config_digest,
            )
        except Exception as exc:
            log.warning(
                "[Worker %s] Failed to install %s artifact %s: %s",
                ctx.global_rank,
                engine_label,
                transfer.name,
                exc,
            )


def schedule_artifact_publish(
    ctx: LoadContext,
    transfers_factory: Callable[[], list[ArtifactEntry]],
    *,
    engine_label: str,
    ready_fn_factory: Callable[[tuple[ArtifactCacheRoot, ...]], Callable[[], bool]],
    artifact_publish_fn: Callable[
        [P2PArtifactTransfer, p2p_pb2.SourceIdentity],
        PublishedArtifactSource,
    ],
    scheduled_publishers: dict[tuple[int, int], PublisherThread],
    log: logging.Logger = logger,
) -> None:
    """Schedule readiness-gated publication of local cache artifacts."""
    if not _artifact_transfer_enabled():
        return
    if not _p2p_metadata_enabled_for_artifacts(ctx, engine_label, log):
        return
    if not _metadata_publication_configured(ctx):
        log.info(
            "[Worker %s] No MX metadata path configured, skipping %s artifacts",
            ctx.global_rank,
            engine_label,
        )
        return
    if ctx.nixl_manager is None:
        log.info(
            "[Worker %s] No NIXL manager, skipping %s artifact publish",
            ctx.global_rank,
            engine_label,
        )
        return

    for transfer, identity in transfers_factory():
        marker_path = mark_publish_scheduled(ctx, transfer, identity)
        if marker_path is None:
            continue

        key = (ctx.device_id, transfer.mx_source_type)
        previous = scheduled_publishers.pop(key, None)
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
                artifact_publish_fn(transfer, identity).endpoint.mx_source_id
            ),
            ready_fn=ready_fn_factory(source_roots),
            publish_timeout_secs=envs.MX_ARTIFACT_READY_TIMEOUT_SECS,
            interval_secs=READY_POLL_SECS,
            heartbeat_after_publish=False,
            cleanup_fn=lambda marker_path=marker_path, publisher_ref=publisher_ref: (
                clear_publish_scheduled(publisher_ref[0], marker_path)
            ),
        )
        publisher_ref[0] = publisher
        scheduled_publishers[key] = publisher
        publisher.start()
        log.info(
            "[Worker %s] Scheduled %s artifact publisher: name=%s roots=%s",
            ctx.global_rank,
            engine_label,
            transfer.name,
            [str(root.source_root) for root in source_roots],
        )


def install_artifact_once(
    ctx: LoadContext,
    transfer: P2PArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
    *,
    engine_label: str,
    on_install_completed: InstallCompleted | None = None,
) -> p2p_pb2.GetArtifactManifestHeaderResponse | None:
    """Install one artifact at most once per pod."""
    marker_path = artifact_marker_path(transfer, identity, "install-attempted")
    with artifact_lock(marker_path):
        if marker_path.exists():
            return None
        if ctx.nixl_manager is None:
            raise RuntimeError(
                f"NIXL manager is required for {engine_label} artifact install"
            )
        write_marker(marker_path, "attempted")
        header = transfer.discover_and_transfer(
            ctx.mx_client,
            identity,
            ctx.nixl_manager,
            worker_rank=None,
            node_rank=ctx.node_rank,
            accelerator=ctx.accelerator_backend.name,
        )
        transfer.install(header)
        if on_install_completed is not None:
            on_install_completed(transfer, identity)
        write_marker(marker_path, header.artifact_id)
        return header


def publish_artifact(
    ctx: LoadContext,
    transfer: P2PArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
    *,
    engine_label: str,
    accelerator: str,
    published_sources: dict[tuple[int, int], PublishedArtifactSource],
    log: logging.Logger = logger,
) -> PublishedArtifactSource:
    """Prepare and publish one local artifact source."""
    if ctx.nixl_manager is None:
        raise RuntimeError(
            f"NIXL manager is required for {engine_label} artifact publish"
        )
    worker_grpc_server = _get_worker_server(ctx.device_id)
    if worker_grpc_server is None:
        raise RuntimeError("P2P worker gRPC server is required for artifact publish")

    required_roots = tuple(
        root.source_root for root in transfer.roots if not root.optional
    )
    if not all(has_files(path) for path in required_roots):
        raise LookupError(
            f"Required {engine_label} artifact sources {transfer.name} are empty "
            f"or missing: "
            f"{required_roots}"
        )

    start = time.perf_counter()
    bundle = transfer.prepare_source()
    key = (ctx.device_id, transfer.mx_source_type)
    previous = published_sources.pop(key, None)
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
        accelerator=accelerator,
    )
    published_sources[key] = published
    elapsed = time.perf_counter() - start
    total_size = sum(file.size for file in bundle.manifest.files)
    log.info(
        "[Worker %s] [TIMING] %s artifact publish complete: "
        "name=%s artifact_id=%s mx_source_id=%s size=%.2f MiB elapsed=%.3fs",
        ctx.global_rank,
        engine_label,
        transfer.name,
        bundle.artifact_id,
        published.endpoint.mx_source_id,
        total_size / (1024 * 1024),
        elapsed,
    )
    return published


def artifact_ready_fn(
    source_roots: tuple[ArtifactCacheRoot, ...],
    health_ready_fn: Callable[[], bool],
) -> Callable[[], bool]:
    """Return a readiness check for a stable, healthy artifact source."""
    server_ready = False
    stable_since: float | None = None
    last_signature: tuple[int, int, int] | None = None

    def ready() -> bool:
        nonlocal server_ready, stable_since, last_signature
        if not server_ready:
            if not health_ready_fn():
                stable_since = None
                last_signature = None
                return False
            server_ready = True

        signature = cache_signature(source_roots)
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
        return time.monotonic() - stable_since >= CACHE_SETTLE_SECS

    return ready


def artifact_health_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1.0) as response:
            return 200 <= response.status < 400
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def resolve_health_url(
    configured: str,
    default_url: str,
    head_addr: str | None = None,
) -> str:
    """Resolve the health endpoint this worker should probe.

    A pod-local loopback URL is only correct on the node that serves it: the
    non-head nodes of a multi-node engine run headless and serve no HTTP. So a
    loopback host is rewritten onto the head, keeping the configured port and
    path. A configured non-loopback host is left alone.

    ``head_addr`` is the engine's own distributed-init address, which the
    orchestrator already resolved and torch.distributed already proved
    reachable. It wins whenever the engine reports one, including a loopback
    value, which means the head is this node. ``LWS_LEADER_ADDRESS`` is only
    consulted for engines that expose no address at all.
    """
    url = (configured or "").strip() or default_url
    if not is_http_url(url):
        logger.warning("Invalid MX_ARTIFACT_READY_URL=%r; using %s", configured, default_url)
        url = default_url

    parsed = urllib.parse.urlparse(url)
    if not _is_loopback(parsed.hostname):
        return url

    # The engine's address is authoritative whenever it reports one: a loopback
    # value means the head is this node, so the local URL is already correct.
    # The orchestrator is consulted only when the engine exposes no address.
    head = (head_addr or "").strip() or envs.LWS_LEADER_ADDRESS.strip()
    if not head or _is_loopback(head):
        return url
    if ":" in head and not head.startswith("["):
        head = f"[{head}]"  # bare IPv6 literal
    netloc = head if parsed.port is None else f"{head}:{parsed.port}"
    return parsed._replace(netloc=netloc).geturl()


def _is_loopback(host: str | None) -> bool:
    """True for hosts only reachable inside the pod that serves them."""
    if not host:
        return True
    candidate = host.strip().strip("[]").lower()
    if candidate == "localhost":
        return True
    try:
        address = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def is_http_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def has_files(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(child.is_file() for child in path.rglob("*"))


def cache_signature(
    roots: tuple[ArtifactCacheRoot, ...],
) -> tuple[int, int, int] | None:
    if not all(has_files(root.source_root) for root in roots if not root.optional):
        return None

    count = 0
    total_size = 0
    max_mtime_ns = 0
    try:
        for root in roots:
            path = root.source_root
            if not path.is_dir():
                continue
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


def mark_publish_scheduled(
    ctx: LoadContext,
    transfer: P2PArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
) -> Path | None:
    """Acquire one process-owned artifact publication lease."""
    lease_path = artifact_marker_path(transfer, identity, "publish-scheduled")
    lease_path.parent.mkdir(parents=True, exist_ok=True)
    lease = lease_path.open("a+")
    try:
        flock(lease.fileno(), LOCK_EX | LOCK_NB)
    except BlockingIOError:
        lease.close()
        return None
    _publish_leases[lease_path] = lease
    logger.debug(
        "[Worker %s] Acquired artifact publish lease: name=%s",
        ctx.global_rank,
        transfer.name,
    )
    return lease_path


def clear_publish_scheduled(
    publisher: PublisherThread | None,
    lease_path: Path,
) -> None:
    if publisher is None or publisher.mx_source_id is not None:
        return
    lease = _publish_leases.pop(lease_path, None)
    if lease is not None:
        lease.close()


def artifact_marker_path(
    transfer: P2PArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
    action: str,
) -> Path:
    return artifact_lock_root() / (
        f"{action}-{transfer.name}-"
        f"{artifact_marker_key(transfer, identity, action)}.done"
    )


def artifact_marker_key(
    transfer: P2PArtifactTransfer,
    identity: p2p_pb2.SourceIdentity,
    action: str,
) -> str:
    digest = sha256()
    digest.update(identity.SerializeToString())
    for root in transfer.roots:
        path = root.source_root if action == "publish-scheduled" else root.target_root
        digest.update(str(path.resolve()).encode())
    digest.update(transfer.name.encode())
    return digest.hexdigest()[:16]


def artifact_lock_root() -> Path:
    return Path(tempfile.gettempdir()) / "modelexpress-artifacts" / "locks"


@contextmanager
def artifact_lock(marker_path: Path) -> Iterator[None]:
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = marker_path.with_suffix(".lock")
    with lock_path.open("w") as lock_file:
        flock(lock_file.fileno(), LOCK_EX)
        try:
            yield
        finally:
            flock(lock_file.fileno(), LOCK_UN)


def write_marker(marker_path: Path, value: str) -> None:
    marker_path.write_text(f"{value}\n", encoding="utf-8")


def _artifact_transfer_enabled() -> bool:
    return envs.MX_ARTIFACT_TRANSFER


def _p2p_metadata_enabled_for_artifacts(
    ctx: LoadContext,
    engine_label: str,
    log: logging.Logger,
) -> bool:
    if _is_p2p_metadata_enabled(ctx.mx_client):
        return True
    log.warning(
        "[Worker %s] MX_ARTIFACT_TRANSFER is enabled but MX_P2P_METADATA is disabled; "
        "skipping %s artifact transfer",
        ctx.global_rank,
        engine_label,
    )
    return False


def _ensure_nixl_manager(
    ctx: LoadContext,
    engine_label: str,
    log: logging.Logger,
) -> None:
    if ctx.nixl_manager is not None:
        return
    try:
        ctx.nixl_manager = _init_nixl_manager(
            ctx.global_rank,
            ctx.device_id,
            "artifact",
            envs.MX_METADATA_PORT + ctx.device_id,
        )
    except Exception as exc:
        log.warning(
            "[Worker %s] NIXL initialization failed, skipping %s artifacts: %s",
            ctx.global_rank,
            engine_label,
            exc,
        )


def triton_cache_root() -> Path:
    configured = envs.TRITON_CACHE_DIR
    return Path(configured) if configured else Path.home() / ".triton" / "cache"


def tvm_ffi_cache_root() -> Path:
    configured = envs.TVM_FFI_CACHE_DIR
    return Path(configured) if configured else Path.home() / ".cache" / "tvm-ffi"


def tilelang_cache_root() -> Path:
    configured = envs.TILELANG_CACHE_DIR
    return Path(configured) if configured else Path.home() / ".tilelang" / "cache"


def cute_dsl_cache_root() -> Path:
    configured = envs.CUTE_DSL_CACHE_DIR
    if configured:
        return Path(configured)
    try:
        user = getuser()
    except (KeyError, OSError):
        user = str(os.getuid())
    return Path(tempfile.gettempdir()) / user / "cutlass_python_cache"


def flashinfer_cache_root() -> Path:
    workspace_base = envs.FLASHINFER_WORKSPACE_BASE
    if workspace_base:
        return Path(workspace_base) / ".cache" / "flashinfer"
    return Path.home() / ".cache" / "flashinfer"


def triton_version() -> str:
    try:
        import triton

        version = getattr(triton, "__version__", "")
        return version if isinstance(version, str) else str(version)
    except Exception:
        return ""


def tvm_ffi_version() -> str:
    try:
        import tvm_ffi
    except ModuleNotFoundError as exc:
        if exc.name != "tvm_ffi":
            raise
        return pkg_version("apache-tvm-ffi")

    version = getattr(tvm_ffi, "__version__", None)
    return str(version) if version else pkg_version("apache-tvm-ffi")


def triton_key() -> str:
    try:
        from triton.runtime.cache import triton_key

        key = triton_key()
        return key if isinstance(key, str) else ""
    except Exception:
        return ""


def deep_gemm_jit_key() -> str:
    try:
        from deep_gemm.jit.compiler import get_deep_gemm_version

        return get_deep_gemm_version()
    except Exception:
        return ""


def tilelang_version() -> str:
    try:
        return pkg_version("tilelang")
    except Exception:
        return ""


def cutlass_dsl_version() -> str:
    try:
        return pkg_version("nvidia-cutlass-dsl")
    except Exception:
        return ""


def flashinfer_version() -> str:
    try:
        return pkg_version("flashinfer-python")
    except Exception:
        return ""


def gpu_arch(device_id: int) -> str:
    if not torch.cuda.is_available():
        return ""
    major, minor = torch.cuda.get_device_capability(device_id)
    return f"sm{major}{minor}"
