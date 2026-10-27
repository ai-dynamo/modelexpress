# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""P2P metadata sync plugin for SGLang."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.metadata import version as pkg_version
import io
import json
import logging
import os
from pathlib import Path
import tarfile
import tempfile
import threading
from typing import Any
from urllib.parse import quote, unquote, urlparse, urlunparse
import urllib.request
import uuid

from ... import envs, p2p_pb2
from ...metadata.client_factory import create_metadata_client
from ...metadata.publish import _get_worker_host
from ...metadata.publisher import PublisherThread

logger = logging.getLogger("modelexpress.engines.sglang.p2p_metadata_sync")

METADATA_EXTRA_PARAM_KEY = "sglang_p2p_metadata_sync"
METADATA_EXTRA_PARAM_VAL = "1"

WEIGHT_SUFFIXES: tuple[str, ...] = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".h5",
    ".msgpack",
    ".gguf",
    ".ckpt.index",
    ".iop",
    ".gas",
)

_server_lock = threading.Lock()
_server_started = False
_http_server: HTTPServer | None = None
_served_paths: dict[str, str] = {}
_default_model_path: str | None = None
_published_paths: set[str] = set()
_publisher_threads: dict[str, PublisherThread] = {}
_worker_id: str | None = None


def is_weight_file(filename: str) -> bool:
    """Return whether ``filename`` is a model weight shard to exclude from metadata."""
    lower = filename.lower()
    return lower.endswith(WEIGHT_SUFFIXES)


def build_metadata_tar_bytes(model_path: str | os.PathLike[str]) -> bytes:
    """Package all non-weight files under ``model_path`` into a gzipped tar archive."""
    root = Path(model_path)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for current_root, dirs, files in os.walk(root):
            dirs[:] = sorted(
                d for d in dirs if d != ".git" and not d.startswith(".modelexpress")
            )
            for fname in sorted(files):
                if is_weight_file(fname) or fname.startswith(".modelexpress"):
                    continue
                full_path = os.path.join(current_root, fname)
                if not os.path.isfile(full_path) or os.path.islink(full_path):
                    continue
                rel_path = os.path.relpath(full_path, root)
                tar.add(full_path, arcname=rel_path)
    return buf.getvalue()


def extract_metadata_tar_bytes(
    data: bytes, target_dir: str | os.PathLike[str]
) -> None:
    """Safely extract a gzipped metadata tar archive into ``target_dir``."""
    dest = Path(target_dir)
    dest.mkdir(parents=True, exist_ok=True)
    resolved_dest = dest.resolve()

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        try:
            tar.extractall(path=dest, filter="data")
        except TypeError:
            # Fallback for Python < 3.12 where filter="data" is not supported.
            safe_members = []
            for member in tar.getmembers():
                if not member.isfile() and not member.isdir():
                    continue
                member_path = (resolved_dest / member.name).resolve()
                if not member_path.is_relative_to(resolved_dest):
                    raise ValueError(f"Unsafe tar member path: {member.name!r}")
                safe_members.append(member)
            tar.extractall(path=dest, members=safe_members)


def _resolve_requested_model_path(request_path: str) -> str | None:
    """Map an HTTP request path to a registered local model directory."""
    parsed = urlparse(request_path)
    clean_path = unquote(parsed.path)
    if clean_path == "/metadata.tar":
        if parsed.query.startswith("path="):
            queried = unquote(parsed.query.split("=", 1)[1])
            if queried in _served_paths:
                return _served_paths[queried]
        return _default_model_path
    return None


def _make_handler():
    class _MetaHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            target_path = _resolve_requested_model_path(self.path)
            if target_path is None or not os.path.isdir(target_path):
                self.send_response(404)
                self.end_headers()
                return

            payload = build_metadata_tar_bytes(target_path)
            self.send_response(200)
            self.send_header("Content-Type", "application/gzip")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            logger.debug(format, *args)

    return _MetaHandler


def _get_worker_id() -> str:
    global _worker_id
    if _worker_id is None:
        _worker_id = f"meta-{uuid.uuid4().hex[:8]}"
    return _worker_id


def _extract_mx_server_url(server_args: Any = None) -> str | None:
    """Resolve explicit ModelExpress server URL from ``server_args`` or env vars."""
    if server_args is not None:
        raw_cfg = getattr(server_args, "modelexpress_config", None)
        if isinstance(raw_cfg, dict):
            url = raw_cfg.get("url")
            if isinstance(url, str) and url.strip():
                return url.strip()
        elif isinstance(raw_cfg, str) and raw_cfg.strip():
            try:
                parsed = json.loads(raw_cfg)
                if isinstance(parsed, dict):
                    url = parsed.get("url")
                    if isinstance(url, str) and url.strip():
                        return url.strip()
            except Exception:
                pass
    return envs.MODEL_EXPRESS_URL or envs.MX_SERVER_ADDRESS


def _is_p2p_discovery_configured(server_url: str | None = None) -> bool:
    """Return True if a ModelExpress coordinator or k8s-service backend is configured."""
    if server_url or envs.MODEL_EXPRESS_URL or envs.MX_SERVER_ADDRESS:
        return True
    return envs.MX_METADATA_BACKEND == "k8s-service"


def _build_metadata_source_identity(model_path: str) -> p2p_pb2.SourceIdentity:
    """Build the ``SourceIdentity`` used to publish/discover SGLang metadata sources."""
    try:
        mx_version = pkg_version("modelexpress")
    except Exception:
        mx_version = "0.0.0"

    normalized = os.path.abspath(model_path)
    return p2p_pb2.SourceIdentity(
        mx_version=mx_version,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name=normalized,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_SGLANG,
        extra_parameters={METADATA_EXTRA_PARAM_KEY: METADATA_EXTRA_PARAM_VAL},
        revision=envs.MX_MODEL_REVISION,
    )


def _extract_host_from_worker(worker: p2p_pb2.WorkerMetadata) -> str | None:
    """Extract a routable host or ``http(s)://host:port`` from ``WorkerMetadata``."""
    for raw in (
        worker.metadata_endpoint,
        worker.worker_grpc_endpoint,
        worker.transfer_engine_session_id,
    ):
        endpoint = (raw or "").strip()
        if not endpoint:
            continue
        if endpoint.startswith(("http://", "https://")):
            return endpoint
        parsed = urlparse(f"http://{endpoint}")
        if parsed.hostname:
            return parsed.hostname
    return None


def _worker_to_meta_source_url(worker: p2p_pb2.WorkerMetadata) -> str | None:
    """Convert a discovered ``WorkerMetadata`` record into an HTTP metadata URL."""
    endpoint = (worker.metadata_endpoint or "").strip()
    if endpoint.startswith(("http://", "https://")):
        return endpoint
    host = _extract_host_from_worker(worker)
    if not host:
        return None
    if host.startswith(("http://", "https://")):
        return host
    return f"http://{host}:{envs.SGLANG_META_PORT}"


def publish_meta_source(
    model_path: str,
    port: int,
    server_url: str | None = None,
) -> bool:
    """Publish this instance's metadata HTTP server to ModelExpress P2P service."""
    if not _is_p2p_discovery_configured(server_url):
        return False

    normalized = os.path.abspath(model_path)
    if normalized in _published_paths:
        return True

    try:
        host = _get_worker_host()
    except Exception:
        host = "127.0.0.1"

    endpoint = f"http://{host}:{port}"
    identity = _build_metadata_source_identity(normalized)
    worker = p2p_pb2.WorkerMetadata(
        worker_rank=0,
        metadata_endpoint=endpoint,
        worker_grpc_endpoint=f"{host}:{port}",
        status=p2p_pb2.SOURCE_STATUS_READY,
    )
    worker_id = _get_worker_id()

    try:
        mx_client = create_metadata_client(worker_rank=0, server_url=server_url)
    except Exception as exc:
        logger.debug("[MetaP2P] Failed to create MX metadata client: %s", exc)
        return False

    def publish_fn() -> str:
        mx_source_id = mx_client.publish_metadata(identity, worker, worker_id)
        logger.info(
            "[MetaP2P] Published metadata source for %s at %s (mx_source_id=%s)",
            normalized,
            endpoint,
            mx_source_id,
        )
        return mx_source_id

    try:
        mx_source_id = publish_fn()
        mx_client.update_status(
            mx_source_id=mx_source_id,
            worker_id=worker_id,
            worker_rank=0,
            status=p2p_pb2.SOURCE_STATUS_READY,
        )
    except Exception as exc:
        logger.debug("[MetaP2P] Initial metadata publish deferred to background: %s", exc)
        mx_source_id = None

    try:
        publisher = PublisherThread(
            mx_client=mx_client,
            mx_source_id=mx_source_id,
            worker_id=worker_id,
            worker_rank=0,
            nixl_manager=None,
            publish_fn=publish_fn,
        )
        publisher.start()
        _publisher_threads[normalized] = publisher
        _published_paths.add(normalized)
        return True
    except Exception as exc:
        logger.debug("[MetaP2P] Failed to start metadata publisher: %s", exc)
        return False


def discover_metadata_source_urls(
    model_path: str,
    server_url: str | None = None,
) -> list[str]:
    """Discover candidate metadata HTTP URLs via ModelExpress P2P source discovery."""
    if not _is_p2p_discovery_configured(server_url):
        return []

    try:
        mx_client = create_metadata_client(worker_rank=0, server_url=server_url)
    except Exception as exc:
        logger.debug("[MetaP2P] Failed to create MX metadata client: %s", exc)
        return []

    normalized = os.path.abspath(model_path)
    identities: list[p2p_pb2.SourceIdentity | None] = [
        _build_metadata_source_identity(normalized),
        None,
    ]
    urls: list[str] = []
    seen_urls: set[str] = set()
    local_worker_id = _get_worker_id()

    try:
        for identity in identities:
            try:
                resp = mx_client.list_sources(
                    identity=identity,
                    status_filter=p2p_pb2.SOURCE_STATUS_READY,
                )
            except Exception as exc:
                logger.debug("[MetaP2P] list_sources failed: %s", exc)
                continue

            if not resp or not resp.instances:
                continue

            # Prefer sources matching model_path first, then any ready rank-0 source
            matching = [
                inst
                for inst in resp.instances
                if inst.worker_id != local_worker_id
                and (
                    inst.model_name in (model_path, normalized)
                    or identity is not None
                )
            ]
            fallback = [
                inst
                for inst in resp.instances
                if inst.worker_id != local_worker_id and inst not in matching
            ]

            for inst in [*matching, *fallback]:
                try:
                    meta_resp = mx_client.get_metadata(
                        mx_source_id=inst.mx_source_id,
                        worker_id=inst.worker_id,
                    )
                except Exception as exc:
                    logger.debug(
                        "[MetaP2P] get_metadata failed for worker %s: %s",
                        inst.worker_id,
                        exc,
                    )
                    continue

                if not meta_resp or not meta_resp.found:
                    continue
                url = _worker_to_meta_source_url(meta_resp.worker)
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    urls.append(url)

            if urls:
                break
    finally:
        try:
            mx_client.close()
        except Exception:
            pass

    return urls


class _MetaHTTPServer(HTTPServer):
    """HTTPServer that skips reverse-DNS lookup on bind."""

    def server_bind(self) -> None:
        import socketserver

        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)


def start_meta_server(
    model_path: str,
    port: int | None = None,
    server_url: str | None = None,
) -> HTTPServer | None:
    """Start the background metadata HTTP server on this instance if not running."""
    global _server_started, _http_server, _default_model_path

    normalized = os.path.abspath(model_path)
    with _server_lock:
        _served_paths[model_path] = normalized
        _served_paths[normalized] = normalized
        if _default_model_path is None:
            _default_model_path = normalized

        if _server_started:
            if _http_server is not None:
                publish_meta_source(
                    normalized, _http_server.server_port, server_url=server_url
                )
            return _http_server

        listen_port = envs.SGLANG_META_PORT if port is None else port
        try:
            server = _MetaHTTPServer(("0.0.0.0", listen_port), _make_handler())
        except OSError:
            # Child processes (Scheduler / TokenizerWorker) or co-located ranks
            # may see the port already bound by the main process.
            _server_started = True
            return None

        thread = threading.Thread(
            target=server.serve_forever,
            name="sglang-meta-p2p-server",
            daemon=True,
        )
        thread.start()
        _http_server = server
        _server_started = True
        logger.info(
            "[MetaP2P] Pod serving metadata from %s on port %d",
            normalized,
            server.server_port,
        )
        publish_meta_source(
            normalized, server.server_port, server_url=server_url
        )
        return server


def _build_metadata_url(source_url: str, model_path: str | None = None) -> str:
    """Construct the full metadata download URL from a peer metadata endpoint."""
    base = source_url.strip()
    if "://" not in base:
        base = f"http://{base}"
    parsed = urlparse(base)
    path = parsed.path.rstrip("/")
    if not path.endswith("/metadata.tar"):
        path = f"{path}/metadata.tar"
    query = parsed.query
    if model_path and not query:
        query = f"path={quote(model_path, safe='/')}"
    return urlunparse(parsed._replace(path=path, query=query))


def fetch_metadata_from_source(
    model_path: str,
    source_url: str,
    timeout: float = 30.0,
) -> None:
    """Download and extract non-weight metadata files from ``source_url`` into ``model_path``."""
    url = _build_metadata_url(source_url, model_path)
    logger.info("[MetaP2P] Target Pod syncing metadata from %s -> %s", url, model_path)

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as resp:
        data = resp.read()

    target_dir = Path(model_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    # Extract via a staging directory on the same filesystem so concurrent local
    # processes do not observe a half-written config.json.
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".meta-p2p-staging-", dir=str(target_dir.parent))
    )
    try:
        extract_metadata_tar_bytes(data, staging_dir)
        for root, dirs, files in os.walk(staging_dir):
            rel_root = Path(root).relative_to(staging_dir)
            dest_root = target_dir / rel_root
            dest_root.mkdir(parents=True, exist_ok=True)
            for d in dirs:
                (dest_root / d).mkdir(parents=True, exist_ok=True)
            for fname in files:
                src_file = Path(root) / fname
                dst_file = dest_root / fname
                os.replace(src_file, dst_file)
    finally:
        for root, dirs, files in os.walk(staging_dir, topdown=False):
            for fname in files:
                try:
                    os.unlink(os.path.join(root, fname))
                except OSError:
                    pass
            for d in dirs:
                try:
                    os.rmdir(os.path.join(root, d))
                except OSError:
                    pass
        try:
            os.rmdir(staging_dir)
        except OSError:
            pass

    logger.info("[MetaP2P] Metadata sync complete for %s", model_path)


def sync_or_serve_metadata(
    path: str | None,
    server_url: str | None = None,
) -> None:
    """Dual-role adaptive entrypoint for a single model/tokenizer/draft path.

    - Role 1 (Source Pod): If ``<path>/config.json`` or ``<path>/tokenizer_config.json``
      already exists locally, serve non-weight files over HTTP on ``SGLANG_META_PORT``
      and publish this instance to the ModelExpress P2P metadata service.
    - Role 2 (Target Pod): If the directory does not have metadata locally,
      discover a ready Source Pod via ModelExpress P2P, pull and extract metadata,
      and then automatically start ``start_meta_server(path)`` so this new instance
      also serves metadata to subsequent pods.
    """
    if not path or path.lower() in ("none", "dummy"):
        return
    if "://" in path:
        return

    config_file = os.path.join(path, "config.json")
    tokenizer_config_file = os.path.join(path, "tokenizer_config.json")
    has_local_metadata = os.path.exists(config_file) or os.path.exists(
        tokenizer_config_file
    )

    if has_local_metadata:
        start_meta_server(path, server_url=server_url)
        return

    candidate_urls = discover_metadata_source_urls(path, server_url=server_url)
    for source_url in candidate_urls:
        try:
            fetch_metadata_from_source(path, source_url)
            start_meta_server(path, server_url=server_url)
            return
        except Exception as exc:
            logger.warning(
                "[MetaP2P] Failed to sync metadata for %s from %s: %s",
                path,
                source_url,
                exc,
            )


def sync_server_args_metadata(server_args: Any) -> None:
    """Sync or serve metadata for all model paths on ``server_args``."""
    server_url = _extract_mx_server_url(server_args)
    seen: set[str] = set()
    for attr in ("model_path", "tokenizer_path", "speculative_draft_model_path"):
        path = getattr(server_args, attr, None)
        if isinstance(path, str) and path and path not in seen:
            seen.add(path)
            sync_or_serve_metadata(path, server_url=server_url)


def before_server_args_post_init(
    self: Any, *args: Any, **kwargs: Any
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """SGLang ``HookType.BEFORE`` hook for ``ServerArgs.__post_init__``."""
    sync_server_args_metadata(self)
    return (self, *args), kwargs


def reset_state_for_test() -> None:
    """Shut down any active test HTTP server and clear module state."""
    global _server_started, _http_server, _default_model_path, _worker_id
    import atexit

    with _server_lock:
        for publisher in list(_publisher_threads.values()):
            try:
                atexit.unregister(publisher._on_exit)
                publisher.stop()
            except Exception:
                pass
        _publisher_threads.clear()
        _published_paths.clear()
        if _http_server is not None:
            try:
                _http_server.shutdown()
                _http_server.server_close()
            except Exception:
                pass
            _http_server = None
        _server_started = False
        _served_paths.clear()
        _default_model_path = None
        _worker_id = None
