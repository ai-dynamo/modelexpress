# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Peer-direct metadata client for ModelExpress.

Drop-in replacement for MxClient that discovers peers via a substrate
probe (mDNS today; K8s, SLURM, DNS, and static lists later) and talks to
those peers' WorkerService gRPC endpoints directly for the metadata
protocol. There is no central coordinator in the loop.

Peers act as their own source of truth: each running worker advertises
its presence on the substrate and answers ListWorkerSources /
GetWorkerMetadata / GetTensorManifest RPCs from its own
WorkerGrpcServer. The client caches peer -> sources mappings locally so
lookups don't hit the network on every call.

The PeerDirectMetadataClient is duck-typed to match MxClient /
DhtMetadataClient exactly, so the vLLM loader can swap backends
transparently.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import grpc

from . import p2p_pb2
from .worker_server import (
    WorkerGrpcServer,
    fetch_worker_metadata,
    fetch_worker_sources,
)

logger = logging.getLogger("modelexpress.peer_direct_client")


# ---------------------------------------------------------------------------
# Source identity hashing (must match Rust compute_mx_source_id and
# DhtMetadataClient's _compute_mx_source_id exactly)
# ---------------------------------------------------------------------------


def _compute_mx_source_id(identity: "p2p_pb2.SourceIdentity") -> str:
    """Compute the 16-char hex source ID from a SourceIdentity proto.

    Must produce identical output to the Rust compute_mx_source_id function
    and to DhtMetadataClient's copy of this helper.
    """
    extra = {}
    for k, v in sorted(identity.extra_parameters.items()):
        extra[k.lower()] = v.lower()

    canonical = json.dumps({
        "backend_framework": identity.backend_framework,
        "dtype": identity.dtype.lower(),
        "expert_parallel_size": identity.expert_parallel_size,
        "extra_parameters": extra,
        "model_name": identity.model_name.lower(),
        "mx_source_type": identity.mx_source_type,
        "mx_version": identity.mx_version.lower(),
        "pipeline_parallel_size": identity.pipeline_parallel_size,
        "quantization": identity.quantization.lower(),
        "tensor_parallel_size": identity.tensor_parallel_size,
    }, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------------------
# Duck-typed response objects (mirror DhtMetadataClient so the loader code
# only needs to know one shape).
# ---------------------------------------------------------------------------


class _FakeGetMetadataResponse:
    """Duck-typed stand-in for p2p_pb2.GetMetadataResponse."""

    def __init__(self, found: bool, worker=None, mx_source_id: str = "", worker_id: str = ""):
        self.found = found
        self.worker = worker
        self.mx_source_id = mx_source_id
        self.worker_id = worker_id


class _FakeListSourcesResponse:
    """Duck-typed stand-in for p2p_pb2.ListSourcesResponse."""

    def __init__(self, instances: list):
        self.instances = instances


# ---------------------------------------------------------------------------
# Peer cache entry
# ---------------------------------------------------------------------------


@dataclass
class _PeerEntry:
    """A discovered peer, keyed by mDNS instance label (or a synthetic
    label in static mode). Populated by the enumerator thread after each
    successful ListWorkerSources."""

    endpoint: str
    txt: dict[str, str] = field(default_factory=dict)
    last_seen: float = 0.0
    sources: list[p2p_pb2.SourceInstanceRef] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Substrate sentinel values
# ---------------------------------------------------------------------------

_SHUTDOWN_SENTINEL: Any = object()
_METADATA_CACHE_TTL_SECONDS = 5.0


class PeerDirectMetadataClient:
    """
    Peer-direct metadata client, duck-typed to match MxClient.

    Discovers peers via a configurable substrate (mDNS / static today) and
    performs all metadata RPCs directly against the serving peer. No
    central coordinator involvement.

    Machinery:
    - One background asyncio loop thread runs the substrate's discovery
      (mDNS ServiceBrowser etc.). Discovery callbacks enqueue resolved
      peers onto a plain ``queue.Queue``.
    - One enumerator thread pulls from that queue and synchronously calls
      ListWorkerSources on each peer to populate the cache. We keep gRPC
      off the asyncio loop thread so callbacks stay non-blocking.
    - The synchronous public API is served out of the cache, with optional
      pass-through gRPC for GetWorkerMetadata on demand.

    Args:
        substrate: ``"mdns"`` (default) or ``"static"``. When ``None``,
            resolved from ``MX_PEER_DISCOVERY_SUBSTRATE`` env var,
            defaulting to ``"mdns"``.
        ip: Advertised IP for mDNS (default from ``MX_PEER_IP`` or
            ``"127.0.0.1"``).
        port: WorkerGrpcServer port to bind on first
            :meth:`publish_metadata`. When ``None``, resolved from
            ``MX_WORKER_GRPC_PORT``; unset defaults to ``6555 +
            MX_WORKER_DEVICE_ID``. Use ``0`` to let the kernel choose.
        hostname: Advertised hostname for the mDNS SRV record. When
            ``None``, derived from the instance name.
        instance_name: Instance label for the mDNS service. When
            ``None``, zeroconf picks a random 32-char alphanumeric.
        peers: Comma-separated static peer list (``host:port,...``). Only
            consulted when ``substrate == "static"``. When ``None``, read
            from ``MX_PEER_ENDPOINTS``.
    """

    def __init__(
        self,
        substrate: str | None = None,
        *,
        ip: str | None = None,
        port: int | None = None,
        hostname: str | None = None,
        instance_name: str | None = None,
        peers: str | None = None,
        **_: Any,
    ):
        self._substrate = (
            substrate
            or os.environ.get("MX_PEER_DISCOVERY_SUBSTRATE", "mdns")
        ).lower()
        if self._substrate not in ("mdns", "static"):
            raise ValueError(
                f"unsupported substrate: {self._substrate!r} "
                "(expected 'mdns' or 'static')"
            )

        self._ip = ip or os.environ.get("MX_PEER_IP", "127.0.0.1")
        self._configured_port = port
        self._hostname = hostname
        self._instance_name = instance_name
        self._peers_str = (
            peers
            if peers is not None
            else os.environ.get("MX_PEER_ENDPOINTS", "")
        )

        # Background asyncio loop (hosts mDNS discovery).
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None

        # Substrate-specific state.
        self._mdns = None

        # Enumerator thread + its work queue. Items are tuples
        # ``(instance_name, port, addresses, txt)`` or ``_SHUTDOWN_SENTINEL``.
        self._pending: "queue.Queue[Any]" = queue.Queue()
        self._enumerator_thread: threading.Thread | None = None

        # Static substrate: periodic re-seed so an enumeration that failed
        # (e.g. peer not listening yet) gets retried. 5s is short enough to
        # recover a cold-start race but long enough not to hammer peers.
        self._static_refresh_interval_sec: float = 5.0
        self._static_refresh_thread: threading.Thread | None = None
        self._shutdown_event = self._shutdown_event if hasattr(self, "_shutdown_event") else threading.Event()

        # Shared peer state, guarded by ``self._lock``.
        self._lock = threading.Lock()
        self._peer_entries: dict[str, _PeerEntry] = {}
        self._source_to_peers: dict[tuple[str, str], str] = {}

        # Brief response cache for repeated GetWorkerMetadata calls - the
        # vLLM loader often fetches the same source twice during startup.
        # Keyed by ``(mx_source_id, worker_id)``.
        self._metadata_cache: dict[
            tuple[str, str], tuple[float, _FakeGetMetadataResponse]
        ] = {}

        # Local WorkerGrpcServers and the sources we own (i.e. sources
        # whose publish_metadata call we handled). Maps from
        # ``(mx_source_id, worker_id)`` to the server serving it.
        self._owned_servers: dict[tuple[str, str], WorkerGrpcServer] = {}
        self._local_port: int | None = None

        # Lifecycle flags.
        self._started = False
        self._closed = False
        self._shutdown_event = threading.Event()

    # -- Lifecycle ------------------------------------------------------------

    def _ensure_started(self) -> None:
        """Lazily start the event loop and substrate on first use."""
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            if self._closed:
                raise RuntimeError("PeerDirectMetadataClient is closed")
            self._start()

    def _start(self) -> None:
        """Start the asyncio loop thread and kick off the enumerator."""
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="peer-direct-loop",
        )
        self._loop_thread.start()

        self._enumerator_thread = threading.Thread(
            target=self._enumerator_run,
            daemon=True,
            name="peer-direct-enum",
        )
        self._enumerator_thread.start()

        if self._substrate == "static":
            self._seed_static_peers()
            self._static_refresh_thread = threading.Thread(
                target=self._static_refresh_run,
                daemon=True,
                name="peer-direct-static-refresh",
            )
            self._static_refresh_thread.start()

        self._started = True
        logger.info(
            "PeerDirectMetadataClient started (substrate=%s)",
            self._substrate,
        )

    def _seed_static_peers(self) -> None:
        """Enqueue every configured endpoint for the enumerator to probe."""
        try:
            from mx_peer_discovery.static import parse_endpoints
        except ImportError as e:
            logger.error(
                "mx_peer_discovery is required for the static substrate: %s", e
            )
            return

        endpoints = parse_endpoints(self._peers_str)
        if not endpoints:
            logger.info(
                "static substrate: no peers configured "
                "(MX_PEER_ENDPOINTS or peers=)"
            )
            return
        for host, port in endpoints:
            synthetic_name = f"static-{host}-{port}"
            self._pending.put((synthetic_name, port, [host], {}))
        logger.info(
            "static substrate: queued %d peer endpoint(s)", len(endpoints)
        )

    def _static_refresh_run(self) -> None:
        """Periodically re-enqueue static peers so failed enumerations retry.

        Static substrate seeds the queue once at startup; if a peer's gRPC
        server isn't up yet (cold-start race), its enumeration fails and
        the peer is never re-probed. This loop ticks every
        ``self._static_refresh_interval_sec`` and re-seeds the queue until
        the client is closed.
        """
        while not self._shutdown_event.wait(self._static_refresh_interval_sec):
            try:
                self._seed_static_peers()
            except Exception as e:
                logger.warning("static refresh failed: %s", e)

    def _start_mdns(self, advertised_port: int) -> None:
        """Start mDNS discovery. Called on first publish_metadata once we
        know our serving port."""
        if self._mdns is not None:
            return
        try:
            from mx_peer_discovery.mdns import Config, MdnsDiscovery
        except ImportError as e:
            logger.error(
                "mx_peer_discovery is required for the mdns substrate: %s", e
            )
            return

        hostname = self._hostname
        instance = self._instance_name
        # zeroconf requires a non-empty hostname; fall back to an instance
        # label if the caller didn't supply one.
        if not hostname:
            base = instance or "mx-peer"
            hostname = f"{base}.local."

        config = Config(
            hostname=hostname,
            ip=self._ip,
            port=advertised_port,
            on_resolved=self._on_peer_resolved,
            instance_name=instance,
        )
        self._mdns = MdnsDiscovery(config)

        fut = asyncio.run_coroutine_threadsafe(self._mdns.start(), self._loop)
        try:
            fut.result(timeout=10)
        except Exception as e:
            logger.error("failed to start mDNS discovery: %s", e)
            self._mdns = None
            return
        logger.info(
            "mDNS discovery started (instance=%s, ip=%s, port=%d)",
            self._mdns.instance_name,
            self._ip,
            advertised_port,
        )

    def close(self) -> None:
        """Stop discovery, join helper threads, stop local servers."""
        # Idempotent: never raise on the second call.
        if self._closed:
            return
        self._closed = True

        if not self._started:
            return

        self._shutdown_event.set()

        # Tear down mDNS on the asyncio loop.
        if self._mdns is not None and self._loop is not None and self._loop.is_running():
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._mdns.stop(), self._loop
                )
                fut.result(timeout=5)
            except Exception as e:
                logger.debug("error stopping mDNS: %s", e)
        self._mdns = None

        # Wake the enumerator so it can exit.
        self._pending.put(_SHUTDOWN_SENTINEL)
        if self._enumerator_thread is not None:
            self._enumerator_thread.join(timeout=5)
        self._enumerator_thread = None

        if self._static_refresh_thread is not None:
            self._static_refresh_thread.join(timeout=5)
        self._static_refresh_thread = None

        # Stop local gRPC servers.
        with self._lock:
            servers = list(self._owned_servers.values())
            self._owned_servers.clear()
        for server in servers:
            try:
                server.stop()
            except Exception as e:
                logger.debug("error stopping WorkerGrpcServer: %s", e)

        # Stop the asyncio loop.
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5)
        self._loop = None
        self._loop_thread = None

        logger.info("PeerDirectMetadataClient stopped")

    # -- MxClient-compatible interface ---------------------------------------

    def publish_metadata(
        self,
        identity: "p2p_pb2.SourceIdentity",
        worker: "p2p_pb2.WorkerMetadata",
        worker_id: str,
    ) -> str:
        """Publish metadata for one worker.

        Returns the mx_source_id computed locally. In peer-direct mode we
        don't forward this to a central server; instead we start (or
        reuse) a local WorkerGrpcServer and let the substrate advertise
        our presence so peers can discover us.
        """
        self._ensure_started()

        source_id = _compute_mx_source_id(identity)
        key = (source_id, worker_id)

        with self._lock:
            existing = self._owned_servers.get(key)
        if existing is not None:
            # Already serving this source; refresh status in place.
            existing.set_status(worker.status)
            logger.debug(
                "publish_metadata: refreshed existing source "
                "(mx_source_id=%s, worker_id=%s)",
                source_id, worker_id,
            )
            return source_id

        tensor_protos = list(worker.tensors)
        # The advertised WorkerMetadata should not echo the full tensor
        # list back to callers who already have it; the WorkerGrpcServer
        # injects tensors into GetWorkerMetadata responses itself when
        # the stored metadata has them stripped.
        advertised_worker = p2p_pb2.WorkerMetadata()
        advertised_worker.CopyFrom(worker)
        del advertised_worker.tensors[:]

        port = self._resolve_serving_port()
        server = WorkerGrpcServer(
            tensor_protos=tensor_protos,
            mx_source_id=source_id,
            port=port,
            worker_id=worker_id,
            model_name=identity.model_name,
            worker_rank=worker.worker_rank,
            status=worker.status,
            worker_metadata=advertised_worker,
        )
        bound_port = server.start()

        with self._lock:
            self._owned_servers[key] = server
            if self._local_port is None:
                self._local_port = bound_port

        # Kick off substrate advertisement the first time we know a port.
        if self._substrate == "mdns":
            self._start_mdns(bound_port)

        logger.info(
            "published source (mx_source_id=%s, worker_id=%s, port=%d)",
            source_id, worker_id, bound_port,
        )
        return source_id

    def list_sources(
        self,
        identity: "p2p_pb2.SourceIdentity | None" = None,
        status_filter: "p2p_pb2.SourceStatus | None" = None,
    ) -> _FakeListSourcesResponse:
        """Enumerate known sources across all cached peers."""
        self._ensure_started()

        filter_source_id = (
            _compute_mx_source_id(identity) if identity is not None else None
        )
        status_value = int(status_filter) if status_filter is not None else None

        instances: list[p2p_pb2.SourceInstanceRef] = []
        with self._lock:
            entries = list(self._peer_entries.values())

        for entry in entries:
            for ref in entry.sources:
                if filter_source_id is not None and ref.mx_source_id != filter_source_id:
                    continue
                if status_value is not None and int(ref.status) != status_value:
                    continue
                instances.append(ref)

        return _FakeListSourcesResponse(instances=instances)

    def get_metadata(
        self,
        mx_source_id: str,
        worker_id: str,
    ) -> _FakeGetMetadataResponse:
        """Fetch WorkerMetadata for one source by forwarding to the
        peer currently advertising it."""
        self._ensure_started()

        cache_key = (mx_source_id, worker_id)

        # Short-TTL cache to avoid re-fetching on back-to-back calls.
        now = time.monotonic()
        with self._lock:
            cached = self._metadata_cache.get(cache_key)
        if cached is not None:
            cached_at, response = cached
            if now - cached_at < _METADATA_CACHE_TTL_SECONDS:
                return response

        # If we own this source, serve from our local server directly.
        with self._lock:
            owned = self._owned_servers.get(cache_key)
        if owned is not None:
            response = self._serve_local_get_metadata(
                owned, mx_source_id, worker_id
            )
            with self._lock:
                self._metadata_cache[cache_key] = (now, response)
            return response

        with self._lock:
            instance_name = self._source_to_peers.get(cache_key)
            entry = (
                self._peer_entries.get(instance_name)
                if instance_name is not None
                else None
            )

        if entry is None:
            logger.debug(
                "get_metadata: source not in cache (mx_source_id=%s, worker_id=%s)",
                mx_source_id, worker_id,
            )
            return _FakeGetMetadataResponse(found=False)

        try:
            remote = fetch_worker_metadata(
                entry.endpoint, mx_source_id, worker_id, timeout=10.0
            )
        except grpc.RpcError as e:
            logger.warning(
                "get_metadata gRPC error from %s: %s", entry.endpoint, e
            )
            return _FakeGetMetadataResponse(found=False)
        except Exception as e:
            logger.error(
                "get_metadata unexpected error from %s: %s", entry.endpoint, e
            )
            return _FakeGetMetadataResponse(found=False)

        response = _FakeGetMetadataResponse(
            found=bool(remote.found),
            worker=remote.worker if remote.found else None,
            mx_source_id=remote.mx_source_id,
            worker_id=remote.worker_id,
        )
        with self._lock:
            self._metadata_cache[cache_key] = (now, response)
        return response

    def update_status(
        self,
        mx_source_id: str,
        worker_id: str,
        worker_rank: int,
        status: "p2p_pb2.SourceStatus",
    ) -> bool:
        """Update our own source's status, if we own it; no-op otherwise.

        Peer-direct mode has no central store to write to. Status updates
        for other peers propagate through their own ListWorkerSources
        responses as we re-enumerate them.
        """
        self._ensure_started()

        key = (mx_source_id, worker_id)
        with self._lock:
            server = self._owned_servers.get(key)
            # Invalidate any cached response: status just changed.
            self._metadata_cache.pop(key, None)
        if server is None:
            # Not ours, nothing to write. Return True so callers treat the
            # call as a soft success (matches the "no central store"
            # semantics we promised).
            logger.debug(
                "update_status: %s/%s not owned locally, no-op",
                mx_source_id, worker_id,
            )
            return True
        try:
            server.set_status(status)
        except Exception as e:
            logger.error(
                "update_status failed for %s/%s: %s",
                mx_source_id, worker_id, e,
            )
            return False
        # worker_rank is part of the method signature but not meaningful
        # here (each local WorkerGrpcServer already knows its own rank).
        del worker_rank
        return True

    # -- Internals -----------------------------------------------------------

    def _resolve_serving_port(self) -> int:
        """Resolve the port we'll bind WorkerGrpcServer to."""
        if self._configured_port is not None:
            return self._configured_port
        env_port = os.environ.get("MX_WORKER_GRPC_PORT")
        if env_port:
            try:
                return int(env_port)
            except ValueError:
                logger.warning(
                    "MX_WORKER_GRPC_PORT=%r is not an integer; ignoring",
                    env_port,
                )
        device_id_str = os.environ.get("MX_WORKER_DEVICE_ID", "0")
        try:
            device_id = int(device_id_str)
        except ValueError:
            device_id = 0
        return 6555 + device_id

    def _on_peer_resolved(
        self,
        instance_name: str,
        port: int,
        addresses: list[str],
        txt: dict[str, str],
    ) -> None:
        """mDNS callback: a new peer resolved. Runs on the asyncio loop
        thread - must not do blocking I/O."""
        if self._closed:
            return
        if not addresses:
            logger.debug("ignoring peer %r with no addresses", instance_name)
            return
        self._pending.put((instance_name, port, list(addresses), dict(txt)))

    def _enumerator_run(self) -> None:
        """Pull resolved peers off the queue and call ListWorkerSources.

        Runs until a ``_SHUTDOWN_SENTINEL`` is received.
        """
        while True:
            try:
                item = self._pending.get()
            except Exception as e:
                logger.debug("enumerator queue get failed: %s", e)
                continue

            if item is _SHUTDOWN_SENTINEL or self._shutdown_event.is_set():
                return

            try:
                instance_name, port, addresses, txt = item
            except (TypeError, ValueError) as e:
                logger.warning("enumerator got malformed item %r: %s", item, e)
                continue

            # Take the first address. The substrate probes are trusted to
            # have deduplicated; if a peer has multiple A records we just
            # use the first one returned.
            endpoint = f"{addresses[0]}:{port}"
            self._enumerate_peer(instance_name, endpoint, txt)

    def _enumerate_peer(
        self,
        instance_name: str,
        endpoint: str,
        txt: dict[str, str],
    ) -> None:
        """Fetch a peer's source list and update the cache."""
        try:
            refs = fetch_worker_sources(endpoint, timeout=5.0)
        except grpc.RpcError as e:
            logger.warning(
                "ListWorkerSources failed for %s (%s): %s",
                instance_name, endpoint, e,
            )
            return
        except Exception as e:
            logger.error(
                "unexpected ListWorkerSources error for %s (%s): %s",
                instance_name, endpoint, e,
            )
            return

        now = time.time()
        with self._lock:
            # Drop any previous source mappings owned by this peer before
            # re-registering - a peer can stop serving a source between
            # enumerations.
            existing = self._peer_entries.get(instance_name)
            if existing is not None:
                for ref in existing.sources:
                    key = (ref.mx_source_id, ref.worker_id)
                    if self._source_to_peers.get(key) == instance_name:
                        self._source_to_peers.pop(key, None)

            self._peer_entries[instance_name] = _PeerEntry(
                endpoint=endpoint,
                txt=dict(txt),
                last_seen=now,
                sources=list(refs),
            )
            for ref in refs:
                self._source_to_peers[(ref.mx_source_id, ref.worker_id)] = instance_name

        logger.info(
            "enumerated peer %s (%s): %d source(s)",
            instance_name, endpoint, len(refs),
        )

    def _serve_local_get_metadata(
        self,
        server: WorkerGrpcServer,
        mx_source_id: str,
        worker_id: str,
    ) -> _FakeGetMetadataResponse:
        """Build a GetMetadata-shaped response from a local server without
        going through gRPC."""
        local_port = server.port or self._local_port
        if local_port is None:
            logger.warning(
                "local server for %s/%s has no bound port yet",
                mx_source_id, worker_id,
            )
            return _FakeGetMetadataResponse(found=False)
        endpoint = f"127.0.0.1:{local_port}"
        try:
            remote = fetch_worker_metadata(
                endpoint, mx_source_id, worker_id, timeout=5.0
            )
        except Exception as e:
            logger.error(
                "local GetWorkerMetadata failed at %s: %s", endpoint, e
            )
            return _FakeGetMetadataResponse(found=False)
        return _FakeGetMetadataResponse(
            found=bool(remote.found),
            worker=remote.worker if remote.found else None,
            mx_source_id=remote.mx_source_id,
            worker_id=remote.worker_id,
        )
