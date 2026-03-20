# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DHT-based metadata client for ModelExpress.

Drop-in replacement for MxClient that uses the mx_libp2p Kademlia DHT
instead of a centralized gRPC server. Workers discover each other through
the DHT without any central coordination.

The DhtMetadataClient is duck-typed to match MxClient's interface so the
vLLM loader can use either one transparently.

Key schema (must match the Rust DhtBackend):
    /mx/{source_id}/attrs                     - source attributes (model_name, etc.)
    /mx/{source_id}/{worker_id}/{rank}        - per-worker record (JSON)
    /mx/{source_id}/{worker_id}/workers       - directory listing ranks
    /mx/{source_id}/instances                 - directory listing worker_ids
    /mx/_sources                              - global source_id list
"""

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from typing import Any

from . import p2p_pb2

logger = logging.getLogger("modelexpress.dht_client")

# Lazy import: mx_libp2p might not be installed in all environments
_DhtNode = None


def _get_dht_node_class():
    global _DhtNode
    if _DhtNode is None:
        from mx_libp2p import DhtNode
        _DhtNode = DhtNode
    return _DhtNode


# ---------------------------------------------------------------------------
# Source identity hashing (must match Rust compute_mx_source_id)
# ---------------------------------------------------------------------------

def _compute_mx_source_id(identity: "p2p_pb2.SourceIdentity") -> str:
    """Compute the 16-char hex source ID from a SourceIdentity proto.

    Must produce identical output to the Rust compute_mx_source_id function.
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
# DHT key helpers
# ---------------------------------------------------------------------------

def _worker_key(source_id: str, worker_id: str, rank: int) -> bytes:
    return f"/mx/{source_id}/{worker_id}/{rank}".encode()


def _worker_directory_key(source_id: str, worker_id: str) -> bytes:
    return f"/mx/{source_id}/{worker_id}/workers".encode()


def _instances_key(source_id: str) -> bytes:
    return f"/mx/{source_id}/instances".encode()


def _attrs_key(source_id: str) -> bytes:
    return f"/mx/{source_id}/attrs".encode()


def _sources_key() -> bytes:
    return b"/mx/_sources"


# ---------------------------------------------------------------------------
# Duck-typed response objects
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


class _FakeSourceInstanceRef:
    """Duck-typed stand-in for p2p_pb2.SourceInstanceRef."""

    def __init__(self, mx_source_id: str, worker_id: str, model_name: str, worker_rank: int):
        self.mx_source_id = mx_source_id
        self.worker_id = worker_id
        self.model_name = model_name
        self.worker_rank = worker_rank


class DhtMetadataClient:
    """
    DHT-based metadata client, duck-typed to match MxClient.

    Wraps a mx_libp2p DhtNode running in a background thread with its own
    asyncio event loop. All public methods are synchronous (matching MxClient)
    and delegate to the async DHT node via run_coroutine_threadsafe.

    Args:
        listen_addr: Address to listen on (default from MX_DHT_LISTEN or 0.0.0.0:4001).
        bootstrap_peers: Comma-separated multiaddr strings (default from MX_DHT_BOOTSTRAP_PEERS).
        bootstrap_dns: K8s headless service hostname (default from MX_DHT_BOOTSTRAP_DNS).
        record_ttl: Record TTL in seconds (default from MX_DHT_RECORD_TTL or 300).
    """

    def __init__(
        self,
        listen_addr: str | None = None,
        bootstrap_peers: str | None = None,
        bootstrap_dns: str | None = None,
        record_ttl: int | None = None,
    ):
        self._listen_addr = listen_addr or os.environ.get("MX_DHT_LISTEN", "0.0.0.0:4001")
        self._bootstrap_peers_str = bootstrap_peers or os.environ.get("MX_DHT_BOOTSTRAP_PEERS", "")
        self._bootstrap_dns = bootstrap_dns or os.environ.get("MX_DHT_BOOTSTRAP_DNS") or None
        self._record_ttl = record_ttl or int(os.environ.get("MX_DHT_RECORD_TTL", "300"))

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._node = None
        self._started = False
        self._lock = threading.Lock()

    def _ensure_started(self):
        """Lazy-start the DHT node on first use."""
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            self._start()

    def _start(self):
        """Start the background event loop and DHT node."""
        DhtNode = _get_dht_node_class()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            daemon=True,
            name="dht-metadata-client",
        )
        self._thread.start()

        # Create and start the DHT node in the background loop
        self._node = self._run(self._async_start(DhtNode))
        self._started = True
        logger.info("DHT metadata client started (listen=%s, ttl=%ds)", self._listen_addr, self._record_ttl)

    async def _async_start(self, DhtNode):
        """Create and start the DHT node (runs in background loop)."""
        node = DhtNode(record_ttl=self._record_ttl)

        # Parse listen address
        host, _, port_str = self._listen_addr.rpartition(":")
        port = int(port_str) if port_str else 4001
        if not host:
            host = "0.0.0.0"

        # Parse bootstrap peers
        bootstrap_peers = None
        if self._bootstrap_peers_str:
            bootstrap_peers = [p.strip() for p in self._bootstrap_peers_str.split(",") if p.strip()]

        await node.start(
            listen_host=host,
            listen_port=port,
            bootstrap_peers=bootstrap_peers or None,
            bootstrap_dns=self._bootstrap_dns,
        )
        return node

    def _run(self, coro):
        """Run a coroutine in the background event loop and wait for the result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=30)

    # -- MxClient-compatible interface -----------------------------------------

    def publish_metadata(
        self,
        identity: "p2p_pb2.SourceIdentity",
        worker: "p2p_pb2.WorkerMetadata",
        worker_id: str,
    ) -> str:
        """Publish metadata for one worker. Returns the mx_source_id."""
        self._ensure_started()
        source_id = _compute_mx_source_id(identity)
        self._run(self._async_publish(source_id, identity, worker, worker_id))
        return source_id

    def get_metadata(
        self,
        mx_source_id: str,
        worker_id: str,
    ) -> _FakeGetMetadataResponse:
        """Fetch full tensor metadata for one specific worker."""
        self._ensure_started()
        try:
            return self._run(self._async_get_metadata(mx_source_id, worker_id))
        except Exception as e:
            logger.error("DHT get_metadata failed: %s", e)
            return _FakeGetMetadataResponse(found=False)

    def list_sources(
        self,
        identity: "p2p_pb2.SourceIdentity | None" = None,
        status_filter: "p2p_pb2.SourceStatus | None" = None,
    ) -> _FakeListSourcesResponse:
        """List available source workers, optionally filtered by identity and status."""
        self._ensure_started()
        try:
            return self._run(self._async_list_sources(identity, status_filter))
        except Exception as e:
            logger.error("DHT list_sources failed: %s", e)
            return _FakeListSourcesResponse(instances=[])

    def update_status(
        self,
        mx_source_id: str,
        worker_id: str,
        worker_rank: int,
        status: "p2p_pb2.SourceStatus",
    ) -> bool:
        """Update worker status in the DHT. Returns True on success."""
        self._ensure_started()
        try:
            self._run(self._async_update_status(mx_source_id, worker_id, worker_rank, status))
            return True
        except Exception as e:
            logger.error("DHT update_status failed: %s", e)
            return False

    def close(self) -> None:
        """Shut down the DHT node and background event loop."""
        if not self._started:
            return
        try:
            if self._node and self._loop and self._loop.is_running():
                self._run(self._node.stop())
        except Exception as e:
            logger.debug("Error stopping DHT node: %s", e)
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._started = False
        logger.info("DHT metadata client stopped")

    # -- Async implementations -------------------------------------------------

    async def _async_publish(
        self,
        source_id: str,
        identity: "p2p_pb2.SourceIdentity",
        worker: "p2p_pb2.WorkerMetadata",
        worker_id: str,
    ) -> None:
        """Publish one worker's record and update all directory structures."""
        now = int(time.time() * 1000)
        rank = worker.worker_rank

        # Store the worker record
        record = _worker_to_json(worker)
        await self._node.put(
            _worker_key(source_id, worker_id, rank),
            json.dumps(record).encode(),
            ttl=self._record_ttl,
        )

        # Merge rank into this worker_id's directory
        existing_dir = await self._get_json(_worker_directory_key(source_id, worker_id))
        existing_ranks = set(existing_dir.get("ranks", [])) if existing_dir else set()
        existing_ranks.add(rank)
        directory = {"ranks": sorted(existing_ranks), "updated_at": now}
        await self._node.put(
            _worker_directory_key(source_id, worker_id),
            json.dumps(directory).encode(),
            ttl=self._record_ttl,
        )

        # Merge worker_id into instances directory
        existing_instances = await self._get_json(_instances_key(source_id))
        worker_ids = set(existing_instances.get("worker_ids", [])) if existing_instances else set()
        worker_ids.add(worker_id)
        instances = {"worker_ids": sorted(worker_ids), "updated_at": now}
        await self._node.put(
            _instances_key(source_id),
            json.dumps(instances).encode(),
            ttl=self._record_ttl,
        )

        # Store source attributes
        attrs = {
            "model_name": identity.model_name,
            "updated_at": now,
        }
        await self._node.put(
            _attrs_key(source_id),
            json.dumps(attrs).encode(),
            ttl=self._record_ttl,
        )

        # Add to global sources list
        existing_sources = await self._get_json(_sources_key())
        sources = set(existing_sources.get("source_ids", [])) if existing_sources else set()
        sources.add(source_id)
        sources_list = {"source_ids": sorted(sources), "updated_at": now}
        await self._node.put(
            _sources_key(),
            json.dumps(sources_list).encode(),
            ttl=self._record_ttl,
        )

        logger.debug(
            "Published metadata for '%s' (source_id=%s, worker_id=%s, rank=%d)",
            identity.model_name, source_id, worker_id, rank,
        )

    async def _async_get_metadata(
        self,
        mx_source_id: str,
        worker_id: str,
    ) -> _FakeGetMetadataResponse:
        """Read the worker directory and fetch all worker records."""
        directory = await self._get_json(_worker_directory_key(mx_source_id, worker_id))
        if not directory or not directory.get("ranks"):
            return _FakeGetMetadataResponse(found=False)

        # GetMetadata returns one WorkerMetadata combining all ranks for this worker_id.
        # The proto response has a single `worker` field. In practice each worker_id
        # publishes exactly one rank, but we handle multiple for correctness.
        workers = []
        for rank in directory["ranks"]:
            record = await self._get_json(_worker_key(mx_source_id, worker_id, rank))
            if record:
                workers.append(_json_to_worker_metadata(record))

        if not workers:
            return _FakeGetMetadataResponse(found=False)

        # Return the first worker (typical case: one rank per worker_id).
        # If multiple ranks exist, the caller accesses them via list_sources.
        return _FakeGetMetadataResponse(
            found=True,
            worker=workers[0],
            mx_source_id=mx_source_id,
            worker_id=worker_id,
        )

    async def _async_list_sources(
        self,
        identity: "p2p_pb2.SourceIdentity | None",
        status_filter: "p2p_pb2.SourceStatus | None",
    ) -> _FakeListSourcesResponse:
        """List source instances, optionally filtered by identity and status."""
        # Determine which source_ids to scan
        if identity is not None:
            source_ids = [_compute_mx_source_id(identity)]
        else:
            sources_list = await self._get_json(_sources_key())
            source_ids = sources_list.get("source_ids", []) if sources_list else []

        instances = []
        for sid in source_ids:
            attrs = await self._get_json(_attrs_key(sid))
            model_name = attrs.get("model_name", "") if attrs else ""

            inst_dir = await self._get_json(_instances_key(sid))
            if not inst_dir:
                continue

            for wid in inst_dir.get("worker_ids", []):
                worker_dir = await self._get_json(_worker_directory_key(sid, wid))
                if not worker_dir:
                    continue

                for rank in worker_dir.get("ranks", []):
                    record = await self._get_json(_worker_key(sid, wid, rank))
                    if not record:
                        continue

                    if status_filter is not None and record.get("status", 0) != int(status_filter):
                        continue

                    instances.append(_FakeSourceInstanceRef(
                        mx_source_id=sid,
                        worker_id=wid,
                        model_name=model_name,
                        worker_rank=rank,
                    ))

        return _FakeListSourcesResponse(instances=instances)

    async def _async_update_status(
        self,
        mx_source_id: str,
        worker_id: str,
        worker_rank: int,
        status: int,
    ) -> None:
        """Read-modify-write a worker's status in the DHT."""
        key = _worker_key(mx_source_id, worker_id, worker_rank)
        record = await self._get_json(key)
        if record is None:
            raise ValueError(
                f"worker record not found: source_id={mx_source_id}, "
                f"worker_id={worker_id}, rank={worker_rank}"
            )

        record["status"] = int(status)
        record["updated_at"] = int(time.time() * 1000)

        await self._node.put(key, json.dumps(record).encode(), ttl=self._record_ttl)
        logger.debug(
            "Updated status for source_id=%s worker_id=%s rank=%d -> %d",
            mx_source_id, worker_id, worker_rank, int(status),
        )

    async def _get_json(self, key: bytes) -> dict | None:
        """GET a key from the DHT and JSON-decode it."""
        raw = await self._node.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning("Failed to decode DHT record at %s: %s", key, e)
            return None


# ---------------------------------------------------------------------------
# JSON serialization helpers (must match Rust DhtBackend schema)
# ---------------------------------------------------------------------------

def _worker_to_json(worker: p2p_pb2.WorkerMetadata) -> dict[str, Any]:
    """Convert a WorkerMetadata proto to the DHT JSON schema."""
    # Determine backend type
    if worker.transfer_engine_session_id:
        backend_type = "transfer_engine"
    elif worker.metadata_endpoint:
        backend_type = "nixl"
    else:
        backend_type = "none"

    return {
        "worker_rank": worker.worker_rank,
        "backend_type": backend_type,
        "metadata_endpoint": worker.metadata_endpoint or None,
        "agent_name": worker.agent_name or None,
        "transfer_engine_session_id": worker.transfer_engine_session_id or None,
        "tensors": [
            {
                "name": t.name,
                "addr": t.addr,
                "size": t.size,
                "device_id": t.device_id,
                "dtype": t.dtype,
            }
            for t in worker.tensors
        ],
        "status": worker.status,
        "updated_at": worker.updated_at,
    }


def _json_to_worker_metadata(record: dict) -> p2p_pb2.WorkerMetadata:
    """Convert a DHT JSON record back to a WorkerMetadata proto."""
    tensors = [
        p2p_pb2.TensorDescriptor(
            name=t["name"],
            addr=t["addr"],
            size=t["size"],
            device_id=t["device_id"],
            dtype=t["dtype"],
        )
        for t in record.get("tensors", [])
    ]
    return p2p_pb2.WorkerMetadata(
        worker_rank=record.get("worker_rank", 0),
        metadata_endpoint=record.get("metadata_endpoint") or "",
        agent_name=record.get("agent_name") or "",
        transfer_engine_session_id=record.get("transfer_engine_session_id") or "",
        tensors=tensors,
        status=record.get("status", 0),
        updated_at=record.get("updated_at", 0),
    )
