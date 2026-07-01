# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DHT-backed metadata client using kademlite.

Decentralized peer-to-peer metadata discovery via Kademlia DHT. Each
worker publishes a pointer record into the DHT under a rank-keyed,
content-addressed key; receivers compute the same key from the
identity they want to load and pull the matching pointer with a
single GET. The pointer carries the worker's gRPC endpoint, and the
full tensor manifest stays on the worker, served via
``GetTensorManifest``.

Bootstrap mechanisms (in priority order):

- ``MX_DHT_BOOTSTRAP_LEASES``: K8s Lease-based self-organizing anchor
  quorum (the value is the anchor Lease name-prefix). Workers elect a
  small set of bootstrap anchors among themselves via
  ``coordination.k8s.io`` Leases - no dedicated seed pods. Requires
  RBAC to get/update Leases and ``POD_IP`` from the downward API.
- ``MX_DHT_BOOTSTRAP_PEERS``: comma-separated libp2p multiaddrs
  (e.g. ``/ip4/10.0.0.1/tcp/4001/p2p/Qm...``).
- ``MX_DHT_BOOTSTRAP_DNS``: K8s headless Service hostname; resolves
  to all backing pod IPs at port ``MX_DHT_BOOTSTRAP_PORT``
  (default ``4001``).
- ``MX_DHT_BOOTSTRAP_SLURM`` (or auto-detected ``SLURM_JOB_NODELIST``):
  Slurm-style hostlist (e.g. ``node[01-04]``); peers are dialed at
  ``MX_DHT_BOOTSTRAP_PORT`` on each resolved host.
- mDNS: auto-enabled when no other bootstrap is wired, which covers
  single-host and LAN tests.

Storage schema:

    key:   /mx/{mx_source_id}/rank/{worker_rank}
    value: JSON {worker_id, worker_rank, worker_grpc_endpoint,
                 metadata_endpoint, agent_name}

Rank-keyed entries mean a receiver can find the source serving its
own rank with one GET, no directory enumeration. Multi-instance
discovery (multiple pods serving the same rank) is out of scope for
this v1: this backend assumes one publisher per rank, matching what
the K8s-Service backend exposes via Service load-balancing.

``REQUIRES_P2P_METADATA = True`` forces ``publish_metadata_and_ready``
to take the P2P branch: each worker starts its own ``WorkerGrpcServer``
and serves the manifest itself. The DHT only carries the pointer.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import threading
import time
from typing import TYPE_CHECKING

import grpc

from . import p2p_pb2
from . import p2p_pb2_grpc
from .client import MxClientBase
from .metadata.source_id import compute_mx_source_id

if TYPE_CHECKING:
    from kademlite import DhtNode
    from kademlite.k8s_lease import LeaseCoordinator

logger = logging.getLogger("modelexpress.dht_client")

_DEFAULT_LISTEN = "0.0.0.0:0"
_DEFAULT_BOOTSTRAP_PORT = 4001
_DEFAULT_RECORD_TTL_SECONDS = 24 * 60 * 60  # match kademlite default
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_BACKOFF_SECONDS = 0.5
_LOOP_START_TIMEOUT = 10.0
_NODE_START_TIMEOUT = 30.0
_NODE_STOP_TIMEOUT = 10.0
_GET_TENSOR_MANIFEST_TIMEOUT = 30.0

# K8s Lease-based self-organizing anchor bootstrap (MX_DHT_BOOTSTRAP_LEASES).
# A fixed set of Lease slots elects a small anchor quorum from the workers
# themselves - no dedicated seed pods. Slot count is locked at 8 (a node maps
# to a slot by the low 3 bits of its peer_id); the anchor quorum converges
# among itself before workers join, which breaks the large-N cold-start stall.
_DHT_LEASE_SLOTS = 8
_DHT_LEASE_TTL_SECONDS = 15
_DHT_LEASE_RENEW_INTERVAL = 5.0
_DHT_LEASE_GATE_TIMEOUT = 60.0


def _key_for(mx_source_id: str, worker_rank: int) -> bytes:
    """Compute the DHT key for a given source/rank."""
    return f"/mx/{mx_source_id}/rank/{worker_rank}".encode("ascii")


def _encode_worker_pointer(
    worker: "p2p_pb2.WorkerMetadata", worker_id: str,
) -> bytes:
    """Encode the published pointer JSON.

    Only the dial information needed to reach the worker's gRPC server
    is stored in the DHT; the full tensor manifest stays on the worker
    and is served via ``GetTensorManifest``.
    """
    payload = {
        "worker_id": worker_id,
        "worker_rank": worker.worker_rank,
        "worker_grpc_endpoint": worker.worker_grpc_endpoint,
        "metadata_endpoint": worker.metadata_endpoint,
        "agent_name": worker.agent_name,
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _decode_worker_pointer(raw: bytes) -> dict:
    """Decode a worker pointer payload, validating required fields."""
    data = json.loads(raw.decode("utf-8"))
    required = {"worker_grpc_endpoint", "worker_rank"}
    missing = required - set(data)
    if missing:
        raise ValueError(
            f"DHT pointer missing required fields: {sorted(missing)}"
        )
    return data


class MxDhtClient(MxClientBase):
    """DHT-backed metadata client using kademlite for peer discovery."""

    REQUIRES_P2P_METADATA = True

    def __init__(
        self,
        worker_rank: int | None = None,
        listen_addr: str | None = None,
        bootstrap_peers: list[str] | None = None,
        bootstrap_dns: str | None = None,
        bootstrap_slurm: str | None = None,
        bootstrap_leases: str | None = None,
        lease_namespace: str | None = None,
        bootstrap_port: int | None = None,
        record_ttl_seconds: float | None = None,
        max_retries: int | None = None,
        backoff_seconds: float | None = None,
        k: int | None = None,
        alpha: int | None = None,
    ):
        self._worker_rank = worker_rank
        self._listen_addr = listen_addr or os.environ.get(
            "MX_DHT_LISTEN", _DEFAULT_LISTEN,
        )
        # Kademlia tuning passed straight to DhtNode. None means
        # "use kademlite's defaults" (K=20, ALPHA=3 in kademlite >= 0.1.1).
        self._k = k
        self._alpha = alpha

        if bootstrap_peers is None:
            env_peers = os.environ.get("MX_DHT_BOOTSTRAP_PEERS", "").strip()
            bootstrap_peers = (
                [p.strip() for p in env_peers.split(",") if p.strip()]
                if env_peers else []
            )
        self._bootstrap_peers = bootstrap_peers
        self._bootstrap_dns = (
            bootstrap_dns
            or os.environ.get("MX_DHT_BOOTSTRAP_DNS", "").strip()
            or None
        )

        # Slurm hostlist: explicit kwarg > MX_DHT_BOOTSTRAP_SLURM env >
        # SLURM_JOB_NODELIST (set automatically inside a Slurm allocation).
        self._bootstrap_slurm = (
            bootstrap_slurm
            or os.environ.get("MX_DHT_BOOTSTRAP_SLURM", "").strip()
            or os.environ.get("SLURM_JOB_NODELIST", "").strip()
            or None
        )

        # K8s Lease-based self-organizing bootstrap: the value of
        # MX_DHT_BOOTSTRAP_LEASES is the anchor Lease name-prefix, and its
        # presence enables anchor election over coordination.k8s.io Leases -
        # no dedicated seed pods. Namespace is auto-detected from the
        # in-cluster service account unless MX_DHT_LEASE_NAMESPACE overrides.
        self._bootstrap_leases = (
            bootstrap_leases
            or os.environ.get("MX_DHT_BOOTSTRAP_LEASES", "").strip()
            or None
        )
        self._lease_namespace = (
            lease_namespace
            or os.environ.get("MX_DHT_LEASE_NAMESPACE", "").strip()
            or None
        )

        env_port = os.environ.get("MX_DHT_BOOTSTRAP_PORT", "")
        self._bootstrap_port = (
            bootstrap_port if bootstrap_port is not None
            else int(env_port) if env_port
            else _DEFAULT_BOOTSTRAP_PORT
        )

        env_ttl = os.environ.get("MX_DHT_RECORD_TTL", "")
        self._record_ttl = (
            record_ttl_seconds if record_ttl_seconds is not None
            else float(env_ttl) if env_ttl
            else _DEFAULT_RECORD_TTL_SECONDS
        )

        env_retries = os.environ.get("MX_DHT_GET_RETRIES", "")
        self._max_retries = (
            max_retries if max_retries is not None
            else int(env_retries) if env_retries
            else _DEFAULT_MAX_RETRIES
        )

        env_backoff = os.environ.get("MX_DHT_GET_BACKOFF_SECONDS", "")
        self._backoff_seconds = (
            backoff_seconds if backoff_seconds is not None
            else float(env_backoff) if env_backoff
            else _DEFAULT_BACKOFF_SECONDS
        )

        self._node: DhtNode | None = None
        self._lease_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._started = threading.Event()
        self._start_lock = threading.Lock()

    # -- lifecycle ----------------------------------------------------------

    def _ensure_started(self) -> None:
        """Spin up the background asyncio loop and DhtNode lazily.

        Two-thread setup so MxClientBase's sync surface can call into a
        kademlite node that lives on its own asyncio loop. Bootstrap
        failures are logged but not fatal; the node still listens and
        can pick up peers via mDNS or future bootstrap attempts.
        """
        if self._started.is_set():
            return
        with self._start_lock:
            if self._started.is_set():
                return
            self._loop = asyncio.new_event_loop()
            loop_ready = threading.Event()

            def _run_loop() -> None:
                asyncio.set_event_loop(self._loop)
                self._loop.call_soon(loop_ready.set)
                try:
                    self._loop.run_forever()
                finally:
                    self._loop.close()

            # Non-daemon thread: when the caller exits, atexit-
            # registered close() fires first and gracefully tears the
            # node down (LEAVE notifications, in-flight RPC drain,
            # peer-side stream cleanup). Daemon=True killed the loop
            # mid-flight, leaving peers with stale routing-table
            # entries pointing at a dead pod for the full 5s TCP
            # dial timeout. atexit guards against callers that
            # forget to close() explicitly.
            self._loop_thread = threading.Thread(
                target=_run_loop, name="mx-dht-loop", daemon=False,
            )
            self._loop_thread.start()
            if not loop_ready.wait(timeout=_LOOP_START_TIMEOUT):
                raise RuntimeError(
                    f"MxDhtClient: event loop failed to start within "
                    f"{_LOOP_START_TIMEOUT}s"
                )

            future = asyncio.run_coroutine_threadsafe(
                self._start_node(), self._loop,
            )
            future.result(timeout=_NODE_START_TIMEOUT)
            self._started.set()
            # Best-effort cleanup on normal interpreter exit. SIGTERM
            # / SIGKILL bypass atexit; callers that need cleanup on
            # those signals must install a handler that calls
            # ``sys.exit`` (which does invoke atexit).
            atexit.register(self.close)

    async def _start_node(self) -> None:
        from kademlite import DhtNode

        host, port = self._parse_listen(self._listen_addr)
        # Pass k/alpha through only when explicitly set; older
        # kademlite versions don't accept these kwargs, so omitting
        # them when None preserves backward compatibility.
        node_kwargs: dict = {"record_ttl": self._record_ttl}
        if self._k is not None:
            node_kwargs["k"] = self._k
        if self._alpha is not None:
            node_kwargs["alpha"] = self._alpha
        self._node = DhtNode(**node_kwargs)
        # Mirror kademlite's intent: auto-disable mDNS when explicit
        # bootstrap is configured. kademlite's auto-detect runs inside
        # _node.start() and only sees args passed to start(); we wire
        # bootstrap via separate calls afterward, so we have to set the
        # flag ourselves or mDNS leaks discovery across the boundaries
        # the bootstrap layer is enforcing (e.g. K8s namespaces).
        has_explicit_bootstrap = bool(
            self._bootstrap_leases
            or self._bootstrap_peers
            or self._bootstrap_dns
            or self._bootstrap_slurm
        )
        await self._node.start(
            host, port, enable_mdns=not has_explicit_bootstrap,
        )
        listen = self._node.listen_addr or (host, port)
        logger.info(
            "MxDhtClient: kademlite DhtNode listening on %s:%d "
            "(peer_id=%s)",
            listen[0], listen[1], self._node.peer_id_short,
        )

        # Bootstrap priority: Leases > explicit peers > DNS > Slurm > mDNS.
        try:
            if self._bootstrap_leases:
                await self._bootstrap_via_leases()
            elif self._bootstrap_peers:
                logger.info(
                    "MxDhtClient: bootstrapping from %d peer multiaddrs",
                    len(self._bootstrap_peers),
                )
                await self._node.bootstrap(self._bootstrap_peers)
            elif self._bootstrap_dns:
                logger.info(
                    "MxDhtClient: bootstrapping from DNS %s:%d",
                    self._bootstrap_dns, self._bootstrap_port,
                )
                await self._node.bootstrap_from_dns(
                    self._bootstrap_dns, self._bootstrap_port,
                )
            elif self._bootstrap_slurm:
                logger.info(
                    "MxDhtClient: bootstrapping from Slurm hostlist "
                    "%r at port %d",
                    self._bootstrap_slurm, self._bootstrap_port,
                )
                await self._node.bootstrap_from_hostlist(
                    self._bootstrap_slurm, self._bootstrap_port,
                )
            else:
                logger.info(
                    "MxDhtClient: no explicit bootstrap configured; "
                    "relying on mDNS",
                )
        except Exception as exc:
            logger.warning(
                "MxDhtClient: bootstrap failed (%s); continuing with "
                "node listening, peers may join later via mDNS or "
                "republish", exc,
            )

    def _advertise_host(self, listen_host: str) -> str | None:
        """Return the dialable host other pods use to reach this node.

        Anchors listen on 0.0.0.0, so the multiaddr published into a Lease
        must carry a routable address. Prefer POD_IP (injected via the k8s
        downward API), then a non-wildcard listen host.
        """
        pod_ip = os.environ.get("POD_IP", "").strip()
        if pod_ip:
            return pod_ip
        if listen_host and listen_host not in ("0.0.0.0", "::", ""):
            return listen_host
        return None

    async def _bootstrap_via_leases(self) -> None:
        """Self-organizing anchor bootstrap over K8s Leases.

        A fixed set of Lease slots elects a small anchor quorum from the
        workers themselves - no dedicated seed pods. This node maps to one
        slot by its peer_id and contests that slot's Lease: winners become
        anchors (converge with the other anchors, then publish converged);
        losers gate on quorum convergence, then bootstrap to the anchor set.
        The anchor quorum converging before workers join is what breaks the
        large-N simultaneous cold-start stall.
        """
        from kademlite.k8s_lease import LeaseCoordinator
        from kademlite.multiaddr import (
            encode_multiaddr_ip_tcp_p2p,
            multiaddr_to_string,
        )

        node = self._node
        listen = node.listen_addr or self._parse_listen(self._listen_addr)
        advertise_host = self._advertise_host(listen[0])
        if advertise_host is None:
            logger.error(
                "MxDhtClient: MX_DHT_BOOTSTRAP_LEASES set but no advertisable "
                "address (set POD_IP via the downward API); skipping lease "
                "bootstrap - node stays up and may join via later republish",
            )
            return

        try:
            coordinator = LeaseCoordinator(
                name_prefix=self._bootstrap_leases,
                num_slots=_DHT_LEASE_SLOTS,
                ttl_seconds=_DHT_LEASE_TTL_SECONDS,
                namespace=self._lease_namespace,
            )
        except ValueError as exc:
            logger.error(
                "MxDhtClient: lease coordinator init failed (%s); skipping "
                "lease bootstrap", exc,
            )
            return

        my_multiaddr = multiaddr_to_string(
            encode_multiaddr_ip_tcp_p2p(advertise_host, listen[1], node.peer_id)
        )
        holder = node.peer_id.hex()
        slot = coordinator.slot_for(node.peer_id)

        won = await coordinator.claim(slot, holder, my_multiaddr)
        if won:
            logger.info(
                "MxDhtClient: won anchor slot %d (%s); converging quorum",
                slot, my_multiaddr,
            )
            # Dial whichever peer anchors are already up, then flag this
            # anchor converged so gated workers can begin joining.
            await self._bootstrap_to_anchors(coordinator, my_multiaddr)
            await coordinator.renew(slot, holder, my_multiaddr, converged=True)
            self._lease_task = asyncio.create_task(
                self._renew_lease_loop(coordinator, slot, holder, my_multiaddr)
            )
        else:
            logger.info(
                "MxDhtClient: not anchor for slot %d; awaiting quorum "
                "convergence (<=%ss)", slot, _DHT_LEASE_GATE_TIMEOUT,
            )
            converged = await coordinator.wait_all_converged(
                _DHT_LEASE_GATE_TIMEOUT
            )
            if not converged:
                logger.warning(
                    "MxDhtClient: anchor quorum not fully converged within "
                    "%ss; joining against whatever anchors are available",
                    _DHT_LEASE_GATE_TIMEOUT,
                )
            await self._bootstrap_to_anchors(coordinator, my_multiaddr)

    async def _bootstrap_to_anchors(
        self, coordinator: "LeaseCoordinator", my_multiaddr: str,
    ) -> None:
        """Bootstrap this node against the current anchor multiaddr set."""
        anchors = await coordinator.anchor_multiaddrs()
        peers = [a for a in anchors if a and a != my_multiaddr]
        if not peers:
            logger.warning(
                "MxDhtClient: no anchor multiaddrs available to bootstrap "
                "against yet",
            )
            return
        logger.info("MxDhtClient: bootstrapping to %d anchor(s)", len(peers))
        await self._node.bootstrap(peers)

    async def _renew_lease_loop(
        self,
        coordinator: "LeaseCoordinator",
        slot: int,
        holder: str,
        my_multiaddr: str,
    ) -> None:
        """Keep this anchor's Lease held and flagged converged until stopped."""
        try:
            while True:
                await asyncio.sleep(_DHT_LEASE_RENEW_INTERVAL)
                ok = await coordinator.renew(
                    slot, holder, my_multiaddr, converged=True,
                )
                if not ok:
                    logger.warning(
                        "MxDhtClient: lost anchor lease slot %d; stopping "
                        "renew loop", slot,
                    )
                    return
        except asyncio.CancelledError:
            return

    async def _stop_node(self) -> None:
        if self._lease_task is not None:
            self._lease_task.cancel()
            try:
                await self._lease_task
            except asyncio.CancelledError:
                pass
            self._lease_task = None
        if self._node is not None:
            await self._node.stop()

    def _run_async(self, coro):
        """Submit a coroutine to the background loop and block for result."""
        if self._loop is None:
            raise RuntimeError("MxDhtClient not started")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    @staticmethod
    def _parse_listen(addr: str) -> tuple[str, int]:
        """Parse ``host:port`` (or ``host``) into a tuple."""
        if ":" in addr:
            host, port_str = addr.rsplit(":", 1)
            return host or "0.0.0.0", int(port_str)
        return addr, 0

    def close(self) -> None:
        """Stop the DHT node and tear down the background loop."""
        if not self._started.is_set():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._stop_node(), self._loop,
            )
            future.result(timeout=_NODE_STOP_TIMEOUT)
        except Exception as exc:
            logger.warning("MxDhtClient: error stopping DhtNode: %s", exc)
        finally:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=_NODE_STOP_TIMEOUT)
            self._node = None
            self._loop = None
            self._loop_thread = None
            self._started.clear()

    # -- MxClientBase --------------------------------------------------------

    def publish_metadata(
        self,
        identity: "p2p_pb2.SourceIdentity",
        worker: "p2p_pb2.WorkerMetadata",
        worker_id: str,
    ) -> str:
        """Compute mx_source_id and PUT a pointer to this worker's gRPC.

        ``publish_metadata_and_ready`` calls this twice: first with an
        empty ``worker_grpc_endpoint`` (server not yet started), then
        with the real endpoint after ``WorkerGrpcServer.start()``. The
        first call is a no-op for the DHT (the pointer would advertise
        a dead endpoint); only the second call actually publishes.
        """
        source_id = compute_mx_source_id(identity)
        self._worker_rank = worker.worker_rank

        if not worker.worker_grpc_endpoint:
            logger.debug(
                "MxDhtClient.publish_metadata: skipping DHT PUT for "
                "worker_rank=%d (empty worker_grpc_endpoint, pre-start "
                "call); returning mx_source_id=%s",
                worker.worker_rank, source_id,
            )
            return source_id

        self._ensure_started()
        key = _key_for(source_id, worker.worker_rank)
        value = _encode_worker_pointer(worker, worker_id)
        replicas = self._run_async(self._node.put(key, value))
        logger.info(
            "MxDhtClient.publish_metadata: PUT /mx/%s/rank/%d "
            "(replicas=%d, endpoint=%s)",
            source_id, worker.worker_rank, replicas,
            worker.worker_grpc_endpoint,
        )
        return source_id

    def list_sources(
        self,
        identity: "p2p_pb2.SourceIdentity | None" = None,
        status_filter: "p2p_pb2.SourceStatus | None" = None,
    ) -> "p2p_pb2.ListSourcesResponse":
        """Return a synthetic single-instance ref for the rank-matched key.

        The DHT GET in ``get_metadata`` is the actual discovery mechanism;
        this method just produces a candidate that the caller's existing
        rank-matching loop in ``rdma_strategy`` can hand back to
        ``get_metadata``.
        """
        if identity is None:
            raise ValueError(
                "list_sources requires an identity so mx_source_id can "
                "be computed locally without a central coordinator"
            )
        if self._worker_rank is None:
            raise RuntimeError(
                "MxDhtClient needs a worker_rank before list_sources "
                "can resolve the rank-keyed DHT entry; pass worker_rank "
                "to the constructor or call publish_metadata first"
            )
        source_id = compute_mx_source_id(identity)
        ref = p2p_pb2.SourceInstanceRef(
            mx_source_id=source_id,
            worker_id="",
            model_name=identity.model_name,
            worker_rank=self._worker_rank,
        )
        return p2p_pb2.ListSourcesResponse(instances=[ref])

    def discover_endpoint(
        self,
        mx_source_id: str,
        worker_rank: int | None = None,
    ) -> str:
        """Look up a worker's gRPC endpoint via the DHT only.

        Discovery-only entry point that isolates the DHT walk from the
        manifest fetch ``get_metadata`` performs after it. Useful for
        benchmarking the cost of the discovery phase (lazy node start +
        bootstrap + iterative lookup) independent of manifest payload
        size. Retry semantics for "key not found" mirror those of
        ``get_metadata``.
        """
        if worker_rank is not None:
            self._worker_rank = worker_rank
        if self._worker_rank is None:
            raise RuntimeError(
                "MxDhtClient.discover_endpoint requires worker_rank; "
                "pass it explicitly or set it via publish_metadata / "
                "constructor"
            )
        self._ensure_started()
        key = _key_for(mx_source_id, self._worker_rank)

        for attempt in range(1, self._max_retries + 2):
            raw = self._run_async(self._node.get(key))
            if raw is not None:
                pointer = _decode_worker_pointer(raw)
                endpoint = pointer["worker_grpc_endpoint"]
                if not endpoint:
                    raise RuntimeError(
                        f"DHT pointer at /mx/{mx_source_id}"
                        f"/rank/{self._worker_rank} has empty "
                        f"worker_grpc_endpoint"
                    )
                logger.info(
                    "MxDhtClient.discover_endpoint: GET /mx/%s/rank/%d "
                    "-> %s (attempt=%d)",
                    mx_source_id, self._worker_rank, endpoint, attempt,
                )
                return endpoint
            if attempt > self._max_retries:
                break
            logger.warning(
                "MxDhtClient.discover_endpoint: key not found on "
                "attempt %d/%d; retrying after %.2fs",
                attempt, self._max_retries + 1, self._backoff_seconds,
            )
            time.sleep(self._backoff_seconds)

        raise RuntimeError(
            f"MxDhtClient.discover_endpoint: exhausted "
            f"{self._max_retries + 1} attempts for "
            f"mx_source_id={mx_source_id} rank={self._worker_rank}"
        )

    def get_metadata(
        self,
        mx_source_id: str,
        worker_id: str,
    ) -> "p2p_pb2.GetMetadataResponse":
        """Look up the worker pointer in the DHT, then fetch its manifest.

        Two-step protocol: GET the rank-keyed pointer from the DHT to
        learn the publisher's gRPC endpoint, then call
        ``GetTensorManifest`` against that endpoint. Retries cover both
        a transient missing key (publisher hasn't propagated yet) and
        ``FAILED_PRECONDITION`` from the worker (revision skew or wrong
        rank routed in).
        """
        if self._worker_rank is None:
            raise RuntimeError(
                "MxDhtClient.get_metadata requires worker_rank; call "
                "publish_metadata first or set it at construction time"
            )
        self._ensure_started()
        key = _key_for(mx_source_id, self._worker_rank)
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 2):
            try:
                raw = self._run_async(self._node.get(key))
                if raw is None:
                    last_error = RuntimeError(
                        f"DHT key not found: /mx/{mx_source_id}"
                        f"/rank/{self._worker_rank}"
                    )
                    if attempt > self._max_retries:
                        break
                    logger.warning(
                        "MxDhtClient.get_metadata: key not found "
                        "on attempt %d/%d; retrying after %.2fs",
                        attempt, self._max_retries + 1,
                        self._backoff_seconds,
                    )
                    time.sleep(self._backoff_seconds)
                    continue

                pointer = _decode_worker_pointer(raw)
                endpoint = pointer["worker_grpc_endpoint"]
                if not endpoint:
                    raise RuntimeError(
                        f"DHT pointer at /mx/{mx_source_id}"
                        f"/rank/{self._worker_rank} has empty "
                        f"worker_grpc_endpoint"
                    )

                resp = self._call_get_tensor_manifest(
                    endpoint, mx_source_id,
                )

                if resp.mx_source_id != mx_source_id:
                    raise RuntimeError(
                        f"manifest from {endpoint} failed validation: "
                        f"mx_source_id mismatch (expected "
                        f"{mx_source_id!r}, got {resp.mx_source_id!r})"
                    )
                if resp.worker_rank != self._worker_rank:
                    raise RuntimeError(
                        f"manifest from {endpoint} failed validation: "
                        f"worker_rank mismatch (expected "
                        f"{self._worker_rank}, got {resp.worker_rank})"
                    )

                worker = p2p_pb2.WorkerMetadata(
                    worker_rank=resp.worker_rank,
                    metadata_endpoint=resp.metadata_endpoint,
                    agent_name=resp.agent_name,
                    tensors=list(resp.tensors),
                    status=p2p_pb2.SOURCE_STATUS_READY,
                    worker_grpc_endpoint=endpoint,
                )
                logger.info(
                    "MxDhtClient.get_metadata: fetched manifest from "
                    "%s (mx_source_id=%s, rank=%d, %d tensors, "
                    "attempt=%d)",
                    endpoint, resp.mx_source_id, resp.worker_rank,
                    len(resp.tensors), attempt,
                )
                return p2p_pb2.GetMetadataResponse(
                    found=True,
                    worker=worker,
                    mx_source_id=resp.mx_source_id,
                    worker_id=pointer.get("worker_id", ""),
                )
            except grpc.RpcError as exc:
                last_error = exc
                if exc.code() != grpc.StatusCode.FAILED_PRECONDITION:
                    raise
                if attempt > self._max_retries:
                    break
                logger.warning(
                    "MxDhtClient.get_metadata: FAILED_PRECONDITION "
                    "on attempt %d/%d (%s); retrying after %.2fs",
                    attempt, self._max_retries + 1,
                    exc.details(), self._backoff_seconds,
                )
                time.sleep(self._backoff_seconds)
                continue

        message = (
            f"MxDhtClient.get_metadata: exhausted "
            f"{self._max_retries + 1} attempts for "
            f"mx_source_id={mx_source_id} rank={self._worker_rank}"
        )
        logger.error("%s: %s", message, last_error)
        raise RuntimeError(f"{message}: {last_error}") from last_error

    def _call_get_tensor_manifest(
        self, endpoint: str, mx_source_id: str,
    ) -> "p2p_pb2.GetTensorManifestResponse":
        """One-shot gRPC call to ``WorkerService.GetTensorManifest``."""
        channel = grpc.insecure_channel(endpoint)
        try:
            stub = p2p_pb2_grpc.WorkerServiceStub(channel)
            req = p2p_pb2.GetTensorManifestRequest(
                mx_source_id=mx_source_id,
            )
            return stub.GetTensorManifest(
                req, timeout=_GET_TENSOR_MANIFEST_TIMEOUT,
            )
        finally:
            channel.close()

    def update_status(
        self,
        mx_source_id: str,
        worker_id: str,
        worker_rank: int,
        status: "p2p_pb2.SourceStatus",
    ) -> bool:
        """No-op: DHT TTL handles staleness and republish covers liveness."""
        return True
