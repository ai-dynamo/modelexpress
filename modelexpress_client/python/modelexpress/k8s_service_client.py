# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
K8s-Service-routed metadata client.

Duck-typed replacement for :class:`MxClient` that skips the central
coordinator entirely. Each source pool sits behind a Kubernetes Service
(one Service per tensor-parallel rank, selectors pinning to pods that
hold that rank); peers open a gRPC channel directly to the Service DNS
name and call ``GetTensorManifest``. Kube-proxy load-balances across
the ready backends for that rank.

Rank-matching is enforced two ways:

1. The Service selector scopes the backend pool to pods with the right
   ``mx.rank`` label, so kube-proxy only routes the caller to
   rank-compatible sources.
2. ``GetTensorManifest`` validates ``mx_source_id`` server-side. If the
   caller hit a backend that has been silently updated to a different
   revision or config (rolling update, version skew), the server
   returns ``FAILED_PRECONDITION`` and the client retries on a fresh
   channel so kube-proxy re-picks a backend.

There is no substrate advertisement here: the Service's Endpoints
object is the source list, maintained by K8s based on pod readiness.
"""

from __future__ import annotations

import logging
import os
import time

import grpc

from . import p2p_pb2
from . import p2p_pb2_grpc
from .source_id import compute_mx_source_id

logger = logging.getLogger("modelexpress.k8s_service_client")

_DEFAULT_SERVICE_PATTERN = "mx-sources-rank-{rank}:6555"
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_BACKOFF_SECONDS = 0.5


class K8sServiceMetadataClient:
    """K8s-Service-routed, duck-typed MxClient."""

    # Signals to metadata.publish_metadata_and_ready that this backend
    # has no central store to fall back to, so the P2P path (start
    # WorkerGrpcServer, serve tensor manifests directly) is required.
    # `_is_p2p_metadata_enabled` checks for this attribute; its absence
    # (MxClient) defaults to False.
    REQUIRES_P2P_METADATA = True

    def __init__(
        self,
        worker_rank: int | None = None,
        service_pattern: str | None = None,
        max_retries: int | None = None,
        backoff_seconds: float | None = None,
    ):
        self._worker_rank = worker_rank
        self._service_pattern = service_pattern or os.environ.get(
            "MX_K8S_SERVICE_PATTERN", _DEFAULT_SERVICE_PATTERN,
        )
        env_retries = os.environ.get("MX_K8S_SOURCE_RETRIES", "")
        self._max_retries = (
            max_retries if max_retries is not None
            else int(env_retries) if env_retries
            else _DEFAULT_MAX_RETRIES
        )
        env_backoff = os.environ.get("MX_K8S_SOURCE_BACKOFF_SECONDS", "")
        self._backoff_seconds = (
            backoff_seconds if backoff_seconds is not None
            else float(env_backoff) if env_backoff
            else _DEFAULT_BACKOFF_SECONDS
        )

    # -- connection management ------------------------------------------------

    def close(self) -> None:
        """No-op: channels are opened per-call and closed immediately."""

    # -- RPC wrappers (MxClient duck-type) -----------------------------------

    def publish_metadata(
        self,
        identity: "p2p_pb2.SourceIdentity",
        worker: "p2p_pb2.WorkerMetadata",
        worker_id: str,
    ) -> str:
        """Compute mx_source_id locally - there is no central store to hit.

        Caller (metadata.py) is responsible for starting the local
        WorkerGrpcServer; this method only produces the ID so caller
        has something to key against. Also records ``worker_rank`` so
        the DNS pattern can be resolved without a separate call.

        ``REQUIRES_P2P_METADATA = True`` on this class ensures
        ``publish_metadata_and_ready`` always takes the P2P branch
        and starts the WorkerGrpcServer, so no env-var wiring is
        needed from the deployer.
        """
        self._worker_rank = worker.worker_rank
        source_id = compute_mx_source_id(identity)
        logger.info(
            "K8sServiceMetadataClient.publish_metadata: "
            "computed mx_source_id=%s for worker_rank=%d",
            source_id, worker.worker_rank,
        )
        return source_id

    def list_sources(
        self,
        identity: "p2p_pb2.SourceIdentity | None" = None,
        status_filter: "p2p_pb2.SourceStatus | None" = None,
    ) -> "p2p_pb2.ListSourcesResponse":
        """Return a single synthetic source pointing at the rank-matched Service.

        Real source discovery is delegated to Kubernetes: the caller's
        own rank picks a Service whose selector only includes pods
        serving that rank, and kube-proxy handles backend selection.
        The caller's existing rank-matching loop in rdma_strategy just
        sees one candidate with matching rank.
        """
        if identity is None:
            raise ValueError(
                "list_sources requires an identity so mx_source_id can "
                "be computed locally without a central coordinator"
            )
        if self._worker_rank is None:
            raise RuntimeError(
                "K8sServiceMetadataClient needs a worker_rank before "
                "list_sources can resolve the Service endpoint; pass "
                "worker_rank to the constructor or call publish_metadata "
                "first"
            )
        source_id = compute_mx_source_id(identity)
        ref = p2p_pb2.SourceInstanceRef(
            mx_source_id=source_id,
            worker_id=f"svc-rank-{self._worker_rank}",
            model_name=identity.model_name,
            worker_rank=self._worker_rank,
        )
        return p2p_pb2.ListSourcesResponse(instances=[ref])

    def get_metadata(
        self,
        mx_source_id: str,
        worker_id: str,
    ) -> "p2p_pb2.GetMetadataResponse":
        """Call GetTensorManifest against the Service, retrying on mismatch.

        Each retry opens a fresh gRPC channel so kube-proxy re-picks a
        backend (a live channel is sticky to one backend, so reusing it
        would just hit the same wrong-revision pod again).
        """
        if self._worker_rank is None:
            raise RuntimeError(
                "K8sServiceMetadataClient.get_metadata requires "
                "worker_rank; call publish_metadata first or set it "
                "at construction time"
            )
        endpoint = self._resolve_endpoint()
        last_error: grpc.RpcError | None = None

        for attempt in range(1, self._max_retries + 2):
            channel = grpc.insecure_channel(endpoint)
            try:
                stub = p2p_pb2_grpc.WorkerServiceStub(channel)
                req = p2p_pb2.GetTensorManifestRequest(mx_source_id=mx_source_id)
                resp = stub.GetTensorManifest(req, timeout=30)
                worker = p2p_pb2.WorkerMetadata(
                    worker_rank=resp.worker_rank,
                    metadata_endpoint=resp.metadata_endpoint,
                    agent_name=resp.agent_name,
                    tensors=list(resp.tensors),
                    status=p2p_pb2.SOURCE_STATUS_READY,
                )
                logger.info(
                    "K8sServiceMetadataClient.get_metadata: fetched "
                    "manifest from %s (mx_source_id=%s, rank=%d, "
                    "%d tensors, attempt=%d)",
                    endpoint, resp.mx_source_id, resp.worker_rank,
                    len(resp.tensors), attempt,
                )
                return p2p_pb2.GetMetadataResponse(
                    found=True,
                    worker=worker,
                    mx_source_id=resp.mx_source_id,
                    worker_id=worker_id,
                )
            except grpc.RpcError as exc:
                last_error = exc
                if (
                    exc.code() == grpc.StatusCode.FAILED_PRECONDITION
                    and attempt <= self._max_retries
                ):
                    logger.warning(
                        "K8sServiceMetadataClient.get_metadata: "
                        "mx_source_id mismatch on attempt %d/%d "
                        "against %s (server: %s); retrying on fresh "
                        "channel after %.2fs backoff",
                        attempt, self._max_retries + 1, endpoint,
                        exc.details(), self._backoff_seconds,
                    )
                    time.sleep(self._backoff_seconds)
                    continue
                raise
            finally:
                channel.close()

        message = (
            f"K8sServiceMetadataClient.get_metadata: exhausted "
            f"{self._max_retries + 1} attempts against {endpoint} "
            f"with no matching mx_source_id"
        )
        logger.error("%s: %s", message, last_error)
        raise RuntimeError(f"{message}: {last_error}") from last_error

    def update_status(
        self,
        mx_source_id: str,
        worker_id: str,
        worker_rank: int,
        status: "p2p_pb2.SourceStatus",
    ) -> bool:
        """No-op: K8s readiness probes supersede central liveness tracking."""
        return True

    # -- helpers -------------------------------------------------------------

    def _resolve_endpoint(self) -> str:
        """Substitute ``{rank}`` in the Service pattern with the worker's rank."""
        return self._service_pattern.format(rank=self._worker_rank)
