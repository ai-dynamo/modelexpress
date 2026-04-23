# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
K8s-Service-routed metadata client.

Duck-typed replacement for :class:`MxClient` that skips the central
coordinator entirely. Each source pool sits behind a Kubernetes Service;
peers open a gRPC channel directly to the Service DNS name and call
``GetTensorManifest``. Kube-proxy load-balances across the ready
backends.

The ``MX_K8S_SERVICE_PATTERN`` pattern decides how rank is encoded:

- **Pattern with explicit port** (e.g. ``mx-sources-rank-{rank}:6555``):
  used verbatim after ``{rank}`` substitution. Rank is encoded in the
  hostname; caller hits one Service per rank, each with a label
  selector that scopes to pods holding that rank. Fits the
  1-GPU-per-pod topology.
- **Pattern without a port** (e.g. ``mx-sources``): client auto-appends
  ``:{MX_WORKER_GRPC_PORT + rank}``. Rank is encoded in the port;
  caller hits one Service with N named ports, each targeting the
  matching in-pod port. Fits the multi-GPU-per-pod topology where every
  pod has every rank.

Rank-matching is enforced two ways regardless of shape:

1. Shape 1: the Service selector scopes the backend pool to pods with
   the right ``mx.rank`` label. Shape 2: the port differentiation
   naturally picks the right rank-R WorkerGrpcServer inside the pod.
2. ``GetTensorManifest`` validates ``mx_source_id`` server-side and the
   client validates the response's ``mx_source_id`` and ``worker_rank``
   before accepting. Mismatches return ``FAILED_PRECONDITION`` (or
   raise on the client side), and the client retries on a fresh
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
from .client import MxClientBase
from .source_id import compute_mx_source_id

logger = logging.getLogger("modelexpress.k8s_service_client")

_DEFAULT_SERVICE_PATTERN = "mx-sources"
_DEFAULT_WORKER_GRPC_PORT = 6555
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_BACKOFF_SECONDS = 0.5


class MxK8sServiceClient(MxClientBase):
    """K8s-Service-routed metadata client."""

    # Signals to metadata.publish_metadata_and_ready that this backend
    # has no central store to fall back to, so the P2P path (start
    # WorkerGrpcServer, serve tensor manifests directly) is required.
    # `_is_p2p_metadata_enabled` checks for this attribute; MxClient
    # inherits the default False from MxClientBase.
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
            "MxK8sServiceClient.publish_metadata: "
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
                "MxK8sServiceClient needs a worker_rank before "
                "list_sources can resolve the Service endpoint; pass "
                "worker_rank to the constructor or call publish_metadata "
                "first"
            )
        source_id = compute_mx_source_id(identity)
        # worker_id is intentionally empty. This backend has no per-pod
        # addressability (kube-proxy picks the backend at connection
        # time); get_metadata ignores worker_id and routes through the
        # Service instead. Stamping a synthetic ID here would suggest
        # per-pod semantics that don't exist in this backend.
        ref = p2p_pb2.SourceInstanceRef(
            mx_source_id=source_id,
            worker_id="",
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
                "MxK8sServiceClient.get_metadata requires "
                "worker_rank; call publish_metadata first or set it "
                "at construction time"
            )
        endpoint = self._resolve_endpoint()
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 2):
            channel = grpc.insecure_channel(endpoint)
            try:
                stub = p2p_pb2_grpc.WorkerServiceStub(channel)
                req = p2p_pb2.GetTensorManifestRequest(mx_source_id=mx_source_id)
                resp = stub.GetTensorManifest(req, timeout=30)

                # Defense-in-depth: validate the response matches what
                # was asked for. The server-side handshake in
                # WorkerServiceServicer rejects mismatched mx_source_id
                # with FAILED_PRECONDITION, but only when the request
                # carries a non-empty ID AND the server's own storage
                # is correct. A misconfigured Service selector routing
                # the caller to a wrong-rank pool, or the client
                # somehow passing an empty mx_source_id, would slip
                # past that check. Validate both fields here before
                # accepting the manifest.
                mismatch_reason: str | None = None
                if resp.mx_source_id != mx_source_id:
                    mismatch_reason = (
                        f"mx_source_id mismatch: expected "
                        f"{mx_source_id!r}, got {resp.mx_source_id!r}"
                    )
                elif resp.worker_rank != self._worker_rank:
                    mismatch_reason = (
                        f"worker_rank mismatch: expected "
                        f"{self._worker_rank}, got {resp.worker_rank}"
                    )

                if mismatch_reason is not None:
                    last_error = RuntimeError(
                        f"manifest from {endpoint} failed validation: "
                        f"{mismatch_reason}"
                    )
                    if attempt <= self._max_retries:
                        logger.warning(
                            "MxK8sServiceClient.get_metadata: "
                            "%s on attempt %d/%d; retrying on fresh "
                            "channel after %.2fs backoff",
                            mismatch_reason, attempt,
                            self._max_retries + 1, self._backoff_seconds,
                        )
                        time.sleep(self._backoff_seconds)
                        continue
                    raise last_error

                worker = p2p_pb2.WorkerMetadata(
                    worker_rank=resp.worker_rank,
                    metadata_endpoint=resp.metadata_endpoint,
                    agent_name=resp.agent_name,
                    tensors=list(resp.tensors),
                    status=p2p_pb2.SOURCE_STATUS_READY,
                )
                logger.info(
                    "MxK8sServiceClient.get_metadata: fetched "
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
                        "MxK8sServiceClient.get_metadata: "
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
            f"MxK8sServiceClient.get_metadata: exhausted "
            f"{self._max_retries + 1} attempts against {endpoint}"
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
        """Substitute ``{rank}`` into the pattern; auto-append port if absent.

        If the pattern resolves to a string containing ``:`` (explicit
        ``host:port``), it is used verbatim. If the pattern is a bare
        hostname, the client appends ``:{MX_WORKER_GRPC_PORT + rank}`` so
        the multi-GPU-per-pod Shape-2 topology (one Service with N named
        ports) works without any explicit port encoding in the pattern.
        """
        resolved = self._service_pattern.format(rank=self._worker_rank)
        if ":" in resolved:
            return resolved
        base_port = int(
            os.environ.get("MX_WORKER_GRPC_PORT", str(_DEFAULT_WORKER_GRPC_PORT)),
        )
        return f"{resolved}:{base_port + self._worker_rank}"
