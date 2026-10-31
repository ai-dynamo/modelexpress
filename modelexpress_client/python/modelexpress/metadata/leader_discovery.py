# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client side of single-Service, model-instance discovery."""

from __future__ import annotations

import logging
import time
from threading import Lock

import grpc
import torch

from .. import envs, p2p_pb2, p2p_pb2_grpc
from ..topology import apply_policy, blocks_selection, in_domain, resolve_policy
from .instance_metadata import InstanceDirectory, grpc_endpoint
from .payload import accelerators_compatible, tensor_source_metadata
from .source_id import compute_mx_source_id

logger = logging.getLogger("modelexpress.metadata.leader_discovery")


class LeaderDiscovery:
    def __init__(self, service: str, max_retries: int, backoff: float):
        if max_retries < 0 or backoff < 0:
            raise ValueError(
                "leader discovery retries and backoff must not be negative"
            )
        self.service = (
            service
            if ":" in service
            else grpc_endpoint(service, envs.MX_WORKER_GRPC_PORT)
        )
        self._max_retries = max_retries
        self._backoff = backoff
        self._identity = None
        self._world_size = 1
        self._global_rank = 0
        self._node_rank = 0
        self._leader_endpoint = ""
        self._ready_fn = None
        self._server = None
        self._worker = None
        self._snapshots = {}
        self._lock = Lock()
        self._registered_state = None

    def configure(self, ctx) -> None:
        from .artifact_lifecycle import artifact_health_ready, resolve_health_url

        self._identity = p2p_pb2.SourceIdentity()
        self._identity.CopyFrom(ctx.identity)
        self._snapshots.clear()
        self._global_rank = ctx.global_rank
        self._worker_rank = ctx.worker_rank
        self._node_rank = ctx.node_rank
        self._world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else max(1, ctx.identity.tensor_parallel_size)
            * max(1, ctx.identity.pipeline_parallel_size)
        )
        if self._global_rank == 0 and ctx.device_id != 0:
            raise ValueError(
                "leader discovery requires global rank 0 on local device 0"
            )
        head = ctx.head_addr or envs.LWS_LEADER_ADDRESS or ""
        if self._world_size > 1 and not head:
            # Local multi-GPU engines do not always set master_addr.
            if self._node_rank != 0:
                raise ValueError(
                    "multi-node leader discovery requires the engine head address"
                )
            head = "127.0.0.1"
        self._leader_endpoint = grpc_endpoint(
            head or "127.0.0.1", envs.MX_WORKER_GRPC_PORT
        )
        default_url = (
            "http://127.0.0.1:30000/health"
            if ctx.identity.backend_framework == p2p_pb2.BACKEND_FRAMEWORK_SGLANG
            else "http://127.0.0.1:8000/health"
        )
        url = resolve_health_url(envs.MX_ARTIFACT_READY_URL, default_url, ctx.head_addr)
        probe_timeout = (
            5.0
            if ctx.identity.backend_framework == p2p_pb2.BACKEND_FRAMEWORK_SGLANG
            else 1.0
        )
        self._ready_fn = lambda: artifact_health_ready(url, timeout_secs=probe_timeout)

    def attach(self, server, worker, worker_id: str) -> None:
        if self._identity is None:
            raise ValueError(
                "leader discovery must be configured from the engine load context"
            )
        self._server = server
        self._worker = p2p_pb2.InstanceWorkerEndpoint(
            global_rank=self._global_rank,
            worker_rank=worker.worker_rank,
            node_rank=self._node_rank,
            worker_id=worker_id,
            worker_grpc_endpoint=worker.worker_grpc_endpoint,
        )
        directory = None
        if self._global_rank == 0:
            directory = InstanceDirectory(
                compute_mx_source_id(self._identity),
                worker_id,
                self._world_size,
                ready_fn=self._ready_fn,
                registration_ttl=max(15, 3 * envs.MX_HEARTBEAT_INTERVAL_SECS),
            )
            # Self-registration must use the bound address, including port=0 in tests.
            self._leader_endpoint = worker.worker_grpc_endpoint
        server.configure_instance(self._worker, directory)

    def register(self, ready: bool) -> None:
        if self._worker is None:
            raise RuntimeError("instance worker server has not been attached")
        self._server.set_instance_ready(ready)
        request = p2p_pb2.RegisterInstanceWorkerRequest(
            mx_source_id=compute_mx_source_id(self._identity),
            world_size=self._world_size,
            worker=self._worker,
            ready=ready,
        )
        with grpc.insecure_channel(self._leader_endpoint) as channel:
            response = p2p_pb2_grpc.WorkerServiceStub(channel).RegisterInstanceWorker(
                request, timeout=5
            )
        state = (response.leader_worker_id, ready)
        if state != self._registered_state:
            logger.info(
                "Instance registration: global_rank=%d node_rank=%d endpoint=%s leader=%s ready=%s",
                self._global_rank,
                self._node_rank,
                self._worker.worker_grpc_endpoint,
                response.leader_worker_id,
                ready,
            )
            self._registered_state = state

    def update_status(self, source_id: str, status: int) -> None:
        if self._identity is not None and source_id == compute_mx_source_id(
            self._identity
        ):
            self.register(status == p2p_pb2.SOURCE_STATUS_READY)

    def metadata(self):
        if self._identity is None:
            raise ValueError("leader discovery requires a configured weight identity")
        source_id = compute_mx_source_id(self._identity)
        policy = resolve_policy()
        if blocks_selection(policy):
            raise LookupError("required instance topology is unknown")
        with self._lock:
            cached = self._snapshots.get(source_id)
            if isinstance(cached, Exception):
                raise cached
            if cached is not None:
                return cached
            last_error = None
            for attempt in range(self._max_retries + 1):
                try:
                    with grpc.insecure_channel(self.service) as channel:
                        response = p2p_pb2_grpc.WorkerServiceStub(
                            channel
                        ).GetInstanceMetadata(
                            p2p_pb2.GetInstanceMetadataRequest(mx_source_id=source_id),
                            timeout=10,
                        )
                    self._validate(response, source_id)
                    if policy is not None and policy.required:
                        selected = next(
                            (
                                w
                                for w in response.workers
                                if w.worker.worker_rank == self._worker_rank
                            ),
                            None,
                        )
                        if selected is None or not in_domain(selected, policy):
                            raise ValueError(
                                "selected instance worker is outside the required topology"
                            )
                    self._snapshots[source_id] = response
                    logger.info(
                        "Selected instance through %s: leader=%s workers=%d published_artifacts=%d",
                        self.service,
                        response.leader_worker_id,
                        len(response.workers),
                        sum(len(entry.artifacts) for entry in response.workers),
                    )
                    return response
                except grpc.RpcError as exc:
                    if exc.code() not in {
                        grpc.StatusCode.UNAVAILABLE,
                        grpc.StatusCode.FAILED_PRECONDITION,
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                    }:
                        self._snapshots[source_id] = exc
                        raise
                    last_error = exc
                except ValueError as exc:
                    last_error = exc
                if attempt < self._max_retries:
                    logger.warning(
                        "Leader discovery through %s failed on attempt %d/%d: %s",
                        self.service,
                        attempt + 1,
                        self._max_retries + 1,
                        last_error,
                    )
                    time.sleep(self._backoff)
            self._snapshots[source_id] = last_error
            raise last_error

    def _validate(self, response, source_id: str) -> None:
        if response.mx_source_id != source_id or not response.leader_worker_id:
            raise ValueError("instance metadata identity mismatch")
        if (
            response.world_size != self._world_size
            or len(response.workers) != self._world_size
        ):
            raise ValueError("incomplete instance metadata")
        ranks = set()
        shard_ranks = set()
        endpoints = set()
        for entry in response.workers:
            worker = entry.worker
            if (
                worker.global_rank in ranks
                or worker.worker_rank in shard_ranks
                or worker.worker_grpc_endpoint in endpoints
                or not worker.worker_id
                or not worker.worker_grpc_endpoint
            ):
                raise ValueError("invalid or duplicate instance worker")
            ranks.add(worker.global_rank)
            shard_ranks.add(worker.worker_rank)
            endpoints.add(worker.worker_grpc_endpoint)
            if (
                worker.global_rank == 0
                and worker.worker_id != response.leader_worker_id
            ):
                raise ValueError("instance leader generation mismatch")
            if any(
                not artifact.mx_source_id or not artifact.artifact_id
                for artifact in entry.artifacts
            ):
                raise ValueError("incomplete artifact inventory")
        if ranks != set(range(response.world_size)):
            raise ValueError("instance global ranks are not complete")

    def tensor_metadata(self, mx_source_id: str, worker_id: str, worker_rank: int):
        metadata = self.metadata()
        if mx_source_id != metadata.mx_source_id:
            raise ValueError("weight source identity differs from selected instance")
        selected = next(
            (
                entry.worker
                for entry in metadata.workers
                if entry.worker.worker_rank == worker_rank
            ),
            None,
        )
        if selected is None or (worker_id and worker_id != selected.worker_id):
            raise LookupError("selected instance does not contain the requested worker")
        with grpc.insecure_channel(selected.worker_grpc_endpoint) as channel:
            response = p2p_pb2_grpc.WorkerServiceStub(channel).GetTensorManifest(
                p2p_pb2.GetTensorManifestRequest(
                    mx_source_id=mx_source_id, worker_id=selected.worker_id
                ),
                timeout=30,
            )
        if (
            response.mx_source_id != mx_source_id
            or response.worker_rank != worker_rank
            or response.worker_id != selected.worker_id
        ):
            raise ValueError("direct tensor manifest does not match instance directory")
        return p2p_pb2.GetMetadataResponse(
            found=True,
            mx_source_id=mx_source_id,
            worker_id=selected.worker_id,
            worker=p2p_pb2.WorkerMetadata(
                worker_rank=response.worker_rank,
                metadata_endpoint=response.metadata_endpoint,
                agent_name=response.agent_name,
                tensor_source=tensor_source_metadata(response.tensors),
                status=p2p_pb2.SOURCE_STATUS_READY,
                worker_grpc_endpoint=selected.worker_grpc_endpoint,
                accelerator=response.accelerator,
                topology=next(
                    entry.topology
                    for entry in metadata.workers
                    if entry.worker == selected
                ),
            ),
        )

    def artifact_source(
        self, identity, node_rank: int, artifact_id: str, accelerator: str, policy=None
    ):
        from .artifact_transfer import ArtifactSourceEndpoint

        source_id = compute_mx_source_id(identity)
        matches = [
            (entry, artifact)
            for entry in apply_policy(list(self.metadata().workers), policy)
            if entry.worker.node_rank == node_rank
            for artifact in entry.artifacts
            if artifact.mx_source_id == source_id
            and artifact.mx_source_type == identity.mx_source_type
            and (not artifact_id or artifact.artifact_id == artifact_id)
            and accelerators_compatible(
                accelerator, entry.accelerator, mx_source_type=identity.mx_source_type
            )
        ]
        if not matches:
            raise LookupError(
                "selected instance has no published compatible artifact for this node"
            )
        if len(matches) != 1:
            raise ValueError(
                "ambiguous artifact inventory; one owner per node/type is required"
            )
        entry, artifact = matches[0]
        return ArtifactSourceEndpoint(
            mx_source_id=source_id,
            worker_id=entry.worker.worker_id,
            worker_rank=entry.worker.worker_rank,
            worker_grpc_endpoint=entry.worker.worker_grpc_endpoint,
            artifact_id=artifact.artifact_id,
        )
