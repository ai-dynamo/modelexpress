# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A model-instance directory hosted by its rank-zero worker.

Registration is instance-local, using the engine's head address. Only discovery
uses a load-balanced Kubernetes Service. Inventories are read directly from the
registered worker generations so an unavailable node is never mistaken for a
node with no optional caches.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import grpc

from .. import p2p_pb2, p2p_pb2_grpc


def grpc_endpoint(host: str, port: int) -> str:
    """Format a host (including an IPv6 literal) and an actual worker port."""
    host = host.strip().strip("[]")
    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


class InstanceDirectory:
    """Bounded, generation-aware registration table for one engine instance."""

    def __init__(
        self,
        mx_source_id: str,
        leader_worker_id: str,
        world_size: int,
        *,
        ready_fn: Callable[[], bool],
        registration_ttl: float = 90,
        clock: Callable[[], float] = time.monotonic,
    ):
        if (
            not mx_source_id
            or not leader_worker_id
            or world_size < 1
            or registration_ttl <= 0
        ):
            raise ValueError("instance directory requires an identity and world size")
        self.mx_source_id = mx_source_id
        self.leader_worker_id = leader_worker_id
        self.world_size = world_size
        self._ready_fn = ready_fn
        self._ttl = registration_ttl
        self._clock = clock
        self._lock = Lock()
        self._workers: dict[
            int, tuple[p2p_pb2.InstanceWorkerEndpoint, float, bool]
        ] = {}
        self._revision = 0

    def register(self, request, context):
        if (
            request.mx_source_id != self.mx_source_id
            or request.world_size != self.world_size
        ):
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "instance identity/world size mismatch",
            )
        worker = request.worker
        if (
            worker.global_rank >= self.world_size
            or not worker.worker_id
            or not worker.worker_grpc_endpoint
            or (worker.global_rank == 0 and worker.worker_id != self.leader_worker_id)
        ):
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid instance worker")
        with self._lock:
            previous = self._workers.get(worker.global_rank)
            now = self._clock()
            if previous and previous[0] != worker and previous[2] and previous[1] > now:
                context.abort(
                    grpc.StatusCode.FAILED_PRECONDITION,
                    "rank already has a live worker generation",
                )
            copy = p2p_pb2.InstanceWorkerEndpoint()
            copy.CopyFrom(worker)
            self._workers[worker.global_rank] = (copy, now + self._ttl, request.ready)
            # Heartbeats extending a lease do not invalidate an in-flight read.
            if (
                previous is None
                or previous[0] != worker
                or previous[2] != request.ready
            ):
                self._revision += 1
        return p2p_pb2.RegisterInstanceWorkerResponse(
            leader_worker_id=self.leader_worker_id
        )

    def metadata(self, request, context, local_inventory):
        if request.mx_source_id != self.mx_source_id:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, "instance weight identity mismatch"
            )
        if not self._ready_fn():
            context.abort(grpc.StatusCode.UNAVAILABLE, "instance engine is not healthy")
        with self._lock:
            now = self._clock()
            workers = [entry[0] for _, entry in sorted(self._workers.items())]
            if len(workers) != self.world_size or any(
                expiry <= now or not ready
                for _, expiry, ready in self._workers.values()
            ):
                context.abort(
                    grpc.StatusCode.UNAVAILABLE, "instance workers are not all ready"
                )
            revision = self._revision

        remaining = context.time_remaining()
        deadline = time.monotonic() + min(
            5.0, remaining if remaining is not None else 5.0
        )

        def inventory(worker):
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise TimeoutError("instance inventory deadline exceeded")
            req = p2p_pb2.GetWorkerInventoryRequest(
                mx_source_id=self.mx_source_id,
                worker_id=worker.worker_id,
            )
            if worker.global_rank == 0:
                result = local_inventory(req, context)
            else:
                with grpc.insecure_channel(worker.worker_grpc_endpoint) as channel:
                    result = p2p_pb2_grpc.WorkerServiceStub(channel).GetWorkerInventory(
                        req, timeout=timeout
                    )
            if result.worker != worker:
                raise ValueError("worker inventory changed generation or endpoint")
            return result

        try:
            with ThreadPoolExecutor(max_workers=min(16, self.world_size)) as pool:
                inventories = list(pool.map(inventory, workers))
        except (grpc.RpcError, ValueError, TimeoutError) as exc:
            context.abort(
                grpc.StatusCode.UNAVAILABLE, f"instance inventory unavailable: {exc}"
            )
        with self._lock:
            if self._revision != revision or any(
                expiry <= self._clock() for _, expiry, _ in self._workers.values()
            ):
                context.abort(
                    grpc.StatusCode.UNAVAILABLE, "instance changed during discovery"
                )
        return p2p_pb2.GetInstanceMetadataResponse(
            mx_source_id=self.mx_source_id,
            leader_worker_id=self.leader_worker_id,
            world_size=self.world_size,
            workers=inventories,
        )
