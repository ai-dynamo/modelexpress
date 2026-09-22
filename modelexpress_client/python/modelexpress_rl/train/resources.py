# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private transport and manifest ownership for the trainer client."""

from __future__ import annotations

import os
import socket
import uuid
from concurrent import futures
from typing import Any

import grpc

from modelexpress import envs

from .. import envs as rl_envs
from .. import refit_pb2_grpc
from .manifest import WeightVersionShardManifestService


def _required(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} is required")
    return value


class _TrainerResources:
    """Own transport and manifest resources hidden by the trainer client."""

    def __init__(
        self,
        *,
        manager: Any,
        manifest_service: WeightVersionShardManifestService,
        server: grpc.Server,
        worker_endpoint: str,
    ) -> None:
        self._manager = manager
        self._manifest_service = manifest_service
        self._server = server
        self._worker_endpoint = worker_endpoint
        self._closed = False

    @classmethod
    def initialize(
        cls,
        *,
        device_id: int,
        agent_name: str | None = None,
    ) -> _TrainerResources:
        """Start NIXL and the worker-local manifest service."""
        from modelexpress.nixl_transfer import NixlTransferManager

        host = _required(
            envs.MX_WORKER_HOST or socket.gethostbyname(socket.gethostname()),
            "MX_WORKER_HOST",
        )
        os.environ.setdefault("MX_WORKER_HOST", host)
        if agent_name is None:
            process_id = uuid.uuid4().hex[:8]
            rank = rl_envs.RANK
            rank_segment = f"{rank}-" if rank is not None else ""
            agent_name = f"modelexpress-trainer-{rank_segment}{process_id}"
        manager = NixlTransferManager(
            agent_name=agent_name,
            device_id=device_id,
            listen_port=envs.MX_METADATA_PORT + device_id,
        )
        manager.initialize()

        worker_port = envs.MX_WORKER_GRPC_PORT + device_id
        worker_endpoint = f"{host}:{worker_port}"
        manifest_service = WeightVersionShardManifestService(endpoint=worker_endpoint)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        try:
            refit_pb2_grpc.add_RefitWorkerServiceServicer_to_server(
                manifest_service, server
            )
            if server.add_insecure_port(f"[::]:{worker_port}") == 0:
                raise RuntimeError(
                    f"failed to bind ModelExpress worker endpoint {worker_endpoint}"
                )
            server.start()
        except Exception:
            server.stop(grace=None).wait()
            manager.shutdown()
            raise

        return cls(
            manager=manager,
            manifest_service=manifest_service,
            server=server,
            worker_endpoint=worker_endpoint,
        )

    @property
    def manager(self) -> Any:
        return self._manager

    @property
    def manifest_service(self) -> WeightVersionShardManifestService:
        return self._manifest_service

    @property
    def worker_endpoint(self) -> str:
        return self._worker_endpoint

    def close(self) -> None:
        """Close the manifest service and transport exactly once."""
        if self._closed:
            return
        self._server.stop(grace=None).wait()
        self._manager.shutdown()
        self._closed = True


__all__: list[str] = []
