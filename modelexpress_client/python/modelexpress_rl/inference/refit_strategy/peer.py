# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference-peer refit strategy."""

import logging
import random

import grpc

from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from modelexpress.metadata.worker_server import fetch_tensor_manifest
from modelexpress.types import ManifestMismatchError

from ...control import WeightVersion
from ..adapter import GeneratorEngineAdapter
from .base import RefitStrategy

logger = logging.getLogger("modelexpress_rl.inference.refit_strategy.peer")


class _PeerRefitStrategy(RefitStrategy):
    """Prefer an already-updated generator with the same rank and version."""

    def __init__(
        self,
        *,
        adapter: GeneratorEngineAdapter,
        p2p_client: MxClient,
        worker_id: str,
        max_transfer_attempts: int,
        rpc_timeout_seconds: float,
    ) -> None:
        self._adapter = adapter
        self._p2p_client = p2p_client
        self._worker_id = worker_id
        self._max_transfer_attempts = max_transfer_attempts
        self._rpc_timeout_seconds = rpc_timeout_seconds

    def stage(self, version: WeightVersion) -> object | None:
        sources = list(self._list_ready_sources(version.version_id))
        random.Random().shuffle(sources)
        for source in sources[: self._max_transfer_attempts]:
            try:
                worker = self._fetch_source(source)
                staged = self._adapter.stage_peer_weight(worker)
            except (grpc.RpcError, RuntimeError, ManifestMismatchError) as error:
                logger.warning(
                    "P2P peer %s failed for version %s: %s",
                    source.worker_id,
                    version.version_id,
                    error,
                )
                continue
            logger.info(
                "staged weight version %s from P2P peer %s",
                version.version_id,
                source.worker_id,
            )
            return staged
        return None

    def _list_ready_sources(
        self, version_id: str
    ) -> tuple[p2p_pb2.SourceInstanceRef, ...]:
        try:
            identity = self._adapter.build_p2p_identity(version_id)
            response = self._p2p_client.list_sources(
                identity=identity,
                status_filter=p2p_pb2.SOURCE_STATUS_READY,
            )
        except (grpc.RpcError, RuntimeError) as error:
            logger.warning(
                "P2P peer discovery failed for version %s: %s",
                version_id,
                error,
            )
            return ()
        return tuple(
            source
            for source in response.instances
            if source.worker_rank == self._adapter.worker_rank
            and source.worker_id != self._worker_id
        )

    def _fetch_source(
        self, source: p2p_pb2.SourceInstanceRef
    ) -> p2p_pb2.WorkerMetadata:
        response = self._p2p_client.get_metadata(
            mx_source_id=source.mx_source_id,
            worker_id=source.worker_id,
        )
        if not response.found:
            raise RuntimeError(
                f"P2P metadata disappeared for worker {source.worker_id!r}"
            )
        worker = response.worker
        if worker.worker_rank != self._adapter.worker_rank:
            raise RuntimeError(
                f"P2P worker rank changed for worker {source.worker_id!r}"
            )
        if worker.worker_grpc_endpoint:
            tensors, _manifest_bytes = fetch_tensor_manifest(
                endpoint=worker.worker_grpc_endpoint,
                mx_source_id=source.mx_source_id,
                worker_id=source.worker_id,
                timeout=self._rpc_timeout_seconds,
            )
            worker.tensor_source.ClearField("tensors")
            worker.tensor_source.tensors.extend(tensors)
        return worker


__all__: list[str] = []
