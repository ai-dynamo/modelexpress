# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator-peer source resolution."""

import logging
import random
from collections.abc import Callable, Iterator

import grpc
from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from modelexpress.types import ManifestMismatchError

from ...control import WeightVersion
from ...train import WeightPayloadFormat
from ..plan import (
    GeneratorPeerUpdateSource,
    ResolvedSource,
    SourceResolver,
    WeightSource,
)

logger = logging.getLogger("modelexpress_rl.inference.source.generator")


class GeneratorSourceResolver(SourceResolver):
    """Resolve an already-updated generator with matching engine identity."""

    def __init__(
        self,
        *,
        p2p_client: MxClient,
        worker_id: str,
        worker_rank: int,
        build_identity: Callable[[str], p2p_pb2.SourceIdentity],
    ) -> None:
        self._p2p_client = p2p_client
        self._worker_id = worker_id
        self._worker_rank = worker_rank
        self._build_identity = build_identity

    @property
    def kind(self) -> WeightSource:
        return WeightSource.GENERATOR

    def supports(self, version: WeightVersion) -> bool:
        return version.payload_format is not WeightPayloadFormat.UNSPECIFIED

    def payload_format(self, version: WeightVersion) -> WeightPayloadFormat:
        del version
        return WeightPayloadFormat.FULL_TENSOR

    def candidates(self, version: WeightVersion) -> Iterator[ResolvedSource]:
        sources = list(self._list_ready_sources(version.version_id))
        random.Random().shuffle(sources)
        for source in sources:
            try:
                worker = self._fetch_source(source)
            except (grpc.RpcError, RuntimeError, ManifestMismatchError) as error:
                logger.warning(
                    "P2P peer %s failed for version %s: %s",
                    source.worker_id,
                    version.version_id,
                    error,
                )
                continue
            yield GeneratorPeerUpdateSource(
                worker=worker,
                mx_source_id=source.mx_source_id,
                worker_id=source.worker_id,
            )

    def _list_ready_sources(
        self, version_id: str
    ) -> tuple[p2p_pb2.SourceInstanceRef, ...]:
        try:
            response = self._p2p_client.list_sources(
                identity=self._build_identity(version_id),
                status_filter=p2p_pb2.SOURCE_STATUS_READY,
            )
        except (grpc.RpcError, RuntimeError) as error:
            logger.warning(
                "P2P peer discovery failed for version %s: %s", version_id, error
            )
            return ()
        return tuple(
            source
            for source in response.instances
            if source.worker_rank == self._worker_rank
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
        if worker.worker_rank != self._worker_rank:
            raise RuntimeError(
                f"P2P worker rank changed for worker {source.worker_id!r}"
            )
        if not worker.worker_grpc_endpoint:
            raise RuntimeError(
                f"P2P worker {source.worker_id!r} has no tensor lease endpoint"
            )
        return worker


__all__ = ["GeneratorSourceResolver"]
