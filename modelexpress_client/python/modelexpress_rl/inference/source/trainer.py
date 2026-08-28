# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer-memory source resolution."""

import hashlib
import logging
from collections import defaultdict
from collections.abc import Callable, Iterator

import grpc

from ... import refit_pb2, refit_pb2_grpc
from ...control import WeightVersion
from ...train import WeightPayloadFormat
from ..adapter import GeneratorSource, GeneratorTransferInputs, NixlGeneratorSource
from ..plan import ResolvedSource, SourceResolver, TrainerUpdateSource, WeightSource

logger = logging.getLogger("modelexpress_rl.inference.source.trainer")


class TrainerSourceResolver(SourceResolver):
    """Resolve trainer shard manifests without compiling a transfer plan."""

    def __init__(
        self,
        *,
        service: Callable[[], refit_pb2_grpc.RefitServiceStub],
        rpc_timeout_seconds: float,
    ) -> None:
        self._service = service
        self._rpc_timeout_seconds = rpc_timeout_seconds

    @property
    def kind(self) -> WeightSource:
        return WeightSource.TRAINER

    def supports(self, version: WeightVersion) -> bool:
        return version.payload_format is WeightPayloadFormat.FULL_TENSOR

    def payload_format(self, version: WeightVersion) -> WeightPayloadFormat:
        return version.payload_format

    def candidates(self, version: WeightVersion) -> Iterator[ResolvedSource]:
        try:
            response = self._service().ListWeightVersionShards(
                refit_pb2.ListWeightVersionShardsRequest(
                    version_id=version.version_id
                ),
                timeout=self._rpc_timeout_seconds,
            )
        except grpc.RpcError as error:
            logger.warning(
                "trainer source discovery failed for version %s: %s",
                version.version_id,
                error,
            )
            return
        candidates = defaultdict(list)
        for shard in response.shards:
            candidates[shard.source_slot_id].append(shard)

        resolved_slots: list[list[GeneratorSource]] = []
        for source_slot_id in version.expected_source_slots:
            ordered = sorted(
                candidates[source_slot_id], key=lambda item: item.worker_id
            )
            resolved = []
            for shard in ordered:
                try:
                    source = self._resolve_source(shard)
                except (grpc.RpcError, RuntimeError) as error:
                    logger.warning(
                        "trainer source %s failed for slot %s: %s",
                        shard.worker_id,
                        source_slot_id,
                        error,
                    )
                    continue
                resolved.append(source)
            if not resolved:
                logger.warning(
                    "no usable trainer source for required slot %s",
                    source_slot_id,
                )
                return
            resolved_slots.append(resolved)

        seen = set()
        candidate_count = max((len(slot) for slot in resolved_slots), default=1)
        for offset in range(candidate_count):
            selected = tuple(slot[offset % len(slot)] for slot in resolved_slots)
            fingerprint = tuple(
                (source.source_slot_id, source.worker_id) for source in selected
            )
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            yield TrainerUpdateSource(
                inputs=GeneratorTransferInputs(
                    version_id=version.version_id,
                    base_version_id=version.base_version_id,
                    layout_signature=version.layout_signature,
                    payload_format=version.payload_format,
                    sources=selected,
                )
            )

    def _resolve_source(self, shard: refit_pb2.WeightVersionShard) -> GeneratorSource:
        if not shard.manifest_endpoint:
            raise RuntimeError("NIXL source is missing its manifest endpoint")
        with grpc.insecure_channel(shard.manifest_endpoint) as channel:
            response = refit_pb2_grpc.RefitWorkerServiceStub(
                channel
            ).GetWeightVersionShardManifest(
                refit_pb2.GetWeightVersionShardManifestRequest(
                    version_id=shard.version_id,
                    source_slot_id=shard.source_slot_id,
                ),
                timeout=self._rpc_timeout_seconds,
            )
        if not shard.manifest_digest:
            raise RuntimeError("source is missing its manifest digest")
        digest = hashlib.sha256(response.manifest).hexdigest()
        if (
            response.manifest_digest != shard.manifest_digest
            or digest != shard.manifest_digest
        ):
            raise RuntimeError(
                f"manifest digest mismatch for source slot {shard.source_slot_id!r}"
            )
        return GeneratorSource(
            source_slot_id=shard.source_slot_id,
            worker_id=shard.worker_id,
            manifest_digest=shard.manifest_digest,
            transport=NixlGeneratorSource(
                manifest_endpoint=shard.manifest_endpoint,
                manifest=response.manifest,
            ),
        )


__all__ = ["TrainerSourceResolver"]
