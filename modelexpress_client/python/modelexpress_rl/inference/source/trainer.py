# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer-memory source resolution."""

import hashlib
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Iterator

import grpc

from modelexpress.refit.reshard.rendezvous import structural_manifest_digest

from ... import refit_pb2, refit_pb2_grpc, timing
from ...control import WeightVersion
from ...train import WeightPayloadFormat
from ..adapter import GeneratorSource, GeneratorTransferInputs, NixlGeneratorSource
from ..plan import ResolvedSource, SourceResolver, TrainerUpdateSource, WeightSource

logger = logging.getLogger("modelexpress_rl.inference.source.trainer")

_MAX_MANIFEST_MESSAGE_SIZE_BYTES = 100 * 1024 * 1024


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
        self._manifest_cache: dict[tuple[str, str], tuple[str, str, bytes, str]] = {}

    @property
    def kind(self) -> WeightSource:
        return WeightSource.TRAINER

    def supports(self, version: WeightVersion) -> bool:
        return version.payload_format is WeightPayloadFormat.FULL_TENSOR

    def payload_format(self, version: WeightVersion) -> WeightPayloadFormat:
        return version.payload_format

    def candidates(self, version: WeightVersion) -> Iterator[ResolvedSource]:
        list_started = time.perf_counter()
        try:
            response = self._service().ListWeightVersionShards(
                refit_pb2.ListWeightVersionShardsRequest(version_id=version.version_id),
                timeout=self._rpc_timeout_seconds,
            )
        except grpc.RpcError as error:
            logger.warning(
                "trainer source discovery failed for version %s: %s",
                version.version_id,
                error,
            )
            return
        list_s = time.perf_counter() - list_started
        timing.record_measured(
            "source_preparation",
            list_s,
            metadata={
                "manifest_list_s": list_s,
                "manifest_list_count": 1,
                "published_shard_count": len(response.shards),
            },
            accumulate_metadata=True,
        )
        candidates = defaultdict(list)
        for shard in response.shards:
            candidates[shard.source_slot_id].append(shard)

        ordered_slots = []
        for source_slot_id in version.expected_source_slots:
            ordered = sorted(
                candidates[source_slot_id], key=lambda item: item.worker_id
            )
            if not ordered:
                logger.warning(
                    "no trainer source published for required slot %s",
                    source_slot_id,
                )
                return
            ordered_slots.append(ordered)

        seen = set()
        candidate_count = max((len(slot) for slot in ordered_slots), default=1)
        for offset in range(candidate_count):
            selected_shards = tuple(slot[offset % len(slot)] for slot in ordered_slots)
            selection = tuple(
                (shard.source_slot_id, shard.worker_id) for shard in selected_shards
            )
            if selection in seen:
                continue
            seen.add(selection)
            resolved = []
            stats: dict[str, int | float] = {
                "manifest_fetch_s": 0.0,
                "manifest_hash_s": 0.0,
                "manifest_fingerprint_s": 0.0,
                "manifest_bytes": 0,
                "manifest_fetch_bytes": 0,
                "manifest_cache_hits": 0,
                "manifest_cache_misses": 0,
                "manifest_fetch_count": 0,
            }
            for shard in selected_shards:
                try:
                    source, source_stats = self._resolve_source(shard)
                except (grpc.RpcError, RuntimeError) as error:
                    logger.warning(
                        "trainer source %s failed for slot %s: %s",
                        shard.worker_id,
                        shard.source_slot_id,
                        error,
                    )
                    break
                resolved.append(source)
                for name, value in source_stats.items():
                    stats[name] += value
            source_preparation_s = (
                float(stats["manifest_fetch_s"])
                + float(stats["manifest_hash_s"])
                + float(stats["manifest_fingerprint_s"])
            )
            timing.record_measured(
                "source_preparation",
                source_preparation_s,
                metadata=stats,
                accumulate_metadata=True,
            )
            if len(resolved) != len(ordered_slots):
                continue
            yield TrainerUpdateSource(
                inputs=GeneratorTransferInputs(
                    version_id=version.version_id,
                    base_version_id=version.base_version_id,
                    layout_signature=version.layout_signature,
                    payload_format=version.payload_format,
                    sources=tuple(resolved),
                )
            )

    def _resolve_source(
        self, shard: refit_pb2.WeightVersionShard
    ) -> tuple[GeneratorSource, dict[str, int | float]]:
        if not shard.manifest_endpoint:
            raise RuntimeError("NIXL source is missing its manifest endpoint")
        if not shard.manifest_digest:
            raise RuntimeError("source is missing its manifest digest")
        key = (shard.source_slot_id, shard.worker_id)
        cached = self._manifest_cache.get(key)
        cache_hit = (
            cached is not None
            and cached[0] == shard.manifest_endpoint
            and cached[1] == shard.manifest_digest
        )
        fetch_s = 0.0
        hash_s = 0.0
        fingerprint_s = 0.0
        if cache_hit:
            assert cached is not None
            manifest = cached[2]
            structure_digest = cached[3]
        else:
            fetch_started = time.perf_counter()
            with grpc.insecure_channel(
                shard.manifest_endpoint,
                options=[
                    (
                        "grpc.max_receive_message_length",
                        _MAX_MANIFEST_MESSAGE_SIZE_BYTES,
                    )
                ],
            ) as channel:
                response = refit_pb2_grpc.RefitWorkerServiceStub(
                    channel
                ).GetWeightVersionShardManifest(
                    refit_pb2.GetWeightVersionShardManifestRequest(
                        version_id=shard.version_id,
                        source_slot_id=shard.source_slot_id,
                    ),
                    timeout=self._rpc_timeout_seconds,
                )
            fetch_s = time.perf_counter() - fetch_started
            hash_started = time.perf_counter()
            digest = hashlib.sha256(response.manifest).hexdigest()
            hash_s = time.perf_counter() - hash_started
            if (
                response.manifest_digest != shard.manifest_digest
                or digest != shard.manifest_digest
            ):
                raise RuntimeError(
                    f"manifest digest mismatch for source slot {shard.source_slot_id!r}"
                )
            fingerprint_started = time.perf_counter()
            try:
                structure_digest = structural_manifest_digest(response.manifest)
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"invalid manifest for source slot {shard.source_slot_id!r}"
                ) from error
            fingerprint_s = time.perf_counter() - fingerprint_started
            manifest = response.manifest
            self._manifest_cache[key] = (
                shard.manifest_endpoint,
                shard.manifest_digest,
                manifest,
                structure_digest,
            )
        source = GeneratorSource(
            source_slot_id=shard.source_slot_id,
            worker_id=shard.worker_id,
            manifest_digest=shard.manifest_digest,
            transport=NixlGeneratorSource(
                manifest_endpoint=shard.manifest_endpoint,
                manifest=manifest,
                structural_digest=structure_digest,
            ),
        )
        return source, {
            "manifest_fetch_s": fetch_s,
            "manifest_hash_s": hash_s,
            "manifest_fingerprint_s": fingerprint_s,
            "manifest_bytes": len(manifest),
            "manifest_fetch_bytes": len(manifest) * int(not cache_hit),
            "manifest_cache_hits": int(cache_hit),
            "manifest_cache_misses": int(not cache_hit),
            "manifest_fetch_count": int(not cache_hit),
        }


__all__ = ["TrainerSourceResolver"]
