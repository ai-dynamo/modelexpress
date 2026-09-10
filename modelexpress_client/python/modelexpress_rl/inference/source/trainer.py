# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer-memory source resolution."""

import hashlib
import logging
from collections import defaultdict
from collections.abc import Callable, Iterator

import grpc
from modelexpress.refit.reshard.rendezvous import structural_manifest_digest
from modelexpress.refit.timing import refit_span

from ... import refit_pb2, refit_pb2_grpc
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
        try:
            with refit_span(
                "source_preparation",
                metadata={"manifest_list_count": 1},
                accumulate_metadata=True,
                duration_key="manifest_list_s",
            ) as counters:
                response = self._service().ListWeightVersionShards(
                    refit_pb2.ListWeightVersionShardsRequest(
                        version_id=version.version_id
                    ),
                    timeout=self._rpc_timeout_seconds,
                )
                counters["published_shard_count"] = len(response.shards)
        except grpc.RpcError as error:
            logger.warning(
                "trainer source discovery failed for version %s: %s",
                version.version_id,
                error,
            )
            return
        published = defaultdict(list)
        for shard in response.shards:
            published[shard.source_slot_id].append(shard)

        ordered_slots = []
        for source_slot_id in version.expected_source_slots:
            ordered = sorted(published[source_slot_id], key=lambda item: item.worker_id)
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
            for shard in selected_shards:
                try:
                    resolved.append(self._resolve_source(shard))
                except (grpc.RpcError, RuntimeError) as error:
                    logger.warning(
                        "trainer source %s failed for slot %s: %s",
                        shard.worker_id,
                        shard.source_slot_id,
                        error,
                    )
                    break
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

    def _resolve_source(self, shard: refit_pb2.WeightVersionShard) -> GeneratorSource:
        if not shard.manifest_endpoint:
            raise RuntimeError("NIXL source is missing its manifest endpoint")
        if not shard.manifest_digest:
            raise RuntimeError("source is missing its manifest digest")
        key = (shard.source_slot_id, shard.worker_id)
        cached = self._manifest_cache.get(key)
        reusable = (
            cached is not None
            and cached[0] == shard.manifest_endpoint
            and cached[1] == shard.manifest_digest
        )
        with refit_span(
            "source_preparation",
            metadata={
                "manifest_cache_hits": int(reusable),
                "manifest_cache_misses": int(not reusable),
            },
            accumulate_metadata=True,
        ) as counters:
            if reusable:
                # These bytes hashed to this digest when they were stored, so
                # verifying them again would be checking them against
                # themselves.
                assert cached is not None
                manifest = cached[2]
                structure_digest = cached[3]
            else:
                manifest, structure_digest = self._fetch_manifest(shard)
                self._manifest_cache[key] = (
                    shard.manifest_endpoint,
                    shard.manifest_digest,
                    manifest,
                    structure_digest,
                )
                counters["manifest_fetch_bytes"] = len(manifest)
                counters["manifest_fetch_count"] = 1
            counters["manifest_bytes"] = len(manifest)
        return GeneratorSource(
            source_slot_id=shard.source_slot_id,
            worker_id=shard.worker_id,
            manifest_digest=shard.manifest_digest,
            transport=NixlGeneratorSource(
                manifest_endpoint=shard.manifest_endpoint,
                manifest=manifest,
                structural_digest=structure_digest,
            ),
        )

    def _fetch_manifest(self, shard: refit_pb2.WeightVersionShard) -> tuple[bytes, str]:
        """Fetch, verify and fingerprint one worker's manifest.

        Three spans on one stage rather than one, because the stage total
        cannot say whether a slow warm refit is waiting on the wire or on the
        CPU, and with digests published these manifests are refetched by
        construction on every version.
        """
        with (
            refit_span(
                "source_preparation",
                accumulate_metadata=True,
                duration_key="manifest_fetch_s",
            ),
            grpc.insecure_channel(
                shard.manifest_endpoint,
                options=[
                    (
                        "grpc.max_receive_message_length",
                        _MAX_MANIFEST_MESSAGE_SIZE_BYTES,
                    )
                ],
            ) as channel,
        ):
            response = refit_pb2_grpc.RefitWorkerServiceStub(
                channel
            ).GetWeightVersionShardManifest(
                refit_pb2.GetWeightVersionShardManifestRequest(
                    version_id=shard.version_id,
                    source_slot_id=shard.source_slot_id,
                ),
                timeout=self._rpc_timeout_seconds,
            )
        with refit_span(
            "source_preparation",
            accumulate_metadata=True,
            duration_key="manifest_hash_s",
        ):
            digest = hashlib.sha256(response.manifest).hexdigest()
        if (
            response.manifest_digest != shard.manifest_digest
            or digest != shard.manifest_digest
        ):
            raise RuntimeError(
                f"manifest digest mismatch for source slot {shard.source_slot_id!r}"
            )
        try:
            with refit_span(
                "source_preparation",
                accumulate_metadata=True,
                duration_key="manifest_fingerprint_s",
            ):
                structure_digest = structural_manifest_digest(response.manifest)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                f"invalid manifest for source slot {shard.source_slot_id!r}"
            ) from error
        return response.manifest, structure_digest


__all__ = ["TrainerSourceResolver"]
