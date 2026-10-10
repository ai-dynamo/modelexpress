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
from ..adapter import TrainerSourceShard
from ..plan import ResolvedSource, SourceResolver, TrainerSourceSnapshot, WeightSource

logger = logging.getLogger("modelexpress_rl.inference.source.trainer")

_MAX_MANIFEST_MESSAGE_SIZE_BYTES = 100 * 1024 * 1024


class _SlotReplicas:
    """Replicas published for one source slot, resolved as far as asked.

    Health is decided per slot and independently of every other slot. Pairing
    replicas by a shared offset instead means a fleet whose healthy replicas
    sit at different offsets in different slots yields no candidate at all,
    even though a complete healthy set exists: with only ``a0`` up in one slot
    and only ``b1`` in the next, the aligned pairs ``(a0, b0)`` and
    ``(a1, b1)`` both contain a dead source and ``(a0, b1)`` is never tried.

    Resolution stays lazy because it is charged per manifest fetched, and the
    first replica of each slot is all a healthy fleet ever needs.
    """

    def __init__(
        self,
        slot_id: str,
        shards: list[refit_pb2.WeightVersionShard],
        resolve: Callable[[refit_pb2.WeightVersionShard], TrainerSourceShard],
    ) -> None:
        self.slot_id = slot_id
        self._pending = list(shards)
        self._resolve = resolve
        self._usable: list[TrainerSourceShard] = []

    def usable(self, index: int) -> TrainerSourceShard | None:
        """Return the index-th usable replica, resolving no further than needed."""
        while len(self._usable) <= index and self._pending:
            shard = self._pending.pop(0)
            try:
                self._usable.append(self._resolve(shard))
            except (grpc.RpcError, RuntimeError) as error:
                logger.warning(
                    "trainer source %s failed for slot %s: %s",
                    shard.worker_id,
                    self.slot_id,
                    error,
                )
        if index < len(self._usable):
            return self._usable[index]
        return None

    @property
    def exhausted(self) -> bool:
        """Whether every published replica has been tried."""
        return not self._pending

    @property
    def usable_count(self) -> int:
        """How many replicas resolved so far; final once exhausted."""
        return len(self._usable)


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
        if version.trainer_mesh_id is None:
            raise RuntimeError("trainer publication requires trainer_mesh_id")
        try:
            mesh_response = self._service().GetTrainerMesh(
                refit_pb2.GetTrainerMeshRequest(mesh_id=version.trainer_mesh_id),
                timeout=self._rpc_timeout_seconds,
            )
        except grpc.RpcError as error:
            logger.warning(
                "trainer mesh lookup failed for version %s: %s",
                version.version_id,
                error,
            )
            return
        if not mesh_response.HasField("mesh"):
            raise RuntimeError("MX GetTrainerMesh response is missing mesh")
        mesh = mesh_response.mesh
        if mesh.mesh_id != version.trainer_mesh_id:
            raise RuntimeError("MX GetTrainerMesh returned a different mesh ID")
        if mesh.generation != version.trainer_mesh_generation:
            raise RuntimeError(
                "weight version trainer mesh generation differs from the current mesh"
            )
        expected_slots = tuple(sorted({
            metadata.logical_shard_id for metadata in mesh_response.mesh.workers.values()
        }))
        mesh_workers = mesh_response.mesh.workers
        mesh_generation = version.trainer_mesh_generation
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
            metadata = mesh_workers.get(shard.worker_id)
            if metadata is None or metadata.logical_shard_id != shard.logical_shard_id:
                continue
            published[shard.logical_shard_id].append(shard)

        slots = []
        for source_slot_id in expected_slots:
            ordered = sorted(
                published[source_slot_id], key=lambda item: item.worker_id
            )
            if not ordered:
                logger.warning(
                    "no trainer source published for required slot %s",
                    source_slot_id,
                )
                return
            slots.append(_SlotReplicas(source_slot_id, ordered, self._resolve_source))

        seen: set[tuple[tuple[str, str], ...]] = set()
        offset = 0
        while True:
            selected = []
            for slot in slots:
                source = slot.usable(offset)
                if source is None:
                    if not slot.usable_count:
                        logger.warning(
                            "no usable trainer source for required slot %s",
                            slot.slot_id,
                        )
                        return
                    # Exhausted and shorter than the candidate index, so cycle
                    # its healthy replicas rather than give up on a slot that
                    # simply has fewer of them.
                    source = slot.usable(offset % slot.usable_count)
                selected.append(source)
            selection = tuple(
                (source.source_slot_id, source.worker_id) for source in selected
            )
            if selection not in seen:
                # Fetching worker manifests can outlive the original mesh snapshot.
                current = self._service().GetTrainerMesh(
                    refit_pb2.GetTrainerMeshRequest(mesh_id=version.trainer_mesh_id),
                    timeout=self._rpc_timeout_seconds,
                )
                if (
                    not current.HasField("mesh")
                    or current.mesh.mesh_id != version.trainer_mesh_id
                    or current.mesh.generation != mesh_generation
                ):
                    raise RuntimeError("trainer mesh generation changed during source resolution")
                seen.add(selection)
                yield TrainerSourceSnapshot(
                    mesh_id=version.trainer_mesh_id,
                    mesh_generation=mesh_generation,
                    shards=tuple(selected),
                )
            deepest = max((slot.usable_count for slot in slots), default=1)
            if all(slot.exhausted for slot in slots) and offset + 1 >= deepest:
                return
            offset += 1

    def _resolve_source(
        self, shard: refit_pb2.WeightVersionShard
    ) -> TrainerSourceShard:
        if not shard.manifest_endpoint:
            raise RuntimeError("NIXL source is missing its manifest endpoint")
        if not shard.manifest_digest:
            raise RuntimeError("source is missing its manifest digest")
        key = (shard.logical_shard_id, shard.worker_id)
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
        return TrainerSourceShard(
            source_slot_id=shard.logical_shard_id,
            worker_id=shard.worker_id,
            manifest_digest=shard.manifest_digest,
            manifest_endpoint=shard.manifest_endpoint,
            manifest=manifest,
            structural_digest=structure_digest,
        )

    def _fetch_manifest(
        self, shard: refit_pb2.WeightVersionShard
    ) -> tuple[bytes, str]:
        """Fetch, verify and fingerprint one worker's manifest.

        Three spans on one stage rather than one, because the stage total
        cannot say whether a slow warm refit is waiting on the wire or on the
        CPU, and with digests published these manifests are refetched by
        construction on every version.
        """
        with refit_span(
            "source_preparation",
            accumulate_metadata=True,
            duration_key="manifest_fetch_s",
        ), grpc.insecure_channel(
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
                    logical_shard_id=shard.logical_shard_id,
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
                f"manifest digest mismatch for source slot {shard.logical_shard_id!r}"
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
                f"invalid manifest for source slot {shard.logical_shard_id!r}"
            ) from error
        return response.manifest, structure_digest


__all__ = ["TrainerSourceResolver"]
