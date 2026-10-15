# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer-memory source resolution."""

import hashlib
import logging
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from time import perf_counter

import grpc
from modelexpress import envs
from modelexpress.refit.reshard.rendezvous import structural_manifest_digest, unwrap_rendezvous_blob
from modelexpress.refit.timing import add_refit_duration, refit_span

from ... import refit_pb2, refit_pb2_grpc
from ...control import WeightVersion
from ...train.manifest import ShardChecksumKey
from ...train import WeightPayloadFormat
from ..adapter import TrainerSourceShard
from ..plan import (
    ResolvedSource,
    ResolvedTrainerSource,
    SourceResolver,
    TrainerSourceSnapshot,
    WeightSource,
)

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


class _ResolutionRound:
    """Per-update source resolution, checksums, and timing."""

    def __init__(self, resolve_source, manifest_cache) -> None:
        self._resolve_source = resolve_source
        self._manifest_cache = manifest_cache
        self._verify = envs.MX_RESHARD_PUBLISH_DIGEST
        self.sources = {}
        self.checksums = {}
        self._counters: dict[str, int | float] = defaultdict(int)
        self._seconds = 0.0
        self._failed = False

    def resolve(self, shard: refit_pb2.WeightVersionShard) -> TrainerSourceShard:
        identity = (shard.logical_shard_id, shard.worker_id)
        if identity in self.sources:
            return self.sources[identity]
        started = perf_counter()
        try:
            source, checksums = self._resolve_source(
                shard, self._manifest_cache.get(identity),
                verify=self._verify, counters=self._counters,
            )
        except BaseException:
            self._failed = True
            raise
        finally:
            self._seconds += perf_counter() - started
        self._manifest_cache[identity] = (source, checksums)
        self.sources[identity] = source
        self.checksums[identity] = checksums if self._verify else {}
        return source

    def flush(self) -> None:
        if not self._counters:
            return
        add_refit_duration(
            "source_preparation", self._seconds,
            status="error" if self._failed else "ok", metadata=dict(self._counters),
            accumulate_metadata=True,
        )
        self._counters.clear()
        self._seconds = 0.0
        self._failed = False


class TrainerSourceResolver(SourceResolver):
    """Resolve trainer metadata and round checksums without compiling a plan."""

    def __init__(
        self,
        *,
        service: Callable[[], refit_pb2_grpc.RefitServiceStub],
        rpc_timeout_seconds: float,
    ) -> None:
        self._service = service
        self._manifest_cache: dict[
            tuple[str, str], tuple[TrainerSourceShard, dict[ShardChecksumKey, str] | None]
        ] = {}
        self._rpc_timeout_seconds = rpc_timeout_seconds

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
        mesh = self._get_mesh(version)
        if mesh is None:
            return
        mesh_workers, expected_slots, mesh_generation = mesh
        published = self._list_published_shards(version, mesh_workers)
        if published is None:
            return

        resolution = _ResolutionRound(self._resolve_source, self._manifest_cache)

        try:
            slots = []
            for source_slot_id in expected_slots:
                ordered = sorted(
                    published[source_slot_id],
                    key=lambda item: item.worker_id
                )
                if not ordered:
                    logger.warning(
                        "no trainer source published for required slot %s",
                        source_slot_id,
                    )
                    return
                slots.append(_SlotReplicas(source_slot_id, ordered, resolution.resolve))

            seen: set[tuple[tuple[str, str], ...]] = set()
            for selected in self._candidate_sets(slots, seen):
                selection = tuple(
                    (source.source_slot_id, source.worker_id) for source in selected
                )
                self._check_current_mesh(version.trainer_mesh_id, mesh_generation)
                seen.add(selection)
                resolution.flush()
                yield ResolvedTrainerSource(
                    snapshot=TrainerSourceSnapshot(
                        mesh_id=version.trainer_mesh_id,
                        mesh_generation=mesh_generation,
                        shards=selected,
                    ),
                    checksums={
                        key: digest
                        for source in selected
                        for key, digest in resolution.checksums[
                            (source.source_slot_id, source.worker_id)
                        ].items()
                    },
                )
        finally:
            resolution.flush()

    def _get_mesh(
        self, version: WeightVersion
    ) -> tuple[Mapping[str, refit_pb2.TrainerTensorsMetadata], tuple[str, ...], int] | None:
        try:
            response = self._service().GetTrainerMesh(
                refit_pb2.GetTrainerMeshRequest(mesh_id=version.trainer_mesh_id),
                timeout=self._rpc_timeout_seconds,
            )
        except grpc.RpcError as error:
            logger.warning(
                "trainer mesh lookup failed for version %s: %s",
                version.version_id,
                error,
            )
            return None
        if not response.HasField("mesh"):
            raise RuntimeError("MX GetTrainerMesh response is missing mesh")
        mesh = response.mesh
        if mesh.mesh_id != version.trainer_mesh_id:
            raise RuntimeError("MX GetTrainerMesh returned a different mesh ID")
        if mesh.generation != version.trainer_mesh_generation:
            raise RuntimeError(
                "weight version trainer mesh generation differs from the current mesh"
            )
        workers = mesh.workers
        expected_slots = tuple(sorted({
            metadata.logical_shard_id for metadata in workers.values()
        }))
        return workers, expected_slots, version.trainer_mesh_generation

    def _list_published_shards(
        self,
        version: WeightVersion,
        mesh_workers: Mapping[str, refit_pb2.TrainerTensorsMetadata],
    ) -> dict[str, list[refit_pb2.WeightVersionShard]] | None:
        try:
            with refit_span(
                "source_preparation",
                metadata={"manifest_list_count": 1},
                accumulate_metadata=True,
                duration_key="manifest_list_s",
            ) as counters:
                response = self._service().ListWeightVersionShards(
                    refit_pb2.ListWeightVersionShardsRequest(version_id=version.version_id),
                    timeout=self._rpc_timeout_seconds,
                )
                counters["published_shard_count"] = len(response.shards)
        except grpc.RpcError as error:
            logger.warning(
                "trainer source discovery failed for version %s: %s",
                version.version_id,
                error,
            )
            return None
        published = defaultdict(list)
        for shard in response.shards:
            if shard.version_id != version.version_id:
                raise RuntimeError("publication listing returned a different weight version")
            metadata = mesh_workers.get(shard.worker_id)
            if metadata is None or metadata.logical_shard_id != shard.logical_shard_id:
                continue
            published[shard.logical_shard_id].append(shard)
        return published

    @staticmethod
    def _candidate_sets(
        slots: list[_SlotReplicas],
        seen: set[tuple[tuple[str, str], ...]],
    ) -> Iterator[tuple[TrainerSourceShard, ...]]:
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
                yield tuple(selected)
            deepest = max((slot.usable_count for slot in slots), default=1)
            if all(slot.exhausted for slot in slots) and offset + 1 >= deepest:
                return
            offset += 1

    def _check_current_mesh(self, mesh_id: str, generation: int) -> None:
        current = self._service().GetTrainerMesh(
            refit_pb2.GetTrainerMeshRequest(mesh_id=mesh_id),
            timeout=self._rpc_timeout_seconds,
        )
        if (
            not current.HasField("mesh")
            or current.mesh.mesh_id != mesh_id
            or current.mesh.generation != generation
        ):
            raise RuntimeError("trainer mesh generation changed during source resolution")

    def _resolve_source(
        self,
        shard: refit_pb2.WeightVersionShard,
        cached: tuple[TrainerSourceShard, dict[ShardChecksumKey, str] | None] | None,
        *,
        verify: bool,
        counters: dict[str, int | float],
    ) -> tuple[TrainerSourceShard, dict[ShardChecksumKey, str] | None]:
        if not shard.manifest_endpoint:
            raise RuntimeError("NIXL source is missing its manifest endpoint")
        if not shard.manifest_digest:
            raise RuntimeError("source is missing its manifest digest")
        source, checksums = cached if cached is not None else (None, None)
        reusable = (
            source is not None
            and source.manifest_endpoint == shard.manifest_endpoint
            and source.manifest_digest == shard.manifest_digest
        )
        counters["manifest_cache_hits"] += int(reusable)
        counters["manifest_cache_misses"] += int(not reusable)
        if not reusable:
            blob, structural_digest = self._fetch_manifest(shard, counters)
            source = TrainerSourceShard(
                source_slot_id=shard.logical_shard_id,
                worker_id=shard.worker_id,
                manifest_digest=shard.manifest_digest,
                manifest_endpoint=shard.manifest_endpoint,
                manifest=blob,
                structural_digest=structural_digest,
            )
            checksums = None
            counters["manifest_fetch_bytes"] += len(blob)
            counters["manifest_fetch_count"] += 1
        counters["manifest_bytes"] += len(source.manifest)
        if verify and checksums is None:
            checksums = {}
            try:
                tensors = unwrap_rendezvous_blob(source.manifest).tensors
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    f"invalid manifest for source slot {shard.logical_shard_id!r}"
                ) from error
            for tensor in tensors:
                for item in tensor.shards:
                    if not item.digest:
                        raise RuntimeError("source did not publish a shard checksum")
                    key = (tensor.name, item.agent_name, item.addr,
                           tuple(item.shard_offset), tuple(item.shape))
                    checksums[key] = item.digest
        return source, checksums

    def _fetch_manifest(
        self, shard: refit_pb2.WeightVersionShard, counters: dict[str, int | float],
    ) -> tuple[bytes, str]:
        started = perf_counter()
        try:
            with grpc.insecure_channel(
                shard.manifest_endpoint,
                options=[("grpc.max_receive_message_length", _MAX_MANIFEST_MESSAGE_SIZE_BYTES)],
            ) as channel:
                response = refit_pb2_grpc.RefitWorkerServiceStub(channel).GetWeightVersionShardManifest(
                    refit_pb2.GetWeightVersionShardManifestRequest(
                        version_id=shard.version_id, logical_shard_id=shard.logical_shard_id,
                    ),
                    timeout=self._rpc_timeout_seconds,
                )
        finally:
            counters["manifest_fetch_s"] += perf_counter() - started
        started = perf_counter()
        digest = hashlib.sha256(response.manifest).hexdigest()
        counters["manifest_hash_s"] += perf_counter() - started
        if response.manifest_digest != shard.manifest_digest or digest != shard.manifest_digest:
            raise RuntimeError(f"manifest digest mismatch for source slot {shard.logical_shard_id!r}")
        started = perf_counter()
        try:
            structural_digest = structural_manifest_digest(response.manifest)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"invalid manifest for source slot {shard.logical_shard_id!r}") from error
        finally:
            counters["manifest_fingerprint_s"] += perf_counter() - started
        return response.manifest, structural_digest


__all__ = ["TrainerSourceResolver"]
