# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer-memory source resolution."""

import hashlib
import logging
from functools import partial
from itertools import chain
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from time import perf_counter

import grpc
from modelexpress_rl import envs
from modelexpress_rl.refit.reshard.rendezvous import unwrap_rendezvous_blob
from modelexpress_rl.refit.timing import add_refit_duration, refit_span

from ... import refit_pb2, refit_pb2_grpc
from ...control import WeightVersion
from ...train.manifest import ShardChecksumKey, version_metadata_digest
from ...train import WeightPayloadFormat
from ..adapter import ReplicaGroup, TrainerSourceShard
from ..plan import (
    ResolvedSource,
    ResolvedTrainerSource,
    SourceResolver,
    TrainerSourceSnapshot,
    WeightSource,
)

logger = logging.getLogger("modelexpress_rl.inference.source.trainer")

_MAX_MANIFEST_MESSAGE_SIZE_BYTES = 100 * 1024 * 1024


def _coverage_signature(blob: bytes) -> tuple:
    """Describe logical tensor coverage without physical placement or digests."""
    tensors = unwrap_rendezvous_blob(blob).tensors
    return tuple(sorted(
        (
            tensor.name,
            str(tensor.dtype),
            tuple(tensor.full_shape),
            tuple(sorted(
                (tuple(shard.shard_offset), tuple(shard.shape))
                for shard in tensor.shards
            )),
        )
        for tensor in tensors
    ))


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

    def __init__(self, resolve_source, stable_sources) -> None:
        self._resolve_source = resolve_source
        self._stable_sources = stable_sources
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
                shard, self._stable_sources.get(identity),
                verify=self._verify, counters=self._counters,
            )
        except BaseException:
            self._failed = True
            raise
        finally:
            self._seconds += perf_counter() - started
        self._stable_sources[identity] = source
        self.sources[identity] = source
        self.checksums[identity] = checksums
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
        cached_source: Callable[[], TrainerSourceSnapshot | None] | None = None,
    ) -> None:
        self._service = service
        self._cached_source = cached_source
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
        mesh_workers, mesh_slots, mesh_generation = mesh
        published = self._list_published_shards(version, mesh_workers)
        if published is None:
            return

        cached = self._cached_source() if self._cached_source is not None else None
        if cached is not None and (
            cached.mesh_id != version.trainer_mesh_id
            or cached.mesh_generation != mesh_generation
        ):
            cached = None
        groups = cached.replica_groups if cached is not None else ()
        if not groups:
            groups = tuple(
                ReplicaGroup(slot, [
                    TrainerSourceShard(
                        source_slot_id=slot, worker_id=worker_id,
                        metadata_endpoint=member.metadata_endpoint,
                        stable_metadata_digest=None, metadata=None,
                    )
                    for worker_id, member in sorted(mesh_workers.items())
                    if member.logical_shard_id == slot
                ])
                for slot in mesh_slots
            )
        stable_sources = self._cached_replicas(cached) if cached is not None else {}
        resolution = _ResolutionRound(self._resolve_source, stable_sources)

        try:
            slots = self._published_slots(
                groups, published, resolution.resolve,
            )
            if slots is None:
                return

            seen: set[tuple[tuple[str, str], ...]] = set()
            for selected in self._candidate_sets(slots, seen):
                selection = tuple(
                    (source.source_slot_id, source.worker_id) for source in selected
                )
                self._check_current_mesh(version.trainer_mesh_id, mesh_generation)
                seen.add(selection)
                for group, source in zip(groups, selected, strict=True):
                    self._update_replica(group, source)
                snapshot = TrainerSourceSnapshot(
                    mesh_id=version.trainer_mesh_id,
                    mesh_generation=mesh_generation,
                    shards=selected,
                    replica_groups=groups,
                )
                resolution.flush()
                yield ResolvedTrainerSource(
                    snapshot=snapshot,
                    checksums={
                        key: digest
                        for source in selected
                        for key, digest in resolution.checksums[
                            (source.source_slot_id, source.worker_id)
                        ].items()
                    },
                    prepare_groups=partial(
                        self._prepare_groups, snapshot, published, stable_sources,
                        resolution.sources, prefetch=envs.MX_REFIT_PREFETCH_REPLICA_METADATA,
                        debug_validate=envs.MX_REFIT_DEBUG_VALIDATE_REPLICAS,
                    ),
                )
        finally:
            resolution.flush()

    @staticmethod
    def _published_slots(groups, published, resolve) -> list[_SlotReplicas] | None:
        slots = []
        for group in groups:
            publications = {item.worker_id: item for item in published[group.source_slot_id]}
            ordered = [
                publications[replica.worker_id] for replica in group.replicas
                if replica.worker_id in publications
            ]
            if not ordered:
                logger.warning("no trainer source published for required slot %s", group.source_slot_id)
                return None
            slots.append(_SlotReplicas(group.source_slot_id, ordered, resolve))
        return slots

    @staticmethod
    def _cached_replicas(snapshot: TrainerSourceSnapshot) -> dict:
        sources = (
            source for group in snapshot.replica_groups for source in group.replicas
        )
        return {
            (source.source_slot_id, source.worker_id): source
            for source in chain(sources, snapshot.shards)
            if source.metadata is not None and source.stable_metadata_digest
        }

    @staticmethod
    def _update_replica(group: ReplicaGroup, source: TrainerSourceShard) -> None:
        for index, replica in enumerate(group.replicas):
            if replica.worker_id == source.worker_id:
                group.replicas[index] = source
                return
        raise RuntimeError("cached trainer source group is missing a selected replica")

    def _prepare_groups(
        self, snapshot, published, stable_sources, round_sources, needed_slots,
        *, prefetch, debug_validate,
    ) -> None:
        if not prefetch and not debug_validate:
            return
        started = perf_counter()
        counters: dict[str, int | float] = defaultdict(int)
        failed = False
        try:
            groups = {group.source_slot_id: group for group in snapshot.replica_groups}
            for slot_id in sorted(needed_slots):
                group = groups.get(slot_id)
                if group is not None:
                    self._prepare_group(
                        group, published[slot_id], stable_sources, round_sources,
                        counters, debug_validate,
                    )
        except (grpc.RpcError, RuntimeError):
            failed = True
            raise
        finally:
            if counters:
                add_refit_duration(
                    "source_preparation", perf_counter() - started,
                    status="error" if failed else "ok", metadata=dict(counters),
                    accumulate_metadata=True,
                )

    def _prepare_group(
        self, group, publications, stable_sources, round_sources, counters, debug_validate,
    ) -> None:
        publications = {item.worker_id: item for item in publications}
        metadata = []
        for index, replica in enumerate(group.replicas):
            publication = publications.get(replica.worker_id)
            if publication is None:
                continue
            source = self._prefetch_replica(
                publication, stable_sources, round_sources, counters, debug_validate,
            )
            if source is None:
                continue
            group.replicas[index] = source
            metadata.append(source.metadata)
        if debug_validate and metadata and len({_coverage_signature(blob) for blob in metadata}) != 1:
            raise RuntimeError(
                f"trainer replicas disagree on tensor coverage for slot {group.source_slot_id!r}"
            )

    def _prefetch_replica(
        self, publication, stable_sources, round_sources, counters, debug_validate,
    ) -> TrainerSourceShard | None:
        identity = (publication.logical_shard_id, publication.worker_id)
        if identity in round_sources:
            return round_sources[identity]
        try:
            source, _ = self._resolve_source(
                publication, stable_sources.get(identity), verify=False, counters=counters,
            )
        except grpc.RpcError as error:
            if debug_validate or error.code() != grpc.StatusCode.UNAVAILABLE:
                raise
            counters["replica_metadata_prefetch_unavailable"] += 1
            return None
        stable_sources[identity] = source
        round_sources[identity] = source
        return source

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
        cached: TrainerSourceShard | None,
        *,
        verify: bool,
        counters: dict[str, int | float],
    ) -> tuple[TrainerSourceShard, dict[ShardChecksumKey, str]]:
        if not shard.metadata_endpoint:
            raise RuntimeError("NIXL source is missing its metadata endpoint")
        if not shard.stable_metadata_digest:
            raise RuntimeError("source is missing its stable metadata digest")
        checksums = {}
        counters["manifest_cache_hits"] += int(cached is not None)
        counters["manifest_cache_misses"] += int(cached is None)
        if cached is None or verify:
            with grpc.insecure_channel(
                shard.metadata_endpoint,
                options=[
                    ("grpc.max_receive_message_length", _MAX_MANIFEST_MESSAGE_SIZE_BYTES)
                ],
            ) as channel:
                stub = refit_pb2_grpc.RefitWorkerServiceStub(channel)
                if cached is None:
                    blob, keys = self._fetch_stable_metadata(stub, shard, counters)
                    cached = TrainerSourceShard(
                        source_slot_id=shard.logical_shard_id,
                        worker_id=shard.worker_id,
                        stable_metadata_digest=shard.stable_metadata_digest,
                        metadata_endpoint=shard.metadata_endpoint,
                        metadata=blob,
                        checksum_keys=keys,
                    )
                    counters["manifest_fetch_bytes"] += len(blob)
                    counters["manifest_fetch_count"] += 1
                if verify:
                    checksums = self._fetch_version_metadata(stub, shard, cached.checksum_keys)
                    counters["version_metadata_fetch_count"] += 1
        counters["manifest_bytes"] += len(cached.metadata)
        return cached, checksums

    def _fetch_stable_metadata(
        self,
        stub: refit_pb2_grpc.RefitWorkerServiceStub,
        shard: refit_pb2.WeightVersionShard,
        counters: dict[str, int | float],
    ) -> tuple[bytes, dict[tuple[str, int], ShardChecksumKey]]:
        started = perf_counter()
        try:
            response = stub.GetTrainerShardMetadata(
                refit_pb2.GetTrainerShardMetadataRequest(
                    metadata_digest=shard.stable_metadata_digest
                ),
                timeout=self._rpc_timeout_seconds,
            )
        finally:
            counters["manifest_fetch_s"] += perf_counter() - started
        blob = response.metadata
        if (
            response.metadata_digest != shard.stable_metadata_digest
            or hashlib.sha256(blob).hexdigest() != shard.stable_metadata_digest
        ):
            raise RuntimeError(f"stable metadata digest mismatch for source slot {shard.logical_shard_id!r}")
        try:
            tensors = unwrap_rendezvous_blob(blob, stable=True).tensors
            keys = {
                (tensor.name, index): (
                    tensor.name, item.agent_name, item.addr,
                    tuple(item.shard_offset), tuple(item.shape),
                )
                for tensor in tensors for index, item in enumerate(tensor.shards)
            }
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"invalid stable metadata for source slot {shard.logical_shard_id!r}") from error
        return blob, keys

    def _fetch_version_metadata(
        self,
        stub: refit_pb2_grpc.RefitWorkerServiceStub,
        shard: refit_pb2.WeightVersionShard,
        keys: Mapping[tuple[str, int], ShardChecksumKey],
    ) -> dict[ShardChecksumKey, str]:
        if not shard.version_metadata_digest:
            raise RuntimeError("source did not publish version verification metadata")
        response = stub.GetWeightVersionShardMetadata(
            refit_pb2.GetWeightVersionShardMetadataRequest(
                version_id=shard.version_id, logical_shard_id=shard.logical_shard_id
            ),
            timeout=self._rpc_timeout_seconds,
        )
        metadata = response.metadata
        try:
            digest = version_metadata_digest(metadata)
        except ValueError as error:
            raise RuntimeError("invalid version checksum metadata") from error
        if (
            not response.HasField("metadata")
            or response.version_metadata_digest != shard.version_metadata_digest
            or digest != shard.version_metadata_digest
            or metadata.version_id != shard.version_id
            or metadata.worker_id != shard.worker_id
            or metadata.logical_shard_id != shard.logical_shard_id
            or metadata.stable_metadata_digest != shard.stable_metadata_digest
        ):
            raise RuntimeError("version checksum metadata identity or digest mismatch")
        published = {(row.tensor_name, row.shard_index): row.digest for row in metadata.checksums}
        if published.keys() != keys.keys() or any(not digest for digest in published.values()):
            raise RuntimeError("version checksum metadata does not cover the stable source shards")
        return {keys[key]: digest for key, digest in published.items()}


__all__ = ["TrainerSourceResolver"]
