# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainer-memory source resolution."""

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from time import perf_counter

import grpc
from modelexpress_rl import envs
from modelexpress_rl.refit.reshard.rendezvous import unwrap_rendezvous_blob
from modelexpress_rl.refit.timing import add_refit_duration, refit_span

from ... import refit_pb2, refit_pb2_grpc
from ...control import WeightVersion
from ...train.manifest import ShardChecksumKey, bound_tensor_manifest, version_metadata_digest
from ...train import WeightPayloadFormat
from ...version import MeshIdentity
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


@dataclass(frozen=True)
class _TrainerReplicaGroup:
    """Mesh members and cached stable metadata for one logical source slot."""

    source_slot_id: str
    worker_ids: tuple[str, ...]
    sources: dict[str, TrainerSourceShard] = field(default_factory=dict)

    def published(
        self, publications: list[refit_pb2.WeightVersionShard],
    ) -> list[refit_pb2.WeightVersionShard]:
        by_worker = {shard.worker_id: shard for shard in publications}
        return [by_worker[worker] for worker in self.worker_ids if worker in by_worker]


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

    def __init__(self, resolve_source, groups) -> None:
        self._resolve_source = resolve_source
        self._groups = groups
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
        group = self._groups[shard.logical_shard_id]
        started = perf_counter()
        try:
            source, checksums = self._resolve_source(
                shard, group.sources.get(shard.worker_id),
                verify=self._verify, counters=self._counters,
            )
        except BaseException:
            self._failed = True
            raise
        finally:
            self._seconds += perf_counter() - started
        group.sources[shard.worker_id] = source
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
    ) -> None:
        self._service = service
        self._mesh_identity: MeshIdentity | None = None
        self._groups: dict[str, _TrainerReplicaGroup] = {}
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
        mesh_workers, expected_slots, mesh_identity = mesh
        published = self._list_published_shards(version, mesh_workers)
        if published is None:
            return

        if self._mesh_identity != mesh_identity:
            self._groups = {
                slot: _TrainerReplicaGroup(slot, tuple(
                    worker_id for worker_id, member in sorted(mesh_workers.items())
                    if member.logical_shard_id == slot
                ))
                for slot in expected_slots
            }
            self._mesh_identity = mesh_identity
        resolution = _ResolutionRound(self._resolve_source, self._groups)

        try:
            slots = []
            for group in self._groups.values():
                ordered = group.published(published[group.source_slot_id])
                if not ordered:
                    logger.warning(
                        "no trainer source published for required slot %s",
                        group.source_slot_id,
                    )
                    return
                slots.append(_SlotReplicas(group.source_slot_id, ordered, resolution.resolve))

            seen: set[tuple[tuple[str, str], ...]] = set()
            for selected in self._candidate_sets(slots, seen):
                selection = tuple(
                    (source.source_slot_id, source.worker_id) for source in selected
                )
                if not seen:
                    self._prepare_replicas(published)
                self._check_current_mesh(mesh_identity)
                seen.add(selection)
                resolution.flush()
                yield ResolvedTrainerSource(
                    snapshot=TrainerSourceSnapshot(
                        mesh_identity=mesh_identity,
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

    def _prepare_replicas(self, published) -> None:
        debug_validate = envs.MX_REFIT_DEBUG_VALIDATE_REPLICAS
        if not envs.MX_REFIT_PREFETCH_REPLICA_METADATA and not debug_validate:
            return
        counters: dict[str, int | float] = defaultdict(int)
        with refit_span("source_preparation", metadata=counters, accumulate_metadata=True):
            for group in self._groups.values():
                coverage = set()
                for publication in group.published(published[group.source_slot_id]):
                    source = group.sources.get(publication.worker_id)
                    if source is None:
                        try:
                            source, _ = self._resolve_source(
                                publication, None, verify=False, counters=counters,
                            )
                        except grpc.RpcError as error:
                            if debug_validate or error.code() not in (
                                grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED
                            ):
                                raise
                            counters["replica_metadata_prefetch_unavailable"] += 1
                            continue
                        group.sources[publication.worker_id] = source
                    if debug_validate:
                        coverage.add(bound_tensor_manifest(json.loads(source.metadata)["tensors"]))
                if len(coverage) > 1:
                    raise RuntimeError(
                        f"trainer replicas disagree on tensor coverage for slot {group.source_slot_id!r}"
                    )

    def _get_mesh(
        self, version: WeightVersion
    ) -> tuple[Mapping[str, refit_pb2.TrainerTensorsMetadata], tuple[str, ...], MeshIdentity] | None:
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
        return workers, expected_slots, MeshIdentity(mesh.mesh_id, mesh.generation)

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

    def _check_current_mesh(self, identity: MeshIdentity) -> None:
        current = self._service().GetTrainerMesh(
            refit_pb2.GetTrainerMeshRequest(mesh_id=identity.mesh_id),
            timeout=self._rpc_timeout_seconds,
        )
        if (
            not current.HasField("mesh")
            or MeshIdentity(current.mesh.mesh_id, current.mesh.generation) != identity
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
