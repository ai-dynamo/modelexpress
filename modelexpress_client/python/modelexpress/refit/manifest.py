# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable revision-domain records and their exact protobuf mapping.

The wire contract lives in ``modelexpress_common/proto/revision.proto``. This
module mirrors that rank-oriented hierarchy as frozen Python records so
publisher, receiver, and orchestrator code never passes protobuf messages
around. Conversion is presence-exact in both directions: an absent optional
field stays absent on the wire, because the catalog reads presence (not
emptiness) when it validates exact-base and integrity rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from modelexpress import revision_pb2


class ChangeState(IntEnum):
    """Per-rank or per-shard change state. Absence never means ``CLEAN``."""

    CLEAN = revision_pb2.CHANGE_STATE_CLEAN
    DIRTY = revision_pb2.CHANGE_STATE_DIRTY


class DeltaTransferMethod(IntEnum):
    """Revision-wide publication unit and legal byte path."""

    CANONICAL = revision_pb2.DELTA_TRANSFER_METHOD_CANONICAL
    RANK_LOCAL = revision_pb2.DELTA_TRANSFER_METHOD_RANK_LOCAL
    P2P_CPU_RANK = revision_pb2.DELTA_TRANSFER_METHOD_P2P_CPU_RANK
    P2P_GPU_SHARD = revision_pb2.DELTA_TRANSFER_METHOD_P2P_GPU_SHARD


class RevisionLifecycleState(IntEnum):
    """Active P0 revision lifecycle. Receiver reports never mutate it."""

    READY = revision_pb2.REVISION_LIFECYCLE_STATE_READY
    COMMITTED = revision_pb2.REVISION_LIFECYCLE_STATE_COMMITTED


class ReceiverRevisionState(IntEnum):
    """Externally useful receiver facts; preparation stays internal."""

    BYTES_RECEIVED = revision_pb2.RECEIVER_REVISION_STATE_BYTES_RECEIVED
    VERIFIED = revision_pb2.RECEIVER_REVISION_STATE_VERIFIED
    FAILED = revision_pb2.RECEIVER_REVISION_STATE_FAILED
    POISONED = revision_pb2.RECEIVER_REVISION_STATE_POISONED


class RecoveryCandidateKind(IntEnum):
    """Shape of a legal lineage from the receiver's installed base to a target."""

    DIRECT_DELTA = revision_pb2.RECOVERY_CANDIDATE_KIND_DIRECT_DELTA
    DELTA_REPLAY = revision_pb2.RECOVERY_CANDIDATE_KIND_DELTA_REPLAY
    FULL_ANCHOR_REPLAY = revision_pb2.RECOVERY_CANDIDATE_KIND_FULL_ANCHOR_REPLAY
    FULL_TARGET = revision_pb2.RECOVERY_CANDIDATE_KIND_FULL_TARGET


class PublicationMode(IntEnum):
    """Publisher progress policy after a revision reaches ``READY``.

    ``ASYNC`` may return once immutable publication reaches ``READY``.
    ``BLOCK`` waits read-only for ``COMMITTED``; cancelling that wait stops
    waiting only and leaves the catalog revision ``READY``. The concrete wait
    and cancellation behavior belongs to the publisher, not to this contract.
    """

    BLOCK = revision_pb2.PUBLICATION_MODE_BLOCK
    ASYNC = revision_pb2.PUBLICATION_MODE_ASYNC


def _optional(message, field: str):
    """Return an optional scalar field's value, or ``None`` when absent."""
    return getattr(message, field) if message.HasField(field) else None


def _present(**fields) -> dict:
    """Drop ``None`` entries so absent optionals are never set on the wire."""
    return {name: value for name, value in fields.items() if value is not None}


@dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str
    object_version: str | None = None

    def to_proto(self) -> revision_pb2.S3Location:
        return revision_pb2.S3Location(
            bucket=self.bucket,
            key=self.key,
            **_present(object_version=self.object_version),
        )

    @classmethod
    def from_proto(cls, proto: revision_pb2.S3Location) -> S3Location:
        return cls(
            bucket=proto.bucket,
            key=proto.key,
            object_version=_optional(proto, "object_version"),
        )


@dataclass(frozen=True)
class ZeroMqLocation:
    endpoint: str
    payload_id: str

    def to_proto(self) -> revision_pb2.ZeroMqLocation:
        return revision_pb2.ZeroMqLocation(endpoint=self.endpoint, payload_id=self.payload_id)

    @classmethod
    def from_proto(cls, proto: revision_pb2.ZeroMqLocation) -> ZeroMqLocation:
        return cls(endpoint=proto.endpoint, payload_id=proto.payload_id)


@dataclass(frozen=True)
class FilesystemLocation:
    path: str

    def to_proto(self) -> revision_pb2.FilesystemLocation:
        return revision_pb2.FilesystemLocation(path=self.path)

    @classmethod
    def from_proto(cls, proto: revision_pb2.FilesystemLocation) -> FilesystemLocation:
        return cls(path=proto.path)


TRANSPORT_KINDS: tuple[str, ...] = ("s3", "zeromq", "filesystem")


@dataclass(frozen=True)
class DeltaLocation:
    """Exactly one transport, mirroring the wire ``oneof``."""

    s3: S3Location | None = None
    zeromq: ZeroMqLocation | None = None
    filesystem: FilesystemLocation | None = None

    def __post_init__(self) -> None:
        selected = [name for name in TRANSPORT_KINDS if getattr(self, name) is not None]
        if len(selected) != 1:
            raise ValueError(
                f"DeltaLocation needs exactly one transport, got {sorted(selected)}"
            )

    @property
    def kind(self) -> str:
        return next(name for name in TRANSPORT_KINDS if getattr(self, name) is not None)

    def to_proto(self) -> revision_pb2.DeltaLocation:
        kind = self.kind
        return revision_pb2.DeltaLocation(**{kind: getattr(self, kind).to_proto()})

    @classmethod
    def from_proto(cls, proto: revision_pb2.DeltaLocation) -> DeltaLocation:
        kind = proto.WhichOneof("transport")
        if kind is None:
            raise ValueError("DeltaLocation has no transport set")
        location = {
            "s3": S3Location,
            "zeromq": ZeroMqLocation,
            "filesystem": FilesystemLocation,
        }[kind].from_proto(getattr(proto, kind))
        return cls(**{kind: location})


@dataclass(frozen=True)
class DeltaDescriptor:
    address: int
    length: int
    dtype: str

    def to_proto(self) -> revision_pb2.DeltaDescriptor:
        return revision_pb2.DeltaDescriptor(
            address=self.address, length=self.length, dtype=self.dtype
        )

    @classmethod
    def from_proto(cls, proto: revision_pb2.DeltaDescriptor) -> DeltaDescriptor:
        return cls(address=proto.address, length=proto.length, dtype=proto.dtype)


@dataclass(frozen=True)
class RankDelta:
    """One trainer rank's encoded delta. ``CLEAN`` carries no transfer reference."""

    change_state: ChangeState
    checksum: str | None = None
    location: DeltaLocation | None = None
    delta_descriptor: DeltaDescriptor | None = None

    def to_proto(self) -> revision_pb2.RankDelta:
        return revision_pb2.RankDelta(
            change_state=self.change_state,
            **_present(
                checksum=self.checksum,
                location=self.location.to_proto() if self.location else None,
                delta_descriptor=(
                    self.delta_descriptor.to_proto() if self.delta_descriptor else None
                ),
            ),
        )

    @classmethod
    def from_proto(cls, proto: revision_pb2.RankDelta) -> RankDelta:
        return cls(
            change_state=ChangeState(proto.change_state),
            checksum=_optional(proto, "checksum"),
            location=(
                DeltaLocation.from_proto(proto.location) if proto.HasField("location") else None
            ),
            delta_descriptor=(
                DeltaDescriptor.from_proto(proto.delta_descriptor)
                if proto.HasField("delta_descriptor")
                else None
            ),
        )


@dataclass(frozen=True)
class TensorDescriptor:
    tensor_name: str
    dtype: str
    byte_size: int
    address: int | None = None
    device_id: int | None = None

    def to_proto(self) -> revision_pb2.TensorDescriptor:
        return revision_pb2.TensorDescriptor(
            tensor_name=self.tensor_name,
            dtype=self.dtype,
            byte_size=self.byte_size,
            **_present(address=self.address, device_id=self.device_id),
        )

    @classmethod
    def from_proto(cls, proto: revision_pb2.TensorDescriptor) -> TensorDescriptor:
        return cls(
            tensor_name=proto.tensor_name,
            dtype=proto.dtype,
            byte_size=proto.byte_size,
            address=_optional(proto, "address"),
            device_id=_optional(proto, "device_id"),
        )


@dataclass(frozen=True)
class TensorRegion:
    full_shape: tuple[int, ...]
    global_offset: tuple[int, ...]
    region_shape: tuple[int, ...]
    target_digest: str

    def to_proto(self) -> revision_pb2.TensorRegion:
        return revision_pb2.TensorRegion(
            full_shape=self.full_shape,
            global_offset=self.global_offset,
            region_shape=self.region_shape,
            target_digest=self.target_digest,
        )

    @classmethod
    def from_proto(cls, proto: revision_pb2.TensorRegion) -> TensorRegion:
        return cls(
            full_shape=tuple(proto.full_shape),
            global_offset=tuple(proto.global_offset),
            region_shape=tuple(proto.region_shape),
            target_digest=proto.target_digest,
        )


@dataclass(frozen=True)
class TensorShard:
    change_state: ChangeState
    tensor_descriptor: TensorDescriptor
    tensor_region: TensorRegion

    def to_proto(self) -> revision_pb2.TensorShard:
        return revision_pb2.TensorShard(
            change_state=self.change_state,
            tensor_descriptor=self.tensor_descriptor.to_proto(),
            tensor_region=self.tensor_region.to_proto(),
        )

    @classmethod
    def from_proto(cls, proto: revision_pb2.TensorShard) -> TensorShard:
        return cls(
            change_state=ChangeState(proto.change_state),
            tensor_descriptor=TensorDescriptor.from_proto(proto.tensor_descriptor),
            tensor_region=TensorRegion.from_proto(proto.tensor_region),
        )


@dataclass(frozen=True)
class RevisionRank:
    """One trainer rank's contribution: a rank delta or explicit shards."""

    trainer_rank: int
    producer_id: str
    source_layout_digest: str
    delta: RankDelta | None = None
    shards: tuple[TensorShard, ...] = ()

    def to_proto(self) -> revision_pb2.RevisionRank:
        return revision_pb2.RevisionRank(
            trainer_rank=self.trainer_rank,
            producer_id=self.producer_id,
            source_layout_digest=self.source_layout_digest,
            shards=[shard.to_proto() for shard in self.shards],
            **_present(delta=self.delta.to_proto() if self.delta else None),
        )

    @classmethod
    def from_proto(cls, proto: revision_pb2.RevisionRank) -> RevisionRank:
        return cls(
            trainer_rank=proto.trainer_rank,
            producer_id=proto.producer_id,
            source_layout_digest=proto.source_layout_digest,
            delta=RankDelta.from_proto(proto.delta) if proto.HasField("delta") else None,
            shards=tuple(TensorShard.from_proto(shard) for shard in proto.shards),
        )


@dataclass(frozen=True, kw_only=True)
class RevisionManifest:
    """One immutable, model-global revision keyed by ``(model_id, version)``."""

    model_id: str
    version: str
    base_version: str | None = None
    transfer_method: DeltaTransferMethod
    delta_method: str | None = None
    compression_algorithm: str | None = None
    format_digest: str
    base_digest: str | None = None
    target_digest: str
    ranks: tuple[RevisionRank, ...] = ()

    def to_proto(self) -> revision_pb2.RevisionManifest:
        return revision_pb2.RevisionManifest(
            model_id=self.model_id,
            version=self.version,
            transfer_method=self.transfer_method,
            format_digest=self.format_digest,
            target_digest=self.target_digest,
            ranks=[rank.to_proto() for rank in self.ranks],
            **_present(
                base_version=self.base_version,
                delta_method=self.delta_method,
                compression_algorithm=self.compression_algorithm,
                base_digest=self.base_digest,
            ),
        )

    @classmethod
    def from_proto(cls, proto: revision_pb2.RevisionManifest) -> RevisionManifest:
        return cls(
            model_id=proto.model_id,
            version=proto.version,
            base_version=_optional(proto, "base_version"),
            transfer_method=DeltaTransferMethod(proto.transfer_method),
            delta_method=_optional(proto, "delta_method"),
            compression_algorithm=_optional(proto, "compression_algorithm"),
            format_digest=proto.format_digest,
            base_digest=_optional(proto, "base_digest"),
            target_digest=proto.target_digest,
            ranks=tuple(RevisionRank.from_proto(rank) for rank in proto.ranks),
        )


@dataclass(frozen=True)
class RevisionRecord:
    manifest: RevisionManifest
    state: RevisionLifecycleState
    created_at_unix_ms: int = 0
    state_changed_at_unix_ms: int = 0

    def to_proto(self) -> revision_pb2.RevisionRecord:
        return revision_pb2.RevisionRecord(
            manifest=self.manifest.to_proto(),
            state=self.state,
            created_at_unix_ms=self.created_at_unix_ms,
            state_changed_at_unix_ms=self.state_changed_at_unix_ms,
        )

    @classmethod
    def from_proto(cls, proto: revision_pb2.RevisionRecord) -> RevisionRecord:
        return cls(
            manifest=RevisionManifest.from_proto(proto.manifest),
            state=RevisionLifecycleState(proto.state),
            created_at_unix_ms=proto.created_at_unix_ms,
            state_changed_at_unix_ms=proto.state_changed_at_unix_ms,
        )


@dataclass(frozen=True)
class RevisionSummary:
    model_id: str
    version: str
    state: RevisionLifecycleState
    ready_at_unix_ms: int = 0

    def to_proto(self) -> revision_pb2.RevisionSummary:
        return revision_pb2.RevisionSummary(
            model_id=self.model_id,
            version=self.version,
            state=self.state,
            ready_at_unix_ms=self.ready_at_unix_ms,
        )

    @classmethod
    def from_proto(cls, proto: revision_pb2.RevisionSummary) -> RevisionSummary:
        return cls(
            model_id=proto.model_id,
            version=proto.version,
            state=RevisionLifecycleState(proto.state),
            ready_at_unix_ms=proto.ready_at_unix_ms,
        )


@dataclass(frozen=True)
class RecoveryCandidate:
    """Ordered lineage; a full anchor, when present, is first."""

    kind: RecoveryCandidateKind
    revisions: tuple[RevisionRecord, ...] = ()

    def to_proto(self) -> revision_pb2.RecoveryCandidate:
        return revision_pb2.RecoveryCandidate(
            kind=self.kind,
            revisions=[revision.to_proto() for revision in self.revisions],
        )

    @classmethod
    def from_proto(cls, proto: revision_pb2.RecoveryCandidate) -> RecoveryCandidate:
        return cls(
            kind=RecoveryCandidateKind(proto.kind),
            revisions=tuple(RevisionRecord.from_proto(item) for item in proto.revisions),
        )


@dataclass(frozen=True)
class ReceiverStateRecord:
    model_id: str
    version: str
    receiver_id: str
    state: ReceiverRevisionState
    installed_version: str | None = None
    detail: str = ""
    observed_at_unix_ms: int = 0

    def to_proto(self) -> revision_pb2.ReceiverStateRecord:
        return revision_pb2.ReceiverStateRecord(
            model_id=self.model_id,
            version=self.version,
            receiver_id=self.receiver_id,
            state=self.state,
            detail=self.detail,
            observed_at_unix_ms=self.observed_at_unix_ms,
            **_present(installed_version=self.installed_version),
        )

    @classmethod
    def from_proto(cls, proto: revision_pb2.ReceiverStateRecord) -> ReceiverStateRecord:
        return cls(
            model_id=proto.model_id,
            version=proto.version,
            receiver_id=proto.receiver_id,
            state=ReceiverRevisionState(proto.state),
            installed_version=_optional(proto, "installed_version"),
            detail=proto.detail,
            observed_at_unix_ms=proto.observed_at_unix_ms,
        )
