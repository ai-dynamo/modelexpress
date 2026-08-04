# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable revision-domain types and their exact protobuf mapping."""

from __future__ import annotations

import dataclasses
import re

import pytest

from modelexpress import revision_pb2
from modelexpress.refit import manifest as manifest_module
from modelexpress.refit.manifest import (
    ChangeState,
    DeltaDescriptor,
    DeltaLocation,
    DeltaTransferMethod,
    FilesystemLocation,
    PublicationMode,
    RankDelta,
    ReceiverRevisionState,
    ReceiverStateRecord,
    RecoveryCandidate,
    RecoveryCandidateKind,
    RevisionLifecycleState,
    RevisionManifest,
    RevisionRank,
    RevisionRecord,
    RevisionSummary,
    S3Location,
    TensorDescriptor,
    TensorRegion,
    TensorShard,
    ZeroMqLocation,
)


# modelexpress_common::revision::is_crc32c accepts exactly eight lowercase hex
# digits, and RevisionStateManager::publish validates every manifest, so any
# other spelling of `checksum` is rejected at PublishRevision.
CRC32C_WIRE_FORMAT = re.compile(r"[0-9a-f]{8}")


def _canonical_manifest(**overrides) -> RevisionManifest:
    fields = {
        "model_id": "Qwen/Qwen3-30B-A3B",
        "version": "v2",
        "base_version": "v1",
        "transfer_method": DeltaTransferMethod.CANONICAL,
        "delta_method": "tensor_byte_xor",
        "compression_algorithm": "zstd",
        "format_digest": "sha256:format",
        "base_digest": "sha256:target-v1",
        "target_digest": "sha256:target-v2",
        "ranks": (
            RevisionRank(
                trainer_rank=0,
                producer_id="trainer-0",
                source_layout_digest="sha256:layout",
                delta=RankDelta(
                    change_state=ChangeState.DIRTY,
                    checksum="8e1f2b3c",
                    location=DeltaLocation(
                        s3=S3Location(
                            bucket="mx-delta",
                            key="Qwen3-30B-A3B/v2/root.json",
                            object_version="obj-1",
                        )
                    ),
                ),
            ),
        ),
    }
    fields.update(overrides)
    return RevisionManifest(**fields)


def test_enum_members_map_exactly_onto_the_protobuf_values():
    assert {member.name: int(member) for member in ChangeState} == {
        "CLEAN": revision_pb2.CHANGE_STATE_CLEAN,
        "DIRTY": revision_pb2.CHANGE_STATE_DIRTY,
    }
    assert {member.name: int(member) for member in DeltaTransferMethod} == {
        "CANONICAL": revision_pb2.DELTA_TRANSFER_METHOD_CANONICAL,
        "RANK_LOCAL": revision_pb2.DELTA_TRANSFER_METHOD_RANK_LOCAL,
        "P2P_CPU_RANK": revision_pb2.DELTA_TRANSFER_METHOD_P2P_CPU_RANK,
        "P2P_GPU_SHARD": revision_pb2.DELTA_TRANSFER_METHOD_P2P_GPU_SHARD,
    }
    assert {member.name: int(member) for member in RevisionLifecycleState} == {
        "READY": revision_pb2.REVISION_LIFECYCLE_STATE_READY,
        "COMMITTED": revision_pb2.REVISION_LIFECYCLE_STATE_COMMITTED,
    }
    assert {member.name: int(member) for member in ReceiverRevisionState} == {
        "BYTES_RECEIVED": revision_pb2.RECEIVER_REVISION_STATE_BYTES_RECEIVED,
        "VERIFIED": revision_pb2.RECEIVER_REVISION_STATE_VERIFIED,
        "FAILED": revision_pb2.RECEIVER_REVISION_STATE_FAILED,
        "POISONED": revision_pb2.RECEIVER_REVISION_STATE_POISONED,
    }
    assert {member.name: int(member) for member in RecoveryCandidateKind} == {
        "DIRECT_DELTA": revision_pb2.RECOVERY_CANDIDATE_KIND_DIRECT_DELTA,
        "DELTA_REPLAY": revision_pb2.RECOVERY_CANDIDATE_KIND_DELTA_REPLAY,
        "FULL_ANCHOR_REPLAY": revision_pb2.RECOVERY_CANDIDATE_KIND_FULL_ANCHOR_REPLAY,
        "FULL_TARGET": revision_pb2.RECOVERY_CANDIDATE_KIND_FULL_TARGET,
    }
    assert {member.name: int(member) for member in PublicationMode} == {
        "BLOCK": revision_pb2.PUBLICATION_MODE_BLOCK,
        "ASYNC": revision_pb2.PUBLICATION_MODE_ASYNC,
    }


def test_canonical_manifest_round_trips_through_protobuf():
    domain = _canonical_manifest()

    proto = domain.to_proto()

    assert proto.transfer_method == revision_pb2.DELTA_TRANSFER_METHOD_CANONICAL
    assert len(proto.ranks) == 1
    assert proto.ranks[0].trainer_rank == 0
    assert proto.ranks[0].delta.location.s3.bucket == "mx-delta"
    assert proto.ranks[0].delta.location.s3.HasField("object_version")
    assert not proto.ranks[0].delta.HasField("delta_descriptor")
    assert not proto.ranks[0].shards
    assert RevisionManifest.from_proto(proto) == domain


def test_absent_optional_manifest_fields_stay_absent_on_the_wire():
    domain = _canonical_manifest(
        base_version=None,
        base_digest=None,
        delta_method=None,
        compression_algorithm=None,
    )

    proto = domain.to_proto()

    assert not proto.HasField("base_version")
    assert not proto.HasField("base_digest")
    assert not proto.HasField("delta_method")
    assert not proto.HasField("compression_algorithm")
    assert RevisionManifest.from_proto(proto) == domain


def test_clean_rank_delta_carries_no_transfer_reference():
    domain = _canonical_manifest(
        ranks=(
            RevisionRank(
                trainer_rank=0,
                producer_id="trainer-0",
                source_layout_digest="sha256:layout",
                delta=RankDelta(change_state=ChangeState.CLEAN),
            ),
        )
    )

    proto = domain.to_proto()

    assert not proto.ranks[0].delta.HasField("checksum")
    assert not proto.ranks[0].delta.HasField("location")
    assert not proto.ranks[0].delta.HasField("delta_descriptor")
    assert RevisionManifest.from_proto(proto) == domain


def test_delta_location_holds_exactly_one_transport():
    assert DeltaLocation(filesystem=FilesystemLocation(path="/mxdelta/v2")).filesystem
    assert DeltaLocation(zeromq=ZeroMqLocation(endpoint="tcp://h:5555", payload_id="p")).zeromq

    with pytest.raises(ValueError):
        DeltaLocation()
    with pytest.raises(ValueError):
        DeltaLocation(
            s3=S3Location(bucket="b", key="k"),
            filesystem=FilesystemLocation(path="/tmp/x"),
        )


def test_optional_s3_object_version_presence_round_trips():
    without_version = DeltaLocation(s3=S3Location(bucket="b", key="k"))

    proto = without_version.to_proto()

    assert not proto.s3.HasField("object_version")
    assert DeltaLocation.from_proto(proto) == without_version


def test_gpu_shard_manifest_round_trips_with_descriptor_presence():
    domain = _canonical_manifest(
        transfer_method=DeltaTransferMethod.P2P_GPU_SHARD,
        base_version=None,
        base_digest=None,
        delta_method=None,
        compression_algorithm=None,
        ranks=(
            RevisionRank(
                trainer_rank=3,
                producer_id="trainer-3",
                source_layout_digest="sha256:layout",
                shards=(
                    TensorShard(
                        change_state=ChangeState.DIRTY,
                        tensor_descriptor=TensorDescriptor(
                            tensor_name="model.layers.0.mlp.gate_proj.weight",
                            dtype="bfloat16",
                            byte_size=4096,
                            address=140_737_488_355_328,
                            device_id=3,
                        ),
                        tensor_region=TensorRegion(
                            full_shape=(2048, 1024),
                            global_offset=(0, 0),
                            region_shape=(512, 1024),
                            target_digest="sha256:shard",
                        ),
                    ),
                    TensorShard(
                        change_state=ChangeState.CLEAN,
                        tensor_descriptor=TensorDescriptor(
                            tensor_name="model.layers.0.mlp.up_proj.weight",
                            dtype="bfloat16",
                            byte_size=4096,
                        ),
                        tensor_region=TensorRegion(
                            full_shape=(2048, 1024),
                            global_offset=(512, 0),
                            region_shape=(512, 1024),
                            target_digest="sha256:shard-clean",
                        ),
                    ),
                ),
            ),
        ),
    )

    proto = domain.to_proto()

    assert not proto.ranks[0].HasField("delta")
    assert proto.ranks[0].shards[0].tensor_descriptor.HasField("address")
    assert proto.ranks[0].shards[0].tensor_descriptor.HasField("device_id")
    assert not proto.ranks[0].shards[1].tensor_descriptor.HasField("address")
    assert not proto.ranks[0].shards[1].tensor_descriptor.HasField("device_id")
    assert RevisionManifest.from_proto(proto) == domain


def test_dirty_delta_checksums_use_the_crc32c_wire_format():
    canonical = _canonical_manifest().ranks[0].delta
    cpu_direct = RankDelta(
        change_state=ChangeState.DIRTY,
        checksum="1234abcd",
        delta_descriptor=DeltaDescriptor(address=4096, length=8192, dtype="uint8"),
    )

    for delta in (canonical, cpu_direct):
        assert CRC32C_WIRE_FORMAT.fullmatch(delta.checksum)
        assert CRC32C_WIRE_FORMAT.fullmatch(delta.to_proto().checksum)


def test_cpu_direct_rank_delta_descriptor_round_trips():
    delta = RankDelta(
        change_state=ChangeState.DIRTY,
        checksum="1234abcd",
        delta_descriptor=DeltaDescriptor(address=4096, length=8192, dtype="uint8"),
    )

    proto = delta.to_proto()

    assert not proto.HasField("location")
    assert proto.delta_descriptor.length == 8192
    assert RankDelta.from_proto(proto) == delta


def test_revision_record_and_summary_round_trip():
    record = RevisionRecord(
        manifest=_canonical_manifest(),
        state=RevisionLifecycleState.COMMITTED,
        created_at_unix_ms=1_700_000_000_000,
        state_changed_at_unix_ms=1_700_000_000_500,
    )
    summary = RevisionSummary(
        model_id="Qwen/Qwen3-30B-A3B",
        version="v2",
        state=RevisionLifecycleState.READY,
        ready_at_unix_ms=1_700_000_000_000,
    )

    assert RevisionRecord.from_proto(record.to_proto()) == record
    assert RevisionSummary.from_proto(summary.to_proto()) == summary


def test_recovery_candidate_and_receiver_state_round_trip():
    candidate = RecoveryCandidate(
        kind=RecoveryCandidateKind.DELTA_REPLAY,
        revisions=(
            RevisionRecord(
                manifest=_canonical_manifest(version="v2"),
                state=RevisionLifecycleState.READY,
                created_at_unix_ms=1,
                state_changed_at_unix_ms=2,
            ),
            RevisionRecord(
                manifest=_canonical_manifest(version="v3", base_version="v2"),
                state=RevisionLifecycleState.READY,
                created_at_unix_ms=3,
                state_changed_at_unix_ms=4,
            ),
        ),
    )
    receiver = ReceiverStateRecord(
        model_id="Qwen/Qwen3-30B-A3B",
        version="v2",
        receiver_id="rollout-tp0",
        state=ReceiverRevisionState.VERIFIED,
        installed_version="v2",
        detail="device verified",
        observed_at_unix_ms=1_700_000_000_000,
    )

    assert RecoveryCandidate.from_proto(candidate.to_proto()) == candidate
    assert ReceiverStateRecord.from_proto(receiver.to_proto()) == receiver
    assert receiver.to_proto().HasField("installed_version")


def test_receiver_state_record_keeps_installed_version_optional():
    receiver = ReceiverStateRecord(
        model_id="Qwen/Qwen3-30B-A3B",
        version="v2",
        receiver_id="rollout-tp0",
        state=ReceiverRevisionState.FAILED,
        detail="checksum mismatch",
    )

    proto = receiver.to_proto()

    assert not proto.HasField("installed_version")
    assert ReceiverStateRecord.from_proto(proto) == receiver


def test_unspecified_wire_enum_values_are_rejected():
    proto = _canonical_manifest().to_proto()
    proto.transfer_method = revision_pb2.DELTA_TRANSFER_METHOD_UNSPECIFIED

    with pytest.raises(ValueError):
        RevisionManifest.from_proto(proto)

    record = revision_pb2.RevisionRecord(
        manifest=_canonical_manifest().to_proto(),
        state=revision_pb2.REVISION_LIFECYCLE_STATE_UNSPECIFIED,
    )
    with pytest.raises(ValueError):
        RevisionRecord.from_proto(record)


def test_revision_domain_records_are_frozen():
    domain = _canonical_manifest()

    for record in (domain, domain.ranks[0], domain.ranks[0].delta):
        assert dataclasses.is_dataclass(record)
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.model_id = "other"


def test_legacy_payload_and_generation_types_are_absent():
    for removed in (
        "PayloadDescriptor",
        "RevisionPayload",
        "PayloadKind",
        "PayloadReference",
        "PublisherGeneration",
        "ReceiverGeneration",
    ):
        assert not hasattr(manifest_module, removed)

    manifest_fields = {field.name for field in dataclasses.fields(RevisionManifest)}
    assert "payloads" not in manifest_fields
    assert "generation" not in manifest_fields
