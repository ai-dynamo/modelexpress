# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses

from modelexpress.refit.manifest import (
    RevisionManifest,
    RevisionRecord,
    RevisionState,
    S3Object,
)


def target_manifest() -> RevisionManifest:
    return RevisionManifest(
        model_id="model",
        target_version="1",
        base_version="0",
        base_digest="sha256:target-0",
        target_digest="sha256:target-1",
        format_digest="sha256:format",
        payload=S3Object(
            bucket="bucket",
            key="model/1/index.json",
            object_version="object-version",
            checksum="crc32c:01020304",
        ),
    )


def test_minimal_manifest_round_trips_through_protobuf():
    manifest = target_manifest()
    proto = manifest.to_proto()

    assert proto.model_id == "model"
    assert proto.target_version == "1"
    assert proto.HasField("base_version")
    assert proto.HasField("base_digest")
    assert proto.HasField("payload")
    assert proto.payload.HasField("object_version")
    assert RevisionManifest.from_proto(proto) == manifest


def test_launch_manifest_preserves_absence():
    manifest = RevisionManifest(
        model_id="model",
        target_version="0",
        target_digest="sha256:target-0",
        format_digest="sha256:format",
    )
    proto = manifest.to_proto()

    assert not proto.HasField("base_version")
    assert not proto.HasField("base_digest")
    assert not proto.HasField("payload")
    assert RevisionManifest.from_proto(proto) == manifest


def test_revision_record_round_trips_and_is_frozen():
    record = RevisionRecord(manifest=target_manifest(), state=RevisionState.READY)

    assert RevisionRecord.from_proto(record.to_proto()) == record
    try:
        record.state = RevisionState.COMMITTED
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("RevisionRecord must be frozen")


def test_revision_state_matches_wire_values():
    assert [(state.name, state.value) for state in RevisionState] == [
        ("UNSPECIFIED", 0),
        ("READY", 1),
        ("COMMITTED", 2),
    ]
