# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal revision-catalog DTOs and their exact protobuf mapping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from modelexpress import revision_pb2


class RevisionState(IntEnum):
    UNSPECIFIED = revision_pb2.REVISION_STATE_UNSPECIFIED
    READY = revision_pb2.REVISION_STATE_READY
    COMMITTED = revision_pb2.REVISION_STATE_COMMITTED


@dataclass(frozen=True)
class S3Object:
    bucket: str
    key: str
    checksum: str
    object_version: str | None = None

    def to_proto(self) -> revision_pb2.S3Object:
        fields = {
            "bucket": self.bucket,
            "key": self.key,
            "checksum": self.checksum,
        }
        if self.object_version is not None:
            fields["object_version"] = self.object_version
        return revision_pb2.S3Object(**fields)

    @classmethod
    def from_proto(cls, proto: revision_pb2.S3Object) -> S3Object:
        return cls(
            bucket=proto.bucket,
            key=proto.key,
            checksum=proto.checksum,
            object_version=(
                proto.object_version if proto.HasField("object_version") else None
            ),
        )


@dataclass(frozen=True)
class RevisionManifest:
    model_id: str
    target_version: str
    target_digest: str
    format_digest: str
    base_version: str | None = None
    base_digest: str | None = None
    payload: S3Object | None = None

    def to_proto(self) -> revision_pb2.RevisionManifest:
        fields = {
            "model_id": self.model_id,
            "target_version": self.target_version,
            "target_digest": self.target_digest,
            "format_digest": self.format_digest,
        }
        if self.base_version is not None:
            fields["base_version"] = self.base_version
        if self.base_digest is not None:
            fields["base_digest"] = self.base_digest
        if self.payload is not None:
            fields["payload"] = self.payload.to_proto()
        return revision_pb2.RevisionManifest(**fields)

    @classmethod
    def from_proto(cls, proto: revision_pb2.RevisionManifest) -> RevisionManifest:
        return cls(
            model_id=proto.model_id,
            target_version=proto.target_version,
            target_digest=proto.target_digest,
            format_digest=proto.format_digest,
            base_version=(proto.base_version if proto.HasField("base_version") else None),
            base_digest=(proto.base_digest if proto.HasField("base_digest") else None),
            payload=(S3Object.from_proto(proto.payload) if proto.HasField("payload") else None),
        )


@dataclass(frozen=True)
class RevisionRecord:
    manifest: RevisionManifest
    state: RevisionState

    def to_proto(self) -> revision_pb2.RevisionRecord:
        return revision_pb2.RevisionRecord(
            manifest=self.manifest.to_proto(),
            state=int(self.state),
        )

    @classmethod
    def from_proto(cls, proto: revision_pb2.RevisionRecord) -> RevisionRecord:
        if not proto.HasField("manifest"):
            raise ValueError("revision record is missing manifest")
        return cls(
            manifest=RevisionManifest.from_proto(proto.manifest),
            state=RevisionState(proto.state),
        )
