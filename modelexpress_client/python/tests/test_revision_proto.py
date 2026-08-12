# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import grpc
from google.protobuf import descriptor_pb2

from modelexpress import p2p_pb2, revision_pb2, revision_pb2_grpc


def _file_descriptor_proto() -> descriptor_pb2.FileDescriptorProto:
    descriptor = descriptor_pb2.FileDescriptorProto()
    descriptor.ParseFromString(revision_pb2.DESCRIPTOR.serialized_pb)
    return descriptor


def _message(name: str) -> descriptor_pb2.DescriptorProto:
    return next(
        message
        for message in _file_descriptor_proto().message_type
        if message.name == name
    )


def test_revision_catalog_service_has_exact_public_methods():
    service = revision_pb2.DESCRIPTOR.services_by_name["RevisionCatalogService"]

    assert [method.name for method in service.methods] == [
        "PublishRevision",
        "GetRevision",
        "CommitRevision",
    ]
    assert not any(method.server_streaming for method in service.methods)
    assert not any(method.client_streaming for method in service.methods)


def test_revision_catalog_rpc_types_are_minimal():
    service = revision_pb2.DESCRIPTOR.services_by_name["RevisionCatalogService"]

    assert {
        method.name: (method.input_type.name, method.output_type.name)
        for method in service.methods
    } == {
        "PublishRevision": ("PublishRevisionRequest", "RevisionRecord"),
        "GetRevision": ("GetRevisionRequest", "RevisionRecord"),
        "CommitRevision": ("CommitRevisionRequest", "RevisionRecord"),
    }


def test_revision_proto_has_only_minimal_messages():
    assert [
        message.name for message in _file_descriptor_proto().message_type
    ] == [
        "S3Object",
        "RevisionManifest",
        "RevisionRecord",
        "PublishRevisionRequest",
        "GetRevisionRequest",
        "CommitRevisionRequest",
    ]


def test_manifest_has_exact_minimal_structure():
    expected_fields = {
        "S3Object": ["bucket", "key", "object_version", "checksum"],
        "RevisionManifest": [
            "model_id",
            "target_version",
            "base_version",
            "base_digest",
            "target_digest",
            "format_digest",
            "payload",
        ],
        "RevisionRecord": ["manifest", "state"],
        "PublishRevisionRequest": ["manifest"],
        "GetRevisionRequest": ["model_id", "target_version"],
        "CommitRevisionRequest": ["model_id", "target_version"],
    }

    for message_name, field_names in expected_fields.items():
        message = _message(message_name)
        assert [field.name for field in message.field] == field_names
        assert [field.number for field in message.field] == list(
            range(1, len(field_names) + 1)
        )
        assert not message.reserved_range


def test_revision_state_has_only_ready_and_committed():
    enum = revision_pb2.DESCRIPTOR.enum_types_by_name["RevisionState"]

    assert [(value.name, value.number) for value in enum.values] == [
        ("REVISION_STATE_UNSPECIFIED", 0),
        ("REVISION_STATE_READY", 1),
        ("REVISION_STATE_COMMITTED", 2),
    ]


def test_exact_base_fields_and_s3_object_version_preserve_presence():
    manifest = revision_pb2.RevisionManifest(
        model_id="model",
        target_version="1",
        base_version="0",
        base_digest="sha256:base",
        target_digest="sha256:target",
        format_digest="sha256:format",
        payload=revision_pb2.S3Object(
            bucket="bucket",
            key="model/1/index.json",
            object_version="object-version",
            checksum="crc32c:01020304",
        ),
    )

    assert manifest.HasField("base_version")
    assert manifest.HasField("base_digest")
    assert manifest.HasField("payload")
    assert manifest.payload.HasField("object_version")

    launch = revision_pb2.RevisionManifest(
        model_id="model",
        target_version="0",
        target_digest="sha256:target",
        format_digest="sha256:format",
    )
    assert not launch.HasField("base_version")
    assert not launch.HasField("base_digest")
    assert not launch.HasField("payload")


def test_generated_client_exposes_only_minimal_revision_methods():
    with grpc.insecure_channel("localhost:1") as channel:
        stub = revision_pb2_grpc.RevisionCatalogServiceStub(channel)

    assert callable(stub.PublishRevision)
    assert callable(stub.GetRevision)
    assert callable(stub.CommitRevision)
    assert not hasattr(stub, "ListReadyRevisions")
    assert not hasattr(stub, "GetRecoveryCandidates")
    assert not hasattr(stub, "UpdateReceiverState")
    assert not hasattr(stub, "CommitVersion")


def test_revision_contract_is_independent_from_p2p_service():
    p2p_service = p2p_pb2.DESCRIPTOR.services_by_name["P2pService"]
    revision_methods = {"PublishRevision", "GetRevision", "CommitRevision"}

    assert revision_methods.isdisjoint(method.name for method in p2p_service.methods)
    assert "RevisionRecord" not in p2p_pb2.DESCRIPTOR.message_types_by_name
    assert "RevisionState" not in p2p_pb2.DESCRIPTOR.enum_types_by_name
