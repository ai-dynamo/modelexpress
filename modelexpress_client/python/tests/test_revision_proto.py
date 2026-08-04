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


def _enum(name: str) -> descriptor_pb2.EnumDescriptorProto:
    return next(item for item in _file_descriptor_proto().enum_type if item.name == name)


def test_revision_catalog_service_has_exact_public_methods():
    service = revision_pb2.DESCRIPTOR.services_by_name["RevisionCatalogService"]

    assert [method.name for method in service.methods] == [
        "PublishRevision",
        "GetRevision",
        "ListReadyRevisions",
        "GetRecoveryCandidates",
        "UpdateReceiverState",
        "CommitVersion",
    ]
    assert not any(method.server_streaming for method in service.methods)
    assert not any(method.client_streaming for method in service.methods)


def test_revision_catalog_rpcs_and_states_are_absent_from_p2p_service():
    p2p_service = p2p_pb2.DESCRIPTOR.services_by_name["P2pService"]
    revision_methods = {
        "PublishRevision",
        "GetRevision",
        "ListReadyRevisions",
        "GetRecoveryCandidates",
        "UpdateReceiverState",
        "CommitVersion",
    }

    assert revision_methods.isdisjoint(method.name for method in p2p_service.methods)
    assert "RevisionRecord" not in p2p_pb2.DESCRIPTOR.message_types_by_name
    assert "RevisionLifecycleState" not in p2p_pb2.DESCRIPTOR.enum_types_by_name


def test_generated_revision_catalog_client_exposes_public_methods():
    with grpc.insecure_channel("localhost:1") as channel:
        stub = revision_pb2_grpc.RevisionCatalogServiceStub(channel)

    assert all(
        callable(getattr(stub, method))
        for method in (
            "PublishRevision",
            "GetRevision",
            "ListReadyRevisions",
            "GetRecoveryCandidates",
            "UpdateReceiverState",
            "CommitVersion",
        )
    )


def test_watch_messages_and_old_receiver_rpc_name_are_absent():
    message_names = {
        message.name for message in _file_descriptor_proto().message_type
    }

    assert "WatchRevisionEventsRequest" not in message_names
    assert "RevisionEvent" not in message_names
    assert "ReportReceiverStateRequest" not in message_names
    assert "ReportReceiverStateResponse" not in message_names
    assert "UpdateReceiverStateRequest" in message_names
    assert "UpdateReceiverStateResponse" in message_names


def test_manifest_preserves_exact_base_and_integrity_presence():
    manifest = revision_pb2.RevisionManifest(
        model_id="model",
        version="1",
        base_version="0",
        base_digest="sha256:target-0",
        delta_method="xor",
        compression_algorithm="zstd",
        format_digest="sha256:format",
        target_digest="sha256:target-1",
    )

    assert manifest.HasField("base_version")
    assert manifest.HasField("base_digest")
    assert manifest.HasField("delta_method")
    assert manifest.HasField("compression_algorithm")
    manifest.ClearField("base_version")
    manifest.ClearField("base_digest")
    manifest.ClearField("delta_method")
    manifest.ClearField("compression_algorithm")
    assert not manifest.HasField("base_version")
    assert not manifest.HasField("base_digest")
    assert not manifest.HasField("delta_method")
    assert not manifest.HasField("compression_algorithm")


def test_manifest_has_exact_rank_oriented_structure():
    expected_fields = {
        "RevisionManifest": [
            "model_id",
            "version",
            "base_version",
            "transfer_method",
            "delta_method",
            "compression_algorithm",
            "format_digest",
            "base_digest",
            "target_digest",
            "ranks",
        ],
        "RevisionRank": [
            "trainer_rank",
            "producer_id",
            "source_layout_digest",
            "delta",
            "shards",
        ],
        "RankDelta": ["change_state", "checksum", "location", "delta_descriptor"],
        "DeltaLocation": ["s3", "zeromq", "filesystem"],
        "S3Location": ["bucket", "key", "object_version"],
        "ZeroMqLocation": ["endpoint", "payload_id"],
        "FilesystemLocation": ["path"],
        "DeltaDescriptor": ["address", "length", "dtype"],
        "TensorShard": ["change_state", "tensor_descriptor", "tensor_region"],
        "TensorDescriptor": [
            "tensor_name",
            "dtype",
            "byte_size",
            "address",
            "device_id",
        ],
        "TensorRegion": ["full_shape", "global_offset", "region_shape", "target_digest"],
    }

    for message_name, field_names in expected_fields.items():
        message = _message(message_name)
        assert [field.name for field in message.field] == field_names
        assert [field.number for field in message.field] == list(
            range(1, len(field_names) + 1)
        )
        assert not message.reserved_range


def test_canonical_manifest_uses_one_rank_zero_delta():
    manifest = revision_pb2.RevisionManifest(
        model_id="model",
        version="1",
        base_version="0",
        transfer_method=revision_pb2.DELTA_TRANSFER_METHOD_CANONICAL,
        delta_method="xor",
        compression_algorithm="zstd",
        format_digest="sha256:format",
        base_digest="sha256:target-0",
        target_digest="sha256:target-1",
        ranks=[
            revision_pb2.RevisionRank(
                trainer_rank=0,
                producer_id="publisher-0",
                source_layout_digest="sha256:layout",
                delta=revision_pb2.RankDelta(
                    change_state=revision_pb2.CHANGE_STATE_DIRTY,
                    checksum="a1b2c3d4",
                    location=revision_pb2.DeltaLocation(
                        s3=revision_pb2.S3Location(
                            bucket="bucket",
                            key="models/policy/versions/1/canonical/index.json",
                            object_version="version-id",
                        )
                    ),
                ),
            )
        ],
    )

    assert manifest.transfer_method == revision_pb2.DELTA_TRANSFER_METHOD_CANONICAL
    assert len(manifest.ranks) == 1
    assert manifest.ranks[0].trainer_rank == 0
    assert manifest.ranks[0].HasField("delta")
    assert manifest.ranks[0].delta.location.WhichOneof("transport") == "s3"
    assert not manifest.ranks[0].shards


def test_transfer_and_change_state_enums_match_stable_values():
    expected = {
        "ChangeState": [
            "CHANGE_STATE_UNSPECIFIED",
            "CHANGE_STATE_CLEAN",
            "CHANGE_STATE_DIRTY",
        ],
        "DeltaTransferMethod": [
            "DELTA_TRANSFER_METHOD_UNSPECIFIED",
            "DELTA_TRANSFER_METHOD_CANONICAL",
            "DELTA_TRANSFER_METHOD_RANK_LOCAL",
            "DELTA_TRANSFER_METHOD_P2P_CPU_RANK",
            "DELTA_TRANSFER_METHOD_P2P_GPU_SHARD",
        ],
    }

    for enum_name, names in expected.items():
        enum = _enum(enum_name)
        assert [value.name for value in enum.value] == names
        assert [value.number for value in enum.value] == list(range(len(names)))


def test_legacy_detached_payload_contract_is_absent():
    message_names = {
        message.name for message in _file_descriptor_proto().message_type
    }
    enum_names = {enum.name for enum in _file_descriptor_proto().enum_type}

    assert message_names.isdisjoint(
        {"PayloadDescriptor", "PayloadReference", "RevisionPayload"}
    )
    assert "PayloadKind" not in enum_names


def test_publication_mode_has_only_block_and_async():
    publication_mode = _enum("PublicationMode")

    assert [(value.name, value.number) for value in publication_mode.value] == [
        ("PUBLICATION_MODE_BLOCK", 0),
        ("PUBLICATION_MODE_ASYNC", 1),
    ]
    request = revision_pb2.PublishRevisionRequest(
        publication_mode=revision_pb2.PUBLICATION_MODE_BLOCK
    )
    assert request.HasField("publication_mode")


def test_mutation_requests_exclude_deferred_idempotency_and_generation_fields():
    expected_fields = {
        "PublishRevisionRequest": ["manifest", "publisher_id", "publication_mode"],
        "UpdateReceiverStateRequest": [
            "model_id",
            "version",
            "receiver_id",
            "state",
            "installed_version",
            "detail",
        ],
        "ReceiverStateRecord": [
            "model_id",
            "version",
            "receiver_id",
            "state",
            "installed_version",
            "detail",
            "observed_at_unix_ms",
        ],
        "CommitVersionRequest": ["model_id", "version"],
    }

    for message_name, field_names in expected_fields.items():
        message = _message(message_name)
        assert [field.name for field in message.field] == field_names
        assert [field.number for field in message.field] == list(
            range(1, len(field_names) + 1)
        )


def test_state_enums_keep_unspecified_as_invalid_wire_default():
    expected = {
        "RevisionLifecycleState": [
            "REVISION_LIFECYCLE_STATE_UNSPECIFIED",
            "REVISION_LIFECYCLE_STATE_READY",
            "REVISION_LIFECYCLE_STATE_COMMITTED",
        ],
        "RecoveryCandidateKind": [
            "RECOVERY_CANDIDATE_KIND_UNSPECIFIED",
            "RECOVERY_CANDIDATE_KIND_DIRECT_DELTA",
            "RECOVERY_CANDIDATE_KIND_DELTA_REPLAY",
            "RECOVERY_CANDIDATE_KIND_FULL_ANCHOR_REPLAY",
            "RECOVERY_CANDIDATE_KIND_FULL_TARGET",
        ],
        "ReceiverRevisionState": [
            "RECEIVER_REVISION_STATE_UNSPECIFIED",
            "RECEIVER_REVISION_STATE_BYTES_RECEIVED",
            "RECEIVER_REVISION_STATE_VERIFIED",
            "RECEIVER_REVISION_STATE_FAILED",
            "RECEIVER_REVISION_STATE_POISONED",
        ],
    }

    for enum_name, names in expected.items():
        enum = _enum(enum_name)
        assert [value.name for value in enum.value] == names
        assert [value.number for value in enum.value] == list(range(len(names)))
        assert not enum.reserved_range


def test_ready_revision_pagination_uses_opaque_page_tokens():
    request = revision_pb2.ListReadyRevisionsRequest(
        model_id="model", page_token="opaque-token", limit=50
    )
    response = revision_pb2.ListReadyRevisionsResponse(
        next_page_token="next-opaque-token"
    )

    assert request.HasField("page_token")
    assert request.page_token == "opaque-token"
    assert response.next_page_token == "next-opaque-token"


def test_recovery_candidate_pagination_uses_opaque_page_tokens():
    request = revision_pb2.GetRecoveryCandidatesRequest(
        model_id="model",
        target_version="10",
        page_token="opaque-token",
        limit=20,
    )
    response = revision_pb2.GetRecoveryCandidatesResponse(
        next_page_token="next-opaque-token"
    )

    assert request.HasField("page_token")
    assert request.page_token == "opaque-token"
    assert request.limit == 20
    assert response.next_page_token == "next-opaque-token"
