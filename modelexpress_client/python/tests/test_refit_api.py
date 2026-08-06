# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import pytest

from modelexpress import revision_pb2
from modelexpress.refit import (
    PreparedUpdate,
    PublicationMode,
    PublisherConfig,
    PublisherProtocol,
    ReceiverConfig,
    ReceiverProtocol,
    ReceiverRevisionState,
    ReceiverStatus,
    S3Config,
    normalize_layer_scope,
)

MODEL_ID = "Qwen/Qwen3-30B-A3B"
S3 = S3Config(bucket="mx-delta", prefix="qwen3")


def test_publisher_is_s3_only_and_blocking_by_default():
    config = PublisherConfig(
        model_id=MODEL_ID,
        catalog_endpoint="http://mx:8001",
        s3=S3,
    )

    assert [field.name for field in dataclasses.fields(config)] == [
        "model_id",
        "catalog_endpoint",
        "s3",
        "publication_mode",
    ]
    assert config.publication_mode is PublicationMode.BLOCK
    assert (
        "publication_mode"
        not in revision_pb2.PublishRevisionRequest.DESCRIPTOR.fields_by_name
    )


def test_s3_configs_expose_locations_but_no_transport_or_credential_choices():
    assert [field.name for field in dataclasses.fields(S3Config)] == [
        "bucket",
        "prefix",
        "endpoint_url",
        "region_name",
    ]
    assert [field.name for field in dataclasses.fields(ReceiverConfig)] == [
        "model_id",
        "catalog_endpoint",
        "s3",
    ]


def test_v0_has_no_generic_transport_recovery_or_codec_knobs():
    import modelexpress.refit as refit

    for name in (
        "DeltaTransferMethod",
        "TransportAdapter",
        "TransportConfig",
        "TransportKind",
        "RecoveryStoreConfig",
        "RecoveryStore",
        "DeltaCodec",
        "CompressionAlgorithm",
    ):
        assert not hasattr(refit, name)


def test_publication_mode_is_block_only_client_behavior():
    assert [mode.value for mode in PublicationMode] == ["block"]
    assert "PublicationMode" not in revision_pb2.DESCRIPTOR.enum_types_by_name


def test_receiver_outcomes_are_runtime_local_only():
    assert [state.value for state in ReceiverRevisionState] == [
        "bytes_received",
        "verified",
        "failed",
        "poisoned",
    ]
    assert "ReceiverRevisionState" not in revision_pb2.DESCRIPTOR.enum_types_by_name

    def status(state):
        return ReceiverStatus(
            receiver_id="rollout-tp0",
            model_id=MODEL_ID,
            installed_version="1",
            state=state,
        )

    assert status(ReceiverRevisionState.POISONED).recovery_required is True
    assert status(ReceiverRevisionState.VERIFIED).recovery_required is False
    assert status(None).recovery_required is False


def test_prepared_update_keeps_exact_mutation_boundary_identity():
    update = PreparedUpdate(
        model_id=MODEL_ID,
        base_version="0",
        base_digest="sha256:target-0",
        target_version="1",
        target_digest="sha256:target-1",
        format_digest="sha256:format",
        receiver_incarnation="receiver-1",
        model_generation=0,
        layer_scope=("decoder.0", "decoder.1"),
    )

    assert update.layer_scope == ("decoder.0", "decoder.1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        update.target_version = "2"


def test_layer_scope_is_deterministic():
    assert normalize_layer_scope(None) is None
    assert normalize_layer_scope(["b", "a", "b"]) == ("a", "b")


def test_publisher_and_receiver_contracts_remain_structural_protocols():
    assert getattr(PublisherProtocol, "_is_protocol", False)
    assert getattr(ReceiverProtocol, "_is_protocol", False)
    assert not hasattr(ReceiverProtocol, "recover")


def test_generic_engine_adapter_frameworks_are_not_public():
    import modelexpress.refit as refit

    for name in (
        "EngineAdapter",
        "LoadResult",
        "SourceAdapter",
    ):
        assert not hasattr(refit, name)
