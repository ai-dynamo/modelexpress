# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses

from modelexpress import revision_pb2
from modelexpress.refit import (
    PublisherConfig,
    ReceiverRevisionState,
    ReceiverStatus,
    S3Config,
)

MODEL_ID = "Qwen/Qwen3-30B-A3B"


def test_public_config_contains_only_consumed_s3_and_publisher_values():
    assert [field.name for field in dataclasses.fields(S3Config)] == [
        "bucket",
        "prefix",
        "endpoint_url",
        "region_name",
    ]
    assert [field.name for field in dataclasses.fields(PublisherConfig)] == [
        "model_id",
        "catalog_endpoint",
        "s3",
    ]


def test_unconsumed_lifecycle_api_is_not_public():
    import modelexpress.refit as refit

    for name in (
        "PreparedUpdate",
        "PublicationMode",
        "PublishResult",
        "PublisherProtocol",
        "PublisherStatus",
        "ReceiverConfig",
        "ReceiverProtocol",
        "normalize_layer_scope",
    ):
        assert not hasattr(refit, name)
    assert "PublicationMode" not in revision_pb2.DESCRIPTOR.enum_types_by_name


def test_receiver_outcomes_consumed_by_sglang_remain_local():
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


def test_generic_transport_and_adapter_frameworks_are_not_public():
    import modelexpress.refit as refit

    for name in (
        "CompressionAlgorithm",
        "DeltaCodec",
        "DeltaTransferMethod",
        "EngineAdapter",
        "LoadResult",
        "RecoveryStore",
        "RecoveryStoreConfig",
        "SourceAdapter",
        "TransportAdapter",
        "TransportConfig",
        "TransportKind",
    ):
        assert not hasattr(refit, name)
