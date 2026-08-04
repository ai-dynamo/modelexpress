# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen public refit contract: configuration, results, and lifecycle protocols."""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from modelexpress import refit, revision_pb2
from modelexpress.refit import api as api_module
from modelexpress.refit.api import (
    PreparedUpdate,
    Publisher,
    PublisherConfig,
    PublisherStatus,
    PublishResult,
    Receiver,
    ReceiverConfig,
    ReceiverStatus,
    RecoveryStoreConfig,
    TransportConfig,
    TransportKind,
    WeightUpdateResult,
    normalize_layer_scope,
)
from modelexpress.refit.manifest import (
    DeltaTransferMethod,
    PublicationMode,
    ReceiverRevisionState,
    RevisionLifecycleState,
)

MODEL_ID = "Qwen/Qwen3-30B-A3B"
S3 = TransportConfig(kind=TransportKind.S3, root_uri="s3://mx-delta/qwen3")


def _signature(function) -> tuple[tuple[str, str, object], ...]:
    return tuple(
        (name, parameter.kind.name, parameter.default)
        for name, parameter in inspect.signature(function).parameters.items()
    )


EMPTY = inspect.Parameter.empty


def _public_methods(protocol) -> tuple[str, ...]:
    return tuple(
        name
        for name, member in vars(protocol).items()
        if not name.startswith("_") and inspect.isfunction(member)
    )


def test_publisher_config_matches_the_design_fields_and_defaults():
    config = PublisherConfig(model_id=MODEL_ID, catalog_endpoint="http://mx:8001", transport=S3)

    assert [field.name for field in dataclasses.fields(PublisherConfig)] == [
        "model_id",
        "catalog_endpoint",
        "transport",
        "delta_transfer_method",
        "recovery_store",
        "delta_method",
        "compression_algorithm",
        "full_anchor_interval",
        "publication_mode",
    ]
    assert config.delta_transfer_method is DeltaTransferMethod.RANK_LOCAL
    assert config.publication_mode is PublicationMode.BLOCK
    assert config.recovery_store is None
    assert config.delta_method is None
    assert config.compression_algorithm is None
    assert config.full_anchor_interval is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.model_id = "other"


def test_receiver_config_matches_the_design_fields_and_defaults():
    config = ReceiverConfig(model_id=MODEL_ID, catalog_endpoint="http://mx:8001", transport=S3)

    assert [field.name for field in dataclasses.fields(ReceiverConfig)] == [
        "model_id",
        "catalog_endpoint",
        "transport",
        "delta_transfer_method",
        "recovery_store",
        "delta_method",
        "compression_algorithm",
        "max_delta_replay_length",
    ]
    assert config.delta_transfer_method is DeltaTransferMethod.RANK_LOCAL
    assert config.max_delta_replay_length is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.transport = S3


def test_delta_and_compression_stay_wire_strings():
    config = PublisherConfig(
        model_id=MODEL_ID,
        catalog_endpoint="http://mx:8001",
        transport=S3,
        delta_method="tensor_byte_xor",
        compression_algorithm="zstd",
    )

    assert config.delta_method == "tensor_byte_xor"
    assert not hasattr(api_module, "DeltaMethod")
    assert not hasattr(api_module, "CompressionAlgorithm")


def test_publication_mode_exposes_only_block_and_async():
    assert [member.name for member in PublicationMode] == ["BLOCK", "ASYNC"]


def test_transport_and_recovery_storage_are_distinct_config_types():
    recovery = RecoveryStoreConfig(kind=TransportKind.S3, root_uri="s3://mx-recovery/qwen3")

    assert TransportConfig is not RecoveryStoreConfig
    assert not isinstance(recovery, TransportConfig)
    assert dataclasses.is_dataclass(recovery)
    with pytest.raises(dataclasses.FrozenInstanceError):
        recovery.root_uri = "s3://other"

    for data_plane in ("publish", "fetch", "get", "put", "upload", "download", "read", "write"):
        assert not hasattr(S3, data_plane)
        assert not hasattr(recovery, data_plane)


def test_transport_kinds_mirror_the_wire_location_oneof():
    oneof = revision_pb2.DeltaLocation.DESCRIPTOR.oneofs_by_name["transport"]

    assert [member.value for member in TransportKind] == [field.name for field in oneof.fields]


def test_publisher_protocol_has_the_exact_design_signatures():
    assert _public_methods(Publisher) == ("initialize", "publish_version", "status", "deregister")
    assert _signature(Publisher.initialize) == (
        ("self", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("config", "POSITIONAL_OR_KEYWORD", EMPTY),
    )
    assert _signature(Publisher.publish_version) == (
        ("self", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("version", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("layers", "POSITIONAL_OR_KEYWORD", None),
        ("base_version", "KEYWORD_ONLY", None),
    )
    assert _signature(Publisher.status) == (("self", "POSITIONAL_OR_KEYWORD", EMPTY),)
    assert _signature(Publisher.deregister) == (("self", "POSITIONAL_OR_KEYWORD", EMPTY),)
    assert Publisher.initialize.__annotations__ == {"config": "PublisherConfig", "return": "None"}
    assert Publisher.publish_version.__annotations__["return"] == "PublishResult"
    assert Publisher.status.__annotations__["return"] == "PublisherStatus"


def test_receiver_protocol_has_the_exact_design_signatures():
    assert _public_methods(Receiver) == (
        "initialize",
        "start_weight_update",
        "update_weights",
        "recover",
        "status",
    )
    assert _signature(Receiver.start_weight_update) == (
        ("self", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("version", "POSITIONAL_OR_KEYWORD", EMPTY),
    )
    assert _signature(Receiver.update_weights) == (
        ("self", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("layers", "POSITIONAL_OR_KEYWORD", None),
    )
    assert _signature(Receiver.recover) == (
        ("self", "POSITIONAL_OR_KEYWORD", EMPTY),
        ("version", "POSITIONAL_OR_KEYWORD", EMPTY),
    )
    assert Receiver.start_weight_update.__annotations__["return"] == "None"
    assert Receiver.update_weights.__annotations__["return"] == "WeightUpdateResult"
    assert Receiver.recover.__annotations__["return"] == "WeightUpdateResult"
    assert Receiver.status.__annotations__["return"] == "ReceiverStatus"


def test_lifecycle_protocols_are_structural_and_unimplemented_in_this_package():
    class ForeignReceiver:
        """An engine-native receiver that inherits nothing from ModelExpress."""

        def initialize(self, config) -> None: ...

        def start_weight_update(self, version) -> None: ...

        def update_weights(self, layers=None): ...

        def recover(self, version): ...

        def status(self): ...

    assert Publisher._is_protocol is True
    assert Receiver._is_protocol is True
    assert isinstance(ForeignReceiver(), Receiver)
    assert not issubclass(ForeignReceiver, Receiver.__mro__[1])

    with pytest.raises(TypeError):
        Receiver()


def test_prepared_update_freezes_the_exact_installation_identity():
    prepared = PreparedUpdate(
        model_id=MODEL_ID,
        base_version="v1",
        base_digest="sha256:target-v1",
        target_version="v2",
        target_digest="sha256:target-v2",
        format_digest="sha256:format",
        receiver_incarnation="rollout-tp0-7f3a",
        model_generation=4,
    )

    assert [field.name for field in dataclasses.fields(PreparedUpdate)] == [
        "model_id",
        "base_version",
        "base_digest",
        "target_version",
        "target_digest",
        "format_digest",
        "receiver_incarnation",
        "model_generation",
        "layer_scope",
    ]
    assert prepared.layer_scope is None
    assert prepared == dataclasses.replace(prepared)
    assert prepared != dataclasses.replace(prepared, base_version="v0")
    with pytest.raises(dataclasses.FrozenInstanceError):
        prepared.target_version = "v3"


def test_prepared_update_requires_a_normalized_layer_scope():
    def build(layer_scope):
        return PreparedUpdate(
            model_id=MODEL_ID,
            base_version="v1",
            base_digest="sha256:target-v1",
            target_version="v2",
            target_digest="sha256:target-v2",
            format_digest="sha256:format",
            receiver_incarnation="rollout-tp0-7f3a",
            model_generation=4,
            layer_scope=layer_scope,
        )

    assert build(("layers.0", "layers.1")).layer_scope == ("layers.0", "layers.1")
    with pytest.raises(ValueError):
        build(("layers.1", "layers.0"))
    with pytest.raises(ValueError):
        build(("layers.0", "layers.0"))
    with pytest.raises(ValueError):
        build(["layers.0"])

    assert normalize_layer_scope(None) is None
    assert normalize_layer_scope(["layers.1", "layers.0", "layers.1"]) == (
        "layers.0",
        "layers.1",
    )
    assert build(normalize_layer_scope(["layers.1", "layers.0"])).layer_scope == (
        "layers.0",
        "layers.1",
    )


def test_prepared_update_keeps_the_payload_representation_out_of_the_contract():
    field_names = {field.name for field in dataclasses.fields(PreparedUpdate)}

    assert field_names.isdisjoint({"payload", "tensors", "weights", "path", "buffers"})


def test_result_records_are_frozen_and_report_only_public_facts():
    publish = PublishResult(
        model_id=MODEL_ID, version="v2", state=RevisionLifecycleState.READY, created=True
    )
    publisher_status = PublisherStatus(
        model_id=MODEL_ID,
        current_version="v2",
        state=RevisionLifecycleState.READY,
        publication_mode=PublicationMode.ASYNC,
    )
    update = WeightUpdateResult(
        success=True,
        receiver_id="rollout-tp0",
        installed_version="v2",
        state=ReceiverRevisionState.VERIFIED,
        target_digest="sha256:target-v2",
    )

    assert publish.state is RevisionLifecycleState.READY
    assert publisher_status.current_version == "v2"
    assert update.detail == ""
    for record in (publish, publisher_status, update):
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.model_id = "other"


def test_receiver_status_requires_recovery_only_when_poisoned():
    def status(state):
        return ReceiverStatus(
            receiver_id="rollout-tp0",
            model_id=MODEL_ID,
            installed_version="v2",
            state=state,
        )

    assert status(ReceiverRevisionState.POISONED).recovery_required is True
    assert status(ReceiverRevisionState.VERIFIED).recovery_required is False
    assert status(ReceiverRevisionState.FAILED).recovery_required is False
    assert status(None).recovery_required is False


def test_no_generic_engine_source_or_codec_mechanics_are_defined():
    forbidden = (
        "EngineAdapter",
        "SglangEngineAdapter",
        "MxModelLoader",
        "LoadStrategy",
        "LoadStrategyChain",
        "LoadResult",
        "TransportAdapter",
        "RecoveryStore",
        "SourceAdapter",
        "DeltaCodec",
        "Codec",
        "install_callback",
        "pull_weights",
        "update_weights_from_disk",
    )

    for name in forbidden:
        assert not hasattr(api_module, name)
        assert not hasattr(refit, name)


def test_refit_package_exports_exactly_the_stable_public_surface():
    assert refit.__all__ == sorted(refit.__all__)
    assert set(refit.__all__) == {
        # revision domain
        "ChangeState",
        "DeltaDescriptor",
        "DeltaLocation",
        "DeltaTransferMethod",
        "FilesystemLocation",
        "PublicationMode",
        "RankDelta",
        "ReceiverRevisionState",
        "ReceiverStateRecord",
        "RecoveryCandidate",
        "RecoveryCandidateKind",
        "RevisionLifecycleState",
        "RevisionManifest",
        "RevisionRank",
        "RevisionRecord",
        "RevisionSummary",
        "S3Location",
        "TensorDescriptor",
        "TensorRegion",
        "TensorShard",
        "ZeroMqLocation",
        # catalog boundary
        "CatalogProtocolError",
        "GrpcRevisionCatalog",
        "PublishedRevision",
        "RecoveryCandidatePage",
        "RevisionCatalog",
        "RevisionPage",
        # lifecycle contract
        "PreparedUpdate",
        "PublishResult",
        "Publisher",
        "PublisherConfig",
        "PublisherStatus",
        "Receiver",
        "ReceiverConfig",
        "ReceiverStatus",
        "RecoveryStoreConfig",
        "TransportConfig",
        "TransportKind",
        "WeightUpdateResult",
        "normalize_layer_scope",
        # refit timing (pre-existing)
        "MX_REFIT_TIMING_PREFIX",
        "REFIT_TIMING_STAGES",
        "RefitTimingRecorder",
        "add_refit_bytes",
        "current_refit_timing",
        "refit_span",
        "use_refit_timing",
    }
    for name in refit.__all__:
        assert hasattr(refit, name)
