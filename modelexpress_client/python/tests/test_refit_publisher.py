# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete CANONICAL Publisher lifecycle and publication ordering."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import replace

import pytest
import torch

import modelexpress.refit.publisher as publisher_module
from modelexpress import revision_pb2
from modelexpress.refit.api import (
    PublishResult,
    PublisherConfig,
    TransportConfig,
    TransportKind,
)
from modelexpress.refit.catalog import GrpcRevisionCatalog, PublishedRevision
from modelexpress.refit.codec import TENSOR_BYTE_XOR, ZSTD_COMPRESSION
from modelexpress.refit.manifest import (
    ChangeState,
    DeltaLocation,
    DeltaTransferMethod,
    PublicationMode,
    RankDelta,
    RevisionLifecycleState,
    RevisionManifest,
    RevisionRank,
    RevisionRecord,
    S3Location,
)
from modelexpress.refit.publisher import (
    PublicationCancelled,
    Publisher,
    PublisherError,
    PublisherStateError,
)
from modelexpress.refit.source.canonical import (
    CanonicalCapture,
    CanonicalDeltaError,
    CanonicalFormatIdentity,
    FilesystemCanonicalBaseStore,
    decode_root_index,
)
from modelexpress.refit.source.base import CanonicalTensorSpec
from modelexpress.refit.transport import StoredObject, TransportClosedError
from modelexpress.refit.transport.filesystem import FilesystemCanonicalTransport
from modelexpress.refit.transport.s3 import S3CanonicalTransport


class _Catalog:
    def __init__(self, *, committed_after: int | None = None, events=None) -> None:
        self.published = []
        self.records = {}
        self.get_calls = 0
        self.commit_calls = 0
        self.committed_after = committed_after
        self.closed = False
        self.events = events

    def publish_revision(self, manifest, *, publisher_id, publication_mode=None):
        if self.events is not None:
            self.events.append("catalog:publish")
        self.published.append((manifest, publisher_id, publication_mode))
        record = RevisionRecord(manifest, RevisionLifecycleState.READY)
        self.records[(manifest.model_id, manifest.version)] = record
        return PublishedRevision(record, created=True)

    def get_revision(self, model_id, version):
        if version != "v1":
            self.get_calls += 1
        record = self.records[(model_id, version)]
        if (
            version != "v1"
            and self.committed_after is not None
            and self.get_calls >= self.committed_after
        ):
            record = RevisionRecord(record.manifest, RevisionLifecycleState.COMMITTED)
            self.records[(model_id, version)] = record
        return record

    def _get_revision_with_timeout(self, model_id, version, *, timeout):
        assert timeout > 0
        return self.get_revision(model_id, version)

    def _publish_revision_with_timeout(
        self, manifest, *, publisher_id, publication_mode=None, timeout
    ):
        assert timeout > 0
        return self.publish_revision(
            manifest,
            publisher_id=publisher_id,
            publication_mode=publication_mode,
        )

    def commit_version(self, *_args, **_kwargs):
        self.commit_calls += 1
        raise AssertionError("Publisher must never commit a revision")

    def close(self):
        self.closed = True
        if self.events is not None:
            self.events.append("catalog:close")


def test_grpc_catalog_publisher_calls_forward_explicit_deadlines():
    manifest = RevisionManifest(
        model_id="model",
        version="v2",
        base_version="v1",
        transfer_method=DeltaTransferMethod.CANONICAL,
        delta_method=TENSOR_BYTE_XOR,
        compression_algorithm=ZSTD_COMPRESSION,
        format_digest="sha256:" + "1" * 64,
        base_digest="sha256:" + "2" * 64,
        target_digest="sha256:" + "3" * 64,
        ranks=(
            RevisionRank(
                trainer_rank=0,
                producer_id="trainer-0",
                source_layout_digest="sha256:" + "1" * 64,
                delta=RankDelta(change_state=ChangeState.CLEAN),
            ),
        ),
    )
    record = RevisionRecord(manifest, RevisionLifecycleState.READY)

    class Rpc:
        def __init__(self, response):
            self.response = response
            self.timeouts = []

        def __call__(self, _request, *, timeout):
            self.timeouts.append(timeout)
            return self.response

    class Stub:
        PublishRevision = Rpc(
            revision_pb2.PublishRevisionResponse(
                revision=record.to_proto(), created=True
            )
        )
        GetRevision = Rpc(revision_pb2.GetRevisionResponse(revision=record.to_proto()))

    catalog = GrpcRevisionCatalog(stub=Stub())
    catalog._publish_revision_with_timeout(
        manifest,
        publisher_id="trainer-0",
        publication_mode=PublicationMode.ASYNC,
        timeout=1.25,
    )
    catalog._get_revision_with_timeout("model", "v2", timeout=2.5)

    assert Stub.PublishRevision.timeouts == [1.25]
    assert Stub.GetRevision.timeouts == [2.5]


class _RecordingTransport:
    def __init__(self, delegate, events) -> None:
        self.delegate = delegate
        self.events = events

    @property
    def identity(self):
        return self.delegate.identity

    def publish(self, key, data, checksum):
        kind = "root" if key.endswith("root.json") else "bucket"
        self.events.append(f"transport:publish:{kind}")
        return self.delegate.publish(key, data, checksum)

    def fetch(self, stored):
        return self.delegate.fetch(stored)

    def resolve(self, location, checksum, maximum_size):
        return self.delegate.resolve(location, checksum, maximum_size)

    def verify(self, stored):
        kind = (
            "root"
            if stored.location.filesystem.path.endswith("root.json")
            else "bucket"
        )
        self.events.append(f"transport:verify:{kind}")
        self.delegate.verify(stored)

    def close(self):
        self.events.append("transport:close")
        self.delegate.close()


def _base_store(tmp_path):
    store = FilesystemCanonicalBaseStore(tmp_path / "bases")
    store.create_snapshot(
        "v1",
        (
            (
                ("a.weight", torch.zeros(256)),
                ("b.weight", torch.arange(8, dtype=torch.float32)),
            ),
        ),
    )
    return store


def _config(
    tmp_path, *, mode=PublicationMode.ASYNC, method=DeltaTransferMethod.CANONICAL
):
    return PublisherConfig(
        model_id="model",
        catalog_endpoint="catalog:1234",
        transport=TransportConfig(
            TransportKind.FILESYSTEM,
            f"file://{(tmp_path / 'objects').resolve()}",
        ),
        delta_transfer_method=method,
        delta_method=TENSOR_BYTE_XOR,
        compression_algorithm=ZSTD_COMPRESSION,
        publication_mode=mode,
    )


def _dirty_capture(version, consume):
    assert version in {"v2", "v3"}
    offset = 1 if version == "v2" else 2
    consume(
        (
            ("a.weight", torch.cat((torch.tensor([offset]), torch.zeros(255)))),
            ("b.weight", torch.arange(8, dtype=torch.float32)),
        )
    )


def _publisher(
    tmp_path, *, capture=_dirty_capture, catalog=None, transport=None, **kwargs
):
    base_store = _base_store(tmp_path)
    catalog = catalog or _Catalog()
    transport = transport or FilesystemCanonicalTransport(tmp_path / "objects")
    kwargs.setdefault("poll_interval_seconds", 0.005)
    kwargs.setdefault("maximum_encoded_ratio", 8.0)
    producer_id = kwargs.pop("producer_id", "trainer-0")
    base = base_store.open_snapshot("v1")
    records = getattr(catalog, "records", None)
    if isinstance(records, dict):
        base_manifest = RevisionManifest(
            model_id="model",
            version=base.version,
            base_version="bootstrap",
            transfer_method=DeltaTransferMethod.CANONICAL,
            delta_method=TENSOR_BYTE_XOR,
            compression_algorithm=ZSTD_COMPRESSION,
            format_digest=base.format_digest,
            base_digest="sha256:" + "0" * 64,
            target_digest=base.target_digest,
            ranks=(
                RevisionRank(
                    trainer_rank=0,
                    producer_id="bootstrap",
                    source_layout_digest=base.format_digest,
                    delta=RankDelta(change_state=ChangeState.CLEAN),
                ),
            ),
        )
        records.setdefault(
            (base_manifest.model_id, base_manifest.version),
            RevisionRecord(base_manifest, RevisionLifecycleState.COMMITTED),
        )
    format_identity = base.format_identity
    canonical_schema = tuple(
        CanonicalTensorSpec(
            metadata.name,
            getattr(torch, metadata.dtype),
            metadata.shape,
        )
        for metadata in base.tensors
    )
    if not isinstance(capture, CanonicalCapture):
        capture = CanonicalCapture(capture, format_identity, canonical_schema)
    try:
        publisher = Publisher(
            capture=capture,
            base_store=base_store,
            initial_base_version="v1",
            producer_id=producer_id,
            format_identity=format_identity,
            catalog=catalog,
            transport=transport,
            allow_filesystem_transport=True,
            **kwargs,
        )
    except Exception:
        transport.close()
        base_store.close()
        raise
    return publisher, base_store, catalog, transport


def _synthetic_intent(request, *, base_digest="sha256:" + "2" * 64):
    return publisher_module._PublicationIntent(
        request=request,
        format_digest="sha256:" + "1" * 64,
        base_digest=base_digest,
        producer_id="trainer-0",
    )


def test_initialize_is_fail_closed_and_conforms_to_the_frozen_protocol(tmp_path):
    publisher, _store, _catalog, _transport = _publisher(tmp_path)

    with pytest.raises(PublisherStateError, match="initialized"):
        publisher.publish_version("v2")
    with pytest.raises(PublisherError, match="CANONICAL"):
        publisher.initialize(_config(tmp_path, method=DeltaTransferMethod.RANK_LOCAL))
    publisher.deregister()

    publisher, _store, _catalog, _transport = _publisher(tmp_path / "missing-codec")
    config = _config(tmp_path / "missing-codec")
    with pytest.raises(PublisherError, match="delta_method"):
        publisher.initialize(
            PublisherConfig(
                model_id=config.model_id,
                catalog_endpoint=config.catalog_endpoint,
                transport=config.transport,
                delta_transfer_method=DeltaTransferMethod.CANONICAL,
                compression_algorithm=ZSTD_COMPRESSION,
            )
        )
    publisher.deregister()

    publisher, _store, _catalog, _transport = _publisher(tmp_path / "valid")
    publisher.initialize(_config(tmp_path / "valid"))
    assert publisher.status().current_version == "v1"
    with pytest.raises(PublisherStateError, match="already initialized"):
        publisher.initialize(_config(tmp_path / "valid"))
    with pytest.raises(PublisherError, match="complete-model"):
        publisher.publish_version("v2", layers=["model.layers.0"])
    publisher.deregister()


@pytest.mark.parametrize("mode", [None, 2, "ASYNC"])
def test_initialize_rejects_non_enum_publication_modes(tmp_path, mode):
    publisher, _store, _catalog, _transport = _publisher(tmp_path)
    try:
        with pytest.raises(PublisherError, match="publication_mode"):
            publisher.initialize(_config(tmp_path, mode=mode))
    finally:
        publisher.deregister()


def test_initialize_rejects_a_different_canonical_representation(tmp_path):
    base_store = _base_store(tmp_path)
    source_identity = CanonicalFormatIdentity(atomic_groups=(("a.weight", "b.weight"),))
    base = base_store.open_snapshot("v1")
    canonical_schema = tuple(
        CanonicalTensorSpec(
            metadata.name, getattr(torch, metadata.dtype), metadata.shape
        )
        for metadata in base.tensors
    )
    publisher = Publisher(
        capture=CanonicalCapture(_dirty_capture, source_identity, canonical_schema),
        base_store=base_store,
        initial_base_version="v1",
        producer_id="trainer-0",
        format_identity=source_identity,
        catalog=_Catalog(),
        transport=FilesystemCanonicalTransport(tmp_path / "objects"),
        allow_filesystem_transport=True,
    )

    with pytest.raises(PublisherError, match="format identity"):
        publisher.initialize(_config(tmp_path))

    publisher.deregister()


def test_publisher_rejects_capture_bound_to_a_different_representation(tmp_path):
    base_store = _base_store(tmp_path)
    base = base_store.open_snapshot("v1")
    schema = tuple(
        CanonicalTensorSpec(
            metadata.name, getattr(torch, metadata.dtype), metadata.shape
        )
        for metadata in base.tensors
    )
    capture = CanonicalCapture(
        _dirty_capture,
        CanonicalFormatIdentity(atomic_groups=(("a.weight", "b.weight"),)),
        schema,
    )

    transport = FilesystemCanonicalTransport(tmp_path / "objects")
    try:
        with pytest.raises(ValueError, match="capture.*format identity"):
            Publisher(
                capture=capture,
                base_store=base_store,
                initial_base_version="v1",
                producer_id="trainer-0",
                format_identity=base_store.open_snapshot("v1").format_identity,
                catalog=_Catalog(),
                transport=transport,
                allow_filesystem_transport=True,
            )
    finally:
        transport.close()
        base_store.close()


def test_initialize_rejects_capture_bound_to_a_different_hf_schema(tmp_path):
    base_store = _base_store(tmp_path)
    base = base_store.open_snapshot("v1")

    capture = CanonicalCapture(
        _dirty_capture,
        base.format_identity,
        (
            CanonicalTensorSpec("a.weight", torch.float32, (256,)),
            CanonicalTensorSpec("b.weight", torch.float32, (9,)),
        ),
    )

    transport = FilesystemCanonicalTransport(tmp_path / "objects")
    publisher = Publisher(
        capture=capture,
        base_store=base_store,
        initial_base_version="v1",
        producer_id="trainer-0",
        format_identity=base.format_identity,
        catalog=_Catalog(),
        transport=transport,
        allow_filesystem_transport=True,
    )
    try:
        with pytest.raises(PublisherError, match="format digest"):
            publisher.initialize(_config(tmp_path))
    finally:
        publisher.deregister()


def test_canonical_requires_explicit_base_before_capture(tmp_path):
    captured = []
    publisher, _store, catalog, _transport = _publisher(
        tmp_path,
        capture=lambda version, consume: captured.append((version, consume)),
    )
    publisher.initialize(_config(tmp_path))

    with pytest.raises(PublisherError, match="explicit base_version"):
        publisher.publish_version("v2")

    assert captured == []
    assert catalog.published == []
    publisher.deregister()


def test_publication_intent_mismatch_fails_before_capture(tmp_path):
    captured = []

    class Coordinator:
        rank = 0

        def agree(self, value):
            if isinstance(value, publisher_module._PublicationRequest):
                raise PublisherError("publication intent differs across trainer ranks")
            return value

        def broadcast(self, value):
            return value

    publisher, _store, catalog, _transport = _publisher(
        tmp_path,
        capture=lambda version, consume: captured.append((version, consume)),
        coordinator=Coordinator(),
    )
    publisher.initialize(_config(tmp_path))

    with pytest.raises(PublisherError, match="intent differs"):
        publisher.publish_version("v2", base_version="v1")

    assert captured == []
    assert catalog.published == []
    publisher.deregister()


def test_catalog_base_attestation_fails_before_capture(tmp_path):
    captured = []

    def capture(version, consume):
        captured.append(version)
        _dirty_capture(version, consume)

    publisher, _store, catalog, _transport = _publisher(
        tmp_path,
        capture=capture,
    )
    base_record = catalog.records[("model", "v1")]
    catalog.records[("model", "v1")] = RevisionRecord(
        replace(
            base_record.manifest,
            target_digest="sha256:" + "f" * 64,
        ),
        RevisionLifecycleState.COMMITTED,
    )
    publisher.initialize(_config(tmp_path))

    with pytest.raises(PublisherError, match="catalog.*base digest"):
        publisher.publish_version("v2", base_version="v1")

    assert captured == []
    assert catalog.published == []
    publisher.deregister()


def test_peer_capture_failure_prevents_root_and_catalog_publication(tmp_path):
    class Coordinator:
        rank = 0

        def __init__(self):
            self.agreements = 0

        def agree(self, value):
            self.agreements += 1
            if isinstance(value, publisher_module._CaptureComplete):
                raise PublisherError("rank 1 capture failed")
            return value

        def broadcast(self, value):
            return value

    publisher, _store, catalog, _transport = _publisher(
        tmp_path, coordinator=Coordinator()
    )
    publisher.initialize(_config(tmp_path))

    with pytest.raises(PublisherError, match="rank 1 capture failed"):
        publisher.publish_version("v2", base_version="v1")

    assert catalog.published == []
    assert list((tmp_path / "objects").rglob("root.json")) == []
    publisher.deregister()


def test_ready_handoff_precedes_each_rank_block_poll(tmp_path):
    class Coordinator:
        rank = 0

        def __init__(self):
            self.broadcasts = 0

        def agree(self, value):
            return value

        def broadcast(self, value):
            self.broadcasts += 1
            return value

    coordinator = Coordinator()

    class Catalog(_Catalog):
        def _get_revision_with_timeout(self, model_id, version, *, timeout):
            if version == "v1":
                return super()._get_revision_with_timeout(
                    model_id, version, timeout=timeout
                )
            assert coordinator.broadcasts == 2, "READY was not handed off first"
            record = self.records[(model_id, version)]
            return RevisionRecord(record.manifest, RevisionLifecycleState.COMMITTED)

    publisher, _store, _catalog, _transport = _publisher(
        tmp_path, catalog=Catalog(), coordinator=coordinator
    )
    publisher.initialize(_config(tmp_path, mode=PublicationMode.BLOCK))

    result = publisher.publish_version("v2", base_version="v1")

    assert result.state is RevisionLifecycleState.COMMITTED
    publisher.deregister()


def test_many_async_publications_reuse_one_observer_thread(tmp_path):
    publisher, _store, _catalog, _transport = _publisher(tmp_path)
    publisher.initialize(_config(tmp_path, mode=PublicationMode.ASYNC))
    try:
        publisher.publish_version("v2", base_version="v1")
        publisher.publish_version("v3", base_version="v2")

        observers = [
            thread
            for thread in threading.enumerate()
            if thread.name.startswith("mx-canonical-observer-")
        ]
        assert len(observers) == 1
    finally:
        publisher.deregister()


def test_nonzero_rank_can_publish_two_successive_async_versions_without_local_base(
    tmp_path,
):
    captured = []

    class Coordinator:
        rank = 1

        def __init__(self):
            self.request = None
            self.intent = None
            self.broadcasts = 0

        def agree(self, value):
            if not isinstance(value, publisher_module._CaptureComplete):
                self.request = value
            return value

        def broadcast(self, value):
            assert value is None
            self.broadcasts += 1
            if self.broadcasts % 2:
                if isinstance(self.request, publisher_module._PublicationIntent):
                    self.intent = self.request
                else:
                    previous_digest = (
                        "sha256:" + "2" * 64
                        if self.request.base_version == "v1"
                        else "sha256:" + "3" * 64
                    )
                    self.intent = _synthetic_intent(
                        self.request, base_digest=previous_digest
                    )
                return publisher_module._PreflightReady(self.intent)

            target_digest = (
                "sha256:" + "3" * 64
                if self.intent.target_version == "v2"
                else "sha256:" + "4" * 64
            )
            manifest = RevisionManifest(
                model_id=self.intent.model_id,
                version=self.intent.target_version,
                base_version=self.intent.base_version,
                transfer_method=self.intent.transfer_method,
                delta_method=self.intent.delta_method,
                compression_algorithm=self.intent.compression_algorithm,
                format_digest=self.intent.format_digest,
                base_digest=self.intent.base_digest,
                target_digest=target_digest,
                ranks=(
                    RevisionRank(
                        trainer_rank=0,
                        producer_id=self.intent.producer_id,
                        source_layout_digest=self.intent.format_digest,
                        delta=RankDelta(change_state=ChangeState.CLEAN),
                    ),
                ),
            )
            return publisher_module._ReadyPublication(
                manifest,
                PublishResult(
                    manifest.model_id,
                    manifest.version,
                    RevisionLifecycleState.READY,
                    created=True,
                ),
            )

    publisher, store, _catalog, _transport = _publisher(
        tmp_path,
        capture=lambda version, _consume: captured.append(version),
        coordinator=Coordinator(),
        producer_id="trainer-1",
    )
    publisher.initialize(_config(tmp_path, mode=PublicationMode.ASYNC))

    assert publisher.publish_version("v2", base_version="v1").version == "v2"
    assert not (store._root / "versions" / "v2").exists()
    assert publisher.publish_version("v3", base_version="v2").version == "v3"
    assert captured == ["v2", "v3"]
    publisher.deregister()


def test_policy_ineligible_base_enters_rank_agreement_before_failure(tmp_path):
    class Coordinator:
        rank = 1

        def __init__(self):
            self.contribution = None

        def agree(self, value):
            self.contribution = value
            return value

        def broadcast(self, _value):
            raise AssertionError("policy failure must precede rank-0 preflight")

    coordinator = Coordinator()
    publisher, _store, _catalog, _transport = _publisher(
        tmp_path, coordinator=coordinator
    )
    publisher.initialize(_config(tmp_path))

    with pytest.raises(PublisherError, match="eligible exact base"):
        publisher.publish_version("v3", base_version="v2")

    assert isinstance(coordinator.contribution, publisher_module._RemoteFailure)
    publisher.deregister()


def test_async_observer_start_failure_does_not_fail_ready_publication(
    tmp_path, monkeypatch
):
    publisher, _store, _catalog, _transport = _publisher(tmp_path)
    publisher.initialize(_config(tmp_path, mode=PublicationMode.ASYNC))
    monkeypatch.setattr(
        publisher_module.threading.Thread,
        "start",
        lambda _thread: (_ for _ in ()).throw(RuntimeError("thread unavailable")),
    )

    result = publisher.publish_version("v2", base_version="v1")

    assert result.state is RevisionLifecycleState.READY
    assert publisher.status().state is RevisionLifecycleState.READY
    publisher.deregister()


def test_publisher_enforces_the_configured_encoder_bucket_bound(tmp_path):
    catalog = _Catalog()
    publisher, _store, _catalog, _transport = _publisher(
        tmp_path,
        catalog=catalog,
        maximum_bucket_bytes=1024,
    )
    publisher.initialize(_config(tmp_path))
    try:
        with pytest.raises(CanonicalDeltaError, match="maximum decoded size"):
            publisher.publish_version("v2", base_version="v1")
        assert catalog.published == []
    finally:
        publisher.deregister()


def test_reentrant_deregister_fails_without_deadlocking_publication(tmp_path):
    owner = {}

    def capture(_version, _consume):
        owner["publisher"].deregister()

    publisher, _store, catalog, _transport = _publisher(tmp_path, capture=capture)
    owner["publisher"] = publisher
    publisher.initialize(_config(tmp_path))

    with pytest.raises(PublisherStateError, match="reentrantly"):
        publisher.publish_version("v2", base_version="v1")

    assert catalog.published == []
    publisher.deregister()


@pytest.mark.parametrize("poll_interval", [math.inf, math.nan])
def test_publisher_rejects_nonfinite_poll_interval(tmp_path, poll_interval):
    with pytest.raises(ValueError, match="poll_interval_seconds"):
        _publisher(tmp_path, poll_interval_seconds=poll_interval)


def test_transport_cannot_redirect_publication_within_namespace(tmp_path):
    class RedirectingTransport:
        def __init__(self, delegate):
            self.delegate = delegate

        @property
        def identity(self):
            return self.delegate.identity

        def publish(self, key, data, checksum):
            return self.delegate.publish(
                f"redirect/{key.rsplit('/', 1)[-1]}", data, checksum
            )

        def fetch(self, stored):
            return self.delegate.fetch(stored)

        def verify(self, stored):
            return self.delegate.verify(stored)

        def close(self):
            self.delegate.close()

    catalog = _Catalog()
    transport = RedirectingTransport(FilesystemCanonicalTransport(tmp_path / "objects"))
    publisher, _store, _catalog, _transport = _publisher(
        tmp_path, catalog=catalog, transport=transport
    )
    publisher.initialize(_config(tmp_path))
    try:
        with pytest.raises(PublisherError, match="exact canonical object key"):
            publisher.publish_version("v2", base_version="v1")
        assert catalog.published == []
    finally:
        publisher.deregister()


def test_async_publishes_verified_root_then_one_rank_zero_manifest_and_retains_next_base(
    tmp_path,
):
    events = []
    catalog = _Catalog(events=events)
    raw_transport = FilesystemCanonicalTransport(tmp_path / "objects")
    transport = _RecordingTransport(raw_transport, events)
    publisher, store, _catalog, _transport = _publisher(
        tmp_path,
        catalog=catalog,
        transport=transport,
    )
    publisher.initialize(_config(tmp_path, mode=PublicationMode.ASYNC))

    first = publisher.publish_version("v2", base_version="v1")
    second = publisher.publish_version("v3", base_version="v2")

    assert first.state is RevisionLifecycleState.READY
    assert second.state is RevisionLifecycleState.READY
    assert (
        store.open_snapshot("v2").target_digest == catalog.published[0][0].target_digest
    )
    assert (
        store.open_snapshot("v3").target_digest == catalog.published[1][0].target_digest
    )
    first_manifest, producer_id, mode = catalog.published[0]
    assert producer_id == "trainer-0"
    assert mode is PublicationMode.ASYNC
    assert first_manifest.base_version == "v1"
    assert catalog.published[1][0].base_version == "v2"
    assert first_manifest.transfer_method is DeltaTransferMethod.CANONICAL
    assert first_manifest.delta_method == TENSOR_BYTE_XOR
    assert first_manifest.compression_algorithm == ZSTD_COMPRESSION
    assert len(first_manifest.ranks) == 1
    rank = first_manifest.ranks[0]
    assert rank.trainer_rank == 0
    assert rank.producer_id == "trainer-0"
    assert rank.source_layout_digest == first_manifest.format_digest
    assert rank.shards == ()
    assert rank.delta.change_state is ChangeState.DIRTY
    assert rank.delta.delta_descriptor is None
    assert rank.delta.location.filesystem is not None
    assert rank.delta.checksum and len(rank.delta.checksum) == 8

    root_stored = raw_transport.resolve(
        rank.delta.location,
        rank.delta.checksum,
        64 * 1024 * 1024,
    )
    root_bytes = raw_transport.fetch(root_stored)
    root = decode_root_index(root_bytes, rank.delta.checksum)
    assert root.target_digest == first_manifest.target_digest
    assert len(root.buckets) == 1
    assert [item.change_state for item in root.tensors] == ["DIRTY", "CLEAN"]
    assert events.index("transport:publish:root") < events.index("catalog:publish")
    assert events.index("transport:verify:bucket") < events.index(
        "transport:publish:root"
    )
    assert publisher.status().current_version == "v3"
    publisher.deregister()


def test_clean_publication_has_no_payload_reference(tmp_path):
    def clean_capture(_version, consume):
        consume(
            (
                ("a.weight", torch.zeros(256)),
                ("b.weight", torch.arange(8, dtype=torch.float32)),
            )
        )

    publisher, _store, catalog, _transport = _publisher(tmp_path, capture=clean_capture)
    publisher.initialize(_config(tmp_path))

    result = publisher.publish_version("v2", base_version="v1")

    assert result.state is RevisionLifecycleState.READY
    manifest = catalog.published[0][0]
    assert len(manifest.ranks) == 1
    delta = manifest.ranks[0].delta
    assert delta.change_state is ChangeState.CLEAN
    assert delta.checksum is None
    assert delta.location is None
    assert delta.delta_descriptor is None
    assert list((tmp_path / "objects").rglob("*.*")) == []
    publisher.deregister()


def test_block_waits_read_only_for_commit(tmp_path):
    catalog = _Catalog(committed_after=3)
    publisher, _store, _catalog, _transport = _publisher(tmp_path, catalog=catalog)
    publisher.initialize(_config(tmp_path, mode=PublicationMode.BLOCK))

    result = publisher.publish_version("v2", base_version="v1")

    assert result.state is RevisionLifecycleState.COMMITTED
    assert catalog.get_calls == 3
    assert catalog.commit_calls == 0
    assert publisher.status().state is RevisionLifecycleState.COMMITTED
    publisher.deregister()


def test_async_observer_updates_ready_status_to_committed(tmp_path):
    catalog = _Catalog(committed_after=2)
    publisher, _store, _catalog, _transport = _publisher(tmp_path, catalog=catalog)
    publisher.initialize(_config(tmp_path, mode=PublicationMode.ASYNC))

    result = publisher.publish_version("v2", base_version="v1")
    deadline = time.monotonic() + 5
    while (
        publisher.status().state is not RevisionLifecycleState.COMMITTED
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)

    assert result.state is RevisionLifecycleState.READY
    assert publisher.status().state is RevisionLifecycleState.COMMITTED
    publisher.deregister()


def test_block_poll_failure_does_not_advance_the_eligible_exact_base(tmp_path):
    class PollFailsOnce(_Catalog):
        def __init__(self):
            super().__init__()
            self.failures = 1

        def _get_revision_with_timeout(self, model_id, version, *, timeout):
            if version == "v1":
                return super()._get_revision_with_timeout(
                    model_id, version, timeout=timeout
                )
            if self.failures:
                self.failures -= 1
                raise OSError("bounded poll failed")
            record = self.records[(model_id, version)]
            record = RevisionRecord(record.manifest, RevisionLifecycleState.COMMITTED)
            self.records[(model_id, version)] = record
            return record

    catalog = PollFailsOnce()
    publisher, _store, _catalog, _transport = _publisher(tmp_path, catalog=catalog)
    publisher.initialize(_config(tmp_path, mode=PublicationMode.BLOCK))

    with pytest.raises(PublisherError, match="bounded poll failed"):
        publisher.publish_version("v2", base_version="v1")
    result = publisher.publish_version("v3", base_version="v1")

    assert result.state is RevisionLifecycleState.COMMITTED
    assert catalog.published[1][0].base_version == "v1"
    publisher.deregister()


def test_explicit_base_must_be_the_policy_eligible_exact_base(tmp_path):
    publisher, store, _catalog, _transport = _publisher(tmp_path)
    base = store.open_snapshot("v1")
    store.create_snapshot(
        "v2",
        (
            tuple(
                (metadata.name, store.read_tensor(base, metadata.name))
                for metadata in base.tensors
            ),
        ),
    )
    publisher.initialize(_config(tmp_path))

    with pytest.raises(PublisherError, match="eligible exact base"):
        publisher.publish_version("v3", base_version="v2")

    publisher.deregister()


def test_initialize_rejects_injected_transport_outside_configured_namespace(tmp_path):
    publisher, _store, _catalog, _transport = _publisher(tmp_path)
    config = _config(tmp_path)
    s3_config = PublisherConfig(
        model_id=config.model_id,
        catalog_endpoint=config.catalog_endpoint,
        transport=TransportConfig(TransportKind.S3, "s3://bucket/prefix"),
        delta_transfer_method=config.delta_transfer_method,
        delta_method=config.delta_method,
        compression_algorithm=config.compression_algorithm,
        publication_mode=config.publication_mode,
    )

    with pytest.raises(PublisherError, match="transport.*configured"):
        publisher.initialize(s3_config)
    publisher.deregister()


def test_nonzero_rank_does_not_enter_capture_after_rank_zero_preflight_failure(
    tmp_path,
):
    captured = []

    class Coordinator:
        rank = 1

        def agree(self, value):
            return value

        def broadcast(self, value):
            assert value is None
            return publisher_module._RemoteFailure(
                "CanonicalDeltaError", "base unavailable"
            )

    publisher, _store, _catalog, _transport = _publisher(
        tmp_path,
        capture=lambda version, consume: captured.append((version, consume)),
        coordinator=Coordinator(),
    )
    publisher.initialize(_config(tmp_path))

    with pytest.raises(PublisherError, match="base unavailable"):
        publisher.publish_version("v2", base_version="v1")

    assert captured == []
    publisher.deregister()


def test_nonzero_initialize_does_not_require_a_local_exact_base_snapshot(tmp_path):
    seed_store = _base_store(tmp_path / "seed")
    base = seed_store.open_snapshot("v1")
    schema = tuple(
        CanonicalTensorSpec(
            metadata.name, getattr(torch, metadata.dtype), metadata.shape
        )
        for metadata in base.tensors
    )

    class MissingLocalBase:
        closed = False

        def open_snapshot(self, _version):
            raise AssertionError("nonzero rank must not open the rank-0 exact base")

        def close(self):
            self.closed = True

    class Coordinator:
        rank = 1

        @staticmethod
        def agree(value):
            return value

    missing = MissingLocalBase()
    publisher = Publisher(
        capture=CanonicalCapture(_dirty_capture, base.format_identity, schema),
        base_store=missing,
        initial_base_version="v1",
        producer_id="trainer-1",
        format_identity=base.format_identity,
        catalog=_Catalog(),
        transport=FilesystemCanonicalTransport(tmp_path / "objects"),
        coordinator=Coordinator(),
        allow_filesystem_transport=True,
    )
    seed_store.close()

    publisher.initialize(_config(tmp_path))
    assert publisher.status().current_version == "v1"
    publisher.deregister()
    assert missing.closed


def test_peer_initialize_failure_rolls_back_all_local_resources(tmp_path):
    events = []

    class Coordinator:
        rank = 1

        def __init__(self):
            self.agreements = 0

        def agree(self, _value):
            self.agreements += 1
            return publisher_module._RemoteFailure(
                "CanonicalDeltaError", "rank 0 base unavailable"
            )

    coordinator = Coordinator()
    catalog = _Catalog(events=events)
    raw_transport = FilesystemCanonicalTransport(tmp_path / "objects")
    transport = _RecordingTransport(raw_transport, events)
    publisher, store, _catalog, _transport = _publisher(
        tmp_path,
        catalog=catalog,
        transport=transport,
        coordinator=coordinator,
        source_close=lambda: events.append("source:close"),
    )

    with pytest.raises(PublisherError, match="rank 0 base unavailable"):
        publisher.initialize(_config(tmp_path))

    assert coordinator.agreements == 1
    assert events == ["source:close", "transport:close", "catalog:close"]
    with pytest.raises(CanonicalDeltaError, match="closed"):
        store.open_snapshot("v1")
    with pytest.raises(PublisherStateError, match="deregistered"):
        publisher.initialize(_config(tmp_path))
    publisher.deregister()


def test_block_requires_bounded_catalog_polling_capability(tmp_path):
    class UnboundedCatalog(_Catalog):
        _get_revision_with_timeout = None

    publisher, _store, _catalog, _transport = _publisher(
        tmp_path, catalog=UnboundedCatalog()
    )

    with pytest.raises(PublisherError, match="bounded revision polling"):
        publisher.initialize(_config(tmp_path, mode=PublicationMode.BLOCK))
    publisher.deregister()


def test_initialize_closes_each_resource_it_created_when_validation_fails(
    tmp_path, monkeypatch
):
    class CreatedCatalog(_Catalog):
        pass

    class CreatedTransport:
        identity = object()

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    catalog = CreatedCatalog()
    transport = CreatedTransport()
    monkeypatch.setattr(
        publisher_module, "GrpcRevisionCatalog", lambda _endpoint: catalog
    )
    monkeypatch.setattr(Publisher, "_build_transport", lambda _self, _config: transport)
    base_store = _base_store(tmp_path)
    base = base_store.open_snapshot("v1")
    format_identity = base.format_identity
    canonical_schema = tuple(
        CanonicalTensorSpec(
            metadata.name, getattr(torch, metadata.dtype), metadata.shape
        )
        for metadata in base.tensors
    )
    publisher = Publisher(
        capture=CanonicalCapture(_dirty_capture, format_identity, canonical_schema),
        base_store=base_store,
        initial_base_version="v1",
        producer_id="trainer-0",
        format_identity=format_identity,
        allow_filesystem_transport=True,
    )

    with pytest.raises(PublisherError, match="does not match configured"):
        publisher.initialize(_config(tmp_path))

    assert transport.closed
    assert catalog.closed
    publisher.deregister()


def test_publisher_rejects_a_mixed_transport_location_before_root_or_catalog(tmp_path):
    class MixedLocationTransport:
        def __init__(self, delegate):
            self.delegate = delegate

        @property
        def identity(self):
            return self.delegate.identity

        def publish(self, key, data, checksum):
            stored = self.delegate.publish(key, data, checksum)
            return StoredObject(
                DeltaLocation(s3=S3Location(bucket="wrong", key=key)),
                stored.checksum,
                stored.size,
            )

        def fetch(self, stored):
            raise AssertionError("mixed locations must fail before fetch")

        def verify(self, stored):
            raise AssertionError("mixed locations must fail before verify")

        def close(self):
            self.delegate.close()

    catalog = _Catalog()
    transport = MixedLocationTransport(
        FilesystemCanonicalTransport(tmp_path / "objects")
    )
    publisher, _store, _catalog, _transport = _publisher(
        tmp_path, catalog=catalog, transport=transport
    )
    publisher.initialize(_config(tmp_path))

    with pytest.raises(PublisherError, match="non-filesystem"):
        publisher.publish_version("v2", base_version="v1")

    assert catalog.published == []
    assert list((tmp_path / "objects").rglob("root.json")) == []
    publisher.deregister()
    publisher.deregister()


def test_s3_publication_exposes_only_the_verified_immutable_root(tmp_path):
    from tests.test_refit_canonical_transport import _FakeS3

    client = _FakeS3()
    transport = S3CanonicalTransport("bucket", "prefix", client=client)
    catalog = _Catalog()
    publisher, _store, _catalog, _transport = _publisher(
        tmp_path, catalog=catalog, transport=transport
    )
    config = _config(tmp_path)
    publisher.initialize(
        PublisherConfig(
            model_id=config.model_id,
            catalog_endpoint=config.catalog_endpoint,
            transport=TransportConfig(TransportKind.S3, "s3://bucket/prefix"),
            delta_transfer_method=config.delta_transfer_method,
            delta_method=config.delta_method,
            compression_algorithm=config.compression_algorithm,
            publication_mode=PublicationMode.ASYNC,
        )
    )

    result = publisher.publish_version("v2", base_version="v1")
    manifest = catalog.published[0][0]
    root_delta = manifest.ranks[0].delta
    assert result.state is RevisionLifecycleState.READY
    assert root_delta.location.s3 is not None
    assert root_delta.location.filesystem is None
    assert root_delta.location.s3.key.startswith("prefix/canonical/")
    assert root_delta.location.s3.key.endswith("/root.json")
    root_stored = transport.resolve(
        root_delta.location,
        root_delta.checksum,
        64 * 1024 * 1024,
    )
    stored_bytes = transport.fetch(root_stored)
    root = decode_root_index(stored_bytes, root_delta.checksum)
    assert len(root.buckets) == 1
    assert all(reference.location.s3 is not None for reference in root.buckets)
    assert manifest.ranks[0].shards == ()
    publisher.deregister()


def test_publisher_applies_its_rpc_timeout_to_owned_s3_transport(tmp_path, monkeypatch):
    captured = {}

    def transport(bucket, prefix, *, request_timeout_seconds):
        captured.update(
            bucket=bucket,
            prefix=prefix,
            request_timeout_seconds=request_timeout_seconds,
        )
        return object()

    monkeypatch.setattr(publisher_module, "S3CanonicalTransport", transport)
    publisher = object.__new__(Publisher)
    publisher._rpc_timeout_seconds = 2.75
    config = _config(tmp_path)
    config = PublisherConfig(
        model_id=config.model_id,
        catalog_endpoint=config.catalog_endpoint,
        transport=TransportConfig(TransportKind.S3, "s3://bucket/prefix"),
        delta_transfer_method=config.delta_transfer_method,
        delta_method=config.delta_method,
        compression_algorithm=config.compression_algorithm,
        publication_mode=config.publication_mode,
    )

    Publisher._build_transport(publisher, config)

    assert captured == {
        "bucket": "bucket",
        "prefix": "prefix",
        "request_timeout_seconds": 2.75,
    }


def test_concurrent_deregister_has_one_close_owner(tmp_path):
    events = []
    catalog = _Catalog(events=events)
    transport = _RecordingTransport(
        FilesystemCanonicalTransport(tmp_path / "objects"), events
    )

    def close_source():
        events.append("source:close")

    publisher, _store, _catalog, _transport = _publisher(
        tmp_path,
        catalog=catalog,
        transport=transport,
        source_close=close_source,
    )
    publisher.initialize(_config(tmp_path, mode=PublicationMode.BLOCK))
    publish_errors = []
    publish_thread = threading.Thread(
        target=lambda: _capture_error(
            publish_errors,
            publisher.publish_version,
            "v2",
            base_version="v1",
        )
    )
    publish_thread.start()
    deadline = time.monotonic() + 5
    while catalog.get_calls == 0 and time.monotonic() < deadline:
        time.sleep(0.001)

    barrier = threading.Barrier(3)
    close_errors = []

    def close():
        barrier.wait()
        try:
            publisher.deregister()
        except Exception as exc:  # pragma: no cover - asserted below
            close_errors.append(exc)

    closers = [threading.Thread(target=close) for _ in range(2)]
    for closer in closers:
        closer.start()
    barrier.wait()
    for closer in closers:
        closer.join(timeout=5)
    publish_thread.join(timeout=5)

    assert not publish_thread.is_alive()
    assert all(not closer.is_alive() for closer in closers)
    assert close_errors == []
    assert len(publish_errors) == 1
    assert isinstance(publish_errors[0], PublicationCancelled)
    assert events.count("source:close") == 1
    assert events.count("transport:close") == 1
    assert events.count("catalog:close") == 1


def _capture_error(errors, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - asserted by callers
        errors.append(exc)


def test_deregister_cancels_block_wait_leaves_ready_drains_then_closes_resources(
    tmp_path,
):
    events = []
    catalog = _Catalog(events=events)
    raw_transport = FilesystemCanonicalTransport(tmp_path / "objects")
    transport = _RecordingTransport(raw_transport, events)
    source_closed = threading.Event()

    def close_source():
        events.append("source:close")
        source_closed.set()

    publisher, _store, _catalog, _transport = _publisher(
        tmp_path,
        catalog=catalog,
        transport=transport,
        source_close=close_source,
    )
    publisher.initialize(_config(tmp_path, mode=PublicationMode.BLOCK))
    outcome = []

    def publish():
        try:
            publisher.publish_version("v2", base_version="v1")
        except Exception as exc:  # pragma: no cover - asserted below
            outcome.append(exc)

    thread = threading.Thread(target=publish)
    thread.start()
    deadline = time.monotonic() + 5
    while catalog.get_calls == 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    publisher.deregister()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(outcome) == 1 and isinstance(outcome[0], PublicationCancelled)
    assert catalog.records[("model", "v2")].state is RevisionLifecycleState.READY
    assert catalog.commit_calls == 0
    assert source_closed.is_set()
    assert catalog.closed
    assert events.index("source:close") < events.index("transport:close")
    assert events.index("transport:close") < events.index("catalog:close")
    assert list((tmp_path / "objects").rglob("root.json"))
    with pytest.raises(PublisherStateError, match="deregistered"):
        publisher.publish_version("v3", base_version="v1")
    with pytest.raises(TransportClosedError):
        raw_transport.resolve(
            catalog.published[0][0].ranks[0].delta.location,
            catalog.published[0][0].ranks[0].delta.checksum,
            64 * 1024 * 1024,
        )


def test_nonzero_rank_block_wait_is_locally_cancellable_and_drains(tmp_path):
    catalog = _Catalog()

    class Coordinator:
        rank = 1

        def __init__(self):
            self.request = None
            self.intent = None
            self.broadcasts = 0

        def agree(self, value):
            if isinstance(value, publisher_module._PublicationRequest):
                self.request = value
            return value

        def broadcast(self, value):
            assert value is None
            self.broadcasts += 1
            if self.broadcasts == 1:
                self.intent = _synthetic_intent(self.request)
                return publisher_module._PreflightReady(self.intent)
            manifest = RevisionManifest(
                model_id=self.intent.model_id,
                version=self.intent.target_version,
                base_version=self.intent.base_version,
                transfer_method=self.intent.transfer_method,
                delta_method=self.intent.delta_method,
                compression_algorithm=self.intent.compression_algorithm,
                format_digest=self.intent.format_digest,
                base_digest=self.intent.base_digest,
                target_digest="sha256:" + "0" * 64,
                ranks=(
                    RevisionRank(
                        trainer_rank=0,
                        producer_id=self.intent.producer_id,
                        source_layout_digest=self.intent.format_digest,
                        delta=RankDelta(change_state=ChangeState.CLEAN),
                    ),
                ),
            )
            catalog.records[(manifest.model_id, manifest.version)] = RevisionRecord(
                manifest, RevisionLifecycleState.READY
            )
            return publisher_module._ReadyPublication(
                manifest,
                PublishResult(
                    manifest.model_id,
                    manifest.version,
                    RevisionLifecycleState.READY,
                    created=True,
                ),
            )

    publisher, _store, _catalog, _transport = _publisher(
        tmp_path,
        capture=lambda _version, _consume: None,
        catalog=catalog,
        coordinator=Coordinator(),
    )
    publisher.initialize(_config(tmp_path, mode=PublicationMode.BLOCK))
    errors = []
    thread = threading.Thread(
        target=lambda: _capture_error(
            errors,
            publisher.publish_version,
            "v2",
            base_version="v1",
        )
    )
    thread.start()
    deadline = time.monotonic() + 5
    while catalog.get_calls == 0 and time.monotonic() < deadline:
        time.sleep(0.001)

    publisher.deregister()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], PublicationCancelled)
    assert catalog.records[("model", "v2")].state is RevisionLifecycleState.READY
    assert catalog.closed


def test_stale_ready_outcome_cannot_advance_status_or_exact_base(tmp_path, monkeypatch):
    publisher, _store, _catalog, _transport = _publisher(tmp_path)
    publisher.initialize(_config(tmp_path))

    def stale_outcome(prepared, _version):
        manifest = RevisionManifest(
            model_id=prepared.config.model_id,
            version="stale",
            base_version=prepared.base.version,
            transfer_method=DeltaTransferMethod.CANONICAL,
            delta_method=prepared.config.delta_method,
            compression_algorithm=prepared.config.compression_algorithm,
            format_digest=prepared.base.format_digest,
            base_digest=prepared.base.target_digest,
            target_digest="sha256:" + "0" * 64,
            ranks=(
                RevisionRank(
                    trainer_rank=0,
                    producer_id="trainer-0",
                    source_layout_digest=prepared.base.format_digest,
                    delta=RankDelta(change_state=ChangeState.CLEAN),
                ),
            ),
        )
        return publisher_module._ReadyPublication(
            manifest,
            PublishResult("model", "stale", RevisionLifecycleState.READY, created=True),
        )

    monkeypatch.setattr(publisher, "_publish_ready_rank_zero", stale_outcome)
    with pytest.raises(PublisherError, match="agreed intent"):
        publisher.publish_version("v2", base_version="v1")

    assert publisher.status().current_version == "v1"
    assert publisher.status().state is None
    publisher.deregister()


def test_failure_before_verified_root_never_reaches_catalog(tmp_path):
    class FailingTransport(FilesystemCanonicalTransport):
        def publish(self, key, data, checksum):
            if "bucket-" in key:
                raise OSError("child upload failed")
            return super().publish(key, data, checksum)

    catalog = _Catalog()
    publisher, _store, _catalog, _transport = _publisher(
        tmp_path,
        catalog=catalog,
        transport=FailingTransport(tmp_path / "objects"),
    )
    publisher.initialize(_config(tmp_path))

    with pytest.raises(OSError, match="child upload failed"):
        publisher.publish_version("v2", base_version="v1")

    assert catalog.published == []
    assert list((tmp_path / "objects").rglob("root.json")) == []
    publisher.deregister()


def test_nonzero_rank_captures_but_never_publishes_catalog_or_objects(tmp_path):
    captured = []

    def rank_one_capture(version, consume):
        captured.append(version)
        assert consume is not None

    class Coordinator:
        rank = 1

        def __init__(self):
            self.calls = 0
            self.request = None
            self.intent = None

        def agree(self, value):
            if isinstance(value, publisher_module._PublicationRequest):
                self.request = value
            return value

        def broadcast(self, value):
            assert value is None
            self.calls += 1
            if self.calls == 1:
                self.intent = _synthetic_intent(self.request)
                return publisher_module._PreflightReady(self.intent)
            manifest = RevisionManifest(
                model_id=self.intent.model_id,
                version=self.intent.target_version,
                base_version=self.intent.base_version,
                transfer_method=self.intent.transfer_method,
                delta_method=self.intent.delta_method,
                compression_algorithm=self.intent.compression_algorithm,
                format_digest=self.intent.format_digest,
                base_digest=self.intent.base_digest,
                target_digest="sha256:" + "0" * 64,
                ranks=(
                    RevisionRank(
                        trainer_rank=0,
                        producer_id=self.intent.producer_id,
                        source_layout_digest=self.intent.format_digest,
                        delta=RankDelta(change_state=ChangeState.CLEAN),
                    ),
                ),
            )
            return publisher_module._ReadyPublication(
                manifest,
                PublishResult(
                    self.intent.model_id,
                    self.intent.target_version,
                    RevisionLifecycleState.READY,
                    created=True,
                ),
            )

    catalog = _Catalog()
    publisher, _store, _catalog, _transport = _publisher(
        tmp_path,
        capture=rank_one_capture,
        catalog=catalog,
        coordinator=Coordinator(),
    )
    publisher.initialize(_config(tmp_path))
    expected = PublishResult("model", "v2", RevisionLifecycleState.READY, created=True)

    assert publisher.publish_version("v2", base_version="v1") == expected
    assert captured == ["v2"]
    assert catalog.published == []
    assert list((tmp_path / "objects").rglob("*.*")) == []
    publisher.deregister()
