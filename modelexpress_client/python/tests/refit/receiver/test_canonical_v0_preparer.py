# SPDX-License-Identifier: Apache-2.0

import torch
from modelexpress.refit.codec import TENSOR_BYTE_XOR, ZSTD_COMPRESSION
from modelexpress.refit.manifest import (
    ChangeState,
    DeltaTransferMethod,
    RankDelta,
    RevisionLifecycleState,
    RevisionManifest,
    RevisionRank,
    RevisionRecord,
)
from modelexpress.refit.source.canonical import (
    CanonicalDeltaEncoder,
    FilesystemCanonicalBaseStore,
)
from modelexpress.refit.transport.filesystem import FilesystemCanonicalTransport
from safetensors.torch import load_file

import modelexpress.refit.receiver.canonical as canonical_module
from modelexpress.refit.receiver import CanonicalV0Preparer


class CountingTransport:
    def __init__(self, transport):
        self.transport = transport
        self.fetch_count = 0

    def resolve(self, *args):
        return self.transport.resolve(*args)

    def fetch(self, *args):
        self.fetch_count += 1
        return self.transport.fetch(*args)


class Catalog:
    def __init__(self, revision):
        self.revision = revision

    def get_revision(self, model_id, version):
        assert model_id == "policy"
        assert version == "2"
        return self.revision


def test_canonical_preparer_reconstructs_exact_target_and_keeps_path_private(
    tmp_path, monkeypatch
):
    publisher_store = FilesystemCanonicalBaseStore(tmp_path / "publisher-base")
    publisher_base = publisher_store.create_snapshot(
        "1", ((("model.weight", torch.tensor([1.0, 2.0])),),)
    )
    base_store = FilesystemCanonicalBaseStore(tmp_path / "receiver-base")
    base = base_store.create_snapshot(
        "1", ((("model.weight", torch.tensor([1.0, 2.0])),),)
    )
    transport = FilesystemCanonicalTransport(tmp_path / "objects")
    counting_transport = CountingTransport(transport)

    def publish_bucket(bucket):
        stored = transport.publish(
            f"bucket-{bucket.ordinal}.mxcd", bucket.data, bucket.checksum
        )
        return stored.location

    encoder = CanonicalDeltaEncoder(
        model_id="policy",
        target_version="2",
        base_store=publisher_store,
        base=publisher_base,
        delta_method=TENSOR_BYTE_XOR,
        compression_algorithm=ZSTD_COMPRESSION,
        publish_bucket=publish_bucket,
        maximum_encoded_ratio=1024.0,
    )
    encoder.consume_bucket((("model.weight", torch.tensor([3.0, 4.0])),))
    publication = encoder.finish()
    root = transport.publish(
        "root.json", publication.root_bytes, publication.root_checksum
    )
    manifest = RevisionManifest(
        model_id="policy",
        version="2",
        base_version="1",
        transfer_method=DeltaTransferMethod.CANONICAL,
        delta_method=TENSOR_BYTE_XOR,
        compression_algorithm=ZSTD_COMPRESSION,
        format_digest=publication.root_index.format_digest,
        base_digest=publication.root_index.base_digest,
        target_digest=publication.root_index.target_digest,
        ranks=(
            RevisionRank(
                trainer_rank=0,
                producer_id="trainer",
                source_layout_digest=publication.root_index.format_digest,
                delta=RankDelta(
                    change_state=ChangeState.DIRTY,
                    checksum=root.checksum,
                    location=root.location,
                ),
            ),
        ),
    )
    materialize_count = 0
    original_materialize = canonical_module.materialize_snapshot_to_safetensors

    def count_materialization(*args, **kwargs):
        nonlocal materialize_count
        materialize_count += 1
        return original_materialize(*args, **kwargs)

    monkeypatch.setattr(
        canonical_module,
        "materialize_snapshot_to_safetensors",
        count_materialization,
    )
    preparer = CanonicalV0Preparer(
        model_id="policy",
        receiver_incarnation="incarnation",
        model_generation=lambda: 7,
        base_store=base_store,
        base_snapshot=lambda: base,
        target_root=tmp_path / "targets",
        catalog=Catalog(RevisionRecord(manifest, RevisionLifecycleState.READY)),
        transport=counting_transport,
    )

    payload = preparer.prepare("2")

    assert payload.identity.base_version == "1"
    assert payload.identity.base_digest == base.target_digest
    assert payload.identity.target_version == "2"
    assert payload.identity.target_digest == publication.root_index.target_digest
    assert payload.identity.model_generation == 7
    assert (payload.model_path / "model.safetensors").is_file()
    first_fetch_count = counting_transport.fetch_count
    assert first_fetch_count > 0

    second_store = FilesystemCanonicalBaseStore(tmp_path / "receiver-base")
    second = CanonicalV0Preparer(
        model_id="policy",
        receiver_incarnation="second-incarnation",
        model_generation=lambda: 9,
        base_store=second_store,
        base_snapshot=lambda: second_store.open_snapshot("1"),
        target_root=tmp_path / "targets",
        catalog=Catalog(RevisionRecord(manifest, RevisionLifecycleState.READY)),
        transport=counting_transport,
    ).prepare("2")

    assert counting_transport.fetch_count == first_fetch_count
    assert materialize_count == 1
    assert second.model_path == payload.model_path
    assert second.identity.target_digest == payload.identity.target_digest
    assert second.identity.receiver_incarnation == "second-incarnation"
    assert second.identity.model_generation == 9
    assert second_store.attest_snapshot(second_store.open_snapshot("1")) == base
    assert second_store.open_snapshot("2") == second.canonical_snapshot

    checkpoint = payload.model_path / "model.safetensors"
    corrupt = tmp_path / "corrupt.safetensors"
    corrupt.write_bytes(b"corrupt")
    corrupt.replace(checkpoint)
    third_store = FilesystemCanonicalBaseStore(tmp_path / "receiver-base")
    third = CanonicalV0Preparer(
        model_id="policy",
        receiver_incarnation="third-incarnation",
        model_generation=lambda: 10,
        base_store=third_store,
        base_snapshot=lambda: third_store.open_snapshot("1"),
        target_root=tmp_path / "targets",
        catalog=Catalog(RevisionRecord(manifest, RevisionLifecycleState.READY)),
        transport=counting_transport,
    ).prepare("2")

    assert counting_transport.fetch_count == first_fetch_count
    assert materialize_count == 2
    assert third.model_path == payload.model_path
    assert torch.equal(load_file(checkpoint)["model.weight"], torch.tensor([3.0, 4.0]))


def test_clean_canonical_revision_is_prepared_without_object_fetch(tmp_path):
    base_store = FilesystemCanonicalBaseStore(tmp_path / "base")
    base = base_store.create_snapshot(
        "1", ((("model.weight", torch.tensor([1.0, 2.0])),),)
    )
    manifest = RevisionManifest(
        model_id="policy",
        version="2",
        base_version="1",
        transfer_method=DeltaTransferMethod.CANONICAL,
        delta_method=TENSOR_BYTE_XOR,
        compression_algorithm=ZSTD_COMPRESSION,
        format_digest=base.format_digest,
        base_digest=base.target_digest,
        target_digest=base.target_digest,
        ranks=(
            RevisionRank(
                trainer_rank=0,
                producer_id="trainer",
                source_layout_digest=base.format_digest,
                delta=RankDelta(change_state=ChangeState.CLEAN),
            ),
        ),
    )

    class NoFetchTransport:
        def resolve(self, *_args):
            raise AssertionError("clean revision must not resolve an object")

        def fetch(self, *_args):
            raise AssertionError("clean revision must not fetch an object")

    preparer = CanonicalV0Preparer(
        model_id="policy",
        receiver_incarnation="incarnation",
        model_generation=lambda: 7,
        base_store=base_store,
        base_snapshot=lambda: base,
        target_root=tmp_path / "targets",
        catalog=Catalog(RevisionRecord(manifest, RevisionLifecycleState.READY)),
        transport=NoFetchTransport(),
    )

    payload = preparer.prepare("2")

    assert payload.noop is True
    assert payload.identity.target_version == "2"
    assert payload.identity.target_digest == base.target_digest
    assert base_store.open_snapshot("2") == payload.canonical_snapshot
