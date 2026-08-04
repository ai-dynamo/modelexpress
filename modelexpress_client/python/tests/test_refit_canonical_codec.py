# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-base canonical codec, framing, coverage, and attestation tests."""

from __future__ import annotations

import json
import os
import re
import threading

import pytest
import torch
import zstandard

import modelexpress.refit.source.canonical as canonical_module
from modelexpress.refit.codec import (
    NO_COMPRESSION,
    TENSOR_BYTE_XOR,
    ZSTD_COMPRESSION,
    CodecError,
    compress_payload,
    crc32c_hex,
    decode_delta,
    decompress_payload,
    encode_delta,
)
from modelexpress.refit.manifest import DeltaLocation, FilesystemLocation
from modelexpress.refit.source.canonical import (
    CanonicalDeltaEncoder,
    CanonicalDeltaError,
    CanonicalFormatIdentity,
    CanonicalSnapshot,
    FilesystemCanonicalBaseStore,
    decode_root_index,
    reconstruct_canonical_delta,
)


def _seed(store: FilesystemCanonicalBaseStore):
    return store.create_snapshot(
        "v1",
        (
            (
                ("a.weight", torch.zeros(256, dtype=torch.float32)),
                ("b.weight", torch.arange(8, dtype=torch.float32)),
            ),
        ),
    )


def _assert_fifo_call_finishes(call, fifo_path):
    errors = []

    def invoke():
        try:
            call()
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    thread = threading.Thread(target=invoke, daemon=True)
    thread.start()
    thread.join(timeout=0.25)
    completed_without_writer = not thread.is_alive()
    if not completed_without_writer:
        writer = os.open(fifo_path, os.O_WRONLY | os.O_NONBLOCK)
        os.close(writer)
        thread.join(timeout=1)
    assert completed_without_writer, "canonical base FIFO blocked before fstat"
    assert len(errors) == 1
    assert isinstance(errors[0], CanonicalDeltaError)


def test_format_digest_binds_normalization_quantization_and_atomic_groups(tmp_path):
    identities = (
        CanonicalFormatIdentity(),
        CanonicalFormatIdentity(normalization_profile="hf-save-pretrained-v2"),
        CanonicalFormatIdentity(quantization_profile="fp8-e4m3fn-v1"),
        CanonicalFormatIdentity(atomic_groups=(("a.weight", "b.weight"),)),
    )
    snapshots = []
    for index, identity in enumerate(identities):
        store = FilesystemCanonicalBaseStore(tmp_path / f"store-{index}")
        snapshots.append(
            store.create_snapshot(
                "v1",
                (
                    (
                        ("a.weight", torch.zeros(2)),
                        ("b.weight", torch.ones(2)),
                    ),
                ),
                format_identity=identity,
            )
        )

    assert len({snapshot.format_digest for snapshot in snapshots}) == len(identities)
    assert len({snapshot.target_digest for snapshot in snapshots}) == 1
    assert [snapshot.format_identity for snapshot in snapshots] == list(identities)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO required")
def test_exact_base_reader_rejects_fifo_without_blocking(tmp_path):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    _seed(store)
    snapshot_path = store._snapshot_path("v1")
    snapshot_path.unlink()
    os.mkfifo(snapshot_path)

    _assert_fifo_call_finishes(lambda: store.open_snapshot("v1"), snapshot_path)


def _encode(store, base, buckets, *, maximum_ratio=8.0):
    objects = {}

    def publish_bucket(bucket):
        objects[bucket.ordinal] = bucket.data
        return DeltaLocation(
            filesystem=FilesystemLocation(
                path=f"objects/bucket-{bucket.ordinal:08d}.mxcd"
            )
        )

    encoder = CanonicalDeltaEncoder(
        model_id="model",
        target_version="v2",
        base_store=store,
        base=base,
        delta_method=TENSOR_BYTE_XOR,
        compression_algorithm=ZSTD_COMPRESSION,
        publish_bucket=publish_bucket,
        maximum_encoded_ratio=maximum_ratio,
    )
    for bucket in buckets:
        encoder.consume_bucket(bucket)
    return encoder.finish(), objects


def test_delta_and_compression_are_separate_versioned_algorithms():
    base = bytes.fromhex("00112233")
    target = bytes.fromhex("10213243")

    delta = encode_delta(TENSOR_BYTE_XOR, base, target)

    assert delta == bytes.fromhex("10301070")
    assert decode_delta(TENSOR_BYTE_XOR, base, delta) == target
    assert (
        decompress_payload(
            NO_COMPRESSION,
            compress_payload(NO_COMPRESSION, delta),
            expected_size=len(delta),
        )
        == delta
    )
    assert (
        decompress_payload(
            ZSTD_COMPRESSION,
            compress_payload(ZSTD_COMPRESSION, delta),
            expected_size=len(delta),
        )
        == delta
    )
    with pytest.raises(CodecError, match="same byte length"):
        encode_delta(TENSOR_BYTE_XOR, b"a", b"ab")
    with pytest.raises(CodecError, match="unsupported delta_method"):
        encode_delta("xor-ish", base, target)
    with pytest.raises(CodecError, match="decoded size"):
        decompress_payload(NO_COMPRESSION, delta, expected_size=99)
    assert compress_payload(
        ZSTD_COMPRESSION, b"MX deterministic payload"
    ) == compress_payload(ZSTD_COMPRESSION, b"MX deterministic payload")


def test_zstd_decode_rejects_declared_size_bombs_and_trailing_bytes():
    oversized = zstandard.ZstdCompressor(write_content_size=True).compress(
        b"x" * 1_000_000
    )
    with pytest.raises(CodecError, match="decoded size"):
        decompress_payload(ZSTD_COMPRESSION, oversized, expected_size=10)

    encoded = compress_payload(ZSTD_COMPRESSION, b"payload")
    with pytest.raises(CodecError, match="zstd payload"):
        decompress_payload(
            ZSTD_COMPRESSION,
            encoded + b"trailing",
            expected_size=len(b"payload"),
        )


def test_crc32c_is_the_bare_lowercase_wire_checksum():
    assert crc32c_hex(b"123456789") == "e3069283"
    assert re.fullmatch(r"[0-9a-f]{8}", crc32c_hex(b"physical bytes"))


def test_canonical_framing_is_deterministic_complete_and_reconstructs_exact_target(
    tmp_path,
):
    store = FilesystemCanonicalBaseStore(tmp_path / "base-store")
    base = _seed(store)
    target = (
        ("a.weight", torch.cat((torch.ones(1), torch.zeros(255)))),
        ("b.weight", torch.arange(8, dtype=torch.float32)),
    )

    publication, objects = _encode(store, base, (target,))
    repeated, repeated_objects = _encode(store, base, (target,))

    assert publication.changed
    assert publication.root_bytes == repeated.root_bytes
    assert objects == repeated_objects
    assert publication.root_checksum == crc32c_hex(publication.root_bytes)
    assert publication.root_checksum and ":" not in publication.root_checksum
    assert publication.root_index.base_digest == base.target_digest
    assert publication.root_index.format_digest == base.format_digest
    assert publication.root_index.target_digest.startswith("sha256:")
    assert [item.name for item in publication.root_index.tensors] == [
        "a.weight",
        "b.weight",
    ]
    assert [item.change_state for item in publication.root_index.tensors] == [
        "DIRTY",
        "CLEAN",
    ]
    assert publication.root_index.buckets[0].checksum == crc32c_hex(objects[0])
    assert base.format_digest == (
        "sha256:c8e3b8d5b628e9ec956c5aa14ad2ab20ebc5e6738dd8098dc455723bd1fcf44e"
    )
    assert base.target_digest == (
        "sha256:70c2f4408f0cf0258073479046c930c4786b9550fd37755cc7617e3baa7244e9"
    )
    assert publication.root_index.target_digest == (
        "sha256:149f9fdf38a42e59374f9e0eb75f445826935dfbcb8e1621747815ccbb6004df"
    )
    reconstructed_store = FilesystemCanonicalBaseStore(tmp_path / "reconstructed")
    reconstructed = reconstruct_canonical_delta(
        root_bytes=publication.root_bytes,
        expected_root_checksum=publication.root_checksum,
        base_store=store,
        base=base,
        target_store=reconstructed_store,
        fetch_bucket=lambda reference: objects[reference.ordinal],
    )

    assert reconstructed.version == "v2"
    for name, tensor in target:
        assert reconstructed_store.read_tensor_bytes(reconstructed, name) == (
            tensor.contiguous().view(torch.uint8).numpy().tobytes()
        )
    assert reconstructed.target_digest == publication.root_index.target_digest


def test_clean_target_has_no_root_payload_or_bucket_reference(tmp_path):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    base = _seed(store)
    clean_bucket = tuple(
        (metadata.name, store.read_tensor(base, metadata.name))
        for metadata in base.tensors
    )

    publication, objects = _encode(store, base, (clean_bucket,))

    assert not publication.changed
    assert publication.root_bytes is None
    assert publication.root_checksum is None
    assert publication.root_index.buckets == ()
    assert publication.target_snapshot.target_digest == base.target_digest
    assert objects == {}


def test_encoder_rejects_a_fabricated_snapshot_aggregate_attestation(tmp_path):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    base = _seed(store)
    forged = CanonicalSnapshot(
        version=base.version,
        format_digest=base.format_digest,
        target_digest=f"sha256:{'0' * 64}",
        tensors=base.tensors,
    )

    with pytest.raises(CanonicalDeltaError, match="snapshot attestation"):
        _encode(
            store,
            forged,
            (
                (
                    ("a.weight", torch.ones(256)),
                    ("b.weight", torch.arange(8, dtype=torch.float32)),
                ),
            ),
        )


def test_scalar_tensor_has_complete_exact_byte_coverage(tmp_path):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    base = store.create_snapshot("v1", ((("scalar", torch.tensor(1.25)),),))

    publication, objects = _encode(
        store,
        base,
        ((("scalar", torch.tensor(2.5)),),),
        maximum_ratio=1024,
    )
    reconstructed_store = FilesystemCanonicalBaseStore(tmp_path / "target")
    reconstructed = reconstruct_canonical_delta(
        root_bytes=publication.root_bytes,
        expected_root_checksum=publication.root_checksum,
        base_store=store,
        base=base,
        target_store=reconstructed_store,
        fetch_bucket=lambda reference: objects[reference.ordinal],
    )

    assert reconstructed.tensors[0].shape == ()
    assert reconstructed_store.read_tensor(reconstructed, "scalar").item() == 2.5


def test_root_checksum_is_verified_before_parse_and_bucket_checksum_before_decode(
    tmp_path,
):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    base = _seed(store)
    publication, objects = _encode(
        store,
        base,
        (
            (
                ("a.weight", torch.ones(256)),
                ("b.weight", torch.arange(8, dtype=torch.float32)),
            ),
        ),
    )

    corrupt_root = publication.root_bytes[:-1] + bytes([publication.root_bytes[-1] ^ 1])
    with pytest.raises(CanonicalDeltaError, match="root checksum"):
        decode_root_index(corrupt_root, publication.root_checksum)

    corrupt_objects = dict(objects)
    corrupt_objects[0] = objects[0][:-1] + bytes([objects[0][-1] ^ 1])
    with pytest.raises(CanonicalDeltaError, match="bucket checksum"):
        reconstruct_canonical_delta(
            root_bytes=publication.root_bytes,
            expected_root_checksum=publication.root_checksum,
            base_store=store,
            base=base,
            target_store=FilesystemCanonicalBaseStore(tmp_path / "bad-target"),
            fetch_bucket=lambda reference: corrupt_objects[reference.ordinal],
        )


def test_root_json_rejects_duplicate_fields_even_with_a_valid_physical_checksum(
    tmp_path,
):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    base = _seed(store)
    publication, _objects = _encode(
        store,
        base,
        (
            (
                ("a.weight", torch.ones(256)),
                ("b.weight", torch.arange(8, dtype=torch.float32)),
            ),
        ),
    )
    duplicate = publication.root_bytes.replace(
        b'"model_id":"model"',
        b'"model_id":"model","model_id":"other"',
        1,
    )

    with pytest.raises(CanonicalDeltaError, match="duplicate JSON field"):
        decode_root_index(duplicate, crc32c_hex(duplicate))


def test_bucket_bounds_are_enforced_before_fetch_or_unbounded_encode(tmp_path):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    base = _seed(store)
    encoder = CanonicalDeltaEncoder(
        model_id="model",
        target_version="bounded-v2",
        base_store=store,
        base=base,
        delta_method=TENSOR_BYTE_XOR,
        compression_algorithm=ZSTD_COMPRESSION,
        publish_bucket=lambda _bucket: pytest.fail("oversized bucket must not publish"),
        maximum_bucket_bytes=16,
    )
    with pytest.raises(CanonicalDeltaError, match="maximum decoded size"):
        encoder.consume_bucket(
            (
                ("a.weight", torch.ones(256)),
                ("b.weight", torch.arange(8, dtype=torch.float32)),
            )
        )

    publication, objects = _encode(
        store,
        base,
        (
            (
                ("a.weight", torch.ones(256)),
                ("b.weight", torch.arange(8, dtype=torch.float32)),
            ),
        ),
    )
    fetches = []
    with pytest.raises(CanonicalDeltaError, match="encoded or decoded size"):
        reconstruct_canonical_delta(
            root_bytes=publication.root_bytes,
            expected_root_checksum=publication.root_checksum,
            base_store=store,
            base=base,
            target_store=FilesystemCanonicalBaseStore(tmp_path / "target"),
            fetch_bucket=lambda reference: fetches.append(reference),
            maximum_bucket_bytes=16,
        )
    assert fetches == []


def test_economic_ratio_must_be_finite(tmp_path):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    base = _seed(store)
    with pytest.raises(ValueError, match="finite and positive"):
        CanonicalDeltaEncoder(
            model_id="model",
            target_version="v2",
            base_store=store,
            base=base,
            delta_method=TENSOR_BYTE_XOR,
            compression_algorithm=ZSTD_COMPRESSION,
            publish_bucket=lambda _bucket: pytest.fail("must not publish"),
            maximum_encoded_ratio=float("nan"),
        )


def test_snapshot_writer_never_creates_an_index_its_reader_rejects(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(canonical_module, "_MAXIMUM_RECORDS", 1)
    store = FilesystemCanonicalBaseStore(tmp_path / "store")

    with pytest.raises(CanonicalDeltaError, match="too many tensors"):
        store.create_snapshot(
            "v1",
            (
                (
                    ("a", torch.tensor([1])),
                    ("b", torch.tensor([2])),
                ),
            ),
        )

    assert list((tmp_path / "store" / "snapshots").iterdir()) == []


def test_base_store_bounds_snapshot_index_reads_before_parsing(tmp_path, monkeypatch):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    _seed(store)
    requested_sizes = []
    real_read = canonical_module.os.read

    def bounded_read(descriptor, size):
        requested_sizes.append(size)
        return real_read(descriptor, size)

    monkeypatch.setattr(canonical_module, "_DEFAULT_MAXIMUM_ROOT_BYTES", 32)
    monkeypatch.setattr(canonical_module.os, "read", bounded_read)

    with pytest.raises(CanonicalDeltaError, match="index exceeds maximum size"):
        store.open_snapshot("v1")

    assert requested_sizes
    assert max(requested_sizes) <= 33


def test_base_store_bounds_tensor_blob_reads_to_declared_size(tmp_path, monkeypatch):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    base = _seed(store)
    metadata = base.tensors[0]
    (tmp_path / "store" / "blobs" / metadata.blob).write_bytes(
        b"x" * (metadata.byte_size + 100)
    )
    requested_sizes = []
    real_read = canonical_module.os.read

    def bounded_read(descriptor, size):
        requested_sizes.append(size)
        return real_read(descriptor, size)

    monkeypatch.setattr(canonical_module.os, "read", bounded_read)

    with pytest.raises(CanonicalDeltaError, match="content verification"):
        store.read_tensor_bytes(base, metadata.name)

    assert requested_sizes
    assert max(requested_sizes) <= metadata.byte_size + 1


def test_base_store_bounds_existing_blob_reads_during_immutable_retry(
    tmp_path, monkeypatch
):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    base = _seed(store)
    metadata = base.tensors[0]
    original = torch.zeros(256, dtype=torch.float32).view(torch.uint8).numpy().tobytes()
    (tmp_path / "store" / "blobs" / metadata.blob).write_bytes(
        original + b"unexpected trailing bytes"
    )
    requested_sizes = []
    real_read = canonical_module.os.read

    def bounded_read(descriptor, size):
        requested_sizes.append(size)
        return real_read(descriptor, size)

    monkeypatch.setattr(canonical_module.os, "read", bounded_read)

    with pytest.raises(CanonicalDeltaError, match="immutable canonical blob conflict"):
        store._ensure_blob(metadata.blob, original)

    assert requested_sizes
    assert max(requested_sizes) <= len(original) + 1


def test_base_store_bounds_existing_snapshot_reads_during_immutable_retry(
    tmp_path, monkeypatch
):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    base = _seed(store)
    encoded = canonical_module._snapshot_bytes(base)
    store._snapshot_path(base.version).write_bytes(encoded + b"trailing bytes")
    requested_sizes = []
    real_read = canonical_module.os.read

    def bounded_read(descriptor, size):
        requested_sizes.append(size)
        return real_read(descriptor, size)

    monkeypatch.setattr(canonical_module.os, "read", bounded_read)

    with pytest.raises(
        CanonicalDeltaError, match="immutable canonical snapshot conflict"
    ):
        store._write_snapshot(base)

    assert requested_sizes
    assert max(requested_sizes) <= len(encoded) + 1


def test_wrong_base_incomplete_coverage_and_uneconomic_deltas_fail_closed(tmp_path):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    base = _seed(store)

    encoder = CanonicalDeltaEncoder(
        model_id="model",
        target_version="v2",
        base_store=store,
        base=base,
        delta_method=TENSOR_BYTE_XOR,
        compression_algorithm=ZSTD_COMPRESSION,
        publish_bucket=lambda _bucket: DeltaLocation(
            filesystem=FilesystemLocation(path="unused")
        ),
    )
    encoder.consume_bucket((("a.weight", torch.zeros(256)),))
    with pytest.raises(CanonicalDeltaError, match="complete coverage"):
        encoder.finish()

    wrong_store = FilesystemCanonicalBaseStore(tmp_path / "wrong-store")
    wrong_base = wrong_store.create_snapshot(
        "v1",
        ((("a.weight", torch.ones(1)),),),
    )
    publication, objects = _encode(
        store,
        base,
        (
            (
                ("a.weight", torch.ones(256)),
                ("b.weight", torch.arange(8, dtype=torch.float32)),
            ),
        ),
    )
    with pytest.raises(CanonicalDeltaError, match="base format digest"):
        reconstruct_canonical_delta(
            root_bytes=publication.root_bytes,
            expected_root_checksum=publication.root_checksum,
            base_store=wrong_store,
            base=wrong_base,
            target_store=FilesystemCanonicalBaseStore(tmp_path / "wrong-target"),
            fetch_bucket=lambda reference: objects[reference.ordinal],
        )

    noisy = torch.arange(256, dtype=torch.float32).sin()
    with pytest.raises(CanonicalDeltaError, match="uneconomic"):
        _encode(
            store,
            base,
            (
                (
                    ("a.weight", noisy),
                    ("b.weight", torch.arange(8, dtype=torch.float32)),
                ),
            ),
            maximum_ratio=0.01,
        )


def test_economic_gate_counts_framing_and_root_bytes_against_the_complete_target(
    tmp_path,
):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    base = _seed(store)

    with pytest.raises(CanonicalDeltaError, match="uneconomic"):
        _encode(
            store,
            base,
            (
                (
                    ("a.weight", torch.cat((torch.ones(1), torch.zeros(255)))),
                    ("b.weight", torch.arange(8, dtype=torch.float32)),
                ),
            ),
            maximum_ratio=0.5,
        )


def test_root_rejects_non_monotonic_bucket_coverage_before_fetch(tmp_path):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    base = _seed(store)
    publication, _objects = _encode(
        store,
        base,
        (
            (("a.weight", torch.ones(256)),),
            (("b.weight", torch.arange(8, dtype=torch.float32) + 1),),
        ),
    )
    document = json.loads(publication.root_bytes)
    document["tensors"][0]["bucket_ordinal"] = 1
    document["tensors"][1]["bucket_ordinal"] = 0
    document["buckets"][0]["tensor_names"] = ["b.weight"]
    document["buckets"][1]["tensor_names"] = ["a.weight"]
    malformed = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()

    with pytest.raises(CanonicalDeltaError, match="bucket coverage order"):
        decode_root_index(malformed, crc32c_hex(malformed))


def test_reconstruction_verifies_aggregate_digest_before_immutable_snapshot_seal(
    tmp_path,
):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    base = _seed(store)
    publication, objects = _encode(
        store,
        base,
        (
            (
                ("a.weight", torch.ones(256)),
                ("b.weight", torch.arange(8, dtype=torch.float32)),
            ),
        ),
    )
    document = json.loads(publication.root_bytes)
    document["target_digest"] = "sha256:" + "0" * 64
    forged = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    target_store = FilesystemCanonicalBaseStore(tmp_path / "target")

    with pytest.raises(CanonicalDeltaError, match="target digest"):
        reconstruct_canonical_delta(
            root_bytes=forged,
            expected_root_checksum=crc32c_hex(forged),
            base_store=store,
            base=base,
            target_store=target_store,
            fetch_bucket=lambda reference: objects[reference.ordinal],
        )
    with pytest.raises(CanonicalDeltaError, match="unavailable"):
        target_store.open_snapshot("v2")


def test_missing_clean_base_blob_fails_instead_of_being_recreated_from_target(tmp_path):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    base = _seed(store)
    (tmp_path / "store" / "blobs" / base.tensors[0].blob).unlink()
    encoder = CanonicalDeltaEncoder(
        model_id="model",
        target_version="v2",
        base_store=store,
        base=base,
        delta_method=TENSOR_BYTE_XOR,
        compression_algorithm=ZSTD_COMPRESSION,
        publish_bucket=lambda _bucket: pytest.fail(
            "clean target must not publish a bucket"
        ),
    )

    with pytest.raises(CanonicalDeltaError, match="missing"):
        encoder.consume_bucket(
            (
                ("a.weight", torch.zeros(256, dtype=torch.float32)),
                ("b.weight", torch.arange(8, dtype=torch.float32)),
            )
        )


def test_decoder_rejects_unbounded_or_malformed_root_metadata(tmp_path):
    store = FilesystemCanonicalBaseStore(tmp_path / "store")
    base = _seed(store)
    publication, _objects = _encode(
        store,
        base,
        (
            (
                ("a.weight", torch.ones(256)),
                ("b.weight", torch.arange(8, dtype=torch.float32)),
            ),
        ),
    )

    with pytest.raises(CanonicalDeltaError, match="maximum root index size"):
        decode_root_index(
            publication.root_bytes,
            publication.root_checksum,
            maximum_bytes=len(publication.root_bytes) - 1,
        )
