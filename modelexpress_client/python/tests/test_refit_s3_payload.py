# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import hashlib
import io
import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from modelexpress.refit import S3Config
from modelexpress.refit.s3 import ImmutableS3Conflict, S3Uploader
from modelexpress.refit.source import canonical
from modelexpress.refit.source.canonical import (
    decode_bucket,
    encode_bucket,
    load_hf_snapshot,
)


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.puts = []
        self.head_calls = 0
        self.get_calls = 0

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        identity = (kwargs["Bucket"], kwargs["Key"])
        if identity in self.objects:
            error = RuntimeError("precondition failed")
            error.response = {"Error": {"Code": "PreconditionFailed"}}
            raise error
        data = bytes(kwargs["Body"])
        self.objects[identity] = (data, kwargs["ChecksumCRC32C"], "version-1")
        return {"VersionId": "version-1"}

    def head_object(self, **kwargs):
        self.head_calls += 1
        data, checksum, version = self.objects[(kwargs["Bucket"], kwargs["Key"])]
        return {
            "ContentLength": len(data),
            "ChecksumCRC32C": checksum,
            "VersionId": version,
        }

    def get_object(self, **kwargs):
        self.get_calls += 1
        data, _checksum, version = self.objects[(kwargs["Bucket"], kwargs["Key"])]
        return {"Body": io.BytesIO(data), "VersionId": version}


def test_s3_uploader_uses_one_conditional_put_and_returns_object():
    client = FakeS3()
    uploader = S3Uploader(S3Config(bucket="bucket", prefix="run"), client=client)

    stored = uploader.put("models/m/revisions/1/root.json", b"root")

    assert stored.bucket == "bucket"
    assert stored.key == "run/models/m/revisions/1/root.json"
    assert stored.object_version == "version-1"
    assert stored.checksum.startswith("crc32c:")
    request = client.puts[0]
    assert request["IfNoneMatch"] == "*"
    assert request["ChecksumAlgorithm"] == "CRC32C"
    assert base64.b64decode(request["ChecksumCRC32C"])
    assert client.head_calls == 0
    assert client.get_calls == 0


def test_s3_uploader_allows_identical_retry_but_rejects_immutable_conflict():
    client = FakeS3()
    uploader = S3Uploader(S3Config(bucket="bucket"), client=client)

    first = uploader.put("root.json", b"same")
    assert uploader.put("root.json", b"same") == first
    with pytest.raises(ImmutableS3Conflict):
        uploader.put("root.json", b"different")


def test_launch_schema_orders_names_after_canonical_prefix_normalization(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    save_file(
        {
            "a.weight": torch.tensor([1.0]),
            "module.0.weight": torch.tensor([2.0]),
        },
        checkpoint,
    )

    snapshot, metadata, _format_digest, _target_digest = load_hf_snapshot(checkpoint)

    assert sorted(snapshot) == ["0.weight", "a.weight"]
    assert sorted(metadata) == ["0.weight", "a.weight"]


@pytest.mark.parametrize(
    ("tied", "expected"),
    [
        (True, ["model.embed_tokens.weight"]),
        (False, ["lm_head.weight", "model.embed_tokens.weight"]),
    ],
)
def test_tied_output_head_is_dropped_from_the_canonical_set(tmp_path, tied, expected):
    # HF serializes a tied output head as a second copy of the input embedding, but the
    # trainer holds one parameter for both and never gathers the copy, so counting it
    # would leave an unreachable tensor in the canonical set.
    embedding = torch.arange(4, dtype=torch.float32)
    save_file(
        {"lm_head.weight": embedding.clone(), "model.embed_tokens.weight": embedding},
        tmp_path / "model.safetensors",
    )
    (tmp_path / "config.json").write_text(json.dumps({"tie_word_embeddings": tied}))

    snapshot, metadata, _format_digest, _target_digest = load_hf_snapshot(tmp_path)

    assert sorted(snapshot) == expected
    assert sorted(metadata) == expected


def test_pack_source_rank_raw_deltas_into_canonical_bucket(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    launch = torch.arange(4, dtype=torch.float32)
    save_file({"model.weight": launch}, checkpoint)
    snapshot, metadata, format_digest, base_digest = load_hf_snapshot(checkpoint)
    target = launch + 1
    old = snapshot["model.weight"].tobytes()
    new = target.contiguous().view(torch.uint8).numpy().tobytes()
    delta = np.frombuffer(
        bytes(left ^ right for left, right in zip(old, new, strict=True)),
        dtype=np.uint8,
    )
    metadata["model.weight"]["target_digest"] = (
        f"sha256:{hashlib.sha256(new).hexdigest()}"
    )

    encoded, decoded_size = encode_bucket(
        model_id="model",
        base_version="0",
        target_version="1",
        base_digest=base_digest,
        format_digest=format_digest,
        ordinal=3,
        tensors=[("model.weight", delta)],
        metadata=metadata,
    )

    restored, restored_metadata, _format, _digest = load_hf_snapshot(checkpoint)
    header = decode_bucket(encoded, restored, restored_metadata)
    assert header["ordinal"] == 3
    assert decoded_size == len(delta)
    assert restored["model.weight"].tobytes() == new


def test_bucket_streams_numpy_deltas_without_decoded_copy(monkeypatch):
    seen = []

    class Compressor:
        def compressobj(self):
            return self

        def compress(self, data):
            seen.append(data)
            return b"frame"

        def flush(self):
            return b"end"

    monkeypatch.setattr(
        canonical.zstandard, "ZstdCompressor", lambda level: Compressor()
    )
    first = np.arange(4, dtype=np.uint8)
    second = np.arange(3, dtype=np.uint8)
    metadata = {
        "first": {
            "name": "first",
            "shape": [4],
            "dtype": "uint8",
            "byte_size": first.nbytes,
            "target_digest": "sha256:first",
        },
        "second": {
            "name": "second",
            "shape": [3],
            "dtype": "uint8",
            "byte_size": second.nbytes,
            "target_digest": "sha256:second",
        },
    }

    encoded, decoded_size = encode_bucket(
        model_id="model",
        base_version="0",
        target_version="1",
        base_digest="sha256:base",
        format_digest="sha256:format",
        ordinal=0,
        tensors=[("first", first), ("second", second)],
        metadata=metadata,
    )

    header, compressed = canonical.bucket_parts(encoded)
    assert all(isinstance(data, memoryview) for data in seen)
    assert [data.obj for data in seen] == [first, second]
    assert decoded_size == first.nbytes + second.nbytes
    assert [entry["offset"] for entry in header["entries"]] == [0, first.nbytes]
    assert bytes(compressed) == b"frameframeend"
