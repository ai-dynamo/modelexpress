# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import io

import pytest
import torch
from safetensors.torch import save_file

from modelexpress.refit import S3Config
from modelexpress.refit.s3 import ImmutableS3Conflict, S3Uploader
from modelexpress.refit.source.canonical import load_hf_snapshot


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
