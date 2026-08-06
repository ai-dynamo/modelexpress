# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json

import pytest
import torch
from safetensors.torch import save_file

from modelexpress.refit import (
    Publisher,
    PublisherConfig,
    RevisionRecord,
    RevisionState,
    S3Config,
)
from modelexpress.refit.source.canonical import decode_bucket, load_hf_snapshot


class FakeCatalog:
    def __init__(self):
        self.records = {}
        self.published = []
        self.gets = {}

    def publish_revision(self, manifest):
        self.published.append(manifest)
        record = RevisionRecord(manifest, RevisionState.READY)
        self.records[(manifest.model_id, manifest.target_version)] = record
        return record

    def get_revision(self, model_id, version):
        key = (model_id, version)
        self.gets[key] = self.gets.get(key, 0) + 1
        record = self.records[key]
        if self.gets[key] >= 1:
            record = RevisionRecord(record.manifest, RevisionState.COMMITTED)
            self.records[key] = record
        return record

    def close(self):
        pass


class FakeS3:
    def __init__(self):
        self.objects = {}
        self.puts = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        key = (kwargs["Bucket"], kwargs["Key"])
        self.objects[key] = (
            bytes(kwargs["Body"]),
            kwargs["ChecksumCRC32C"],
            f"version-{len(self.puts)}",
        )
        return {"VersionId": self.objects[key][2]}

    def head_object(self, **kwargs):
        data, checksum, version = self.objects[(kwargs["Bucket"], kwargs["Key"])]
        return {
            "ContentLength": len(data),
            "ChecksumCRC32C": checksum,
            "VersionId": version,
        }

    def get_object(self, **kwargs):
        data, _checksum, version = self.objects[(kwargs["Bucket"], kwargs["Key"])]
        return {"Body": io.BytesIO(data), "VersionId": version}


def checkpoint(tmp_path):
    path = tmp_path / "hf"
    path.mkdir()
    tensors = {
        "model.a.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "model.b.weight": torch.ones((2, 2), dtype=torch.float32),
    }
    save_file(tensors, path / "model.safetensors")
    return path, tensors


def make_publisher(tmp_path, catalog, s3):
    hf_path, _weights = checkpoint(tmp_path)
    publisher = Publisher(
        launch_checkpoint=hf_path,
        bucket_bytes=64,
        catalog=catalog,
        s3_client=s3,
        sleep=lambda _seconds: None,
    )
    publisher.initialize(PublisherConfig("model", "mx:8001", S3Config("bucket", "run")))
    snapshot, _metadata, _format, _digest = load_hf_snapshot(hf_path)
    publisher.capture_baseline(gather(_weights), lambda name: snapshot[name])
    return publisher


def gather(weights):
    def run(encode_bucket):
        encode_bucket(list(weights.items()), None)

    return run


def test_launch_zero_publishes_metadata_without_weights(tmp_path):
    catalog = FakeCatalog()
    s3 = FakeS3()
    publisher = make_publisher(tmp_path, catalog, s3)

    publisher.publish_version("0")

    manifest = catalog.published[-1]
    assert manifest.target_version == "0"
    assert manifest.base_version is None
    assert manifest.payload is None
    assert s3.puts == []


def test_exact_base_update_uses_miles_hf_buckets_and_uploads_delta(tmp_path):
    catalog = FakeCatalog()
    s3 = FakeS3()
    hf_path, launch = checkpoint(tmp_path)
    target = {name: tensor.clone() for name, tensor in launch.items()}
    target["model.b.weight"] += 7
    publisher = Publisher(
        launch_checkpoint=hf_path,
        bucket_bytes=64,
        catalog=catalog,
        s3_client=s3,
        sleep=lambda _seconds: None,
    )
    publisher.initialize(PublisherConfig("model", "mx:8001", S3Config("bucket", "run")))
    snapshot, _metadata, _format, _digest = load_hf_snapshot(hf_path)
    publisher.capture_baseline(gather(launch), lambda name: snapshot[name])
    publisher.publish_version("0")
    publisher.wait_for_commit("0")

    publisher.publish_version("1", base_version="0", gather_hf_buckets=gather(target))
    publisher.wait_for_commit("1")

    assert publisher.current_version == "1"
    manifest = catalog.published[-1]
    root = json.loads(s3.objects[(manifest.payload.bucket, manifest.payload.key)][0])
    assert root["base_version"] == "0"
    assert root["target_version"] == "1"
    [bucket] = root["buckets"]
    encoded = s3.objects[(bucket["object"]["bucket"], bucket["object"]["key"])][0]
    snapshot, metadata, _format_digest, _base_digest = load_hf_snapshot(hf_path)
    decode_bucket(encoded, snapshot, metadata)
    for name, tensor in target.items():
        assert (
            snapshot[name].tobytes()
            == tensor.contiguous().view(torch.uint8).numpy().tobytes()
        )


def test_wrong_base_is_rejected_before_gather(tmp_path):
    publisher = make_publisher(tmp_path, FakeCatalog(), FakeS3())
    publisher.publish_version("0")
    called = False

    def should_not_run(_encode):
        nonlocal called
        called = True

    with pytest.raises(RuntimeError, match="base"):
        publisher.publish_version(
            "2", base_version="1", gather_hf_buckets=should_not_run
        )

    assert not called


def test_s3_failure_prevents_catalog_publication(tmp_path):
    class FailingS3(FakeS3):
        fail = False

        def put_object(self, **kwargs):
            if self.fail:
                raise RuntimeError("upload failed")
            return super().put_object(**kwargs)

    catalog = FakeCatalog()
    s3 = FailingS3()
    hf_path, launch = checkpoint(tmp_path)
    publisher = Publisher(
        launch_checkpoint=hf_path,
        bucket_bytes=64,
        catalog=catalog,
        s3_client=s3,
        sleep=lambda _seconds: None,
    )
    publisher.initialize(PublisherConfig("model", "mx:8001", S3Config("bucket", "run")))
    snapshot, _metadata, _format, _digest = load_hf_snapshot(hf_path)
    publisher.capture_baseline(gather(launch), lambda name: snapshot[name])
    publisher.publish_version("0")
    publisher.wait_for_commit("0")
    s3.fail = True

    with pytest.raises(Exception, match="upload failed"):
        publisher.publish_version(
            "1",
            base_version="0",
            gather_hf_buckets=gather(
                {name: tensor + 1 for name, tensor in launch.items()}
            ),
        )

    assert [manifest.target_version for manifest in catalog.published] == ["0"]
