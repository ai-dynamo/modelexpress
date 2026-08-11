# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import inspect
import json
import threading

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
from modelexpress.refit import publisher as publisher_module
from modelexpress.refit.delta import decode_bucket
from modelexpress.refit.source.canonical import load_hf_snapshot


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


def configure_services(monkeypatch, catalog, s3):
    monkeypatch.setattr(
        publisher_module, "GrpcRevisionCatalog", lambda _endpoint: catalog
    )
    monkeypatch.setattr("boto3.client", lambda *_args, **_kwargs: s3)


def make_publisher(tmp_path, monkeypatch, catalog, s3):
    configure_services(monkeypatch, catalog, s3)
    hf_path, _weights = checkpoint(tmp_path)
    publisher = Publisher(
        launch_checkpoint=hf_path,
        bucket_bytes=64,
    )
    publisher.initialize(PublisherConfig("model", "mx:8001", S3Config("bucket", "run")))
    snapshot, _metadata, _format, _digest = load_hf_snapshot(hf_path)
    publisher.capture_baseline(gather(_weights), lambda name: snapshot[name])
    return publisher


def gather(weights):
    def run(encode_bucket):
        encode_bucket(list(weights.items()), None)

    return run


def test_initialize_constructs_catalog_and_s3_services(tmp_path, monkeypatch):
    hf_path, _weights = checkpoint(tmp_path)
    catalog = FakeCatalog()
    s3 = object()
    created = []

    monkeypatch.setattr(
        publisher_module,
        "GrpcRevisionCatalog",
        lambda endpoint: created.append(("catalog", endpoint)) or catalog,
    )
    monkeypatch.setattr(
        publisher_module,
        "S3Client",
        lambda **kwargs: created.append(("s3", kwargs)) or s3,
        raising=False,
    )

    parameters = inspect.signature(Publisher).parameters
    assert "catalog" not in parameters
    assert "s3_client" not in parameters
    assert "sleep" not in parameters
    assert "poll_interval_seconds" not in parameters

    publisher = Publisher(launch_checkpoint=hf_path)
    publisher.initialize(
        PublisherConfig(
            "model",
            "mx:8001",
            S3Config(
                "bucket",
                endpoint_url="https://s3.example",
                region_name="us-west-2",
            ),
        )
    )

    assert publisher.catalog is catalog
    assert publisher.s3 is s3
    assert created == [
        ("catalog", "mx:8001"),
        (
            "s3",
            {"endpoint_url": "https://s3.example", "region_name": "us-west-2"},
        ),
    ]


def test_publisher_uses_direct_capture_and_transport_calls():
    assert hasattr(Publisher, "_capture_deltas")
    assert not hasattr(Publisher, "_encode_delta")
    assert not hasattr(Publisher, "_agree_error")
    assert not hasattr(Publisher, "_put")
    assert not hasattr(Publisher, "_barrier")


def test_launch_zero_publishes_metadata_without_weights(tmp_path, monkeypatch):
    catalog = FakeCatalog()
    s3 = FakeS3()
    publisher = make_publisher(tmp_path, monkeypatch, catalog, s3)

    publisher.publish_version("0")

    manifest = catalog.published[-1]
    assert manifest.target_version == "0"
    assert manifest.base_version is None
    assert manifest.payload is None
    assert s3.puts == []


def test_exact_base_update_uses_miles_hf_buckets_and_uploads_delta(
    tmp_path, monkeypatch
):
    workers = []
    executor = publisher_module.ThreadPoolExecutor

    def thread_pool(*args, **kwargs):
        workers.append(kwargs["max_workers"])
        return executor(*args, **kwargs)

    monkeypatch.setenv("MX_REFIT_S3_UPLOAD_WORKERS", "1")
    monkeypatch.setattr(publisher_module, "ThreadPoolExecutor", thread_pool)
    catalog = FakeCatalog()
    s3 = FakeS3()
    configure_services(monkeypatch, catalog, s3)
    hf_path, launch = checkpoint(tmp_path)
    target = {name: tensor.clone() for name, tensor in launch.items()}
    target["model.b.weight"] += 7
    publisher = Publisher(
        launch_checkpoint=hf_path,
        bucket_bytes=64,
    )
    publisher.initialize(PublisherConfig("model", "mx:8001", S3Config("bucket", "run")))
    snapshot, _metadata, _format, _digest = load_hf_snapshot(hf_path)
    publisher.capture_baseline(gather(launch), lambda name: snapshot[name])
    publisher.publish_version("0")
    publisher.wait_for_commit("0")

    publisher.publish_version("1", base_version="0", gather_hf_buckets=gather(target))
    publisher.wait_for_commit("1")

    metrics = publisher.pop_metrics()
    assert 1 in workers
    old_bytes = launch["model.b.weight"].contiguous().view(torch.uint8).numpy()
    new_bytes = target["model.b.weight"].contiguous().view(torch.uint8).numpy()
    changed = int(torch.from_numpy(old_bytes != new_bytes).sum())
    total = sum(tensor.numel() * tensor.element_size() for tensor in target.values())
    assert metrics["perf/update_weights_density"] == changed / total
    assert metrics["perf/update_weights_wire_bytes"] > 0
    assert metrics["perf/mx_encode_delta"] >= 0
    assert metrics["perf/mx_publish_setup"] >= 0
    assert metrics["perf/mx_publish_pool"] >= 0
    assert metrics["perf/mx_publish_finalize"] >= 0
    assert metrics["perf/mx_publish_time"] == pytest.approx(
        metrics["perf/mx_publish_setup"]
        + metrics["perf/mx_publish_pool"]
        + metrics["perf/mx_publish_finalize"],
        abs=1e-4,
    )
    assert publisher.pop_metrics() == {}
    assert publisher.current_version == "1"
    manifest = catalog.published[-1]
    assert manifest.payload.key.endswith("/delta-index.json")
    index = json.loads(s3.objects[(manifest.payload.bucket, manifest.payload.key)][0])
    assert index["base_version"] == "0"
    assert index["target_version"] == "1"
    [bucket] = index["buckets"]
    encoded = s3.objects[(bucket["object"]["bucket"], bucket["object"]["key"])][0]
    snapshot, metadata, _format_digest, _base_digest = load_hf_snapshot(hf_path)
    decode_bucket(encoded, snapshot, metadata)
    for name, tensor in target.items():
        assert (
            snapshot[name].tobytes()
            == tensor.contiguous().view(torch.uint8).numpy().tobytes()
        )


def test_wrong_base_is_rejected_before_gather(tmp_path, monkeypatch):
    publisher = make_publisher(tmp_path, monkeypatch, FakeCatalog(), FakeS3())
    publisher.publish_version("0")
    publisher.wait_for_commit("0")
    called = False

    def should_not_run(_encode):
        nonlocal called
        called = True

    with pytest.raises(RuntimeError, match="base"):
        publisher.publish_version(
            "2", base_version="1", gather_hf_buckets=should_not_run
        )

    assert not called


def test_s3_failure_prevents_catalog_publication(tmp_path, monkeypatch):
    class FailingS3(FakeS3):
        fail = False

        def put_object(self, **kwargs):
            if self.fail:
                raise RuntimeError("upload failed")
            return super().put_object(**kwargs)

    catalog = FakeCatalog()
    s3 = FailingS3()
    configure_services(monkeypatch, catalog, s3)
    hf_path, launch = checkpoint(tmp_path)
    publisher = Publisher(
        launch_checkpoint=hf_path,
        bucket_bytes=64,
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
    with pytest.raises(RuntimeError, match="publisher is poisoned"):
        publisher.publish_version(
            "1", base_version="0", gather_hf_buckets=gather(launch)
        )


def test_bucket_uploads_run_in_parallel(tmp_path, monkeypatch):
    class ConcurrentS3(FakeS3):
        def __init__(self):
            super().__init__()
            self.barrier = threading.Barrier(2)
            self.threads = set()

        def put_object(self, **kwargs):
            if kwargs["Key"].endswith(".mxcd"):
                self.threads.add(threading.get_ident())
                self.barrier.wait(timeout=5)
            return super().put_object(**kwargs)

    catalog = FakeCatalog()
    s3 = ConcurrentS3()
    configure_services(monkeypatch, catalog, s3)
    hf_path, launch = checkpoint(tmp_path)
    publisher = Publisher(
        launch_checkpoint=hf_path,
        bucket_bytes=8,
    )
    publisher.initialize(PublisherConfig("model", "mx:8001", S3Config("bucket", "run")))
    snapshot, _metadata, _format, _digest = load_hf_snapshot(hf_path)
    publisher.capture_baseline(gather(launch), lambda name: snapshot[name])
    publisher.publish_version("0")
    publisher.wait_for_commit("0")

    publisher.publish_version(
        "1",
        base_version="0",
        gather_hf_buckets=gather({name: tensor + 1 for name, tensor in launch.items()}),
    )

    assert len(s3.threads) > 1
