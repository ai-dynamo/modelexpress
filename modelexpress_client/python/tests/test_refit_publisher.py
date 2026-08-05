# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import struct

import pytest
import torch
import zstandard
from safetensors.torch import save_file

from modelexpress.refit import (
    PublicationMode,
    Publisher,
    PublisherConfig,
    RevisionRecord,
    RevisionState,
    S3Config,
)
from modelexpress.refit.publisher import PublisherError
from modelexpress.refit.source.megatron_bridge import MegatronBridgeHfBucketConfig


class FakeCatalog:
    def __init__(self):
        self.records = {}
        self.published = []
        self.commit_calls = 0
        self.get_calls = []
        self.commit_after_gets = None

    def publish_revision(self, manifest):
        self.published.append(manifest)
        key = (manifest.model_id, manifest.target_version)
        current = self.records.get(key)
        if current is not None and current.manifest != manifest:
            raise RuntimeError("manifest conflict")
        if current is None:
            current = RevisionRecord(manifest, RevisionState.READY)
            self.records[key] = current
        return current

    def get_revision(self, model_id, target_version):
        self.get_calls.append((model_id, target_version))
        key = (model_id, target_version)
        record = self.records[key]
        if (
            self.commit_after_gets is not None
            and len(self.get_calls) >= self.commit_after_gets
        ):
            record = RevisionRecord(record.manifest, RevisionState.COMMITTED)
            self.records[key] = record
        return record

    def commit_revision(self, model_id, target_version):
        self.commit_calls += 1
        key = (model_id, target_version)
        record = self.records[key]
        committed = RevisionRecord(record.manifest, RevisionState.COMMITTED)
        self.records[key] = committed
        return committed


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


class Bridge:
    def __init__(self, weights):
        self.weights = weights
        self.capture_calls = 0

    def get_conversion_tasks(self, _model):
        return [object()]

    def export_hf_weights(self, _model, **_kwargs):
        self.capture_calls += 1
        for name in reversed(sorted(self.weights)):
            yield name, self.weights[name]


def checkpoint(tmp_path):
    path = tmp_path / "hf"
    path.mkdir()
    tensors = {
        "model.a.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "model.b.weight": torch.ones((2, 2), dtype=torch.float32),
    }
    save_file(tensors, path / "model.safetensors")
    return path, tensors


def make_publisher(tmp_path, catalog, s3, bridge, mode=PublicationMode.ASYNC):
    hf_path, _weights = checkpoint(tmp_path)
    publisher = Publisher(
        model=object(),
        launch_checkpoint=hf_path,
        scratch_directory=tmp_path / "scratch",
        megatron_config=MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=64,
            spool_directory=tmp_path / "spool",
        ),
        catalog=catalog,
        s3_client=s3,
        sleep=lambda _seconds: None,
    )
    publisher.initialize(
        PublisherConfig(
            model_id="model",
            catalog_endpoint="mx:8001",
            s3=S3Config(bucket="bucket", prefix="run"),
            publication_mode=mode,
        )
    )
    return publisher


def test_launch_zero_is_attested_and_published_ready_without_any_s3_upload(tmp_path):
    catalog = FakeCatalog()
    s3 = FakeS3()
    bridge = Bridge({})
    publisher = make_publisher(tmp_path, catalog, s3, bridge)

    result = publisher.publish_version("0")

    assert result.state is RevisionState.READY
    manifest = catalog.published[-1]
    assert manifest.target_version == "0"
    assert manifest.base_version is None
    assert manifest.base_digest is None
    assert manifest.payload is None
    assert manifest.target_digest.startswith("sha256:")
    assert manifest.format_digest.startswith("sha256:")
    assert s3.puts == []
    assert bridge.capture_calls == 0
    assert catalog.commit_calls == 0
    assert list((tmp_path / "scratch").iterdir()) == []


def test_exact_base_update_uploads_buckets_and_one_root_then_publishes_root_only(
    tmp_path,
):
    catalog = FakeCatalog()
    s3 = FakeS3()
    hf_path, launch = checkpoint(tmp_path)
    updated = {name: tensor.clone() for name, tensor in launch.items()}
    updated["model.b.weight"] += 2
    bridge = Bridge(updated)
    publisher = Publisher(
        model=object(),
        launch_checkpoint=hf_path,
        scratch_directory=tmp_path / "scratch",
        megatron_config=MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=64,
            spool_directory=tmp_path / "spool",
        ),
        catalog=catalog,
        s3_client=s3,
        sleep=lambda _seconds: None,
    )
    publisher.initialize(
        PublisherConfig(
            "model", "mx:8001", S3Config("bucket", "run"), PublicationMode.ASYNC
        )
    )
    publisher.publish_version("0")
    catalog.commit_revision("model", "0")

    result = publisher.publish_version("1", base_version="0")

    assert result.state is RevisionState.READY
    manifest = catalog.published[-1]
    assert manifest.base_version == "0"
    assert (
        manifest.base_digest == catalog.records[("model", "0")].manifest.target_digest
    )
    assert manifest.payload is not None
    assert manifest.payload.bucket == "bucket"
    assert manifest.payload.key.endswith("/root.json")
    assert catalog.commit_calls == 1  # the explicit test-orchestrator call only
    assert len(s3.puts) == 2  # one dirty bucket plus exactly one root
    root_bytes = s3.objects[(manifest.payload.bucket, manifest.payload.key)][0]
    root = json.loads(root_bytes)
    assert root["schema"] == "mx.canonical.delta.v0"
    assert root["encoding"] == {"compression": "zstd", "delta": "xor"}
    assert len(root["buckets"]) == 1
    assert root["base_digest"] == manifest.base_digest
    assert root["format_digest"] == manifest.format_digest
    assert root["target_digest"] == manifest.target_digest
    assert "buckets" not in vars(manifest)
    assert "encoding" not in vars(manifest)


def test_bucket_payload_decodes_against_exact_base_to_canonical_target(tmp_path):
    catalog = FakeCatalog()
    s3 = FakeS3()
    hf_path, launch = checkpoint(tmp_path)
    target = {name: tensor.clone() for name, tensor in launch.items()}
    target["model.b.weight"] += 7
    publisher = Publisher(
        model=object(),
        launch_checkpoint=hf_path,
        scratch_directory=tmp_path / "scratch",
        megatron_config=MegatronBridgeHfBucketConfig(
            bridge=Bridge(target),
            bucket_bytes=64,
            spool_directory=tmp_path / "spool",
        ),
        catalog=catalog,
        s3_client=s3,
    )
    publisher.initialize(
        PublisherConfig(
            "model", "mx:8001", S3Config("bucket", "run"), PublicationMode.ASYNC
        )
    )
    publisher.publish_version("0")
    catalog.commit_revision("model", "0")

    publisher.publish_version("1", base_version="0")

    manifest = catalog.published[-1]
    root = json.loads(s3.objects[(manifest.payload.bucket, manifest.payload.key)][0])
    [bucket] = root["buckets"]
    location = bucket["object"]
    encoded = s3.objects[(location["bucket"], location["key"])][0]
    assert encoded.startswith(b"MXCDV0\0")
    header_size = struct.unpack(">I", encoded[7:11])[0]
    header = json.loads(encoded[11 : 11 + header_size])
    decoded = zstandard.ZstdDecompressor().decompress(
        encoded[11 + header_size :],
        max_output_size=header["decoded_size"],
    )
    [entry] = header["entries"]
    delta = decoded[entry["offset"] : entry["offset"] + entry["byte_size"]]
    base = launch[entry["name"]].contiguous().view(torch.uint8).numpy().tobytes()
    recovered = bytes(left ^ right for left, right in zip(base, delta, strict=True))
    expected = target[entry["name"]].contiguous().view(torch.uint8).numpy().tobytes()
    assert recovered == expected
    assert root["tensors"] == [
        {
            "byte_size": tensor.numel() * tensor.element_size(),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "name": name,
            "shape": list(tensor.shape),
            "state": "clean" if torch.equal(tensor, launch[name]) else "dirty",
            "target_digest": next(
                item["target_digest"]
                for item in root["tensors"]
                if item["name"] == name
            ),
            **({"bucket_ordinal": 0} if not torch.equal(tensor, launch[name]) else {}),
        }
        for name, tensor in sorted(target.items())
    ]


def test_s3_verification_failure_prevents_target_catalog_publication(tmp_path):
    class FailingReadbackS3(FakeS3):
        fail = False

        def get_object(self, **kwargs):
            if self.fail:
                raise RuntimeError("readback unavailable")
            return super().get_object(**kwargs)

    catalog = FakeCatalog()
    s3 = FailingReadbackS3()
    hf_path, launch = checkpoint(tmp_path)
    publisher = Publisher(
        model=object(),
        launch_checkpoint=hf_path,
        scratch_directory=tmp_path / "scratch",
        megatron_config=MegatronBridgeHfBucketConfig(
            bridge=Bridge({name: tensor + 1 for name, tensor in launch.items()}),
            bucket_bytes=64,
            spool_directory=tmp_path / "spool",
        ),
        catalog=catalog,
        s3_client=s3,
    )
    publisher.initialize(
        PublisherConfig(
            "model", "mx:8001", S3Config("bucket", "run"), PublicationMode.ASYNC
        )
    )
    publisher.publish_version("0")
    catalog.commit_revision("model", "0")
    s3.fail = True

    with pytest.raises(PublisherError, match="unreadable"):
        publisher.publish_version("1", base_version="0")

    assert [manifest.target_version for manifest in catalog.published] == ["0"]


def test_async_requires_committed_launch_before_target_capture(tmp_path):
    catalog = FakeCatalog()
    s3 = FakeS3()
    hf_path, launch = checkpoint(tmp_path)
    bridge = Bridge({name: tensor + 1 for name, tensor in launch.items()})
    publisher = Publisher(
        model=object(),
        launch_checkpoint=hf_path,
        scratch_directory=tmp_path / "scratch",
        megatron_config=MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=64,
            spool_directory=tmp_path / "spool",
        ),
        catalog=catalog,
        s3_client=s3,
    )
    publisher.initialize(
        PublisherConfig(
            "model", "mx:8001", S3Config("bucket", "run"), PublicationMode.ASYNC
        )
    )
    publisher.publish_version("0")

    with pytest.raises(PublisherError, match="not committed"):
        publisher.publish_version("1", base_version="0")

    assert bridge.capture_calls == 0
    assert s3.puts == []
    assert [manifest.target_version for manifest in catalog.published] == ["0"]


def test_async_target_advances_retained_base_only_after_external_commit(tmp_path):
    catalog = FakeCatalog()
    s3 = FakeS3()
    hf_path, launch = checkpoint(tmp_path)
    bridge = Bridge({name: tensor + 1 for name, tensor in launch.items()})
    publisher = Publisher(
        model=object(),
        launch_checkpoint=hf_path,
        scratch_directory=tmp_path / "scratch",
        megatron_config=MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=64,
            spool_directory=tmp_path / "spool",
        ),
        catalog=catalog,
        s3_client=s3,
    )
    publisher.initialize(
        PublisherConfig(
            "model", "mx:8001", S3Config("bucket", "run"), PublicationMode.ASYNC
        )
    )
    publisher.publish_version("0")
    catalog.commit_revision("model", "0")

    first = publisher.publish_version("1", base_version="0")

    assert first.state is RevisionState.READY
    assert publisher.status().current_version == "0"
    assert publisher._base_store.current.version == "0"
    capture_calls = bridge.capture_calls
    with pytest.raises(PublisherError, match="pending target is not committed"):
        publisher.publish_version("2", base_version="1")
    assert bridge.capture_calls == capture_calls

    catalog.commit_revision("model", "1")
    bridge.weights = {name: tensor + 2 for name, tensor in launch.items()}
    second = publisher.publish_version("2", base_version="1")

    assert second.state is RevisionState.READY
    assert publisher.status().current_version == "1"
    assert publisher._base_store.current.version == "1"


def test_publisher_requires_the_one_exact_retained_base_before_capture(tmp_path):
    catalog = FakeCatalog()
    bridge = Bridge({})
    publisher = make_publisher(tmp_path, catalog, FakeS3(), bridge)
    publisher.publish_version("0")
    catalog.commit_revision("model", "0")

    with pytest.raises(PublisherError, match="exact retained base '0'"):
        publisher.publish_version("1", base_version="stale")

    assert bridge.capture_calls == 0
    assert [manifest.target_version for manifest in catalog.published] == ["0"]


def test_clean_update_still_uploads_and_publishes_one_self_describing_root(tmp_path):
    catalog = FakeCatalog()
    s3 = FakeS3()
    hf_path, launch = checkpoint(tmp_path)
    publisher = Publisher(
        model=object(),
        launch_checkpoint=hf_path,
        scratch_directory=tmp_path / "scratch",
        megatron_config=MegatronBridgeHfBucketConfig(
            bridge=Bridge(launch),
            bucket_bytes=64,
            spool_directory=tmp_path / "spool",
        ),
        catalog=catalog,
        s3_client=s3,
        sleep=lambda _seconds: None,
    )
    publisher.initialize(
        PublisherConfig(
            "model",
            "mx:8001",
            S3Config("bucket", "run"),
            PublicationMode.ASYNC,
        )
    )
    publisher.publish_version("0")
    catalog.commit_revision("model", "0")

    result = publisher.publish_version("1", base_version="0")

    manifest = catalog.published[-1]
    assert result.state is RevisionState.READY
    assert manifest.payload is not None
    assert len(s3.puts) == 1
    root = json.loads(s3.objects[(manifest.payload.bucket, manifest.payload.key)][0])
    assert root["buckets"] == []
    assert [tensor["state"] for tensor in root["tensors"]] == ["clean", "clean"]


def test_successive_updates_retain_only_the_new_exact_base(tmp_path):
    catalog = FakeCatalog()
    s3 = FakeS3()
    hf_path, launch = checkpoint(tmp_path)
    bridge = Bridge({name: tensor + 1 for name, tensor in launch.items()})
    publisher = Publisher(
        model=object(),
        launch_checkpoint=hf_path,
        scratch_directory=tmp_path / "scratch",
        megatron_config=MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=64,
            spool_directory=tmp_path / "spool",
        ),
        catalog=catalog,
        s3_client=s3,
        sleep=lambda _seconds: None,
    )
    publisher.initialize(
        PublisherConfig(
            "model",
            "mx:8001",
            S3Config("bucket", "run"),
            PublicationMode.ASYNC,
        )
    )
    publisher.publish_version("0")
    catalog.commit_revision("model", "0")
    publisher.publish_version("1", base_version="0")
    catalog.commit_revision("model", "1")
    bridge.weights = {name: tensor + 2 for name, tensor in launch.items()}

    publisher.publish_version("2", base_version="1")

    retained = list((tmp_path / "scratch").iterdir())
    assert len(retained) == 2  # committed base 1 plus uncommitted candidate 2
    catalog.commit_revision("model", "2")
    publisher.publish_version("2", base_version="1")
    retained = list((tmp_path / "scratch").iterdir())
    assert len(retained) == 1
    assert retained[0].is_dir()
    assert publisher._base_store.current.version == "2"
    capture_calls = bridge.capture_calls
    with pytest.raises(PublisherError, match="exact retained base '2'"):
        publisher.publish_version("3", base_version="0")
    assert bridge.capture_calls == capture_calls


def test_block_polls_exact_get_until_committed_and_never_commits(tmp_path):
    catalog = FakeCatalog()
    catalog.commit_after_gets = 2
    publisher = make_publisher(
        tmp_path,
        catalog,
        FakeS3(),
        Bridge({}),
        mode=PublicationMode.BLOCK,
    )

    result = publisher.publish_version("0")

    assert result.state is RevisionState.COMMITTED
    assert catalog.get_calls == [("model", "0"), ("model", "0")]
    assert catalog.commit_calls == 0


def test_block_rejects_a_changed_manifest_from_exact_get(tmp_path):
    class ChangedManifestCatalog(FakeCatalog):
        def get_revision(self, model_id, target_version):
            record = super().get_revision(model_id, target_version)
            changed = type(record.manifest)(
                model_id=record.manifest.model_id,
                target_version=record.manifest.target_version,
                target_digest="sha256:changed",
                format_digest=record.manifest.format_digest,
            )
            return RevisionRecord(changed, RevisionState.COMMITTED)

    publisher = make_publisher(
        tmp_path,
        ChangedManifestCatalog(),
        FakeS3(),
        Bridge({}),
        mode=PublicationMode.BLOCK,
    )

    with pytest.raises(PublisherError, match="different immutable manifest"):
        publisher.publish_version("0")


def test_block_promotes_target_only_after_commit(tmp_path):
    catalog = FakeCatalog()
    s3 = FakeS3()
    hf_path, launch = checkpoint(tmp_path)
    bridge = Bridge({name: tensor + 1 for name, tensor in launch.items()})
    observations = []
    publisher = Publisher(
        model=object(),
        launch_checkpoint=hf_path,
        scratch_directory=tmp_path / "scratch",
        megatron_config=MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=64,
            spool_directory=tmp_path / "spool",
        ),
        catalog=catalog,
        s3_client=s3,
        sleep=lambda _seconds: observations.append(
            (
                publisher.status().current_version,
                publisher._base_store.current.version,
                len(list((tmp_path / "scratch").iterdir())),
            )
        ),
    )
    publisher.initialize(
        PublisherConfig(
            "model",
            "mx:8001",
            S3Config("bucket", "run"),
            PublicationMode.BLOCK,
        )
    )
    catalog.commit_after_gets = 1
    publisher.publish_version("0")
    catalog.get_calls.clear()
    catalog.commit_after_gets = 3

    result = publisher.publish_version("1", base_version="0")

    assert observations == [("0", "0", 1)]
    assert result.state is RevisionState.COMMITTED
    assert publisher.status().current_version == "1"
    assert publisher._base_store.current.version == "1"
    assert len(list((tmp_path / "scratch").iterdir())) == 1


def test_block_poll_failure_keeps_base_and_resumes_without_republishing(tmp_path):
    class TransientGetCatalog(FakeCatalog):
        fail_next_get = False

        def get_revision(self, model_id, target_version):
            if self.fail_next_get and target_version == "1":
                self.fail_next_get = False
                raise RuntimeError("temporary exact-get failure")
            return super().get_revision(model_id, target_version)

    catalog = TransientGetCatalog()
    s3 = FakeS3()
    hf_path, launch = checkpoint(tmp_path)
    bridge = Bridge({name: tensor + 1 for name, tensor in launch.items()})
    publisher = Publisher(
        model=object(),
        launch_checkpoint=hf_path,
        scratch_directory=tmp_path / "scratch",
        megatron_config=MegatronBridgeHfBucketConfig(
            bridge=bridge,
            bucket_bytes=64,
            spool_directory=tmp_path / "spool",
        ),
        catalog=catalog,
        s3_client=s3,
        sleep=lambda _seconds: None,
    )
    publisher.initialize(
        PublisherConfig(
            "model",
            "mx:8001",
            S3Config("bucket", "run"),
            PublicationMode.BLOCK,
        )
    )
    catalog.commit_after_gets = 1
    publisher.publish_version("0")
    catalog.get_calls.clear()
    catalog.commit_after_gets = None
    catalog.fail_next_get = True

    with pytest.raises(PublisherError, match="temporary exact-get failure"):
        publisher.publish_version("1", base_version="0")

    assert publisher.status().current_version == "0"
    assert publisher._base_store.current.version == "0"
    published = len(catalog.published)
    capture_calls = bridge.capture_calls
    s3_puts = len(s3.puts)
    catalog.records[("model", "1")] = RevisionRecord(
        catalog.records[("model", "1")].manifest,
        RevisionState.COMMITTED,
    )

    result = publisher.publish_version("1", base_version="0")

    assert result.state is RevisionState.COMMITTED
    assert publisher._base_store.current.version == "1"
    assert len(catalog.published) == published
    assert bridge.capture_calls == capture_calls
    assert len(s3.puts) == s3_puts
