# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json

import google_crc32c
import numpy as np
import torch
from safetensors.torch import load_file, save_file

from modelexpress.refit.manifest import (
    RevisionManifest,
    RevisionRecord,
    RevisionState,
    S3Object,
)
from modelexpress.refit.receiver import ReceiverConfig, build_weight_receiver
from modelexpress.refit.source.canonical import (
    encode_bucket,
    load_hf_snapshot,
    snapshot_digest,
)


class Catalog:
    def __init__(self, records, misses=0):
        self.records = records
        self.misses = misses

    def get_revision(self, _model_id, version):
        if self.misses:
            self.misses -= 1
            raise KeyError(version)
        return self.records[version]


class S3:
    def __init__(self, objects):
        self.objects = objects
        self.calls = []

    def get_object(self, **request):
        self.calls.append(request["Key"])
        return {"Body": io.BytesIO(self.objects[request["Key"]])}


def location(key, data):
    return {
        "bucket": "bucket",
        "key": key,
        "checksum": f"crc32c:{google_crc32c.value(data):08x}",
    }


def test_receiver_downloads_and_patches_one_persistent_exact_checkpoint(tmp_path):
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    save_file(
        {"model.weight": torch.tensor([1.0, 2.0])}, checkpoint / "model.safetensors"
    )
    snapshot, metadata, format_digest, base_digest = load_hf_snapshot(checkpoint)
    target = torch.tensor([3.0, 4.0])
    name = "model.weight"
    old = snapshot[name].tobytes()
    new = target.contiguous().view(torch.uint8).numpy().tobytes()
    delta = np.frombuffer(
        bytes(left ^ right for left, right in zip(old, new, strict=True)),
        dtype=np.uint8,
    )
    metadata[name]["target_digest"] = f"sha256:{hashlib.sha256(new).hexdigest()}"
    ordinal = 0
    bucket, decoded_size = encode_bucket(
        model_id="model",
        base_version="0",
        target_version="1",
        base_digest=base_digest,
        format_digest=format_digest,
        ordinal=ordinal,
        tensors=[(name, delta)],
        metadata=metadata,
    )
    target_digest = snapshot_digest(metadata)
    coverage = [{**metadata[name], "state": "dirty", "bucket_ordinal": ordinal}]
    bucket_location = location("bucket-0.mxdb", bucket)
    root = json.dumps(
        {
            "model_id": "model",
            "base_version": "0",
            "target_version": "1",
            "base_digest": base_digest,
            "target_digest": target_digest,
            "format_digest": format_digest,
            "buckets": [
                {
                    "ordinal": ordinal,
                    "decoded_size": decoded_size,
                    "tensors": [name],
                    "object": bucket_location,
                }
            ],
            "tensors": coverage,
        }
    ).encode()
    launch = RevisionRecord(
        RevisionManifest(
            model_id="model",
            target_version="0",
            target_digest=base_digest,
            format_digest=format_digest,
        ),
        RevisionState.READY,
    )
    target_record = RevisionRecord(
        RevisionManifest(
            model_id="model",
            target_version="1",
            base_version="0",
            base_digest=base_digest,
            target_digest=target_digest,
            format_digest=format_digest,
            payload=S3Object(**location("root.json", root)),
        ),
        RevisionState.READY,
    )
    config = ReceiverConfig(
        model_id="model",
        catalog_endpoint="mx:8001",
        initial_version="0",
        preparation_cache_dir=tmp_path / "cache",
        ready_timeout_seconds=1,
    )
    catalog = Catalog({"0": launch, "1": target_record}, misses=1)
    s3 = S3({"root.json": root, "bucket-0.mxdb": bucket})

    def build():
        return build_weight_receiver(
            config=config,
            receiver_id="host:0",
            launch_checkpoint=checkpoint,
            install_target=lambda _target: None,
            catalog=catalog,
            s3_client=s3,
        )

    receiver = build()
    follower = build()
    local_checkpoint = tmp_path / "cache" / "model" / "checkpoint"
    local_file = local_checkpoint / "model.safetensors"
    inode = local_file.stat().st_ino

    receiver.start_weight_update("1")

    assert receiver.status().installed_version == "0"
    assert receiver.prepared["digest"] == target_digest
    assert receiver.prepared["path"] == local_checkpoint
    assert local_file.stat().st_ino == inode
    assert torch.equal(load_file(local_file)["model.weight"], target)
    assert not (tmp_path / "cache" / "model" / "1").exists()
    assert torch.equal(
        load_file(checkpoint / "model.safetensors")["model.weight"],
        torch.tensor([1.0, 2.0]),
    )
    assert s3.calls == ["root.json", "bucket-0.mxdb"]

    follower.start_weight_update("1")
    assert s3.calls == ["root.json", "bucket-0.mxdb"]

    restarted = build()
    assert restarted.status().installed_version == "0"
    assert torch.equal(load_file(local_file)["model.weight"], torch.tensor([1.0, 2.0]))

    result = follower.update_weights()
    assert not result.success
    assert "changed before installation" in result.detail

    with local_file.open("r+b") as handle:
        handle.seek(-1, 2)
        value = handle.read(1)[0]
        handle.seek(-1, 2)
        handle.write(bytes([value ^ 1]))
    recovered = build()
    assert recovered.status().installed_version == "0"
    assert torch.equal(load_file(local_file)["model.weight"], torch.tensor([1.0, 2.0]))
