# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import multiprocessing
from datetime import timedelta

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


class Catalog:
    def __init__(self):
        self.records = {}
        self.published = []

    def publish_revision(self, manifest):
        self.published.append(manifest.target_version)
        record = RevisionRecord(manifest, RevisionState.READY)
        self.records[(manifest.model_id, manifest.target_version)] = record
        return record

    def get_revision(self, model_id, version):
        record = self.records[(model_id, version)]
        record = RevisionRecord(record.manifest, RevisionState.COMMITTED)
        self.records[(model_id, version)] = record
        return record

    def close(self):
        pass


class S3:
    def __init__(self):
        self.objects = {}
        self.puts = 0

    def put_object(self, **kwargs):
        self.puts += 1
        key = (kwargs["Bucket"], kwargs["Key"])
        self.objects[key] = (bytes(kwargs["Body"]), kwargs["ChecksumCRC32C"])
        return {"VersionId": f"version-{self.puts}"}

    def head_object(self, **kwargs):
        data, checksum = self.objects[(kwargs["Bucket"], kwargs["Key"])]
        return {
            "ContentLength": len(data),
            "ChecksumCRC32C": checksum,
            "VersionId": f"version-{self.puts}",
        }

    def get_object(self, **kwargs):
        data, _checksum = self.objects[(kwargs["Bucket"], kwargs["Key"])]
        return {"Body": io.BytesIO(data), "VersionId": f"version-{self.puts}"}


def _run(rank, world_size, init_file, checkpoint, queue):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    torch.distributed.broadcast_object_list = lambda *_args, **_kwargs: (
        _ for _ in ()
    ).throw(AssertionError("publisher must not broadcast objects"))
    catalog = Catalog() if rank == 0 else None
    s3 = S3() if rank == 0 else None
    publisher = Publisher(
        launch_checkpoint=checkpoint,
        bucket_bytes=64,
        catalog=catalog,
        s3_client=s3,
        sleep=lambda _seconds: None,
    )
    publisher.initialize(PublisherConfig("model", "mx:8001", S3Config("bucket")))
    publisher.publish_version("0")

    target = {
        "model.a.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2) + 1,
        "model.b.weight": torch.ones((2, 2), dtype=torch.float32) + 1,
    }

    def gather(encode_bucket):
        encode_bucket(list(target.items()), None)

    publisher.publish_version("1", base_version="0", gather_hf_buckets=gather)
    queue.put(
        (
            rank,
            publisher.status().current_version,
            catalog.published if catalog is not None else [],
            s3.puts if s3 is not None else 0,
        )
    )
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed is unavailable"
)
def test_two_rank_publisher_uses_miles_buckets_without_object_broadcast(tmp_path):
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    save_file(
        {
            "model.a.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "model.b.weight": torch.ones((2, 2), dtype=torch.float32),
        },
        checkpoint / "model.safetensors",
    )
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_file = tmp_path / "gloo"
    processes = [
        context.Process(
            target=_run,
            args=(rank, 2, str(init_file), str(checkpoint), queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0

    results = sorted(queue.get(timeout=5) for _ in processes)
    assert results[0][1:] == ("1", ["0", "1"], 2)
    assert results[1][1:] == ("1", [], 0)
