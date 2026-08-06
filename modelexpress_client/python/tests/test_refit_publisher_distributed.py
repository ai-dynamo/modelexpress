# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import multiprocessing
from datetime import timedelta

import numpy as np
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
from modelexpress.refit.source.canonical import load_hf_snapshot


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

    def get_object(self, **kwargs):
        data, _checksum = self.objects[(kwargs["Bucket"], kwargs["Key"])]
        return {"Body": io.BytesIO(data), "VersionId": f"version-{self.puts}"}


def _contains_model_bytes(value):
    if isinstance(value, (bytes, bytearray, memoryview, np.ndarray, torch.Tensor)):
        return True
    if isinstance(value, dict):
        return any(
            _contains_model_bytes(key) or _contains_model_bytes(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_model_bytes(item) for item in value)
    return False


def _run_with_non_source_rank(rank, world_size, init_file, checkpoint, queue):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    catalog = Catalog() if rank == 0 else None
    s3 = S3()
    publisher = Publisher(
        launch_checkpoint=checkpoint,
        bucket_bytes=16,
        catalog=catalog,
        s3_client=s3,
        sleep=lambda _seconds: None,
    )
    publisher.initialize(PublisherConfig("model", "mx:8001", S3Config("bucket")))
    publisher.publish_version("0")
    publisher.wait_for_commit("0")

    launch, _metadata, _format, _digest = load_hf_snapshot(checkpoint)
    # Only rank 0 is a source rank, as with Miles at TP>1 or DP>1: every other rank
    # joins the collectives but is handed no tensors at all.
    local_names = sorted(launch) if rank == 0 else ()
    baseline = {
        name: torch.from_numpy(launch[name].copy()).view(torch.float32).reshape(2, 2)
        for name in local_names
    }

    def gather(weights):
        def run(consume):
            consume(list(weights.items()), None)

        return run

    publisher.capture_baseline(gather(baseline), lambda name: launch[name])
    target = {name: tensor + 1 for name, tensor in baseline.items()}
    publisher.publish_version("1", base_version="0", gather_hf_buckets=gather(target))
    publisher.wait_for_commit("1")
    queue.put((rank, s3.puts, publisher.current_version))
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed is unavailable"
)
def test_non_source_ranks_publish_without_holding_any_tensor(tmp_path):
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    save_file(
        {
            "model.a.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "model.b.weight": torch.ones((2, 2), dtype=torch.float32) * 2,
        },
        checkpoint / "model.safetensors",
    )
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    init_file = tmp_path / "gloo"
    processes = [
        context.Process(
            target=_run_with_non_source_rank,
            args=(rank, 2, str(init_file), str(checkpoint), queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(60)
        assert process.exitcode == 0

    results = {
        rank: (puts, version)
        for rank, puts, version in [queue.get() for _ in processes]
    }
    # The non-source rank uploads nothing yet still advances to the committed revision.
    assert results[1][0] == 0
    assert results[0][1] == "1"
    assert results[1][1] == "1"


def _run(rank, world_size, init_file, checkpoint, queue):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    gather_object = torch.distributed.all_gather_object

    def gather_metadata(output, value, group=None):
        if _contains_model_bytes(value):
            raise AssertionError("model bytes must not cross all_gather_object")
        return gather_object(output, value, group=group)

    torch.distributed.all_gather_object = gather_metadata
    catalog = Catalog() if rank == 0 else None
    s3 = S3()
    publisher = Publisher(
        launch_checkpoint=checkpoint,
        bucket_bytes=16,
        catalog=catalog,
        s3_client=s3,
        sleep=lambda _seconds: None,
    )
    publisher.initialize(PublisherConfig("model", "mx:8001", S3Config("bucket")))
    publisher.publish_version("0")
    publisher.wait_for_commit("0")

    launch, _metadata, _format, _digest = load_hf_snapshot(checkpoint)
    local_names = (
        ("duplicate", "model.a.weight")
        if rank == 0
        else ("duplicate", "model.b.weight")
    )
    baseline = {
        name: torch.from_numpy(launch[name].copy()).view(torch.float32).reshape(2, 2)
        for name in local_names
    }

    def gather(weights):
        def run(consume):
            consume(list(weights.items()), None)

        return run

    publisher.capture_baseline(gather(baseline), lambda name: launch[name])
    target = {name: tensor + 1 for name, tensor in baseline.items()}
    publisher.publish_version("1", base_version="0", gather_hf_buckets=gather(target))
    publisher.wait_for_commit("1")

    root = None
    if rank == 0:
        manifest = catalog.records[("model", "1")].manifest
        root = json.loads(
            s3.objects[(manifest.payload.bucket, manifest.payload.key)][0]
        )
    queue.put((rank, s3.puts, root, publisher.pop_metrics()))
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.distributed.is_available(), reason="torch.distributed is unavailable"
)
def test_two_source_ranks_upload_disjoint_s3_buckets_and_one_root(tmp_path):
    checkpoint = tmp_path / "hf"
    checkpoint.mkdir()
    save_file(
        {
            "duplicate": torch.ones((2, 2), dtype=torch.float32),
            "model.a.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
            "model.b.weight": torch.ones((2, 2), dtype=torch.float32) * 2,
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
        process.join(60)
        assert process.exitcode == 0

    results = {
        rank: (puts, root, metrics)
        for rank, puts, root, metrics in [queue.get() for _ in processes]
    }
    assert results[0][0] == 3
    assert results[1][0] == 1
    assert results[0][2]["perf/update_weights_density"] == 0.375
    assert results[0][2]["perf/update_weights_wire_bytes"] > 0
    assert (
        results[0][2]["perf/update_weights_density"]
        == results[1][2]["perf/update_weights_density"]
    )
    assert (
        results[0][2]["perf/update_weights_wire_bytes"]
        == results[1][2]["perf/update_weights_wire_bytes"]
    )
    root = results[0][1]
    assert [bucket["ordinal"] for bucket in root["buckets"]] == [0, 1, 2]
    assert [tensor["name"] for tensor in root["tensors"]] == [
        "duplicate",
        "model.a.weight",
        "model.b.weight",
    ]
    assert sum("duplicate" in bucket["tensors"] for bucket in root["buckets"]) == 1
