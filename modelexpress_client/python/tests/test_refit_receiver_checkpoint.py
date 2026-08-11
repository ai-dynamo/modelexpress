# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import fcntl
import hashlib
import json
import threading
from types import SimpleNamespace

import google_crc32c
import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from modelexpress.refit import receiver as receiver_module
from modelexpress.refit.manifest import (
    RevisionManifest,
    RevisionRecord,
    RevisionState,
    S3Object,
)
from modelexpress.refit.receiver import ModelExpressWeightReceiver, ReceiverConfig
from modelexpress.refit.source.canonical import (
    encode_bucket,
    load_hf_snapshot,
    snapshot_digest,
)


def test_receiver_download_workers_follow_receiver_configuration(monkeypatch):
    monkeypatch.setenv("MX_REFIT_S3_DOWNLOAD_WORKERS", "7")

    assert receiver_module._download_worker_count(20) == 7
    assert receiver_module._download_worker_count(3) == 3


def test_receiver_has_no_callback_or_forwarding_helpers():
    import inspect

    parameters = inspect.signature(ModelExpressWeightReceiver).parameters
    builder_parameters = inspect.signature(
        receiver_module.build_weight_receiver
    ).parameters

    assert "model_runner" in parameters
    assert "launch_checkpoint" not in parameters
    assert "catalog" not in parameters
    assert "s3_client" not in parameters
    assert "prepare_target" not in parameters
    assert "install_target" not in parameters
    assert "catalog" not in builder_parameters
    assert "s3_client" not in builder_parameters
    assert "checkpoint" not in builder_parameters
    assert not hasattr(ModelExpressWeightReceiver, "_prepare_revision")
    assert not hasattr(ModelExpressWeightReceiver, "_installation")
    assert not hasattr(receiver_module, "_location")
    assert not hasattr(receiver_module, "load_prepared_checkpoint")


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

    def get(self, location):
        self.calls.append(location.key)
        data = self.objects[location.key]
        assert location.checksum == f"crc32c:{google_crc32c.value(data):08x}"
        return data


class Receiver(ModelExpressWeightReceiver):
    install_started: threading.Event | None = None
    install_release: threading.Event | None = None

    def install_prepared_checkpoint(self, _prepared):
        if self.install_started is not None:
            self.install_started.set()
            release = self.install_release
            assert release is not None
            assert release.wait(5)


def location(key, data, include_size=False):
    value = {
        "bucket": "bucket",
        "key": key,
        "checksum": f"crc32c:{google_crc32c.value(data):08x}",
    }
    if include_size:
        value["size"] = len(data)
    return value


def test_receiver_downloads_and_patches_one_persistent_exact_checkpoint(
    tmp_path, monkeypatch
):
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
    bucket_location = location("bucket-0.mxdb", bucket, include_size=True)
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
    monkeypatch.setattr(
        receiver_module,
        "GrpcRevisionCatalog",
        lambda _endpoint: catalog,
        raising=False,
    )
    monkeypatch.setattr(
        receiver_module,
        "S3Client",
        lambda **_kwargs: s3,
        raising=False,
    )

    def build():
        runner = SimpleNamespace(
            loader=SimpleNamespace(
                _prepare_weights=lambda *_args: (checkpoint, None, None)
            ),
            model_config=SimpleNamespace(model_path=str(checkpoint), revision=None),
        )
        value = Receiver(config, "host:0", runner)
        value.initialize()
        return value

    receiver = build()
    assert receiver.catalog is catalog
    assert receiver.s3 is s3
    follower = build()
    local_checkpoint = tmp_path / "cache" / "model" / "checkpoint"
    local_file = local_checkpoint / "model.safetensors"
    inode = local_file.stat().st_ino

    receiver.start_weight_update("1")

    metrics = receiver.pop_metrics()
    assert metrics["perf/mx_receive_prepare_time"] >= 0
    assert metrics["perf/mx_receive_root_download"] >= 0
    assert metrics["perf/mx_receive_pool"] >= 0
    assert "perf/mx_receive_wire_bytes" not in metrics

    assert receiver.status().installed_version == "0"
    assert receiver.prepared.target_digest == target_digest
    assert receiver.prepared.path == local_checkpoint
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
    follower_metrics = follower.pop_metrics()
    assert follower_metrics["perf/mx_receive_prepare_time"] >= 0
    assert "perf/mx_receive_wire_bytes" not in follower_metrics

    install_started = threading.Event()
    install_release = threading.Event()
    follower.install_started = install_started
    follower.install_release = install_release
    install_result = []
    install_thread = threading.Thread(
        target=lambda: install_result.append(follower.update_weights())
    )
    install_thread.start()
    assert install_started.wait(5)
    checkpoint_state = follower.checkpoint
    assert checkpoint_state is not None
    try:
        with checkpoint_state.lock_path.open("a+") as handle:
            with pytest.raises(BlockingIOError):
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        install_release.set()
        install_thread.join(5)
    assert not install_thread.is_alive()
    assert install_result[0].success

    restarted = build()
    assert restarted.status().installed_version == "0"
    assert torch.equal(load_file(local_file)["model.weight"], torch.tensor([1.0, 2.0]))

    result = receiver.update_weights()
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
