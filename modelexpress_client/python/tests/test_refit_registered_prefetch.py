# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
from unittest.mock import Mock

import pytest
from safetensors.torch import load_file
import torch

from modelexpress_rl import refit_pb2, envs
from modelexpress_rl.inference import prefetch
from modelexpress_rl.inference.client import ModelExpressGeneratorClient
from modelexpress_rl.inference.receiver import PreparationSuperseded
from tests.test_refit_s3_receiver import _build, _two_delta_objects, _two_delta_chain


class Registry:
    def __init__(self):
        self.versions = {}
        for uid, base, label in [
            ("target-a", "base-a", 1),
            ("target-b", "target-a", 2),
        ]:
            self.versions[uid] = refit_pb2.WeightVersion(
                uid=uid,
                model_name="test/model",
                base_version_id=base,
                payload_format=refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA,
                state=refit_pb2.WEIGHT_VERSION_STATE_READY,
                created_at_unix_ms=label,
                object_storage=refit_pb2.ObjectStorageSource(
                    uri=f"s3://weights/test/v{label}/model.safetensors.index.json",
                    storage_type=refit_pb2.OBJECT_STORAGE_TYPE_S3,
                ),
            )

    def ListWeightVersions(self, request, timeout):
        return refit_pb2.ListWeightVersionsResponse(
            versions=list(self.versions.values())
        )

    def GetWeightVersion(self, request, timeout):
        return refit_pb2.GetWeightVersionResponse(version=self.versions[request.uid])


@pytest.fixture
def scene(monkeypatch, tmp_path):
    adapter, storage = _build(monkeypatch, tmp_path, _two_delta_objects())
    settings = prefetch.PrefetchSettings(
        model_name="test/model",
        cache_model_name="test/model",
        server_url="mx:8001",
        refit_checkpoint_dir=str(tmp_path / "cache"),
        initial_base_version_id="base-a",
        seed_checkpoint_path=str(tmp_path / "launch"),
        refit_checkpoint_max_size_gb=1,
        endpoint_url=None,
        region_name=None,
        max_replay_chain_length=64,
        rpc_timeout_seconds=1,
        parent_pid=os.getpid(),
    )
    worker = prefetch.RefitPrefetcher(
        settings, service=Registry(), checkpoint=adapter._checkpoint
    )
    yield adapter, storage, worker
    adapter.close()


def test_prepares_only_next_version_and_reuses_exact_target(scene):
    adapter, storage, worker = scene
    assert worker.poll_once() == "target-a"
    downloads = len(storage.calls)
    assert worker.poll_once() is None
    staged = adapter.stage_chain(_two_delta_chain()[:1])
    assert len(storage.calls) == downloads
    assert torch.equal(
        load_file(staged.path / "model.safetensors")["weight"], torch.tensor([3.0, 4.0])
    )
    adapter._checkpoint.activate(staged)
    adapter.release_staged_weight(staged)
    assert worker.poll_once() == "target-b"


def test_does_not_retry_failure_from_same_head(scene, monkeypatch):
    adapter, _, worker = scene
    prepare = Mock(side_effect=ValueError("bad delta"))
    monkeypatch.setattr(adapter._checkpoint, "prepare_chain", prepare)
    assert worker.poll_once() is None
    assert worker.poll_once() is None
    assert prepare.call_count == 1


def test_rechecks_head_after_waiting_for_preparation(scene):
    adapter, _, _ = scene
    adapter._checkpoint.store.activate("other-version")
    with pytest.raises(PreparationSuperseded):
        from modelexpress_rl.inference.receiver import _ObjectStorageVersion

        adapter._checkpoint.prepare_chain(
            (
                _ObjectStorageVersion(
                    "target-a", "base-a", _two_delta_chain()[0].payload_format, "unused"
                ),
            ),
            expected_head="base-a",
        )


def test_streaming_leaves_full_checkpoint_to_engine(scene):
    _, storage, worker = scene
    worker._settings = dataclasses.replace(worker._settings, streamed=True)
    worker._service.versions = {
        "full": refit_pb2.WeightVersion(
            uid="full",
            model_name="test/model",
            payload_format=refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_HF_CHECKPOINT,
            state=refit_pb2.WEIGHT_VERSION_STATE_READY,
            created_at_unix_ms=4,
            object_storage=refit_pb2.ObjectStorageSource(
                uri="s3://weights/full/index.json",
                storage_type=refit_pb2.OBJECT_STORAGE_TYPE_S3,
            ),
        )
    }
    assert worker.poll_once() is None
    assert storage.calls == []


def test_prefetch_default_and_boolean_validation(monkeypatch):
    monkeypatch.delenv("MX_REFIT_PREFETCH", raising=False)
    assert envs.MX_REFIT_PREFETCH is False
    monkeypatch.setenv("MX_REFIT_PREFETCH", "true")
    assert envs.MX_REFIT_PREFETCH is True
    monkeypatch.setenv("MX_REFIT_PREFETCH", "invalid")
    with pytest.raises(ValueError):
        _ = envs.MX_REFIT_PREFETCH


def test_stop_reaps_killed_child():
    client = ModelExpressGeneratorClient()
    process = Mock()
    process.poll.return_value = None
    import subprocess

    process.wait.side_effect = [subprocess.TimeoutExpired("prefetch", 10), 0]
    client._prefetcher = process
    client._stop_prefetcher()
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2


def test_cache_lock_prevents_duplicate_prefetch_process(scene, monkeypatch):
    import fcntl
    import json

    _, _, worker = scene
    path = worker._settings.store().cache / prefetch.LOCK_FILE
    with path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        connect = Mock(side_effect=AssertionError("duplicate must not connect"))
        monkeypatch.setattr(prefetch.RefitPrefetcher, "connect", connect)
        assert (
            prefetch.main(
                ["--settings", json.dumps(dataclasses.asdict(worker._settings))]
            )
            == 0
        )
        connect.assert_not_called()


def test_azure_prefetch_uses_azure_reader(scene, monkeypatch):
    _, _, worker = scene
    settings = dataclasses.replace(worker._settings, storage_type="AZURE")
    reader = Mock()
    monkeypatch.setattr(prefetch, "AzureBlobReader", lambda: reader)
    checkpoint = Mock(reader=reader)
    checkpoint.store = settings.store()
    monkeypatch.setattr(prefetch, "_LocalCheckpoint", Mock(return_value=checkpoint))
    connected = prefetch.RefitPrefetcher.connect(settings)
    connected.close()
    reader.close.assert_called_once()
