# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from safetensors.torch import load_file
import torch

from modelexpress_rl import refit_pb2, envs
from modelexpress_rl.inference import prefetch
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
        max_replay_chain_length=64,
        rpc_timeout_seconds=1,
    )
    worker = prefetch.RefitPrefetcher(
        settings, service=Registry(), checkpoint=adapter._checkpoint
    )
    yield adapter, storage, worker
    worker.close()
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


def test_cache_owner_is_exclusive_and_can_take_over(scene, monkeypatch):
    _, _, worker = scene
    other = prefetch.RefitPrefetcher(
        worker._settings, service=worker._service, checkpoint=worker._checkpoint
    )
    first, second = Mock(), Mock()
    monkeypatch.setattr(worker, "poll_once", first)
    monkeypatch.setattr(other, "poll_once", second)
    worker.poll()
    other.poll()
    first.assert_called_once()
    second.assert_not_called()
    worker.close()
    other._next_poll = 0
    other.poll()
    second.assert_called_once()
    other.close()


def test_bad_background_delta_preserves_serving_head(scene, monkeypatch):
    adapter, storage, worker = scene
    assert worker.poll_once() == "target-a"
    adapter._checkpoint.store.activate("target-a")
    from tests.test_refit_s3_receiver import _artifact

    old = torch.tensor([9.0, 9.0]).view(torch.uint8).numpy()
    target = torch.tensor([5.0, 6.0]).view(torch.uint8).numpy()
    storage.objects.update(
        _artifact(
            old, target, version="target-b", version_label=2, base_version="target-a"
        )
    )
    assert worker.poll_once() is None
    state = adapter._checkpoint.store.state()
    assert state.version == "target-a"
    assert state.status.value == "READY"
    assert torch.equal(
        load_file(adapter._checkpoint.local_checkpoint / "model.safetensors")["weight"],
        torch.tensor([3.0, 4.0]),
    )
    from modelexpress_rl.inference.receiver import _ObjectStorageVersion

    same_base = _ObjectStorageVersion(
        "target-b",
        "target-a",
        _two_delta_chain()[1].payload_format,
        "s3://weights/test/v2/model.safetensors.index.json",
    )
    with pytest.raises(RuntimeError, match="checksum differs"):
        adapter._checkpoint.prepare_chain((same_base,))


def test_unresolved_chain_uses_bounded_retry(scene, monkeypatch):
    _, _, worker = scene
    clock = [10.0]
    monkeypatch.setattr(prefetch.time, "monotonic", lambda: clock[0])
    fetch = Mock(side_effect=RuntimeError("not published yet"))
    monkeypatch.setattr(worker, "_fetch_ready_version", fetch)
    assert worker._next_registered("base-a") is None
    assert worker._next_registered("base-a") is None
    assert fetch.call_count == 1
    clock[0] += prefetch.POLL_SECONDS
    assert worker._next_registered("base-a") is None
    assert fetch.call_count == 2


def test_prefer_prepared_keeps_configured_order_for_other_updates(monkeypatch):
    from modelexpress_rl.inference.session import WeightUpdateSession
    from modelexpress_rl.inference.plan import WeightSource

    resolvers = (
        SimpleNamespace(kind=WeightSource.GENERATOR),
        SimpleNamespace(kind=WeightSource.OBJECT_STORAGE),
    )
    session = WeightUpdateSession(
        resolvers=resolvers,
        methods=(),
        installer=Mock(),
        max_transfer_attempts=1,
        start_lease=Mock(),
        prefer_prepared=lambda v: v == "ready",
    )
    tried = []
    monkeypatch.setattr(
        session,
        "_stage_source",
        lambda version, resolver: tried.append(resolver.kind) or "prepared",
    )
    assert session.stage("ready") == "prepared"
    assert session.stage("other") == "prepared"
    assert tried == [WeightSource.OBJECT_STORAGE, WeightSource.GENERATOR]
    assert session.source_order == (WeightSource.GENERATOR, WeightSource.OBJECT_STORAGE)


def test_s3_watch_adopts_registered_source(scene, monkeypatch):
    adapter, storage, worker = scene
    worker._settings = dataclasses.replace(worker._settings, s3_watch=True)
    registry = worker._service
    first = registry.versions.pop("target-a")
    registry.versions.pop("target-b")
    base = refit_pb2.WeightVersion(
        uid="base-a",
        model_name="test/model",
        payload_format=refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_HF_CHECKPOINT,
        state=refit_pb2.WEIGHT_VERSION_STATE_READY,
        created_at_unix_ms=0,
        object_storage=refit_pb2.ObjectStorageSource(
            uri="s3://weights/test/v0/model.safetensors.index.json",
            storage_type=refit_pb2.OBJECT_STORAGE_TYPE_S3,
        ),
    )
    registry.versions["base-a"] = base
    import json

    storage.objects[base.object_storage.uri] = json.dumps(
        {"metadata": {"version": "base-a"}}
    ).encode()
    storage.list_children = lambda prefix, start_after=None: ["v1", "v2"]
    storage.exists = lambda uri: uri in storage.objects
    speculative = worker.poll_once()
    assert speculative.startswith("mx-speculative-")
    assert adapter._checkpoint.is_prepared(
        version_id="target-a", uri=first.object_storage.uri
    )
    registry.versions["target-a"] = first
    assert worker.poll_once() == "target-a"
    assert adapter._checkpoint.store.speculation() is None
    assert adapter._checkpoint.store.state().version == "target-a"


def test_prefetch_worker_limit_honors_existing_settings(monkeypatch):
    monkeypatch.setenv("MX_REFIT_DELTA_WORKERS", "2")
    monkeypatch.setenv("MX_REFIT_PREFETCH_WORKERS", "16")
    monkeypatch.setattr(prefetch.os, "cpu_count", lambda: 32)
    with prefetch.background_workers():
        assert envs.MX_REFIT_DELTA_WORKERS == 2
    assert os.environ["MX_REFIT_DELTA_WORKERS"] == "2"
    monkeypatch.delenv("MX_REFIT_DELTA_WORKERS")
    with pytest.raises(RuntimeError), prefetch.background_workers():
        assert envs.MX_REFIT_DELTA_WORKERS == 16
        raise RuntimeError("background failure")
    assert "MX_REFIT_DELTA_WORKERS" not in os.environ


def test_foreground_sibling_uses_unchanged_prefetch_base(scene, monkeypatch):
    adapter, storage, worker = scene
    assert worker.poll_once() == "target-a"
    from tests.test_refit_s3_receiver import _artifact, _inputs

    before = torch.tensor([1.0, 2.0]).view(torch.uint8).numpy()
    after = torch.tensor([8.0, 2.0]).view(torch.uint8).numpy()
    storage.objects.update(
        _artifact(
            before, after, version="sibling", version_label=3, base_version="base-a"
        )
    )
    prepared = adapter.stage_weight(
        _inputs(None, version="sibling", version_label=3, base_version="base-a")
    )
    assert prepared.target_version == "sibling"
    assert torch.equal(
        load_file(prepared.path / "model.safetensors")["weight"],
        torch.tensor([8.0, 2.0]),
    )
    adapter.release_staged_weight(prepared)
