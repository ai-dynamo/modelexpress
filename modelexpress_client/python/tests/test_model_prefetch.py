# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pre-engine metadata prefetch."""

import logging
import threading
import time
from pathlib import Path

import pytest

from modelexpress import model_prefetch

REPO = "org/model"
COMMIT = "e" * 40


class FakeClient:
    """Stands in for ModelCacheClient, recording what the prefetch asked for."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.revisions = []
        self.snapshot = None
        self.error = None
        FakeClient.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def install_metadata_snapshot(self, repo_id, *args, **kwargs):
        self.calls.append(repo_id)
        self.revisions.append(kwargs.get("requested_revision"))
        if self.error is not None:
            raise self.error
        return self.snapshot


@pytest.fixture(autouse=True)
def clean_state(monkeypatch, tmp_path):
    model_prefetch.reset()
    FakeClient.instances = []
    for name in (
        "MODEL_EXPRESS_NO_SHARED_STORAGE",
        "MODEL_EXPRESS_URL",
        "MX_SERVER_ADDRESS",
        "MODEL_EXPRESS_TRANSFER_CHUNK_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    model_prefetch.reset()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv("MODEL_EXPRESS_NO_SHARED_STORAGE", "1")
    monkeypatch.setenv("MODEL_EXPRESS_URL", "http://mx:8001")


@pytest.fixture
def fake_client(monkeypatch, tmp_path):
    snapshot = tmp_path / "models--org--model" / "snapshots" / COMMIT
    snapshot.mkdir(parents=True)

    def factory(**kwargs):
        client = FakeClient(**kwargs)
        client.snapshot = snapshot
        return client

    monkeypatch.setattr("modelexpress.model_client.ModelCacheClient", factory)
    return snapshot


class TestIsEnabled:
    def test_off_by_default(self):
        assert model_prefetch.is_enabled() is False

    def test_needs_a_server_address(self, monkeypatch):
        monkeypatch.setenv("MODEL_EXPRESS_NO_SHARED_STORAGE", "1")
        assert model_prefetch.is_enabled() is False

    def test_needs_the_switch(self, monkeypatch):
        monkeypatch.setenv("MODEL_EXPRESS_URL", "http://mx:8001")
        assert model_prefetch.is_enabled() is False

    def test_enabled_with_both(self, enabled):
        assert model_prefetch.is_enabled() is True

    def test_mx_server_address_also_counts(self, monkeypatch):
        monkeypatch.setenv("MODEL_EXPRESS_NO_SHARED_STORAGE", "1")
        monkeypatch.setenv("MX_SERVER_ADDRESS", "mx:8001")
        assert model_prefetch.is_enabled() is True


class TestIsRepoId:
    @pytest.mark.parametrize("model", ["org/model", "model", "org/model-v2.5"])
    def test_repo_ids(self, model):
        assert model_prefetch.is_repo_id(model) is True

    @pytest.mark.parametrize(
        "model",
        ["", "/abs/path", "/home/dynamo/.cache/huggingface/hub/models--org--model", "a/b/c"],
    )
    def test_not_repo_ids(self, model):
        assert model_prefetch.is_repo_id(model) is False

    def test_existing_local_directory_is_not_a_repo_id(self, tmp_path, monkeypatch):
        (tmp_path / "local-model").mkdir()
        monkeypatch.chdir(tmp_path)
        assert model_prefetch.is_repo_id("local-model") is False


class TestEnsureMetadata:
    def test_no_op_when_disabled(self, fake_client):
        assert model_prefetch.ensure_metadata(REPO) is None
        assert FakeClient.instances == []

    def test_no_op_for_local_path(self, enabled, fake_client, tmp_path):
        assert model_prefetch.ensure_metadata(str(tmp_path)) is None
        assert FakeClient.instances == []

    def test_installs_and_returns_snapshot(self, enabled, fake_client):
        assert model_prefetch.ensure_metadata(REPO) == fake_client
        assert FakeClient.instances[0].calls == [REPO]

    def test_second_call_does_not_hit_the_server(self, enabled, fake_client):
        first = model_prefetch.ensure_metadata(REPO)
        second = model_prefetch.ensure_metadata(REPO)

        assert first == second
        assert len(FakeClient.instances) == 1

    def test_failure_is_retryable(self, enabled, fake_client, monkeypatch):
        def failing_factory(**kwargs):
            client = FakeClient(**kwargs)
            client.error = RuntimeError("server down")
            return client

        monkeypatch.setattr("modelexpress.model_client.ModelCacheClient", failing_factory)
        with pytest.raises(RuntimeError, match="server down"):
            model_prefetch.ensure_metadata(REPO)

        def working_factory(**kwargs):
            client = FakeClient(**kwargs)
            client.snapshot = fake_client
            return client

        monkeypatch.setattr("modelexpress.model_client.ModelCacheClient", working_factory)
        assert model_prefetch.ensure_metadata(REPO) == fake_client

    def test_passes_configured_chunk_size(self, enabled, fake_client, monkeypatch):
        monkeypatch.setenv("MODEL_EXPRESS_TRANSFER_CHUNK_SIZE", "65536")
        model_prefetch.ensure_metadata(REPO)
        assert FakeClient.instances[0].kwargs["chunk_size"] == 65536

    @pytest.mark.parametrize("raw", ["not-a-number", "0", "-1", "99999999999999"])
    def test_bad_chunk_size_falls_back(self, enabled, fake_client, monkeypatch, raw):
        """A bad env var must not be the reason a worker fails to start."""
        monkeypatch.setenv("MODEL_EXPRESS_TRANSFER_CHUNK_SIZE", raw)
        model_prefetch.ensure_metadata(REPO)
        assert FakeClient.instances[0].kwargs["chunk_size"] is None


class TestConcurrentEnsureMetadata:
    """A second caller arriving mid-install must get the snapshot, not None.

    The engine resolves the model immediately after ensure_metadata returns, so
    handing back None while another thread is still writing the snapshot sends
    it looking for files that do not exist yet.
    """

    def test_second_caller_waits_and_gets_the_same_snapshot(
        self, enabled, monkeypatch, tmp_path
    ):
        snapshot = tmp_path / "models--org--model" / "snapshots" / ("a" * 40)
        snapshot.mkdir(parents=True)
        installs = []

        class SlowClient:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return None

            def install_metadata_snapshot(self, repo_id, *args, **kwargs):
                installs.append(repo_id)
                time.sleep(0.3)
                return snapshot

        monkeypatch.setattr("modelexpress.model_client.ModelCacheClient", SlowClient)

        results = {}

        def call(tag, delay):
            time.sleep(delay)
            results[tag] = model_prefetch.ensure_metadata(REPO)

        threads = [
            threading.Thread(target=call, args=("first", 0.0)),
            threading.Thread(target=call, args=("second", 0.05)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert results["first"] == snapshot
        assert results["second"] == snapshot
        assert installs == [REPO]


class TestRevisionForwarding:
    """The engine's revision has to reach the client, not just be logged.

    Everything downstream -- which revision the server serves, which directory
    the snapshot lands in, which ref it is reachable by -- follows from this
    one value being passed on.
    """

    def test_commit_hash_is_forwarded(self, enabled, fake_client):
        model_prefetch.ensure_metadata(REPO, COMMIT)
        assert FakeClient.instances[-1].revisions == [COMMIT]

    def test_branch_name_is_forwarded(self, enabled, fake_client):
        model_prefetch.ensure_metadata(REPO, "refs/pr/1")
        assert FakeClient.instances[-1].revisions == ["refs/pr/1"]

    def test_no_revision_stays_unpinned(self, enabled, fake_client):
        model_prefetch.ensure_metadata(REPO)
        assert FakeClient.instances[-1].revisions == [None]

    def test_empty_revision_stays_unpinned(self, enabled, fake_client):
        """An engine that fills the field with "" has not pinned anything."""
        model_prefetch.ensure_metadata(REPO, "")
        assert FakeClient.instances[-1].revisions == [None]


class TestRevisionScopedDedup:
    def test_same_revision_installs_once(self, enabled, fake_client):
        first = model_prefetch.ensure_metadata(REPO, COMMIT)
        second = model_prefetch.ensure_metadata(REPO, COMMIT)

        assert first == second
        assert len(FakeClient.instances) == 1

    def test_second_revision_is_a_second_install(self, enabled, fake_client):
        """Two revisions of one model are two installs, not one.

        Keyed by repo id alone, the second request would be served the first
        revision's snapshot -- a revision the engine never asked for.
        """
        model_prefetch.ensure_metadata(REPO, COMMIT)
        model_prefetch.ensure_metadata(REPO, "f" * 40)

        assert len(FakeClient.instances) == 2
        assert FakeClient.instances[-1].revisions == ["f" * 40]


class TestRepoIdFor:
    def test_maps_snapshot_path_back(self, enabled, fake_client):
        snapshot = model_prefetch.ensure_metadata(REPO)
        assert model_prefetch.repo_id_for(snapshot) == REPO
        assert model_prefetch.repo_id_for(str(snapshot) + "/") == REPO

    def test_passes_through_repo_id(self):
        assert model_prefetch.repo_id_for(REPO) == REPO

    def test_none_for_unknown_path(self, tmp_path):
        assert model_prefetch.repo_id_for(tmp_path / "unknown") is None

    def test_none_for_unregistered_local_model_dir(self, tmp_path):
        assert model_prefetch.repo_id_for("/opt/models/llama") is None

    def test_accepts_path_objects(self, enabled, fake_client):
        snapshot = model_prefetch.ensure_metadata(REPO)
        assert model_prefetch.repo_id_for(Path(snapshot)) == REPO


class TestRepoIdFromCachePath:
    """vLLM loads weights in a separate EngineCore process.

    That process never runs the prefetch, so the in-process record is empty
    there and the cache layout has to carry the repo id on its own.
    """

    def test_recovers_from_snapshot_path_without_any_record(self, tmp_path):
        path = tmp_path / "models--Qwen--Qwen2.5-0.5B-Instruct" / "snapshots" / ("a" * 40)
        assert model_prefetch.repo_id_for(path) == "Qwen/Qwen2.5-0.5B-Instruct"
        assert model_prefetch._snapshot_to_repo_id == {}

    def test_recovers_from_repo_root(self, tmp_path):
        path = tmp_path / "models--org--model"
        assert model_prefetch.repo_id_from_cache_path(path) == "org/model"

    def test_recovers_from_file_inside_snapshot(self, tmp_path):
        path = (
            tmp_path / "models--org--model" / "snapshots" / ("b" * 40) / "config.json"
        )
        assert model_prefetch.repo_id_from_cache_path(path) == "org/model"

    def test_single_segment_repo(self, tmp_path):
        assert model_prefetch.repo_id_from_cache_path(tmp_path / "models--gpt2") == "gpt2"

    @pytest.mark.parametrize(
        "path", ["/opt/models/llama", "/home/dynamo/.cache/huggingface/hub", "/"]
    )
    def test_none_for_non_cache_paths(self, path):
        assert model_prefetch.repo_id_from_cache_path(path) is None
