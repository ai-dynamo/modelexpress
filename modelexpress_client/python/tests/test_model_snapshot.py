# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Hugging Face cache layout used by server-streamed models."""

import pytest

from modelexpress.model_snapshot import (
    MAIN_REF,
    ModelSnapshotCache,
    ModelSnapshotError,
    is_weight_file,
    repo_dir_name,
    resolve_cache_root,
    safe_commit_hash,
    safe_relative_path,
    split_by_weight,
)

COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_EXPRESS_CACHE_DIRECTORY", raising=False)
    return ModelSnapshotCache("org/model", tmp_path)


def _write(cache, files, commit=COMMIT):
    """Publish ``files`` ({path: bytes}) as a snapshot and return its path."""
    staging = cache.staging()
    for relative_path, payload in files.items():
        staging.begin_file(relative_path)
        staging.write(payload)
        staging.end_file()
    expected = {path: len(payload) for path, payload in files.items()}
    return staging.publish(commit, expected)


class TestWeightClassification:
    @pytest.mark.parametrize(
        "path",
        [
            "model.safetensors",
            "pytorch_model-00001-of-00002.bin",
            "sub/dir/model.safetensors",
            "tf_model.h5",
            "flax_model.msgpack",
        ],
    )
    def test_weight_files(self, path):
        assert is_weight_file(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            "config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
            "README.md",
        ],
    )
    def test_metadata_files(self, path):
        assert is_weight_file(path) is False

    def test_split_preserves_order(self):
        metadata, weights = split_by_weight(
            ["config.json", "a.safetensors", "tokenizer.json", "b.bin"]
        )
        assert metadata == ["config.json", "tokenizer.json"]
        assert weights == ["a.safetensors", "b.bin"]


class TestPathValidation:
    @pytest.mark.parametrize(
        "path",
        [
            "",
            "/etc/passwd",
            "../escape",
            "sub/../../escape",
            "sub/./file",
            "back\\slash",
            "nul\x00byte",
            "trailing/",
        ],
    )
    def test_rejects_unsafe_paths(self, path):
        with pytest.raises(ModelSnapshotError):
            safe_relative_path(path)

    def test_accepts_nested_path(self):
        assert safe_relative_path("sub/dir/file.json").parts == ("sub", "dir", "file.json")

    @pytest.mark.parametrize("commit", ["", ".", "..", "a/b", "a\\b", "a\x00b"])
    def test_rejects_unsafe_commit(self, commit):
        with pytest.raises(ModelSnapshotError):
            safe_commit_hash(commit)

    def test_repo_dir_name(self):
        assert repo_dir_name("org/model") == "models--org--model"
        assert repo_dir_name("model") == "models--model"

    @pytest.mark.parametrize("name", ["", "/abs", "org/../model", "back\\slash"])
    def test_repo_dir_name_rejects_unsafe(self, name):
        with pytest.raises(ValueError):
            repo_dir_name(name)


class TestCacheRootResolution:
    def test_explicit_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MODEL_EXPRESS_CACHE_DIRECTORY", str(tmp_path / "env"))
        assert resolve_cache_root(tmp_path / "explicit") == tmp_path / "explicit"

    def test_env_used_when_no_explicit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MODEL_EXPRESS_CACHE_DIRECTORY", str(tmp_path / "env"))
        assert resolve_cache_root() == tmp_path / "env"

    def test_falls_back_to_hf_hub_cache(self, monkeypatch):
        from huggingface_hub.constants import HF_HUB_CACHE

        monkeypatch.delenv("MODEL_EXPRESS_CACHE_DIRECTORY", raising=False)
        assert str(resolve_cache_root()) == str(HF_HUB_CACHE)


class TestPublish:
    def test_layout_and_ref(self, cache):
        snapshot = _write(cache, {"config.json": b"{}", "sub/tok.json": b"[]"})

        assert snapshot == cache.repo_root / "snapshots" / COMMIT
        assert (snapshot / "config.json").read_bytes() == b"{}"
        assert (snapshot / "sub" / "tok.json").read_bytes() == b"[]"
        assert (cache.repo_root / "refs" / MAIN_REF).read_text() == COMMIT
        assert cache.read_main_ref() == COMMIT

    def test_no_staging_directory_left_behind(self, cache):
        _write(cache, {"config.json": b"{}"})
        leftovers = [
            p.name for p in cache.repo_root.iterdir() if p.name.startswith(".modelexpress-")
        ]
        assert leftovers == []

    def test_discard_removes_staging(self, cache):
        staging = cache.staging()
        staging_path = staging.path
        staging.begin_file("config.json")
        staging.write(b"{}")
        staging.discard()

        assert not staging_path.exists()
        assert not (cache.repo_root / "snapshots").exists()

    def test_republish_same_commit_reuses_complete_snapshot(self, cache):
        snapshot = _write(cache, {"config.json": b"{}"})
        (snapshot / "extra.json").write_text("kept")

        again = _write(cache, {"config.json": b"{}"})

        assert again == snapshot
        assert (snapshot / "extra.json").read_text() == "kept"

    def test_republish_updates_incomplete_snapshot(self, cache):
        snapshot = _write(cache, {"config.json": b"{}"})
        (snapshot / "config.json").unlink()

        again = _write(cache, {"config.json": b"{'v': 2}"})

        assert again == snapshot
        assert (snapshot / "config.json").read_bytes() == b"{'v': 2}"

    def test_republish_keeps_files_the_manifest_does_not_mention(self, cache):
        """Installing metadata must not delete an already-installed weight set.

        The commit hash comes from the server resolving ``main``, so a second
        install targets the same ``snapshots/<commit>/``. The manifest passed
        here covers metadata only, so replacing the directory wholesale would
        drop weights that no expected-file check ever looks at.
        """
        snapshot = _write(cache, {"config.json": b"{}"})
        weights = snapshot / "model.safetensors"
        weights.write_bytes(b"W" * 64)
        (snapshot / "shards" / "extra").mkdir(parents=True)
        (snapshot / "shards" / "extra" / "part.safetensors").write_bytes(b"S" * 16)

        again = _write(cache, {"config.json": b"{}", "chat_template.jinja": b"tpl"})

        assert again == snapshot
        assert weights.read_bytes() == b"W" * 64
        assert (snapshot / "shards" / "extra" / "part.safetensors").read_bytes() == b"S" * 16
        assert (snapshot / "chat_template.jinja").read_bytes() == b"tpl"

    def test_republish_leaves_no_staging_or_stale_directories(self, cache):
        snapshot = _write(cache, {"config.json": b"{}"})
        (snapshot / "model.safetensors").write_bytes(b"W")

        _write(cache, {"config.json": b"{}", "chat_template.jinja": b"tpl"})

        leftovers = [
            entry.name
            for entry in cache.repo_root.iterdir()
            if entry.name.startswith((".modelexpress-stale-", ".modelexpress-staging-"))
        ]
        assert leftovers == []

    def test_second_commit_moves_ref(self, cache):
        _write(cache, {"config.json": b"{}"}, commit=COMMIT)
        _write(cache, {"config.json": b"{}"}, commit=OTHER_COMMIT)

        assert cache.read_main_ref() == OTHER_COMMIT
        assert (cache.repo_root / "snapshots" / COMMIT).is_dir()

    def test_rejects_unsafe_streamed_path(self, cache):
        staging = cache.staging()
        with pytest.raises(ModelSnapshotError):
            staging.begin_file("../escape.json")
        staging.discard()

    def test_rejects_overlapping_files(self, cache):
        staging = cache.staging()
        staging.begin_file("a.json")
        with pytest.raises(ModelSnapshotError):
            staging.begin_file("b.json")
        staging.discard()


class TestResolveSnapshot:
    def test_returns_snapshot_when_files_present(self, cache):
        snapshot = _write(cache, {"config.json": b"{}"})
        assert cache.resolve_snapshot({"config.json": 2}) == snapshot

    def test_none_when_file_missing(self, cache):
        _write(cache, {"config.json": b"{}"})
        assert cache.resolve_snapshot({"config.json": 2, "tokenizer.json": 5}) is None

    def test_none_when_size_differs(self, cache):
        _write(cache, {"config.json": b"{}"})
        assert cache.resolve_snapshot({"config.json": 99}) is None

    def test_none_without_ref(self, cache):
        _write(cache, {"config.json": b"{}"})
        (cache.repo_root / "refs" / MAIN_REF).unlink()
        assert cache.resolve_snapshot({"config.json": 2}) is None


class TestPatch:
    def test_adds_files_to_published_snapshot(self, cache):
        snapshot = _write(cache, {"config.json": b"{}"})

        patch = cache.patch(snapshot)
        patch.begin_file("model.safetensors")
        patch.write(b"weights")
        patch.end_file()
        patch.close()

        assert (snapshot / "model.safetensors").read_bytes() == b"weights"
        assert (snapshot / "config.json").read_bytes() == b"{}"
        assert cache.read_main_ref() == COMMIT

    def test_leaves_no_temp_file_on_abort(self, cache):
        snapshot = _write(cache, {"config.json": b"{}"})

        patch = cache.patch(snapshot)
        patch.begin_file("model.safetensors")
        patch.write(b"partial")
        patch.close()

        assert not (snapshot / "model.safetensors").exists()
        assert list(snapshot.iterdir()) == [snapshot / "config.json"]

    def test_rejects_missing_snapshot(self, cache):
        with pytest.raises(ModelSnapshotError):
            cache.patch(cache.repo_root / "snapshots" / COMMIT)


class TestLock:
    def test_released_after_context(self, cache):
        with cache.lock():
            pass
        with cache.lock():
            pass
        assert (cache.repo_root / ".modelexpress.lock").is_file()


def test_published_snapshot_resolves_offline(cache, monkeypatch):
    """huggingface_hub must resolve the published layout with no network.

    Regression guard for issue #569: the engine resolves the model through
    ``snapshot_download(local_files_only=True)`` long before the weight loader
    runs, and that call fails with LocalEntryNotFoundError unless refs/main
    points at a snapshot directory.
    """
    from huggingface_hub import snapshot_download

    snapshot = _write(cache, {"config.json": b"{}", "tokenizer.json": b"[]"})

    resolved = snapshot_download(
        "org/model", cache_dir=str(cache.cache_root), local_files_only=True
    )

    assert resolved == str(snapshot)


def test_snapshot_without_ref_is_unresolvable(cache):
    """The failure mode from the issue, pinned so the ref write cannot regress."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    _write(cache, {"config.json": b"{}"})
    (cache.repo_root / "refs" / MAIN_REF).unlink()

    with pytest.raises(LocalEntryNotFoundError):
        snapshot_download(
            "org/model", cache_dir=str(cache.cache_root), local_files_only=True
        )
