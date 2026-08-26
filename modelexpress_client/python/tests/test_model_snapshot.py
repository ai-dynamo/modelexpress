# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Hugging Face cache layout used by server-streamed models."""

import shutil

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
        assert cache.resolve_snapshot({"config.json": 2}, COMMIT) == snapshot

    def test_none_when_file_missing(self, cache):
        _write(cache, {"config.json": b"{}"})
        assert cache.resolve_snapshot({"config.json": 2, "tokenizer.json": 5}, COMMIT) is None

    def test_none_when_size_differs(self, cache):
        _write(cache, {"config.json": b"{}"})
        assert cache.resolve_snapshot({"config.json": 99}, COMMIT) is None

    def test_none_without_ref(self, cache):
        _write(cache, {"config.json": b"{}"})
        (cache.repo_root / "refs" / MAIN_REF).unlink()
        assert cache.resolve_snapshot({"config.json": 2}, COMMIT) is None

    def test_none_when_server_named_no_revision(self, cache):
        """No reported revision is not proof the local snapshot is current."""
        _write(cache, {"config.json": b"{}"})
        assert cache.resolve_snapshot({"config.json": 2}, None) is None

    def test_none_when_revision_differs(self, cache):
        """The case a manifest cannot catch: same names and sizes, new commit.

        This is what makes size-only reuse unsafe. Without the commit check
        the stale snapshot is returned and no stream ever opens to notice.
        """
        _write(cache, {"config.json": b"{}"})
        assert cache.resolve_snapshot({"config.json": 2}, OTHER_COMMIT) is None


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

    def test_rollback_restores_a_replaced_shard(self, cache):
        """A refresh that fails part-way must not cost the snapshot a shard.

        Publishing shard A overwrites the copy already on disk. If the patch
        then fails on shard B, rolling back by deleting what it published
        would leave the snapshot short of a shard it had before the patch
        started -- worse than the partial write the rollback exists to avoid.
        """
        snapshot = _write(
            cache,
            {"shard-1.safetensors": b"old-one", "shard-2.safetensors": b"old-two"},
        )

        patch = cache.patch(snapshot)
        patch.begin_file("shard-1.safetensors")
        patch.write(b"new-one")
        patch.end_file()
        patch.rollback()

        assert (snapshot / "shard-1.safetensors").read_bytes() == b"old-one"
        assert (snapshot / "shard-2.safetensors").read_bytes() == b"old-two"
        assert sorted(p.name for p in snapshot.iterdir()) == [
            "shard-1.safetensors",
            "shard-2.safetensors",
        ]

    def test_rollback_removes_a_newly_added_file(self, cache):
        snapshot = _write(cache, {"config.json": b"{}"})

        patch = cache.patch(snapshot)
        patch.begin_file("model.safetensors")
        patch.write(b"weights")
        patch.end_file()
        patch.rollback()

        assert not (snapshot / "model.safetensors").exists()
        assert list(snapshot.iterdir()) == [snapshot / "config.json"]

    def test_commit_drops_the_backups(self, cache):
        snapshot = _write(cache, {"shard-1.safetensors": b"old-one"})

        patch = cache.patch(snapshot)
        patch.begin_file("shard-1.safetensors")
        patch.write(b"new-one")
        patch.end_file()
        patch.commit()
        patch.close()

        assert (snapshot / "shard-1.safetensors").read_bytes() == b"new-one"
        assert list(snapshot.iterdir()) == [snapshot / "shard-1.safetensors"]

    def test_rollback_after_commit_keeps_the_published_files(self, cache):
        snapshot = _write(cache, {"shard-1.safetensors": b"old-one"})

        patch = cache.patch(snapshot)
        patch.begin_file("shard-1.safetensors")
        patch.write(b"new-one")
        patch.end_file()
        patch.commit()
        patch.rollback()

        assert (snapshot / "shard-1.safetensors").read_bytes() == b"new-one"


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


class TestPinnedRevisionLayout:
    """What a pinned install must leave on disk for the engine to find it.

    The engine looks the snapshot up by the revision *it* asked for, so the
    layout has to answer that question rather than the one the server
    happened to answer.
    """

    def test_commit_pin_writes_no_ref(self, cache):
        """A commit hash resolves by directory name; a ref would be noise."""
        cache.write_revision_ref(COMMIT, COMMIT)
        assert not (cache.repo_root / "refs").exists()

    def test_branch_pin_writes_its_own_ref(self, cache):
        cache.write_revision_ref(COMMIT, "refs/pr/1")

        ref = cache.repo_root / "refs" / "refs" / "pr" / "1"
        assert ref.read_text() == COMMIT

    def test_unpinned_still_writes_main(self, cache):
        cache.write_revision_ref(COMMIT, None)
        assert (cache.repo_root / "refs" / MAIN_REF).read_text() == COMMIT

    def test_pinning_leaves_main_alone(self, cache):
        """Pointing main at a pin would misdirect every later unpinned read.

        The default revision and the pinned one are different questions. A
        worker that pinned an older commit must not answer the first with the
        second -- for itself later, or for the next worker sharing the cache.
        """
        cache.write_revision_ref(COMMIT, None)
        cache.write_revision_ref(OTHER_COMMIT, "v1.0")

        assert (cache.repo_root / "refs" / MAIN_REF).read_text() == COMMIT

    def test_rewriting_the_same_hash_leaves_the_ref_untouched(self, cache):
        """Reuse of an unpinned snapshot asks for a ref that already matches.

        resolve_snapshot only returns a path when refs/main already holds the
        commit, so the reuse path lands here with nothing to change. The inode
        is what proves nothing happened: the write lands through a rename, and
        a rename would leave a different one.
        """
        cache.write_revision_ref(COMMIT, None)
        ref = cache.repo_root / "refs" / MAIN_REF
        before = ref.stat().st_ino

        cache.write_revision_ref(COMMIT, None)

        assert ref.stat().st_ino == before
        assert ref.read_text() == COMMIT

    def test_rewriting_a_different_hash_moves_the_ref(self, cache):
        cache.write_revision_ref(COMMIT, None)
        cache.write_revision_ref(OTHER_COMMIT, None)

        assert (cache.repo_root / "refs" / MAIN_REF).read_text() == OTHER_COMMIT

    def test_rejects_a_ref_whose_parent_is_already_a_ref(self, cache):
        """A tag foo and a branch foo/bar coexist upstream; refs/ holds one."""
        cache.write_ref("foo", COMMIT)

        with pytest.raises(ModelSnapshotError):
            cache.write_ref("foo/bar", OTHER_COMMIT)

    def test_rejects_a_ref_that_is_already_a_directory(self, cache):
        """The other order: refs/foo exists as a directory of nested refs."""
        cache.write_ref("foo/bar", COMMIT)

        with pytest.raises(ModelSnapshotError):
            cache.write_ref("foo", OTHER_COMMIT)

    @pytest.mark.parametrize("name", ["../escape", "/abs", "", "a/../b"])
    def test_rejects_unsafe_ref_names(self, cache, name):
        """The name comes from engine configuration, so it is not trusted."""
        with pytest.raises(ModelSnapshotError):
            cache.write_ref(name, COMMIT)


def test_branch_pin_resolves_offline(cache):
    """The engine's own lookup, for a revision that is not a commit hash.

    A commit hash resolves by directory name, so only this case proves the
    ref write is what makes a branch or tag pin resolvable.
    """
    from huggingface_hub import snapshot_download

    snapshot = _write(cache, {"config.json": b"{}"})
    cache.write_revision_ref(COMMIT, "v1.0")

    resolved = snapshot_download(
        "org/model",
        revision="v1.0",
        cache_dir=str(cache.cache_root),
        local_files_only=True,
    )

    assert resolved == str(snapshot)


def test_commit_pin_resolves_offline_without_any_ref(cache):
    """Confirms the rule the layout relies on, in huggingface_hub itself."""
    from huggingface_hub import snapshot_download

    snapshot = _write(cache, {"config.json": b"{}"})
    shutil.rmtree(cache.repo_root / "refs")

    resolved = snapshot_download(
        "org/model",
        revision=COMMIT,
        cache_dir=str(cache.cache_root),
        local_files_only=True,
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
