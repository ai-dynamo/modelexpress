# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import errno
import logging
import os
import shutil
import sys
from types import SimpleNamespace

import pytest
from modelexpress_rl.inference import checkpoint_store as store_module
from modelexpress_rl.inference.checkpoint_store import (
    CheckpointCacheCapacityError,
    CheckpointState,
    LocalCheckpointStore,
)


@pytest.mark.parametrize("clone", [True, False])
def test_materialization_is_independent_and_preserves_copy_metadata(
    monkeypatch, tmp_path, clone
):
    monkeypatch.setattr(store_module.sys, "platform", "linux" if clone else "darwin")
    calls = []

    def clone_file(destination, operation, source):
        calls.append(operation)
        os.write(destination, os.read(source, 1024))

    monkeypatch.setattr(store_module.fcntl, "ioctl", clone_file)
    store = LocalCheckpointStore(root=tmp_path, model_name="model")
    store.initialize()
    base = store.full_path("base")
    base.mkdir()
    weights = base / "weights"
    weights.write_bytes(b"canonical")
    weights.chmod(0o640)
    os.utime(weights, ns=(1_700_000_000_000_000_000,) * 2)
    (base / "alias").symlink_to("weights")
    (base / "nested").mkdir()
    (base / "nested" / "config").write_bytes(b"config")
    target = store.materialized_path("delta")

    with store.replace_directory(target, copy_from=base) as temporary:
        assert not target.exists()
        copied = temporary / "weights"
        assert copied.read_bytes() == b"canonical"
        assert copied.stat().st_ino != weights.stat().st_ino
        assert copied.stat().st_mode == weights.stat().st_mode
        assert copied.stat().st_mtime_ns == weights.stat().st_mtime_ns
        assert not (temporary / "alias").is_symlink()
        assert (temporary / "alias").read_bytes() == b"canonical"
        assert (temporary / "nested" / "config").read_bytes() == b"config"
        copied.write_bytes(b"delta")

    assert weights.read_bytes() == b"canonical"
    assert (target / "weights").read_bytes() == b"delta"
    assert bool(calls) == clone
    assert not target.with_suffix(".tmp").exists()


@pytest.mark.parametrize(
    "error",
    [errno.EBADF, errno.EOPNOTSUPP, errno.ENOTTY, errno.EXDEV, errno.EINVAL, errno.ENOSYS],
)
def test_unsupported_clone_falls_back_to_complete_copy(monkeypatch, tmp_path, error):
    monkeypatch.setattr(store_module.sys, "platform", "linux")

    def unsupported(destination, operation, source):
        os.write(destination, b"partial clone longer than source")
        raise OSError(error, "unsupported clone")

    monkeypatch.setattr(store_module.fcntl, "ioctl", unsupported)
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"original")
    store_module._copy_checkpoint_file(str(source), str(destination))
    assert destination.read_bytes() == b"original"
    destination.write_bytes(b"changed")
    assert source.read_bytes() == b"original"


@pytest.mark.parametrize(
    "error", [errno.EIO, errno.ENOSPC, errno.EACCES, errno.EPERM]
)
@pytest.mark.parametrize("existing_target", [False, True])
def test_clone_failure_cleans_partial_tree_without_publishing(
    monkeypatch, tmp_path, error, existing_target
):
    monkeypatch.setattr(store_module.sys, "platform", "linux")

    def fail(destination, operation, source):
        os.write(destination, b"partial")
        raise OSError(error, "clone failed")

    monkeypatch.setattr(store_module.fcntl, "ioctl", fail)
    store = LocalCheckpointStore(root=tmp_path, model_name="model")
    store.initialize()
    base = store.full_path("base")
    base.mkdir()
    (base / "weights").write_bytes(b"canonical")
    target = store.materialized_path("delta")
    if existing_target:
        target.mkdir()
        (target / "old").write_bytes(b"old")
    with (
        pytest.raises(shutil.Error, match="clone failed"),
        store.replace_directory(target, copy_from=base),
    ):
        pytest.fail("must not publish a failed copy")
    assert not target.with_suffix(".tmp").exists()
    assert target.exists() == existing_target
    if existing_target:
        assert list(target.iterdir()) == [target / "old"]
        assert (target / "old").read_bytes() == b"old"
    assert (base / "weights").read_bytes() == b"canonical"


@pytest.mark.parametrize("failure", ["fallback", "metadata", "delta"])
def test_materialization_failure_retains_original_target(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(store_module.sys, "platform", "linux")

    def clone(destination, operation, source):
        if failure == "fallback":
            raise OSError(errno.EOPNOTSUPP, "unsupported")
        os.write(destination, os.read(source, 1024))

    def fail(*args, **kwargs):
        raise OSError(errno.ENOSPC, "injected failure")

    monkeypatch.setattr(store_module.fcntl, "ioctl", clone)
    if failure == "fallback":
        monkeypatch.setattr(store_module.shutil, "copy2", fail)
    elif failure == "metadata":
        monkeypatch.setattr(store_module.shutil, "copystat", fail)
    store = LocalCheckpointStore(root=tmp_path, model_name="model")
    store.initialize()
    base = store.full_path("base")
    base.mkdir()
    (base / "weights").write_bytes(b"canonical")
    target = store.materialized_path("delta")
    target.mkdir()
    (target / "old").write_bytes(b"old")

    with (
        pytest.raises((OSError, shutil.Error), match="injected failure"),
        store.replace_directory(target, copy_from=base) as temporary,
    ):
        (temporary / "weights").write_bytes(b"partial delta")
        fail()

    assert not target.with_suffix(".tmp").exists()
    assert list(target.iterdir()) == [target / "old"]
    assert (target / "old").read_bytes() == b"old"
    assert (base / "weights").read_bytes() == b"canonical"


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux FICLONE")
def test_real_reflink_keeps_canonical_bytes_private(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.write_bytes(b"canonical" * 8192)
    # Probe directly so fallback cannot masquerade as real reflink coverage.
    with source.open("rb") as src, destination.open("xb") as dst:
        try:
            store_module.fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())
        except OSError as error:
            if error.errno in {
                errno.EBADF,
                errno.EOPNOTSUPP,
                errno.ENOTTY,
                errno.EXDEV,
                errno.EINVAL,
                errno.ENOSYS,
            }:
                pytest.skip(f"test filesystem does not support FICLONE: {error}")
            raise
    destination.unlink()
    store_module._copy_checkpoint_file(str(source), str(destination))
    assert destination.read_bytes() == source.read_bytes()
    assert destination.stat().st_ino != source.stat().st_ino
    with destination.open("r+b") as handle:
        handle.write(b"modified!")
    assert source.read_bytes() == b"canonical" * 8192


def test_store_owns_the_versioned_layout_and_state(tmp_path):
    store = LocalCheckpointStore(root=tmp_path, model_name="test/model")
    store.initialize()
    checkpoint = store.full_path("base/a")
    checkpoint.mkdir()
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"weights")
    chain = {
        "version": "base/a",
        "full_version": "base/a",
        "deltas": [],
    }

    store.write_chain("base/a", chain)
    store.write_state(
        status=CheckpointState.READY,
        version="base/a",
        checkpoint_paths=[weights],
    )
    store.activate("base/a")

    assert store.cache == tmp_path / "test%2Fmodel"
    assert checkpoint == store.full_cache / "base%2Fa"
    assert store.chain("base/a") == chain
    assert store.checkpoint_path("base/a") == checkpoint
    state = store.state()
    assert state is not None
    assert state.status is CheckpointState.READY
    assert state.version == "base/a"
    assert state.files is not None
    assert state.files["model.safetensors"][0] == len(b"weights")
    assert store.active_version() == "base/a"


def test_store_encodes_dot_path_components(tmp_path):
    store = LocalCheckpointStore(root=tmp_path, model_name="..")
    store.initialize()

    assert store.cache == tmp_path / "%2E%2E"
    assert store.full_path(".") == store.full_cache / "%2E"
    assert store.delta_path("..") == store.delta_cache / "%2E%2E"
    assert store.chain_path("..") == store.chain_cache / "%2E%2E.json"


def test_store_directory_replacement_rolls_back_on_failure(tmp_path):
    store = LocalCheckpointStore(root=tmp_path, model_name="test/model")
    store.initialize()
    target = store.full_path("v1")
    target.mkdir()
    (target / "original").write_text("original")

    with pytest.raises(RuntimeError, match="injected failure"):
        with store.replace_directory(target) as temporary:
            (temporary / "replacement").write_text("replacement")
            raise RuntimeError("injected failure")

    assert (target / "original").read_text() == "original"
    assert not target.with_name("v1.tmp").exists()


def test_store_rejects_changed_artifacts_and_source_identity(tmp_path):
    store = LocalCheckpointStore(root=tmp_path, model_name="test/model")
    store.initialize()
    artifact = store.delta_path("v1")
    artifact.mkdir()
    shard = artifact / "model.safetensors"
    shard.write_bytes(b"delta")
    source = {"uri": "s3://weights/v1/index.json"}
    store.record_artifact(artifact, source=source)

    store.verify_artifact_source(artifact, source)
    with pytest.raises(ValueError, match="different source identity"):
        store.verify_artifact_source(
            artifact,
            {"uri": "s3://weights/other/index.json"},
        )

    shard.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        store.verify_artifact(artifact)


def test_store_evicts_stale_lineage_without_touching_active_lineage(
    monkeypatch, tmp_path
):
    store = LocalCheckpointStore(root=tmp_path, model_name="test/model")
    store.initialize()

    active_full = store.full_path("base")
    active_full.mkdir()
    (active_full / "weights").write_bytes(b"base")
    store.record_artifact(active_full)
    active_delta = store.delta_path("active")
    active_delta.mkdir()
    (active_delta / "weights").write_bytes(b"delta")
    store.record_artifact(active_delta)
    active_materialized = store.materialized_path("active")
    active_materialized.mkdir()
    (active_materialized / "weights").write_bytes(b"active")
    store.write_chain(
        "active",
        {"version": "active", "full_version": "base", "deltas": ["active"]},
    )
    store.activate("active")

    stale_full = store.full_path("stale-full")
    stale_full.mkdir()
    (stale_full / "weights").write_bytes(b"canonical")
    store.record_artifact(stale_full)
    stale_delta = store.delta_path("stale-delta")
    stale_delta.mkdir()
    (stale_delta / "weights").write_bytes(b"stale-delta")
    store.record_artifact(stale_delta)
    store.write_chain(
        "stale-full",
        {
            "version": "stale-full",
            "full_version": "stale-full",
            "deltas": [],
        },
    )
    store.write_chain(
        "stale-delta",
        {
            "version": "stale-delta",
            "full_version": "stale-full",
            "deltas": ["stale-delta"],
        },
    )
    stale_materialized = store.materialized_path("stale-derived")
    stale_materialized.mkdir()
    (stale_materialized / "weights").write_bytes(b"derived")

    active_size = sum(
        (path / "weights").stat().st_size
        for path in (active_full, active_delta, active_materialized)
    )
    limited = LocalCheckpointStore(
        root=tmp_path,
        model_name="test/model",
        max_size_bytes=active_size + (stale_full / "weights").stat().st_size,
    )
    cache_size_bytes = limited.cache_size_bytes
    cache_size_calls = 0

    def count_cache_size_bytes():
        nonlocal cache_size_calls
        cache_size_calls += 1
        return cache_size_bytes()

    monkeypatch.setattr(limited, "cache_size_bytes", count_cache_size_bytes)
    limited.enforce_capacity(protected_versions={"active"})

    assert active_full.exists()
    assert active_delta.exists()
    assert active_materialized.exists()
    assert stale_full.exists()
    assert not stale_delta.exists()
    assert not stale_materialized.exists()
    assert store.chain_path("stale-full").exists()
    assert not store.chain_path("stale-delta").exists()
    assert cache_size_calls == 1

    limited.max_size_bytes = active_size
    limited.enforce_capacity(protected_versions={"active"})

    assert not stale_full.exists()
    assert not store.chain_path("stale-full").exists()
    assert cache_size_bytes() <= active_size


def test_store_rejects_quota_smaller_than_protected_lineage(tmp_path):
    store = LocalCheckpointStore(root=tmp_path, model_name="test/model")
    store.initialize()
    active = store.full_path("active")
    active.mkdir()
    (active / "weights").write_bytes(b"active")
    store.record_artifact(active)
    store.write_chain(
        "active",
        {"version": "active", "full_version": "active", "deltas": []},
    )
    store.activate("active")

    limited = LocalCheckpointStore(
        root=tmp_path,
        model_name="test/model",
        max_size_bytes=store.cache_size_bytes(),
    )
    with pytest.raises(
        CheckpointCacheCapacityError, match="checkpoint cache quota"
    ):
        limited.ensure_capacity(1, protected_versions={"active"})

    assert active.exists()
    assert limited.active_version() == "active"


def test_store_rejects_write_larger_than_filesystem_free_space(
    monkeypatch, tmp_path
):
    store = LocalCheckpointStore(root=tmp_path, model_name="test/model")
    store.initialize()
    monkeypatch.setattr(
        "modelexpress_rl.inference.checkpoint_store.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=3),
    )

    with pytest.raises(
        CheckpointCacheCapacityError, match="filesystem has 3 bytes free"
    ):
        store.ensure_capacity(4)


@pytest.mark.parametrize("quota_gb, expected_gb", [(500, 300), (None, 300), (150, 150)])
def test_store_caps_quota_to_disk_space_and_logs(
    monkeypatch, tmp_path, caplog, quota_gb, expected_gb
):
    gb = 1_000_000_000
    monkeypatch.setattr(
        "modelexpress_rl.inference.checkpoint_store.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=300 * gb),
    )
    store = LocalCheckpointStore(
        root=tmp_path,
        model_name="test/model",
        max_size_bytes=quota_gb * gb if quota_gb is not None else None,
    )

    with caplog.at_level(logging.INFO):
        store.initialize()

    assert store.max_size_bytes == expected_gb * gb
    if expected_gb == 300:
        assert "capped at 300.00 GB (free=300.00 GB)" in caplog.text
    else:
        assert not caplog.records


def test_store_disk_cap_accounts_for_existing_cache(monkeypatch, tmp_path):
    store = LocalCheckpointStore(root=tmp_path, model_name="test/model")
    store.initialize()
    active = store.full_path("active")
    active.mkdir()
    (active / "weights").write_bytes(b"x" * 100)
    monkeypatch.setattr(
        "modelexpress_rl.inference.checkpoint_store.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=300),
    )
    resumed = LocalCheckpointStore(
        root=tmp_path, model_name="test/model", max_size_bytes=500
    )

    resumed.initialize()
    resumed.ensure_capacity(300, protected_versions={"active"})

    assert resumed.max_size_bytes == 400
    with pytest.raises(CheckpointCacheCapacityError):
        resumed.ensure_capacity(301, protected_versions={"active"})
    assert (active / "weights").read_bytes() == b"x" * 100


@pytest.mark.parametrize("free_bytes", [0, 50, 100])
def test_store_caps_quota_to_all_available_disk_space(
    monkeypatch, tmp_path, free_bytes
):
    monkeypatch.setattr(
        "modelexpress_rl.inference.checkpoint_store.shutil.disk_usage",
        lambda _path: SimpleNamespace(total=1000, free=free_bytes),
    )
    store = LocalCheckpointStore(root=tmp_path, model_name="test/model")
    store.initialize()

    assert store.max_size_bytes == free_bytes
    store.ensure_capacity(free_bytes)
    with pytest.raises(CheckpointCacheCapacityError, match="filesystem has"):
        store.ensure_capacity(free_bytes + 1)


def test_store_evicts_stale_checkpoint_to_free_disk_space(monkeypatch, tmp_path):
    store = LocalCheckpointStore(
        root=tmp_path, model_name="test/model", max_size_bytes=500
    )
    store.initialize()
    active = store.full_path("active")
    active.mkdir()
    (active / "weights").write_bytes(b"x" * 60)
    stale = store.materialized_path("stale")
    stale.mkdir()
    (stale / "weights").write_bytes(b"x" * 40)
    monkeypatch.setattr(
        "modelexpress_rl.inference.checkpoint_store.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=20 if stale.exists() else 60),
    )

    store.ensure_capacity(50, protected_versions={"active"})

    assert not stale.exists()
    assert (active / "weights").read_bytes() == b"x" * 60
