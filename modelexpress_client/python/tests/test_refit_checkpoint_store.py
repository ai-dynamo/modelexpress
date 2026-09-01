# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from modelexpress_rl.inference.checkpoint_store import (
    CheckpointState,
    LocalCheckpointStore,
)


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
    assert state["files"]["model.safetensors"][0] == len(b"weights")
    assert store.active_version() == "base/a"


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
