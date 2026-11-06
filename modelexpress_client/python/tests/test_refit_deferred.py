# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import pytest
import torch
from safetensors.torch import load_file

from modelexpress_rl import envs
from modelexpress_rl.inference.checkpoint_store import CheckpointState
from modelexpress_rl.inference.engines.vllm.installer import _with_staged_deltas
from tests.refit_fixtures import _checkpoint, _delta, _delta_snapshot, _FakeS3


def scene(tmp_path):
    objects = _delta_snapshot(
        [1.0, 2.0], [1.0, 3.0], identity="v1", base_identity="base"
    )
    objects.update(
        _delta_snapshot([1.0, 3.0], [4.0, 3.0], identity="v2", base_identity="v1")
    )
    objects.update(
        _delta_snapshot([4.0, 3.0], [4.0, 8.0], identity="v3", base_identity="v2")
    )
    checkpoint = _checkpoint(tmp_path, _FakeS3(objects), stage_deltas=True)
    return checkpoint


def prepare(checkpoint, version, base):
    return checkpoint.prepare_chain((_delta(version, base=base, identity=version),))


def weight(checkpoint):
    return load_file(checkpoint.local_checkpoint / "model.safetensors")["weight"]


def test_deferred_writes_are_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MX_REFIT_STAGED_LAYERS", raising=False)
    assert envs.MX_REFIT_STAGED_LAYERS == 0
    checkpoint = scene(tmp_path)
    prepared = prepare(checkpoint, "v1", "base")
    assert prepared.materialized
    assert checkpoint.store.state().status is CheckpointState.READY
    assert torch.equal(weight(checkpoint), torch.tensor([1.0, 3.0]))


def test_staged_chain_preserves_files_and_reloads_through_overlays(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MX_REFIT_STAGED_LAYERS", "2")
    checkpoint = scene(tmp_path)
    prepare(checkpoint, "v1", "base")
    prepared = prepare(checkpoint, "v2", "v1")
    assert not prepared.materialized
    assert checkpoint.store.state().base == "base"
    assert torch.equal(weight(checkpoint), torch.tensor([1.0, 2.0]))
    target = dict(
        _with_staged_deltas([("weight", weight(checkpoint))], prepared.overlay)
    )
    assert torch.equal(target["weight"], torch.tensor([4.0, 3.0]))
    checkpoint.activate(prepared)
    checkpoint.materialize_head()
    assert checkpoint.store.state().status is CheckpointState.READY
    assert torch.equal(weight(checkpoint), torch.tensor([4.0, 3.0]))


def test_bad_delta_at_limit_keeps_valid_rewritten_head(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_REFIT_STAGED_LAYERS", "1")
    checkpoint = scene(tmp_path)
    prepared = prepare(checkpoint, "v1", "base")
    checkpoint.activate(prepared)
    # The incoming delta has checksums for another base, so verification fails.
    checkpoint.reader.objects.update(
        _delta_snapshot([9.0, 9.0], [4.0, 3.0], identity="v2", base_identity="v1")
    )
    with pytest.raises(RuntimeError, match="checksum differs"):
        prepare(checkpoint, "v2", "v1")
    state = checkpoint.store.state()
    assert (state.version, state.status) == ("v1", CheckpointState.READY)
    assert torch.equal(weight(checkpoint), torch.tensor([1.0, 3.0]))
    checkpoint.recover_incomplete_preparation()
    assert checkpoint.store.state().version == "v1"


def test_failed_copy_rewrite_keeps_staged_head(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_REFIT_STAGED_LAYERS", "1")
    checkpoint = scene(tmp_path)
    prepared = prepare(checkpoint, "v1", "base")
    checkpoint.activate(prepared)

    def fail(*args, **kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(checkpoint, "_run_pass", fail)
    with pytest.raises(OSError, match="injected"):
        checkpoint.materialize_head()
    state = checkpoint.store.state()
    assert (state.version, state.status) == ("v1", CheckpointState.STAGED)
    assert torch.equal(weight(checkpoint), torch.tensor([1.0, 2.0]))
    checkpoint.recover_incomplete_preparation()
    assert checkpoint.store.state().status is CheckpointState.STAGED


def test_failed_materialized_rewrite_requires_recovery(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_REFIT_STAGED_LAYERS", "1")
    checkpoint = scene(tmp_path)
    checkpoint.activate(prepare(checkpoint, "v1", "base"))
    checkpoint.materialize_head()
    checkpoint.activate(prepare(checkpoint, "v2", "v1"))

    def fail(*args, **kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(checkpoint, "_run_pass", fail)
    with pytest.raises(OSError, match="injected"):
        checkpoint.materialize_head()
    assert checkpoint.store.state().status is CheckpointState.UPDATING
    checkpoint.recover_incomplete_preparation()
    assert checkpoint.store.state().version == "base"
    assert checkpoint.store.state().status is CheckpointState.READY


def test_disabling_staging_materializes_existing_head(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_REFIT_STAGED_LAYERS", "2")
    checkpoint = scene(tmp_path)
    prepare(checkpoint, "v1", "base")
    monkeypatch.setenv("MX_REFIT_STAGED_LAYERS", "0")
    prepared = prepare(checkpoint, "v1", "base")
    assert prepared.materialized
    assert torch.equal(weight(checkpoint), torch.tensor([1.0, 3.0]))


def test_staged_state_survives_reinitialization(tmp_path, monkeypatch):
    monkeypatch.setenv("MX_REFIT_STAGED_LAYERS", "2")
    checkpoint = scene(tmp_path)
    prepare(checkpoint, "v1", "base")
    from modelexpress_rl.inference.receiver import _LocalCheckpoint
    from tests.refit_fixtures import _config

    checkpoint = _LocalCheckpoint(
        model_name="test/model",
        config=dataclasses.replace(_config(tmp_path), seed_checkpoint_path=None),
        reader=checkpoint.reader,
        stage_deltas=True,
    )
    assert checkpoint.initialize()
    assert checkpoint.store.state().status is CheckpointState.STAGED
    assert prepare(checkpoint, "v1", "base").overlay


def test_layer_limit_rejects_negative_value(monkeypatch):
    monkeypatch.setenv("MX_REFIT_STAGED_LAYERS", "-1")
    with pytest.raises(ValueError, match="non-negative"):
        _ = envs.MX_REFIT_STAGED_LAYERS


def test_capacity_rejection_keeps_every_staged_dependency(tmp_path, monkeypatch):
    from modelexpress_rl.inference.receiver import _protected_versions
    from modelexpress_rl.inference.checkpoint_store import CheckpointCacheCapacityError

    monkeypatch.setenv("MX_REFIT_STAGED_LAYERS", "3")
    checkpoint = scene(tmp_path)
    prepare(checkpoint, "v1", "base")
    prepare(checkpoint, "v2", "v1")
    store = checkpoint.store
    store.max_size_bytes = store.cache_size_bytes()
    with pytest.raises(CheckpointCacheCapacityError):
        store.ensure_capacity(
            1, protected_versions=_protected_versions(store, head=True)
        )
    assert store.staging_path("v1").is_dir()
    assert store.staging_path("v2").is_dir()
    assert torch.equal(
        dict(
            _with_staged_deltas(
                [("weight", weight(checkpoint))],
                prepare(checkpoint, "v2", "v1").overlay,
            )
        )["weight"],
        torch.tensor([4.0, 3.0]),
    )
