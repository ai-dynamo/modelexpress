# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from safetensors.torch import load_file, save_file

from modelexpress_rl import envs
from modelexpress_rl.inference import receiver
from modelexpress_rl.inference.methods import canonical_delta
from tests.test_refit_s3_receiver import (
    _Adapter,
    _MemoryS3,
    _artifact,
    _full_artifact,
    _full_inputs,
    _inputs,
)


def build(monkeypatch, tmp_path, *, checksums=False):
    objects = _full_artifact(torch.tensor([7.0, 8.0]))
    uri = "s3://weights/test/v2/model.safetensors.index.json"
    if not checksums:
        index = json.loads(objects[uri])
        del index["metadata"]["checksum_format"]
        objects[uri] = json.dumps(index).encode()
    storage = _MemoryS3(objects)
    monkeypatch.setattr(canonical_delta, "S3Client", lambda **kwargs: storage)
    launch = tmp_path / "launch"
    launch.mkdir()
    save_file({"weight": torch.tensor([1.0, 2.0])}, launch / "model.safetensors")
    adapter = _Adapter(
        model_name="test/model",
        config=receiver.ObjectStorageGeneratorConfig(
            storage_type=receiver.ObjectStorageType.S3,
            initial_base_version_id="base-a",
            seed_checkpoint_path=launch,
            refit_checkpoint_dir=tmp_path / "cache",
        ),
        stream_full_checkpoints=True,
    )
    return adapter, storage


def consume(full, values=None, *, cache=True):
    values = values if values is not None else [("weight", torch.tensor([7.0, 8.0]))]
    return list(full.streaming.validate(iter(values), cache=cache))


def test_streaming_is_opt_in(monkeypatch):
    monkeypatch.delenv("MX_REFIT_FULL_STREAMING", raising=False)
    assert envs.MX_REFIT_FULL_STREAMING is False
    monkeypatch.setenv("MX_REFIT_FULL_STREAMING", "true")
    assert envs.MX_REFIT_FULL_STREAMING is True
    monkeypatch.setenv("MX_REFIT_FULL_STREAMING", "invalid")
    with pytest.raises(ValueError, match="boolean"):
        _ = envs.MX_REFIT_FULL_STREAMING


def test_prepare_only_reads_index(monkeypatch, tmp_path):
    adapter, storage = build(monkeypatch, tmp_path)
    try:
        full = adapter.stage_weight(_full_inputs())
        assert full.streaming.shard_uris == (
            "s3://weights/test/v2/model-00001-of-00001.safetensors",
        )
        assert storage.calls == ["s3://weights/test/v2/model.safetensors.index.json"]
        assert not full.path.exists()
        adapter.release_staged_weight(full)
    finally:
        adapter.close()


def test_declared_checksums_keep_verified_disk_path(monkeypatch, tmp_path):
    adapter, storage = build(monkeypatch, tmp_path, checksums=True)
    try:
        full = adapter.stage_weight(_full_inputs())
        assert full.streaming is None
        assert full.path.is_dir()
        assert any(uri.endswith(".safetensors") for uri in storage.calls)
    finally:
        adapter.close()


def test_bad_tensor_set_fails_before_any_shard_read(monkeypatch, tmp_path):
    adapter, storage = build(monkeypatch, tmp_path)
    uri = "s3://weights/test/v2/model.safetensors.index.json"
    storage.objects[uri] = json.dumps(
        {"weight_map": {"wrong": "model.safetensors"}}
    ).encode()
    try:
        with pytest.raises(RuntimeError, match="tensor set differs"):
            adapter.stage_weight(_full_inputs())
        assert storage.calls == [uri]
    finally:
        adapter.close()


def test_cache_uses_stream_bytes_and_next_delta_never_downloads_full(
    monkeypatch, tmp_path
):
    adapter, storage = build(monkeypatch, tmp_path)
    full = adapter.stage_weight(_full_inputs())
    old_state = adapter._checkpoint.store.state()
    try:
        with adapter._method.installation_context(adapter._active):
            consume(full)
        full.streaming.wait_cache()
        assert old_state.version == "base-a"
        assert adapter._checkpoint.store.state().version == "full-a"
        assert torch.equal(
            load_file(full.path / "model-00001-of-00001.safetensors")["weight"],
            torch.tensor([7.0, 8.0]),
        )
        adapter.release_staged_weight(full)
        storage.objects.update(
            _artifact(
                torch.tensor([7.0, 8.0]).view(torch.uint8).numpy(),
                torch.tensor([9.0, 10.0]).view(torch.uint8).numpy(),
                version="delta-b",
                version_label=3,
                base_version="full-a",
            )
        )
        # Removing the remote shard proves delta replay only needs the local base.
        del storage.objects["s3://weights/test/v2/model-00001-of-00001.safetensors"]
        delta = adapter.stage_chain(
            [
                _full_inputs(),
                _inputs(
                    None, base_version="full-a", version="delta-b", version_label=3
                ),
            ]
        )
        assert torch.equal(
            load_file(delta.path / "model-00001-of-00001.safetensors")["weight"],
            torch.tensor([9.0, 10.0]),
        )
        assert (
            "s3://weights/test/v2/model-00001-of-00001.safetensors" not in storage.calls
        )
        adapter.release_staged_weight(delta)
    finally:
        adapter.close()


def test_install_returns_before_disk_tail_completes(monkeypatch, tmp_path):
    from modelexpress_rl.inference.streaming_checkpoint import _CheckpointWriter

    adapter, _ = build(monkeypatch, tmp_path)
    started, unblock = threading.Event(), threading.Event()
    original = _CheckpointWriter._write_tensor

    def blocked(self, *args):
        started.set()
        assert unblock.wait(10)
        return original(self, *args)

    monkeypatch.setattr(_CheckpointWriter, "_write_tensor", blocked)
    full = adapter.stage_weight(_full_inputs())
    try:
        with adapter._method.installation_context(adapter._active):
            consume(full)
        assert started.wait(5)
        assert not full.path.exists()
        unblock.set()
        full.streaming.wait_cache()
        assert full.path.exists()
    finally:
        unblock.set()
        adapter.close()


def test_failed_install_discards_partial_cache(monkeypatch, tmp_path):
    adapter, _ = build(monkeypatch, tmp_path)
    full = adapter.stage_weight(_full_inputs())
    try:
        with pytest.raises(RuntimeError, match="install failed"):
            with adapter._method.installation_context(adapter._active):
                consume(full)
                raise RuntimeError("install failed")
        adapter.close()
        assert not full.path.exists()
        assert not full.path.with_name(full.path.name + ".tmp").exists()
    finally:
        adapter.close()


def test_non_writer_rank_does_not_create_cache(monkeypatch, tmp_path):
    adapter, _ = build(monkeypatch, tmp_path)
    full = adapter.stage_weight(_full_inputs())
    try:
        with adapter._method.installation_context(adapter._active):
            consume(full, cache=False)
        assert full.streaming._writer is None
        assert not full.path.exists()
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "tensors,match",
    [
        ([("wrong", torch.ones(2))], "unexpected"),
        ([("weight", torch.ones(3))], "metadata"),
        ([("weight", torch.ones(2, dtype=torch.float16))], "metadata"),
        ([("weight", torch.ones(2)), ("weight", torch.ones(2))], "duplicate"),
        ([], "missing"),
    ],
)
def test_invalid_stream_is_rejected(monkeypatch, tmp_path, tensors, match):
    adapter, _ = build(monkeypatch, tmp_path)
    full = adapter.stage_weight(_full_inputs())
    try:
        with pytest.raises(ValueError, match=match):
            consume(full, tensors)
    finally:
        adapter.close()


def test_snapshot_is_owned_and_preserves_raw_bytes(monkeypatch, tmp_path):
    adapter, _ = build(monkeypatch, tmp_path)
    full = adapter.stage_weight(_full_inputs())
    tensor = torch.tensor([7.0, 8.0])
    try:
        with adapter._method.installation_context(adapter._active):
            values = consume(full, [("weight", tensor)])
            assert values[0][1] is tensor
            tensor.zero_()
        full.streaming.wait_cache()
        assert torch.equal(
            load_file(full.path / "model-00001-of-00001.safetensors")["weight"],
            torch.tensor([7.0, 8.0]),
        )
    finally:
        adapter.close()


def test_cache_failure_is_reported_before_delta_without_second_download(
    monkeypatch, tmp_path
):
    from modelexpress_rl.inference.streaming_checkpoint import _CheckpointWriter

    adapter, storage = build(monkeypatch, tmp_path)
    full = adapter.stage_weight(_full_inputs())

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(_CheckpointWriter, "_write_tensor", fail)
    try:
        with adapter._method.installation_context(adapter._active):
            consume(full)
        with pytest.raises(OSError, match="disk unavailable"):
            full.streaming.wait_cache()
        adapter.release_staged_weight(full)
        calls = list(storage.calls)
        with pytest.raises(OSError, match="disk unavailable"):
            adapter.stage_chain(
                [
                    _full_inputs(),
                    _inputs(
                        None, base_version="full-a", version="delta-b", version_label=3
                    ),
                ]
            )
        assert storage.calls == calls
        assert not full.path.exists()
    finally:
        adapter.close()


def test_writer_backpressure_and_close_drain(monkeypatch, tmp_path):
    from modelexpress_rl.inference import streaming_checkpoint as module

    adapter, _ = build(monkeypatch, tmp_path)
    full = adapter.stage_weight(_full_inputs())
    started, unblock = threading.Event(), threading.Event()
    original = module._CheckpointWriter._write_tensor
    monkeypatch.setattr(module, "_CACHE_PENDING_BYTES", 8)

    def blocked(self, *args):
        started.set()
        assert unblock.wait(10)
        return original(self, *args)

    monkeypatch.setattr(module._CheckpointWriter, "_write_tensor", blocked)
    writer = module._CheckpointWriter(full.streaming)
    full.streaming._writer = writer
    try:
        writer.submit("weight", torch.ones(2))
        assert started.wait(5)
        with ThreadPoolExecutor() as pool:
            entered = threading.Event()

            def submit():
                entered.set()
                writer.submit("weight", torch.ones(2))

            pending = pool.submit(submit)
            assert entered.wait(5)
            with writer._condition:
                assert writer._pending_bytes == 8
            assert not pending.done()
            unblock.set()
            pending.result(timeout=5)
        writer.finish(False)
        adapter.close()
        assert writer._future.done()
        assert not full.path.exists()
    finally:
        unblock.set()
        adapter.close()


def test_incomplete_iterator_is_never_published(monkeypatch, tmp_path):
    adapter, _ = build(monkeypatch, tmp_path)
    full = adapter.stage_weight(_full_inputs())
    try:
        with pytest.raises(RuntimeError, match="not completely consumed"):
            with adapter._method.installation_context(adapter._active):
                iterator = full.streaming.validate(
                    iter([("weight", torch.ones(2))]), cache=True
                )
                next(iterator)
                iterator.close()
        adapter.close()
        assert not full.path.exists()
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "dtype,header_dtype", [(torch.bfloat16, "BF16"), (torch.float8_e4m3fn, "F8_E4M3")]
)
def test_cache_preserves_quantized_tensor_bytes(
    monkeypatch, tmp_path, dtype, header_dtype
):
    adapter, _ = build(monkeypatch, tmp_path)
    full = adapter.stage_weight(_full_inputs())
    full.streaming.tensor_metadata = {
        "weight": {
            "dtype": header_dtype,
            "shape": [2],
            "byte_size": 2 * torch.tensor([], dtype=dtype).element_size(),
        }
    }
    tensor = torch.tensor([1.0, 2.0]).to(dtype)
    try:
        with adapter._method.installation_context(adapter._active):
            consume(full, [("weight", tensor)])
        full.streaming.wait_cache()
        restored = load_file(full.path / "model-00001-of-00001.safetensors")["weight"]
        assert restored.dtype == tensor.dtype
        assert torch.equal(restored.view(torch.uint8), tensor.view(torch.uint8))
    finally:
        adapter.close()


def test_repeat_full_reuses_cache_without_replacing_state_files(monkeypatch, tmp_path):
    from modelexpress_rl.inference.checkpoint_store import checkpoint_files_state

    adapter, _ = build(monkeypatch, tmp_path)
    try:
        for attempt in range(2):
            full = adapter.stage_weight(_full_inputs())
            assert full.streaming is not None
            with adapter._method.installation_context(adapter._active):
                consume(full)
            full.streaming.wait_cache()
            current = checkpoint_files_state(full.path.glob("*.safetensors"))
            if attempt == 0:
                original = current
            else:
                assert current == original
            adapter.release_staged_weight(full)
    finally:
        adapter.close()


def test_other_rank_waits_for_shared_cache_then_replays_delta(monkeypatch, tmp_path):
    from modelexpress_rl.inference.streaming_checkpoint import _CheckpointWriter

    owner, storage = build(monkeypatch, tmp_path)
    follower = _Adapter(
        model_name="test/model",
        config=receiver.ObjectStorageGeneratorConfig(
            storage_type=receiver.ObjectStorageType.S3,
            initial_base_version_id="base-a",
            seed_checkpoint_path=tmp_path / "launch",
            refit_checkpoint_dir=tmp_path / "cache",
        ),
        stream_full_checkpoints=True,
    )
    started, unblock = threading.Event(), threading.Event()
    original = _CheckpointWriter._write_tensor
    writes = []

    def blocked(self, *args):
        writes.append(args[1])
        started.set()
        assert unblock.wait(10)
        return original(self, *args)

    monkeypatch.setattr(_CheckpointWriter, "_write_tensor", blocked)
    try:
        first = owner.stage_weight(_full_inputs())
        second = follower.stage_weight(_full_inputs())
        with owner._method.installation_context(owner._active):
            consume(first)
        assert started.wait(5)
        with follower._method.installation_context(follower._active):
            consume(second, cache=False)
        owner.release_staged_weight(first)
        follower.release_staged_weight(second)
        storage.objects.update(
            _artifact(
                torch.tensor([7.0, 8.0]).view(torch.uint8).numpy(),
                torch.tensor([9.0, 10.0]).view(torch.uint8).numpy(),
                version="delta-b",
                version_label=3,
                base_version="full-a",
            )
        )
        del storage.objects["s3://weights/test/v2/model-00001-of-00001.safetensors"]
        with ThreadPoolExecutor() as pool:
            entered = threading.Event()

            def replay():
                entered.set()
                return follower.stage_chain(
                    [
                        _full_inputs(),
                        _inputs(
                            None,
                            base_version="full-a",
                            version="delta-b",
                            version_label=3,
                        ),
                    ]
                )

            pending = pool.submit(replay)
            assert entered.wait(5)
            assert not pending.done()
            unblock.set()
            delta = pending.result(timeout=10)
        assert torch.equal(
            load_file(delta.path / "model-00001-of-00001.safetensors")["weight"],
            torch.tensor([9.0, 10.0]),
        )
        assert writes == ["weight"]
        assert (
            "s3://weights/test/v2/model-00001-of-00001.safetensors" not in storage.calls
        )
    finally:
        unblock.set()
        owner.close()
        follower.close()


def _head(store):
    state = store.state()
    return state.version, state.status, store.active_version()


def test_streamed_full_becomes_cache_head_so_old_chain_is_evictable(
    monkeypatch, tmp_path
):
    from modelexpress_rl.inference.checkpoint_store import (
        CheckpointState,
        checkpoint_files_state,
    )
    from modelexpress_rl.utils import index_checkpoint_tensors

    adapter, _ = build(monkeypatch, tmp_path)
    store = adapter._checkpoint.store
    try:
        assert _head(store)[0] == _head(store)[2] == "base-a"
        full = adapter.stage_weight(_full_inputs())
        with adapter._method.installation_context(adapter._active):
            consume(full)
        full.streaming.wait_cache()
        assert _head(store) == ("full-a", CheckpointState.READY, "full-a")
        paths, _, _ = index_checkpoint_tensors(full.path)
        assert store.state().files == checkpoint_files_state(paths)
        assert store.full_path("base-a").exists()
        # The next replay protects its base plus the active head. The old head
        # must now be evictable instead of exhausting the quota.
        protected = receiver._protected_versions(store, "full-a", "delta-b")
        assert "base-a" not in protected
        store.max_size_bytes = store.cache_size_bytes()
        store.ensure_capacity(
            store._payload_size_bytes(store.full_path("base-a")),
            protected_versions=protected,
        )
        assert not store.full_path("base-a").exists()
        assert full.path.exists()
        adapter.release_staged_weight(full)
    finally:
        adapter.close()


def test_failed_cache_keeps_previous_cache_head(monkeypatch, tmp_path):
    from modelexpress_rl.inference.streaming_checkpoint import _CheckpointWriter

    adapter, _ = build(monkeypatch, tmp_path)
    store = adapter._checkpoint.store
    before = _head(store)

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(_CheckpointWriter, "_write_tensor", fail)
    try:
        full = adapter.stage_weight(_full_inputs())
        with adapter._method.installation_context(adapter._active):
            consume(full)
        with pytest.raises(OSError, match="disk unavailable"):
            full.streaming.wait_cache()
        assert _head(store) == before
    finally:
        adapter.close()


def test_deferred_activation_adopts_stream_only_on_activate(monkeypatch, tmp_path):
    from modelexpress_rl.inference.checkpoint_store import CheckpointState

    adapter, _ = build(monkeypatch, tmp_path)
    store = adapter._checkpoint.store
    before = _head(store)
    try:
        full = adapter.stage_weight(_full_inputs())
        with adapter._method.installation_context(adapter._active, activate=False):
            consume(full)
        full.streaming.wait_cache()
        assert full.path.exists()
        assert _head(store) == before
        adapter._method.activate(adapter._active)
        assert _head(store) == ("full-a", CheckpointState.READY, "full-a")
    finally:
        adapter.close()
