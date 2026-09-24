# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mooncake artifact protocol unit tests; no mooncake.store dependency."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
import torch

from modelexpress import p2p_pb2
from modelexpress.metadata import mooncake_artifact_store as mc
from modelexpress.metadata.artifact_transfer import (
    ArtifactCacheRoot,
    TarredArtifactTransfer,
)


class _MemoryStore:
    """In-memory model of Mooncake's immutable, first-writer-wins objects."""

    def __init__(self):
        self.values: dict[str, bytes] = {}
        self.remove_results: list[int] = []
        self.put_buffer_counts: dict[str, int] = {}
        self.get_buffer_sizes: dict[str, list[int]] = {}
        self.put_soft_pins: list[tuple[str, bool]] = []

    def get_size(self, key):
        value = self.values.get(key)
        return len(value) if value is not None else None

    def get_buffers(self, key, sizes):
        value = self.values.get(key)
        if value is None:
            return None
        if sum(sizes) != len(value):
            raise AssertionError(f"incorrect receive size for {key}")
        self.get_buffer_sizes[key] = list(sizes)
        buffers = []
        offset = 0
        for size in sizes:
            buffer = torch.empty(size, dtype=torch.uint8)
            memoryview(buffer.numpy()).cast("B")[:] = value[offset : offset + size]
            buffers.append(buffer)
            offset += size
        return tuple(buffers)

    def put_buffers(self, key, buffers, *, soft_pin):
        self.put_soft_pins.append((key, soft_pin))
        self.put_buffer_counts[key] = len(buffers)
        value = b"".join(mc._tensor_to_bytes(buffer) for buffer in buffers)
        self.values.setdefault(key, value)
        return 0

    def get_bytes(self, key, expected_size=None):
        value = self.values.get(key)
        if value is not None and expected_size is not None:
            assert len(value) == expected_size
        return value

    def put_bytes(self, key, data, *, soft_pin=True):
        self.put_soft_pins.append((key, soft_pin))
        self.values.setdefault(key, bytes(data))
        return 0

    def remove(self, key):
        result = self.remove_results.pop(0) if self.remove_results else 0
        if result in (0, -704):
            self.values.pop(key, None)
        return result


@pytest.fixture
def memory_store(monkeypatch):
    store = _MemoryStore()

    @contextmanager
    def session():
        yield store

    monkeypatch.setattr(mc, "_store_session", session)
    monkeypatch.setenv("MX_ARTIFACT_MOONCAKE_NAMESPACE", "test-mooncake")
    return store


def _identity(model_name="org/Test-Model"):
    return p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        model_name=model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
    )


def _transfer(tmp_path, *, target_name="target", chunk_size=4, source_name="source"):
    source = tmp_path / source_name
    source.mkdir(exist_ok=True)
    (source / "one.bin").write_bytes(b"abcdefgh")
    nested = source / "nested"
    nested.mkdir(exist_ok=True)
    (nested / "two.bin").write_bytes(b"0123456789")
    return TarredArtifactTransfer(
        name="triton_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        roots=(ArtifactCacheRoot("cache", source, tmp_path / target_name),),
        bundle_root=tmp_path / f"bundle-{target_name}",
        chunk_size=chunk_size,
    )


def _cache_key(transfer, identity):
    return mc.compute_artifact_cache_key(
        transfer,
        identity,
        node_rank=0,
        accelerator="cuda",
    )


def _publish(transfer, identity, bundle=None, *, stale_fingerprint=None):
    return mc.publish_to_mooncake(
        transfer,
        identity,
        bundle or transfer.prepare_source(),
        node_rank=0,
        accelerator="cuda",
        repair_from_fingerprint=stale_fingerprint,
    )


def _fetch(transfer, identity):
    return mc.install_from_mooncake(
        transfer,
        identity,
        node_rank=0,
        accelerator="cuda",
    )


def _corrupt_payload(store, cache_key, value=b"x"):
    data = bytearray(store.values[cache_key])
    offset = mc._ARTIFACT_ENVELOPE_BYTES
    data[offset : offset + len(value)] = value
    store.values[cache_key] = bytes(data)


def test_native_config_is_logged_only_when_store_is_initialized(monkeypatch, caplog):
    config = mc._MooncakeNativeConfig(
        local_hostname="127.0.0.1",
        metadata_server="metadata",
        master_server="master",
        protocol="rdma",
        device_name="mlx5_0",
        global_segment_size=0,
        local_buffer_size=1024,
    )
    native_store = MagicMock()
    native_store_type = MagicMock(return_value=native_store)
    monkeypatch.setattr(mc, "_shared_store", None)
    monkeypatch.setattr(mc, "_shared_store_config", None)
    monkeypatch.setattr(mc, "_store_atexit_registered", True)
    monkeypatch.setattr(mc, "_mooncake_native_config", MagicMock(return_value=config))
    monkeypatch.setattr(mc, "_MooncakeNativeStore", native_store_type)

    with caplog.at_level(logging.INFO, logger=mc.logger.name):
        assert mc._new_store() is native_store
        assert mc._new_store() is native_store

    native_store_type.assert_called_once_with(config)
    assert caplog.text.count("[MX][Mooncake] native store initialized") == 1


def test_publish_and_fetch_round_trip_uses_one_multibuffer_object(
    tmp_path, memory_store
):
    source_transfer = _transfer(tmp_path, target_name="source-target")
    target_transfer = _transfer(tmp_path, target_name="installed")
    identity = _identity()
    bundle = source_transfer.prepare_source()

    cache_key = _publish(source_transfer, identity, bundle)
    header = _fetch(target_transfer, identity)
    target_transfer.install(header)

    assert len(bundle.manifest.chunks) > 1
    assert set(memory_store.values) == {cache_key}
    assert memory_store.put_buffer_counts[cache_key] == 1 + len(
        bundle.manifest.chunks
    )
    # The receive layout deliberately differs from the publishing layout.
    assert len(memory_store.get_buffer_sizes[cache_key]) == 2
    assert (tmp_path / "installed" / "one.bin").read_bytes() == b"abcdefgh"
    assert (tmp_path / "installed" / "nested" / "two.bin").read_bytes() == (
        b"0123456789"
    )


def test_cache_key_is_stable_and_isolates_compatibility_dimensions(
    tmp_path, memory_store
):
    transfer = _transfer(tmp_path)
    base = _cache_key(transfer, _identity())
    assert base == _cache_key(transfer, _identity())
    assert base != mc.compute_artifact_cache_key(
        transfer, _identity(), node_rank=1, accelerator="cuda"
    )
    assert base != mc.compute_artifact_cache_key(
        transfer, _identity(), node_rank=0, accelerator="rocm"
    )
    assert base != _cache_key(transfer, _identity("other/model"))


def test_cache_key_is_stable_when_map_entries_are_inserted_differently(
    tmp_path, memory_store
):
    transfer = _transfer(tmp_path)
    first = _identity()
    first.extra_parameters["z-key"] = "z-value"
    first.extra_parameters["a-key"] = "a-value"
    second = _identity()
    second.extra_parameters["a-key"] = "a-value"
    second.extra_parameters["z-key"] = "z-value"

    assert _cache_key(transfer, first) == _cache_key(transfer, second)


def test_cache_key_uses_case_insensitive_compatibility_identity(
    tmp_path, memory_store
):
    transfer = _transfer(tmp_path)
    upper = _identity("ORG/TEST-MODEL")
    upper.revision = "MAIN"
    upper.gpu_arch = "SM90"
    upper.backend_framework_version = "VLLM-0.10.0"
    upper.extra_parameters["CACHE_BACKEND"] = "CUDA"
    lower = _identity("org/test-model")
    lower.revision = "main"
    lower.gpu_arch = "sm90"
    lower.backend_framework_version = "vllm-0.10.0"
    lower.extra_parameters["cache_backend"] = "cuda"

    assert _cache_key(transfer, upper) == _cache_key(transfer, lower)


def test_missing_object_is_logged_only_below_info(tmp_path, memory_store, caplog):
    with caplog.at_level(logging.INFO, logger=mc.logger.name):
        with pytest.raises(mc.MooncakeArtifactCacheMiss, match="miss"):
            _fetch(_transfer(tmp_path), _identity())

    assert not caplog.records


def _truncate_object(store, cache_key):
    store.values[cache_key] = store.values[cache_key][:-1]


def _corrupt_checksum(store, cache_key):
    _corrupt_payload(store, cache_key)


def _corrupt_envelope(store, cache_key):
    data = bytearray(store.values[cache_key])
    data[: len(mc._ARTIFACT_ENVELOPE_MAGIC)] = b"BROKEN!!"
    store.values[cache_key] = bytes(data)


def _corrupt_manifest_payload(store, cache_key):
    object_bytes = store.values[cache_key]
    limit = mc._ARTIFACT_ENVELOPE_BYTES
    envelope = mc._decode_artifact_envelope(object_bytes[:limit])
    frame = mc._encode_artifact_envelope(
        artifact_id=envelope.artifact_id,
        manifest_bytes=b"\x80",
    )
    store.values[cache_key] = frame + object_bytes[limit:]


def _corrupt_artifact_id(store, cache_key):
    object_bytes = store.values[cache_key]
    limit = mc._ARTIFACT_ENVELOPE_BYTES
    envelope = mc._decode_artifact_envelope(object_bytes[:limit])
    frame = mc._encode_artifact_envelope(
        artifact_id="incorrect-artifact-id",
        manifest_bytes=envelope.manifest_bytes,
    )
    store.values[cache_key] = frame + object_bytes[limit:]


@pytest.mark.parametrize(
    ("corrupt", "reason"),
    [
        pytest.param(_truncate_object, "object size mismatch", id="truncated-object"),
        pytest.param(_corrupt_checksum, "checksum mismatch", id="checksum-mismatch"),
        pytest.param(_corrupt_envelope, "magic mismatch", id="invalid-envelope"),
        pytest.param(
            _corrupt_manifest_payload,
            "Error parsing message",
            id="invalid-manifest-protobuf",
        ),
        pytest.param(_corrupt_artifact_id, "artifact_id mismatch", id="artifact-id"),
    ],
)
def test_stale_artifact_is_not_deleted_by_reader_and_cleans_staging(
    tmp_path, memory_store, caplog, corrupt, reason
):
    source_transfer = _transfer(tmp_path, target_name="source-target")
    target_transfer = _transfer(tmp_path, target_name="installed")
    identity = _identity()
    cache_key = _publish(source_transfer, identity)
    corrupt(memory_store, cache_key)
    stale_bytes = memory_store.values[cache_key]

    with caplog.at_level(logging.WARNING, logger=mc.logger.name):
        with pytest.raises(mc.MooncakeArtifactCacheStale, match=reason) as raised:
            _fetch(target_transfer, identity)

    assert raised.value.fingerprint
    assert memory_store.values[cache_key] == stale_bytes
    assert not (tmp_path / "bundle-installed" / "artifact.tar").exists()
    assert not caplog.records


def test_stale_artifact_owner_repairs_same_object(tmp_path, memory_store):
    source_transfer = _transfer(tmp_path, target_name="source-target")
    target_transfer = _transfer(tmp_path, target_name="installed")
    identity = _identity()
    cache_key = _publish(source_transfer, identity)
    _corrupt_checksum(memory_store, cache_key)

    with pytest.raises(mc.MooncakeArtifactCacheStale) as raised:
        _fetch(target_transfer, identity)
    _publish(
        source_transfer,
        identity,
        stale_fingerprint=raised.value.fingerprint,
    )
    header = _fetch(target_transfer, identity)
    target_transfer.install(header)

    assert mc._repair_key(cache_key) not in memory_store.values
    assert (tmp_path / "installed" / "one.bin").read_bytes() == b"abcdefgh"


def test_repair_does_not_delete_a_changed_corrupt_object(tmp_path, memory_store):
    transfer = _transfer(tmp_path)
    identity = _identity()
    cache_key = _publish(transfer, identity)
    _corrupt_payload(memory_store, cache_key, b"x")
    with pytest.raises(mc.MooncakeArtifactCacheStale) as raised:
        _fetch(transfer, identity)

    _corrupt_payload(memory_store, cache_key, b"y")
    changed = memory_store.values[cache_key]
    _publish(transfer, identity, stale_fingerprint=raised.value.fingerprint)

    assert memory_store.values[cache_key] == changed
    assert mc._repair_key(cache_key) not in memory_store.values


def test_repair_does_not_replace_an_object_repaired_by_another_pod(
    tmp_path, memory_store
):
    transfer = _transfer(tmp_path)
    identity = _identity()
    cache_key = _publish(transfer, identity)
    _corrupt_checksum(memory_store, cache_key)
    with pytest.raises(mc.MooncakeArtifactCacheStale) as raised:
        _fetch(transfer, identity)

    memory_store.values.pop(cache_key)
    _publish(transfer, identity)
    repaired = memory_store.values[cache_key]
    _publish(transfer, identity, stale_fingerprint=raised.value.fingerprint)

    assert memory_store.values[cache_key] == repaired


def test_repair_loser_does_not_modify_objects_or_remove_owner_lock(
    tmp_path, memory_store
):
    transfer = _transfer(tmp_path)
    identity = _identity()
    cache_key = _publish(transfer, identity)
    _corrupt_checksum(memory_store, cache_key)
    with pytest.raises(mc.MooncakeArtifactCacheStale) as raised:
        _fetch(transfer, identity)
    stale = memory_store.values[cache_key]
    other_lock = mc._encode_repair_lock("other-owner")
    memory_store.values[mc._repair_key(cache_key)] = other_lock

    _publish(transfer, identity, stale_fingerprint=raised.value.fingerprint)

    assert memory_store.values[cache_key] == stale
    assert memory_store.values[mc._repair_key(cache_key)] == other_lock


def test_explicit_object_already_exists_is_repair_lock_contention(memory_store):
    repair_key = "artifact/repair"
    other_lock = mc._encode_repair_lock("other-owner")
    memory_store.values[repair_key] = other_lock
    original_put_bytes = memory_store.put_bytes

    def put_bytes(key, data, *, soft_pin=True):
        if key == repair_key and key in memory_store.values:
            return -705
        return original_put_bytes(key, data, soft_pin=soft_pin)

    memory_store.put_bytes = put_bytes
    token = "competing-owner"

    assert not mc._acquire_repair_lock(
        memory_store,
        repair_key,
        mc._encode_repair_lock(token),
        token=token,
        transfer_name="triton_cache",
        cache_key="artifact",
    )
    assert memory_store.values[repair_key] == other_lock


def test_repair_does_not_reclaim_live_marker(tmp_path, memory_store, monkeypatch):
    transfer = _transfer(tmp_path)
    identity = _identity()
    cache_key = _publish(transfer, identity)
    _corrupt_checksum(memory_store, cache_key)
    with pytest.raises(mc.MooncakeArtifactCacheStale) as raised:
        _fetch(transfer, identity)
    stale = memory_store.values[cache_key]

    created_at = 1000.0
    monkeypatch.setattr(mc.time, "time", lambda: created_at)
    other_lock = mc._encode_repair_lock("live-owner")
    memory_store.values[mc._repair_key(cache_key)] = other_lock
    monkeypatch.setattr(
        mc.time,
        "time",
        lambda: created_at + mc._REPAIR_LOCK_STALE_SECONDS - 1,
    )

    _publish(transfer, identity, stale_fingerprint=raised.value.fingerprint)

    assert memory_store.values[cache_key] == stale
    assert memory_store.values[mc._repair_key(cache_key)] == other_lock


def test_repair_reclaims_marker_older_than_ten_minutes(
    tmp_path, memory_store, monkeypatch
):
    transfer = _transfer(tmp_path)
    identity = _identity()
    cache_key = _publish(transfer, identity)
    _corrupt_checksum(memory_store, cache_key)
    with pytest.raises(mc.MooncakeArtifactCacheStale) as raised:
        _fetch(transfer, identity)

    created_at = 1000.0
    monkeypatch.setattr(mc.time, "time", lambda: created_at)
    memory_store.values[mc._repair_key(cache_key)] = mc._encode_repair_lock(
        "dead-owner"
    )
    monkeypatch.setattr(
        mc.time,
        "time",
        lambda: created_at + mc._REPAIR_LOCK_STALE_SECONDS + 1,
    )

    _publish(transfer, identity, stale_fingerprint=raised.value.fingerprint)
    header = _fetch(transfer, identity)
    transfer.install(header)

    assert mc._repair_key(cache_key) not in memory_store.values
    assert (mc._repair_key(cache_key), True) in memory_store.put_soft_pins
    assert (tmp_path / "target" / "one.bin").read_bytes() == b"abcdefgh"


def test_repair_does_not_reclaim_marker_that_changes_during_confirmation(
    tmp_path, memory_store, monkeypatch
):
    transfer = _transfer(tmp_path)
    identity = _identity()
    cache_key = _publish(transfer, identity)
    _corrupt_checksum(memory_store, cache_key)
    with pytest.raises(mc.MooncakeArtifactCacheStale) as raised:
        _fetch(transfer, identity)
    stale = memory_store.values[cache_key]

    created_at = 1000.0
    monkeypatch.setattr(mc.time, "time", lambda: created_at)
    repair_key = mc._repair_key(cache_key)
    memory_store.values[repair_key] = mc._encode_repair_lock("dead-owner")
    replacement_lock = mc._encode_repair_lock("replacement-owner")
    monkeypatch.setattr(
        mc.time,
        "time",
        lambda: created_at + mc._REPAIR_LOCK_STALE_SECONDS + 1,
    )
    original_get_bytes = memory_store.get_bytes
    repair_reads = 0

    def replace_marker_on_confirmation(key, expected_size=None):
        nonlocal repair_reads
        if key == repair_key:
            repair_reads += 1
            if repair_reads == 2:
                memory_store.values[repair_key] = replacement_lock
        return original_get_bytes(key, expected_size)

    monkeypatch.setattr(memory_store, "get_bytes", replace_marker_on_confirmation)

    _publish(transfer, identity, stale_fingerprint=raised.value.fingerprint)

    assert memory_store.values[cache_key] == stale
    assert memory_store.values[repair_key] == replacement_lock


def test_repair_does_not_reclaim_invalid_marker(tmp_path, memory_store):
    transfer = _transfer(tmp_path)
    identity = _identity()
    cache_key = _publish(transfer, identity)
    _corrupt_checksum(memory_store, cache_key)
    with pytest.raises(mc.MooncakeArtifactCacheStale) as raised:
        _fetch(transfer, identity)
    stale = memory_store.values[cache_key]
    payload = b"[]"
    invalid_lock = mc._REPAIR_LOCK_HEADER.pack(
        mc._REPAIR_LOCK_MAGIC,
        len(payload),
    ) + payload
    invalid_lock += bytes(mc._REPAIR_LOCK_BYTES - len(invalid_lock))
    repair_key = mc._repair_key(cache_key)
    memory_store.values[repair_key] = invalid_lock

    _publish(transfer, identity, stale_fingerprint=raised.value.fingerprint)

    assert memory_store.values[cache_key] == stale
    assert memory_store.values[repair_key] == invalid_lock


def test_old_repair_owner_does_not_release_new_owner_marker(memory_store):
    repair_key = "artifact/repair"
    new_lock = mc._encode_repair_lock("new-owner")
    memory_store.values[repair_key] = new_lock

    mc._release_repair_lock(memory_store, repair_key, "old-owner")

    assert memory_store.values[repair_key] == new_lock


def test_normal_concurrent_publish_is_first_writer_wins(tmp_path, memory_store):
    first = _transfer(tmp_path, target_name="first", source_name="first-source")
    second = _transfer(tmp_path, target_name="second", source_name="second-source")
    (tmp_path / "second-source" / "one.bin").write_bytes(b"different")
    identity = _identity()
    first_bundle = first.prepare_source()
    second_bundle = second.prepare_source()

    cache_key = _publish(first, identity, first_bundle)
    first_object = memory_store.values[cache_key]
    _publish(second, identity, second_bundle)

    assert first_bundle.artifact_id != second_bundle.artifact_id
    assert memory_store.values[cache_key] == first_object


def test_remove_object_retries_while_object_has_lease(memory_store, monkeypatch):
    monkeypatch.setenv("MX_ARTIFACT_MOONCAKE_DELETE_RETRIES", "1")
    monkeypatch.setenv("MX_ARTIFACT_MOONCAKE_DELETE_RETRY_DELAY_SECS", "0")
    memory_store.values["key"] = b"value"
    memory_store.remove_results[:] = [-706, 0]

    mc._remove_object_with_retry(memory_store, "key")

    assert "key" not in memory_store.values


class _NativeBufferStore:
    def __init__(self):
        self.registered = []
        self.unregistered = []

    def register_buffer(self, pointer, size):
        self.registered.append((pointer, size))
        return 0

    def unregister_buffer(self, pointer):
        self.unregistered.append(pointer)
        return 0

    def batch_put_from_multi_buffers(self, keys, pointers, sizes, config):
        del keys, pointers, sizes, config
        return [0]

    def batch_get_into_multi_buffers(self, keys, pointers, sizes):
        del keys, pointers
        return [sum(sizes[0])]


def _native_wrapper(native):
    wrapper = object.__new__(mc._MooncakeNativeStore)
    wrapper._store = native
    wrapper._artifact_replicate_config = object()
    wrapper._repair_replicate_config = object()
    return wrapper


def test_native_multibuffer_calls_register_and_unregister_every_buffer():
    native = _NativeBufferStore()
    wrapper = _native_wrapper(native)
    buffers = (mc._bytes_to_tensor(b"abc"), mc._bytes_to_tensor(b"defgh"))

    assert wrapper.put_buffers("key", buffers, soft_pin=True) == 0
    fetched = wrapper.get_buffers("key", [3, 5])

    assert fetched is not None
    assert [size for _, size in native.registered] == [3, 5, 3, 5]
    assert native.unregistered == [
        native.registered[1][0],
        native.registered[0][0],
        native.registered[3][0],
        native.registered[2][0],
    ]


def test_native_multibuffer_unregisters_prior_buffers_when_registration_fails():
    native = _NativeBufferStore()
    original_register = native.register_buffer

    def fail_second_registration(pointer, size):
        if native.registered:
            return -99
        return original_register(pointer, size)

    native.register_buffer = fail_second_registration
    wrapper = _native_wrapper(native)
    buffers = (mc._bytes_to_tensor(b"abc"), mc._bytes_to_tensor(b"def"))

    with pytest.raises(mc.MooncakeArtifactCacheUnavailable, match="returned -99"):
        wrapper.put_buffers("key", buffers, soft_pin=True)

    assert native.unregistered == [native.registered[0][0]]
