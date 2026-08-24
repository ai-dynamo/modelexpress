# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mooncake artifact protocol unit tests; no mooncake.store dependency."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from modelexpress import p2p_pb2
from modelexpress.metadata import mooncake_artifact_cache as mc
from modelexpress.metadata.artifact_transfer import ArtifactCacheRoot, TarredP2PArtifactTransfer


class _MemoryStore:
    def __init__(self):
        self.values: dict[str, bytes] = {}
        self.remove_results: list[int] = []

    def get_bytes(self, key, expected_size=None):
        return self.values.get(key)

    def put_bytes(self, key, data):
        self.values[key] = bytes(data)
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
    monkeypatch.setenv("MX_ARTIFACT_MOONCAKE_MANIFEST_BYTES", "65536")
    monkeypatch.setenv("MX_ARTIFACT_MOONCAKE_NAMESPACE", "test-mooncake")
    return store


def _identity(model_name="org/Test-Model"):
    return p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        model_name=model_name,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
    )


def _transfer(tmp_path, *, target_name="target", chunk_size=4):
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    (source / "one.bin").write_bytes(b"abcdefgh")
    nested = source / "nested"
    nested.mkdir(exist_ok=True)
    (nested / "two.bin").write_bytes(b"0123456789")
    return TarredP2PArtifactTransfer(
        name="triton_cache",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        roots=(ArtifactCacheRoot("cache", source, tmp_path / target_name),),
        bundle_root=tmp_path / f"bundle-{target_name}",
        chunk_size=chunk_size,
    )


def test_publish_and_fetch_round_trip_with_multiple_chunks(tmp_path, memory_store):
    source_transfer = _transfer(tmp_path, target_name="source-target")
    target_transfer = _transfer(tmp_path, target_name="installed")
    identity = _identity()
    bundle = source_transfer.prepare_source()

    mc.publish_to_mooncake(source_transfer, identity, bundle, node_rank=0, accelerator="cuda")
    header = mc.install_from_mooncake(target_transfer, identity, node_rank=0, accelerator="cuda")
    target_transfer.install(header)

    assert len(bundle.manifest.chunks) > 1
    assert (tmp_path / "installed" / "one.bin").read_bytes() == b"abcdefgh"
    assert (tmp_path / "installed" / "nested" / "two.bin").read_bytes() == b"0123456789"


def test_cache_key_is_stable_and_isolates_compatibility_dimensions(tmp_path, memory_store):
    transfer = _transfer(tmp_path)
    base = mc.compute_artifact_cache_key(transfer, _identity(), node_rank=0, accelerator="cuda")
    assert base == mc.compute_artifact_cache_key(transfer, _identity(), node_rank=0, accelerator="cuda")
    assert base != mc.compute_artifact_cache_key(transfer, _identity(), node_rank=1, accelerator="cuda")
    assert base != mc.compute_artifact_cache_key(transfer, _identity(), node_rank=0, accelerator="rocm")
    assert base != mc.compute_artifact_cache_key(transfer, _identity("other/model"), node_rank=0, accelerator="cuda")


def test_missing_manifest_is_a_cache_miss(tmp_path, memory_store):
    with pytest.raises(mc.MooncakeArtifactCacheMiss, match="miss"):
        mc.install_from_mooncake(_transfer(tmp_path), _identity(), node_rank=0, accelerator="cuda")


def test_chunk_corruption_fails_and_removes_staged_target(tmp_path, memory_store):
    source_transfer = _transfer(tmp_path, target_name="source-target")
    target_transfer = _transfer(tmp_path, target_name="installed")
    identity = _identity()
    mc.publish_to_mooncake(source_transfer, identity, source_transfer.prepare_source(), node_rank=0, accelerator="cuda")
    chunk_key = next(key for key in memory_store.values if "/chunk/" in key)
    memory_store.values[chunk_key] = b"x" * len(memory_store.values[chunk_key])

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        mc.install_from_mooncake(target_transfer, identity, node_rank=0, accelerator="cuda")
    assert not (tmp_path / "bundle-installed" / "artifact.tar").exists()


def test_publish_does_not_commit_manifest_when_a_chunk_put_fails(tmp_path, memory_store, monkeypatch):
    transfer = _transfer(tmp_path)
    identity = _identity()
    bundle = transfer.prepare_source()
    key = mc.compute_artifact_cache_key(transfer, identity, node_rank=0, accelerator="cuda")
    original_put = memory_store.put_bytes

    def fail_first_chunk(name, data):
        if "/chunk/" in name:
            return -99
        return original_put(name, data)

    monkeypatch.setattr(memory_store, "put_bytes", fail_first_chunk)
    with pytest.raises(mc.MooncakeArtifactCacheUnavailable, match="put chunk failed"):
        mc.publish_to_mooncake(transfer, identity, bundle, node_rank=0, accelerator="cuda")
    assert mc._manifest_key(key) not in memory_store.values


def test_remove_manifest_retries_lease_then_commits(tmp_path, memory_store, monkeypatch):
    monkeypatch.setenv("MX_ARTIFACT_MOONCAKE_DELETE_RETRIES", "1")
    monkeypatch.setenv("MX_ARTIFACT_MOONCAKE_DELETE_RETRY_DELAY_SECS", "0")
    transfer = _transfer(tmp_path)
    identity = _identity()
    key = mc.compute_artifact_cache_key(transfer, identity, node_rank=0, accelerator="cuda")
    memory_store.values[mc._manifest_key(key)] = b"old"
    memory_store.remove_results[:] = [-706, 0]

    mc.publish_to_mooncake(transfer, identity, transfer.prepare_source(), node_rank=0, accelerator="cuda")
    assert mc._manifest_key(key) in memory_store.values
