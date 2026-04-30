# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PeerModelServiceServicer and the WorkerGrpcServer integration."""

from __future__ import annotations

import os
from pathlib import Path

import grpc
import pytest

from modelexpress import model_pb2, model_pb2_grpc
from modelexpress.peer_model_service import (
    PeerModelServiceServicer,
    _resolve_hf_model_path,
    _is_safe_relative,
)
from modelexpress.worker_server import WorkerGrpcServer


def _make_hf_model(
    cache_root: Path,
    model_id: str,
    commit: str,
    files: dict[str, bytes],
) -> Path:
    """Create a minimal HuggingFace cache layout for tests.

    Creates ``cache_root/models--<org>--<name>/snapshots/<commit>/<files>``
    with regular files (not symlinks) of the given byte contents.
    """
    folder = "models--" + model_id.replace("/", "--")
    snap = cache_root / folder / "snapshots" / commit
    snap.mkdir(parents=True)
    for rel, data in files.items():
        full = snap / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data)
    return snap


@pytest.fixture
def hf_cache(tmp_path):
    """A tmp HF cache root with one model containing two files."""
    cache_root = tmp_path / "hf-hub"
    cache_root.mkdir()
    _make_hf_model(
        cache_root,
        "google/t5-small",
        "abc123",
        {
            "config.json": b'{"model_type":"t5"}',
            "model.safetensors": b"\x00" * 4096 + b"\xab" * 1024,
        },
    )
    return cache_root


# ---------------------------------------------------------------------------
# Unit tests for the path helpers
# ---------------------------------------------------------------------------


def test_resolve_hf_model_path_with_commit(hf_cache):
    p = _resolve_hf_model_path(hf_cache, "google/t5-small", "abc123")
    assert p.is_dir()
    assert p.name == "abc123"


def test_resolve_hf_model_path_without_commit_picks_latest(hf_cache):
    # Add a second snapshot, mtime ahead of the first
    folder = hf_cache / "models--google--t5-small" / "snapshots" / "def456"
    folder.mkdir()
    (folder / "config.json").write_bytes(b"{}")
    os.utime(folder, (1_700_000_000, 1_700_000_000))
    older = hf_cache / "models--google--t5-small" / "snapshots" / "abc123"
    os.utime(older, (1_500_000_000, 1_500_000_000))
    p = _resolve_hf_model_path(hf_cache, "google/t5-small", None)
    assert p.name == "def456"


def test_resolve_hf_model_path_missing_commit_raises(hf_cache):
    with pytest.raises(FileNotFoundError):
        _resolve_hf_model_path(hf_cache, "google/t5-small", "nope")


def test_resolve_hf_model_path_missing_model_raises(hf_cache):
    with pytest.raises(FileNotFoundError):
        _resolve_hf_model_path(hf_cache, "no/such-model", None)


def test_is_safe_relative_accepts_in_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a.txt").write_bytes(b"x")
    assert _is_safe_relative(root, "a.txt")


def test_is_safe_relative_rejects_dotdot(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    assert not _is_safe_relative(root, "../leak")


def test_is_safe_relative_rejects_symlink_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"secret")
    (root / "link").symlink_to(outside)
    assert not _is_safe_relative(root, "link")


# ---------------------------------------------------------------------------
# End-to-end: real gRPC server + stub against a tmp HF cache
# ---------------------------------------------------------------------------


@pytest.fixture
def running_server(hf_cache):
    server = WorkerGrpcServer(
        tensor_protos=[],
        mx_source_id="test-source-id",
        port=0,
        cache_root=hf_cache,
    )
    server.start()
    yield server
    server.stop(grace=0)


@pytest.fixture
def model_stub(running_server):
    channel = grpc.insecure_channel(f"127.0.0.1:{running_server.port}")
    stub = model_pb2_grpc.ModelServiceStub(channel)
    yield stub
    channel.close()


def test_ensure_model_downloaded_hit(model_stub):
    request = model_pb2.ModelDownloadRequest(
        model_name="google/t5-small",
        provider=model_pb2.HUGGING_FACE,
        ignore_weights=False,
    )
    updates = list(model_stub.EnsureModelDownloaded(request))
    assert len(updates) == 1
    assert updates[0].status == model_pb2.DOWNLOADED


def test_ensure_model_downloaded_miss(model_stub):
    request = model_pb2.ModelDownloadRequest(
        model_name="no/such-model",
        provider=model_pb2.HUGGING_FACE,
        ignore_weights=False,
    )
    with pytest.raises(grpc.RpcError) as excinfo:
        list(model_stub.EnsureModelDownloaded(request))
    assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND


def test_list_model_files(model_stub):
    request = model_pb2.ModelFilesRequest(
        model_name="google/t5-small",
        provider=model_pb2.HUGGING_FACE,
        chunk_size=0,
    )
    response = model_stub.ListModelFiles(request)
    paths = sorted(f.relative_path for f in response.files)
    assert paths == ["config.json", "model.safetensors"]
    sizes = {f.relative_path: f.size for f in response.files}
    assert sizes["config.json"] == 19
    assert sizes["model.safetensors"] == 5120
    assert response.total_size == 19 + 5120


def test_stream_model_files_reconstructs_payload(model_stub):
    import blake3 as _blake3

    request = model_pb2.ModelFilesRequest(
        model_name="google/t5-small",
        provider=model_pb2.HUGGING_FACE,
        chunk_size=1024,
    )
    accumulated: dict[str, bytearray] = {}
    saw_final = False
    first_commit = None
    blake3_announced: dict[str, str] = {}
    for chunk in model_stub.StreamModelFiles(request):
        if first_commit is None and chunk.commit_hash:
            first_commit = chunk.commit_hash
        buf = accumulated.setdefault(chunk.relative_path, bytearray())
        assert chunk.offset == len(buf)
        buf.extend(chunk.data)
        if chunk.is_last_chunk:
            assert chunk.HasField("blake3"), (
                f"server should populate blake3 on final chunk of {chunk.relative_path!r}"
            )
            blake3_announced[chunk.relative_path] = chunk.blake3
        else:
            assert not chunk.HasField("blake3"), (
                f"server populated blake3 mid-stream for {chunk.relative_path!r}"
            )
        if chunk.is_last_file and chunk.is_last_chunk:
            saw_final = True
    assert saw_final
    assert first_commit == "abc123"
    assert bytes(accumulated["config.json"]) == b'{"model_type":"t5"}'
    assert (
        bytes(accumulated["model.safetensors"])
        == b"\x00" * 4096 + b"\xab" * 1024
    )
    # Independently re-hash the bytes we received and compare to the
    # server-announced hex.
    for path, payload in accumulated.items():
        h = _blake3.blake3()
        h.update(bytes(payload))
        assert blake3_announced[path] == h.hexdigest(), (
            f"blake3 mismatch for {path!r}"
        )


def test_stream_model_files_blake3_zero_byte_file(hf_cache, model_stub):
    folder = hf_cache / "models--google--t5-small" / "snapshots" / "abc123"
    (folder / "empty.txt").write_bytes(b"")
    request = model_pb2.ModelFilesRequest(
        model_name="google/t5-small",
        provider=model_pb2.HUGGING_FACE,
        chunk_size=1024,
    )
    chunks_by_path: dict[str, list] = {}
    for chunk in model_stub.StreamModelFiles(request):
        chunks_by_path.setdefault(chunk.relative_path, []).append(chunk)
    empty = chunks_by_path["empty.txt"]
    assert len(empty) == 1
    assert empty[0].is_last_chunk is True
    assert empty[0].HasField("blake3")
    # blake3 of zero bytes is well-known.
    assert empty[0].blake3 == (
        "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    )


def test_stream_model_files_unknown_provider_unimplemented(model_stub):
    request = model_pb2.ModelFilesRequest(
        model_name="any",
        provider=model_pb2.NGC,
        chunk_size=1024,
    )
    with pytest.raises(grpc.RpcError) as excinfo:
        list(model_stub.StreamModelFiles(request))
    assert excinfo.value.code() == grpc.StatusCode.UNIMPLEMENTED


def test_servicer_init_uses_explicit_cache_root(tmp_path):
    cache = tmp_path / "explicit"
    cache.mkdir()
    s = PeerModelServiceServicer(cache_root=cache)
    # Internal attr access in tests is fine
    assert s._cache_root == cache


# ---------------------------------------------------------------------------
# relative_paths filter (P1)
# ---------------------------------------------------------------------------


def test_list_model_files_with_filter(model_stub):
    request = model_pb2.ModelFilesRequest(
        model_name="google/t5-small",
        provider=model_pb2.HUGGING_FACE,
        chunk_size=0,
        relative_paths=["model.safetensors"],
    )
    response = model_stub.ListModelFiles(request)
    paths = [f.relative_path for f in response.files]
    assert paths == ["model.safetensors"]
    sizes = {f.relative_path: f.size for f in response.files}
    assert sizes["model.safetensors"] == 5120
    assert response.total_size == 5120


def test_list_model_files_filter_unknown_path_returns_not_found(model_stub):
    request = model_pb2.ModelFilesRequest(
        model_name="google/t5-small",
        provider=model_pb2.HUGGING_FACE,
        chunk_size=0,
        relative_paths=["does-not-exist.bin"],
    )
    with pytest.raises(grpc.RpcError) as excinfo:
        model_stub.ListModelFiles(request)
    assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND


def test_stream_model_files_filter_emits_in_request_order(model_stub):
    request = model_pb2.ModelFilesRequest(
        model_name="google/t5-small",
        provider=model_pb2.HUGGING_FACE,
        chunk_size=4096,
        relative_paths=["model.safetensors", "config.json"],
    )
    seen_paths: list[str] = []
    accumulated: dict[str, bytearray] = {}
    for chunk in model_stub.StreamModelFiles(request):
        if chunk.relative_path not in seen_paths:
            seen_paths.append(chunk.relative_path)
        accumulated.setdefault(chunk.relative_path, bytearray()).extend(chunk.data)
    # Order matches request, not directory walk.
    assert seen_paths == ["model.safetensors", "config.json"]
    assert bytes(accumulated["config.json"]) == b'{"model_type":"t5"}'
    assert (
        bytes(accumulated["model.safetensors"])
        == b"\x00" * 4096 + b"\xab" * 1024
    )


def test_stream_model_files_filter_path_traversal_rejected(model_stub):
    request = model_pb2.ModelFilesRequest(
        model_name="google/t5-small",
        provider=model_pb2.HUGGING_FACE,
        chunk_size=4096,
        relative_paths=["../../../etc/passwd"],
    )
    with pytest.raises(grpc.RpcError) as excinfo:
        list(model_stub.StreamModelFiles(request))
    assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND


# ---------------------------------------------------------------------------
# LOCAL provider (P3)
# ---------------------------------------------------------------------------


@pytest.fixture
def local_mount_payload(tmp_path):
    """A throwaway directory of metadata-shaped files to expose via LOCAL."""
    payload = tmp_path / "metadata-bundle"
    payload.mkdir()
    (payload / "tokenizer.json").write_bytes(b"TOKEN-DATA")
    (payload / "config.json").write_bytes(b'{"k":"v"}')
    return payload


@pytest.fixture
def local_running_server(hf_cache, local_mount_payload):
    server = WorkerGrpcServer(
        tensor_protos=[],
        mx_source_id="test-source-id",
        port=0,
        cache_root=hf_cache,
    )
    # The WorkerGrpcServer does not currently accept local_mounts directly;
    # we patch the servicer it constructs by wiring a custom one in the test.
    # Use the lower-level grpc.server path for this test instead.
    import grpc
    from concurrent import futures
    from modelexpress.peer_model_service import PeerModelServiceServicer
    from modelexpress import model_pb2_grpc

    raw_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = PeerModelServiceServicer(
        cache_root=hf_cache,
        local_mounts={"metadata-bundle": local_mount_payload},
    )
    model_pb2_grpc.add_ModelServiceServicer_to_server(servicer, raw_server)
    port = raw_server.add_insecure_port("[::]:0")
    raw_server.start()
    yield port
    raw_server.stop(grace=0)
    # Suppress the worker-server fixture: we did not use it here.
    server.stop(grace=0)


@pytest.fixture
def local_stub(local_running_server):
    channel = grpc.insecure_channel(f"127.0.0.1:{local_running_server}")
    stub = model_pb2_grpc.ModelServiceStub(channel)
    yield stub
    channel.close()


def test_stream_model_files_local_serves_mount(local_stub):
    request = model_pb2.ModelFilesRequest(
        model_name="metadata-bundle",
        provider=model_pb2.LOCAL,
        chunk_size=1024,
    )
    accumulated: dict[str, bytearray] = {}
    for chunk in local_stub.StreamModelFiles(request):
        accumulated.setdefault(chunk.relative_path, bytearray()).extend(chunk.data)
        # LOCAL must not emit commit_hash.
        assert not chunk.HasField("commit_hash")
    assert bytes(accumulated["tokenizer.json"]) == b"TOKEN-DATA"
    assert bytes(accumulated["config.json"]) == b'{"k":"v"}'


def test_stream_model_files_local_unknown_mount_not_found(local_stub):
    request = model_pb2.ModelFilesRequest(
        model_name="not-registered",
        provider=model_pb2.LOCAL,
        chunk_size=1024,
    )
    with pytest.raises(grpc.RpcError) as excinfo:
        list(local_stub.StreamModelFiles(request))
    assert excinfo.value.code() == grpc.StatusCode.NOT_FOUND


def test_list_model_files_local_serves_mount(local_stub):
    request = model_pb2.ModelFilesRequest(
        model_name="metadata-bundle",
        provider=model_pb2.LOCAL,
        chunk_size=0,
    )
    response = local_stub.ListModelFiles(request)
    paths = sorted(f.relative_path for f in response.files)
    assert paths == ["config.json", "tokenizer.json"]


def test_local_mount_with_filter(local_stub):
    request = model_pb2.ModelFilesRequest(
        model_name="metadata-bundle",
        provider=model_pb2.LOCAL,
        chunk_size=1024,
        relative_paths=["tokenizer.json"],
    )
    seen = []
    for chunk in local_stub.StreamModelFiles(request):
        if chunk.relative_path not in seen:
            seen.append(chunk.relative_path)
    assert seen == ["tokenizer.json"]
