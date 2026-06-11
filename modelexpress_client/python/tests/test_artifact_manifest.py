# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for artifact manifest helpers and worker RPC serving."""

from __future__ import annotations

import time
from concurrent import futures

import grpc
import pytest

from modelexpress import p2p_pb2, p2p_pb2_grpc
from modelexpress.metadata.artifact_manifest import (
    MAX_ARTIFACT_TRANSFER_CHUNK_SIZE,
    MX_ARTIFACT_TRANSFER_CHUNK_SIZE_ENV,
    artifact_manifest_id,
    artifact_transfer_chunk_size,
    artifact_source_metadata,
    build_artifact_manifest,
)
from modelexpress.metadata.artifact_transfer import NixlArtifactChunkManager
from modelexpress.metadata.worker_server import (
    WorkerServiceServicer,
    fetch_artifact_manifest_chunks,
    fetch_artifact_manifest_header,
    prepare_artifact_chunk,
    release_artifact_chunk,
)


def test_build_artifact_manifest_sorts_hashes_and_chunks_files(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"abcdef")
    (tmp_path / "a.txt").write_bytes(b"xyz")

    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=4,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )

    assert manifest.manifest_version == 1
    assert manifest.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE
    assert manifest.chunk_size == 4
    assert [file.path for file in manifest.files] == [
        (tmp_path / "a.txt").resolve().as_posix(),
        (nested / "b.bin").resolve().as_posix(),
    ]
    assert [file.file_index for file in manifest.files] == [0, 1]
    assert [file.checksum for file in manifest.files] == ["25236885", "53bceff1"]
    assert [
        (
            chunk.chunk_index,
            chunk.file_index,
            chunk.file_offset,
            chunk.length,
            chunk.checksum,
        )
        for chunk in manifest.chunks
    ] == [
        (0, 0, 0, 3, "25236885"),
        (1, 1, 0, 4, "92c80a31"),
        (2, 1, 4, 2, "6bb2dff5"),
    ]

    metadata = artifact_source_metadata(manifest)
    assert metadata.artifact_id == artifact_manifest_id(manifest)
    assert metadata.total_size == 9
    assert metadata.file_count == 2
    assert metadata.chunk_count == 3


def test_build_artifact_manifest_sorts_by_manifest_path_with_prefix_collision(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "inner.bin").write_bytes(b"inner")
    (tmp_path / "sub.txt").write_bytes(b"text")

    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=8,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )

    assert [file.path for file in manifest.files] == [
        (tmp_path / "sub.txt").resolve().as_posix(),
        (sub / "inner.bin").resolve().as_posix(),
    ]
    assert [
        (chunk.chunk_index, chunk.file_index, chunk.file_offset, chunk.length)
        for chunk in manifest.chunks
    ] == [
        (0, 0, 0, 4),
        (1, 1, 0, 5),
    ]


def test_build_artifact_manifest_rejects_invalid_chunk_size(tmp_path):
    (tmp_path / "artifact.bin").write_bytes(b"artifact")

    with pytest.raises(ValueError, match="chunk_size"):
        build_artifact_manifest(
            tmp_path,
            chunk_size=0,
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        )
    with pytest.raises(ValueError, match="exceeds maximum"):
        build_artifact_manifest(
            tmp_path,
            chunk_size=MAX_ARTIFACT_TRANSFER_CHUNK_SIZE + 1,
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        )


def test_build_artifact_manifest_skips_symlinks(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"target")
    (tmp_path / "link").symlink_to(target)

    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=4,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )

    assert [file.path for file in manifest.files] == [target.resolve().as_posix()]


def test_build_artifact_manifest_keeps_empty_file_without_transfer_chunk(tmp_path):
    (tmp_path / "empty.bin").write_bytes(b"")
    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=3,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )

    assert [(file.path, file.size) for file in manifest.files] == [
        ((tmp_path / "empty.bin").resolve().as_posix(), 0)
    ]
    assert list(manifest.chunks) == []


def test_artifact_transfer_chunk_size_env(monkeypatch):
    monkeypatch.setenv(MX_ARTIFACT_TRANSFER_CHUNK_SIZE_ENV, "12345")

    assert artifact_transfer_chunk_size() == 12345


def test_artifact_manifest_uses_env_chunk_size(monkeypatch, tmp_path):
    monkeypatch.setenv(MX_ARTIFACT_TRANSFER_CHUNK_SIZE_ENV, "4")
    (tmp_path / "artifact.bin").write_bytes(b"abcdef")

    manifest = build_artifact_manifest(
        tmp_path,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )

    assert manifest.chunk_size == 4
    assert [chunk.length for chunk in manifest.chunks] == [4, 2]


def test_worker_service_serves_artifact_manifest(tmp_path):
    (tmp_path / "artifact.bin").write_bytes(b"artifact")
    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=3,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )
    artifact_id = artifact_manifest_id(manifest)
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id="source-123",
        artifact_manifests={artifact_id: manifest},
    )
    server, port = _start_server(servicer)

    try:
        header, response_bytes = fetch_artifact_manifest_header(
            f"127.0.0.1:{port}",
            mx_source_id="source-123",
            artifact_id=artifact_id,
        )
        chunks, _ = fetch_artifact_manifest_chunks(
            f"127.0.0.1:{port}",
            mx_source_id="source-123",
            artifact_id=artifact_id,
            max_chunks=2,
        )
    finally:
        server.stop(grace=None)

    assert response_bytes > 0
    assert header.artifact_id == artifact_id
    assert header.manifest_version == manifest.manifest_version
    assert header.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE
    assert header.chunk_size == manifest.chunk_size
    assert list(header.files) == list(manifest.files)
    assert len(chunks.chunks) == 2
    assert chunks.next_page_token == "2"


def test_worker_service_defaults_and_caps_chunk_pages():
    manifest = p2p_pb2.ArtifactManifest(
        manifest_version=1,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        chunk_size=1,
        files=[
            p2p_pb2.ArtifactManifestFile(
                file_index=0,
                path="/tmp/artifact.bin",
                size=1025,
                checksum="file",
            )
        ],
        chunks=[
            p2p_pb2.ArtifactManifestChunk(
                chunk_index=index,
                file_index=0,
                file_offset=index,
                length=1,
                checksum=f"chunk-{index}",
            )
            for index in range(1025)
        ],
    )
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id="source-123",
        artifact_manifests={"artifact": manifest},
    )
    server, port = _start_server(servicer)

    try:
        default_page, _ = fetch_artifact_manifest_chunks(
            f"127.0.0.1:{port}",
            mx_source_id="source-123",
            artifact_id="artifact",
        )
        capped_page, _ = fetch_artifact_manifest_chunks(
            f"127.0.0.1:{port}",
            mx_source_id="source-123",
            artifact_id="artifact",
            max_chunks=2048,
        )
    finally:
        server.stop(grace=None)

    assert len(default_page.chunks) == 1024
    assert default_page.next_page_token == "1024"
    assert len(capped_page.chunks) == 1024
    assert capped_page.next_page_token == "1024"


def test_worker_service_prepares_and_releases_artifact_chunk(tmp_path):
    (tmp_path / "artifact.bin").write_bytes(b"artifact")
    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=3,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )
    artifact_id = artifact_manifest_id(manifest)
    fake_nixl = _FakeNixlManager()
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id="source-123",
        artifact_manifests={artifact_id: manifest},
        artifact_chunk_manager=NixlArtifactChunkManager(fake_nixl, max_buffers=1),
        metadata_endpoint="10.0.0.1:5555",
        agent_name="source-agent",
        worker_rank=4,
    )
    server, port = _start_server(servicer)

    try:
        response, response_bytes = prepare_artifact_chunk(
            f"127.0.0.1:{port}",
            mx_source_id="source-123",
            artifact_id=artifact_id,
            chunk_index=1,
        )
        released, _ = release_artifact_chunk(
            f"127.0.0.1:{port}",
            mx_source_id="source-123",
            artifact_id=artifact_id,
            lease_id=response.lease_id,
        )
    finally:
        server.stop(grace=None)

    assert response_bytes > 0
    assert response.artifact_id == artifact_id
    assert response.lease_id
    assert response.chunk == manifest.chunks[1]
    assert response.source.addr > 0
    assert response.source.length == 3
    assert response.source.device_id == 0
    assert response.source_metadata == b"fake-nixl-metadata"
    assert fake_nixl.registered_sizes == [3]
    assert fake_nixl.deregistered_count == 0
    assert released.chunk == manifest.chunks[1]


def test_artifact_chunk_manager_reuses_registered_buffer(tmp_path):
    (tmp_path / "artifact.bin").write_bytes(b"abcdef")
    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=3,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )
    fake_nixl = _FakeNixlManager()
    manager = NixlArtifactChunkManager(fake_nixl, max_buffers=1)

    first_lease, first_source, _ = manager.prepare(
        manifest,
        "artifact-1",
        manifest.chunks[0],
    )
    manager.release(first_lease)
    second_lease, second_source, _ = manager.prepare(
        manifest,
        "artifact-1",
        manifest.chunks[1],
    )
    manager.release(second_lease)

    assert fake_nixl.registered_sizes == [3]
    assert first_source.addr == second_source.addr
    assert fake_nixl.deregistered_count == 0


def test_artifact_chunk_manager_close_deregisters_buffers(tmp_path):
    (tmp_path / "artifact.bin").write_bytes(b"abcdef")
    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=3,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )
    fake_nixl = _FakeNixlManager()
    manager = NixlArtifactChunkManager(fake_nixl, max_buffers=2)

    lease_id, _, _ = manager.prepare(
        manifest,
        "artifact-1",
        manifest.chunks[0],
    )
    manager.release(lease_id)
    manager.close()
    manager.close()

    assert fake_nixl.registered_sizes == [3, 3]
    assert fake_nixl.deregistered_count == 2


def test_artifact_chunk_manager_releases_expired_lease_slot(tmp_path):
    (tmp_path / "artifact.bin").write_bytes(b"abcdef")
    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=3,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )
    fake_nixl = _FakeNixlManager()
    manager = NixlArtifactChunkManager(
        fake_nixl,
        max_buffers=1,
        lease_ttl_seconds=0.01,
    )

    first_lease, first_source, _ = manager.prepare(
        manifest,
        "artifact-1",
        manifest.chunks[0],
    )
    time.sleep(0.02)
    second_lease, second_source, _ = manager.prepare(
        manifest,
        "artifact-1",
        manifest.chunks[1],
    )
    manager.release(second_lease)
    manager.close()

    assert first_lease != second_lease
    assert fake_nixl.registered_sizes == [3]
    assert fake_nixl.deregistered_count == 1
    assert first_source.addr == second_source.addr
    assert first_lease not in manager._leases


def test_worker_service_rejects_invalid_artifact_chunk_index(tmp_path):
    (tmp_path / "artifact.bin").write_bytes(b"artifact")
    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=3,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )
    artifact_id = artifact_manifest_id(manifest)
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id="source-123",
        artifact_manifests={artifact_id: manifest},
        artifact_chunk_manager=NixlArtifactChunkManager(_FakeNixlManager()),
    )
    server, port = _start_server(servicer)

    try:
        with pytest.raises(grpc.RpcError) as exc_info:
            prepare_artifact_chunk(
                f"127.0.0.1:{port}",
                mx_source_id="source-123",
                artifact_id=artifact_id,
                chunk_index=len(manifest.chunks),
            )
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        server.stop(grace=None)


def test_worker_service_rejects_prepare_without_chunk_manager(tmp_path):
    (tmp_path / "artifact.bin").write_bytes(b"artifact")
    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=3,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )
    artifact_id = artifact_manifest_id(manifest)
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id="source-123",
        artifact_manifests={artifact_id: manifest},
    )
    server, port = _start_server(servicer)

    try:
        with pytest.raises(grpc.RpcError) as exc_info:
            prepare_artifact_chunk(
                f"127.0.0.1:{port}",
                mx_source_id="source-123",
                artifact_id=artifact_id,
                chunk_index=0,
            )
        assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    finally:
        server.stop(grace=None)


def test_worker_service_rejects_mismatched_artifact_id(tmp_path):
    (tmp_path / "artifact.bin").write_bytes(b"artifact")
    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=3,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id="source-123",
        artifact_manifests={artifact_manifest_id(manifest): manifest},
    )
    server, port = _start_server(servicer)

    try:
        with pytest.raises(grpc.RpcError) as exc_info:
            fetch_artifact_manifest_header(
                f"127.0.0.1:{port}",
                mx_source_id="source-123",
                artifact_id="wrong",
            )
        assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    finally:
        server.stop(grace=None)


def test_worker_service_returns_not_found_without_artifact_manifest():
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id="source-123",
    )
    server, port = _start_server(servicer)

    try:
        with pytest.raises(grpc.RpcError) as exc_info:
            fetch_artifact_manifest_header(
                f"127.0.0.1:{port}",
                mx_source_id="source-123",
            )
        assert exc_info.value.code() == grpc.StatusCode.NOT_FOUND
    finally:
        server.stop(grace=None)


def test_worker_service_selects_artifact_manifest_by_id(tmp_path):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    (left_dir / "torchinductor.bin").write_bytes(b"torchinductor")
    (right_dir / "triton.cubin").write_bytes(b"triton")
    left = build_artifact_manifest(
        left_dir,
        chunk_size=8,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )
    right = build_artifact_manifest(
        right_dir,
        chunk_size=8,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
    )
    left_id = artifact_manifest_id(left)
    right_id = artifact_manifest_id(right)
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id="source-123",
        artifact_manifests={
            left_id: left,
            right_id: right,
        },
    )
    server, port = _start_server(servicer)

    try:
        header, _ = fetch_artifact_manifest_header(
            f"127.0.0.1:{port}",
            mx_source_id="source-123",
            artifact_id=right_id,
        )
    finally:
        server.stop(grace=None)

    assert header.artifact_id == right_id
    assert header.mx_source_type == p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE
    assert list(header.files) == list(right.files)


def test_worker_service_requires_artifact_id_for_multiple_manifests(tmp_path):
    left_dir = tmp_path / "left"
    right_dir = tmp_path / "right"
    left_dir.mkdir()
    right_dir.mkdir()
    (left_dir / "torchinductor.bin").write_bytes(b"torchinductor")
    (right_dir / "triton.cubin").write_bytes(b"triton")
    left = build_artifact_manifest(
        left_dir,
        chunk_size=8,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )
    right = build_artifact_manifest(
        right_dir,
        chunk_size=8,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
    )
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id="source-123",
        artifact_manifests={
            artifact_manifest_id(left): left,
            artifact_manifest_id(right): right,
        },
    )
    server, port = _start_server(servicer)

    try:
        with pytest.raises(grpc.RpcError) as exc_info:
            fetch_artifact_manifest_header(
                f"127.0.0.1:{port}",
                mx_source_id="source-123",
            )
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    finally:
        server.stop(grace=None)


def test_pinned_artifact_manifest_id_cross_checked_with_rust():
    manifest = p2p_pb2.ArtifactManifest(
        manifest_version=1,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        chunk_size=8,
        files=[
            p2p_pb2.ArtifactManifestFile(
                file_index=0,
                path="/cache/sub.txt",
                size=4,
                checksum="text-file-checksum",
            ),
            p2p_pb2.ArtifactManifestFile(
                file_index=1,
                path="/cache/sub/inner.bin",
                size=5,
                checksum="inner-file-checksum",
            ),
        ],
        chunks=[
            p2p_pb2.ArtifactManifestChunk(
                chunk_index=0,
                file_index=0,
                file_offset=0,
                length=4,
                checksum="text-chunk-checksum",
            ),
            p2p_pb2.ArtifactManifestChunk(
                chunk_index=1,
                file_index=1,
                file_offset=0,
                length=5,
                checksum="inner-chunk-checksum",
            ),
        ],
    )

    assert (
        artifact_manifest_id(manifest)
        == "a0f08392f2abc45f78bd59f0fe2c601750c2b270dc5cc37c2166d86a65398466"
    )


def _start_server(servicer) -> tuple[grpc.Server, int]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    p2p_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, port


class _FakeNixlManager:
    def __init__(self):
        self.registered_sizes = []
        self.deregistered_count = 0
        self.nixl_metadata = b"fake-nixl-metadata"

    def register_dram_buffer(self, buffer):
        self.registered_sizes.append(buffer.numel())
        return object()

    def deregister_memory(self, registered):
        del registered
        self.deregistered_count += 1
