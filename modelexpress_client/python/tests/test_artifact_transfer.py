# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI-safe artifact transfer protocol tests."""

from __future__ import annotations

import io
import logging
import tarfile
from concurrent import futures
from pathlib import Path
from unittest.mock import MagicMock

import grpc
import pytest
import torch

import modelexpress.metadata.artifact_transfer as artifact_transfer_module
from modelexpress import p2p_pb2, p2p_pb2_grpc
from modelexpress.metadata.artifact_manifest import (
    artifact_manifest_id,
    build_artifact_manifest,
)
from modelexpress.metadata.artifact_transfer import (
    ArtifactBundle,
    ArtifactCacheRoot,
    P2PArtifactTransfer,
    cute_dsl_cache_artifact_transfer,
    deep_gemm_cache_artifact_transfer,
    discover_artifact_source,
    flashinfer_cache_artifact_transfer,
    publish_artifact_source,
    transfer_artifact_from_worker,
    triton_cache_artifact_transfer,
    tilelang_cache_artifact_transfer,
    tvm_ffi_cache_artifact_transfer,
    torch_compile_cache_artifact_transfer,
)
from modelexpress.metadata.source_id import compute_mx_source_id
from modelexpress.metadata.worker_server import (
    WorkerGrpcServer,
    WorkerServiceServicer,
    fetch_tensor_manifest,
)


def test_transfer_artifact_from_worker_reconstructs_file_and_releases_leases(tmp_path):
    artifact_file = tmp_path / "cache.bin"
    artifact_file.write_bytes(b"0123456789abcdef")
    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=5,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )
    artifact_id = artifact_manifest_id(manifest)
    source_bytes_by_path = {
        file.path: artifact_file.read_bytes()
        for file in manifest.files
    }
    chunk_manager = _FakeArtifactChunkManager(source_bytes_by_path)
    target_nixl = _FakeTargetNixlManager(chunk_manager)
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id="source-123",
        artifact_manifests={artifact_id: manifest},
        artifact_chunk_manager=chunk_manager,
        metadata_endpoint="127.0.0.1:5555",
        agent_name="source-agent",
        worker_rank=0,
    )
    server, port = _start_server(servicer)

    try:
        header = transfer_artifact_from_worker(
            f"127.0.0.1:{port}",
            mx_source_id="source-123",
            artifact_id=artifact_id,
            nixl_manager=target_nixl,
            timeout=1.0,
        )
    finally:
        server.stop(grace=None)

    assert header.artifact_id == artifact_id
    assert artifact_file.read_bytes() == b"0123456789abcdef"
    assert target_nixl.loaded_metadata == [b"fake-source-metadata"]
    assert sorted(target_nixl.received_addrs) == [
        chunk_manager.prepared_addrs_by_chunk[index]
        for index in range(len(manifest.chunks))
    ]
    assert target_nixl.registered_sizes == [5, 5, 5, 5]
    assert target_nixl.deregistered_count == 4
    assert not chunk_manager.leases
    assert chunk_manager.released_chunks == [0, 1, 2, 3]


def test_worker_grpc_server_shares_port_for_tensor_and_artifact_sources(tmp_path):
    artifact_file = tmp_path / "cache.bin"
    artifact_file.write_bytes(b"compiled-cache")
    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=8,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )
    artifact_id = artifact_manifest_id(manifest)
    server = WorkerGrpcServer(
        tensor_protos=[
            p2p_pb2.TensorDescriptor(
                name="weight",
                addr=1234,
                size=8,
                device_id=0,
                dtype="torch.float16",
            )
        ],
        mx_source_id=None,
        port=0,
        metadata_endpoint="127.0.0.1:5555",
        agent_name="source-agent",
        worker_rank=0,
        worker_id="weight-generation",
    )
    port = server.start()
    endpoint = f"127.0.0.1:{port}"
    server.set_mx_source_id("weight-source")
    server.register_artifact_source(
        "artifact-source",
        artifact_id,
        manifest,
        object(),
    )

    try:
        tensors, _ = fetch_tensor_manifest(
            endpoint,
            "weight-source",
            timeout=1.0,
            worker_id="weight-generation",
        )
        header, _ = artifact_transfer_module.fetch_artifact_manifest_header(
            endpoint,
            "artifact-source",
            artifact_id,
            timeout=1.0,
        )
        with pytest.raises(grpc.RpcError):
            artifact_transfer_module.fetch_artifact_manifest_header(
                endpoint,
                "weight-source",
                artifact_id,
                timeout=1.0,
            )
    finally:
        server.stop(grace=None)

    assert [tensor.name for tensor in tensors] == ["weight"]
    assert header.mx_source_id == "artifact-source"
    assert header.artifact_id == artifact_id


def test_fetch_tensor_manifest_rejects_stale_worker_generation():
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id="source-123",
        worker_id="new-generation",
    )
    server, port = _start_server(servicer)

    try:
        with pytest.raises(grpc.RpcError) as exc_info:
            fetch_tensor_manifest(
                f"127.0.0.1:{port}",
                "source-123",
                worker_id="old-generation",
                timeout=1.0,
            )
    finally:
        server.stop(grace=None)

    assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    assert "worker_id mismatch" in exc_info.value.details()


def test_fetch_tensor_manifest_default_timeout_is_five_seconds(monkeypatch):
    channel = MagicMock()
    stub = MagicMock()
    stub.GetTensorManifest.return_value = p2p_pb2.GetTensorManifestResponse()
    monkeypatch.setattr(grpc, "insecure_channel", MagicMock(return_value=channel))
    monkeypatch.setattr(
        p2p_pb2_grpc,
        "WorkerServiceStub",
        MagicMock(return_value=stub),
    )

    fetch_tensor_manifest("source:6555", "source-123")

    assert stub.GetTensorManifest.call_args.kwargs["timeout"] == 5.0
    channel.close.assert_called_once()


def test_fetch_tensor_manifest_closes_channel_on_rpc_error(monkeypatch):
    channel = MagicMock()
    stub = MagicMock()
    stub.GetTensorManifest.side_effect = grpc.RpcError("manifest failed")
    monkeypatch.setattr(grpc, "insecure_channel", MagicMock(return_value=channel))
    monkeypatch.setattr(
        p2p_pb2_grpc,
        "WorkerServiceStub",
        MagicMock(return_value=stub),
    )

    with pytest.raises(grpc.RpcError, match="manifest failed"):
        fetch_tensor_manifest("source:6555", "source-123")

    channel.close.assert_called_once()


def test_fetch_tensor_manifest_accepts_legacy_source_without_worker_id():
    class LegacyServicer(p2p_pb2_grpc.WorkerServiceServicer):
        def GetTensorManifest(self, request, context):
            return p2p_pb2.GetTensorManifestResponse(
                mx_source_id=request.mx_source_id,
            )

    server, port = _start_server(LegacyServicer())

    try:
        tensors, _ = fetch_tensor_manifest(
            f"127.0.0.1:{port}",
            "source-123",
            worker_id="selected-generation",
            timeout=1.0,
        )
    finally:
        server.stop(grace=None)

    assert tensors == []


def test_new_server_accepts_legacy_request_without_worker_id():
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id="source-123",
        worker_id="new-generation",
    )
    server, port = _start_server(servicer)

    try:
        tensors, _ = fetch_tensor_manifest(
            f"127.0.0.1:{port}",
            "source-123",
            timeout=1.0,
        )
    finally:
        server.stop(grace=None)

    assert tensors == []


def test_transfer_artifact_from_worker_retries_after_source_buffer_exhaustion(
    tmp_path,
):
    artifact_file = tmp_path / "cache.bin"
    artifact_file.write_bytes(b"0123456789abcdef")
    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=4,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )
    artifact_id = artifact_manifest_id(manifest)
    source_bytes_by_path = {
        file.path: artifact_file.read_bytes()
        for file in manifest.files
    }
    chunk_manager = _LimitedFakeArtifactChunkManager(source_bytes_by_path, max_active=2)
    target_nixl = _FakeTargetNixlManager(chunk_manager)
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id="source-123",
        artifact_manifests={artifact_id: manifest},
        artifact_chunk_manager=chunk_manager,
        metadata_endpoint="127.0.0.1:5555",
        agent_name="source-agent",
        worker_rank=0,
    )
    server, port = _start_server(servicer)

    try:
        transfer_artifact_from_worker(
            f"127.0.0.1:{port}",
            mx_source_id="source-123",
            artifact_id=artifact_id,
            nixl_manager=target_nixl,
            timeout=1.0,
            max_inflight_chunks=4,
        )
    finally:
        server.stop(grace=None)

    assert artifact_file.read_bytes() == b"0123456789abcdef"
    assert not chunk_manager.leases
    assert chunk_manager.max_observed_active == 2
    assert chunk_manager.released_chunks == [0, 1, 2, 3]


def test_transfer_artifact_from_worker_cleans_partial_files_on_checksum_error(tmp_path):
    artifact_file = tmp_path / "cache.bin"
    artifact_file.write_bytes(b"0123456789abcdef")
    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=16,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )
    artifact_id = artifact_manifest_id(manifest)
    source_bytes_by_path = {file.path: b"xxxxxxxxxxxxxxxx" for file in manifest.files}
    chunk_manager = _FakeArtifactChunkManager(source_bytes_by_path)
    target_nixl = _FakeTargetNixlManager(chunk_manager)
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id="source-123",
        artifact_manifests={artifact_id: manifest},
        artifact_chunk_manager=chunk_manager,
        metadata_endpoint="127.0.0.1:5555",
        agent_name="source-agent",
        worker_rank=0,
    )
    server, port = _start_server(servicer)

    try:
        with pytest.raises(RuntimeError, match="crc32c mismatch"):
            transfer_artifact_from_worker(
                f"127.0.0.1:{port}",
                mx_source_id="source-123",
                artifact_id=artifact_id,
                nixl_manager=target_nixl,
                timeout=1.0,
            )
    finally:
        server.stop(grace=None)

    assert not artifact_file.exists()
    assert not chunk_manager.leases
    assert target_nixl.deregistered_count == 1


def test_validate_fetched_manifest_rejects_missing_chunk(tmp_path):
    artifact_file = tmp_path / "cache.bin"
    artifact_file.write_bytes(b"0123456789abcdef")
    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=5,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )
    header = _artifact_header_from_manifest(manifest)

    with pytest.raises(RuntimeError, match="manifest count mismatch"):
        artifact_transfer_module._validate_fetched_artifact_manifest(
            header,
            list(manifest.chunks[:-1]),
            header.artifact_id,
        )


def test_validate_fetched_manifest_rejects_overlapping_chunks(tmp_path):
    artifact_file = tmp_path / "cache.bin"
    artifact_file.write_bytes(b"0123456789abcdef")
    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=8,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )
    chunks = [p2p_pb2.ArtifactManifestChunk() for _ in manifest.chunks]
    for target, source in zip(chunks, manifest.chunks, strict=True):
        target.CopyFrom(source)
    chunks[1].file_offset = 0
    header = _artifact_header_from_manifest(manifest, chunks=chunks)

    with pytest.raises(RuntimeError, match="coverage gap or overlap"):
        artifact_transfer_module._validate_fetched_artifact_manifest(
            header,
            chunks,
            header.artifact_id,
        )


def test_validate_fetched_manifest_rejects_manifest_id_mismatch(tmp_path):
    artifact_file = tmp_path / "cache.bin"
    artifact_file.write_bytes(b"0123456789abcdef")
    manifest = build_artifact_manifest(
        tmp_path,
        chunk_size=8,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
    )
    header = _artifact_header_from_manifest(manifest)
    header.artifact_id = "bad-artifact-id"

    with pytest.raises(RuntimeError, match="artifact header id mismatch"):
        artifact_transfer_module._validate_fetched_artifact_manifest(
            header,
            list(manifest.chunks),
            artifact_manifest_id(manifest),
        )


def test_tarred_p2p_artifact_transfer_prepares_single_file_bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    nested = source / "nested"
    nested.mkdir()
    (source / "a.txt").write_text("alpha")
    (nested / "b.bin").write_bytes(b"beta")

    transfer = torch_compile_cache_artifact_transfer(
        source,
        tmp_path / "extract",
        tmp_path / "bundle",
        chunk_size=4,
    )
    bundle = transfer.prepare_source()
    header = p2p_pb2.GetArtifactManifestHeaderResponse(files=bundle.manifest.files)
    transfer.install(header)

    assert bundle.tar_path == (tmp_path / "bundle" / "artifact.tar").resolve()
    assert bundle.artifact_id == artifact_manifest_id(bundle.manifest)
    assert [file.path for file in bundle.manifest.files] == [
        bundle.tar_path.as_posix()
    ]
    assert bundle.manifest.files[0].size == bundle.tar_path.stat().st_size
    assert len(bundle.manifest.chunks) > 1
    assert (transfer.roots[0].target_root / "a.txt").read_text() == "alpha"
    assert (transfer.roots[0].target_root / "nested" / "b.bin").read_bytes() == b"beta"
    assert isinstance(bundle, ArtifactBundle)


def test_flashinfer_cache_transfer_includes_engine_autotune_files(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="modelexpress.metadata.artifact_transfer")
    jit_source = tmp_path / "jit-source"
    autotune_source = tmp_path / "autotune-source"
    jit_source.mkdir()
    autotune_source.mkdir()
    (jit_source / "kernel.so").write_bytes(b"compiled")
    (autotune_source / "configs.json").write_text("{}")
    jit_target = tmp_path / "jit-target"
    autotune_target = tmp_path / "autotune-target"

    source_transfer = flashinfer_cache_artifact_transfer(
        jit_source,
        tmp_path / "unused-jit-target",
        tmp_path / "source-bundle",
        additional_roots=(
            ArtifactCacheRoot(
                name="autotune",
                source_root=autotune_source,
                target_root=tmp_path / "unused-autotune-target",
                optional=True,
            ),
        ),
    )
    target_transfer = flashinfer_cache_artifact_transfer(
        jit_source,
        jit_target,
        tmp_path / "target-bundle",
        additional_roots=(
            ArtifactCacheRoot(
                name="autotune",
                source_root=autotune_source,
                target_root=autotune_target,
                optional=True,
            ),
        ),
    )
    bundle = source_transfer.prepare_source()
    source_bytes_by_path = {
        file.path: Path(file.path).read_bytes() for file in bundle.manifest.files
    }
    chunk_manager = _FakeArtifactChunkManager(source_bytes_by_path)
    target_nixl = _FakeTargetNixlManager(chunk_manager)
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id="source-123",
        artifact_manifests={bundle.artifact_id: bundle.manifest},
        artifact_chunk_manager=chunk_manager,
        metadata_endpoint="127.0.0.1:5555",
        agent_name="source-agent",
        worker_rank=0,
    )
    server, port = _start_server(servicer)

    try:
        header = target_transfer.transfer_from_worker(
            f"127.0.0.1:{port}",
            mx_source_id="source-123",
            artifact_id=bundle.artifact_id,
            nixl_manager=target_nixl,
            timeout=1.0,
        )
        target_transfer.install(header)
    finally:
        server.stop(grace=None)

    assert [Path(file.path).name for file in bundle.manifest.files] == [
        "artifact.tar",
        "autotune.tar",
    ]
    assert [Path(file.path).name for file in header.files] == [
        "artifact.tar",
        "autotune.tar",
    ]
    assert (jit_target / "kernel.so").read_bytes() == b"compiled"
    assert (autotune_target / "configs.json").read_text() == "{}"
    assert f"targets=['{jit_target}', '{autotune_target}']" in caplog.text


def test_flashinfer_cache_transfer_allows_missing_optional_root(tmp_path):
    jit_source = tmp_path / "jit-source"
    jit_source.mkdir()
    (jit_source / "kernel.so").write_bytes(b"compiled")

    transfer = flashinfer_cache_artifact_transfer(
        jit_source,
        tmp_path / "jit-target",
        tmp_path / "bundle",
        additional_roots=(
            ArtifactCacheRoot(
                name="autotune",
                source_root=tmp_path / "missing-autotune-source",
                target_root=tmp_path / "autotune-target",
                optional=True,
            ),
        ),
    )
    bundle = transfer.prepare_source()
    transfer.install(
        p2p_pb2.GetArtifactManifestHeaderResponse(files=bundle.manifest.files)
    )

    assert (transfer.roots[0].target_root / "kernel.so").read_bytes() == b"compiled"
    assert (tmp_path / "autotune-target").is_dir()


def test_tarred_p2p_artifact_transfer_rejects_unsafe_staging(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    transfer = torch_compile_cache_artifact_transfer(
        source,
        tmp_path / "target",
        source / "bundle",
    )

    with pytest.raises(ValueError, match="bundle_root"):
        transfer.prepare_source()


def test_tarred_p2p_artifact_transfer_rejects_stale_bundle_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "stale.bin").write_bytes(b"stale")
    transfer = torch_compile_cache_artifact_transfer(
        source,
        tmp_path / "target",
        bundle,
    )

    with pytest.raises(ValueError, match="dedicated"):
        transfer.prepare_source()


def test_tarred_p2p_artifact_transfer_rejects_target_bundle_symlink(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    bundle = tmp_path / "target-bundle"
    bundle.mkdir()
    outside = tmp_path / "outside.tar"
    outside.write_bytes(b"keep-me")
    (bundle / "artifact.tar").symlink_to(outside)
    transfer = torch_compile_cache_artifact_transfer(
        source,
        tmp_path / "target",
        bundle,
    )

    with pytest.raises(ValueError, match="symlink"):
        transfer.transfer_from_worker(
            "127.0.0.1:1",
            mx_source_id="source-123",
            artifact_id="artifact",
            nixl_manager=object(),
        )
    assert outside.read_bytes() == b"keep-me"


def test_tarred_p2p_artifact_transfer_rejects_symlink(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    target = source / "target.txt"
    target.write_text("target")
    (source / "link.txt").symlink_to(target)
    transfer = torch_compile_cache_artifact_transfer(
        source,
        tmp_path / "target",
        tmp_path / "bundle",
    )

    with pytest.raises(ValueError, match="symlink"):
        transfer.prepare_source()


def test_extract_tarred_artifact_rejects_unsafe_member(tmp_path):
    tar_path = tmp_path / "artifact.tar"
    data = b"escape"
    info = tarfile.TarInfo("../escape.txt")
    info.size = len(data)
    with tarfile.open(tar_path, "w") as archive:
        archive.addfile(info, io.BytesIO(data))
    transfer = torch_compile_cache_artifact_transfer(
        tmp_path,
        tmp_path / "extract",
        tmp_path / "bundle",
    )
    header = p2p_pb2.GetArtifactManifestHeaderResponse(
        files=[p2p_pb2.ArtifactManifestFile(path=tar_path.as_posix())]
    )

    with pytest.raises(ValueError, match="unsafe tar member"):
        transfer.install(header)


def test_extract_tarred_artifact_rejects_duplicate_archive_names(tmp_path):
    transfer = torch_compile_cache_artifact_transfer(
        tmp_path,
        tmp_path / "extract",
        tmp_path / "bundle",
    )
    header = p2p_pb2.GetArtifactManifestHeaderResponse(
        files=[
            p2p_pb2.ArtifactManifestFile(path="left/artifact.tar"),
            p2p_pb2.ArtifactManifestFile(path="right/artifact.tar"),
        ]
    )

    with pytest.raises(ValueError, match="unique by archive name"):
        transfer.install(header)


def test_tarred_p2p_artifact_transfer_splits_transfer_and_install(tmp_path, caplog):
    source = tmp_path / "source"
    source.mkdir()
    (source / "bucket-000").mkdir()
    (source / "bucket-001").mkdir()
    (source / "bucket-000" / "kernel-00000.bin").write_bytes(b"compiled-cache-a")
    (source / "bucket-001" / "kernel-00001.bin").write_bytes(b"compiled-cache-b")
    source_transfer = torch_compile_cache_artifact_transfer(
        source,
        tmp_path / "unused-source-target",
        tmp_path / "source-bundle",
        chunk_size=5,
    )
    target_transfer = torch_compile_cache_artifact_transfer(
        source,
        tmp_path / "extract",
        tmp_path / "target-bundle",
        chunk_size=5,
    )
    with caplog.at_level(
        logging.INFO,
        logger="modelexpress.metadata.artifact_transfer",
    ):
        bundle = source_transfer.prepare_source()
    source_bytes_by_path = {
        file.path: Path(file.path).read_bytes()
        for file in bundle.manifest.files
    }
    chunk_manager = _FakeArtifactChunkManager(source_bytes_by_path)
    target_nixl = _FakeTargetNixlManager(chunk_manager)
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id="source-123",
        artifact_manifests={bundle.artifact_id: bundle.manifest},
        artifact_chunk_manager=chunk_manager,
        metadata_endpoint="127.0.0.1:5555",
        agent_name="source-agent",
        worker_rank=0,
    )
    server, port = _start_server(servicer)

    try:
        with caplog.at_level(
            logging.INFO,
            logger="modelexpress.metadata.artifact_transfer",
        ):
            header = target_transfer.transfer_from_worker(
                f"127.0.0.1:{port}",
                mx_source_id="source-123",
                artifact_id=bundle.artifact_id,
                nixl_manager=target_nixl,
                timeout=1.0,
            )
        assert header.files[0].path == (
            tmp_path / "target-bundle" / "artifact.tar"
        ).resolve().as_posix()
        assert not (target_transfer.roots[0].target_root / "bucket-000").exists()
        with caplog.at_level(
            logging.INFO,
            logger="modelexpress.metadata.artifact_transfer",
        ):
            target_transfer.install(header)
    finally:
        server.stop(grace=None)

    assert header.artifact_id == bundle.artifact_id
    installed_files = {
        file.relative_to(target_transfer.roots[0].target_root).as_posix(): file.read_bytes()
        for file in target_transfer.roots[0].target_root.rglob("*")
        if file.is_file()
    }
    assert installed_files == {
        "bucket-000/kernel-00000.bin": b"compiled-cache-a",
        "bucket-001/kernel-00001.bin": b"compiled-cache-b",
    }
    assert bundle.tar_path == (tmp_path / "source-bundle" / "artifact.tar").resolve()
    assert Path(header.files[0].path) == (
        tmp_path / "target-bundle" / "artifact.tar"
    ).resolve()
    assert not chunk_manager.leases
    assert chunk_manager.released_chunks == [
        chunk.chunk_index for chunk in bundle.manifest.chunks
    ]
    assert "[TIMING] Artifact prepare complete: name=torch_compile_cache" in caplog.text
    assert "[TIMING] Artifact transfer complete" in caplog.text
    assert "[TIMING] Artifact install complete" in caplog.text


@pytest.mark.parametrize(
    ("factory", "name", "mx_source_type"),
    [
        (
            torch_compile_cache_artifact_transfer,
            "torch_compile_cache",
            p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        ),
        (
            triton_cache_artifact_transfer,
            "triton_cache",
            p2p_pb2.MX_SOURCE_TYPE_TRITON_CACHE,
        ),
        (
            tvm_ffi_cache_artifact_transfer,
            "tvm_ffi_cache",
            p2p_pb2.MX_SOURCE_TYPE_TVM_FFI_CACHE,
        ),
        (
            deep_gemm_cache_artifact_transfer,
            "deep_gemm_cache",
            p2p_pb2.MX_SOURCE_TYPE_DEEP_GEMM_CACHE,
        ),
        (
            tilelang_cache_artifact_transfer,
            "tilelang_cache",
            p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE,
        ),
        (
            cute_dsl_cache_artifact_transfer,
            "cute_dsl_cache",
            p2p_pb2.MX_SOURCE_TYPE_CUTE_DSL_CACHE,
        ),
        (
            flashinfer_cache_artifact_transfer,
            "flashinfer_cache",
            p2p_pb2.MX_SOURCE_TYPE_FLASHINFER_CACHE,
        ),
    ],
)
def test_cache_artifact_transfers_share_p2p_interface(
    tmp_path,
    factory,
    name,
    mx_source_type,
):
    source = tmp_path / f"{name}-source"
    source.mkdir()
    (source / "kernel.so").write_bytes(f"{name}-bytes".encode())
    transfer = factory(
        source,
        tmp_path / f"{name}-target",
        tmp_path / f"{name}-bundle",
        chunk_size=6,
    )

    assert isinstance(transfer, P2PArtifactTransfer)
    assert transfer.name == name
    assert transfer.mx_source_type == mx_source_type
    bundle = transfer.prepare_source()
    source_bytes_by_path = {
        file.path: Path(file.path).read_bytes()
        for file in bundle.manifest.files
    }
    chunk_manager = _FakeArtifactChunkManager(source_bytes_by_path)
    target_nixl = _FakeTargetNixlManager(chunk_manager)
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id="source-123",
        artifact_manifests={bundle.artifact_id: bundle.manifest},
        artifact_chunk_manager=chunk_manager,
        metadata_endpoint="127.0.0.1:5555",
        agent_name="source-agent",
        worker_rank=0,
    )
    server, port = _start_server(servicer)

    try:
        header = transfer.transfer_from_worker(
            f"127.0.0.1:{port}",
            mx_source_id="source-123",
            artifact_id=bundle.artifact_id,
            nixl_manager=target_nixl,
            timeout=1.0,
        )
        assert not (transfer.roots[0].target_root / "kernel.so").exists()
        transfer.install(header)
    finally:
        server.stop(grace=None)

    assert header.mx_source_type == mx_source_type
    assert (transfer.roots[0].target_root / "kernel.so").read_bytes() == (
        f"{name}-bytes".encode()
    )


def test_publish_artifact_source_registers_mx_discovery_metadata(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.bin").write_bytes(b"compiled-cache")
    transfer = torch_compile_cache_artifact_transfer(
        source,
        tmp_path / "target",
        tmp_path / "bundle",
        chunk_size=8,
    )
    bundle = transfer.prepare_source()
    identity = p2p_pb2.SourceIdentity(
        mx_version="0.5.0",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        model_name="test/model",
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        backend_framework_version="vllm-0.10.0",
        torch_version="2.8.0",
        cuda_version="12.8",
        triton_version="3.4.0",
        gpu_arch="sm90",
        compile_config_digest="cache-key",
    )
    mx_client = _FakeMxClient(mx_source_id="server-artifact-source-id")
    source_nixl = _FakeSourceNixlManager(listen_port=7010)
    worker_server = _FakeWorkerGrpcServer()

    published = publish_artifact_source(
        mx_client,
        transfer,
        bundle,
        identity,
        source_nixl,
        worker_id="source-worker-0",
        worker_grpc_server=worker_server,
        host="127.0.0.1",
    )
    try:
        discovered = discover_artifact_source(mx_client, identity)
    finally:
        published.stop()

    assert source_nixl.refresh_metadata_count == 1
    assert discovered == published.endpoint
    assert published.endpoint.mx_source_id == "server-artifact-source-id"
    assert worker_server.registered == [
        ("server-artifact-source-id", bundle.artifact_id)
    ]
    assert worker_server.unregistered == [
        ("server-artifact-source-id", bundle.artifact_id)
    ]
    assert mx_client.published_worker.worker_grpc_endpoint == (
        published.endpoint.worker_grpc_endpoint
    )
    assert mx_client.published_worker.metadata_endpoint == "127.0.0.1:7010"
    assert mx_client.published_worker.artifact_source.artifact_id == bundle.artifact_id
    assert mx_client.published_worker.artifact_source.file_count == 1


def test_discover_artifact_source_does_not_rank_match_by_default(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.bin").write_bytes(b"compiled-cache")
    transfer = torch_compile_cache_artifact_transfer(
        source,
        tmp_path / "target",
        tmp_path / "bundle",
        chunk_size=8,
    )
    bundle = transfer.prepare_source()
    identity = p2p_pb2.SourceIdentity(
        mx_version="0.5.0",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        model_name="test/model",
    )
    mx_client = _FakeMxClient()
    published = publish_artifact_source(
        mx_client,
        transfer,
        bundle,
        identity,
        _FakeSourceNixlManager(listen_port=7010),
        worker_id="source-worker-3",
        worker_rank=3,
        worker_grpc_server=_FakeWorkerGrpcServer(),
        host="127.0.0.1",
    )
    try:
        assert discover_artifact_source(mx_client, identity) == published.endpoint
        with pytest.raises(LookupError):
            discover_artifact_source(mx_client, identity, worker_rank=0)
    finally:
        published.stop()


def test_discover_artifact_source_matches_node_rank():
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE,
        model_name="test/model",
    )
    mx_source_id = compute_mx_source_id(identity)
    instances = [
        p2p_pb2.SourceInstanceRef(
            mx_source_id=mx_source_id,
            worker_id=f"source-pod-{node_rank}",
            worker_rank=worker_rank,
        )
        for node_rank, worker_rank in [(0, 3), (1, 7)]
    ]
    metadata = {
        f"source-pod-{node_rank}": p2p_pb2.GetMetadataResponse(
            found=True,
            mx_source_id=mx_source_id,
            worker_id=f"source-pod-{node_rank}",
            worker=p2p_pb2.WorkerMetadata(
                worker_rank=worker_rank,
                worker_grpc_endpoint=f"source-pod-{node_rank}:6555",
                artifact_source=p2p_pb2.ArtifactSourceMetadata(
                    artifact_id=f"artifact-{node_rank}",
                    node_rank=node_rank,
                ),
            ),
        )
        for node_rank, worker_rank in [(0, 3), (1, 7)]
    }
    mx_client = MagicMock()
    mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
        instances=instances
    )
    mx_client.get_metadata.side_effect = (
        lambda source_id, worker_id: metadata[worker_id]
    )

    source_0 = discover_artifact_source(mx_client, identity, node_rank=0)
    source_1 = discover_artifact_source(mx_client, identity, node_rank=1)

    assert source_0.worker_id == "source-pod-0"
    assert source_0.artifact_id == "artifact-0"
    assert source_1.worker_id == "source-pod-1"
    assert source_1.worker_rank == 7
    assert source_1.artifact_id == "artifact-1"


def _artifact_discovery_client(instances_and_workers):
    """Build a MagicMock mx_client from (SourceInstanceRef, WorkerMetadata) pairs."""
    identity = p2p_pb2.SourceIdentity(
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE,
        model_name="test/model",
    )
    metadata = {
        ref.worker_id: p2p_pb2.GetMetadataResponse(
            found=True,
            mx_source_id=ref.mx_source_id,
            worker_id=ref.worker_id,
            worker=worker,
        )
        for ref, worker in instances_and_workers
    }
    mx_client = MagicMock()
    mx_client.list_sources.return_value = p2p_pb2.ListSourcesResponse(
        instances=[ref for ref, _ in instances_and_workers]
    )
    mx_client.get_metadata.side_effect = (
        lambda source_id, worker_id: metadata[worker_id]
    )
    return mx_client, identity


def _artifact_worker(worker_grpc_endpoint, artifact_id, accelerator=""):
    return p2p_pb2.WorkerMetadata(
        worker_grpc_endpoint=worker_grpc_endpoint,
        accelerator=accelerator,
        artifact_source=p2p_pb2.ArtifactSourceMetadata(artifact_id=artifact_id),
    )


def test_discover_artifact_source_skips_incompatible_accelerator():
    # A compatible source listed after an incompatible one must still be
    # chosen; the incompatible source is dropped before GetMetadata.
    mx_source_id = compute_mx_source_id(
        p2p_pb2.SourceIdentity(
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE,
            model_name="test/model",
        )
    )
    incompatible = p2p_pb2.SourceInstanceRef(
        mx_source_id=mx_source_id, worker_id="other-0", accelerator="other"
    )
    compatible = p2p_pb2.SourceInstanceRef(
        mx_source_id=mx_source_id, worker_id="cuda-0", accelerator="cuda"
    )
    mx_client, identity = _artifact_discovery_client(
        [
            (incompatible, _artifact_worker("other-0:6555", "a-other", "other")),
            (compatible, _artifact_worker("cuda-0:6555", "a-cuda", "cuda")),
        ]
    )

    discovered = discover_artifact_source(mx_client, identity, accelerator="cuda")

    assert discovered.worker_id == "cuda-0"
    # The incompatible source was filtered before its metadata was fetched.
    mx_client.get_metadata.assert_called_once_with(mx_source_id, "cuda-0")


def test_discover_artifact_source_rejects_cross_family_hetero_pair():
    # cuda<->xpu is enabled for weights but artifacts stay strict same-family:
    # a cuda target must not install an xpu-published cache bundle.
    mx_source_id = compute_mx_source_id(
        p2p_pb2.SourceIdentity(
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE,
            model_name="test/model",
        )
    )
    xpu_source = p2p_pb2.SourceInstanceRef(
        mx_source_id=mx_source_id, worker_id="xpu-0", accelerator="xpu"
    )
    mx_client, identity = _artifact_discovery_client(
        [(xpu_source, _artifact_worker("xpu-0:6555", "a-xpu", "xpu"))]
    )

    with pytest.raises(LookupError):
        discover_artifact_source(mx_client, identity, accelerator="cuda")

    # The incompatible source was dropped before any metadata fetch.
    mx_client.get_metadata.assert_not_called()


def test_discover_artifact_source_empty_accelerator_is_compatible():
    # A source whose ref predates the accelerator field (empty) is accepted;
    # a populated mismatch is still skipped.
    mx_source_id = compute_mx_source_id(
        p2p_pb2.SourceIdentity(
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE,
            model_name="test/model",
        )
    )
    legacy = p2p_pb2.SourceInstanceRef(
        mx_source_id=mx_source_id, worker_id="legacy-0", accelerator=""
    )
    mx_client, identity = _artifact_discovery_client(
        [(legacy, _artifact_worker("legacy-0:6555", "a-legacy", ""))]
    )

    discovered = discover_artifact_source(mx_client, identity, accelerator="cuda")

    assert discovered.worker_id == "legacy-0"


def test_discover_artifact_source_rechecks_worker_metadata_accelerator():
    # Defense-in-depth: an empty ref accelerator passes the pre-fetch filter,
    # but the authoritative WorkerMetadata.accelerator mismatch is caught after
    # GetMetadata, so no incompatible source is returned.
    mx_source_id = compute_mx_source_id(
        p2p_pb2.SourceIdentity(
            mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TILELANG_CACHE,
            model_name="test/model",
        )
    )
    drifted = p2p_pb2.SourceInstanceRef(
        mx_source_id=mx_source_id, worker_id="drift-0", accelerator=""
    )
    mx_client, identity = _artifact_discovery_client(
        [(drifted, _artifact_worker("drift-0:6555", "a-drift", "other"))]
    )

    with pytest.raises(LookupError, match="no ready artifact source"):
        discover_artifact_source(mx_client, identity, accelerator="cuda")


def test_publish_artifact_source_stops_server_when_refresh_fails(
    tmp_path,
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.bin").write_bytes(b"compiled-cache")
    transfer = torch_compile_cache_artifact_transfer(
        source,
        tmp_path / "target",
        tmp_path / "bundle",
        chunk_size=8,
    )
    bundle = transfer.prepare_source()
    identity = p2p_pb2.SourceIdentity(
        mx_version="0.5.0",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        model_name="test/model",
    )
    mx_client = _FakeMxClient()
    source_nixl = _FailingRefreshNixlManager(listen_port=7010)
    worker_server = _FakeWorkerGrpcServer()

    with pytest.raises(RuntimeError, match="refresh failed"):
        publish_artifact_source(
            mx_client,
            transfer,
            bundle,
            identity,
            source_nixl,
            worker_id="source-worker-0",
            worker_grpc_server=worker_server,
            host="127.0.0.1",
        )

    assert mx_client.published_worker is not None
    mx_source_id = compute_mx_source_id(identity)
    assert worker_server.registered == [(mx_source_id, bundle.artifact_id)]
    assert worker_server.unregistered == [(mx_source_id, bundle.artifact_id)]


def test_torch_compile_cache_transfer_discovers_source_through_mx_server(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "bucket-000").mkdir()
    (source / "bucket-001").mkdir()
    (source / "bucket-000" / "kernel-00000.bin").write_bytes(b"compiled-cache-a")
    (source / "bucket-001" / "kernel-00001.bin").write_bytes(b"compiled-cache-b")
    transfer = torch_compile_cache_artifact_transfer(
        source,
        tmp_path / "target",
        tmp_path / "bundle",
        chunk_size=8,
    )
    bundle = transfer.prepare_source()
    identity = p2p_pb2.SourceIdentity(
        mx_version="0.5.0",
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_TORCH_COMPILE_CACHE,
        model_name="test/model",
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        backend_framework_version="vllm-0.10.0",
        torch_version="2.8.0",
        cuda_version="12.8",
        triton_version="3.4.0",
        gpu_arch="sm90",
        compile_config_digest="cache-key",
    )
    mx_source_id = compute_mx_source_id(identity)
    source_bytes_by_path = {
        file.path: Path(file.path).read_bytes()
        for file in bundle.manifest.files
    }
    chunk_manager = _FakeArtifactChunkManager(source_bytes_by_path)
    target_nixl = _FakeTargetNixlManager(chunk_manager)
    servicer = WorkerServiceServicer(
        tensor_protos=[],
        mx_source_id=mx_source_id,
        artifact_manifests={bundle.artifact_id: bundle.manifest},
        artifact_chunk_manager=chunk_manager,
        metadata_endpoint="127.0.0.1:7010",
        agent_name="source-agent",
        worker_rank=0,
    )
    server, port = _start_server(servicer)
    mx_client = _FakeMxClient()
    mx_client.publish_metadata(
        identity,
        p2p_pb2.WorkerMetadata(
            worker_rank=0,
            metadata_endpoint="127.0.0.1:7010",
            agent_name="source-agent",
            worker_grpc_endpoint=f"127.0.0.1:{port}",
            artifact_source=p2p_pb2.ArtifactSourceMetadata(
                artifact_id=bundle.artifact_id,
                total_size=sum(file.size for file in bundle.manifest.files),
                file_count=len(bundle.manifest.files),
                chunk_count=len(bundle.manifest.chunks),
            ),
        ),
        "source-worker-0",
    )

    try:
        discovered = discover_artifact_source(mx_client, identity)
        header = transfer.discover_and_transfer(
            mx_client,
            identity,
            target_nixl,
            timeout=1.0,
            max_inflight_chunks=2,
        )
        assert not transfer.roots[0].target_root.exists()
        transfer.install(header)
    finally:
        server.stop(grace=None)

    assert discovered.mx_source_id == mx_source_id
    assert discovered.artifact_id == bundle.artifact_id
    assert header.artifact_id == bundle.artifact_id
    installed_files = {
        file.relative_to(transfer.roots[0].target_root).as_posix(): file.read_bytes()
        for file in transfer.roots[0].target_root.rglob("*")
        if file.is_file()
    }
    assert installed_files == {
        "bucket-000/kernel-00000.bin": b"compiled-cache-a",
        "bucket-001/kernel-00001.bin": b"compiled-cache-b",
    }


def test_fetch_all_chunks_rejects_non_advancing_page_token(monkeypatch):
    def fake_fetch_artifact_manifest_chunks(*args, **kwargs):
        del args, kwargs
        return (
            p2p_pb2.GetArtifactManifestChunksResponse(
                mx_source_id="source-123",
                artifact_id="artifact",
                start_chunk_index=0,
                next_page_token="0",
            ),
            0,
        )

    monkeypatch.setattr(
        artifact_transfer_module,
        "fetch_artifact_manifest_chunks",
        fake_fetch_artifact_manifest_chunks,
    )

    with pytest.raises(RuntimeError, match="did not advance"):
        artifact_transfer_module._fetch_all_chunks(
            "127.0.0.1:1",
            "source-123",
            "artifact",
            expected_chunk_count=1,
        )


def _artifact_header_from_manifest(
    manifest: p2p_pb2.ArtifactManifest,
    *,
    chunks: list[p2p_pb2.ArtifactManifestChunk] | None = None,
) -> p2p_pb2.GetArtifactManifestHeaderResponse:
    if chunks is None:
        chunks = list(manifest.chunks)
    reconstructed = p2p_pb2.ArtifactManifest(
        manifest_version=manifest.manifest_version,
        mx_source_type=manifest.mx_source_type,
        chunk_size=manifest.chunk_size,
        files=manifest.files,
        chunks=chunks,
    )
    return p2p_pb2.GetArtifactManifestHeaderResponse(
        mx_source_id="source-123",
        artifact_id=artifact_manifest_id(reconstructed),
        manifest_version=manifest.manifest_version,
        mx_source_type=manifest.mx_source_type,
        total_size=sum(file.size for file in manifest.files),
        file_count=len(manifest.files),
        chunk_count=len(chunks),
        chunk_size=manifest.chunk_size,
        files=manifest.files,
    )


def _start_server(servicer) -> tuple[grpc.Server, int]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    p2p_pb2_grpc.add_WorkerServiceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, port


class _FakeArtifactChunkManager:
    def __init__(self, source_bytes_by_path: dict[str, bytes]):
        self._source_bytes_by_path = source_bytes_by_path
        self.buffers: dict[int, bytes] = {}
        self.leases: dict[str, tuple[str, p2p_pb2.ArtifactManifestChunk, int]] = {}
        self.prepared_addrs_by_chunk: dict[int, int] = {}
        self.released_chunks: list[int] = []

    def prepare(
        self,
        manifest: p2p_pb2.ArtifactManifest,
        artifact_id: str,
        chunk: p2p_pb2.ArtifactManifestChunk,
    ) -> tuple[str, p2p_pb2.ArtifactChunkTransferDescriptor, bytes]:
        file = manifest.files[chunk.file_index]
        source = self._source_bytes_by_path[file.path]
        start = chunk.file_offset
        end = start + chunk.length
        addr = chunk.chunk_index + 1
        self.buffers[addr] = source[start:end]
        lease_id = f"lease-{chunk.chunk_index}"
        self.leases[lease_id] = (artifact_id, chunk, addr)
        self.prepared_addrs_by_chunk[chunk.chunk_index] = addr
        return (
            lease_id,
            p2p_pb2.ArtifactChunkTransferDescriptor(
                addr=addr,
                length=chunk.length,
                device_id=0,
            ),
            b"fake-source-metadata",
        )

    def release(self, lease_id: str) -> tuple[str, p2p_pb2.ArtifactManifestChunk]:
        artifact_id, chunk, addr = self.leases.pop(lease_id)
        self.buffers.pop(addr)
        self.released_chunks.append(chunk.chunk_index)
        return artifact_id, chunk


class _LimitedFakeArtifactChunkManager(_FakeArtifactChunkManager):
    def __init__(self, source_bytes_by_path: dict[str, bytes], max_active: int):
        super().__init__(source_bytes_by_path)
        self._max_active = max_active
        self.max_observed_active = 0

    def prepare(
        self,
        manifest: p2p_pb2.ArtifactManifest,
        artifact_id: str,
        chunk: p2p_pb2.ArtifactManifestChunk,
    ) -> tuple[str, p2p_pb2.ArtifactChunkTransferDescriptor, bytes]:
        if len(self.leases) >= self._max_active:
            raise RuntimeError("all artifact transfer buffers are currently leased")
        result = super().prepare(manifest, artifact_id, chunk)
        self.max_observed_active = max(self.max_observed_active, len(self.leases))
        return result


class _FakeTargetNixlManager:
    def __init__(self, source: _FakeArtifactChunkManager):
        self._source = source
        self.loaded_metadata: list[bytes] = []
        self.received_addrs: list[int] = []
        self.registered_sizes: list[int] = []
        self.deregistered_count = 0

    def add_remote_agent(self, source_metadata: bytes) -> str:
        self.loaded_metadata.append(source_metadata)
        return "source-agent"

    def register_dram_buffer(self, buffer):
        self.registered_sizes.append(buffer.numel())
        return object()

    def deregister_memory(self, registered):
        del registered
        self.deregistered_count += 1

    def receive_dram_into_buffer(
        self,
        remote_agent_name: str,
        remote_addr: int,
        local_buffer: torch.Tensor,
        size: int,
        remote_device_id: int = 0,
        remote_mem_type: str = "DRAM",
        timeout_seconds: float | None = None,
    ) -> float:
        del remote_agent_name, remote_device_id, remote_mem_type, timeout_seconds
        self.received_addrs.append(remote_addr)
        data = self._source.buffers[remote_addr]
        assert len(data) == size
        view = memoryview(local_buffer.numpy()).cast("B")
        view[:size] = data
        return 0.0


class _FakeMxClient:
    def __init__(self, mx_source_id: str | None = None):
        self.mx_source_id = mx_source_id
        self.published_identity = None
        self.published_worker = None
        self.published_worker_id = ""
        self.status_updates = []

    def publish_metadata(self, identity, worker, worker_id):
        self.published_identity = p2p_pb2.SourceIdentity()
        self.published_identity.CopyFrom(identity)
        self.published_worker = p2p_pb2.WorkerMetadata()
        self.published_worker.CopyFrom(worker)
        self.published_worker_id = worker_id
        return self.mx_source_id or compute_mx_source_id(identity)

    def list_sources(self, identity=None, status_filter=None):
        assert identity == self.published_identity
        assert status_filter == p2p_pb2.SOURCE_STATUS_READY
        return p2p_pb2.ListSourcesResponse(
            instances=[
                p2p_pb2.SourceInstanceRef(
                    mx_source_id=self.mx_source_id
                    or compute_mx_source_id(self.published_identity),
                    worker_id=self.published_worker_id,
                    model_name=self.published_identity.model_name,
                    worker_rank=self.published_worker.worker_rank,
                )
            ]
        )

    def get_metadata(self, mx_source_id, worker_id):
        assert mx_source_id == (
            self.mx_source_id or compute_mx_source_id(self.published_identity)
        )
        assert worker_id == self.published_worker_id
        worker = p2p_pb2.WorkerMetadata()
        worker.CopyFrom(self.published_worker)
        return p2p_pb2.GetMetadataResponse(
            found=True,
            worker=worker,
            mx_source_id=mx_source_id,
            worker_id=worker_id,
        )

    def update_status(self, mx_source_id, worker_id, worker_rank, status):
        self.status_updates.append((mx_source_id, worker_id, worker_rank, status))
        return True


class _FakeSourceNixlManager:
    def __init__(self, listen_port: int):
        self.agent_name = "source-agent"
        self.nixl_metadata = b"fake-source-metadata"
        self._listen_port = listen_port
        self.buffers = {}
        self.refresh_metadata_count = 0

    def register_dram_buffer(self, buffer):
        self.buffers[buffer.data_ptr()] = buffer
        return object()

    def refresh_agent_metadata(self):
        self.refresh_metadata_count += 1
        return self.nixl_metadata

    def is_healthy(self):
        return True


class _FailingRefreshNixlManager(_FakeSourceNixlManager):
    def refresh_agent_metadata(self):
        raise RuntimeError("refresh failed")


class _FakeWorkerGrpcServer:
    def __init__(self, *args, artifact_chunk_manager=None, **kwargs):
        del args, kwargs
        self.artifact_chunk_manager = artifact_chunk_manager
        self.mx_source_id = None
        self.port = 7100
        self.registered = []
        self.unregistered = []
        self.stopped = False

    def start(self):
        return 7100

    def set_mx_source_id(self, mx_source_id: str):
        self.mx_source_id = mx_source_id

    def register_artifact_source(
        self,
        mx_source_id,
        artifact_id,
        manifest,
        artifact_chunk_manager,
    ):
        del manifest
        self.artifact_chunk_manager = artifact_chunk_manager
        self.registered.append((mx_source_id, artifact_id))

    def unregister_artifact_source(self, mx_source_id, artifact_id):
        self.unregistered.append((mx_source_id, artifact_id))

    def stop(self, grace: float = 5.0):
        del grace
        self.stopped = True
