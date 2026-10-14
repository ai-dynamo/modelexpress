# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tarfile
from pathlib import Path

import pytest

import modelexpress.metadata.artifact_transfer as artifact_transfer_module
from modelexpress import p2p_pb2
from modelexpress.metadata.artifact_symlinks import METADATA_NAME, read_links
from modelexpress.metadata.artifact_transfer import (
    flashinfer_cache_artifact_transfer,
    tilelang_cache_artifact_transfer,
    triton_cache_artifact_transfer,
)


def _bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "headers").mkdir()
    (source / "headers" / "header.h").write_text("header")
    transfer = flashinfer_cache_artifact_transfer(source, tmp_path / "target", tmp_path / "bundle")
    return source, transfer


def _install(transfer):
    bundle = transfer.prepare_source()
    header = p2p_pb2.GetArtifactManifestHeaderResponse(files=bundle.manifest.files)
    transfer.install(header)
    return bundle, header


def _append_metadata(tar_path, links):
    payload = json.dumps({"version": 1, "links": links}).encode()
    with tarfile.open(tar_path, "a") as archive:
        member = tarfile.TarInfo(METADATA_NAME)
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))


def test_relative_directory_link_and_file_link_chain(tmp_path):
    source, transfer = _bundle(tmp_path)
    (source / "include").symlink_to("headers", target_is_directory=True)
    (source / "header.h").symlink_to("include/header.h")
    bundle, header = _install(transfer)
    target = transfer.roots[0].target_root
    assert (target / "header.h").read_text() == "header"
    assert (target / "include").readlink() == Path("headers")
    with tarfile.open(bundle.tar_path) as archive:
        assert METADATA_NAME in archive.getnames()
        assert all(member.isfile() or member.isdir() for member in archive)
    transfer.install(header)
    assert (target / "header.h").read_text() == "header"
    assert not (target / METADATA_NAME).exists()


@pytest.mark.parametrize(
    "transfer_factory",
    [triton_cache_artifact_transfer, tilelang_cache_artifact_transfer],
)
def test_other_cache_artifacts_restore_symlinks(tmp_path, transfer_factory):
    source = tmp_path / "source"
    source.mkdir()
    (source / "kernel.bin").write_bytes(b"compiled")
    (source / "kernel-link.bin").symlink_to("kernel.bin")
    target = tmp_path / "target"
    transfer = transfer_factory(source, target, tmp_path / "bundle")

    bundle = transfer.prepare_source()
    with tarfile.open(bundle.tar_path) as archive:
        assert METADATA_NAME in archive.getnames()
    transfer.install(p2p_pb2.GetArtifactManifestHeaderResponse(files=bundle.manifest.files))

    assert (target / "kernel-link.bin").is_symlink()
    assert (target / "kernel-link.bin").read_bytes() == b"compiled"
    assert not (target / METADATA_NAME).exists()


def test_absolute_internal_link_does_not_depend_on_source_cache(tmp_path):
    source, transfer = _bundle(tmp_path)
    (source / "header.h").symlink_to(source / "headers" / "header.h")
    _install(transfer)
    source.rename(tmp_path / "retired-source")
    assert (transfer.roots[0].target_root / "header.h").read_text() == "header"


def test_cycle_is_rejected_before_extracting_regular_files(tmp_path):
    source, transfer = _bundle(tmp_path)
    (source / "a").symlink_to("b")
    (source / "b").symlink_to("a")
    with pytest.raises(ValueError, match="cyclic"):
        _install(transfer)
    assert not (transfer.roots[0].target_root / "headers" / "header.h").exists()


def test_external_target_missing_on_receiver_rejects_install(tmp_path):
    source, transfer = _bundle(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    (source / "include").symlink_to(external)
    bundle = transfer.prepare_source()
    external.rename(tmp_path / "retired-external")
    with pytest.raises(ValueError, match="source missing"):
        transfer.install(p2p_pb2.GetArtifactManifestHeaderResponse(files=bundle.manifest.files))
    assert not (transfer.roots[0].target_root / "headers" / "header.h").exists()


def test_source_cannot_use_reserved_metadata_name(tmp_path):
    source, transfer = _bundle(tmp_path)
    (source / METADATA_NAME).write_text("user data")
    with pytest.raises(ValueError, match="reserved metadata"):
        transfer.prepare_source()


@pytest.mark.parametrize("indirect", [False, True])
def test_source_link_cannot_target_reserved_metadata(tmp_path, indirect):
    source, transfer = _bundle(tmp_path)
    (source / "final").symlink_to(METADATA_NAME)
    if indirect:
        (source / "first").symlink_to("final")

    with pytest.raises(ValueError, match="targets reserved metadata"):
        transfer.prepare_source()


@pytest.mark.parametrize("indirect", [False, True])
def test_received_link_cannot_target_reserved_metadata(tmp_path, indirect):
    _, transfer = _bundle(tmp_path)
    bundle = transfer.prepare_source()
    links = [{"path": "final", "target": METADATA_NAME}]
    if indirect:
        links.insert(0, {"path": "first", "target": "final"})
    _append_metadata(bundle.tar_path, links)

    with pytest.raises(ValueError, match="targets reserved metadata"):
        transfer.install(p2p_pb2.GetArtifactManifestHeaderResponse(files=bundle.manifest.files))
    assert not (transfer.roots[0].target_root / "headers" / "header.h").exists()


def test_no_metadata_preserves_existing_target_file(tmp_path):
    _, transfer = _bundle(tmp_path)
    target = transfer.roots[0].target_root
    target.mkdir()
    existing = target / METADATA_NAME
    existing.write_text("keep")
    bundle = transfer.prepare_source()
    assert read_links(bundle.tar_path, target) is None

    transfer.install(p2p_pb2.GetArtifactManifestHeaderResponse(files=bundle.manifest.files))

    assert existing.read_text() == "keep"
    assert (target / "headers" / "header.h").read_text() == "header"


def test_empty_metadata_is_removed_after_install(tmp_path):
    _, transfer = _bundle(tmp_path)
    bundle = transfer.prepare_source()
    _append_metadata(bundle.tar_path, [])
    target = transfer.roots[0].target_root
    assert read_links(bundle.tar_path, target) == {}

    transfer.install(p2p_pb2.GetArtifactManifestHeaderResponse(files=bundle.manifest.files))

    assert not (target / METADATA_NAME).exists()
    assert (target / "headers" / "header.h").read_text() == "header"


def test_existing_metadata_file_rejects_sidecar_before_extract(tmp_path):
    source, transfer = _bundle(tmp_path)
    (source / "include").symlink_to("headers")
    target = transfer.roots[0].target_root
    target.mkdir()
    existing = target / METADATA_NAME
    existing.write_text("keep")

    with pytest.raises(ValueError, match="metadata destination"):
        _install(transfer)

    assert existing.read_text() == "keep"
    assert not (target / "headers" / "header.h").exists()


def test_restore_failure_rolls_back_links_and_removes_sidecar(tmp_path, monkeypatch):
    source, transfer = _bundle(tmp_path)
    external = tmp_path / "external"
    external.write_text("header")
    (source / "external-link").symlink_to(external)
    bundle = transfer.prepare_source()
    run_tar = artifact_transfer_module._run_tar

    def extract_then_remove_target(args):
        run_tar(args)
        if args[0] == "-xf":
            external.unlink()

    monkeypatch.setattr(artifact_transfer_module, "_run_tar", extract_then_remove_target)
    with pytest.raises(ValueError, match="target disappeared"):
        transfer.install(p2p_pb2.GetArtifactManifestHeaderResponse(files=bundle.manifest.files))

    target = transfer.roots[0].target_root
    assert not (target / "external-link").is_symlink()
    assert not (target / METADATA_NAME).exists()


def test_extract_failure_removes_extracted_sidecar(tmp_path, monkeypatch):
    source, transfer = _bundle(tmp_path)
    (source / "include").symlink_to("headers")
    bundle = transfer.prepare_source()
    run_tar = artifact_transfer_module._run_tar

    def extract_then_fail(args):
        run_tar(args)
        if args[0] == "-xf":
            raise RuntimeError("simulated extraction failure")

    monkeypatch.setattr(artifact_transfer_module, "_run_tar", extract_then_fail)
    with pytest.raises(RuntimeError, match="simulated extraction failure"):
        transfer.install(p2p_pb2.GetArtifactManifestHeaderResponse(files=bundle.manifest.files))

    target = transfer.roots[0].target_root
    assert not (target / METADATA_NAME).exists()
    assert not (target / "include").is_symlink()


@pytest.mark.parametrize("metadata", [
    [], {"version": 2, "links": []}, {"version": 1, "links": None},
    {"version": 1, "links": [None]},
    {"version": 1, "links": [{"path": "../escape", "target": "/tmp"}]},
    {"version": 1, "links": [{"path": "/escape", "target": "/tmp"}]},
    {"version": 1, "links": [{"path": "a/./b", "target": "/tmp"}]},
    {"version": 1, "links": [{"path": METADATA_NAME, "target": "/tmp"}]},
    {"version": 1, "links": [{"path": "a", "target": ""}]},
    {"version": 1, "links": [{"path": "a", "target": "\0"}]},
    {"version": 1, "links": [{"path": "a", "target": "/tmp"}] * 2},
    {"version": 1, "links": [{"path": "a", "target": "/tmp"}, {"path": "a/b", "target": "/tmp"}]},
])
def test_invalid_metadata_is_rejected(tmp_path, metadata):
    archive_path = tmp_path / "artifact.tar"
    payload = json.dumps(metadata).encode()
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo(METADATA_NAME)
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(ValueError):
        read_links(archive_path, target)


def test_existing_symlink_parent_is_rejected_without_touching_external_files(tmp_path):
    source, transfer = _bundle(tmp_path)
    (source / "nested").mkdir()
    (source / "nested" / "link").symlink_to("../headers/header.h")
    target = transfer.roots[0].target_root
    target.mkdir()
    external = tmp_path / "outside"
    external.mkdir()
    (target / "nested").symlink_to(external)
    with pytest.raises(ValueError, match="unsafe.*parent"):
        _install(transfer)
    assert not list(external.iterdir())


def test_metadata_destination_symlink_is_rejected_before_extract(tmp_path):
    source, transfer = _bundle(tmp_path)
    (source / "include").symlink_to("headers")
    target = transfer.roots[0].target_root
    target.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("keep")
    (target / METADATA_NAME).symlink_to(outside)
    with pytest.raises(ValueError, match="metadata destination"):
        _install(transfer)
    assert outside.read_text() == "keep"
