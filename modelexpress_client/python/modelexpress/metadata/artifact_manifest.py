# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic artifact manifest helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import google_crc32c

from .. import p2p_pb2

ARTIFACT_MANIFEST_VERSION = 1
MAX_ARTIFACT_TRANSFER_CHUNK_SIZE = 64 * 1024 * 1024


def build_artifact_manifest(
    root: str | Path,
    chunk_size: int,
    mx_source_type: int,
) -> p2p_pb2.ArtifactManifest:
    if chunk_size <= 0:
        raise ValueError("artifact manifest chunk_size must be greater than zero")
    if chunk_size > MAX_ARTIFACT_TRANSFER_CHUNK_SIZE:
        raise ValueError(
            "artifact manifest chunk_size "
            f"{chunk_size} exceeds maximum {MAX_ARTIFACT_TRANSFER_CHUNK_SIZE}"
        )

    root_path = Path(root).resolve(strict=True)
    if not root_path.is_dir():
        raise ValueError(f"artifact root is not a directory: {root_path}")

    manifest_files = [
        (_manifest_path(path), path) for path in _collect_regular_files(root_path)
    ]
    manifest_files.sort(key=lambda item: item[0])

    files = []
    chunks = []
    for file_index, (manifest_path, path) in enumerate(manifest_files):
        size = path.stat().st_size
        files.append(
            p2p_pb2.ArtifactManifestFile(
                file_index=file_index,
                path=manifest_path,
                size=size,
                checksum=_file_checksum(path),
            )
        )
        chunks.extend(_chunks_for_file(path, file_index, len(chunks), chunk_size))

    return p2p_pb2.ArtifactManifest(
        manifest_version=ARTIFACT_MANIFEST_VERSION,
        mx_source_type=mx_source_type,
        chunk_size=chunk_size,
        files=files,
        chunks=chunks,
    )


def artifact_manifest_id(manifest: p2p_pb2.ArtifactManifest) -> str:
    canonical = json.dumps(
        _manifest_to_dict(manifest),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def artifact_source_metadata(
    manifest: p2p_pb2.ArtifactManifest,
) -> p2p_pb2.ArtifactSourceMetadata:
    return p2p_pb2.ArtifactSourceMetadata(
        artifact_id=artifact_manifest_id(manifest),
        total_size=sum(file.size for file in manifest.files),
        file_count=len(manifest.files),
        chunk_count=len(manifest.chunks),
    )


def _collect_regular_files(root: Path) -> list[Path]:
    files: list[Path] = []

    def visit(directory: Path) -> None:
        for path in sorted(directory.iterdir()):
            if path.is_symlink():
                raise ValueError(f"artifact manifest does not support symlink {path}")
            if path.is_dir():
                visit(path)
            elif path.is_file():
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(root):
                    raise ValueError(
                        f"artifact file resolves outside artifact root: {path}"
                    )
                files.append(resolved)

    visit(root)
    return files


def _manifest_path(path: Path) -> str:
    if not path.is_absolute():
        raise ValueError(f"artifact manifest path must be absolute: {path}")
    if not path.parts:
        raise ValueError("empty artifact absolute path")
    for part in path.parts[1:]:
        if part in ("", ".", ".."):
            raise ValueError(f"unsafe artifact absolute path {path}")
    return path.as_posix()


def _chunks_for_file(
    path: Path,
    file_index: int,
    first_chunk_index: int,
    chunk_size: int,
) -> Iterable[p2p_pb2.ArtifactManifestChunk]:
    size = path.stat().st_size
    if size == 0:
        yield p2p_pb2.ArtifactManifestChunk(
            chunk_index=first_chunk_index,
            file_index=file_index,
            file_offset=0,
            length=0,
            checksum=_crc32c_hex(b""),
        )
        return

    chunk_index = first_chunk_index
    offset = 0
    with path.open("rb") as file:
        while offset < size:
            data = file.read(min(chunk_size, size - offset))
            if not data:
                break
            yield p2p_pb2.ArtifactManifestChunk(
                chunk_index=chunk_index,
                file_index=file_index,
                file_offset=offset,
                length=len(data),
                checksum=_crc32c_hex(data),
            )
            offset += len(data)
            chunk_index += 1


def _file_checksum(path: Path) -> str:
    checksum = google_crc32c.Checksum()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(64 * 1024), b""):
            checksum.update(chunk)
    digest = checksum.hexdigest()
    if isinstance(digest, bytes):
        digest = digest.decode("ascii")
    return digest.lower()


def _crc32c_hex(data: bytes) -> str:
    return f"{google_crc32c.value(data):08x}"


def _manifest_to_dict(manifest: p2p_pb2.ArtifactManifest) -> dict:
    return {
        "manifest_version": manifest.manifest_version,
        "mx_source_type": manifest.mx_source_type,
        "chunk_size": manifest.chunk_size,
        "files": [
            {
                "file_index": file.file_index,
                "path": file.path,
                "size": file.size,
                "checksum": file.checksum,
            }
            for file in manifest.files
        ],
        "chunks": [
            {
                "chunk_index": chunk.chunk_index,
                "file_index": chunk.file_index,
                "file_offset": chunk.file_offset,
                "length": chunk.length,
                "checksum": chunk.checksum,
            }
            for chunk in manifest.chunks
        ],
    }
