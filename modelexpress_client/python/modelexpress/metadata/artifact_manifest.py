# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic artifact manifest helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import google_crc32c

from .. import p2p_pb2

ARTIFACT_MANIFEST_VERSION = 1
MAX_ARTIFACT_TRANSFER_CHUNK_SIZE = 4 * 1024 * 1024 * 1024
DEFAULT_ARTIFACT_TRANSFER_CHUNK_SIZE = 64 * 1024 * 1024
MX_ARTIFACT_TRANSFER_CHUNK_SIZE_ENV = "MX_ARTIFACT_TRANSFER_CHUNK_SIZE"

logger = logging.getLogger("modelexpress.metadata.artifact_manifest")


def build_artifact_manifest(
    root: str | Path,
    *,
    chunk_size: int | None = None,
    mx_source_type: int,
) -> p2p_pb2.ArtifactManifest:
    if chunk_size is None:
        chunk_size = artifact_transfer_chunk_size()
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
        file_checksum, file_chunks = _checksums_for_file(
            path,
            file_index,
            len(chunks),
            chunk_size,
        )
        files.append(
            p2p_pb2.ArtifactManifestFile(
                file_index=file_index,
                path=manifest_path,
                size=size,
                checksum=file_checksum,
            )
        )
        chunks.extend(file_chunks)

    return p2p_pb2.ArtifactManifest(
        manifest_version=ARTIFACT_MANIFEST_VERSION,
        mx_source_type=mx_source_type,
        chunk_size=chunk_size,
        files=files,
        chunks=chunks,
    )


def artifact_transfer_chunk_size(
    default: int = DEFAULT_ARTIFACT_TRANSFER_CHUNK_SIZE,
) -> int:
    raw = os.environ.get(MX_ARTIFACT_TRANSFER_CHUNK_SIZE_ENV)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid artifact transfer chunk size %r; using %d",
            raw,
            default,
        )
        return default
    if value <= 0:
        logger.warning(
            "Ignoring non-positive artifact transfer chunk size %d; using %d",
            value,
            default,
        )
        return default
    return value


def artifact_manifest_id(manifest: p2p_pb2.ArtifactManifest) -> str:
    # Compact UTF-8 JSON is the canonical byte representation for artifact_id.
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
                logger.warning("Skipping artifact symlink %s", path)
                continue
            if path.is_dir():
                visit(path)
            elif path.is_file():
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(root):
                    logger.warning(
                        "Skipping artifact file outside artifact root: %s",
                        path,
                    )
                    continue
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


def _checksums_for_file(
    path: Path,
    file_index: int,
    first_chunk_index: int,
    chunk_size: int,
) -> tuple[str, list[p2p_pb2.ArtifactManifestChunk]]:
    size = path.stat().st_size
    if size == 0:
        del first_chunk_index, file_index
        return _crc32c_hex(b""), []

    chunks = []
    file_crc = google_crc32c.Checksum()
    chunk_index = first_chunk_index
    offset = 0
    with path.open("rb") as file:
        while offset < size:
            data = file.read(min(chunk_size, size - offset))
            if not data:
                break
            file_crc.update(data)
            chunks.append(
                p2p_pb2.ArtifactManifestChunk(
                    chunk_index=chunk_index,
                    file_index=file_index,
                    file_offset=offset,
                    length=len(data),
                    checksum=_crc32c_hex(data),
                )
            )
            offset += len(data)
            chunk_index += 1
    return file_crc.hexdigest().decode("ascii"), chunks


def _crc32c_hex(data) -> str:
    if isinstance(data, memoryview):
        data = data.tobytes()
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
