# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated sidecar metadata for links in file-backed cache archives."""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path, PurePosixPath

METADATA_NAME = "mx.symlink.metadata.v1"
_MAX_METADATA_BYTES = 4 * 1024 * 1024


def encode_links(source_root: Path, links: list[Path]) -> bytes:
    entries = []
    reserved_path = source_root.resolve() / METADATA_NAME
    for link in links:
        try:
            resolved = link.resolve(strict=False)
        except RuntimeError as exc:
            raise ValueError(f"cyclic artifact symlink: {link}") from exc
        if resolved == reserved_path:
            raise ValueError(f"artifact symlink targets reserved metadata: {link}")
        target = os.readlink(link)
        # Absolute links within a cache must follow that cache when relocated.
        if os.path.isabs(target):
            normalized = Path(os.path.abspath(target))
            if normalized.is_relative_to(source_root):
                target = os.path.relpath(normalized, link.parent)
        entries.append({"path": link.relative_to(source_root).as_posix(), "target": target})
    payload = json.dumps(
        {"version": 1, "links": entries}, sort_keys=True, separators=(",", ":")
    ).encode()
    if len(payload) > _MAX_METADATA_BYTES:
        raise ValueError("artifact symlink metadata exceeds size limit")
    return payload


def read_links(tar_path: Path, target_root: Path) -> dict[str, str] | None:
    """Validate destinations and final targets before extracting an archive."""
    with tarfile.open(tar_path, "r:") as archive:
        members = archive.getmembers()
        metadata = [m for m in members if str(PurePosixPath(m.name)) == METADATA_NAME]
        if not metadata:
            return None
        if len(metadata) != 1 or not metadata[0].isfile():
            raise ValueError("invalid artifact symlink metadata member")
        if metadata[0].size > _MAX_METADATA_BYTES:
            raise ValueError("artifact symlink metadata exceeds size limit")
        destination = target_root / METADATA_NAME
        if destination.is_symlink() or destination.exists():
            raise ValueError("unsafe artifact symlink metadata destination")
        handle = archive.extractfile(metadata[0])
        if handle is None:
            raise ValueError("cannot read artifact symlink metadata")
        try:
            data = json.loads(handle.read())
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid artifact symlink metadata") from exc
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("unsupported artifact symlink metadata version")
        entries = data.get("links")
        if not isinstance(entries, list):
            raise ValueError("invalid artifact symlink entries")
        archive_paths = {str(PurePosixPath(m.name)) for m in members}

    links: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid artifact symlink entry")
        relative, target = entry.get("path"), entry.get("target")
        if not isinstance(relative, str) or not isinstance(target, str) or not target:
            raise ValueError("invalid artifact symlink entry")
        parts = PurePosixPath(relative).parts
        if (
            not parts or relative.startswith("/") or ".." in parts
            or "\0" in relative or "\0" in target
            or str(PurePosixPath(relative)) != relative or relative == METADATA_NAME
        ):
            raise ValueError(f"unsafe artifact symlink path: {relative!r}")
        if relative in links or relative in archive_paths:
            raise ValueError(f"duplicate or conflicting artifact symlink: {relative}")
        destination = target_root / relative
        parent = destination.parent
        while parent != target_root:
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise ValueError(f"unsafe artifact symlink parent: {parent}")
            parent = parent.parent
        if destination.is_symlink():
            if os.readlink(destination) != target:
                raise ValueError(f"artifact symlink destination already exists: {destination}")
        elif destination.exists():
            raise ValueError(f"artifact symlink destination already exists: {destination}")
        links[relative] = target

    for relative in links:
        prefix = relative + "/"
        if any(path.startswith(prefix) for path in archive_paths | links.keys()):
            raise ValueError(f"artifact member has a symlink ancestor: {relative}")

    def resolve(path: Path, visiting: frozenset[str]) -> Path:
        path = Path(os.path.abspath(path))
        if not path.is_relative_to(target_root):
            return path
        parts = path.relative_to(target_root).parts
        for index in range(1, len(parts) + 1):
            prefix = PurePosixPath(*parts[:index]).as_posix()
            if prefix not in links:
                continue
            if prefix in visiting or len(visiting) >= 64:
                raise ValueError(f"cyclic artifact symlink: {prefix}")
            replacement = Path(links[prefix])
            if not replacement.is_absolute():
                replacement = (target_root / prefix).parent / replacement
            return resolve(replacement.joinpath(*parts[index:]), visiting | {prefix})
        return path

    reserved_target = Path(os.path.abspath(target_root / METADATA_NAME))
    for relative in links:
        resolved = resolve(target_root / relative, frozenset())
        if resolved == reserved_target:
            raise ValueError(f"artifact symlink targets reserved metadata: {relative}")
        if resolved.is_relative_to(target_root):
            archive_target = resolved.relative_to(target_root).as_posix()
            if archive_target in archive_paths:
                continue
        try:
            exists = resolved.exists()
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"invalid artifact symlink target: {relative}") from exc
        if not exists:
            raise ValueError(f"artifact symlink source missing: {relative} -> {resolved}")
    return links


def restore_links(target_root: Path, links: dict[str, str] | None) -> int:
    """Restore validated links only after regular members have been extracted."""
    if links is None:
        return 0
    created: list[Path] = []
    try:
        for relative, target in links.items():
            destination = target_root / relative
            if destination.is_symlink() and os.readlink(destination) == target:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(target)
            created.append(destination)
        if any(not (target_root / relative).exists() for relative in links):
            raise ValueError("artifact symlink target disappeared during installation")
    except Exception:
        for destination in reversed(created):
            destination.unlink(missing_ok=True)
        raise
    finally:
        (target_root / METADATA_NAME).unlink(missing_ok=True)
    return len(links)
