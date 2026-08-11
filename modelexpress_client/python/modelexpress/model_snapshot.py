# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face cache layout for model files streamed from ModelExpress Server.

Streamed files carry paths relative to the server's snapshot directory. This
module turns them into a cache an engine can resolve while offline::

    <cache_root>/models--<org>--<name>/
        refs/main            commit hash of the published snapshot
        snapshots/<commit>/  the files themselves

``refs/main`` is what lets ``snapshot_download(local_files_only=True)`` resolve
a repo id. Without it the engine raises ``LocalEntryNotFoundError`` even when
every file is already on disk.

There are two write paths because their atomicity requirements differ:

- :class:`SnapshotStaging` builds a snapshot out of band and publishes the
  whole directory with a single rename. Use it before the engine starts.
- :class:`SnapshotPatch` adds files to a snapshot the engine has already
  resolved, renaming one file at a time so the directory is never swapped
  out from under it.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from typing import Iterator, Mapping

from huggingface_hub.constants import HF_HUB_CACHE

from . import envs

logger = logging.getLogger("modelexpress.model_snapshot")

# Mirrors ModelProviderExt::is_weight_file in
# modelexpress_common/src/providers.rs. The server uses that list to decide
# what `ignore_weights` skips, so the two must stay in sync.
WEIGHT_FILE_SUFFIXES = (
    ".bin",
    ".safetensors",
    ".h5",
    ".msgpack",
    ".ckpt.index",
    ".iop",
    ".gas",
)

MAIN_REF = "main"

_LOCK_FILE = ".modelexpress.lock"
_STAGING_PREFIX = ".modelexpress-staging-"
_STALE_PREFIX = ".modelexpress-stale-"
_TEMP_PREFIX = ".modelexpress-tmp-"


class ModelSnapshotError(RuntimeError):
    """Raised when server-provided paths or the local cache cannot be trusted."""


def is_weight_file(relative_path: str) -> bool:
    """Return whether a repo-relative path holds model weights."""
    return relative_path.endswith(WEIGHT_FILE_SUFFIXES)


def split_by_weight(paths) -> tuple[list[str], list[str]]:
    """Split repo-relative paths into (metadata, weights), preserving order."""
    metadata: list[str] = []
    weights: list[str] = []
    for path in paths:
        (weights if is_weight_file(path) else metadata).append(path)
    return metadata, weights


def safe_relative_path(relative_path: str) -> Path:
    """Validate a server-provided path and return it as a relative Path."""
    if (
        not relative_path
        or "\x00" in relative_path
        or "\\" in relative_path
        or relative_path.startswith("/")
    ):
        raise ModelSnapshotError(f"Unsafe model file path: {relative_path!r}")
    parts = relative_path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ModelSnapshotError(f"Unsafe model file path: {relative_path!r}")
    return Path(*parts)


def safe_commit_hash(commit_hash: str) -> str:
    """Validate a server-provided commit hash used as a directory name."""
    if (
        not commit_hash
        or commit_hash in (".", "..")
        or "\x00" in commit_hash
        or "/" in commit_hash
        or "\\" in commit_hash
    ):
        raise ModelSnapshotError(f"Unsafe commit hash: {commit_hash!r}")
    return commit_hash


def repo_dir_name(model_name: str) -> str:
    """Return the Hugging Face cache directory name for a model id."""
    if (
        not model_name
        or "\x00" in model_name
        or "\\" in model_name
        or model_name.startswith("/")
    ):
        raise ValueError(f"Invalid Hugging Face model name: {model_name!r}")
    parts = model_name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Invalid Hugging Face model name: {model_name!r}")
    return f"models--{'--'.join(parts)}"


def resolve_cache_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the local cache root.

    Priority: explicit argument, ``MODEL_EXPRESS_CACHE_DIRECTORY``, then
    huggingface_hub's own ``HF_HUB_CACHE``.
    """
    if explicit is not None:
        return Path(explicit).expanduser()
    configured = envs.MODEL_EXPRESS_CACHE_DIRECTORY
    if configured:
        return Path(configured).expanduser()
    return Path(HF_HUB_CACHE).expanduser()


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_contained(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root)
    except OSError:
        return False


def _ensure_directory(directory: Path, cache_root: Path) -> None:
    if directory.is_symlink():
        raise ModelSnapshotError(f"Refusing to use symlinked cache directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    if not _is_contained(directory, cache_root):
        raise ModelSnapshotError(f"Cache directory resolves outside the cache root: {directory}")


class SnapshotSink:
    """Writes one streamed file at a time below ``root``."""

    def __init__(self, root: Path, cache_root: Path):
        self._root = root
        self._cache_root = cache_root
        self._handle = None
        self._target: Path | None = None
        self._relative_path: str | None = None

    @property
    def current_file(self) -> str | None:
        """Repo-relative path of the file currently open, if any."""
        return self._relative_path

    def begin_file(self, relative_path: str) -> None:
        """Open ``relative_path`` for writing."""
        if self._handle is not None:
            raise ModelSnapshotError(
                f"Cannot start {relative_path!r} while {self._relative_path!r} is open"
            )
        target = self._root / safe_relative_path(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not _is_contained(target.parent, self._cache_root):
            raise ModelSnapshotError(f"File path resolves outside the cache root: {target}")
        self._target = target
        self._relative_path = relative_path
        self._handle = self._open(target)

    def write(self, data: bytes) -> None:
        """Append a chunk to the open file."""
        if self._handle is None:
            raise ModelSnapshotError("No model file is open for writing")
        written = self._handle.write(data)
        if written != len(data):
            raise ModelSnapshotError(
                f"Short local write for {self._relative_path!r}: "
                f"wrote {written} of {len(data)} bytes"
            )

    def end_file(self) -> None:
        """Flush, sync and finalize the open file."""
        if self._handle is None or self._target is None:
            raise ModelSnapshotError("No model file is open for writing")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._handle = None
        self._finalize(self._target)
        self._target = None
        self._relative_path = None

    def close(self) -> None:
        """Drop a partially written file. Safe to call more than once."""
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None
        if self._target is not None:
            self._discard(self._target)
        self._target = None
        self._relative_path = None

    def _open(self, target: Path):
        raise NotImplementedError

    def _finalize(self, target: Path) -> None:
        raise NotImplementedError

    def _discard(self, target: Path) -> None:
        raise NotImplementedError


class SnapshotStaging(SnapshotSink):
    """Collects a snapshot in a staging directory, then publishes it atomically."""

    def __init__(self, cache: "ModelSnapshotCache"):
        _ensure_directory(cache.repo_root, cache.cache_root)
        staging_path = Path(
            tempfile.mkdtemp(prefix=_STAGING_PREFIX, dir=cache.repo_root)
        )
        super().__init__(staging_path, cache.cache_root)
        self._cache = cache
        self._staging_path: Path | None = staging_path

    @property
    def path(self) -> Path:
        """Staging directory backing this snapshot."""
        if self._staging_path is None:
            raise ModelSnapshotError("Staging directory has already been consumed")
        return self._staging_path

    def publish(self, commit_hash: str, expected_files: Mapping[str, int]) -> Path:
        """Move the staged files into ``snapshots/<commit>`` and update refs/main."""
        staging_path = self.path
        commit_hash = safe_commit_hash(commit_hash)
        snapshots_root = self._cache.repo_root / "snapshots"
        _ensure_directory(snapshots_root, self._cache.cache_root)
        snapshot_path = snapshots_root / commit_hash

        if self._cache.has_files(snapshot_path, expected_files):
            logger.info(
                "Snapshot %s already complete, discarding staged copy", snapshot_path
            )
            shutil.rmtree(staging_path, ignore_errors=True)
            self._staging_path = None
            self._cache.write_main_ref(commit_hash)
            return snapshot_path

        if snapshot_path.is_dir() and not snapshot_path.is_symlink():
            # Same commit means same content, so the directory already on disk
            # holds files this manifest never mentions -- weights, above all.
            # Merge into it rather than swapping it out, or installing metadata
            # would delete a weight set nothing here checks for.
            self._merge_into(snapshot_path)
            self._cache.write_main_ref(commit_hash)
            self._staging_path = None
            return snapshot_path

        stale_path: Path | None = None
        if snapshot_path.exists() or snapshot_path.is_symlink():
            stale_path = self._cache.repo_root / f"{_STALE_PREFIX}{uuid.uuid4().hex}"
            os.replace(snapshot_path, stale_path)

        try:
            os.replace(staging_path, snapshot_path)
            _fsync_directory(snapshots_root)
            self._cache.write_main_ref(commit_hash)
        except Exception:
            if stale_path is not None and not snapshot_path.exists():
                os.replace(stale_path, snapshot_path)
            raise
        self._staging_path = None

        if stale_path is not None:
            try:
                shutil.rmtree(stale_path)
            except OSError:
                logger.warning("Failed to clean up stale snapshot %s", stale_path)
        return snapshot_path

    def _merge_into(self, snapshot_path: Path) -> None:
        """Move every staged file into an existing snapshot, one rename at a time.

        Staging and the snapshot share a filesystem, so each rename is atomic:
        a reader sees either the old file or the new one, never a partial write.
        """
        staging_path = self.path
        touched_dirs: set[Path] = set()
        for source in sorted(staging_path.rglob("*")):
            if source.is_dir():
                continue
            target = snapshot_path / source.relative_to(staging_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not _is_contained(target.parent, self._cache.cache_root):
                raise ModelSnapshotError(
                    f"Staged file resolves outside the cache root: {target}"
                )
            os.replace(source, target)
            touched_dirs.add(target.parent)
        for directory in touched_dirs:
            _fsync_directory(directory)
        shutil.rmtree(staging_path, ignore_errors=True)

    def discard(self) -> None:
        """Remove the staging directory if it was never published."""
        self.close()
        if self._staging_path is not None:
            shutil.rmtree(self._staging_path, ignore_errors=True)
            self._staging_path = None

    def _open(self, target: Path):
        return target.open("xb")

    def _finalize(self, target: Path) -> None:
        return None

    def _discard(self, target: Path) -> None:
        target.unlink(missing_ok=True)


class SnapshotPatch(SnapshotSink):
    """Adds files to a published snapshot one atomic rename at a time."""

    def __init__(self, cache: "ModelSnapshotCache", snapshot_path: Path):
        if not snapshot_path.is_dir():
            raise ModelSnapshotError(f"Snapshot directory does not exist: {snapshot_path}")
        if not _is_contained(snapshot_path, cache.cache_root):
            raise ModelSnapshotError(
                f"Snapshot resolves outside the cache root: {snapshot_path}"
            )
        super().__init__(snapshot_path, cache.cache_root)
        self._temp_paths: dict[Path, Path] = {}
        self._published: list[Path] = []

    def rollback(self) -> None:
        """Undo this patch, leaving the snapshot as it was before it started.

        A half-applied patch is worse than none: the engine would see a subset
        of the weights and load it as if it were complete.
        """
        self.close()
        while self._published:
            self._published.pop().unlink(missing_ok=True)

    def _open(self, target: Path):
        temp_path = target.parent / f"{_TEMP_PREFIX}{uuid.uuid4().hex}-{target.name}"
        self._temp_paths[target] = temp_path
        return temp_path.open("xb")

    def _finalize(self, target: Path) -> None:
        temp_path = self._temp_paths.pop(target)
        os.replace(temp_path, target)
        _fsync_directory(target.parent)
        self._published.append(target)

    def _discard(self, target: Path) -> None:
        temp_path = self._temp_paths.pop(target, None)
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


class ModelSnapshotCache:
    """One Hugging Face repo directory inside a local cache root."""

    def __init__(
        self,
        model_name: str,
        cache_root: str | os.PathLike[str] | None = None,
    ):
        self.model_name = model_name
        root = resolve_cache_root(cache_root)
        root.mkdir(parents=True, exist_ok=True)
        self.cache_root = root.resolve()
        self.repo_root = self.cache_root / repo_dir_name(model_name)

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Serialize cache writes across the workers sharing this directory."""
        _ensure_directory(self.repo_root, self.cache_root)
        lock_path = self.repo_root / _LOCK_FILE
        with lock_path.open("a", encoding="utf-8") as handle:
            flock(handle.fileno(), LOCK_EX)
            try:
                yield
            finally:
                flock(handle.fileno(), LOCK_UN)

    def snapshot_path(self, commit_hash: str) -> Path:
        """Return the directory a given commit's snapshot lives in."""
        return self.repo_root / "snapshots" / safe_commit_hash(commit_hash)

    def read_main_ref(self) -> str | None:
        """Return the commit hash refs/main points at, or None."""
        ref_path = self.repo_root / "refs" / MAIN_REF
        if not ref_path.is_file() or ref_path.is_symlink():
            return None
        try:
            return safe_commit_hash(ref_path.read_text(encoding="utf-8").strip())
        except (OSError, UnicodeError, ModelSnapshotError):
            return None

    def write_main_ref(self, commit_hash: str) -> None:
        """Point refs/main at ``commit_hash``, replacing any previous value."""
        commit_hash = safe_commit_hash(commit_hash)
        refs_root = self.repo_root / "refs"
        _ensure_directory(refs_root, self.cache_root)
        temp_ref = refs_root / f"{_TEMP_PREFIX}{uuid.uuid4().hex}"
        try:
            with temp_ref.open("x", encoding="utf-8") as ref_file:
                ref_file.write(commit_hash)
                ref_file.flush()
                os.fsync(ref_file.fileno())
            os.replace(temp_ref, refs_root / MAIN_REF)
            _fsync_directory(refs_root)
        finally:
            temp_ref.unlink(missing_ok=True)

    def has_files(self, snapshot_path: Path, expected_files: Mapping[str, int]) -> bool:
        """Return whether every expected file is present at its expected size."""
        if not snapshot_path.is_dir():
            return False
        try:
            if not _is_contained(snapshot_path, self.cache_root):
                return False
            for relative_path, expected_size in expected_files.items():
                file_path = snapshot_path / safe_relative_path(relative_path)
                if not file_path.is_file():
                    return False
                if not _is_contained(file_path, self.cache_root):
                    return False
                if file_path.stat().st_size != expected_size:
                    return False
        except (OSError, ModelSnapshotError):
            return False
        return True

    def resolve_snapshot(self, expected_files: Mapping[str, int]) -> Path | None:
        """Return the snapshot refs/main points at when it holds every file."""
        commit_hash = self.read_main_ref()
        if commit_hash is None:
            return None
        snapshot_path = self.snapshot_path(commit_hash)
        if self.has_files(snapshot_path, expected_files):
            return snapshot_path
        return None

    def staging(self) -> SnapshotStaging:
        """Open a staging directory for a fresh snapshot."""
        return SnapshotStaging(self)

    def patch(self, snapshot_path: Path) -> SnapshotPatch:
        """Open a writer that adds files to an already published snapshot."""
        return SnapshotPatch(self, snapshot_path)
