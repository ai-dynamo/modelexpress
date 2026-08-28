# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hugging Face cache layout for model files streamed from ModelExpress Server.

Streamed files carry paths relative to the server's snapshot directory. This
module turns them into a cache an engine can resolve while offline::

    <cache_root>/models--<org>--<name>/
        refs/<revision>      commit hash the engine's revision resolves to
        snapshots/<commit>/  the files themselves

A ref is what lets ``snapshot_download(local_files_only=True)`` resolve a
revision that is not itself a commit hash -- a branch, a tag, or no revision
at all, which resolves through ``refs/main``. Without it the engine raises
``LocalEntryNotFoundError`` even when every file is already on disk. A request
pinned to a commit hash needs no ref: it resolves by snapshot directory name.

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
_BACKUP_PREFIX = ".modelexpress-backup-"


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

    def publish(
        self,
        commit_hash: str,
        expected_files: Mapping[str, int],
        requested_revision: str | None = None,
    ) -> Path:
        """Move the staged files into ``snapshots/<commit>`` and record its ref.

        ``requested_revision`` is what the engine asked for, which decides
        which ref the snapshot becomes reachable under. See
        :meth:`ModelSnapshotCache.write_revision_ref`.
        """
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
            self._cache.write_revision_ref(commit_hash, requested_revision)
            return snapshot_path

        if snapshot_path.is_dir() and not snapshot_path.is_symlink():
            # Same commit means same content, so the directory already on disk
            # holds files this manifest never mentions -- weights, above all.
            # Merge into it rather than swapping it out, or installing metadata
            # would delete a weight set nothing here checks for.
            self._merge_into(snapshot_path)
            self._cache.write_revision_ref(commit_hash, requested_revision)
            self._staging_path = None
            return snapshot_path

        stale_path: Path | None = None
        if snapshot_path.exists() or snapshot_path.is_symlink():
            stale_path = self._cache.repo_root / f"{_STALE_PREFIX}{uuid.uuid4().hex}"
            os.replace(snapshot_path, stale_path)

        try:
            os.replace(staging_path, snapshot_path)
            _fsync_directory(snapshots_root)
            self._cache.write_revision_ref(commit_hash, requested_revision)
        except BaseException:
            # BaseException, not Exception: a KeyboardInterrupt here would
            # otherwise strand the moved-aside directory with no owner and no
            # cleanup path, leaking a partial model's worth of disk.
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
        self._backups: dict[Path, Path] = {}

    def commit(self) -> None:
        """Make this patch final, dropping the backups it took.

        Call this only once every file has arrived. Afterwards
        :meth:`rollback` has nothing to undo, so a caller that commits and
        then fails cannot delete the files it just published.
        """
        touched: set[Path] = set()
        for backup in self._backups.values():
            backup.unlink(missing_ok=True)
            touched.add(backup.parent)
        self._backups.clear()
        self._published.clear()
        for directory in touched:
            _fsync_directory(directory)

    def rollback(self) -> None:
        """Undo this patch, leaving the snapshot as it was before it started.

        A half-applied patch is worse than none: the engine would see a subset
        of the weights and load it as if it were complete. A file this patch
        replaced is restored from its backup rather than left deleted --
        rolling back a refresh must not cost the snapshot a shard it already
        had.
        """
        self.close()
        touched: set[Path] = set()
        while self._published:
            target = self._published.pop()
            target.unlink(missing_ok=True)
            backup = self._backups.pop(target, None)
            if backup is not None:
                os.replace(backup, target)
            touched.add(target.parent)
        # A backup with no published file means the rename never landed; the
        # original is the copy to put back, not the one to drop.
        for target, backup in self._backups.items():
            if target.exists():
                backup.unlink(missing_ok=True)
            else:
                os.replace(backup, target)
            touched.add(target.parent)
        self._backups.clear()
        for directory in touched:
            _fsync_directory(directory)

    def _open(self, target: Path):
        temp_path = target.parent / f"{_TEMP_PREFIX}{uuid.uuid4().hex}-{target.name}"
        self._temp_paths[target] = temp_path
        return temp_path.open("xb")

    def _finalize(self, target: Path) -> None:
        temp_path = self._temp_paths.pop(target)
        # os.replace overwrites, so an existing file is gone the moment the
        # rename lands. Move it aside first or rollback has nothing to restore.
        if target.exists() or target.is_symlink():
            backup = target.parent / f"{_BACKUP_PREFIX}{uuid.uuid4().hex}-{target.name}"
            os.replace(target, backup)
            self._backups[target] = backup
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
        return self.read_ref(MAIN_REF)

    def read_ref(self, ref_name: str) -> str | None:
        """Return the commit hash ``refs/<ref_name>`` points at, or None."""
        try:
            ref_path = self.repo_root / "refs" / safe_relative_path(ref_name)
        except ModelSnapshotError:
            return None
        return self._read_ref_path(ref_path)

    @staticmethod
    def _read_ref_path(ref_path: Path) -> str | None:
        if not ref_path.is_file() or ref_path.is_symlink():
            return None
        try:
            return safe_commit_hash(ref_path.read_text(encoding="utf-8").strip())
        except (OSError, UnicodeError, ModelSnapshotError):
            return None

    def write_main_ref(self, commit_hash: str) -> None:
        """Point refs/main at ``commit_hash``, replacing any previous value."""
        self.write_ref(MAIN_REF, commit_hash)

    def write_ref(self, ref_name: str, commit_hash: str) -> None:
        """Point ``refs/<ref_name>`` at ``commit_hash``.

        A ref name may contain slashes -- ``refs/pr/1`` is a legal revision --
        so the parent directories are created here. The name comes from the
        engine's own configuration, so it is validated against the same rules
        as a streamed file path before it becomes part of one, and refused if
        the layout cannot hold it beside a ref already on disk.

        Writing a value the ref already holds is a no-op.
        """
        commit_hash = safe_commit_hash(commit_hash)
        refs_root = self.repo_root / "refs"
        relative = safe_relative_path(ref_name)
        ref_path = refs_root / relative
        self._reject_ref_collision(refs_root, relative)
        if self._read_ref_path(ref_path) == commit_hash:
            # The rename below is durable, not free: a temp file, an
            # fsync each for the file and its directory, then the rename
            # itself -- all to land bytes that are already there.
            # Snapshot reuse is what hits it: resolve_snapshot returns a
            # path only when refs/main already holds the commit, so reuse
            # arrives here with nothing to change.
            return
        _ensure_directory(ref_path.parent, self.cache_root)
        temp_ref = ref_path.parent / f"{_TEMP_PREFIX}{uuid.uuid4().hex}"
        try:
            with temp_ref.open("x", encoding="utf-8") as ref_file:
                ref_file.write(commit_hash)
                ref_file.flush()
                os.fsync(ref_file.fileno())
            os.replace(temp_ref, ref_path)
            _fsync_directory(ref_path.parent)
        finally:
            temp_ref.unlink(missing_ok=True)

    @staticmethod
    def _reject_ref_collision(refs_root: Path, relative: Path) -> None:
        """Refuse a ref name that the layout cannot hold beside an existing one.

        Git itself refuses a branch ``foo`` beside a branch ``foo/bar``, but
        that rule holds within one namespace. This directory is flatter than
        git's: a revision is written under the name the engine asked for, so
        a tag ``foo`` and a branch ``foo/bar`` -- which coexist upstream --
        both land in one tree, as does a full ref path like ``refs/pr/1``.
        ``refs/foo`` cannot be a file and a directory at the same time. Left to
        the filesystem this surfaces as a bare OSError -- FileExistsError from
        mkdir in one order, "Is a directory" from os.replace in the other --
        which a caller catching this module's own error type never sees.
        """
        ancestor = refs_root
        for part in relative.parts[:-1]:
            ancestor = ancestor / part
            if ancestor.exists() and not ancestor.is_dir():
                raise ModelSnapshotError(
                    f"Ref name {str(relative)!r} needs {ancestor} to be a directory, "
                    "but another ref already holds that name"
                )
        target = refs_root / relative
        if target.is_dir() and not target.is_symlink():
            raise ModelSnapshotError(
                f"Ref name {str(relative)!r} is already a directory holding other refs"
            )

    def write_revision_ref(self, commit_hash: str, requested_revision: str | None) -> None:
        """Record the alias the engine will look the snapshot up by.

        Mirrors what ``huggingface_hub`` caches for itself: a ref is written
        only when the requested revision is not already the commit hash, since
        a full commit hash resolves straight to ``snapshots/<commit>/``.

        A pinned request must not touch ``refs/main``. Pointing the default
        revision at a pin would make every later unpinned resolution -- in this
        worker or the next one to share the cache -- read a revision nobody
        asked for.
        """
        if requested_revision is None:
            self.write_main_ref(commit_hash)
            return
        if requested_revision == commit_hash:
            return
        self.write_ref(requested_revision, commit_hash)

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

    def resolve_pinned_snapshot(
        self,
        expected_files: Mapping[str, int],
        commit_hash: str,
    ) -> Path | None:
        """Return ``snapshots/<commit>`` when it already holds every file.

        A pinned request addresses one snapshot directly, so ``refs/main`` --
        which tracks the default revision, a different thing entirely -- has no
        say in whether this one can be reused.
        """
        snapshot_path = self.snapshot_path(commit_hash)
        if self.has_files(snapshot_path, expected_files):
            return snapshot_path
        return None

    def resolve_snapshot(
        self,
        expected_files: Mapping[str, int],
        expected_commit: str | None,
    ) -> Path | None:
        """Return the snapshot refs/main points at when it holds every file.

        ``expected_commit`` is the revision the server reported for this
        request. Reuse fails closed unless it matches: a file manifest carries
        only paths and sizes, so a revision that changed neither is
        indistinguishable from the one already on disk, and reusing it would
        hand the engine stale files without ever opening a stream to notice.
        ``None`` means the server named no revision, which is not proof of
        anything and so never justifies reuse.
        """
        if expected_commit is None:
            return None
        commit_hash = self.read_main_ref()
        if commit_hash is None or commit_hash != expected_commit:
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
