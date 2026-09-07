# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-local persistence for canonical checkpoint artifacts."""

from __future__ import annotations

import fcntl
import json
import shutil
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from urllib.parse import quote


def _encode_cache_component(value: str) -> str:
    """Encode an opaque ID as exactly one filesystem path component."""
    # quote() never escapes dots, even with safe="". Encode dot-only IDs
    # explicitly so Path does not interpret them as the current or parent path.
    if value in {".", ".."}:
        return value.replace(".", "%2E")
    return quote(value, safe="")


class CheckpointState(str, Enum):
    """Preparation state persisted across receiver processes."""

    READY = "READY"
    UPDATING = "UPDATING"


class CheckpointCacheCapacityError(RuntimeError):
    """The protected cache working set cannot admit an incoming write."""


def checkpoint_files_state(
    paths: Iterable[Path],
) -> dict[str, list[int]]:
    """Return the size and modification time of indexed checkpoint files."""
    return {
        path.name: [path.stat().st_size, path.stat().st_mtime_ns]
        for path in sorted(set(paths))
    }


def _artifact_files_state(path: Path) -> dict[str, list[int]]:
    return {
        str(file.relative_to(path)): [file.stat().st_size, file.stat().st_mtime_ns]
        for file in sorted(path.rglob("*"))
        if file.is_file() and file.name != ".source.json"
    }


class LocalCheckpointStore:
    """Persist one model's immutable lineage and activation state.

    Layout, locking, fingerprints, and temporary-directory promotion stay
    behind this concrete interface. Tensor reconstruction and object-storage
    access remain the receiver's responsibility.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        model_name: str,
        max_size_bytes: int | None = None,
    ) -> None:
        # Per-model cache layout:
        #
        #   <cache>/
        #     full/<version>/           immutable full HF checkpoints
        #     deltas/<version>/         immutable delta index and shards
        #     chains/<version>.json     full ancestor plus ordered deltas
        #     materialized/<version>/   derived, installable full checkpoints
        #     state.json                current preparation transaction
        #     active.json               version committed after engine install
        #     .lock                     cross-process cache coordination
        #     .install.lock             blocks preparation during engine install
        #
        # Full checkpoints, deltas, and chain manifests are the canonical
        # lineage. Materialized checkpoints are rebuildable outputs for engines
        # that require an ordinary checkpoint directory. Preparation updates
        # state.json without changing active.json; installation advances
        # active.json only after the engine reload succeeds.
        self.cache = Path(root) / _encode_cache_component(model_name)
        self.full_cache = self.cache / "full"
        self.delta_cache = self.cache / "deltas"
        self.chain_cache = self.cache / "chains"
        self.materialized_cache = self.cache / "materialized"
        self.state_path = self.cache / "state.json"
        self.active_path = self.cache / "active.json"
        self.lock_path = self.cache / ".lock"
        self.install_lock_path = self.cache / ".install.lock"
        self.max_size_bytes = max_size_bytes

    def initialize(self) -> None:
        self.cache.mkdir(parents=True, exist_ok=True)
        for path in (
            self.full_cache,
            self.delta_cache,
            self.chain_cache,
            self.materialized_cache,
        ):
            path.mkdir(exist_ok=True)

    @contextmanager
    def _locked(self, path: Path, shared: bool) -> Iterator[None]:
        with path.open("a+") as handle:
            operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(handle, operation)
            yield

    def locked(self, *, shared: bool = False):
        return self._locked(self.lock_path, shared=shared)

    def installation_locked(self, *, shared: bool = False):
        return self._locked(self.install_lock_path, shared=shared)

    @contextmanager
    def replace_directory(
        self,
        target: Path,
        *,
        copy_from: Path | None = None,
    ) -> Iterator[Path]:
        """Populate a temporary directory and promote it to ``target``."""
        temporary = target.with_name(f"{target.name}.tmp")
        shutil.rmtree(temporary, ignore_errors=True)
        try:
            if copy_from is None:
                temporary.mkdir(parents=True)
            else:
                shutil.copytree(copy_from, temporary)
            yield temporary
            shutil.rmtree(target, ignore_errors=True)
            temporary.replace(target)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _version_path(self, root: Path, version: str) -> Path:
        return root / _encode_cache_component(version)

    def full_path(self, version: str) -> Path:
        return self._version_path(self.full_cache, version)

    def delta_path(self, version: str) -> Path:
        return self._version_path(self.delta_cache, version)

    def materialized_path(self, version: str) -> Path:
        return self._version_path(self.materialized_cache, version)

    def chain_path(self, version: str) -> Path:
        return self.chain_cache / f"{_encode_cache_component(version)}.json"

    @staticmethod
    def path_size_bytes(path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        if not path.is_dir():
            return 0
        return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())

    @staticmethod
    def _payload_size_bytes(path: Path) -> int:
        if path.is_file():
            return 0 if path.name == ".source.json" else path.stat().st_size
        if not path.is_dir():
            return 0
        return sum(
            file.stat().st_size
            for file in path.rglob("*")
            if file.is_file() and file.name != ".source.json"
        )

    def cache_size_bytes(self) -> int:
        return sum(
            self._payload_size_bytes(root)
            for root in (
                self.full_cache,
                self.delta_cache,
                self.materialized_cache,
            )
        )

    def _protected_paths(self, versions: Iterable[str]) -> set[Path]:
        protected: set[Path] = set()
        for version in versions:
            protected.update(
                {
                    self.chain_path(version),
                    self.full_path(version),
                    self.delta_path(version),
                    self.materialized_path(version),
                }
            )
            chain = self.chain(version)
            if chain is None:
                continue
            full_version = chain.get("full_version")
            if isinstance(full_version, str):
                protected.add(self.full_path(full_version))
            deltas = chain.get("deltas")
            if isinstance(deltas, list):
                protected.update(
                    self.delta_path(delta) for delta in deltas if isinstance(delta, str)
                )
        return protected

    def _eviction_candidates(self, protected: set[Path]) -> list[Path]:
        groups = (
            self.materialized_cache,
            self.delta_cache,
            self.full_cache,
        )
        candidates: list[Path] = []
        for root in groups:
            candidates.extend(
                sorted(
                    (path for path in root.iterdir() if path not in protected),
                    key=lambda path: path.stat().st_mtime_ns,
                )
            )
        return candidates

    def _remove_orphaned_chain(self, encoded_version: str) -> None:
        if any(
            (root / encoded_version).exists()
            for root in (
                self.full_cache,
                self.delta_cache,
                self.materialized_cache,
            )
        ):
            return
        (self.chain_cache / f"{encoded_version}.json").unlink(missing_ok=True)

    def ensure_capacity(
        self,
        additional_bytes: int,
        *,
        protected_versions: Iterable[str] = (),
        protected_paths: Iterable[Path] = (),
    ) -> None:
        """Evict stale entries before adding bytes, or reject the write."""
        if additional_bytes < 0:
            raise ValueError("additional_bytes must not be negative")

        size = self.cache_size_bytes()
        protected = self._protected_paths(protected_versions)
        protected.update(protected_paths)

        def has_capacity() -> bool:
            quota_available = (
                self.max_size_bytes is None
                or size + additional_bytes <= self.max_size_bytes
            )
            return (
                quota_available
                and shutil.disk_usage(self.cache).free >= additional_bytes
            )

        for candidate in self._eviction_candidates(protected):
            if has_capacity():
                break
            freed_bytes = self._payload_size_bytes(candidate)
            if candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()
            self._remove_orphaned_chain(candidate.name)
            size -= freed_bytes

        if not has_capacity():
            reasons = []
            if (
                self.max_size_bytes is not None
                and size + additional_bytes > self.max_size_bytes
            ):
                reasons.append(
                    f"checkpoint cache quota of {self.max_size_bytes} bytes cannot "
                    f"accommodate {additional_bytes} additional bytes"
                )
            free_bytes = shutil.disk_usage(self.cache).free
            if free_bytes < additional_bytes:
                reasons.append(
                    f"checkpoint cache filesystem has {free_bytes} bytes free but "
                    f"requires {additional_bytes} bytes"
                )
            raise CheckpointCacheCapacityError("; ".join(reasons))

    def enforce_capacity(self, *, protected_versions: Iterable[str] = ()) -> None:
        self.ensure_capacity(0, protected_versions=protected_versions)

    @staticmethod
    def _write_json(path: Path, value: dict[str, object]) -> None:
        temporary = path.with_name(f"{path.name}.tmp")
        temporary.write_text(json.dumps(value, sort_keys=True))
        temporary.replace(path)

    @staticmethod
    def _read_json(path: Path) -> dict | None:
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def state(self) -> dict | None:
        return self._read_json(self.state_path)

    def write_state(
        self,
        *,
        status: CheckpointState,
        version: str,
        checkpoint_paths: Iterable[Path],
        source: dict[str, str] | None = None,
    ) -> None:
        state: dict[str, object] = {"status": status, "version": version}
        if status is CheckpointState.READY:
            state["files"] = checkpoint_files_state(checkpoint_paths)
        if source is not None:
            state["source"] = source
        self._write_json(self.state_path, state)

    def chain(self, version: str) -> dict | None:
        return self._read_json(self.chain_path(version))

    def write_chain(self, version: str, chain: dict[str, object]) -> None:
        self._write_json(self.chain_path(version), chain)

    def checkpoint_path(self, version: str) -> Path:
        chain = self.chain(version)
        if chain is None:
            raise RuntimeError(f"checkpoint chain for {version!r} is missing")
        if chain.get("deltas"):
            return self.materialized_path(version)
        return self.full_path(version)

    def active_version(self) -> str:
        active = self._read_json(self.active_path)
        if active is None or not isinstance(active.get("version"), str):
            raise RuntimeError("active checkpoint version is missing")
        return active["version"]

    def activate(self, version: str) -> None:
        self._write_json(self.active_path, {"version": version})

    @staticmethod
    def _source_path(artifact: Path) -> Path:
        return artifact / ".source.json"

    def record_artifact(
        self,
        artifact: Path,
        *,
        source: dict[str, str] | None = None,
    ) -> None:
        self._write_json(
            self._source_path(artifact),
            {
                "source": source,
                "files": _artifact_files_state(artifact),
            },
        )

    def _verified_artifact_metadata(self, artifact: Path) -> dict:
        metadata = self._read_json(self._source_path(artifact))
        if metadata is None or metadata.get("files") != _artifact_files_state(
            artifact
        ):
            raise ValueError("cached checkpoint artifact changed")
        return metadata

    def verify_artifact(self, artifact: Path) -> None:
        self._verified_artifact_metadata(artifact)

    def verify_artifact_source(
        self,
        artifact: Path,
        expected_source: dict[str, str],
    ) -> None:
        metadata = self._verified_artifact_metadata(artifact)
        if metadata.get("source") != expected_source:
            raise ValueError("prepared checkpoint has different source identity")


__all__ = [
    "CheckpointCacheCapacityError",
    "CheckpointState",
    "LocalCheckpointStore",
    "checkpoint_files_state",
]
