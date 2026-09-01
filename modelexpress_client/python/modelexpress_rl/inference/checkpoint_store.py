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


class CheckpointState(str, Enum):
    """Preparation state persisted across receiver processes."""

    READY = "READY"
    UPDATING = "UPDATING"


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
        #
        # Full checkpoints, deltas, and chain manifests are the canonical
        # lineage. Materialized checkpoints are rebuildable outputs for engines
        # that require an ordinary checkpoint directory. Preparation updates
        # state.json without changing active.json; installation advances
        # active.json only after the engine reload succeeds.
        self.cache = Path(root) / quote(model_name, safe="")
        self.full_cache = self.cache / "full"
        self.delta_cache = self.cache / "deltas"
        self.chain_cache = self.cache / "chains"
        self.materialized_cache = self.cache / "materialized"
        self.state_path = self.cache / "state.json"
        self.active_path = self.cache / "active.json"
        self.lock_path = self.cache / ".lock"

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
    def locked(self, *, shared: bool = False) -> Iterator[None]:
        with self.lock_path.open("a+") as handle:
            operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
            fcntl.flock(handle, operation)
            yield

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
        return root / quote(version, safe="")

    def full_path(self, version: str) -> Path:
        return self._version_path(self.full_cache, version)

    def delta_path(self, version: str) -> Path:
        return self._version_path(self.delta_cache, version)

    def materialized_path(self, version: str) -> Path:
        return self._version_path(self.materialized_cache, version)

    def chain_path(self, version: str) -> Path:
        return self.chain_cache / f"{quote(version, safe='')}.json"

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
    "CheckpointState",
    "LocalCheckpointStore",
    "checkpoint_files_state",
]
