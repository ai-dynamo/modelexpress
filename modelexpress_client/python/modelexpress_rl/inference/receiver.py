# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical object-storage checkpoint preparation for generator refit."""

from __future__ import annotations

import json
import logging
import itertools
import os
import shutil
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from modelexpress_rl import envs as rl_envs
from modelexpress_rl.inference.checkpoint_store import (
    CheckpointCacheCapacityError,
    CheckpointCopyCancelled,
    CheckpointRecord,
    CheckpointState,
    LocalCheckpointStore,
    checkpoint_files_state,
)
from modelexpress_rl.object_storage import ObjectStorageType
from modelexpress_rl.object_storage_reader import ObjectStorageReader
from modelexpress_rl.train import WeightPayloadFormat
from modelexpress_rl.utils import (
    Adler32Checksum,
    checksum_factory,
    index_checkpoint_tensors,
    read_safetensors_header,
    threadpool_map,
)

if TYPE_CHECKING:
    from .streaming_checkpoint import StreamedCheckpoint

logger = logging.getLogger(__name__)

_BYTES_PER_GB = 1_000_000_000
DEFAULT_REFIT_CHECKPOINT_MAX_SIZE_GB = 2000


@dataclass(frozen=True)
class ObjectStorageGeneratorConfig:
    """Object-storage checkpoint settings for one generator rank."""

    storage_type: ObjectStorageType
    initial_base_version_id: str
    seed_checkpoint_path: str | Path | None
    refit_checkpoint_dir: str | Path
    refit_checkpoint_max_size_gb: int | None = DEFAULT_REFIT_CHECKPOINT_MAX_SIZE_GB
    endpoint_url: str | None = None
    region_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.storage_type, ObjectStorageType):
            raise TypeError("storage_type must be an ObjectStorageType")
        if self.storage_type is ObjectStorageType.AZURE:
            for field in ("endpoint_url", "region_name"):
                if getattr(self, field) is not None:
                    raise ValueError(
                        f"{field} is not supported for Azure object storage"
                    )
        if not self.initial_base_version_id.strip():
            raise ValueError("initial_base_version_id is required")
        if (
            self.seed_checkpoint_path is not None
            and not str(self.seed_checkpoint_path).strip()
        ):
            raise ValueError("seed_checkpoint_path must be non-empty when provided")
        if not str(self.refit_checkpoint_dir).strip():
            raise ValueError("refit_checkpoint_dir is required")
        if (
            self.refit_checkpoint_max_size_gb is not None
            and self.refit_checkpoint_max_size_gb <= 0
        ):
            raise ValueError("refit_checkpoint_max_size_gb must be positive")


class PreparationSuperseded(RuntimeError):
    """The cache head changed while background preparation waited."""


class ReceiverInstallError(RuntimeError):
    """An engine reload failed."""


@dataclass(frozen=True)
class DeltaChange:
    """Checkpoint tensor names one delta changes from its exact base."""

    base_version: str
    version: str
    tensor_names: frozenset[str]


@dataclass(frozen=True)
class PreparedCheckpoint:
    """One verified host-local checkpoint ready for engine installation."""

    target_version: str
    path: Path
    metrics: dict[str, float]
    # Deltas from the target's full root, oldest first; empty when unknown.
    delta_changes: tuple[DeltaChange, ...] = ()
    streaming: StreamedCheckpoint | None = None


@dataclass(frozen=True)
class _ObjectStorageVersion:
    version_id: str
    base_version_id: str | None
    payload_format: WeightPayloadFormat
    uri: str


_ObjectStorageManifest = tuple[
    _ObjectStorageVersion, bytes, dict[str, Any], dict[str, str]
]


def _source_identity(version: _ObjectStorageVersion) -> dict[str, str]:
    return {"uri": version.uri}


_Decompressor = Callable[[memoryview], Any]


# Delta bytes applied per read and write of the checkpoint file. Tensors larger
# than _SPREAD_TENSOR_BYTES spread their chunks over the apply workers.
_APPLY_CHUNK_BYTES = 16 << 20
_SPREAD_TENSOR_BYTES = 256 << 20


def _read_exactly(fd: int, out: memoryview, offset: int) -> None:
    done = 0
    while done < len(out):
        read = os.pread(fd, len(out) - done, offset + done)
        out[done : done + len(read)] = read
        read = len(read)
        if read <= 0:
            raise ValueError(f"checkpoint file ended at byte {offset + done}")
        done += read


def _write_all(fd: int, data: memoryview, offset: int) -> None:
    done = 0
    while done < len(data):
        written = os.pwrite(fd, data[done:], offset + done)
        if written <= 0:
            raise OSError("checkpoint write made no progress")
        done += written


def _zstd_stream_reader(data: memoryview) -> Any:
    import zstandard

    return zstandard.ZstdDecompressor().stream_reader(data)


_DECOMPRESSORS: dict[str, _Decompressor] = {
    "zstd": _zstd_stream_reader,
}


def _is_safe_shard_basename(value: object) -> bool:
    """Return whether a shard is one file directly under its artifact root."""
    return (
        isinstance(value, str)
        and bool(value)
        and value not in {".", ".."}
        and Path(value).name == value
    )


def _parse_index_manifest(
    data: bytes,
    *,
    is_delta: bool,
    version: _ObjectStorageVersion | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Parse and validate an index manifest."""
    try:
        index = json.loads(data)
    except (TypeError, ValueError) as error:
        raise ValueError("The index manifest is not valid JSON") from error

    weight_map = index.get("weight_map")
    if weight_map is None:
        raise ValueError("The index manifest has no weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("The index manifest has invalid weight_map")
    if not is_delta and not weight_map:
        raise ValueError("The index manifest has no tensors")
    for name, filename in weight_map.items():
        if (
            not isinstance(name, str)
            or not _is_safe_shard_basename(filename)
            or not filename.endswith(".safetensors")
        ):
            raise ValueError("The index manifest has invalid weight_map")

    metadata = index.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("The index manifest has invalid metadata")
    if is_delta:
        assert version is not None
        expected_metadata = {
            "delta_encoding": "xor",
            "checksum_format": "adler32",
        }
        for name, expected in expected_metadata.items():
            if metadata.get(name) != expected:
                raise ValueError(
                    f"The index manifest {name} does not match revision "
                    f"{version.version_id!r}"
                )
        compression_format = metadata.get("compression_format")
        if compression_format not in _DECOMPRESSORS:
            raise ValueError(f"unsupported compression format {compression_format!r}")
    elif "checksum_format" in metadata:
        checksum_factory(metadata["checksum_format"])
    return metadata, weight_map


def _group_tensors_by_shard(
    weight_map: dict[str, str],
) -> defaultdict[str, list[str]]:
    shard_to_tensors = defaultdict(list)
    for name, filename in weight_map.items():
        shard_to_tensors[filename].append(name)
    return shard_to_tensors


def _full_shard_bytes(
    reader: ObjectStorageReader,
    weight_map: dict[str, str],
    root_uri: str,
) -> int:
    """Return the stored size of a full checkpoint's shards."""
    shards = _group_tensors_by_shard(weight_map)
    parent_uri = root_uri.rsplit("/", 1)[0]
    return sum(
        threadpool_map(
            shards,
            lambda filename: reader.size(f"{parent_uri}/{filename}"),
            max_workers=min(rl_envs.MX_REFIT_DOWNLOAD_WORKERS, len(shards)),
            thread_name_prefix="modelexpress-refit-head-full",
        )
    )


def _download_full_checkpoint(
    *,
    reader: ObjectStorageReader,
    store: LocalCheckpointStore,
    target: Path,
    index_metadata: dict[str, Any],
    weight_map: dict[str, str],
    root_uri: str,
    protected_versions: set[str],
    tensor_metadata: dict[str, dict] | None = None,
    shard_bytes: int | None = None,
    working_copy: Path | None = None,
) -> tuple[float, float]:
    """Download and validate one full HF checkpoint artifact.

    Each shard is also written to ``working_copy`` when one is given.
    """
    checksum_format = index_metadata.get("checksum_format")
    if tensor_metadata is not None and set(weight_map) != set(tensor_metadata):
        raise ValueError("full HF checkpoint tensor set differs from local checkpoint")

    shard_to_tensors = _group_tensors_by_shard(weight_map)
    parent_uri = root_uri.rsplit("/", 1)[0]
    workers = min(rl_envs.MX_REFIT_DOWNLOAD_WORKERS, len(shard_to_tensors))
    if shard_bytes is None:
        shard_bytes = _full_shard_bytes(reader, weight_map, root_uri)
    targets = [target] if working_copy is None else [target, working_copy]
    store.ensure_capacity(
        len(targets) * shard_bytes,
        protected_versions=protected_versions,
        protected_paths=set(targets),
    )

    def download_and_validate(filename: str) -> tuple[float, float]:
        download_started = time.perf_counter()
        try:
            data = reader.get(f"{parent_uri}/{filename}")
        except Exception as error:
            raise RuntimeError(
                f"full HF checkpoint download failed for {filename!r}"
            ) from error
        download_time = time.perf_counter() - download_started

        validation_started = time.perf_counter()
        header, data_start = read_safetensors_header(data, repr(filename))
        tensor_names = shard_to_tensors[filename]
        metadata = header.get("__metadata__", {})
        if not isinstance(metadata, dict):
            raise ValueError(
                f"full HF checkpoint shard {filename!r} has invalid metadata"
            )
        if set(header) - {"__metadata__"} != set(tensor_names):
            raise ValueError(
                f"full HF checkpoint shard {filename!r} tensor set differs "
                "from its index"
            )

        view = memoryview(data)
        for name in tensor_names:
            tensor_header = header[name]
            begin, end = tensor_header["data_offsets"]
            if tensor_metadata is not None:
                local_metadata = tensor_metadata[name]
                if (
                    tensor_header["dtype"] != local_metadata["dtype"]
                    or tensor_header["shape"] != local_metadata["shape"]
                    or end - begin != local_metadata["byte_size"]
                ):
                    raise ValueError(
                        f"full HF checkpoint metadata differs for {name!r}"
                    )
            source = np.frombuffer(
                view,
                dtype=np.uint8,
                count=end - begin,
                offset=data_start + begin,
            )
            if checksum_format is not None:
                expected_checksum = metadata.get(name)
                if expected_checksum is None:
                    raise ValueError(
                        f"full HF checkpoint shard {filename!r} is missing "
                        f"checksum for tensor {name!r}"
                    )
                checksum = checksum_factory(checksum_format)
                checksum.update(source)
                if checksum.hexdigest() != expected_checksum:
                    raise ValueError(
                        f"full HF checkpoint checksum differs for {name!r}"
                    )
        for directory in targets:
            (directory / filename).write_bytes(data)
        return download_time, time.perf_counter() - validation_started

    timings = list(
        threadpool_map(
            shard_to_tensors,
            download_and_validate,
            max_workers=workers,
            thread_name_prefix="modelexpress-refit-download-full",
        )
    )
    return (
        max(download for download, _ in timings),
        max(validation for _, validation in timings),
    )


def _active_version(store: LocalCheckpointStore) -> str | None:
    """Return the installed version when the store has committed one."""
    try:
        return store.active_version()
    except RuntimeError:
        return None


def _protected_versions(
    store: LocalCheckpointStore,
    *versions: str,
) -> set[str]:
    """Return versions excluded from cache eviction during this operation.

    Callers supply the operation's working set. The installed version is also
    retained so recovery can rebuild from the checkpoint backing the engine.
    """
    protected = set(versions)
    active = _active_version(store)
    if active is not None:
        protected.add(active)
    return protected


def _ensure_full_checkpoint(
    *,
    store: LocalCheckpointStore,
    target: Path,
    version: _ObjectStorageVersion,
    reader: ObjectStorageReader,
    protected_versions: set[str],
) -> None:
    """Ensure one immutable, source-verified full checkpoint is cached."""
    source = _source_identity(version)
    if target.exists():
        try:
            store.verify_artifact_source(target, source)
            index_checkpoint_tensors(target)
            return
        except (OSError, RuntimeError, ValueError):
            pass

    index_data = reader.get(version.uri)
    metadata, weight_map = _parse_index_manifest(index_data, is_delta=False)
    store.ensure_capacity(
        len(index_data),
        protected_versions=protected_versions,
    )
    with store.replace_directory(target) as temporary:
        (temporary / Path(version.uri).name).write_bytes(index_data)
        _download_full_checkpoint(
            reader=reader,
            store=store,
            target=temporary,
            index_metadata=metadata,
            weight_map=weight_map,
            root_uri=version.uri,
            protected_versions=protected_versions,
        )
        store.record_artifact(temporary, source=source)
    index_checkpoint_tensors(target)


def bootstrap_object_storage_checkpoint(
    *,
    model_name: str,
    version: _ObjectStorageVersion,
    refit_checkpoint_dir: str | Path,
    reader: ObjectStorageReader,
    refit_checkpoint_max_size_gb: int | None = DEFAULT_REFIT_CHECKPOINT_MAX_SIZE_GB,
) -> Path:
    """Download the immutable full root needed by a cold-start replay."""
    if version.payload_format is not WeightPayloadFormat.FULL_HF_CHECKPOINT:
        raise ValueError("cold-start replay root must be FULL_HF_CHECKPOINT")
    if version.base_version_id is not None:
        raise ValueError("FULL_HF_CHECKPOINT must not have base_version_id")

    store = LocalCheckpointStore(
        root=refit_checkpoint_dir,
        model_name=model_name,
        max_size_bytes=(
            refit_checkpoint_max_size_gb * _BYTES_PER_GB
            if refit_checkpoint_max_size_gb is not None
            else None
        ),
    )
    store.initialize()
    target = store.full_path(version.version_id)
    with store.installation_locked(), store.locked():
        protected_versions = _protected_versions(store, version.version_id)
        state = store.state()
        if state is not None:
            protected_versions.add(state.version)
        _ensure_full_checkpoint(
            store=store,
            target=target,
            version=version,
            reader=reader,
            protected_versions=protected_versions,
        )

        store.write_chain(
            version.version_id,
            {
                "version": version.version_id,
                "full_version": version.version_id,
                "deltas": [],
            },
        )
        # Bootstrap only populates immutable cache data. A later TP rank may
        # arrive after another rank has prepared the target, so this path must
        # not change shared preparation or engine-installation state.
        store.enforce_capacity(protected_versions=protected_versions)
    return target


class _LocalCheckpoint:
    """Immutable local artifacts plus a materialized install checkpoint."""

    def __init__(
        self,
        *,
        model_name: str,
        config: ObjectStorageGeneratorConfig,
        reader: ObjectStorageReader,
    ) -> None:
        self.initial_version = config.initial_base_version_id
        self.reader = reader
        self.store = LocalCheckpointStore(
            root=config.refit_checkpoint_dir,
            model_name=model_name,
            max_size_bytes=(
                config.refit_checkpoint_max_size_gb * _BYTES_PER_GB
                if config.refit_checkpoint_max_size_gb is not None
                else None
            ),
        )
        self.local_checkpoint = self.store.full_path(self.initial_version)
        self.seed_checkpoint_path = (
            Path(config.seed_checkpoint_path)
            if config.seed_checkpoint_path is not None
            else self.local_checkpoint
        )
        self.checkpoint_paths: list[Path] = []
        self.locations: dict[str, tuple[Path, int, int]] = {}
        self.tensor_metadata: dict[str, dict] = {}

    def initialize(self, *, allow_unrecorded_seed: bool = False) -> bool:
        """Initialize the disk seed, or return False if bootstrap is deferred."""
        self.store.initialize()
        with self.store.installation_locked(), self.store.locked():
            initial_checkpoint = self.store.full_path(self.initial_version)
            # An omitted seed or an explicit cached-root path reuses the cache.
            cached_seed = (
                self.seed_checkpoint_path.resolve() == initial_checkpoint.resolve()
            )
            if (
                allow_unrecorded_seed
                and cached_seed
                and not self.store.has_artifact_record(initial_checkpoint)
                and self.store.state() is None
            ):
                return False
            if cached_seed and not initial_checkpoint.is_dir():
                raise FileNotFoundError(
                    f"cached seed checkpoint for {self.initial_version!r} "
                    f"not found: {initial_checkpoint}"
                )
            version = self._initialize_checkpoint_state(
                state=self.store.state(),
                initial_checkpoint=initial_checkpoint,
                cached_seed=cached_seed,
            )
            if not cached_seed:
                self.store.activate(self.initial_version)
            self.store.enforce_capacity(
                protected_versions=_protected_versions(self.store, version),
            )

        return True

    def _initialize_checkpoint_state(
        self,
        *,
        state: CheckpointRecord | None,
        initial_checkpoint: Path,
        cached_seed: bool,
    ) -> str:
        version = self.initial_version
        if cached_seed and state is not None and state.status is CheckpointState.READY:
            # A co-located rank may already have reconstructed the desired
            # target. Reuse that verified cache head instead of rewinding it.
            version = state.version
        reusable = self._reuse_ready_checkpoint(state=state, version=version)
        if reusable and (
            cached_seed
            or self.store.chain(self.initial_version)
            == {
                "version": self.initial_version,
                "full_version": self.initial_version,
                "deltas": [],
            }
        ):
            return version

        self.local_checkpoint = initial_checkpoint
        if cached_seed:
            self._restore_cached_initial_checkpoint()
        else:
            self.reset_initial_checkpoint()
        return self.initial_version

    def _reuse_ready_checkpoint(
        self,
        *,
        state: CheckpointRecord | None,
        version: str,
    ) -> bool:
        if (
            state is None
            or state.status is not CheckpointState.READY
            or state.version != version
        ):
            return False
        try:
            checkpoint = self.store.checkpoint_path(version)
            if not any(checkpoint.glob("*.safetensors")):
                return False
            self._set_local_checkpoint(checkpoint)
        except (OSError, RuntimeError, ValueError):
            return False
        if state.files != checkpoint_files_state(self.checkpoint_paths):
            return False
        return True

    def _restore_cached_initial_checkpoint(self) -> None:
        """Recover incomplete preparation without copying over the storage root."""
        self.store.verify_artifact(self.local_checkpoint)
        self._set_local_checkpoint(self.local_checkpoint)
        self.store.write_chain(
            self.initial_version,
            {
                "version": self.initial_version,
                "full_version": self.initial_version,
                "deltas": [],
            },
        )
        self.store.write_state(
            status=CheckpointState.READY,
            version=self.initial_version,
            checkpoint_paths=self.checkpoint_paths,
        )

    def recover_incomplete_preparation(self) -> None:
        """Restore a usable cache head after an in-process replay failure."""
        with self.store.installation_locked(), self.store.locked():
            state = self.store.state()
            if state is None or state.status is CheckpointState.READY:
                return
            self.local_checkpoint = self.store.full_path(self.initial_version)
            self._restore_cached_initial_checkpoint()
            self.store.enforce_capacity(
                protected_versions=_protected_versions(
                    self.store,
                    self.initial_version,
                ),
            )

    def _set_local_checkpoint(self, path: Path) -> None:
        self.local_checkpoint = path
        (
            self.checkpoint_paths,
            self.locations,
            self.tensor_metadata,
        ) = index_checkpoint_tensors(path)

    def reset_initial_checkpoint(self) -> None:
        """Restore the immutable initial full checkpoint from the launch seed.

        Seed files are symlinked, not copied, so they must stay unchanged while
        the initial version is cached. Seed subdirectories are still copied.
        """
        state = self.store.state()
        target = self.store.full_path(self.initial_version)
        seed = self.seed_checkpoint_path
        self.store.ensure_capacity(
            sum(
                self.store.path_size_bytes(entry)
                for entry in seed.iterdir()
                if entry.is_dir()
            )
            if seed.is_dir()
            else 0,
            protected_versions={self.initial_version},
        )
        self.store.write_state(
            status=CheckpointState.UPDATING,
            version=(state.version if state is not None else self.initial_version),
            checkpoint_paths=self.checkpoint_paths,
        )
        # A working copy of the replaced initial checkpoint is stale.
        self.store.drop_working_copy(self.initial_version)
        with self.store.replace_directory(target) as temporary:
            if seed.is_file():
                (temporary / "model.safetensors").symlink_to(seed.resolve())
            elif seed.is_dir():
                for entry in seed.iterdir():
                    if entry.is_file():
                        (temporary / entry.name).symlink_to(entry.resolve())
                    elif entry.is_dir():
                        shutil.copytree(entry, temporary / entry.name)
            else:
                raise FileNotFoundError(f"seed checkpoint does not exist: {seed}")
            self.store.record_artifact(temporary)
        self._set_local_checkpoint(target)
        self.store.write_chain(
            self.initial_version,
            {
                "version": self.initial_version,
                "full_version": self.initial_version,
                "deltas": [],
            },
        )
        self.store.write_state(
            status=CheckpointState.READY,
            version=self.initial_version,
            checkpoint_paths=self.checkpoint_paths,
        )

    def prepare_seed_working_copy(self, cancelled: threading.Event) -> bool:
        """Copy the full cache head, usually the seed, to its working copy.

        Copying holds only the preparation lock: ranks can still initialize and
        install, while the next delta waits for the copy and then renames it.
        Setting ``cancelled`` stops the copy and any wait for the other locks.
        Return whether this call made the copy; another rank may have made it.
        """
        with self.store.preparation_locked(blocking=False) as acquired:
            if not acquired:
                return False
            with (
                self.store.installation_locked(cancelled=cancelled) as installing,
                self.store.locked(cancelled=cancelled) as locked,
            ):
                if not (installing and locked):
                    return False
                state = self.store.state()
                if state is None or state.status is not CheckpointState.READY:
                    return False
                chain = self.store.chain(state.version)
                # The next delta renames a materialized head instead of copying it.
                if chain is None or chain.get("deltas"):
                    return False
                version = state.version
                if self.store.working_copy(version) is not None:
                    return False
                full_checkpoint = self.store.full_path(version)
                working_copy = self.store.materialized_path(version)
                record = self.store.artifact_record(full_checkpoint)
                size = self.store.path_size_bytes(full_checkpoint)
                protected_versions = _protected_versions(self.store, version)
                if not self.store.can_admit(
                    size,
                    protected_versions=protected_versions,
                ):
                    logger.warning(
                        "No room for a working copy of full checkpoint %s, so the "
                        "next delta copies it",
                        version,
                    )
                    return False
                self.store.ensure_capacity(
                    size,
                    protected_versions=protected_versions,
                )
            logger.info(
                "Copying full checkpoint %s (%.1f GB) to its working copy",
                version,
                size / _BYTES_PER_GB,
            )
            started = time.perf_counter()
            # Copy in place: eviction protects the cache head's working copy,
            # and only a recorded copy is used, so a later preparation deletes
            # a partial copy under the store lock.
            try:
                self.store.copy_directory(
                    full_checkpoint,
                    working_copy,
                    cancelled=cancelled,
                )
                with self.store.locked(cancelled=cancelled) as locked:
                    if not locked:
                        raise CheckpointCopyCancelled("checkpoint copy was cancelled")
                    self.store.record_copy(full_checkpoint, working_copy, record)
            except CheckpointCopyCancelled:
                logger.info(
                    "Cancelled copying full checkpoint %s to its working copy", version
                )
                return False
            logger.info(
                "Copied full checkpoint %s to its working copy in %.1f s",
                version,
                time.perf_counter() - started,
            )
        return True

    def prepare(self, version: _ObjectStorageVersion) -> PreparedCheckpoint:
        return self.prepare_chain((version,))

    def prepare_chain(
        self,
        versions: tuple[_ObjectStorageVersion, ...],
        *,
        expected_head: str | None = None,
    ) -> PreparedCheckpoint:
        """Prepare an ordered chain into one immutable target checkpoint."""
        if not versions:
            raise ValueError("canonical replay chain is empty")
        target = versions[-1]
        # Queue preparers separately so followers can reuse the target while
        # the leader installs it, without first waiting for an exclusive fence.
        started = time.perf_counter()
        with self.store.preparation_locked():
            # Includes another rank's preparation and a background working copy.
            lock_wait = time.perf_counter() - started
            with (
                self.store.installation_locked(shared=True),
                self.store.locked(shared=True),
            ):
                state, _active_version = self._load_ready_state()
                if expected_head is not None and (
                    state.version != expected_head
                    or self.store.active_version() != expected_head
                ):
                    raise PreparationSuperseded(
                        "cache head changed before background preparation"
                    )
                prepared = self._reuse_prepared_target(
                    state=state,
                    target=target,
                    lock_wait=lock_wait,
                )
                if prepared is not None:
                    return prepared
            with self.store.installation_locked(), self.store.locked():
                state, active_version = self._load_ready_state()
                if expected_head is not None and (
                    state.version != expected_head
                    or self.store.active_version() != expected_head
                ):
                    raise PreparationSuperseded(
                        "cache head changed before background preparation"
                    )
                prepared = self._reuse_prepared_target(
                    state=state,
                    target=target,
                    lock_wait=lock_wait,
                )
                if prepared is not None:
                    return prepared
                # P2P can advance the engine without advancing this disk checkpoint.
                # Choose the replay suffix from the verified cache head under lock.
                for position, version in enumerate(versions[:-1]):
                    if version.version_id != state.version:
                        continue
                    source = self.store.artifact_source(self._artifact_path(version))
                    if source is None:
                        break
                    if source != _source_identity(version):
                        raise ValueError(
                            "prepared checkpoint has different source identity"
                        )
                    versions = versions[position + 1 :]
                    break
                manifests, index_download_time = self._download_replay_manifests(
                    versions=versions,
                    target=target,
                    base_version=state.version,
                )
                return self._reconstruct_target(
                    manifests=manifests,
                    target=target,
                    active_version=active_version,
                    index_download_time=index_download_time,
                    lock_wait=lock_wait,
                )

    def _load_ready_state(self) -> tuple[CheckpointRecord, str]:
        state = self.store.state()
        if state is None:
            raise RuntimeError("local checkpoint state is missing")
        if state.status is not CheckpointState.READY:
            raise RuntimeError("local checkpoint update is incomplete")
        active_version = _active_version(self.store) or state.version
        state_checkpoint = self.store.checkpoint_path(state.version)
        if self.local_checkpoint != state_checkpoint:
            self._set_local_checkpoint(state_checkpoint)
        if state.files != checkpoint_files_state(self.checkpoint_paths):
            raise RuntimeError("local checkpoint files changed outside ModelExpress")
        return state, active_version

    def _reuse_prepared_target(
        self,
        *,
        state: CheckpointRecord,
        target: _ObjectStorageVersion,
        lock_wait: float,
    ) -> PreparedCheckpoint | None:
        if state.version != target.version_id:
            return None
        delta_changes = self._delta_changes(target.version_id)
        # Followers reach this after the lock holder has reconstructed the
        # target. Source identity prevents attaching to a reused UID artifact.
        self.store.verify_artifact_source(
            self._artifact_path(target),
            _source_identity(target),
        )
        return PreparedCheckpoint(
            target_version=target.version_id,
            path=self.local_checkpoint,
            metrics={
                "perf/mx_receive_prepare_lock_wait": lock_wait,
                "perf/mx_receive_delta_index_download": 0.0,
                "perf/mx_receive_delta_download": 0.0,
                "perf/mx_receive_delta_apply": 0.0,
            },
            delta_changes=delta_changes,
        )

    def _delta_changes(self, version_id: str) -> tuple[DeltaChange, ...]:
        """Changed names per delta from the full root, or ``()`` if unknown."""
        if not rl_envs.MX_REFIT_DELTA_SURGICAL:
            return ()
        chain = self.store.chain(version_id)
        if chain is None:
            return ()
        changes = []
        base = chain["full_version"]
        for delta in chain["deltas"]:
            indexes = list(self.store.delta_path(delta).glob("*.index.json"))
            if len(indexes) != 1:
                return ()
            try:
                index = json.loads(indexes[0].read_text())
            except (OSError, ValueError):
                return ()
            if not isinstance(index, dict):
                return ()
            weight_map = index.get("weight_map")
            metadata = index.get("metadata")
            if (
                not isinstance(weight_map, dict)
                or not isinstance(metadata, dict)
                or metadata.get("delta_encoding") != "xor"
            ):
                return ()
            changes.append(DeltaChange(base, delta, frozenset(weight_map)))
            base = delta
        return tuple(changes)

    def _download_replay_manifests(
        self,
        *,
        versions: tuple[_ObjectStorageVersion, ...],
        target: _ObjectStorageVersion,
        base_version: str,
    ) -> tuple[list[_ObjectStorageManifest], float]:
        """Validate the entire chain before mutating preparation state."""
        expected_base = base_version
        manifests = []
        index_download_time = 0.0
        for position, version in enumerate(versions):
            if (
                position > 0
                and version.payload_format is WeightPayloadFormat.FULL_HF_CHECKPOINT
            ):
                raise RuntimeError(
                    f"full checkpoint revision {version.version_id!r} must be "
                    "the first replay revision"
                )
            if (
                version.payload_format is WeightPayloadFormat.XOR_DELTA
                and expected_base != version.base_version_id
            ):
                raise RuntimeError(
                    f"local checkpoint version {expected_base!r} does not match "
                    f"exact base {version.base_version_id!r} for revision "
                    f"{version.version_id!r}"
                )
            expected_base = version.version_id
            started = time.perf_counter()
            try:
                index_data = self.reader.get(version.uri)
                is_delta = version.payload_format is WeightPayloadFormat.XOR_DELTA
                metadata, weight_map = _parse_index_manifest(
                    index_data,
                    is_delta=is_delta,
                    version=version if is_delta else None,
                )
            except Exception as error:
                raise RuntimeError(
                    f"target {target.version_id!r}: replay validation failed "
                    f"at revision {version.version_id!r}: {error}"
                ) from error
            index_download_time += time.perf_counter() - started
            manifests.append((version, index_data, metadata, weight_map))
        return manifests, index_download_time

    def _reconstruct_target(
        self,
        *,
        manifests: list[_ObjectStorageManifest],
        target: _ObjectStorageVersion,
        active_version: str,
        index_download_time: float,
        lock_wait: float,
    ) -> PreparedCheckpoint:
        restore = self.store.state()
        assert restore is not None
        # READY -> UPDATING is persisted before any derived checkpoint mutates.
        # A crash therefore makes the next initializer rebuild from lineage.
        self.store.write_state(
            status=CheckpointState.UPDATING,
            version=active_version,
            checkpoint_paths=self.checkpoint_paths,
        )

        failed_version = target
        try:
            download_time = 0.0
            apply_time = 0.0
            for version, index_data, metadata, weight_map in manifests:
                failed_version = version
                if version.payload_format is WeightPayloadFormat.XOR_DELTA:
                    downloaded, applied = self._prepare_delta(
                        version,
                        index_data,
                        metadata,
                        weight_map,
                    )
                else:
                    downloaded, applied = self._prepare_full(
                        version,
                        index_data,
                        metadata,
                        weight_map,
                    )
                download_time += downloaded
                apply_time += applied
                restore = CheckpointRecord(
                    status=CheckpointState.READY,
                    version=version.version_id,
                    source=_source_identity(version),
                )
        except CheckpointCacheCapacityError:
            self._set_local_checkpoint(self.store.checkpoint_path(restore.version))
            self.store.write_state(
                status=restore.status,
                version=restore.version,
                checkpoint_paths=self.checkpoint_paths,
                source=restore.source,
            )
            raise
        except Exception as error:
            raise RuntimeError(
                f"target {target.version_id!r}: replay failed at revision "
                f"{failed_version.version_id!r}: {error}"
            ) from error

        self.store.write_state(
            status=CheckpointState.READY,
            version=target.version_id,
            checkpoint_paths=self.checkpoint_paths,
            source=_source_identity(target),
        )
        return PreparedCheckpoint(
            target_version=target.version_id,
            path=self.local_checkpoint,
            metrics={
                "perf/mx_receive_prepare_lock_wait": lock_wait,
                "perf/mx_receive_delta_index_download": index_download_time,
                "perf/mx_receive_delta_download": download_time,
                "perf/mx_receive_delta_apply": apply_time,
            },
            delta_changes=self._delta_changes(target.version_id),
        )

    def _artifact_path(self, version: _ObjectStorageVersion) -> Path:
        if version.payload_format is WeightPayloadFormat.XOR_DELTA:
            return self.store.delta_path(version.version_id)
        return self.store.full_path(version.version_id)

    @staticmethod
    def _copy_non_weight_files(source: Path, target: Path) -> None:
        if not source.is_dir():
            return
        for entry in source.iterdir():
            if entry.name.endswith(".safetensors") or entry.name.endswith(
                ".safetensors.index.json"
            ):
                continue
            destination = target / entry.name
            if entry.is_dir():
                shutil.copytree(entry, destination)
            elif entry.is_file():
                shutil.copy2(entry, destination)

    @staticmethod
    def _non_weight_files_size(source: Path) -> int:
        if not source.is_dir():
            return 0
        return sum(
            LocalCheckpointStore.path_size_bytes(entry)
            for entry in source.iterdir()
            if not entry.name.endswith(".safetensors")
            and not entry.name.endswith(".safetensors.index.json")
        )

    def _prepare_full(
        self,
        version: _ObjectStorageVersion,
        index_data: bytes,
        metadata: dict[str, Any],
        weight_map: dict[str, str],
    ) -> tuple[float, float]:
        # Cached or not, another full checkpoint leaves the active one's working
        # copy unused.
        active_version = _active_version(self.store)
        if active_version not in {None, version.version_id}:
            self.store.drop_unused_working_copy(active_version)
        target = self.store.full_path(version.version_id)
        if target.exists():
            source = self.store.artifact_source(target)
            if source is not None:
                self.store.enforce_capacity(
                    protected_versions=_protected_versions(
                        self.store,
                        version.version_id,
                    ),
                )
                self.store.verify_artifact_source(
                    target,
                    _source_identity(version),
                )
                self._set_local_checkpoint(target)
                self.store.write_chain(
                    version.version_id,
                    {
                        "version": version.version_id,
                        "full_version": version.version_id,
                        "deltas": [],
                    },
                )
                return 0.0, 0.0

        non_weight_source = self.seed_checkpoint_path
        if (
            non_weight_source.resolve()
            == self.store.full_path(self.initial_version).resolve()
        ):
            # Updates carry seed metadata forward; the initial root may be evicted.
            non_weight_source = self.local_checkpoint
        # A working copy of the full checkpoint being replaced is stale.
        self.store.drop_working_copy(version.version_id)
        protected_versions = _protected_versions(self.store, version.version_id)
        shard_bytes = _full_shard_bytes(self.reader, weight_map, version.uri)
        metadata_bytes = self._non_weight_files_size(non_weight_source) + len(
            index_data
        )
        # A working copy lets the next delta mutate it instead of copying.
        directories = [target, self.store.materialized_path(version.version_id)]
        if not self.store.can_admit(
            2 * (metadata_bytes + shard_bytes),
            protected_versions=protected_versions,
            protected_paths={non_weight_source},
        ):
            logger.warning(
                "No room for a working copy of full checkpoint %s, so the next "
                "delta copies it",
                version.version_id,
            )
            directories.pop()
        self.store.ensure_capacity(
            len(directories) * metadata_bytes,
            protected_versions=protected_versions,
            protected_paths={non_weight_source},
        )
        with ExitStack() as stack:
            temporaries = [
                stack.enter_context(self.store.replace_directory(directory))
                for directory in directories
            ]
            for temporary in temporaries:
                self._copy_non_weight_files(non_weight_source, temporary)
                (temporary / Path(version.uri).name).write_bytes(index_data)
            download_time, validation_time = self._download_full_checkpoint(
                target=temporaries[0],
                index_metadata=metadata,
                weight_map=weight_map,
                root_uri=version.uri,
                protected_versions=protected_versions,
                shard_bytes=shard_bytes,
                working_copy=temporaries[1] if len(temporaries) > 1 else None,
            )
            for temporary in temporaries:
                self.store.record_artifact(
                    temporary,
                    source=_source_identity(version),
                )

        self._set_local_checkpoint(target)
        self.store.write_chain(
            version.version_id,
            {
                "version": version.version_id,
                "full_version": version.version_id,
                "deltas": [],
            },
        )
        return download_time, validation_time

    def _prepare_delta(
        self,
        version: _ObjectStorageVersion,
        index_data: bytes,
        metadata: dict[str, Any],
        weight_map: dict[str, str],
    ) -> tuple[float, float]:
        assert version.base_version_id is not None
        base_chain = self.store.chain(version.base_version_id)
        if base_chain is None:
            raise RuntimeError(
                "checkpoint chain for exact base "
                f"{version.base_version_id!r} is missing"
            )

        artifact = self.store.delta_path(version.version_id)
        downloaded = None
        download_started = time.perf_counter()
        if artifact.exists():
            self.store.verify_artifact_source(
                artifact,
                _source_identity(version),
            )
        else:
            protected_versions = _protected_versions(
                self.store,
                version.base_version_id,
                version.version_id,
            )
            downloaded = self._download_deltas(weight_map, version.uri)
            self.store.ensure_capacity(
                len(index_data)
                + sum(len(data) for data, _names in downloaded.values()),
                protected_versions=protected_versions,
            )
            with self.store.replace_directory(artifact) as temporary:
                (temporary / Path(version.uri).name).write_bytes(index_data)
                for filename, (data, _names) in downloaded.items():
                    (temporary / filename).write_bytes(data)
                self.store.record_artifact(
                    temporary,
                    source=_source_identity(version),
                )
        download_time = time.perf_counter() - download_started

        chain = {
            "version": version.version_id,
            "full_version": base_chain["full_version"],
            "deltas": [*base_chain["deltas"], version.version_id],
        }
        target = self.store.materialized_path(version.version_id)
        apply_started = time.perf_counter()
        self.store.verify_artifact(self.store.full_path(chain["full_version"]))
        for delta_version in base_chain["deltas"]:
            self.store.verify_artifact(self.store.delta_path(delta_version))
        base_checkpoint = self.store.checkpoint_path(version.base_version_id)
        if base_checkpoint.parent != self.store.materialized_cache:
            base_checkpoint = (
                self.store.working_copy(version.base_version_id) or base_checkpoint
            )
        reuse_materialized = base_checkpoint.parent == self.store.materialized_cache
        self.store.ensure_capacity(
            0 if reuse_materialized else self.store.path_size_bytes(base_checkpoint),
            protected_versions=_protected_versions(
                self.store,
                version.base_version_id,
                version.version_id,
            ),
        )
        if reuse_materialized:
            # A sequential delta can take ownership of the derived base. Rename
            # it to the target version and mutate it without another full copy.
            shutil.rmtree(target, ignore_errors=True)
            base_checkpoint.replace(target)
            self._set_local_checkpoint(target)
            self._apply_delta_artifact(
                artifact, metadata, weight_map, shards=downloaded
            )
        else:
            # A full checkpoint without a working copy is copied once so
            # applying the delta cannot modify the canonical full artifact.
            with self.store.replace_directory(
                target,
                copy_from=base_checkpoint,
            ) as temporary:
                self._set_local_checkpoint(temporary)
                self._apply_delta_artifact(
                    artifact, metadata, weight_map, shards=downloaded
                )

        self._set_local_checkpoint(target)
        self.store.write_chain(version.version_id, chain)
        return download_time, time.perf_counter() - apply_started

    def _apply_delta_artifact(
        self,
        artifact: Path,
        index_metadata: dict[str, Any],
        weight_map: dict[str, str],
        *,
        shards: dict[str, tuple[bytes, list[str]]] | None = None,
    ) -> None:
        self.store.verify_artifact(artifact)
        if shards is None:
            shards = {
                filename: (
                    (artifact / filename).read_bytes(),
                    names,
                )
                for filename, names in _group_tensors_by_shard(weight_map).items()
            }
        self._apply_shards(shards, index_metadata=index_metadata)

    def _download_deltas(
        self,
        weight_map: dict[str, str],
        root_uri: str,
    ) -> dict[str, tuple[bytes, list[str]]]:
        if not weight_map:
            return {}
        shard_to_tensors = _group_tensors_by_shard(weight_map)
        parent_uri = root_uri.rsplit("/", 1)[0]
        shards = {}

        def download(item: tuple[str, list[str]]):
            filename, names = item
            data = self.reader.get(f"{parent_uri}/{filename}")
            return filename, data, names

        for filename, data, names in threadpool_map(
            shard_to_tensors.items(),
            download,
            max_workers=min(
                rl_envs.MX_REFIT_DOWNLOAD_WORKERS,
                len(shard_to_tensors),
            ),
            thread_name_prefix="modelexpress-refit-download-file",
        ):
            shards[filename] = (data, names)
        return shards

    def _download_full_checkpoint(
        self,
        *,
        target: Path,
        index_metadata: dict[str, Any],
        weight_map: dict[str, str],
        root_uri: str,
        protected_versions: set[str],
        shard_bytes: int | None = None,
        working_copy: Path | None = None,
    ) -> tuple[float, float]:
        return _download_full_checkpoint(
            reader=self.reader,
            store=self.store,
            target=target,
            index_metadata=index_metadata,
            weight_map=weight_map,
            root_uri=root_uri,
            protected_versions=protected_versions,
            tensor_metadata=self.tensor_metadata,
            shard_bytes=shard_bytes,
            working_copy=working_copy,
        )

    def _apply_shards(
        self,
        shards: dict[str, tuple[bytes, list[str]]],
        *,
        index_metadata: dict[str, Any],
    ) -> None:
        if not shards:
            return

        items = []
        for filename, (data, names) in shards.items():
            header, data_start = read_safetensors_header(data, repr(filename))
            checksums = header.get("__metadata__")
            if not isinstance(checksums, dict):
                raise ValueError(
                    f"canonical delta shard {filename!r} is missing checksum metadata"
                )
            view = memoryview(data)
            for name in names:
                if name not in header:
                    raise ValueError(
                        f"canonical delta shard {filename!r} is missing tensor {name!r}"
                    )
                if name not in checksums:
                    raise ValueError(
                        f"canonical delta shard {filename!r} is missing checksum "
                        f"for tensor {name!r}"
                    )
                if name not in self.locations:
                    raise ValueError(
                        f"canonical delta tensor {name!r} from shard {filename!r} "
                        "is absent from the local checkpoint"
                    )
                begin, end = header[name]["data_offsets"]
                items.append(
                    (
                        name,
                        view[data_start + begin : data_start + end],
                        checksums[name],
                    )
                )
        if not items:
            return

        # Take each file's tensors largest first, so a large tensor does not
        # finish last, and alternate files, since writes to one file serialize.
        by_file: defaultdict[Path, list] = defaultdict(list)
        for item in items:
            by_file[self.locations[item[0]][0]].append(item)
        for group in by_file.values():
            group.sort(key=lambda item: -self.locations[item[0]][2])
        items = [
            item
            for row in itertools.zip_longest(*by_file.values())
            for item in row
            if item is not None
        ]
        checksum_format = index_metadata["checksum_format"]
        decompress = _DECOMPRESSORS[index_metadata["compression_format"]]
        workers = min(rl_envs.MX_REFIT_DELTA_WORKERS, len(items))
        local = threading.local()

        def apply_chunk(
            fd: int, offset: int, block: bytes, position: int
        ) -> tuple[Adler32Checksum, int]:
            """XOR one decompressed chunk into the file; checksum the target bytes."""
            delta = np.frombuffer(block, dtype=np.uint8)
            if getattr(local, "buffer", None) is None:
                local.buffer = np.empty(_APPLY_CHUNK_BYTES, dtype=np.uint8)
            target = local.buffer[: delta.size]
            view = memoryview(target)
            _read_exactly(fd, view, offset + position)
            np.bitwise_xor(target, delta, out=target)
            checksum = checksum_factory(checksum_format)
            checksum.update(target)
            _write_all(fd, view, offset + position)
            return checksum, delta.size

        files: dict[Path, int] = {}
        with ThreadPoolExecutor(
            max_workers=rl_envs.MX_REFIT_DELTA_WORKERS,
            thread_name_prefix="modelexpress-delta-chunk",
        ) as chunks:

            def apply_one(item) -> None:
                name, compressed, expected_checksum = item
                path, file_offset, size = self.locations[name]
                fd = files[path]
                checksum = checksum_factory(checksum_format)
                # A large tensor's chunks spread over the workers; others stay here.
                spread = (
                    rl_envs.MX_REFIT_DELTA_WORKERS > 1 and size > _SPREAD_TENSOR_BYTES
                )
                pending: deque[Future] = deque()

                def collect(result) -> None:
                    part, length = result
                    checksum.extend(part, length)

                reader = decompress(compressed)
                position = 0
                extra = b""
                try:
                    while position < size:
                        block = reader.read(min(_APPLY_CHUNK_BYTES, size - position))
                        if not block:
                            break
                        if spread:
                            pending.append(
                                chunks.submit(
                                    apply_chunk, fd, file_offset, block, position
                                )
                            )
                            if len(pending) >= 2 * workers:
                                collect(pending.popleft().result())
                        else:
                            collect(apply_chunk(fd, file_offset, block, position))
                        position += len(block)
                    if position == size:
                        extra = reader.read(1)
                    while pending:
                        collect(pending.popleft().result())
                finally:
                    reader.close()
                    for future in pending:
                        future.cancel()
                    wait(pending)
                if position != size or extra:
                    raise ValueError(f"canonical delta byte size differs for {name!r}")
                # Checksums describe the delta's target, not a reverted base.
                if checksum.hexdigest() != expected_checksum:
                    raise ValueError(f"canonical target checksum differs for {name!r}")

            try:
                for path in {
                    self.locations[name][0] for name, _data, _checksum in items
                }:
                    files[path] = os.open(path, os.O_RDWR)
                list(
                    threadpool_map(
                        items,
                        apply_one,
                        max_workers=workers,
                        thread_name_prefix="modelexpress-delta-apply",
                    )
                )
            finally:
                for descriptor in files.values():
                    os.close(descriptor)

    def _validate_prepared(
        self,
        prepared: PreparedCheckpoint,
        *,
        message: str,
    ) -> None:
        """Ensure the prepared checkpoint still matches durable store state."""
        state = self.store.state()
        if (
            state is None
            or state.status is not CheckpointState.READY
            or state.version != prepared.target_version
            or state.files != checkpoint_files_state(self.checkpoint_paths)
        ):
            raise ReceiverInstallError(message)

    def activate(self, prepared: PreparedCheckpoint) -> None:
        """Commit a prepared checkpoint after every local installer succeeds."""
        with self.store.installation_locked(), self.store.locked():
            self._validate_prepared(
                prepared,
                message="prepared checkpoint changed before activation",
            )
            self.store.activate(prepared.target_version)
            self.store.enforce_capacity(
                protected_versions={prepared.target_version},
            )

    @contextmanager
    def installation_context(
        self,
        prepared: PreparedCheckpoint,
        *,
        activate: bool = True,
    ):
        """Hold installation locks and optionally activate on successful exit."""
        with self.store.installation_locked(shared=True):
            with self.store.locked(shared=True):
                self._validate_prepared(
                    prepared,
                    message="prepared checkpoint changed before installation",
                )
                yield
            if activate:
                with self.store.locked():
                    self._validate_prepared(
                        prepared,
                        message="prepared checkpoint changed during installation",
                    )
                    self.store.activate(prepared.target_version)
                    self.store.enforce_capacity(
                        protected_versions={prepared.target_version},
                    )


__all__ = [
    "DEFAULT_REFIT_CHECKPOINT_MAX_SIZE_GB",
    "PreparedCheckpoint",
    "ReceiverInstallError",
    "ObjectStorageGeneratorConfig",
    "bootstrap_object_storage_checkpoint",
]
