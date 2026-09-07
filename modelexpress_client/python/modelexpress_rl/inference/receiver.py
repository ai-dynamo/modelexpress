# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical S3 checkpoint preparation for generator refit."""

from __future__ import annotations

import json
import mmap
import shutil
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from modelexpress_rl import envs as rl_envs
from modelexpress_rl.inference.checkpoint_store import (
    CheckpointCacheCapacityError,
    CheckpointState,
    LocalCheckpointStore,
    checkpoint_files_state,
)
from modelexpress_rl.object_storage import ObjectStorageType
from modelexpress_rl.s3 import S3Client
from modelexpress_rl.train import WeightPayloadFormat
from modelexpress_rl.utils import (
    checksum_factory,
    index_checkpoint_tensors,
    read_safetensors_header,
    threadpool_map,
)

_BYTES_PER_GB = 1_000_000_000
DEFAULT_REFIT_CHECKPOINT_MAX_SIZE_GB = 500


@dataclass(frozen=True)
class ObjectStorageGeneratorConfig:
    """Object-storage checkpoint settings for one generator rank."""

    storage_type: ObjectStorageType
    initial_base_version_id: str
    seed_checkpoint_path: str | Path
    refit_checkpoint_dir: str | Path
    refit_checkpoint_max_size_gb: int | None = DEFAULT_REFIT_CHECKPOINT_MAX_SIZE_GB
    endpoint_url: str | None = None
    region_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.storage_type, ObjectStorageType):
            raise TypeError("storage_type must be an ObjectStorageType")
        if not self.initial_base_version_id.strip():
            raise ValueError("initial_base_version_id is required")
        if not str(self.seed_checkpoint_path).strip():
            raise ValueError("seed_checkpoint_path is required")
        if not str(self.refit_checkpoint_dir).strip():
            raise ValueError("refit_checkpoint_dir is required")
        if (
            self.refit_checkpoint_max_size_gb is not None
            and self.refit_checkpoint_max_size_gb <= 0
        ):
            raise ValueError("refit_checkpoint_max_size_gb must be positive")


class ReceiverInstallError(RuntimeError):
    """An engine reload failed."""


@dataclass(frozen=True)
class PreparedCheckpoint:
    """One verified host-local checkpoint ready for engine installation."""

    target_version: str
    path: Path
    metrics: dict[str, float]


@dataclass(frozen=True)
class _S3Version:
    version_id: str
    base_version_id: str | None
    payload_format: WeightPayloadFormat
    uri: str


def _source_identity(version: _S3Version) -> dict[str, str]:
    return {"uri": version.uri}


_Decompressor = Callable[[memoryview], Any]


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
    version: _S3Version | None = None,
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
            "version": version.version_id,
            "base_version": version.base_version_id,
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
            raise ValueError(
                f"unsupported compression format {compression_format!r}"
            )
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


class _LocalCheckpoint:
    """Immutable local artifacts plus a materialized install checkpoint."""

    def __init__(
        self,
        *,
        model_name: str,
        config: ObjectStorageGeneratorConfig,
        s3: S3Client,
    ) -> None:
        self.initial_version = config.initial_base_version_id
        self.seed_checkpoint_path = Path(config.seed_checkpoint_path)
        self.s3 = s3
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
        self.checkpoint_paths: list[Path] = []
        self.locations: dict[str, tuple[Path, int, int]] = {}
        self.tensor_metadata: dict[str, dict] = {}

    def initialize(self) -> None:
        self.store.initialize()
        with self.store.installation_locked(), self.store.locked():
            self.local_checkpoint = self.store.full_path(self.initial_version)
            state = self.store.state()
            reusable = (
                state is not None
                and state.get("status") == CheckpointState.READY
                and state.get("version") == self.initial_version
                and any(self.local_checkpoint.glob("*.safetensors"))
            )
            if reusable:
                (
                    self.checkpoint_paths,
                    self.locations,
                    self.tensor_metadata,
                ) = index_checkpoint_tensors(self.local_checkpoint)
                reusable = state.get("files") == checkpoint_files_state(
                    self.checkpoint_paths
                )
                reusable = reusable and self.store.chain(self.initial_version) == {
                    "version": self.initial_version,
                    "full_version": self.initial_version,
                    "deltas": [],
                }
            if not reusable:
                self.reset_initial_checkpoint()
            self.store.activate(self.initial_version)
            self.store.enforce_capacity(
                protected_versions={self.initial_version},
            )

    def _set_local_checkpoint(self, path: Path) -> None:
        self.local_checkpoint = path
        (
            self.checkpoint_paths,
            self.locations,
            self.tensor_metadata,
        ) = index_checkpoint_tensors(path)

    def reset_initial_checkpoint(self) -> None:
        """Restore the immutable initial full checkpoint from the launch seed."""
        state = self.store.state()
        target = self.store.full_path(self.initial_version)
        self.store.ensure_capacity(
            self.store.path_size_bytes(self.seed_checkpoint_path),
            protected_versions={self.initial_version},
        )
        self.store.write_state(
            status=CheckpointState.UPDATING,
            version=(
                state.get("version", self.initial_version)
                if state is not None
                else self.initial_version
            ),
            checkpoint_paths=self.checkpoint_paths,
        )
        with self.store.replace_directory(target) as temporary:
            if self.seed_checkpoint_path.is_file():
                shutil.copy2(
                    self.seed_checkpoint_path,
                    temporary / "model.safetensors",
                )
            elif self.seed_checkpoint_path.is_dir():
                for entry in self.seed_checkpoint_path.iterdir():
                    if entry.is_file():
                        shutil.copy2(entry, temporary / entry.name)
                    elif entry.is_dir():
                        shutil.copytree(entry, temporary / entry.name)
            else:
                raise FileNotFoundError(
                    f"seed checkpoint does not exist: {self.seed_checkpoint_path}"
                )
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

    def prepare(self, version: _S3Version) -> PreparedCheckpoint:
        return self.prepare_chain((version,))

    def prepare_chain(
        self,
        versions: tuple[_S3Version, ...],
    ) -> PreparedCheckpoint:
        """Prepare an ordered chain into one immutable target checkpoint."""
        if not versions:
            raise ValueError("canonical replay chain is empty")
        with self.store.installation_locked(), self.store.locked():
            state = self.store.state()
            if state is None:
                raise RuntimeError("local checkpoint state is missing")
            active_version = self.store.active_version()
            target = versions[-1]
            if state.get("status") != CheckpointState.READY:
                raise RuntimeError("local checkpoint update is incomplete")
            state_checkpoint = self.store.checkpoint_path(state["version"])
            if self.local_checkpoint != state_checkpoint:
                self._set_local_checkpoint(state_checkpoint)
            if state.get("files") != checkpoint_files_state(self.checkpoint_paths):
                raise RuntimeError(
                    "local checkpoint files changed outside ModelExpress"
                )

            # The requested immutable artifact was already prepared.
            if state["version"] == target.version_id:
                self.store.enforce_capacity(
                    protected_versions={active_version, target.version_id},
                )
                self.store.verify_artifact_source(
                    self._artifact_path(target),
                    _source_identity(target),
                )
                return PreparedCheckpoint(
                    target_version=target.version_id,
                    path=self.local_checkpoint,
                    metrics={
                        "perf/mx_receive_delta_index_download": 0.0,
                        "perf/mx_receive_delta_download": 0.0,
                        "perf/mx_receive_delta_apply": 0.0,
                    },
                )
            expected_base = active_version
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
                        f"active checkpoint version {expected_base!r} does not match "
                        f"exact base {version.base_version_id!r} for revision "
                        f"{version.version_id!r}"
                    )
                expected_base = version.version_id
                started = time.perf_counter()
                try:
                    index_data = self.s3.get(version.uri)
                    is_delta = (
                        version.payload_format is WeightPayloadFormat.XOR_DELTA
                    )
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

            self.store.write_state(
                status=CheckpointState.UPDATING,
                version=active_version,
                checkpoint_paths=self.checkpoint_paths,
            )

            try:
                download_time = 0.0
                apply_time = 0.0
                for version, index_data, metadata, weight_map in manifests:
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
            except CheckpointCacheCapacityError:
                self._set_local_checkpoint(
                    self.store.checkpoint_path(active_version)
                )
                self.store.write_state(
                    status=CheckpointState.READY,
                    version=active_version,
                    checkpoint_paths=self.checkpoint_paths,
                )
                raise
            except Exception as error:
                raise RuntimeError(
                    f"target {target.version_id!r}: replay failed at revision "
                    f"{version.version_id!r}: {error}"
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
                    "perf/mx_receive_delta_index_download": index_download_time,
                    "perf/mx_receive_delta_download": download_time,
                    "perf/mx_receive_delta_apply": apply_time,
                },
            )

    def _artifact_path(self, version: _S3Version) -> Path:
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
        version: _S3Version,
        index_data: bytes,
        metadata: dict[str, Any],
        weight_map: dict[str, str],
    ) -> tuple[float, float]:
        target = self.store.full_path(version.version_id)
        if target.exists():
            self.store.enforce_capacity(
                protected_versions={
                    self.store.active_version(),
                    version.version_id,
                },
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

        protected_versions = {
            self.store.active_version(),
            version.version_id,
        }
        self.store.ensure_capacity(
            self._non_weight_files_size(self.seed_checkpoint_path) + len(index_data),
            protected_versions=protected_versions,
        )
        with self.store.replace_directory(target) as temporary:
            self._copy_non_weight_files(self.seed_checkpoint_path, temporary)
            index_name = Path(version.uri).name
            (temporary / index_name).write_bytes(index_data)
            download_time, validation_time = self._download_full_checkpoint(
                target=temporary,
                index_metadata=metadata,
                weight_map=weight_map,
                root_uri=version.uri,
                protected_versions=protected_versions,
            )
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
        version: _S3Version,
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
        download_started = time.perf_counter()
        if artifact.exists():
            self.store.verify_artifact_source(
                artifact,
                _source_identity(version),
            )
        else:
            protected_versions = {
                self.store.active_version(),
                version.base_version_id,
                version.version_id,
            }
            shards = self._download_deltas(weight_map, version.uri)
            self.store.ensure_capacity(
                len(index_data) + sum(len(data) for data, _names in shards.values()),
                protected_versions=protected_versions,
            )
            with self.store.replace_directory(artifact) as temporary:
                (temporary / Path(version.uri).name).write_bytes(index_data)
                for filename, (data, _names) in shards.items():
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
        reuse_materialized = base_checkpoint.parent == self.store.materialized_cache
        self.store.ensure_capacity(
            0 if reuse_materialized else self.store.path_size_bytes(base_checkpoint),
            protected_versions={
                self.store.active_version(),
                version.base_version_id,
                version.version_id,
            },
        )
        if reuse_materialized:
            # A sequential delta can take ownership of the derived base. Rename
            # it to the target version and mutate it without another full copy.
            shutil.rmtree(target, ignore_errors=True)
            base_checkpoint.replace(target)
            self._set_local_checkpoint(target)
            self._apply_delta_artifact(artifact, metadata, weight_map)
        else:
            # The first delta after a full checkpoint copies once so applying
            # it cannot modify the canonical full artifact.
            with self.store.replace_directory(
                target,
                copy_from=base_checkpoint,
            ) as temporary:
                self._set_local_checkpoint(temporary)
                self._apply_delta_artifact(artifact, metadata, weight_map)

        self._set_local_checkpoint(target)
        self.store.write_chain(version.version_id, chain)
        return download_time, time.perf_counter() - apply_started

    def _apply_delta_artifact(
        self,
        artifact: Path,
        index_metadata: dict[str, Any],
        weight_map: dict[str, str],
    ) -> None:
        self.store.verify_artifact(artifact)
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
            data = self.s3.get(f"{parent_uri}/{filename}")
            return filename, data, names

        for filename, data, names in threadpool_map(
            shard_to_tensors.items(),
            download,
            max_workers=min(
                rl_envs.MX_S3_DOWNLOAD_WORKERS,
                len(shard_to_tensors),
            ),
            thread_name_prefix="modelexpress-s3-download-file",
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
    ) -> tuple[float, float]:
        checksum_format = index_metadata.get("checksum_format")
        if set(weight_map) != set(self.locations):
            raise ValueError(
                "full HF checkpoint tensor set differs from local checkpoint"
            )

        shard_to_tensors = _group_tensors_by_shard(weight_map)
        parent_uri = root_uri.rsplit("/", 1)[0]
        workers = min(
            rl_envs.MX_S3_DOWNLOAD_WORKERS,
            len(shard_to_tensors),
        )
        shard_sizes = threadpool_map(
            shard_to_tensors,
            lambda filename: self.s3.size(f"{parent_uri}/{filename}"),
            max_workers=workers,
            thread_name_prefix="modelexpress-s3-head-full",
        )
        self.store.ensure_capacity(
            sum(shard_sizes),
            protected_versions=protected_versions,
            protected_paths={target},
        )

        def download_and_validate(filename: str) -> tuple[float, float]:
            download_started = time.perf_counter()
            try:
                data = self.s3.get(f"{parent_uri}/{filename}")
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

            view = memoryview(data)
            for name in tensor_names:
                if name not in header:
                    raise ValueError(
                        f"full HF checkpoint shard {filename!r} is missing tensor "
                        f"{name!r}"
                    )
                tensor_header = header[name]
                begin, end = tensor_header["data_offsets"]
                local_metadata = self.tensor_metadata[name]
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
            (target / filename).write_bytes(data)
            return download_time, time.perf_counter() - validation_started

        timings = list(
            threadpool_map(
                shard_to_tensors,
                download_and_validate,
                max_workers=workers,
                thread_name_prefix="modelexpress-s3-download-full",
            )
        )

        return (
            max((download for download, _ in timings)),
            max((apply for _, apply in timings)),
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

        maps: dict[Path, tuple[Any, mmap.mmap]] = {}
        try:
            for path in {self.locations[name][0] for name, _data, _checksum in items}:
                file_handle = path.open("r+b")
                maps[path] = (file_handle, mmap.mmap(file_handle.fileno(), 0))

            def apply_one(item) -> None:
                name, compressed, expected_checksum = item
                path, file_offset, size = self.locations[name]
                target = np.frombuffer(
                    maps[path][1],
                    dtype=np.uint8,
                    count=size,
                    offset=file_offset,
                )
                checksum = checksum_factory(index_metadata["checksum_format"])
                reader = _DECOMPRESSORS[index_metadata["compression_format"]](
                    compressed
                )
                position = 0
                extra = b""
                try:
                    while position < size:
                        block = reader.read(min(2 << 20, size - position))
                        if not block:
                            break
                        delta = np.frombuffer(block, dtype=np.uint8)
                        end = position + delta.size
                        region = target[position:end]
                        try:
                            np.bitwise_xor(region, delta, out=region)
                            checksum.update(region)
                        finally:
                            del region
                        position = end
                    if position == size:
                        extra = reader.read(1)
                finally:
                    reader.close()
                    del target
                if position != size or extra:
                    raise ValueError(f"canonical delta byte size differs for {name!r}")
                if checksum.hexdigest() != expected_checksum:
                    raise ValueError(f"canonical target checksum differs for {name!r}")

            list(
                threadpool_map(
                    items,
                    apply_one,
                    max_workers=min(rl_envs.MX_REFIT_DELTA_WORKERS, len(items)),
                    thread_name_prefix="modelexpress-delta-apply",
                )
            )
        finally:
            for file_handle, mapped in maps.values():
                mapped.close()
                file_handle.close()

    @contextmanager
    def installation_context(self, prepared: PreparedCheckpoint):
        with self.store.installation_locked(shared=True):
            with self.store.locked(shared=True):
                state = self.store.state()
                if (
                    state is None
                    or state.get("status") != CheckpointState.READY
                    or state.get("version") != prepared.target_version
                    or state.get("files")
                    != checkpoint_files_state(self.checkpoint_paths)
                ):
                    raise ReceiverInstallError(
                        "prepared checkpoint changed before installation"
                    )
                yield
            with self.store.locked():
                state = self.store.state()
                if (
                    state is None
                    or state.get("status") != CheckpointState.READY
                    or state.get("version") != prepared.target_version
                    or state.get("files")
                    != checkpoint_files_state(self.checkpoint_paths)
                ):
                    raise ReceiverInstallError(
                        "prepared checkpoint changed during installation"
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
]
