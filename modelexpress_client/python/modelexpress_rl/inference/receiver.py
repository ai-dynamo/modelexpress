# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical S3 checkpoint preparation for generator refit."""

from __future__ import annotations

import fcntl
import json
import mmap
import shutil
import time
import zlib
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np

from modelexpress_rl import envs as rl_envs
from modelexpress_rl.object_storage import ObjectStorageType
from modelexpress_rl.s3 import S3Client
from modelexpress_rl.train import WeightPayloadFormat
from modelexpress_rl.utils import (
    adler32_checksum,
    index_checkpoint_tensors,
    read_safetensors_header,
    threadpool_map,
)


@dataclass(frozen=True)
class ObjectStorageGeneratorConfig:
    """Object-storage checkpoint settings for one generator rank."""

    storage_type: ObjectStorageType
    initial_base_version_id: str
    seed_checkpoint_path: str | Path
    refit_checkpoint_dir: str | Path
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


class _CheckpointState(str, Enum):
    READY = "READY"
    UPDATING = "UPDATING"


def _source_identity(version: _S3Version) -> dict[str, str]:
    return {"uri": version.uri}


_Decompressor = Callable[[memoryview], Any]


def _zstd_stream_reader(data: memoryview) -> Any:
    import zstandard

    return zstandard.ZstdDecompressor().stream_reader(data)


_DECOMPRESSORS: dict[str, _Decompressor] = {
    "zstd": _zstd_stream_reader,
}


def _parse_delta_manifest(
    data: bytes,
) -> tuple[dict[str, str], _Decompressor]:
    try:
        manifest = json.loads(data)
    except (TypeError, ValueError) as error:
        raise ValueError("canonical delta manifest is not valid JSON") from error
    compression_format = manifest["metadata"]["compression_format"]
    try:
        decompressor = _DECOMPRESSORS[compression_format]
    except KeyError as error:
        raise ValueError(
            f"unsupported canonical delta compression format {compression_format!r}"
        ) from error
    return manifest["weight_map"], decompressor


def _parse_full_checkpoint_manifest(data: bytes) -> dict[str, str]:
    try:
        manifest = json.loads(data)
        weight_map = manifest["weight_map"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("full HF checkpoint index is not valid JSON") from error
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("full HF checkpoint index has no tensors")
    for name, filename in weight_map.items():
        if not isinstance(name, str) or not isinstance(filename, str):
            raise ValueError("full HF checkpoint weight_map must contain strings")
        if not filename or Path(filename).name != filename:
            raise ValueError(f"invalid full HF checkpoint shard filename {filename!r}")
    return weight_map


def _group_tensors_by_shard(
    weight_map: dict[str, str],
) -> defaultdict[str, list[str]]:
    shard_to_tensors = defaultdict(list)
    for name, filename in weight_map.items():
        shard_to_tensors[filename].append(name)
    return shard_to_tensors


class _LocalCheckpoint:
    """One host-local checkpoint updated under an exact-base lock."""

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
        self.cache = Path(config.refit_checkpoint_dir) / quote(model_name, safe="")
        self.local_checkpoint = self.cache / "checkpoint"
        self.state_path = self.cache / "state.json"
        self.lock_path = self.cache / ".lock"
        self.checkpoint_paths: list[Path] = []
        self.locations: dict[str, tuple[Path, int, int]] = {}
        self.decompressor: _Decompressor | None = None

    def initialize(self) -> None:
        self.cache.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            state = self._state()
            reusable = (
                state is not None
                and state.get("status") == _CheckpointState.READY
                and state.get("version") == self.initial_version
                and any(self.local_checkpoint.glob("*.safetensors"))
            )
            if reusable:
                (
                    self.checkpoint_paths,
                    self.locations,
                    _,
                ) = index_checkpoint_tensors(self.local_checkpoint)
            else:
                self.reset_initial_checkpoint()

    def _write_state(
        self,
        *,
        status: _CheckpointState,
        version: str,
        source: dict[str, str] | None = None,
    ) -> None:
        state: dict[str, object] = {"status": status, "version": version}
        if source is not None:
            state["source"] = source
        temporary = self.state_path.with_name(f"{self.state_path.name}.tmp")
        temporary.write_text(json.dumps(state, sort_keys=True))
        temporary.replace(self.state_path)

    def reset_initial_checkpoint(self) -> None:
        """Reset the mutable checkpoint from the configured initial seed."""
        state = self._state()
        self._write_state(
            status=_CheckpointState.UPDATING,
            version=(
                state.get("version", self.initial_version)
                if state is not None
                else self.initial_version
            ),
        )
        shutil.rmtree(self.local_checkpoint, ignore_errors=True)
        self.local_checkpoint.mkdir(parents=True)
        if self.seed_checkpoint_path.is_file():
            shutil.copy2(
                self.seed_checkpoint_path,
                self.local_checkpoint / "model.safetensors",
            )
        elif self.seed_checkpoint_path.is_dir():
            for entry in self.seed_checkpoint_path.iterdir():
                if entry.is_file():
                    shutil.copy2(entry, self.local_checkpoint / entry.name)
        else:
            raise FileNotFoundError(
                f"seed checkpoint does not exist: {self.seed_checkpoint_path}"
            )
        (
            self.checkpoint_paths,
            self.locations,
            _,
        ) = index_checkpoint_tensors(self.local_checkpoint)
        self._write_state(
            status=_CheckpointState.READY,
            version=self.initial_version,
        )

    def _state(self) -> dict | None:
        if not self.state_path.is_file():
            return None
        try:
            value = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def prepare(self, version: _S3Version) -> PreparedCheckpoint:
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            state = self._state()
            if state is None:
                raise RuntimeError("local checkpoint state is missing")
            if state.get("status") != _CheckpointState.READY:
                raise RuntimeError("local checkpoint update is incomplete")

            # The local checkpoint is already up to date with the requested version.
            if state["version"] == version.version_id:
                if state.get("source") != _source_identity(version):
                    raise ValueError(
                        "prepared checkpoint has different source identity"
                    )
                return PreparedCheckpoint(
                    target_version=version.version_id,
                    path=self.local_checkpoint,
                    metrics={
                        "perf/mx_receive_delta_index_download": 0.0,
                        "perf/mx_receive_delta_download": 0.0,
                        "perf/mx_receive_delta_apply": 0.0,
                    },
                )
            if (
                version.payload_format is WeightPayloadFormat.XOR_DELTA
                and state["version"] != version.base_version_id
            ):
                raise RuntimeError(
                    f"local checkpoint version {state['version']!r} does not match "
                    f"exact base {version.base_version_id!r}"
                )

            started = time.perf_counter()
            try:
                index_data = self.s3.get(version.uri)
            except Exception as error:
                raise RuntimeError("canonical root download failed") from error
            index_download_time = time.perf_counter() - started

            self._write_state(
                status=_CheckpointState.UPDATING,
                version=state["version"],
            )

            if version.payload_format is WeightPayloadFormat.XOR_DELTA:
                weight_map, self.decompressor = _parse_delta_manifest(index_data)
                download_started = time.perf_counter()
                shards = self._download_deltas(weight_map, version.uri)
                download_time = time.perf_counter() - download_started
                apply_started = time.perf_counter()
                self._apply_shards(shards)
                apply_time = time.perf_counter() - apply_started
            else:
                weight_map = _parse_full_checkpoint_manifest(index_data)
                download_time, apply_time = self._download_full_checkpoint(
                    weight_map=weight_map,
                    root_uri=version.uri,
                )

            self._write_state(
                status=_CheckpointState.READY,
                version=version.version_id,
                source=_source_identity(version),
            )
            return PreparedCheckpoint(
                target_version=version.version_id,
                path=self.local_checkpoint,
                metrics={
                    "perf/mx_receive_delta_index_download": index_download_time,
                    "perf/mx_receive_delta_download": download_time,
                    "perf/mx_receive_delta_apply": apply_time,
                },
            )

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
        weight_map: dict[str, str],
        root_uri: str,
    ) -> tuple[float, float]:
        shard_to_tensors = _group_tensors_by_shard(weight_map)
        parent_uri = root_uri.rsplit("/", 1)[0]

        maps: dict[Path, tuple[Any, mmap.mmap]] = {}
        try:
            for path in self.checkpoint_paths:
                file_handle = path.open("r+b")
                maps[path] = (file_handle, mmap.mmap(file_handle.fileno(), 0))

            def download_and_apply(filename: str) -> tuple[float, float]:
                download_started = time.perf_counter()
                try:
                    data = self.s3.get(f"{parent_uri}/{filename}")
                except Exception as error:
                    raise RuntimeError(
                        f"full HF checkpoint download failed for {filename!r}"
                    ) from error
                download_time = time.perf_counter() - download_started

                header, data_start = read_safetensors_header(data, repr(filename))
                tensor_names = shard_to_tensors[filename]

                checksums = header.get("__metadata__")
                if not isinstance(checksums, dict) or set(checksums) != set(
                    tensor_names
                ):
                    raise ValueError(
                        f"full HF checkpoint shard {filename!r} has invalid checksums"
                    )

                view = memoryview(data)
                apply_started = time.perf_counter()
                for name in tensor_names:
                    begin, end = header[name]["data_offsets"]
                    source = np.frombuffer(
                        view,
                        dtype=np.uint8,
                        count=end - begin,
                        offset=data_start + begin,
                    )
                    if adler32_checksum(source) != checksums[name]:
                        raise ValueError(
                            f"full HF checkpoint checksum differs for {name!r}"
                        )
                    path, offset, size = self.locations[name]
                    target = np.frombuffer(
                        maps[path][1],
                        dtype=np.uint8,
                        count=size,
                        offset=offset,
                    )
                    try:
                        np.copyto(target, source)
                    finally:
                        del target
                return download_time, time.perf_counter() - apply_started

            workers = min(
                rl_envs.MX_S3_DOWNLOAD_WORKERS,
                len(shard_to_tensors),
            )
            timings = list(
                threadpool_map(
                    shard_to_tensors,
                    download_and_apply,
                    max_workers=workers,
                    thread_name_prefix="modelexpress-s3-download-full",
                )
            )
        finally:
            for file_handle, mapped in maps.values():
                mapped.close()
                file_handle.close()

        return (
            max((download for download, _ in timings)),
            max((apply for _, apply in timings)),
        )

    def _apply_shards(
        self,
        shards: dict[str, tuple[bytes, list[str]]],
    ) -> None:
        if not shards:
            return
        assert self.decompressor is not None

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
                checksum = 1
                position = 0
                extra = b""
                reader = self.decompressor(compressed)
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
                            checksum = zlib.adler32(region, checksum)
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
                if f"{checksum:08x}" != expected_checksum:
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
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_SH)
            state = self._state()
            if (
                state is None
                or state.get("status") != _CheckpointState.READY
                or state.get("version") != prepared.target_version
            ):
                raise ReceiverInstallError(
                    "prepared checkpoint changed before installation"
                )
            yield


__all__ = [
    "PreparedCheckpoint",
    "ReceiverInstallError",
    "ObjectStorageGeneratorConfig",
]
