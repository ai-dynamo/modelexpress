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
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import zstandard

from modelexpress_rl import envs as rl_envs
from modelexpress_rl.object_storage import ObjectStorageType
from modelexpress_rl.s3 import S3Client
from modelexpress_rl.train import WeightPayloadFormat
from modelexpress_rl.utils import index_checkpoint_tensors, read_safetensors_header

from .adapter import (
    GeneratorEngineAdapter,
    GeneratorTransferInputs,
)


@dataclass(frozen=True)
class ObjectStorageGeneratorConfig:
    """Object-storage checkpoint settings for one generator rank."""

    storage_type: ObjectStorageType
    initial_base_version_id: str
    launch_checkpoint: str | Path
    preparation_cache_dir: str | Path
    endpoint_url: str | None = None
    region_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.storage_type, ObjectStorageType):
            raise TypeError("storage_type must be an ObjectStorageType")
        if not self.initial_base_version_id.strip():
            raise ValueError("initial_base_version_id is required")
        if not str(self.launch_checkpoint).strip():
            raise ValueError("launch_checkpoint is required")
        if not str(self.preparation_cache_dir).strip():
            raise ValueError("preparation_cache_dir is required")


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
    base_version_id: str
    uri: str


def _seed_checkpoint(source: Path, target: Path) -> None:
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    if source.is_file():
        shutil.copy2(source, target / "model.safetensors")
        return
    if not source.is_dir():
        raise FileNotFoundError(f"launch checkpoint does not exist: {source}")
    for entry in source.iterdir():
        if entry.is_file():
            shutil.copy2(entry, target / entry.name)


def _checkpoint_files_state(
    locations: dict[str, tuple[Path, int, int]],
) -> dict[str, list[int]]:
    return {
        path.name: [path.stat().st_size, path.stat().st_mtime_ns]
        for path in sorted({item[0] for item in locations.values()})
    }


def _write_state(path: Path, state: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(state, sort_keys=True))
    temporary.replace(path)


def _source_identity(version: _S3Version) -> dict[str, str]:
    return {"uri": version.uri}


_Decompressor = Callable[[memoryview], Any]


_DECOMPRESSORS: dict[str, _Decompressor] = {
    "zstd": lambda data: zstandard.ZstdDecompressor().stream_reader(data),
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
        self.launch_checkpoint = Path(config.launch_checkpoint)
        self.s3 = s3
        self.cache = Path(config.preparation_cache_dir) / quote(model_name, safe="")
        self.local_checkpoint = self.cache / "checkpoint"
        self.state_path = self.cache / "state.json"
        self.lock_path = self.cache / ".lock"
        self.locations: dict[str, tuple[Path, int, int]] = {}
        self.decompressor: _Decompressor | None = None

    def initialize(self) -> None:
        self.cache.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            state = self._state()
            reusable = (
                state is not None
                and state.get("version") == self.initial_version
                and any(self.local_checkpoint.glob("*.safetensors"))
            )
            if reusable:
                self.locations, _ = index_checkpoint_tensors(self.local_checkpoint)
                reusable = state.get("files") == _checkpoint_files_state(self.locations)
            if not reusable:
                _seed_checkpoint(self.launch_checkpoint, self.local_checkpoint)
                self.locations, _ = index_checkpoint_tensors(self.local_checkpoint)
                _write_state(
                    self.state_path,
                    {
                        "version": self.initial_version,
                        "files": _checkpoint_files_state(self.locations),
                    },
                )

    def _state(self) -> dict | None:
        if not self.state_path.is_file():
            return None
        try:
            value = json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    @property
    def current_version(self) -> str:
        state = self._state()
        if state is None:
            raise RuntimeError("local checkpoint state is missing")
        return str(state["version"])

    def prepare(self, version: _S3Version) -> PreparedCheckpoint:
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            state = self._state()
            if state is None:
                raise RuntimeError("local checkpoint state is missing")
            if state.get("files") != _checkpoint_files_state(self.locations):
                raise RuntimeError("local checkpoint files changed outside ModelExpress")
            if state["version"] == version.version_id:
                if state.get("source") != _source_identity(version):
                    raise ValueError(
                        "prepared checkpoint came from a different canonical root"
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
            if state["version"] != version.base_version_id:
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
            weight_map, self.decompressor = _parse_delta_manifest(index_data)

            download_started = time.perf_counter()
            shards = self._download_deltas(weight_map, version.uri)
            download_time = time.perf_counter() - download_started

            apply_started = time.perf_counter()
            self._apply_shards(shards)
            _write_state(
                self.state_path,
                {
                    "version": version.version_id,
                    "source": _source_identity(version),
                    "files": _checkpoint_files_state(self.locations),
                },
            )
            return PreparedCheckpoint(
                target_version=version.version_id,
                path=self.local_checkpoint,
                metrics={
                    "perf/mx_receive_delta_index_download": index_download_time,
                    "perf/mx_receive_delta_download": download_time,
                    "perf/mx_receive_delta_apply": time.perf_counter() - apply_started,
                },
            )

    def _download_deltas(
        self,
        weight_map: dict[str, str],
        root_uri: str,
    ) -> dict[str, tuple[bytes, list[str]]]:
        if not weight_map:
            return {}
        by_file: dict[str, list[str]] = {}
        for name, filename in weight_map.items():
            by_file.setdefault(filename, []).append(name)
        parent_uri = root_uri.rsplit("/", 1)[0]
        shards = {}
        with ThreadPoolExecutor(
            max_workers=min(rl_envs.MX_S3_DOWNLOAD_WORKERS, len(by_file)),
            thread_name_prefix="modelexpress-s3-download-file",
        ) as pool:
            downloads = {
                filename: pool.submit(
                    self.s3.get,
                    f"{parent_uri}/{filename}",
                )
                for filename in by_file
            }
            for filename, names in by_file.items():
                try:
                    data = downloads[filename].result()
                except Exception as error:
                    raise RuntimeError(
                        f"canonical delta download failed for {filename!r}"
                    ) from error
                shards[filename] = (data, names)
        return shards

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
                    raise ValueError(
                        f"canonical delta byte size differs for {name!r}"
                    )
                if f"{checksum:08x}" != expected_checksum:
                    raise ValueError(
                        f"canonical target checksum differs for {name!r}"
                    )

            with ThreadPoolExecutor(
                max_workers=min(rl_envs.MX_REFIT_DELTA_WORKERS, len(items)),
                thread_name_prefix="modelexpress-delta-apply",
            ) as pool:
                list(pool.map(apply_one, items))
        finally:
            for file_handle, mapped in maps.values():
                mapped.close()
                file_handle.close()

    @contextmanager
    def installation(self, prepared: PreparedCheckpoint):
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_SH)
            state = self._state()
            if (
                state is None
                or state.get("version") != prepared.target_version
                or state.get("files") != _checkpoint_files_state(self.locations)
            ):
                raise ReceiverInstallError(
                    "prepared checkpoint changed before installation"
                )
            yield


class CanonicalS3GeneratorAdapter(GeneratorEngineAdapter):
    """Prepare one canonical S3 delta, then reload it through an engine hook."""

    def __init__(
        self,
        *,
        model_name: str,
        config: ObjectStorageGeneratorConfig,
    ) -> None:
        if config.storage_type is not ObjectStorageType.S3:
            raise ValueError("only S3 object storage is currently supported")
        self._s3 = S3Client(
            endpoint_url=config.endpoint_url,
            region_name=config.region_name,
        )
        self._checkpoint = _LocalCheckpoint(
            model_name=model_name,
            config=config,
            s3=self._s3,
        )
        self._checkpoint.initialize()
        self._active_staged: PreparedCheckpoint | None = None

    @property
    def supported_payload_formats(self) -> frozenset[WeightPayloadFormat]:
        return frozenset({WeightPayloadFormat.XOR_DELTA})

    def stage_weight(self, inputs: GeneratorTransferInputs) -> PreparedCheckpoint:
        if self._active_staged is not None:
            raise RuntimeError("release staged weight before staging another version")
        if inputs.payload_format is not WeightPayloadFormat.XOR_DELTA:
            raise ValueError("canonical S3 requires XOR_DELTA payloads")
        if not inputs.base_version_id:
            raise ValueError("canonical S3 version is missing base_version_id")
        if inputs.object_storage is None:
            raise ValueError("canonical S3 requires a version-level URI")
        if inputs.object_storage.storage_type is not ObjectStorageType.S3:
            raise ValueError("canonical S3 requires S3 object storage")
        if self._checkpoint.current_version not in {
            inputs.base_version_id,
            inputs.version_id,
        }:
            raise ValueError("canonical S3 target does not match the exact local base")
        version = _S3Version(
            version_id=inputs.version_id,
            base_version_id=inputs.base_version_id,
            uri=inputs.object_storage.uri,
        )
        started = time.perf_counter()
        try:
            staged = self._checkpoint.prepare(version)
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        staged.metrics["perf/mx_receive_prepare_time"] = time.perf_counter() - started
        self._active_staged = staged
        return staged

    def apply_weight(self, staged: object) -> dict[str, float]:
        if staged is not self._active_staged:
            raise RuntimeError("canonical S3 staged weight is no longer active")
        started = time.perf_counter()
        with self._checkpoint.installation(self._active_staged):
            self.install_prepared_checkpoint(self._active_staged)
        return {"perf/mx_receive_install_time": time.perf_counter() - started}

    def install_prepared_checkpoint(self, prepared: PreparedCheckpoint) -> None:
        """Load ``prepared.path`` into the live engine."""
        raise NotImplementedError

    def release_staged_weight(self, staged: object) -> None:
        if staged is not self._active_staged:
            raise RuntimeError("canonical S3 staged weight is no longer active")
        self._active_staged = None

    def close(self) -> None:
        self._active_staged = None
        self._s3.close()


__all__ = [
    "CanonicalS3GeneratorAdapter",
    "PreparedCheckpoint",
    "ReceiverInstallError",
    "ObjectStorageGeneratorConfig",
]
