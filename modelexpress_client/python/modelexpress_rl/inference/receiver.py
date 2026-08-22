# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical S3 checkpoint preparation for generator refit."""

from __future__ import annotations

import fcntl
import hashlib
import json
import mmap
import shutil
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import numpy as np
import zstandard

from modelexpress_rl.s3 import S3Client, S3Object
from modelexpress_rl.train import WeightPayloadFormat
from modelexpress_rl.utils import index_checkpoint_tensors, read_safetensors_header
from modelexpress_rl.version import CANONICAL_DELTA_SOURCE_SLOT

from .adapter import (
    GeneratorEngineAdapter,
    GeneratorTransferInputs,
    S3GeneratorSource,
)


@dataclass(frozen=True)
class S3GeneratorConfig:
    """Canonical checkpoint and S3 settings for one generator rank."""

    initial_base_version_id: str
    launch_checkpoint: str | Path
    preparation_cache_dir: str | Path
    endpoint_url: str | None = None
    region_name: str | None = None

    def __post_init__(self) -> None:
        if not self.initial_base_version_id.strip():
            raise ValueError("initial_base_version_id is required")
        if not str(self.launch_checkpoint).strip():
            raise ValueError("launch_checkpoint is required")
        if not str(self.preparation_cache_dir).strip():
            raise ValueError("preparation_cache_dir is required")


class ReceiverInstallError(RuntimeError):
    """An engine reload failed before or after live mutation began."""

    def __init__(self, detail: str, *, mutation_started: bool) -> None:
        super().__init__(detail)
        self.mutation_started = mutation_started


class PoisonedCheckpointError(Exception):
    """The private checkpoint requires recovery before another delta."""


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
    root: S3Object
    manifest_digest: str


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


def _validate_filename(filename: str) -> None:
    path = PurePosixPath(filename)
    if not filename or path.name != filename or filename in {".", ".."}:
        raise ValueError(f"invalid delta shard filename {filename!r}")


def _source_identity(version: _S3Version) -> dict[str, str | None]:
    return {
        "bucket": version.root.bucket,
        "key": version.root.key,
        "object_version": version.root.object_version,
        "checksum": version.root.checksum,
        "manifest_digest": version.manifest_digest,
    }


class _LocalCheckpoint:
    """One crash-safe, host-local checkpoint updated under an exact-base lock."""

    def __init__(
        self,
        *,
        model_name: str,
        config: S3GeneratorConfig,
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

    def initialize(self) -> None:
        self.cache.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            state = self._state()
            reusable = (
                state is not None
                and not state.get("poisoned")
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
        if state is None or state.get("poisoned"):
            raise PoisonedCheckpointError("local checkpoint is poisoned")
        return str(state["version"])

    def prepare(self, version: _S3Version) -> PreparedCheckpoint:
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            state = self._state()
            if state is None or state.get("poisoned"):
                raise PoisonedCheckpointError("local checkpoint is poisoned")
            if state.get("files") != _checkpoint_files_state(self.locations):
                _write_state(self.state_path, {**state, "poisoned": True})
                raise PoisonedCheckpointError(
                    "local checkpoint files changed outside ModelExpress"
                )
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
                index_data = self.s3.get(version.root)
            except Exception as error:
                raise RuntimeError("canonical root download failed") from error
            index_download_time = time.perf_counter() - started
            if hashlib.sha256(index_data).hexdigest() != version.manifest_digest:
                raise ValueError("canonical root manifest digest mismatch")
            index = self._validate_index(index_data, version)
            shards = self._download_shards(index, version.root)

            _write_state(self.state_path, {**state, "poisoned": True})
            apply_started = time.perf_counter()
            try:
                self._apply_shards(shards)
            except Exception as error:
                raise PoisonedCheckpointError(
                    f"canonical delta left the local checkpoint poisoned: {error}"
                ) from error
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
                    "perf/mx_receive_delta_apply": time.perf_counter() - apply_started,
                },
            )

    def _validate_index(self, data: bytes, version: _S3Version) -> dict:
        try:
            index = json.loads(data)
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("canonical root is not valid JSON") from error
        if not isinstance(index, dict):
            raise ValueError("canonical root must be a JSON object")
        metadata = index.get("metadata")
        weight_map = index.get("weight_map")
        expected = {
            "version": version.version_id,
            "base_version": version.base_version_id,
            "delta_encoding": "xor",
            "compression_format": "zstd",
            "checksum_format": "adler32",
        }
        if metadata != expected:
            raise ValueError(
                "canonical root metadata does not match the requested version"
            )
        if not isinstance(weight_map, dict) or any(
            not isinstance(name, str) or not isinstance(filename, str)
            for name, filename in weight_map.items()
        ):
            raise ValueError("canonical root weight_map must map strings to strings")
        unknown = set(weight_map) - set(self.locations)
        if unknown:
            raise ValueError(f"canonical root contains unknown tensor {min(unknown)!r}")
        for filename in set(weight_map.values()):
            _validate_filename(filename)
        return index

    def _download_shards(
        self,
        index: dict,
        root: S3Object,
    ) -> dict[str, list[tuple[str, bytes, str]]]:
        weight_map: dict[str, str] = index["weight_map"]
        if not weight_map:
            return {}
        by_file: dict[str, set[str]] = {}
        for name, filename in weight_map.items():
            by_file.setdefault(filename, set()).add(name)
        parent = PurePosixPath(root.key).parent
        shards = {}
        for filename, names in sorted(by_file.items()):
            try:
                data = self.s3.get_key(
                    bucket=root.bucket,
                    key=str(parent / filename),
                )
            except Exception as error:
                raise RuntimeError(
                    f"canonical shard download failed for {filename!r}"
                ) from error
            shards[filename] = self._decode_shard(data, filename, names)
        return shards

    def _apply_shards(
        self,
        shards: dict[str, list[tuple[str, bytes, str]]],
    ) -> None:
        if not shards:
            return

        maps: dict[Path, tuple[Any, mmap.mmap]] = {}
        try:
            for path in {location[0] for location in self.locations.values()}:
                file_handle = path.open("r+b")
                maps[path] = (file_handle, mmap.mmap(file_handle.fileno(), 0))
            for entries in shards.values():
                for name, compressed, expected_checksum in entries:
                    path, file_offset, size = self.locations[name]
                    delta = zstandard.ZstdDecompressor().decompress(
                        compressed,
                        max_output_size=size,
                    )
                    if len(delta) != size:
                        raise ValueError(
                            f"canonical delta byte size differs for {name!r}"
                        )
                    target = np.frombuffer(
                        maps[path][1],
                        dtype=np.uint8,
                        count=size,
                        offset=file_offset,
                    )
                    try:
                        np.bitwise_xor(
                            target,
                            np.frombuffer(delta, dtype=np.uint8),
                            out=target,
                        )
                        actual_checksum = f"{zlib.adler32(target):08x}"
                    finally:
                        del target
                    if actual_checksum != expected_checksum:
                        raise ValueError(
                            f"canonical target checksum differs for {name!r}"
                        )
            for _, mapped in maps.values():
                mapped.flush()
        finally:
            for file_handle, mapped in maps.values():
                mapped.close()
                file_handle.close()

    def _decode_shard(
        self,
        data: bytes,
        filename: str,
        expected_names: set[str],
    ) -> list[tuple[str, bytes, str]]:
        header, data_start = read_safetensors_header(data, repr(filename))
        checksums = header.pop("__metadata__", None)
        if (
            not isinstance(checksums, dict)
            or set(header) != expected_names
            or set(checksums) != expected_names
        ):
            raise ValueError(f"safetensors entries do not match {filename!r}")
        ordered = sorted(header.items(), key=lambda item: item[1]["data_offsets"][0])
        position = 0
        decoded = []
        for name, info in ordered:
            if not isinstance(info, dict):
                raise ValueError(f"invalid safetensors entry for {name!r}")
            offsets = info.get("data_offsets")
            if (
                info.get("dtype") != "U8"
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or offsets[0] != position
                or not isinstance(offsets[1], int)
                or offsets[1] < offsets[0]
            ):
                raise ValueError(f"invalid canonical delta entry for {name!r}")
            begin, end = offsets
            shape = info.get("shape")
            if shape != [end - begin] or data_start + end > len(data):
                raise ValueError(f"invalid canonical delta size for {name!r}")
            expected_checksum = checksums.get(name)
            if not isinstance(expected_checksum, str) or len(expected_checksum) != 8:
                raise ValueError(f"invalid canonical checksum for {name!r}")
            try:
                int(expected_checksum, 16)
            except ValueError as error:
                raise ValueError(f"invalid canonical checksum for {name!r}") from error
            compressed = data[data_start + begin : data_start + end]
            if not compressed.startswith(b"\x28\xb5\x2f\xfd"):
                raise ValueError(f"canonical delta is not zstd for {name!r}")
            expected_size = self.locations[name][2]
            reader = zstandard.ZstdDecompressor().stream_reader(compressed)
            decoded_size = 0
            try:
                while block := reader.read(2 << 20):
                    decoded_size += len(block)
            finally:
                reader.close()
            if decoded_size != expected_size:
                raise ValueError(f"canonical delta byte size differs for {name!r}")
            decoded.append((name, compressed, expected_checksum))
            position = end
        if data_start + position != len(data):
            raise ValueError(f"safetensors data has unused bytes in {filename!r}")
        return decoded

    @contextmanager
    def installation(self, prepared: PreparedCheckpoint):
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_SH)
            state = self._state()
            if (
                state is None
                or state.get("poisoned")
                or state.get("version") != prepared.target_version
                or state.get("files") != _checkpoint_files_state(self.locations)
            ):
                raise ReceiverInstallError(
                    "prepared checkpoint changed before installation",
                    mutation_started=False,
                )
            yield


class CanonicalS3GeneratorAdapter(GeneratorEngineAdapter):
    """Prepare one canonical S3 delta, then reload it through an engine hook."""

    def __init__(
        self,
        *,
        model_name: str,
        config: S3GeneratorConfig,
    ) -> None:
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
        self._poisoned = False

    @property
    def supported_payload_formats(self) -> frozenset[WeightPayloadFormat]:
        return frozenset({WeightPayloadFormat.XOR_DELTA})

    def stage_weight(self, inputs: GeneratorTransferInputs) -> PreparedCheckpoint:
        if self._poisoned:
            raise PoisonedCheckpointError(
                "poisoned generator cannot install another update"
            )
        if self._active_staged is not None:
            raise RuntimeError("release staged weight before staging another version")
        if inputs.payload_format is not WeightPayloadFormat.XOR_DELTA:
            raise ValueError("canonical S3 requires XOR_DELTA payloads")
        if not inputs.base_version_id:
            raise ValueError("canonical S3 version is missing base_version_id")
        if len(inputs.sources) != 1:
            raise ValueError("canonical S3 requires one global root source")
        source = inputs.sources[0]
        if source.source_slot_id != CANONICAL_DELTA_SOURCE_SLOT or not isinstance(
            source.transport, S3GeneratorSource
        ):
            raise ValueError("canonical S3 requires the canonical.delta.root source")
        if self._checkpoint.current_version not in {
            inputs.base_version_id,
            inputs.version_id,
        }:
            raise ValueError("canonical S3 target does not match the exact local base")
        version = _S3Version(
            version_id=inputs.version_id,
            base_version_id=inputs.base_version_id,
            root=source.transport.location,
            manifest_digest=source.manifest_digest,
        )
        started = time.perf_counter()
        try:
            staged = self._checkpoint.prepare(version)
        except PoisonedCheckpointError:
            self._poisoned = True
            raise
        except ValueError as error:
            raise RuntimeError(str(error)) from error
        staged.metrics["perf/mx_receive_prepare_time"] = time.perf_counter() - started
        self._active_staged = staged
        return staged

    def apply_weight(self, staged: object) -> dict[str, float]:
        if self._poisoned or staged is not self._active_staged:
            raise RuntimeError("canonical S3 staged weight is no longer active")
        started = time.perf_counter()
        try:
            with self._checkpoint.installation(self._active_staged):
                self.install_prepared_checkpoint(self._active_staged)
        except ReceiverInstallError as error:
            self._poisoned = error.mutation_started
            raise
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
    "CANONICAL_DELTA_SOURCE_SLOT",
    "CanonicalS3GeneratorAdapter",
    "PoisonedCheckpointError",
    "PreparedCheckpoint",
    "ReceiverInstallError",
    "S3GeneratorConfig",
]
