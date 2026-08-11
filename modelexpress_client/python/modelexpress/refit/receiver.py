# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical S3 receiver and persistent checkpoint installer."""

from __future__ import annotations

import fcntl
import hashlib
import json
import mmap
import socket
import shutil
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np
import torch
import zstandard

from .. import envs
from .api import ReceiverRevisionState, ReceiverStatus, WeightUpdateResult
from .catalog import GrpcRevisionCatalog
from .manifest import RevisionState, S3Object
from .s3 import S3Client
from .source.canonical import bucket_parts, snapshot_digest
from .source.canonical import format_digest as canonical_format_digest


@dataclass(frozen=True)
class ReceiverConfig:
    model_id: str
    catalog_endpoint: str
    initial_version: str
    preparation_cache_dir: str | Path
    ready_timeout_seconds: float = 600.0
    s3_endpoint_url: str | None = None


class ReceiverInstallError(RuntimeError):
    def __init__(self, detail: str, mutation_started: bool) -> None:
        super().__init__(detail)
        self.mutation_started = mutation_started


def _download_worker_count(bucket_count: int) -> int:
    return min(max(1, envs.MX_REFIT_S3_DOWNLOAD_WORKERS), bucket_count)


@dataclass(frozen=True)
class PreparedRevision:
    target_version: str
    target_digest: str
    path: Path
    metrics: dict[str, float]


class ModelExpressWeightReceiver:
    def __init__(
        self,
        config: ReceiverConfig,
        receiver_id: str,
        model_runner,
    ) -> None:
        self.config = config
        self.receiver_id = receiver_id
        self.model_id = config.model_id
        self.model_runner = model_runner
        checkpoint, _, _ = model_runner.loader._prepare_weights(
            model_runner.model_config.model_path,
            model_runner.model_config.revision,
            False,
        )
        self.launch_checkpoint = Path(checkpoint)
        self.installed_version = config.initial_version
        self.installed_digest: str
        self.catalog = None
        self.s3 = None
        self.checkpoint: _LocalCheckpoint
        self.prepared: PreparedRevision | None = None
        self.state = None
        self.detail = ""
        self._metrics: dict[str, float] = {}

    def initialize(self) -> None:
        self.catalog = GrpcRevisionCatalog(self.config.catalog_endpoint)
        self.s3 = S3Client(endpoint_url=self.config.s3_endpoint_url)
        deadline = time.monotonic() + self.config.ready_timeout_seconds
        while True:
            try:
                launch = self.catalog.get_revision(
                    self.model_id, self.installed_version
                )
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        if launch.state not in {RevisionState.READY, RevisionState.COMMITTED}:
            raise ValueError("installed launch revision is not ready")
        self.checkpoint = _LocalCheckpoint(
            self.config, self.catalog, self.s3, self.launch_checkpoint
        )
        self.installed_digest = self.checkpoint.initialize(launch)
        self.state = ReceiverRevisionState.VERIFIED

    @property
    def prepared_identity(self):
        return self.prepared

    def install_prepared_checkpoint(self, prepared: PreparedRevision) -> None:
        try:
            from sglang.srt.configs.load_config import LoadConfig, LoadFormat
            from sglang.srt.model_loader.loader import (
                DefaultModelLoader,
                get_model_loader,
            )

            loader = get_model_loader(
                LoadConfig(
                    load_format=LoadFormat.SAFETENSORS,
                    download_dir=self.model_runner.server_args.download_dir,
                    model_loader_extra_config=(
                        self.model_runner.server_args.model_loader_extra_config
                    ),
                ),
                self.model_runner.model_config,
            )
            if not isinstance(loader, DefaultModelLoader):
                raise TypeError("ModelExpress requires DefaultModelLoader")
            source = SimpleNamespace(
                model_or_path=str(prepared.path),
                revision=None,
                prefix="",
                fall_back_to_pt=False,
                model_config=self.model_runner.model_config,
            )
            weights = loader._get_weights_iterator(source)
        except Exception as error:
            raise ReceiverInstallError(str(error), False) from error

        try:
            from sglang.srt.model_loader.utils import set_default_torch_dtype

            with set_default_torch_dtype(self.model_runner.model_config.dtype):
                loader.load_weights_and_postprocess(
                    self.model_runner.model,
                    weights,
                    torch.device(self.model_runner.device),
                )
            device = torch.get_device_module(self.model_runner.device)
            if hasattr(device, "synchronize"):
                device.synchronize()
        except Exception as error:
            raise ReceiverInstallError(str(error), True) from error

    def start_weight_update(self, version: str) -> None:
        if self.state is ReceiverRevisionState.POISONED:
            raise RuntimeError("poisoned receiver cannot install another update")
        self._metrics = {}
        self.prepared = None
        started = time.perf_counter()
        try:
            self.prepared = self.checkpoint.prepare(
                version, self.installed_version, self.installed_digest
            )
            self._metrics.update(self.prepared.metrics)
        finally:
            self._metrics["perf/mx_receive_prepare_time"] = (
                time.perf_counter() - started
            )
        self.detail = ""

    def update_weights(self, layers=None) -> WeightUpdateResult:
        if layers is not None:
            raise ValueError("ModelExpress supports complete-model updates only")
        if self.prepared is None:
            raise RuntimeError("no prepared ModelExpress update")
        if self.state is ReceiverRevisionState.POISONED:
            raise RuntimeError("poisoned receiver cannot install another update")
        target = self.prepared
        started = time.perf_counter()
        try:
            with self.checkpoint.installation(target):
                self.install_prepared_checkpoint(target)
        except ReceiverInstallError as error:
            self._metrics = {
                "perf/mx_receive_install_time": time.perf_counter() - started
            }
            self.state = (
                ReceiverRevisionState.POISONED
                if error.mutation_started
                else ReceiverRevisionState.FAILED
            )
            self.detail = str(error)
            return self._result(False)
        self.installed_version = target.target_version
        self.installed_digest = target.target_digest
        self.prepared = None
        self.state = ReceiverRevisionState.VERIFIED
        self._metrics = {"perf/mx_receive_install_time": time.perf_counter() - started}
        return self._result(True)

    def pop_metrics(self) -> dict[str, float]:
        metrics, self._metrics = self._metrics, {}
        return metrics

    def mark_poisoned(self, detail: str):
        self.state = ReceiverRevisionState.POISONED
        self.detail = detail
        return self._result(False)

    def status(self) -> ReceiverStatus:
        return ReceiverStatus(
            receiver_id=self.receiver_id,
            model_id=self.model_id,
            installed_version=self.installed_version,
            target_digest=self.installed_digest,
            state=self.state,
            detail=self.detail,
        )

    def _result(self, success: bool) -> WeightUpdateResult:
        return WeightUpdateResult(
            success=success,
            receiver_id=self.receiver_id,
            installed_version=self.installed_version,
            state=self.state,
            target_digest=self.installed_digest,
            detail=self.detail,
        )


def _seed_checkpoint(source: Path, target: Path) -> None:
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    if source.is_file():
        shutil.copy2(source, target / "model.safetensors")
        return
    for entry in source.iterdir():
        if entry.is_file():
            shutil.copy2(entry, target / entry.name)


_SAFETENSORS_DTYPES = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "F16": "float16",
    "BF16": "bfloat16",
    "F32": "float32",
    "F64": "float64",
}


def _tensor_locations(checkpoint: Path):
    locations = {}
    metadata = {}
    tied = set()
    config = checkpoint / "config.json"
    if config.is_file() and json.loads(config.read_text()).get(
        "tie_word_embeddings", False
    ):
        tied.add("lm_head.weight")
    for path in checkpoint.glob("*.safetensors"):
        with path.open("rb") as handle:
            (header_size,) = struct.unpack("<Q", handle.read(8))
            header = json.loads(handle.read(header_size))
        for name, item in header.items():
            if name == "__metadata__":
                continue
            name = name.removeprefix("module.")
            if name in tied:
                continue
            begin, end = item["data_offsets"]
            locations[name] = (path, 8 + header_size + begin, end - begin)
            metadata[name] = {
                "name": name,
                "shape": item["shape"],
                "dtype": _SAFETENSORS_DTYPES[item["dtype"]],
                "byte_size": end - begin,
            }
    return locations, metadata


def _checkpoint_identity(locations, metadata):
    values = {name: dict(item) for name, item in metadata.items()}
    maps = {}
    try:
        for path in {item[0] for item in locations.values()}:
            handle = path.open("rb")
            maps[path] = (
                handle,
                mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ),
            )
        for name, (path, offset, size) in locations.items():
            view = memoryview(maps[path][1])[offset : offset + size]
            try:
                values[name]["target_digest"] = (
                    f"sha256:{hashlib.sha256(view).hexdigest()}"
                )
            finally:
                view.release()
    finally:
        for handle, mapped in maps.values():
            mapped.close()
            handle.close()
    return canonical_format_digest(values), snapshot_digest(values)


def _checkpoint_files_state(locations):
    return {
        path.name: [path.stat().st_size, path.stat().st_mtime_ns]
        for path in sorted({item[0] for item in locations.values()})
    }


def _apply_bucket(data, locations, maps, root, descriptor, coverage) -> None:
    header, compressed = bucket_parts(data)
    if (
        header["model_id"] != root["model_id"]
        or header["base_version"] != root["base_version"]
        or header["target_version"] != root["target_version"]
        or header["base_digest"] != root["base_digest"]
        or header["format_digest"] != root["format_digest"]
        or header["ordinal"] != descriptor["ordinal"]
        or header["decoded_size"] != descriptor["decoded_size"]
        or header["compression"] != "zstd"
        or header["delta"] != "xor"
        or [item["name"] for item in header["entries"]] != descriptor["tensors"]
    ):
        raise ValueError("canonical bucket does not match its root descriptor")
    reader = zstandard.ZstdDecompressor().stream_reader(compressed)
    decoded_position = 0
    for entry in header["entries"]:
        name = entry["name"]
        if entry["offset"] != decoded_position:
            raise ValueError(f"{name} has a non-contiguous canonical delta offset")
        target_identity = coverage[name]
        if (
            target_identity.get("state") != "dirty"
            or target_identity.get("bucket_ordinal") != descriptor["ordinal"]
            or any(
                entry[field] != target_identity[field]
                for field in ("shape", "dtype", "byte_size", "target_digest")
            )
        ):
            raise ValueError(f"canonical bucket identity differs for {name}")
        path, file_offset, size = locations[name]
        if size != entry["byte_size"]:
            raise ValueError(f"{name} byte size differs from the local checkpoint")
        mapped = maps[path][1]
        target = np.frombuffer(mapped, dtype=np.uint8, count=size, offset=file_offset)
        try:
            hasher = hashlib.sha256()
            position = 0
            while position < size:
                block = reader.read(min(2 << 20, size - position))
                if not block:
                    break
                delta = np.frombuffer(block, dtype=np.uint8)
                region = target[position : position + delta.size]
                try:
                    np.bitwise_xor(region, delta, out=region)
                    hasher.update(region)
                finally:
                    del region
                position += delta.size
            if position != size:
                raise ValueError(
                    f"{name} delta byte size differs from canonical metadata"
                )
            digest = f"sha256:{hasher.hexdigest()}"
            if digest != entry["target_digest"]:
                raise ValueError(f"canonical target checksum differs for {name}")
            decoded_position += size
        finally:
            del target
    if decoded_position != header["decoded_size"]:
        raise ValueError("canonical bucket decoded size differs from its entries")
    reader.close()


def _write_state(path: Path, state: dict) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(state))
    temporary.replace(path)


class _LocalCheckpoint:
    def __init__(self, config, catalog, s3, launch_checkpoint):
        self.model_id = config.model_id
        self.initial_version = config.initial_version
        self.catalog = catalog
        self.s3 = s3
        self.launch_checkpoint = Path(launch_checkpoint)
        self.cache = Path(config.preparation_cache_dir) / quote(self.model_id, safe="")
        self.local_checkpoint = self.cache / "checkpoint"
        self.state_path = self.cache / "state.json"
        self.lock_path = self.cache / ".lock"
        self.locations = None
        self.metadata = None
        self.format_digest = None

    def initialize(self, launch) -> str:
        installed_digest = launch.manifest.target_digest
        self.format_digest = launch.manifest.format_digest
        self.cache.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            state = (
                json.loads(self.state_path.read_text())
                if self.state_path.exists()
                else None
            )
            seeded = (
                state is None
                or state.get("poisoned")
                or state.get("version") != self.initial_version
                or state.get("digest") != installed_digest
                or not any(self.local_checkpoint.glob("*.safetensors"))
            )
            if not seeded:
                self.locations, self.metadata = _tensor_locations(self.local_checkpoint)
                seeded = state.get("files") != _checkpoint_files_state(self.locations)
            if seeded:
                _seed_checkpoint(self.launch_checkpoint, self.local_checkpoint)
                self.locations, self.metadata = _tensor_locations(self.local_checkpoint)
            if canonical_format_digest(self.metadata) != self.format_digest:
                raise ValueError("local checkpoint format differs from revision 0")
            if seeded:
                actual_format, actual_digest = _checkpoint_identity(
                    self.locations, self.metadata
                )
                if (
                    actual_format != self.format_digest
                    or actual_digest != installed_digest
                ):
                    raise ValueError("local checkpoint differs from revision 0")
                _write_state(
                    self.state_path,
                    {
                        "version": self.initial_version,
                        "digest": installed_digest,
                        "format_digest": self.format_digest,
                        "files": _checkpoint_files_state(self.locations),
                    },
                )
            elif state["format_digest"] != self.format_digest:
                raise ValueError("existing local checkpoint state is not reusable")
        return installed_digest

    def prepare(
        self, version: str, base_version: str, base_digest: str
    ) -> PreparedRevision:
        record = self.catalog.get_revision(self.model_id, version)
        if record.state not in {RevisionState.READY, RevisionState.COMMITTED}:
            raise ValueError("revision is not ready")
        manifest = record.manifest
        if (
            manifest.base_version != base_version
            or manifest.base_digest != base_digest
            or manifest.format_digest != self.format_digest
        ):
            raise ValueError("revision does not match the installed exact base")

        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            state = json.loads(self.state_path.read_text())
            if state.get("poisoned"):
                raise ValueError("local checkpoint is poisoned")
            if state.get("files") != _checkpoint_files_state(self.locations):
                raise ValueError("local checkpoint files changed outside ModelExpress")
            if (
                state["version"] == version
                and state["digest"] == manifest.target_digest
            ):
                return PreparedRevision(
                    version,
                    manifest.target_digest,
                    self.local_checkpoint,
                    {
                        "perf/mx_receive_root_download": 0.0,
                        "perf/mx_receive_pool": 0.0,
                    },
                )
            if state["version"] != base_version or state["digest"] != base_digest:
                raise ValueError("local checkpoint does not match the exact base")
            if manifest.payload is None:
                raise ValueError("revision has no canonical payload")
            root_started = time.perf_counter()
            root_payload = self.s3.get(manifest.payload)
            root_download_time = time.perf_counter() - root_started
            root = json.loads(root_payload)
            if (
                root["model_id"] != self.model_id
                or root["base_version"] != base_version
                or root["target_version"] != version
                or root["base_digest"] != base_digest
                or root["target_digest"] != manifest.target_digest
                or root["format_digest"] != self.format_digest
            ):
                raise ValueError("root does not match the requested revision")
            coverage = {item["name"]: item for item in root["tensors"]}
            if len(coverage) != len(root["tensors"]) or set(coverage) != set(
                self.locations
            ):
                raise ValueError(
                    "root tensor coverage differs from the local checkpoint"
                )
            for name, item in self.metadata.items():
                target = coverage[name]
                if any(
                    target[field] != item[field]
                    for field in ("shape", "dtype", "byte_size")
                ):
                    raise ValueError(f"root metadata differs for {name}")
            identity = {
                name: {
                    field: item[field]
                    for field in (
                        "name",
                        "shape",
                        "dtype",
                        "byte_size",
                        "target_digest",
                    )
                }
                for name, item in coverage.items()
            }
            if (
                canonical_format_digest(identity) != self.format_digest
                or snapshot_digest(identity) != manifest.target_digest
            ):
                raise ValueError("root target identity differs from the revision")
            bucket_ordinals = [item["ordinal"] for item in root["buckets"]]
            bucket_names = [
                name for item in root["buckets"] for name in item["tensors"]
            ]
            dirty_names = [
                name for name, item in coverage.items() if item["state"] == "dirty"
            ]
            if (
                len(bucket_ordinals) != len(set(bucket_ordinals))
                or len(bucket_names) != len(set(bucket_names))
                or set(bucket_names) != set(dirty_names)
            ):
                raise ValueError("root buckets do not cover the dirty tensor set")
            _write_state(self.state_path, {**state, "poisoned": True})
            maps = {}
            for path in {item[0] for item in self.locations.values()}:
                file_handle = path.open("r+b")
                maps[path] = (file_handle, mmap.mmap(file_handle.fileno(), 0))

            def apply(bucket):
                location = bucket["object"]
                payload = self.s3.get(
                    S3Object(
                        bucket=location["bucket"],
                        key=location["key"],
                        checksum=location["checksum"],
                        object_version=location.get("object_version"),
                    )
                )
                _apply_bucket(payload, self.locations, maps, root, bucket, coverage)

            bucket_pool_time = 0.0
            try:
                buckets = root["buckets"]
                if buckets:
                    pool_started = time.perf_counter()
                    with ThreadPoolExecutor(
                        max_workers=_download_worker_count(len(buckets))
                    ) as pool:
                        list(pool.map(apply, buckets))
                    bucket_pool_time = time.perf_counter() - pool_started
            finally:
                for file_handle, mapped in maps.values():
                    mapped.close()
                    file_handle.close()
            _write_state(
                self.state_path,
                {
                    "version": version,
                    "digest": manifest.target_digest,
                    "format_digest": self.format_digest,
                    "files": _checkpoint_files_state(self.locations),
                },
            )
            return PreparedRevision(
                version,
                manifest.target_digest,
                self.local_checkpoint,
                {
                    "perf/mx_receive_root_download": root_download_time,
                    "perf/mx_receive_pool": bucket_pool_time,
                },
            )

    @contextmanager
    def installation(self, prepared: PreparedRevision):
        with self.lock_path.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_SH)
            state = json.loads(self.state_path.read_text())
            if (
                state.get("poisoned")
                or state["version"] != prepared.target_version
                or state["digest"] != prepared.target_digest
                or state.get("files") != _checkpoint_files_state(self.locations)
            ):
                raise ReceiverInstallError(
                    "prepared checkpoint changed before installation",
                    False,
                )
            yield


def build_weight_receiver(model_runner):
    args = model_runner.server_args
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    receiver = ModelExpressWeightReceiver(
        ReceiverConfig(
            model_id=args.modelexpress_model_id,
            catalog_endpoint=args.modelexpress_catalog_endpoint,
            initial_version=args.modelexpress_initial_version,
            preparation_cache_dir=args.modelexpress_preparation_cache_dir,
            ready_timeout_seconds=args.modelexpress_ready_timeout_seconds,
            s3_endpoint_url=args.modelexpress_delta_s3_endpoint,
        ),
        f"{socket.gethostname()}:{rank}",
        model_runner,
    )
    receiver.initialize()
    return receiver
