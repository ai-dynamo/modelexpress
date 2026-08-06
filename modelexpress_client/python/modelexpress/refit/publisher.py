# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Miles-specific source-rank canonical publisher."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
import zstandard

from .api import PublisherConfig
from .catalog import GrpcRevisionCatalog
from .manifest import RevisionManifest, RevisionState
from .s3 import S3Uploader
from .source.canonical import (
    canonical_json,
    encode_compressed_bucket,
    format_digest,
    load_hf_snapshot,
    snapshot_digest,
)


def _key(model_id: str, version: str, filename: str) -> str:
    return (
        f"models/{quote(model_id, safe='')}/revisions/"
        f"{quote(version, safe='')}/canonical/{filename}"
    )


def _object(stored, size: int) -> dict[str, object]:
    result = {
        "bucket": stored.bucket,
        "checksum": stored.checksum,
        "key": stored.key,
        "size": size,
    }
    if stored.object_version is not None:
        result["object_version"] = stored.object_version
    return result


def _merge_metadata(contributions) -> tuple[dict[str, dict], dict[str, int]]:
    metadata = {}
    owners = {}
    for rank, local in sorted(contributions):
        for name, item in local.items():
            if name in metadata:
                if item["target_digest"] != metadata[name]["target_digest"]:
                    raise RuntimeError(
                        f"{name!r} differs between source ranks {owners[name]} and {rank}"
                    )
                continue
            metadata[name] = item
            owners[name] = rank
    return metadata, owners


def _bucket_groups(names: list[str], metadata: dict[str, dict], limit: int):
    groups = []
    current = []
    size = 0
    for name in names:
        tensor_size = metadata[name]["byte_size"]
        if current and size + tensor_size > limit:
            groups.append(current)
            current = []
            size = 0
        current.append(name)
        size += tensor_size
    if current:
        groups.append(current)
    return groups


class Publisher:
    def __init__(
        self,
        *,
        launch_checkpoint: str | Path,
        bucket_bytes: int = 256 * 1024 * 1024,
        group=None,
        catalog=None,
        s3_client=None,
        sleep=time.sleep,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self.launch_checkpoint = Path(launch_checkpoint)
        self.bucket_bytes = bucket_bytes
        self.group = group
        self.catalog = catalog
        self.s3_client = s3_client
        self.sleep = sleep
        self.poll_interval_seconds = poll_interval_seconds
        self.uploader = None
        self.snapshot = {}
        self.metadata = {}
        self.captured = False

    def initialize(self, config: PublisherConfig) -> None:
        self.config = config
        self.distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        self.rank = torch.distributed.get_rank() if self.distributed else 0
        self.world = (
            torch.distributed.get_world_size(self.group) if self.distributed else 1
        )

        identity = None
        if self.rank == 0:
            _snapshot, _metadata, launch_format, launch_digest = load_hf_snapshot(
                self.launch_checkpoint
            )
            identity = (launch_format, launch_digest)
            if self.catalog is None:
                self.catalog = GrpcRevisionCatalog(config.catalog_endpoint)
        identities = self._gather(identity)
        self.format_digest, self.target_digest = next(
            item for item in identities if item is not None
        )
        self.current_version = "0"
        self.pending_version = None
        self.pending_digest = None

    def publish_version(
        self,
        version: str,
        *,
        base_version: str | None = None,
        gather_hf_buckets=None,
    ) -> None:
        if version == "0":
            if base_version is not None or self.current_version != "0":
                raise RuntimeError("version 0 is the launch revision")
            if self.rank == 0:
                self.catalog.publish_revision(
                    RevisionManifest(
                        model_id=self.config.model_id,
                        target_version="0",
                        target_digest=self.target_digest,
                        format_digest=self.format_digest,
                    )
                )
            self.pending_version = "0"
            self.pending_digest = self.target_digest
            self._barrier()
            return

        if base_version != self.current_version:
            raise RuntimeError(
                f"base {base_version!r} does not match current version "
                f"{self.current_version!r}"
            )
        if gather_hf_buckets is None or not self.captured:
            raise RuntimeError("Miles source-rank baseline is not captured")

        if self.rank == 0:
            base = self.catalog.get_revision(self.config.model_id, base_version)
            if (
                base.state is not RevisionState.COMMITTED
                or base.manifest.target_digest != self.target_digest
                or base.manifest.format_digest != self.format_digest
            ):
                raise RuntimeError("catalog base does not match the current snapshot")
        self._barrier()

        deltas, checksums = self._encode_delta(gather_hf_buckets)
        self._drop_duplicate_names(deltas, checksums)
        metadata, owners = self._collect_metadata()
        if format_digest(metadata) != self.format_digest:
            raise RuntimeError("canonical format changed during publication")
        target_digest = snapshot_digest(metadata)
        self._publish_root(
            version,
            base_version,
            deltas,
            metadata,
            owners,
            target_digest,
        )
        self.pending_version = version
        self.pending_digest = target_digest
        self._barrier()

    def capture_baseline(self, gather_hf_buckets, read_hf_tensor) -> None:
        def seed_bucket(bucket, _pbar=None):
            for name, tensor in bucket:
                name = name.removeprefix("module.")
                self._record_metadata(name, tensor)
                try:
                    value = read_hf_tensor(name)
                except KeyError:
                    value = (
                        tensor.detach()
                        .cpu()
                        .contiguous()
                        .view(torch.uint8)
                        .numpy()
                        .reshape(-1)
                    )
                self.snapshot[name] = np.asarray(value, dtype=np.uint8).copy()

        gather_hf_buckets(seed_bucket)
        metadata, _owners = self._collect_metadata()
        if (
            format_digest(metadata) != self.format_digest
            or snapshot_digest(metadata) != self.target_digest
        ):
            raise RuntimeError(
                "Miles source-rank baseline differs from launch revision"
            )
        self.captured = True

    def wait_for_commit(self, version: str, completion=None) -> None:
        if self.pending_version != version:
            raise RuntimeError(f"revision {version!r} is not pending")
        if self.rank == 0:
            while (
                self.catalog.get_revision(self.config.model_id, version).state
                is not RevisionState.COMMITTED
            ):
                if completion is not None and completion.done():
                    completion.result()
                self.sleep(self.poll_interval_seconds)
        self._barrier()
        self.current_version = version
        self.target_digest = self.pending_digest
        self.pending_version = None
        self.pending_digest = None

    def deregister(self) -> None:
        if self.uploader is not None:
            self.uploader.close()
        close = getattr(self.catalog, "close", None)
        if close is not None:
            close()

    def _encode_delta(self, gather_hf_buckets):
        deltas = {}
        checksums = {}
        compressor = zstandard.ZstdCompressor(level=1)

        def encode_bucket(bucket, _pbar=None):
            for name, tensor in bucket:
                name = name.removeprefix("module.")
                self._record_metadata(name, tensor)
                data = (
                    tensor.detach()
                    .cpu()
                    .contiguous()
                    .view(torch.uint8)
                    .numpy()
                    .reshape(-1)
                    .copy()
                )
                old = self.snapshot[name]
                if len(data) != len(old):
                    raise RuntimeError(f"{name} changed byte size")
                delta = np.bitwise_xor(data, old)
                self.snapshot[name] = data
                if np.any(delta):
                    digest = f"sha256:{hashlib.sha256(data.tobytes()).hexdigest()}"
                    deltas[name] = compressor.compress(delta.tobytes())
                    checksums[name] = digest

        gather_hf_buckets(encode_bucket)
        return deltas, checksums

    def _drop_duplicate_names(self, deltas, checksums) -> None:
        contributions = self._gather((self.rank, checksums))
        for rank, other in sorted(contributions):
            if rank >= self.rank:
                break
            for name in deltas.keys() & other.keys():
                if other[name] != checksums[name]:
                    raise RuntimeError(
                        f"{name!r} published by rank {rank} and rank {self.rank} "
                        "with different bytes"
                    )
                del deltas[name]
                del checksums[name]

    def _record_metadata(self, name, tensor) -> None:
        self.metadata[name] = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "byte_size": tensor.numel() * tensor.element_size(),
        }

    def _collect_metadata(self):
        local = {}
        for name, item in self.metadata.items():
            if name not in self.snapshot:
                continue
            data = self.snapshot[name].tobytes()
            local[name] = {
                **item,
                "target_digest": f"sha256:{hashlib.sha256(data).hexdigest()}",
            }
        return _merge_metadata(self._gather((self.rank, local)))

    def _publish_root(
        self,
        version,
        base_version,
        deltas,
        metadata,
        owners,
        target_digest,
    ) -> None:
        local_metadata = {
            name: item for name, item in metadata.items() if owners[name] == self.rank
        }
        groups = _bucket_groups(
            sorted(name for name in deltas if owners[name] == self.rank),
            local_metadata,
            self.bucket_bytes,
        )
        counts = sorted(self._gather((self.rank, len(groups))))
        offset = sum(count for rank, count in counts if rank < self.rank)
        total = sum(count for _rank, count in counts)
        descriptors = []
        dirty_ordinals = {}
        for local_ordinal, names in enumerate(groups):
            ordinal = offset + local_ordinal
            data, decoded_size, tensor_names = encode_compressed_bucket(
                model_id=self.config.model_id,
                base_version=base_version,
                target_version=version,
                base_digest=self.target_digest,
                format_digest=self.format_digest,
                ordinal=ordinal,
                names=names,
                compressed_deltas=deltas,
                metadata=local_metadata,
            )
            stored = self._uploader().put(
                _key(
                    self.config.model_id,
                    version,
                    f"bucket-{ordinal:08d}-of-{total:08d}.mxcd",
                ),
                data,
            )
            descriptors.append(
                {
                    "decoded_size": decoded_size,
                    "object": _object(stored, len(data)),
                    "ordinal": ordinal,
                    "tensors": list(tensor_names),
                }
            )
            for name in names:
                dirty_ordinals[name] = ordinal

        coverage = []
        for name, item in local_metadata.items():
            value = {**item, "state": "clean"}
            if name in dirty_ordinals:
                value["state"] = "dirty"
                value["bucket_ordinal"] = dirty_ordinals[name]
            coverage.append(value)
        all_descriptors = self._gather((self.rank, descriptors))
        all_coverage = self._gather((self.rank, coverage))
        if self.rank != 0:
            return
        buckets = sorted(
            (item for _rank, values in all_descriptors for item in values),
            key=lambda item: item["ordinal"],
        )
        tensors = sorted(
            (item for _rank, values in all_coverage for item in values),
            key=lambda item: item["name"],
        )
        root = canonical_json(
            {
                "base_digest": self.target_digest,
                "base_version": base_version,
                "buckets": buckets,
                "encoding": {"compression": "zstd", "delta": "xor"},
                "format_digest": self.format_digest,
                "model_id": self.config.model_id,
                "schema": "mx.canonical.delta.v0",
                "target_digest": target_digest,
                "target_version": version,
                "tensors": tensors,
            }
        )
        payload = self._uploader().put(
            _key(self.config.model_id, version, "root.json"), root
        )
        self.catalog.publish_revision(
            RevisionManifest(
                model_id=self.config.model_id,
                target_version=version,
                base_version=base_version,
                base_digest=self.target_digest,
                target_digest=target_digest,
                format_digest=self.format_digest,
                payload=payload,
            )
        )

    def _uploader(self):
        if self.uploader is None:
            self.uploader = S3Uploader(self.config.s3, client=self.s3_client)
        return self.uploader

    def _gather(self, value):
        if not self.distributed:
            return [value]
        values = [None] * self.world
        torch.distributed.all_gather_object(values, value, group=self.group)
        return values

    def _barrier(self) -> None:
        if self.distributed:
            torch.distributed.barrier(group=self.group)
