# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-rank canonical publisher for Megatron-based trainers."""

from __future__ import annotations

import hashlib
import os
import queue
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch

from .. import envs
from .api import PublisherConfig
from .catalog import GrpcRevisionCatalog
from .delta import compute_delta, encode_bucket
from .manifest import RevisionManifest, RevisionState
from .s3 import S3Client
from .source.canonical import (
    canonical_json,
    format_digest,
    load_hf_snapshot,
    snapshot_digest,
)


NUM_WORKERS = min(32, os.cpu_count() or 8)
POLL_INTERVAL_SECONDS = 1.0


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


def _compute_staged_delta(name, data, size, pinned, base, free_buffers):
    if pinned:
        current = np.empty(size, dtype=np.uint8)
        np.copyto(current, data.numpy()[:size])
        free_buffers.put(data)
    else:
        current = data
    raw_delta, target_digest, changed_bytes = compute_delta(current, base)
    return name, current, raw_delta, target_digest, changed_bytes


class Publisher:
    def __init__(
        self,
        launch_checkpoint: str | Path,
        bucket_bytes: int = 256 * 1024 * 1024,
        group=None,
    ) -> None:
        self.launch_checkpoint = Path(launch_checkpoint)
        self.bucket_bytes = bucket_bytes
        self.group = group
        self.catalog = None
        self.s3 = None
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
            self.catalog = GrpcRevisionCatalog(config.catalog_endpoint)
        self.s3 = S3Client(
            endpoint_url=config.s3.endpoint_url,
            region_name=config.s3.region_name,
        )
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
        base_version: str | None = None,
        gather_hf_buckets=None,
    ) -> None:
        if self.pending_version is not None:
            raise RuntimeError(f"revision {self.pending_version!r} is still pending")
        if version == "0":
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
            return

        if base_version != self.current_version:
            raise RuntimeError(
                f"base {base_version!r} does not match current version "
                f"{self.current_version!r}"
            )
        capture_started = time.monotonic()
        raw_deltas, changed_bytes = self._capture_deltas(gather_hf_buckets)
        capture_seconds = time.monotonic() - capture_started
        metadata, owners = _merge_metadata(
            self._gather((self.rank, self._local_metadata()))
        )
        for name in list(raw_deltas):
            if owners[name] != self.rank:
                del raw_deltas[name]
                del changed_bytes[name]
        if format_digest(metadata) != self.format_digest:
            raise RuntimeError("canonical format changed during publication")
        target_digest = snapshot_digest(metadata)
        publish_started = time.monotonic()
        wire_bytes, setup_seconds, pool_seconds, finalize_seconds = (
            self._publish_delta_index(
                version,
                base_version,
                raw_deltas,
                metadata,
                owners,
                target_digest,
            )
        )
        publish_seconds = time.monotonic() - publish_started
        self._metrics = self._collect_metrics(
            (
                capture_seconds,
                publish_seconds,
                setup_seconds,
                pool_seconds,
                finalize_seconds,
                sum(changed_bytes.values()),
                wire_bytes,
            ),
            sum(item["byte_size"] for item in metadata.values()),
        )
        self.pending_version = version
        self.pending_digest = target_digest

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
        metadata, _owners = _merge_metadata(
            self._gather((self.rank, self._local_metadata()))
        )
        if (
            format_digest(metadata) != self.format_digest
            or snapshot_digest(metadata) != self.target_digest
        ):
            raise RuntimeError("source-rank baseline differs from launch revision")
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
                time.sleep(POLL_INTERVAL_SECONDS)
        self.current_version = version
        self.target_digest = self.pending_digest
        self.pending_version = None
        self.pending_digest = None

    def _collect_metrics(self, local_metrics, total_bytes: int) -> dict[str, float]:
        phase_metrics = self._gather(local_metrics)
        return {
            "perf/update_weights_density": sum(item[5] for item in phase_metrics)
            / max(total_bytes, 1),
            "perf/update_weights_wire_bytes": sum(item[6] for item in phase_metrics),
            "perf/mx_encode_delta": max(item[0] for item in phase_metrics),
            "perf/mx_publish_time": max(item[1] for item in phase_metrics),
            "perf/mx_publish_setup": max(item[2] for item in phase_metrics),
            "perf/mx_publish_pool": max(item[3] for item in phase_metrics),
            "perf/mx_publish_finalize": max(item[4] for item in phase_metrics),
        }

    def pop_metrics(self) -> dict[str, float]:
        metrics, self._metrics = getattr(self, "_metrics", {}), {}
        return metrics

    def deregister(self) -> None:
        for resource in (self.s3, self.catalog):
            close = getattr(resource, "close", None)
            if close is not None:
                close()

    def _capture_deltas(self, gather_hf_buckets):
        raw_deltas = {}
        changed_bytes = {}
        max_bytes = max(
            (int(value.nbytes) for value in self.snapshot.values()), default=0
        )
        free_buffers = queue.Queue()
        use_pinned = max_bytes <= 32 << 30
        try:
            count = max(1, min(2 * NUM_WORKERS, (32 << 30) // max(max_bytes, 1)))
            if not use_pinned:
                raise RuntimeError("tensor exceeds pinned buffer budget")
            for _ in range(count):
                free_buffers.put(
                    torch.empty(max_bytes, dtype=torch.uint8, pin_memory=True)
                )
        except RuntimeError:
            free_buffers = queue.Queue()
            use_pinned = False

        inflight = deque()
        pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)

        def capture_bucket(bucket, _pbar=None):
            for name, tensor in bucket:
                name = name.removeprefix("module.")
                self._record_metadata(name, tensor)
                flat = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
                size = int(flat.numel())
                if use_pinned and size <= max_bytes:
                    data = free_buffers.get()
                    data[:size].copy_(flat, non_blocking=True)
                    torch.cuda.current_stream().synchronize()
                    pinned = True
                else:
                    data = flat.cpu().numpy()
                    pinned = False
                inflight.append(
                    pool.submit(
                        _compute_staged_delta,
                        name,
                        data,
                        size,
                        pinned,
                        self.snapshot[name],
                        free_buffers,
                    )
                )
                if len(inflight) >= 2 * NUM_WORKERS:
                    name, current, raw_delta, digest, changed = (
                        inflight.popleft().result()
                    )
                    self.snapshot[name] = current
                    if changed:
                        raw_deltas[name] = raw_delta
                        changed_bytes[name] = changed
                        self.metadata[name]["target_digest"] = digest

        try:
            gather_hf_buckets(capture_bucket)
            while inflight:
                name, current, raw_delta, digest, changed = inflight.popleft().result()
                self.snapshot[name] = current
                if changed:
                    raw_deltas[name] = raw_delta
                    changed_bytes[name] = changed
                    self.metadata[name]["target_digest"] = digest
        finally:
            pool.shutdown()
        return raw_deltas, changed_bytes

    def _record_metadata(self, name, tensor) -> None:
        digest = self.metadata.get(name, {}).get("target_digest")
        self.metadata[name] = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "byte_size": tensor.numel() * tensor.element_size(),
        }
        if digest is not None:
            self.metadata[name]["target_digest"] = digest

    def _local_metadata(self):
        local = {}
        for name, item in self.metadata.items():
            if name not in self.snapshot:
                continue
            if "target_digest" not in item:
                item["target_digest"] = (
                    f"sha256:{hashlib.sha256(self.snapshot[name].tobytes()).hexdigest()}"
                )
            local[name] = dict(item)
        return local

    def _publish_delta_index(
        self,
        version,
        base_version,
        raw_deltas,
        metadata,
        owners,
        target_digest,
    ) -> tuple[int, float, float, float]:
        setup_started = time.monotonic()
        local_metadata = {
            name: item for name, item in metadata.items() if owners[name] == self.rank
        }
        groups = _bucket_groups(
            sorted(name for name in raw_deltas if owners[name] == self.rank),
            local_metadata,
            self.bucket_bytes,
        )
        counts = sorted(self._gather((self.rank, len(groups))))
        offset = sum(count for rank, count in counts if rank < self.rank)
        total = sum(count for _rank, count in counts)

        setup_seconds = time.monotonic() - setup_started

        tasks = [
            (ordinal, [(name, raw_deltas[name]) for name in names])
            for ordinal, names in enumerate(groups)
        ]
        raw_deltas.clear()

        def upload(item):
            local_ordinal, tensors = item
            ordinal = offset + local_ordinal
            tensor_names = tuple(name for name, _delta in tensors)
            data, decoded_size = encode_bucket(
                model_id=self.config.model_id,
                base_version=base_version,
                target_version=version,
                base_digest=self.target_digest,
                format_digest=self.format_digest,
                ordinal=ordinal,
                tensors=tensors,
                metadata=local_metadata,
            )
            key = _key(
                self.config.model_id,
                version,
                f"bucket-{ordinal:08d}-of-{total:08d}.mxcd",
            )
            key = "/".join(
                part for part in (self.config.s3.prefix.strip("/"), key) if part
            )
            stored = self.s3.put(
                bucket=self.config.s3.bucket,
                key=key,
                data=data,
            )
            return (
                {
                    "decoded_size": decoded_size,
                    "object": _object(stored, len(data)),
                    "ordinal": ordinal,
                    "tensors": list(tensor_names),
                },
                tensor_names,
            )

        pool_started = time.monotonic()
        descriptors = []
        dirty_ordinals = {}
        if groups:
            with ThreadPoolExecutor(
                max_workers=min(max(1, envs.MX_REFIT_S3_UPLOAD_WORKERS), len(groups))
            ) as pool:
                uploaded = pool.map(upload, tasks)
                for descriptor, names in uploaded:
                    descriptors.append(descriptor)
                    for name in names:
                        dirty_ordinals[name] = descriptor["ordinal"]
        pool_seconds = time.monotonic() - pool_started
        finalize_started = time.monotonic()

        coverage = []
        for name, item in local_metadata.items():
            value = {**item, "state": "clean"}
            if name in dirty_ordinals:
                value["state"] = "dirty"
                value["bucket_ordinal"] = dirty_ordinals[name]
            coverage.append(value)
        wire_bytes = sum(item["object"]["size"] for item in descriptors)
        contributions = self._gather((self.rank, descriptors, coverage))

        if self.rank == 0:
            buckets = sorted(
                (item for _rank, values, _coverage in contributions for item in values),
                key=lambda item: item["ordinal"],
            )
            tensors = sorted(
                (item for _rank, _values, values in contributions for item in values),
                key=lambda item: item["name"],
            )
            index_payload = canonical_json(
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
            key = _key(self.config.model_id, version, "delta-index.json")
            key = "/".join(
                part for part in (self.config.s3.prefix.strip("/"), key) if part
            )
            payload = self.s3.put(
                bucket=self.config.s3.bucket,
                key=key,
                data=index_payload,
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
        finalize_seconds = time.monotonic() - finalize_started
        return wire_bytes, setup_seconds, pool_seconds, finalize_seconds

    def _gather(self, value):
        if not self.distributed:
            return [value]
        values = [None] * self.world
        torch.distributed.all_gather_object(values, value, group=self.group)
        return values
