# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
S3-over-RDMA safetensors loader built on NIXL OBJ.

Metadata (listing, index, safetensors headers, integrity canaries) is read
over plain HTTP with boto3. Tensor payloads are read with ranged S3-over-RDMA
GETs into a small pool of registered staging slots, then copied into fresh
per-tensor device allocations that are yielded to the engine. Slots live in
device memory, or in pinned host memory when the GPU cannot be registered for
RDMA (``MX_S3_RDMA_STAGING``). Yielded tensors never alias the slots, so
engines may keep references across iterations while the slots are reused.

Usage::

    loader = MxS3RdmaLoader(accelerator_backend, device_id, S3RdmaLoadConfig.from_env())
    loader.prepare("s3://models/llama-3.1-8b")
    try:
        for name, tensor in loader.iter_tensors():
            ...
    finally:
        loader.release()
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence
from urllib.parse import urlparse

import torch

from . import envs
from .accelerators import AcceleratorBackend
from .s3_rdma_transfer import (
    S3RdmaEndpoint,
    S3RdmaTransferError,
    S3RdmaTransferManager,
)
from .safetensors_header import (
    HEADER_LENGTH_BYTES,
    INDEX_FILE,
    SafetensorsHeader,
    header_length,
    index_weight_map,
    parse_header,
    select_shards,
    torch_dtype,
)

logger = logging.getLogger("modelexpress.s3_rdma_loader")

VERIFY_MODES = ("tail", "full", "off")
STAGING_MODES = ("auto", "device", "host")
_MIB = 1024 * 1024
# One ranged GET covers the header of typical shards (tens to hundreds of KiB);
# a larger header costs one more GET.
_HEADER_PROBE_BYTES = _MIB
# Small enough that each canary GET is latency-bound, and the poisoned tail
# is the bitwise inverse, so a window the server skipped or truncated cannot match.
_CANARY_BYTES = 64
# Parallel HTTP header and canary GETs; also the boto3 connection pool size,
# so these requests never wait for a connection.
_METADATA_WORKERS = 32
# Sanity cap on staging slots: GPU memory held is window size x inflight.
_MAX_INFLIGHT = 64
# Default maxConnections of Aws::Client::ClientConfiguration. NIXL's OBJ
# control plane neither sets it nor accepts a parameter for it, so it cannot
# be read back; revisit this if NIXL starts configuring its own pool.
_AWS_SDK_DEFAULT_CONNECTIONS = 25


class S3RdmaIntegrityError(S3RdmaTransferError):
    """Bytes delivered over RDMA do not match the object."""


class S3RdmaTransferTimeout(S3RdmaTransferError):
    """A ranged read did not complete within the configured timeout."""


@dataclass(frozen=True)
class S3RdmaLoadConfig:
    """Tuning for one load; see the MX_S3_RDMA_* environment variables."""

    window_bytes: int = 256 * _MIB
    inflight: int = 8
    threads: int = 0
    timeout_s: float = 120.0
    verify: str = "tail"
    staging: str = "auto"

    def __post_init__(self) -> None:
        if self.window_bytes <= 0:
            raise ValueError(f"window size {self.window_bytes} must be positive")
        if not 1 <= self.inflight <= _MAX_INFLIGHT:
            raise ValueError(f"inflight {self.inflight} must be in [1, {_MAX_INFLIGHT}]")
        if self.threads < 0:
            raise ValueError(f"threads {self.threads} must be >= 0")
        if self.verify not in VERIFY_MODES:
            raise ValueError(f"verify mode {self.verify!r} must be one of {VERIFY_MODES}")
        if self.staging not in STAGING_MODES:
            raise ValueError(f"staging mode {self.staging!r} must be one of {STAGING_MODES}")

    @property
    def num_threads(self) -> int:
        return self.threads or self.inflight

    @classmethod
    def from_env(cls) -> S3RdmaLoadConfig:
        config = cls(
            window_bytes=envs.MX_S3_RDMA_WINDOW_MB * _MIB,
            inflight=envs.MX_S3_RDMA_INFLIGHT,
            threads=envs.MX_S3_RDMA_THREADS,
            timeout_s=envs.MX_S3_RDMA_TIMEOUT,
            verify=envs.MX_S3_RDMA_VERIFY,
            staging=envs.MX_S3_RDMA_STAGING,
        )
        if config.num_threads > _AWS_SDK_DEFAULT_CONNECTIONS - 1:
            logger.warning(
                "MX_S3_RDMA inflight/threads=%d exceeds the NIXL OBJ control plane's "
                "default HTTP connection pool (%d); extra reads will queue",
                config.num_threads,
                _AWS_SDK_DEFAULT_CONNECTIONS,
            )
        return config


def parse_s3_prefix(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket[/prefix]`` into the bucket and a ``/``-terminated prefix."""
    parsed = urlparse(uri)
    if parsed.scheme.lower() != "s3" or not parsed.netloc:
        raise ValueError(f"expected an s3://bucket[/prefix] URI, got {uri!r}")
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return parsed.netloc, prefix


class S3ObjectReader:
    """Small boto3 wrapper for the HTTP metadata path."""

    def __init__(self, client: Any):
        self._client = client

    @classmethod
    def from_session(cls, bucket: str) -> tuple[S3ObjectReader, S3RdmaEndpoint]:
        """Build a reader and the matching NIXL endpoint from the AWS config chain.

        Endpoint, region, credentials and CA bundle come from the standard AWS
        environment (``AWS_ENDPOINT_URL``, ``AWS_ACCESS_KEY_ID``, ...), so the
        HTTP metadata path, NIXL, and the ModelStreamer fallback agree. NIXL
        resolves credentials itself; they are only checked here so a missing
        chain fails before any GPU or RDMA setup.
        """
        import boto3
        import botocore.session
        from botocore.config import Config

        core = botocore.session.get_session()
        session = boto3.session.Session(botocore_session=core)
        client = session.client(
            "s3",
            config=Config(
                s3={"addressing_style": "path"},
                retries={"mode": "standard", "max_attempts": 5},
                max_pool_connections=_METADATA_WORKERS,
            ),
        )
        endpoint_url = client.meta.endpoint_url
        host = urlparse(endpoint_url).hostname or ""
        if host.endswith("amazonaws.com"):
            raise S3RdmaTransferError(
                f"endpoint {endpoint_url} is AWS S3, which does not serve S3-over-RDMA; "
                "set AWS_ENDPOINT_URL to an RDMA-enabled endpoint"
            )
        credentials = session.get_credentials()
        if credentials is None:
            raise S3RdmaTransferError("no S3 credentials found in the AWS config chain")
        frozen = credentials.get_frozen_credentials()
        if not frozen.access_key or not frozen.secret_key:
            raise S3RdmaTransferError("S3 credentials must include an access key and a secret key")
        endpoint = S3RdmaEndpoint(
            bucket=bucket,
            endpoint_url=endpoint_url,
            region=client.meta.region_name or "us-east-1",
            ca_bundle=core.get_config_variable("ca_bundle"),
        )
        return cls(client), endpoint

    def list_objects(self, bucket: str, prefix: str) -> dict[str, tuple[int, str]]:
        """Return ``{key: (size, etag)}`` for objects directly under ``prefix``."""
        paginator = self._client.get_paginator("list_objects_v2")
        objects: dict[str, tuple[int, str]] = {}
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
            for item in page.get("Contents", []):
                objects[item["Key"]] = (int(item["Size"]), item.get("ETag", "").strip('"'))
        return objects

    def get_object(self, bucket: str, key: str) -> bytes:
        return self._client.get_object(Bucket=bucket, Key=key)["Body"].read()

    def get_range(self, bucket: str, key: str, start: int, end: int) -> bytes:
        """Return bytes ``[start, end)`` of an object."""
        if end <= start:
            return b""
        response = self._client.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end - 1}")
        return response["Body"].read()


@dataclass(frozen=True)
class ShardInfo:
    key: str
    size: int
    etag: str
    header: SafetensorsHeader


@dataclass(frozen=True)
class TensorPlan:
    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    nbytes: int
    obj_idx: int
    # Absolute byte offset of the tensor data within its object.
    start: int


@dataclass(frozen=True)
class Piece:
    """Bytes ``[win_off, win_off + nbytes)`` of a window belong at ``dst_off`` of a tensor."""

    tensor_idx: int
    win_off: int
    dst_off: int
    nbytes: int


@dataclass(frozen=True)
class WindowPlan:
    obj_idx: int
    offset: int
    length: int
    pieces: tuple[Piece, ...]


@dataclass(frozen=True)
class S3RdmaLoadPlan:
    bucket: str
    prefix: str
    objects: tuple[tuple[str, int, str], ...]
    tensors: tuple[TensorPlan, ...]
    windows: tuple[WindowPlan, ...]
    fingerprint: str

    @property
    def total_bytes(self) -> int:
        return sum(window.length for window in self.windows)


def _fetch_header(reader: S3ObjectReader, bucket: str, key: str, size: int) -> SafetensorsHeader:
    source = f"s3://{bucket}/{key}"
    prefix = reader.get_range(bucket, key, 0, min(size, _HEADER_PROBE_BYTES))
    total = HEADER_LENGTH_BYTES + header_length(prefix, source)
    if total > size:
        raise ValueError(f"Safetensors header of {source} runs past the end of the object")
    if len(prefix) < total:
        prefix += reader.get_range(bucket, key, len(prefix), total)
    header = parse_header(prefix[HEADER_LENGTH_BYTES:total], source)
    if header.data_end > size:
        raise ValueError(
            f"Tensor data of {source} ends at byte {header.data_end}, past the object size {size}"
        )
    return header


def resolve_checkpoint(
    reader: S3ObjectReader,
    bucket: str,
    prefix: str,
    *,
    listing: dict[str, tuple[int, str]] | None = None,
) -> list[ShardInfo]:
    """Pick the safetensors objects of a checkpoint (see ``select_shards``) and parse their headers."""
    uri = f"s3://{bucket}/{prefix}"
    if listing is None:
        listing = reader.list_objects(bucket, prefix)

    index_key = prefix + INDEX_FILE
    weight_map = None
    if index_key in listing:
        weight_map = index_weight_map(json.loads(reader.get_object(bucket, index_key)), uri + INDEX_FILE)
    keys = [prefix + name for name in select_shards((key[len(prefix):] for key in listing), weight_map)]
    missing = [key for key in keys if key not in listing]
    if missing:
        raise FileNotFoundError(f"{INDEX_FILE} in {uri} references missing objects: {missing}")
    if not keys:
        raise FileNotFoundError(f"No .safetensors objects found under {uri}")

    with ThreadPoolExecutor(max_workers=min(_METADATA_WORKERS, len(keys))) as pool:
        headers = list(pool.map(lambda key: _fetch_header(reader, bucket, key, listing[key][0]), keys))

    shards = []
    seen: dict[str, str] = {}
    for key, header in zip(keys, headers, strict=True):
        for entry in header.tensors:
            if entry.name in seen:
                raise ValueError(
                    f"Tensor '{entry.name}' appears in both {seen[entry.name]} and {key}"
                )
            seen[entry.name] = key
        size, etag = listing[key]
        shards.append(ShardInfo(key=key, size=size, etag=etag, header=header))
    return shards


def plan_windows(
    object_sizes: Sequence[int],
    tensors: Sequence[TensorPlan],
    window_bytes: int,
    tensor_filter: Callable[[TensorPlan], bool] | None = None,
) -> tuple[WindowPlan, ...]:
    """Cut the bytes of the wanted tensors into object-local windows.

    Adjacent tensors share windows; a gap (a filtered-out tensor, or padding)
    starts a new extent. Every wanted tensor with data is covered exactly once.
    """
    by_object: dict[int, list[int]] = {}
    for idx, tensor in enumerate(tensors):
        if tensor.nbytes == 0 or (tensor_filter is not None and not tensor_filter(tensor)):
            continue
        by_object.setdefault(tensor.obj_idx, []).append(idx)

    windows: list[WindowPlan] = []
    for obj_idx in sorted(by_object):
        members = sorted(by_object[obj_idx], key=lambda i: tensors[i].start)
        extents: list[list[int]] = []
        for idx in members:
            if extents:
                last = tensors[extents[-1][-1]]
                if tensors[idx].start == last.start + last.nbytes:
                    extents[-1].append(idx)
                    continue
            extents.append([idx])

        for extent in extents:
            ext_start = tensors[extent[0]].start
            last = tensors[extent[-1]]
            ext_end = last.start + last.nbytes
            cursor = 0
            for win_start in range(ext_start, ext_end, window_bytes):
                win_end = min(win_start + window_bytes, ext_end)
                pieces = []
                while cursor < len(extent):
                    tensor = tensors[extent[cursor]]
                    t_end = tensor.start + tensor.nbytes
                    lo = max(tensor.start, win_start)
                    hi = min(t_end, win_end)
                    if lo < hi:
                        pieces.append(
                            Piece(
                                tensor_idx=extent[cursor],
                                win_off=lo - win_start,
                                dst_off=lo - tensor.start,
                                nbytes=hi - lo,
                            )
                        )
                    if t_end <= win_end:
                        cursor += 1
                    else:
                        break
                windows.append(
                    WindowPlan(
                        obj_idx=obj_idx,
                        offset=win_start,
                        length=win_end - win_start,
                        pieces=tuple(pieces),
                    )
                )

    covered: dict[int, int] = {}
    for window in windows:
        if window.length > window_bytes or window.offset + window.length > object_sizes[window.obj_idx]:
            raise AssertionError(f"window {window.offset}+{window.length} outside its object")
        for piece in window.pieces:
            covered[piece.tensor_idx] = covered.get(piece.tensor_idx, 0) + piece.nbytes
    for idx_list in by_object.values():
        for idx in idx_list:
            if covered.get(idx) != tensors[idx].nbytes:
                raise AssertionError(f"tensor '{tensors[idx].name}' is not covered exactly once")
    return tuple(windows)


def build_plan(
    bucket: str,
    prefix: str,
    shards: Sequence[ShardInfo],
    window_bytes: int,
) -> S3RdmaLoadPlan:
    """Turn resolved shards into tensors and windows."""
    objects = tuple((shard.key, shard.size, shard.etag) for shard in shards)
    tensors: list[TensorPlan] = []
    for obj_idx, shard in enumerate(shards):
        for entry in shard.header.tensors:
            tensors.append(
                TensorPlan(
                    name=entry.name,
                    dtype=torch_dtype(entry.dtype, entry.name),
                    shape=entry.shape,
                    nbytes=entry.nbytes,
                    obj_idx=obj_idx,
                    start=shard.header.data_start + entry.begin,
                )
            )
    windows = plan_windows([shard.size for shard in shards], tensors, window_bytes)
    digest = hashlib.sha256()
    for key, size, etag in sorted(objects):
        digest.update(f"{key}\0{size}\0{etag}\n".encode())
    return S3RdmaLoadPlan(
        bucket=bucket,
        prefix=prefix,
        objects=objects,
        tensors=tuple(tensors),
        windows=windows,
        fingerprint=digest.hexdigest(),
    )


class MxS3RdmaLoader:
    """Load safetensors weights from an S3-over-RDMA endpoint into device memory."""

    def __init__(
        self,
        accelerator_backend: AcceleratorBackend,
        device_id: int,
        config: S3RdmaLoadConfig,
        *,
        reader: S3ObjectReader | None = None,
        endpoint: S3RdmaEndpoint | None = None,
    ):
        self._backend = accelerator_backend
        self._device_id = device_id
        self._config = config
        self._reader = reader
        self._endpoint = endpoint
        self._device: torch.device | None = None
        self._plan: S3RdmaLoadPlan | None = None
        self._manager: S3RdmaTransferManager | None = None
        self._slots: list[torch.Tensor] = []
        self._tail_host: list[torch.Tensor] = []
        self._canaries: list[bytes] = []
        self._poison: torch.Tensor | None = None
        self._free: deque[int] = deque()
        self._done: deque[int] = deque()
        self._slot_window: dict[int, int] = {}
        self._slot_handle: dict[int, Any] = {}
        self._slot_posted: dict[int, float] = {}
        self._next = 0
        self._dest: dict[int, torch.Tensor] = {}
        self._remaining: dict[int, int] = {}
        self._yielded = 0
        self._finished = False
        self._drain_failed = False
        self._timing: dict[str, float] = {}
        self._staging = "device"

    @property
    def plan(self) -> S3RdmaLoadPlan | None:
        return self._plan

    @property
    def finished(self) -> bool:
        """True once every planned tensor was yielded and the objects were re-checked."""
        return self._finished

    # ------------------------------------------------------------------
    # Prepare
    # ------------------------------------------------------------------

    def prepare(self, model_uri: str) -> S3RdmaLoadPlan:
        """Resolve the checkpoint, set up RDMA, and prove the data path on real reads.

        Everything that can fail before the engine sees a tensor happens here:
        backend creation, metadata, registration, and the first verified window.
        Backend creation comes first because it is the likeliest failure on a
        host without cuObject or an RDMA fabric.
        """
        t0 = time.perf_counter()
        bucket, prefix = parse_s3_prefix(model_uri)
        if self._reader is None or self._endpoint is None:
            self._reader, self._endpoint = S3ObjectReader.from_session(bucket)
        if self._endpoint.bucket != bucket:
            raise ValueError(f"endpoint bucket {self._endpoint.bucket!r} != URI bucket {bucket!r}")

        self._backend.set_device(self._device_id)
        self._device = self._backend.torch_device(self._device_id)
        self._manager = S3RdmaTransferManager(
            agent_name=f"mx-s3-rdma-{self._device_id}-{uuid.uuid4().hex[:8]}",
            endpoint=self._endpoint,
            device_id=self._device_id,
            num_threads=self._config.num_threads,
            accelerator_backend=self._backend,
        )
        self._manager.initialize()
        marks = {"backend": time.perf_counter()}

        shards = resolve_checkpoint(self._reader, bucket, prefix)
        plan = build_plan(bucket, prefix, shards, self._config.window_bytes)
        marks["metadata"] = time.perf_counter()
        self._plan = plan
        self._remaining = {
            idx: tensor.nbytes for idx, tensor in enumerate(plan.tensors) if tensor.nbytes
        }

        if plan.windows:
            self._fetch_canaries(plan)
            marks["canaries"] = time.perf_counter()
            self._manager.register_objects([(key, size) for key, size, _etag in plan.objects])
            self._register_slots(plan)
            marks["register"] = time.perf_counter()
            self._prime()
            marks["prime"] = time.perf_counter()
        self._timing["prepare_s"] = time.perf_counter() - t0
        previous = t0
        phases = []
        for phase, mark in marks.items():
            phases.append(f"{phase}={mark - previous:.2f}s")
            previous = mark
        logger.info(
            "[TIMING] NIXL OBJ prepare: %d objects, %d tensors, %d windows, %.2f GB, "
            "%s staging, in %.2fs (%s)",
            len(plan.objects),
            len(plan.tensors),
            len(plan.windows),
            plan.total_bytes / 1e9,
            self._staging,
            self._timing["prepare_s"],
            ", ".join(phases),
        )
        return plan

    def _fetch_canaries(self, plan: S3RdmaLoadPlan) -> None:
        if self._config.verify != "tail":
            return
        reader = self._reader

        def fetch(window: WindowPlan) -> bytes:
            key = plan.objects[window.obj_idx][0]
            end = window.offset + window.length
            return reader.get_range(plan.bucket, key, end - min(_CANARY_BYTES, window.length), end)

        with ThreadPoolExecutor(max_workers=min(_METADATA_WORKERS, len(plan.windows))) as pool:
            self._canaries = list(pool.map(fetch, plan.windows))
        poison = bytearray(_CANARY_BYTES * len(plan.windows))
        for idx, (window, canary) in enumerate(zip(plan.windows, self._canaries, strict=True)):
            if len(canary) != min(_CANARY_BYTES, window.length):
                raise S3RdmaIntegrityError("canary fetch returned a short read")
            base = idx * _CANARY_BYTES
            poison[base : base + len(canary)] = bytes(b ^ 0xFF for b in canary)
        table = torch.frombuffer(poison, dtype=torch.uint8).view(len(plan.windows), _CANARY_BYTES)
        self._poison = table.pin_memory() if self._device.type == "cuda" else table

    def _register_slots(self, plan: S3RdmaLoadPlan) -> None:
        """Register device staging slots, or pinned host slots if device registration fails."""
        staging = self._config.staging
        if staging != "host":
            self._allocate_slots(plan, host=False)
            try:
                self._manager.register_slots(self._slots)
                self._staging = "device"
                return
            except S3RdmaTransferError as exc:
                if staging == "device":
                    raise
                logger.warning(
                    "NIXL OBJ could not register device staging slots (%s); "
                    "staging windows in pinned host memory instead",
                    exc,
                )
                self._slots = []
                self._tail_host = []
                self._backend.empty_cache()
        self._allocate_slots(plan, host=True)
        self._manager.register_slots(self._slots, host=True)
        self._staging = "host"

    def _allocate_slots(self, plan: S3RdmaLoadPlan, *, host: bool) -> None:
        slot_bytes = max(window.length for window in plan.windows)
        count = min(self._config.inflight, len(plan.windows))
        pin = self._device.type == "cuda"
        for _ in range(count):
            if host:
                slot = torch.empty(slot_bytes, dtype=torch.uint8, pin_memory=pin)
            else:
                slot = torch.empty(slot_bytes, dtype=torch.uint8, device=self._device)
            self._slots.append(slot)
            tail = torch.empty(_CANARY_BYTES, dtype=torch.uint8, pin_memory=pin)
            self._tail_host.append(tail)
        self._free = deque(range(count))

    def _prime(self) -> None:
        """Post the first wave of reads and verify the first one to complete."""
        assert self._plan is not None
        for window_idx, slot in enumerate(self._free):
            self._poison_tail(slot, window_idx)
        self._fence()
        while self._free and self._next < len(self._plan.windows):
            self._post(self._free.popleft())
        slot = self._wait_one()
        window = self._plan.windows[self._slot_window[slot]]
        self._capture_tail(slot, window)
        self._fence()
        self._verify(slot, self._slot_window[slot])
        self._done.append(slot)

    # ------------------------------------------------------------------
    # Stream
    # ------------------------------------------------------------------

    def iter_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield ``(name, tensor)`` as each tensor's bytes have all arrived.

        Order follows read completion, not file order. Closing the iterator
        early drains every outstanding read before returning.
        """
        plan = self._plan
        if plan is None:
            raise RuntimeError("prepare() must succeed before iter_tensors()")
        t0 = time.perf_counter()
        try:
            for tensor in plan.tensors:
                if tensor.nbytes == 0:
                    self._yielded += 1
                    yield tensor.name, torch.empty(tensor.shape, dtype=tensor.dtype, device=self._device)

            while self._done or self._slot_handle:
                slot = self._done.popleft() if self._done else self._wait_one()
                window_idx = self._slot_window.pop(slot)
                window = plan.windows[window_idx]
                ready = self._scatter(slot, window)
                self._capture_tail(slot, window)
                refill = self._next < len(plan.windows)
                if refill:
                    if self._staging == "host":
                        # A poison write to a host slot lands at once, so queued
                        # copies out of the slot must finish first.
                        self._fence()
                    self._poison_tail(slot, self._next)
                self._fence()
                self._verify(slot, window_idx)
                if refill:
                    self._post(slot)
                else:
                    self._free.append(slot)
                for idx in ready:
                    tensor = plan.tensors[idx]
                    buffer = self._dest.pop(idx)
                    self._yielded += 1
                    yield tensor.name, buffer.view(tensor.dtype).reshape(tensor.shape)

            if self._yielded != len(plan.tensors):
                raise S3RdmaIntegrityError(
                    f"yielded {self._yielded} of {len(plan.tensors)} planned tensors"
                )
            self._check_unchanged(plan)
            self._finished = True
            self._timing["stream_s"] = time.perf_counter() - t0
        finally:
            if not self._finished:
                self._abort()

    def _scatter(self, slot: int, window: WindowPlan) -> list[int]:
        """Copy a landed window into its tensors; return tensors that are now complete."""
        src = self._slots[slot]
        ready = []
        for piece in window.pieces:
            dst = self._dest.get(piece.tensor_idx)
            if dst is None:
                nbytes = self._plan.tensors[piece.tensor_idx].nbytes
                dst = torch.empty(nbytes, dtype=torch.uint8, device=self._device)
                self._dest[piece.tensor_idx] = dst
            dst[piece.dst_off : piece.dst_off + piece.nbytes].copy_(
                src[piece.win_off : piece.win_off + piece.nbytes], non_blocking=True
            )
            self._remaining[piece.tensor_idx] -= piece.nbytes
            if self._remaining[piece.tensor_idx] == 0:
                ready.append(piece.tensor_idx)
        return ready

    def _post(self, slot: int) -> None:
        window_idx = self._next
        window = self._plan.windows[window_idx]
        handle = self._manager.post(slot, window.obj_idx, window.offset, window.length)
        self._next += 1
        self._slot_window[slot] = window_idx
        self._slot_handle[slot] = handle
        self._slot_posted[slot] = time.monotonic()

    def _wait_one(self) -> int:
        """Poll in-flight reads round-robin until one completes; return its slot."""
        spins = 0
        while True:
            now = time.monotonic()
            for slot, handle in list(self._slot_handle.items()):
                try:
                    done = self._manager.poll(handle)
                except S3RdmaTransferError:
                    self._manager.release(handle)
                    del self._slot_handle[slot]
                    del self._slot_posted[slot]
                    raise
                if done:
                    self._manager.release(handle)
                    del self._slot_handle[slot]
                    del self._slot_posted[slot]
                    return slot
                if now - self._slot_posted[slot] > self._config.timeout_s:
                    window = self._plan.windows[self._slot_window[slot]]
                    raise S3RdmaTransferTimeout(
                        f"read of {self._plan.objects[window.obj_idx][0]} "
                        f"[{window.offset}, {window.offset + window.length}) did not complete "
                        f"within {self._config.timeout_s}s"
                    )
            spins += 1
            if spins > 100:
                time.sleep(0.0001)
                spins = 0

    def _poison_tail(self, slot: int, window_idx: int) -> None:
        if self._config.verify != "tail":
            return
        end = self._plan.windows[window_idx].length
        count = min(_CANARY_BYTES, end)
        self._slots[slot][end - count : end].copy_(self._poison[window_idx, :count], non_blocking=True)

    def _capture_tail(self, slot: int, window: WindowPlan) -> None:
        if self._config.verify != "tail":
            return
        count = min(_CANARY_BYTES, window.length)
        self._tail_host[slot][:count].copy_(
            self._slots[slot][window.length - count : window.length], non_blocking=True
        )

    def _verify(self, slot: int, window_idx: int) -> None:
        """Check a landed window after the fence; raise S3RdmaIntegrityError on mismatch."""
        mode = self._config.verify
        if mode == "off":
            return
        window = self._plan.windows[window_idx]
        key = self._plan.objects[window.obj_idx][0]
        if mode == "tail":
            expected = self._canaries[window_idx]
            actual = bytes(self._tail_host[slot][: len(expected)].tolist())
        else:
            end = window.offset + window.length
            expected = self._reader.get_range(self._plan.bucket, key, window.offset, end)
            actual = bytes(self._slots[slot][: window.length].cpu().numpy().tobytes())
        if actual != expected:
            raise S3RdmaIntegrityError(
                f"RDMA read of {key} [{window.offset}, {window.offset + window.length}) "
                f"did not deliver the expected bytes (verify={mode}); "
                "the server may have declined or truncated the transfer"
            )

    def _fence(self) -> None:
        """Wait for queued device copies; RDMA writes are not ordered on any stream."""
        self._backend.synchronize(self._device_id)

    def _check_unchanged(self, plan: S3RdmaLoadPlan) -> None:
        listing = self._reader.list_objects(plan.bucket, plan.prefix)
        for key, size, etag in plan.objects:
            if listing.get(key) != (size, etag):
                raise S3RdmaIntegrityError(f"s3://{plan.bucket}/{key} changed while it was being loaded")

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _abort(self) -> None:
        if self._manager is None:
            return
        if not self._manager.drain(self._config.timeout_s):
            self._drain_failed = True
            logger.error("NIXL OBJ reads did not drain within %.0fs", self._config.timeout_s)
        self._slot_handle.clear()
        self._slot_posted.clear()

    def release(self) -> None:
        """Drain outstanding reads and free the agent, registrations, and staging memory."""
        if self._manager is not None:
            drain_timeout_s = 0.0 if self._drain_failed else self._config.timeout_s
            self._manager.shutdown(drain_timeout_s=drain_timeout_s)
            self._manager = None
        had_slots = bool(self._slots)
        self._slots = []
        self._tail_host = []
        self._poison = None
        self._dest.clear()
        self._slot_handle.clear()
        self._slot_posted.clear()
        self._done.clear()
        self._free.clear()
        if had_slots:
            self._backend.empty_cache()

    def shutdown(self) -> None:
        self.release()
        self._reader = None

    def log_timing(self) -> None:
        plan = self._plan
        if plan is None:
            return
        prepare_s = self._timing.get("prepare_s", 0.0)
        stream_s = self._timing.get("stream_s", 0.0)
        total_s = prepare_s + stream_s
        rate = plan.total_bytes / total_s / 1e9 if total_s > 0 else 0.0
        logger.info(
            "[TIMING] NIXL OBJ load: %.2f GB, %d tensors, prepare %.2fs, stream %.2fs, "
            "total %.2fs (%.2f GB/s)",
            plan.total_bytes / 1e9,
            self._yielded,
            prepare_s,
            stream_s,
            total_s,
            rate,
        )
