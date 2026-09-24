# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-read full refit with bounded asynchronous checkpoint persistence."""

from __future__ import annotations

import json
import logging
import struct
import threading
import time
from collections.abc import Iterator
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Queue

import torch
from safetensors.torch import _getdtype

from .checkpoint_store import LocalCheckpointStore
from .receiver import _S3Version, _protected_versions, _source_identity
from ..utils import index_checkpoint_tensors

logger = logging.getLogger(__name__)
_CACHE_PENDING_BYTES = 2 * 1024**3


@dataclass
class StreamedCheckpoint:
    """Full HF source plus an optional node-local writer fed by its iterator."""

    version: _S3Version
    index_data: bytes
    weight_map: dict[str, str]
    tensor_metadata: dict[str, dict]
    store: LocalCheckpointStore
    _writer: _CheckpointWriter | None = field(default=None, init=False)
    _complete: bool = field(default=False, init=False)

    @property
    def shard_uris(self) -> tuple[str, ...]:
        parent = self.version.uri.rsplit("/", 1)[0]
        return tuple(
            f"{parent}/{name}" for name in sorted(set(self.weight_map.values()))
        )

    def validate(self, weights: Iterator, *, cache: bool = False) -> Iterator:
        """Check each tensor before yielding it; snapshot before engine mutation."""
        if cache:
            self._writer = _CheckpointWriter(self)
        seen = set()
        for name, tensor in weights:
            if name not in self.tensor_metadata:
                raise ValueError(f"unexpected full checkpoint tensor {name!r}")
            if name in seen:
                raise ValueError(f"duplicate full checkpoint tensor {name!r}")
            metadata = self.tensor_metadata[name]
            if (
                tensor.dtype != _getdtype(metadata["dtype"])
                or list(tensor.shape) != metadata["shape"]
                or tensor.numel() * tensor.element_size() != metadata["byte_size"]
            ):
                raise ValueError(f"full HF checkpoint metadata differs for {name!r}")
            seen.add(name)
            if self._writer is not None:
                self._writer.submit(name, tensor)
            yield name, tensor
        if seen != set(self.tensor_metadata):
            raise ValueError("full checkpoint stream is missing tensors")
        self._complete = True

    def finish_cache(self, *, success: bool) -> None:
        if self._writer is not None:
            self._writer.finish(success and self._complete)
        if success and not self._complete:
            raise RuntimeError("full checkpoint stream was not completely consumed")

    def wait_cache(self) -> None:
        if self._writer is not None:
            self._writer.wait()

    @property
    def cache_metrics(self) -> dict[str, float]:
        if self._writer is None:
            return {}
        return {
            "perf/mx_stream_cache_snapshot_time": self._writer.snapshot_seconds,
            "perf/mx_stream_cache_backpressure_time": self._writer.backpressure_seconds,
        }

    def close(self) -> None:
        if self._writer is not None:
            self._writer.finish(False)
            try:
                self._writer.wait()
            except Exception:
                logger.warning(
                    "Streamed checkpoint cache did not complete", exc_info=True
                )


class _CheckpointWriter:
    """One writer per node, with owned CPU snapshots and bounded backpressure."""

    def __init__(self, checkpoint: StreamedCheckpoint) -> None:
        self.checkpoint = checkpoint
        self._queue: Queue = Queue()
        self._condition = threading.Condition()
        self._pending_bytes = 0
        self._error: Exception | None = None
        self._finished = False
        self._success = False
        self.snapshot_seconds = 0.0
        self.backpressure_seconds = 0.0
        self._locations: dict[str, tuple[str, int]] = {}
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mx-stream-cache")
        self._future = pool.submit(self._run)
        pool.shutdown(wait=False)

    def submit(self, name: str, tensor: torch.Tensor) -> None:
        size = tensor.numel() * tensor.element_size()
        started = time.perf_counter()
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._error is not None
                    or self._pending_bytes == 0
                    or self._pending_bytes + size <= _CACHE_PENDING_BYTES
                )
            )
            if self._error is not None:
                # Disk failures must not strand other ranks in stream collectives.
                return
            self._pending_bytes += size
        self.backpressure_seconds += time.perf_counter() - started
        started = time.perf_counter()
        try:
            snapshot = tensor.detach().to(
                device="cpu", copy=True, memory_format=torch.contiguous_format
            )
        except Exception:
            with self._condition:
                if self._error is None:
                    self._pending_bytes -= size
                self._condition.notify_all()
            raise
        self.snapshot_seconds += time.perf_counter() - started
        with self._condition:
            if self._error is None:
                self._queue.put((name, snapshot, size))

    def finish(self, success: bool) -> None:
        if not self._finished:
            self._finished = True
            self._success = success
            self._queue.put(None)

    def wait(self) -> None:
        self._future.result()

    def _headers(self) -> dict[str, bytes]:
        headers = {}
        checkpoint = self.checkpoint
        for filename in sorted(set(checkpoint.weight_map.values())):
            offset = 0
            header = {}
            names = sorted(
                name
                for name, shard in checkpoint.weight_map.items()
                if shard == filename
            )
            for name in names:
                metadata = checkpoint.tensor_metadata[name]
                size = metadata["byte_size"]
                header[name] = {
                    "dtype": metadata["dtype"],
                    "shape": metadata["shape"],
                    "data_offsets": [offset, offset + size],
                }
                offset += size
            encoded = json.dumps(header, separators=(",", ":")).encode()
            encoded += b" " * (-len(encoded) % 8)
            prefix = struct.pack("<Q", len(encoded)) + encoded
            headers[filename] = prefix
            for name in names:
                self._locations[name] = (
                    filename,
                    len(prefix) + header[name]["data_offsets"][0],
                )
        return headers

    def _write_tensor(self, temporary, name: str, tensor: torch.Tensor) -> None:
        filename, offset = self._locations[name]
        data = memoryview(tensor.reshape(-1).view(torch.uint8).numpy())
        with (temporary / filename).open("r+b") as output:
            output.seek(offset)
            output.write(data)

    def _run(self) -> None:
        started = time.perf_counter()
        checkpoint = self.checkpoint
        store = checkpoint.store
        version = checkpoint.version.version_id
        target = store.full_path(version)
        logger.info("Streamed checkpoint cache started version=%s", version)
        try:
            with store.installation_locked(), store.locked():
                protected = _protected_versions(store, version)
                state = store.state()
                if state is not None:
                    protected.add(state.version)
                headers = self._headers()
                size = len(checkpoint.index_data) + sum(map(len, headers.values()))
                size += sum(
                    meta["byte_size"] for meta in checkpoint.tensor_metadata.values()
                )
                cached = target.exists()
                if cached:
                    store.verify_artifact_source(
                        target, _source_identity(checkpoint.version)
                    )
                    index_checkpoint_tensors(target)
                else:
                    store.ensure_capacity(size, protected_versions=protected)
                context = (
                    nullcontext(target) if cached else store.replace_directory(target)
                )
                with context as temporary:
                    if not cached:
                        (temporary / "model.safetensors.index.json").write_bytes(
                            checkpoint.index_data
                        )
                        for filename, header in headers.items():
                            (temporary / filename).write_bytes(header)
                    seen = set()
                    while (item := self._queue.get()) is not None:
                        name, tensor, size = item
                        if not cached:
                            self._write_tensor(temporary, name, tensor)
                        seen.add(name)
                        del tensor, item
                        with self._condition:
                            self._pending_bytes -= size
                            self._condition.notify_all()
                    if not self._success or seen != set(checkpoint.weight_map):
                        raise RuntimeError(
                            "streamed checkpoint cache aborted before complete installation"
                        )
                    if not cached:
                        index_checkpoint_tensors(temporary)
                        store.record_artifact(
                            temporary, source=_source_identity(checkpoint.version)
                        )
                store.write_chain(
                    version, {"version": version, "full_version": version, "deltas": []}
                )
                store.enforce_capacity(protected_versions=protected)
            logger.info(
                "Streamed checkpoint cache ready version=%s seconds=%.3f",
                version,
                time.perf_counter() - started,
            )
        except Exception as error:
            with self._condition:
                self._error = error
                self._pending_bytes = 0
                while not self._queue.empty():
                    self._queue.get_nowait()
                self._condition.notify_all()
            logger.error(
                "Streamed checkpoint cache failed version=%s seconds=%.3f",
                version,
                time.perf_counter() - started,
                exc_info=True,
            )
            raise
