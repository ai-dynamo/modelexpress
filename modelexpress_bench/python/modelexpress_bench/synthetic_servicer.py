# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synthetic-source ModelService for benchmarking.

Standalone implementation that emits the same `FileChunk` shape as the
production Rust and Python servers, sourced from a single pre-filled
in-memory buffer. Designed for benchmarking the Python gRPC plane in
isolation from disk.

The production peer is intentionally NOT a dependency: the bench Docker
image only needs proto stubs + grpcio, not the full modelexpress package
(which transitively pulls torch / vLLM / NIXL). The production path is
tested independently; this servicer measures network-shape throughput.
"""

from __future__ import annotations

import logging
from typing import Iterator

import grpc

from modelexpress import model_pb2, model_pb2_grpc

from .model_name import BENCH_PREFIX, BenchSpec

logger = logging.getLogger("modelexpress_bench.synthetic_servicer")


# Mirrors modelexpress_common::constants::DEFAULT_TRANSFER_CHUNK_SIZE.
DEFAULT_TRANSFER_CHUNK_SIZE = 256 * 1024


def _make_source_buffer(size: int, seed: int = 0xDEADBEEFCAFEBABE) -> bytes:
    """Build a deterministic source buffer of the requested size.

    Uses a tiny xorshift64 pattern so the bytes aren't all-zero (would let
    a transport-layer compressor inflate the apparent rate).
    """
    out = bytearray(size)
    state = seed | 1
    pos = 0
    while pos < size:
        state ^= (state << 13) & 0xFFFFFFFFFFFFFFFF
        state ^= (state >> 7) & 0xFFFFFFFFFFFFFFFF
        state ^= (state << 17) & 0xFFFFFFFFFFFFFFFF
        block = state.to_bytes(8, "little")
        end = min(pos + 8, size)
        out[pos:end] = block[: end - pos]
        pos = end
    return bytes(out)


class SyntheticBenchServicer(model_pb2_grpc.ModelServiceServicer):
    """ModelService that handles bench: model_names from a memory buffer."""

    def __init__(self, source_buf_size: int = 16 * 1024 * 1024) -> None:
        self._source = _make_source_buffer(max(source_buf_size, 1))
        logger.info(
            "SyntheticBenchServicer ready: source_buf=%d bytes", len(self._source)
        )

    # ------------------------------------------------------------------
    # ModelService RPCs
    # ------------------------------------------------------------------

    def EnsureModelDownloaded(self, request, context):
        if not request.model_name.startswith(BENCH_PREFIX):
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "synthetic bench server only handles bench:* model_names",
            )
            return
        yield model_pb2.ModelStatusUpdate(
            model_name=request.model_name,
            status=model_pb2.DOWNLOADED,
            message="bench: synthetic source",
            provider=request.provider,
        )

    def ListModelFiles(self, request, context):
        try:
            spec = BenchSpec.parse(request.model_name)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return model_pb2.ModelFileList()
        files = [
            model_pb2.ModelFileInfo(
                relative_path=f"bench-{i:04d}.bin",
                size=spec.bytes_per_file,
            )
            for i in range(spec.file_count)
        ]
        return model_pb2.ModelFileList(
            model_name=request.model_name,
            files=files,
            total_size=spec.total_bytes,
        )

    def StreamModelFiles(
        self, request, context
    ) -> Iterator[model_pb2.FileChunk]:
        try:
            spec = BenchSpec.parse(request.model_name)
        except ValueError as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
            return

        chunk_size = (
            request.chunk_size
            if request.chunk_size > 0
            else DEFAULT_TRANSFER_CHUNK_SIZE
        )
        if chunk_size > len(self._source):
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"chunk_size {chunk_size} > source buffer {len(self._source)}",
            )
            return

        total_files = spec.file_count
        total_size = spec.bytes_per_file
        for file_idx in range(total_files):
            relative_path = f"bench-{file_idx:04d}.bin"
            is_last_file = (file_idx + 1) == total_files
            offset = 0
            if total_size == 0:
                yield model_pb2.FileChunk(
                    relative_path=relative_path,
                    data=b"",
                    offset=0,
                    total_size=0,
                    is_last_chunk=True,
                    is_last_file=is_last_file,
                    commit_hash=None,
                )
                continue
            while offset < total_size:
                remaining = total_size - offset
                take = min(remaining, chunk_size)
                # bytes(self._source[:take]) does the per-chunk copy that
                # mirrors the Rust server's `buffer[..take].to_vec()` shape.
                data = bytes(self._source[:take])
                is_last_chunk = take == remaining
                yield model_pb2.FileChunk(
                    relative_path=relative_path,
                    data=data,
                    offset=offset,
                    total_size=total_size,
                    is_last_chunk=is_last_chunk,
                    is_last_file=is_last_file and is_last_chunk,
                    commit_hash=None,
                )
                offset += take
