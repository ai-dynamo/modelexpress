# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical XOR-delta bucket framing and zstd compression."""

from __future__ import annotations

import hashlib
import json
import struct

import numpy as np
import zstandard

from .source.canonical import canonical_json

_BUCKET_MAGIC = b"MXCDV0\0"
_SCHEMA = "mx.canonical.delta.v0"


def compute_delta(
    current: np.ndarray, base: np.ndarray
) -> tuple[np.ndarray | None, str | None, int]:
    if len(current) != len(base):
        raise RuntimeError("tensor changed byte size")
    raw_delta = np.bitwise_xor(current, base)
    changed_bytes = int(np.count_nonzero(raw_delta))
    if not changed_bytes:
        return None, None, 0
    target_digest = f"sha256:{hashlib.sha256(memoryview(current)).hexdigest()}"
    return raw_delta, target_digest, changed_bytes


def encode_bucket(
    model_id: str,
    base_version: str,
    target_version: str,
    base_digest: str,
    format_digest: str,
    ordinal: int,
    tensors: list[tuple[str, np.ndarray]],
    metadata: dict[str, dict],
) -> tuple[bytes, int]:
    entries = []
    offset = 0
    compressor = zstandard.ZstdCompressor(level=3).compressobj()
    chunks = []
    for name, delta in tensors:
        entries.append({**metadata[name], "offset": offset})
        offset += delta.nbytes
        chunks.append(compressor.compress(memoryview(delta)))
    chunks.append(compressor.flush())

    header = canonical_json(
        {
            "base_digest": base_digest,
            "base_version": base_version,
            "compression": "zstd",
            "decoded_size": offset,
            "delta": "xor",
            "entries": entries,
            "format_digest": format_digest,
            "model_id": model_id,
            "ordinal": ordinal,
            "schema": f"{_SCHEMA}.bucket",
            "target_version": target_version,
        }
    )
    compressed = b"".join(chunks)
    return _BUCKET_MAGIC + struct.pack(">I", len(header)) + header + compressed, offset


def bucket_parts(data: bytes) -> tuple[dict, memoryview]:
    if not data.startswith(_BUCKET_MAGIC):
        raise ValueError("invalid canonical bucket")
    header_size = struct.unpack(
        ">I", data[len(_BUCKET_MAGIC) : len(_BUCKET_MAGIC) + 4]
    )[0]
    header_start = len(_BUCKET_MAGIC) + 4
    header = json.loads(data[header_start : header_start + header_size])
    return header, memoryview(data)[header_start + header_size :]


def parse_bucket(data: bytes) -> tuple[dict, bytes]:
    header, compressed = bucket_parts(data)
    decoded = zstandard.ZstdDecompressor().decompress(
        compressed, max_output_size=header["decoded_size"]
    )
    return header, decoded


def decode_bucket(
    data: bytes, snapshot: dict[str, np.ndarray], metadata: dict[str, dict]
) -> dict:
    header, decoded = parse_bucket(data)
    for entry in header["entries"]:
        name = entry["name"]
        start = entry["offset"]
        delta = np.frombuffer(
            decoded[start : start + entry["byte_size"]], dtype=np.uint8
        )
        target = np.bitwise_xor(snapshot[name], delta)
        digest = f"sha256:{hashlib.sha256(target.tobytes()).hexdigest()}"
        if digest != entry["target_digest"]:
            raise ValueError(f"canonical target checksum differs for {name}")
        snapshot[name] = target
        metadata[name]["target_digest"] = digest
    return header
