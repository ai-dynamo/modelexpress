# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load one HF snapshot and encode XOR+zstd deltas against it."""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import zstandard

_BUCKET_MAGIC = b"MXCDV0\0"
_SCHEMA = "mx.canonical.delta.v0"


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: object) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(value)).hexdigest()}"


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _checkpoint_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    index = root / "model.safetensors.index.json"
    if index.exists():
        names = json.loads(index.read_text())["weight_map"].values()
        return [root / name for name in sorted(set(names))]
    return sorted(root.glob("*.safetensors"))


def load_hf_snapshot(
    checkpoint: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, dict], str, str]:
    """Load the launch HF checkpoint as the byte snapshot used by Miles delta sync."""
    from safetensors import safe_open

    snapshot = {}
    metadata = {}
    for path in _checkpoint_files(Path(checkpoint)):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for source_name in handle.keys():
                tensor = handle.get_tensor(source_name)
                name = source_name.removeprefix("module.")
                data = _tensor_bytes(tensor)
                snapshot[name] = np.frombuffer(data, dtype=np.uint8).copy()
                metadata[name] = {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "byte_size": len(data),
                    "target_digest": f"sha256:{hashlib.sha256(data).hexdigest()}",
                }

    return snapshot, metadata, format_digest(metadata), snapshot_digest(metadata)


def format_digest(metadata: dict[str, dict]) -> str:
    return _digest(
        [
            {
                "name": metadata[name]["name"],
                "shape": metadata[name]["shape"],
                "dtype": metadata[name]["dtype"],
                "byte_size": metadata[name]["byte_size"],
            }
            for name in sorted(metadata)
        ]
    )


def snapshot_digest(metadata: dict[str, dict]) -> str:
    return _digest([metadata[name] for name in sorted(metadata)])


def encode_compressed_bucket(
    *,
    model_id: str,
    base_version: str,
    target_version: str,
    base_digest: str,
    format_digest: str,
    ordinal: int,
    names: list[str],
    compressed_deltas: Mapping[str, bytes | np.ndarray],
    metadata: dict[str, dict],
) -> tuple[bytes, int, tuple[str, ...]]:
    decoded = bytearray()
    entries = []
    decompressor = zstandard.ZstdDecompressor()
    for name in names:
        item = metadata[name]
        delta = decompressor.decompress(
            bytes(compressed_deltas[name]), max_output_size=item["byte_size"]
        )
        if len(delta) != item["byte_size"]:
            raise ValueError(f"{name} delta byte size differs from canonical metadata")
        entries.append({**item, "offset": len(decoded)})
        decoded.extend(delta)

    header = canonical_json(
        {
            "base_digest": base_digest,
            "base_version": base_version,
            "compression": "zstd",
            "decoded_size": len(decoded),
            "delta": "xor",
            "entries": entries,
            "format_digest": format_digest,
            "model_id": model_id,
            "ordinal": ordinal,
            "schema": f"{_SCHEMA}.bucket",
            "target_version": target_version,
        }
    )
    compressed = zstandard.ZstdCompressor(level=3).compress(bytes(decoded))
    return (
        _BUCKET_MAGIC + struct.pack(">I", len(header)) + header + compressed,
        len(decoded),
        tuple(names),
    )


def decode_bucket(
    data: bytes, snapshot: dict[str, np.ndarray], metadata: dict[str, dict]
) -> dict:
    if not data.startswith(_BUCKET_MAGIC):
        raise ValueError("invalid canonical bucket")
    header_size = struct.unpack(
        ">I", data[len(_BUCKET_MAGIC) : len(_BUCKET_MAGIC) + 4]
    )[0]
    header_start = len(_BUCKET_MAGIC) + 4
    header = json.loads(data[header_start : header_start + header_size])
    decoded = zstandard.ZstdDecompressor().decompress(
        data[header_start + header_size :], max_output_size=header["decoded_size"]
    )
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
