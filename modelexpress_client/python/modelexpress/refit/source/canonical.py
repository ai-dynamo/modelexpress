# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load one HF snapshot and encode XOR+zstd deltas against it."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
import zstandard

CanonicalBucket = list[tuple[str, torch.Tensor]] | tuple[tuple[str, torch.Tensor], ...]
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

    ordered = [metadata[name] for name in sorted(metadata)]
    format_digest = _digest(
        [
            {
                "name": item["name"],
                "shape": item["shape"],
                "dtype": item["dtype"],
                "byte_size": item["byte_size"],
            }
            for item in ordered
        ]
    )
    return snapshot, metadata, format_digest, snapshot_digest(metadata)


def snapshot_digest(metadata: dict[str, dict]) -> str:
    return _digest([metadata[name] for name in sorted(metadata)])


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


class CanonicalDeltaEncoder:
    """Diff gathered HF tensors against a mutable byte snapshot."""

    def __init__(
        self,
        model_id: str,
        base_version: str,
        target_version: str,
        snapshot: dict[str, np.ndarray],
        metadata: dict[str, dict],
        format_digest: str,
        base_digest: str,
        bucket_bytes: int,
    ) -> None:
        self.model_id = model_id
        self.base_version = base_version
        self.target_version = target_version
        self.snapshot = snapshot
        self.metadata = metadata
        self.format_digest = format_digest
        self.base_digest = base_digest
        self.bucket_bytes = bucket_bytes
        self.ordinal = 0
        self.coverage = {
            name: {**item, "state": "clean"} for name, item in metadata.items()
        }

    def encode_bucket(
        self, bucket: CanonicalBucket
    ) -> tuple[int, bytes, int, tuple[str, ...]] | None:
        entries = []
        decoded = bytearray()
        ordinal = self.ordinal

        for name, tensor in bucket:
            name = name.removeprefix("module.")
            old = self.snapshot[name]
            data = _tensor_bytes(tensor)
            new = np.frombuffer(data, dtype=np.uint8).copy()
            if len(new) != len(old):
                raise ValueError(f"{name} changed byte size")

            delta = np.bitwise_xor(new, old)
            digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
            self.snapshot[name] = new
            self.metadata[name]["target_digest"] = digest
            coverage = {**self.metadata[name], "state": "clean"}

            if np.any(delta):
                offset = len(decoded)
                decoded.extend(delta.tobytes())
                entries.append(
                    {
                        **self.metadata[name],
                        "offset": offset,
                        "target_digest": digest,
                    }
                )
                coverage["state"] = "dirty"
                coverage["bucket_ordinal"] = ordinal
            self.coverage[name] = coverage

        if not entries:
            return None

        header = canonical_json(
            {
                "base_digest": self.base_digest,
                "base_version": self.base_version,
                "compression": "zstd",
                "decoded_size": len(decoded),
                "delta": "xor",
                "entries": entries,
                "format_digest": self.format_digest,
                "model_id": self.model_id,
                "ordinal": ordinal,
                "schema": f"{_SCHEMA}.bucket",
                "target_version": self.target_version,
            }
        )
        compressed = zstandard.ZstdCompressor(level=3).compress(bytes(decoded))
        self.ordinal += 1
        return (
            ordinal,
            _BUCKET_MAGIC + struct.pack(">I", len(header)) + header + compressed,
            len(decoded),
            tuple(entry["name"] for entry in entries),
        )

    def finish(self) -> tuple[str, list[dict]]:
        coverage = [self.coverage[name] for name in sorted(self.coverage)]
        return snapshot_digest(self.metadata), coverage
