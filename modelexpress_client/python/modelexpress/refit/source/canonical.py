# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load one canonical HF snapshot and compute its logical identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


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


def _tied_names(root: Path) -> set[str]:
    """Names HF stores as a redundant copy of another parameter.

    With ``tie_word_embeddings`` the checkpoint serializes the output head as a second
    copy of the input embedding, but the trainer holds one parameter for both and so
    never gathers the copy. Counting it would put an unreachable tensor in the canonical
    set, so both the publisher and the receiver drop it.
    """
    config = root / "config.json"
    if not config.is_file():
        return set()
    try:
        with config.open() as handle:
            tied = json.load(handle).get("tie_word_embeddings", False)
    except (OSError, ValueError):
        return set()
    return {"lm_head.weight"} if tied else set()


def load_hf_snapshot(
    checkpoint: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, dict], str, str]:
    """Load the launch HF checkpoint as the byte snapshot used by Miles delta sync."""
    from safetensors import safe_open

    snapshot = {}
    metadata = {}
    tied = _tied_names(Path(checkpoint))
    for path in _checkpoint_files(Path(checkpoint)):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for source_name in handle.keys():
                name = source_name.removeprefix("module.")
                if name in tied:
                    continue
                tensor = handle.get_tensor(source_name)
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
