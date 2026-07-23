# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-addressed digest for Miles delta weight versions.

Hashes the delta shard bytes (not target-state checksums, which collide across
XOR deltas built on different bases). Kept as a pure lineage primitive; the MX
control-plane identity/publish path it once fed is deferred to the cross-region
design (P1).
"""

from __future__ import annotations

import hashlib
import os

_INDEX_NAME = "model.safetensors.index.json"
_READ_CHUNK_SIZE = 8 * 1024 * 1024


def _hash_file(digest, path: str) -> None:
    with open(path, "rb") as file:
        while chunk := file.read(_READ_CHUNK_SIZE):
            digest.update(chunk)


def content_digest(version_dir: str) -> str:
    """Hash delta shard bytes in filename order followed by the index bytes."""
    shard_names = sorted(
        name
        for name in os.listdir(version_dir)
        if name.endswith(".safetensors")
        and os.path.isfile(os.path.join(version_dir, name))
    )

    digest = hashlib.sha256()
    for shard_name in shard_names:
        _hash_file(digest, os.path.join(version_dir, shard_name))
    _hash_file(digest, os.path.join(version_dir, _INDEX_NAME))
    return digest.hexdigest()
