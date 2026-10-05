# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select checkpoint shards before SGLang starts the ModelStreamer read."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlsplit

from .draft_weights import DraftWeightAdapter

logger = logging.getLogger(__name__)
_INDEX_NAME = "model.safetensors.index.json"


def _read_index(model_uri: str) -> dict | None:
    if urlsplit(model_uri).scheme in {"s3", "gs", "az"}:
        from runai_model_streamer import pull_files

        with tempfile.TemporaryDirectory(prefix="mx-index-") as directory:
            pull_files(
                model_uri,
                directory,
                allow_pattern=[_INDEX_NAME, f"*/{_INDEX_NAME}"],
            )
            index_path = Path(directory) / _INDEX_NAME
            if not index_path.is_file():
                return None
            return json.loads(index_path.read_text(encoding="utf-8"))

    path = Path(model_uri) / _INDEX_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def select_role_shards(
    model_uri: str,
    available_files: Sequence[str],
    selector: DraftWeightAdapter,
    role: str,
) -> list[str] | None:
    """Return a file subset, or None to retain SGLang's full-file fallback."""
    try:
        index = _read_index(model_uri)
    except Exception as exc:
        # This is an optional optimization. Storage/index failures must leave
        # SGLang's existing full-file loader available as the fallback.
        logger.warning("Checkpoint index read failed; using all shards: %s", exc)
        return None
    if not isinstance(index, dict):
        return None
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str) and isinstance(filename, str)
        for name, filename in weight_map.items()
    ):
        return None
    selected_names = {
        filename
        for name, filename in weight_map.items()
        if selector.uses_checkpoint_tensor(name, role)
    }
    if not selected_names:
        return None
    available_names = {os.path.basename(path) for path in available_files}
    indexed_names = set(weight_map.values())
    if (
        len(available_names) != len(available_files)
        or indexed_names != available_names
    ):
        return None
    return [
        path
        for path in available_files
        if os.path.basename(path) in selected_names
    ]
