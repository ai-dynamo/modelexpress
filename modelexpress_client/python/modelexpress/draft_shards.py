# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pick the shards a speculative draft head needs out of a shared checkpoint.

MTP / NextN drafts live inside the target model's checkpoint. A draft pass
that reads every shard to keep the handful of tensors it owns re-reads the
whole model from storage, which for a large checkpoint costs more than the
target load it follows. The helpers here read ``model.safetensors.index.json``
and keep only the shards holding draft-prefixed tensors.

Every outcome other than SELECTED means "load everything": a checkpoint whose
draft head cannot be positively identified is never truncated to nothing.
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum, auto

logger = logging.getLogger("modelexpress.draft_shards")

SAFETENSORS_INDEX_NAME = "model.safetensors.index.json"
CONFIG_JSON_NAME = "config.json"

# Checkpoint families that name draft tensors by a fixed prefix rather than by
# a layer index past the target's last layer:
#   - "mtp."              Qwen3-Next / Qwen3.5 (mtp.fc.weight, mtp.layers.0.*)
#   - "model.mtp."        LongCat-Flash (model.mtp.layers.0.*, model.mtp.norm.*)
#   - "model.mtp_layers." MiMo (model.mtp_layers.0.*)
STATIC_DRAFT_PREFIXES: tuple[str, ...] = ("mtp.", "model.mtp.", "model.mtp_layers.")


class DraftShardSelection(Enum):
    """Outcome of picking the draft's own shards out of a checkpoint."""

    SELECTED = auto()
    NO_DRAFT_WEIGHTS = auto()
    UNRESOLVED = auto()


def read_local_json(directory: str, name: str) -> dict | None:
    """Read a JSON file by name from a local directory, or None if absent."""
    path = os.path.join(directory, name)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else None


def _layer_counts(config: dict) -> tuple[object, object]:
    """Return (num_hidden_layers, num_nextn_predict_layers) from a config dict.

    Multimodal checkpoints keep the language model's counts under text_config.
    """
    for section in (config, config.get("text_config")):
        if not isinstance(section, dict):
            continue
        base = section.get("num_hidden_layers")
        extra = section.get("num_nextn_predict_layers")
        if base is not None and extra is not None:
            return base, extra
    return None, None


def extra_layer_draft_prefixes(config: dict | None) -> tuple[str, ...]:
    """Draft prefixes for the extra-decoder-layer convention.

    DeepSeek-V3, GLM-4.5/5.x and Step-3.5 store the MTP head as decoder layers
    model.layers.{num_hidden_layers + i}, i < num_nextn_predict_layers. Both
    counts come from the checkpoint's on-disk config.json, not the engine's
    runtime draft config: SGLang rewrites num_hidden_layers for some families
    (LongCat), which would then alias an ordinary target layer. Any other
    shape returns () so the selector falls back to loading every shard.
    """
    if not config:
        return ()
    base, extra = _layer_counts(config)
    if isinstance(base, bool) or isinstance(extra, bool):
        return ()
    if not isinstance(base, int) or not isinstance(extra, int):
        return ()
    if base <= 0 or extra <= 0:
        return ()
    prefixes: list[str] = []
    for i in range(extra):
        prefixes.append(f"model.layers.{base + i}.")
        prefixes.append(f"layers.{base + i}.")
        prefixes.append(f"model.language_model.layers.{base + i}.")
    return tuple(prefixes)


def draft_prefixes_for(config: dict | None) -> tuple[str, ...]:
    """All tensor-name prefixes that identify draft weights for a checkpoint."""
    return STATIC_DRAFT_PREFIXES + extra_layer_draft_prefixes(config)


def _shard_names(path: str, hf_folder: str | None) -> set[str]:
    names = {os.path.basename(path)}
    if hf_folder and "://" not in hf_folder and "://" not in path:
        try:
            names.add(os.path.relpath(path, hf_folder).replace(os.sep, "/"))
        except ValueError:
            pass
    return names


def select_draft_weight_files(
    index: dict | None,
    weight_files: list[str],
    draft_prefixes: tuple[str, ...],
    hf_folder: str | None = None,
) -> tuple[DraftShardSelection, list[str]]:
    """Return the shards of ``weight_files`` that hold draft-prefixed tensors.

    ``index`` is the parsed safetensors index. Shard names in its weight_map
    are matched against each file's basename and, for a local ``hf_folder``,
    its path relative to that folder.
    """
    if not draft_prefixes:
        return DraftShardSelection.NO_DRAFT_WEIGHTS, []
    if not index:
        return DraftShardSelection.UNRESOLVED, []
    try:
        weight_map = index.get("weight_map") or {}
        wanted = {
            os.path.normpath(fname).replace(os.sep, "/")
            for tname, fname in weight_map.items()
            if isinstance(tname, str)
            and isinstance(fname, str)
            and tname.startswith(draft_prefixes)
        }
        if not wanted:
            return DraftShardSelection.NO_DRAFT_WEIGHTS, []
        subset = [
            path for path in weight_files if _shard_names(path, hf_folder) & wanted
        ]
        if not subset:
            return DraftShardSelection.UNRESOLVED, []
        return DraftShardSelection.SELECTED, subset
    except Exception as exc:
        logger.warning("Draft weight-file selection failed: %s", exc)
        return DraftShardSelection.UNRESOLVED, []


def narrow_to_draft_shards(
    hf_folder: str,
    weight_files: list[str],
    *,
    log_prefix: str = "",
) -> list[str]:
    """Narrow a resolved local checkpoint's shard list to the draft's shards.

    Reads the index and config.json next to the shards. Returns
    ``weight_files`` unchanged whenever the draft head cannot be positively
    identified, logging why.
    """
    if not hf_folder or "://" in hf_folder:
        logger.warning(
            "%s[draft] %s is not a local directory; loading all %d shards",
            log_prefix,
            hf_folder,
            len(weight_files),
        )
        return weight_files
    try:
        index = read_local_json(hf_folder, SAFETENSORS_INDEX_NAME)
        config = read_local_json(hf_folder, CONFIG_JSON_NAME)
    except Exception as exc:
        logger.warning(
            "%s[draft] could not read %s or %s under %s (%s); loading all %d shards",
            log_prefix,
            SAFETENSORS_INDEX_NAME,
            CONFIG_JSON_NAME,
            hf_folder,
            exc,
            len(weight_files),
        )
        return weight_files
    prefixes = draft_prefixes_for(config)
    selection, subset = select_draft_weight_files(
        index, weight_files, prefixes, hf_folder=hf_folder
    )
    if selection is DraftShardSelection.UNRESOLVED:
        logger.warning(
            "%s[draft] could not resolve draft-only shards from %s under %s; "
            "loading all %d shards",
            log_prefix,
            SAFETENSORS_INDEX_NAME,
            hf_folder,
            len(weight_files),
        )
        return weight_files
    if selection is DraftShardSelection.NO_DRAFT_WEIGHTS:
        logger.info(
            "%s[draft] %s under %s holds no draft tensors (checked prefixes %s); "
            "loading all %d shards",
            log_prefix,
            SAFETENSORS_INDEX_NAME,
            hf_folder,
            prefixes,
            len(weight_files),
        )
        return weight_files
    logger.info(
        "%s[draft] loading %d of %d safetensors shards for draft weights: %s",
        log_prefix,
        len(subset),
        len(weight_files),
        [os.path.basename(path) for path in subset],
    )
    return subset
