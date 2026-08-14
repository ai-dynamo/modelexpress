# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata prefetch for workers without shared model storage.

An engine needs a resolvable local snapshot long before it loads weights:
vLLM calls ``snapshot_download`` while parsing engine args, and the tokenizer
follows right after. Neither can wait for the weight loader, and neither is
served by P2P, which transfers GPU tensors and never repository files.

So the two halves are fetched at different times. Everything except the
weights is pulled here, unconditionally, before the engine resolves the model.
The weights stay with the strategy chain, where P2P keeps first refusal and
:mod:`modelexpress.load_strategy.server_cache_strategy` only steps in on a
miss. Fetching metadata early costs one small transfer and does not weaken
P2P-first, because no weight ever moves on this path.

This module also remembers which repo id produced which snapshot directory.
vLLM rewrites ``ModelConfig.model`` in place with the resolved local path, so
by the time a strategy runs, the repo id the server needs is gone unless
something recorded it.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path

from . import envs

logger = logging.getLogger("modelexpress.model_prefetch")

COMMIT_HASH_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")

_REPO_DIR_PREFIX = "models--"

# Reentrant: ensure_metadata holds this across the whole install and calls
# helpers that take it again.
_lock = threading.RLock()
_snapshot_to_repo_id: dict[str, str] = {}
_installed: set[str] = set()


def is_enabled() -> bool:
    """Return whether server-backed model fetching is configured."""
    if not envs.MODEL_EXPRESS_NO_SHARED_STORAGE:
        return False
    return bool(envs.MODEL_EXPRESS_URL or envs.MX_SERVER_ADDRESS)


def is_repo_id(model: str) -> bool:
    """Return whether ``model`` looks like a Hugging Face repo id we can fetch."""
    if not model or os.path.isabs(model) or os.path.sep in os.path.dirname(model or ""):
        return False
    try:
        from huggingface_hub.utils import validate_repo_id
    except ImportError:
        return False
    try:
        validate_repo_id(model)
    except Exception:
        return False
    return not Path(model).exists()


def ensure_metadata(repo_id: str, revision: str | None = None) -> Path | None:
    """Install the model's non-weight files locally, once per process.

    Returns the snapshot directory, or None when the prefetch does not apply.
    Errors from the server propagate; callers on the engine's critical path
    decide whether to fail or fall through.

    The install runs under the lock so that a caller arriving mid-install waits
    for that result instead of walking away with None -- the engine resolves
    the model right after this returns, and a None would send it looking for a
    snapshot that is still being written.
    """
    if not is_enabled() or not is_repo_id(repo_id):
        return None

    from .model_client import ModelCacheClient

    with _lock:
        if repo_id in _installed:
            # Later calls in the same process (tokenizer, processor) resolve
            # from the snapshot the first call installed.
            return _known_snapshot(repo_id)

        with ModelCacheClient(chunk_size=configured_chunk_size()) as client:
            snapshot_path = client.install_metadata_snapshot(repo_id)

        _warn_on_revision_mismatch(repo_id, revision, snapshot_path)
        _installed.add(repo_id)
        _snapshot_to_repo_id[_normalize(snapshot_path)] = repo_id
        return snapshot_path


def repo_id_for(model: str | os.PathLike[str]) -> str | None:
    """Map a resolved snapshot path (or a repo id) back to its repo id.

    The in-process record only covers the process that ran the prefetch. vLLM
    loads weights in a separate EngineCore process, which never sees it, so the
    cache layout itself has to be authoritative: a snapshot path carries the
    repo id in its ``models--<org>--<name>`` directory.
    """
    model = str(model)
    with _lock:
        recorded = _snapshot_to_repo_id.get(_normalize(model))
    if recorded is not None:
        return recorded
    if is_repo_id(model):
        return model
    return repo_id_from_cache_path(model)


def repo_id_from_cache_path(path: str | os.PathLike[str]) -> str | None:
    """Recover a repo id from a Hugging Face cache path, or None."""
    try:
        from huggingface_hub.utils import validate_repo_id
    except ImportError:
        return None

    candidate = Path(str(path))
    for part in (candidate, *candidate.parents):
        name = part.name
        if not name.startswith(_REPO_DIR_PREFIX):
            continue
        repo_id = name[len(_REPO_DIR_PREFIX):].replace("--", "/")
        try:
            validate_repo_id(repo_id)
        except Exception:
            return None
        return repo_id
    return None


def reset() -> None:
    """Forget prefetch state. For tests."""
    with _lock:
        _snapshot_to_repo_id.clear()
        _installed.clear()


def _known_snapshot(repo_id: str) -> Path | None:
    with _lock:
        for snapshot_path, known_repo_id in _snapshot_to_repo_id.items():
            if known_repo_id == repo_id:
                return Path(snapshot_path)
    return None


def _normalize(path: str | os.PathLike[str]) -> str:
    return os.path.normpath(str(path))


def configured_chunk_size() -> int | None:
    """Return the configured transfer chunk size, or None to use the default.

    Every rejection falls back to the default rather than raising: this runs on
    the engine's startup path, and a bad env var should not be the reason a
    worker fails to start.
    """
    from .model_client import MAX_CHUNK_SIZE

    raw = envs.MODEL_EXPRESS_TRANSFER_CHUNK_SIZE
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid MODEL_EXPRESS_TRANSFER_CHUNK_SIZE=%r; using default", raw)
        return None
    if not 0 < value <= MAX_CHUNK_SIZE:
        logger.warning(
            "MODEL_EXPRESS_TRANSFER_CHUNK_SIZE=%r must be between 1 and %d; using default",
            raw,
            MAX_CHUNK_SIZE,
        )
        return None
    return value


def _warn_on_revision_mismatch(repo_id: str, revision: str | None, snapshot_path: Path) -> None:
    """Warn when the server's snapshot is not the revision the engine asked for.

    ``ModelDownloadRequest`` and ``ModelFilesRequest`` both carry an optional
    ``revision``, but this client leaves it unset, so every request is
    unpinned and the server answers with the default revision it resolves. A
    mismatch is not silently unsafe -- Hugging Face addresses pinned revisions
    by directory name, so the engine simply will not see this snapshot -- but
    the failure that follows is unhelpful unless the real reason is logged
    here.
    """
    if not revision or revision == "main":
        return
    if COMMIT_HASH_PATTERN.fullmatch(revision) and snapshot_path.name == revision:
        return
    logger.warning(
        "ModelExpress Server returned commit %s for %s, but this worker asked for "
        "revision %r. This client does not pin a revision on the model RPCs yet, so "
        "the engine will not resolve this snapshot.",
        snapshot_path.name,
        repo_id,
        revision,
    )
