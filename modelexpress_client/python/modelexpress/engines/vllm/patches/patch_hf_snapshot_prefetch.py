# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch model metadata from ModelExpress Server before the engine resolves it."""

from __future__ import annotations

import logging

from .... import model_prefetch

logger = logging.getLogger(__name__)

# snapshot_download is re-exported into several huggingface_hub namespaces, and
# callers bind it from whichever one they imported. vLLM reaches it through
# HfApi.snapshot_download, which calls the name bound in hf_api, so patching the
# package attribute alone would miss the path that actually fails.
_PATCH_TARGETS = (
    "huggingface_hub._snapshot_download",
    "huggingface_hub.hf_api",
    "huggingface_hub",
)


def patch_hf_snapshot_prefetch() -> bool:
    """Route offline snapshot resolution through the ModelExpress model cache.

    Without shared storage the engine has no way to resolve a model before its
    weight loader runs: it calls snapshot_download while parsing engine args,
    finds nothing on disk, and fails under HF_HUB_OFFLINE. This hook fills the
    local cache from the server first, then lets the original call proceed.
    Weights are not fetched here -- P2P keeps first refusal on those.
    """
    if not model_prefetch.is_enabled():
        return False

    try:
        import importlib

        from huggingface_hub import _snapshot_download
    except ImportError:
        return False

    original = _snapshot_download.snapshot_download
    if getattr(original, "__modelexpress_patched__", False):
        return False

    def patched(*args, **kwargs):
        """Install the snapshot from the server, then run the original call.

        The original always runs, so the engine keeps the behaviour and the
        errors it already handles. Prefetch failures are logged and swallowed:
        this sits on the startup path, and a ModelExpress error here would
        replace the diagnostic the engine expects with an unfamiliar one.
        """
        repo_id = kwargs["repo_id"] if "repo_id" in kwargs else (args[0] if args else None)
        repo_type = kwargs.get("repo_type") or "model"
        if isinstance(repo_id, str) and repo_type == "model":
            try:
                model_prefetch.ensure_metadata(repo_id, kwargs.get("revision"))
            except Exception as exc:
                # Let the original call report the failure the engine expects
                # instead of replacing it with a ModelExpress error.
                logger.warning(
                    "ModelExpress metadata prefetch failed for %s: %s", repo_id, exc
                )
        return original(*args, **kwargs)

    patched.__modelexpress_patched__ = True
    patched.__wrapped__ = original

    for module_name in _PATCH_TARGETS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if getattr(module, "snapshot_download", None) is original:
            module.snapshot_download = patched
    return True
