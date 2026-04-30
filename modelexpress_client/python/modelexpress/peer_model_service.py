# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-peer ModelService implementation.

Exposes ``StreamModelFiles``, ``ListModelFiles``, and ``EnsureModelDownloaded``
on the same gRPC server that already hosts the per-worker ``WorkerService``
(see :mod:`worker_server`). This lets a Python ModelExpress peer serve its
local cached model files directly to other Python or Rust clients, without
round-tripping through the central MX server.

Design notes:

* The serving servicer NEVER initiates downloads. ``EnsureModelDownloaded``
  returns ``DOWNLOADED`` if the model is present in the local cache,
  ``ERROR`` otherwise. Download orchestration stays with the central server.
* Path resolution mirrors the Rust ``modelexpress_common::cache::resolve_model_path``
  layout exactly so files served by a Python peer are byte-identical to
  files served by the central Rust server, and the consumer-side code in
  ``modelexpress_client/src/lib.rs::stream_model_files_from_server``
  treats both indistinguishably.
* v1 supports HuggingFace only. NGC and GCS providers return ``NOT_FOUND``;
  add when there is a real customer for peer-served NGC/GCS files.
* Path safety: any resolved file path must be under the expected
  ``model_path`` after symlink resolution. Symlinks pointing outside the
  cache root are rejected.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterator, Optional

import grpc

from . import model_pb2
from . import model_pb2_grpc

logger = logging.getLogger("modelexpress.peer_model_service")


# Mirrors modelexpress_common::constants::DEFAULT_TRANSFER_CHUNK_SIZE.
DEFAULT_TRANSFER_CHUNK_SIZE = 256 * 1024


def _hf_cache_root(local_path: Optional[Path] = None) -> Optional[Path]:
    """Resolve the HuggingFace cache root.

    Priority (matches the Rust HuggingFaceProvider):

    1. Explicit ``local_path`` argument.
    2. ``HF_HUB_CACHE`` env var.
    3. ``$HF_HOME/hub`` if ``HF_HOME`` is set.
    4. ``~/.cache/huggingface/hub`` default.
    """
    if local_path is not None:
        return Path(local_path)
    env = os.environ.get("HF_HUB_CACHE")
    if env:
        return Path(env)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _hf_repo_root(cache_root: Path, model_name: str) -> Path:
    """``cache_root/models--<org>--<name>``."""
    folder = "models--" + model_name.replace("/", "--")
    return cache_root / folder


def _hf_snapshots_dir(cache_root: Path, model_name: str) -> Path:
    return _hf_repo_root(cache_root, model_name) / "snapshots"


def _resolve_hf_model_path(
    cache_root: Path, model_name: str, commit: Optional[str]
) -> Path:
    """Resolve an HF model to its snapshot directory.

    With ``commit``: ``cache_root/models--org--name/snapshots/<commit>``.
    Without: pick the most recently modified subdirectory of ``snapshots/``.
    Raises ``FileNotFoundError`` if not present.
    """
    snapshots = _hf_snapshots_dir(cache_root, model_name)
    if commit:
        target = snapshots / commit
        if not target.is_dir():
            raise FileNotFoundError(
                f"HuggingFace snapshot {commit!r} not found at {target}"
            )
        return target
    if not snapshots.is_dir():
        raise FileNotFoundError(
            f"HuggingFace snapshots dir not found at {snapshots}"
        )
    candidates = [p for p in snapshots.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(
            f"HuggingFace snapshots dir empty at {snapshots}"
        )
    # Pick the most recent. Mirrors `latest_local_snapshot_path` in the
    # Rust HuggingFace provider, which sorts by created/modified time.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _walk_files(root: Path) -> list[tuple[str, int]]:
    """Walk ``root`` recursively and return (relative_path, size) pairs.

    Follows symlinks (HF cache stores files as symlinks to ``blobs/<hash>``).
    Skips entries that fail to stat. Skips directories themselves; only
    regular files are included.
    """
    entries: list[tuple[str, int]] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=True):
        for name in filenames:
            full = Path(dirpath) / name
            try:
                size = full.stat().st_size
            except OSError as exc:
                logger.warning("Skipping %s: %s", full, exc)
                continue
            rel = full.relative_to(root)
            entries.append((str(rel), size))
    entries.sort()
    return entries


def _is_safe_relative(model_root: Path, rel_path: str) -> bool:
    """Reject path-traversal and symlink-escape attempts.

    A file at ``model_root / rel_path`` is safe iff after symlink resolution
    it is still under ``model_root``.
    """
    candidate = (model_root / rel_path).resolve()
    try:
        candidate.relative_to(model_root.resolve())
    except ValueError:
        return False
    return True


class PeerModelServiceServicer(model_pb2_grpc.ModelServiceServicer):
    """ModelService backed by the local cache directory of a Python peer.

    Constructed with the local cache root (HF cache layout). Any HF model
    present under that root is automatically servable.
    """

    def __init__(self, cache_root: Optional[Path] = None) -> None:
        resolved = _hf_cache_root(cache_root)
        if resolved is None:
            raise RuntimeError("Could not resolve HuggingFace cache root")
        self._cache_root = Path(resolved)
        logger.info(
            "PeerModelServiceServicer: HF cache root = %s", self._cache_root
        )

    # ------------------------------------------------------------------
    # Internal: provider dispatch
    # ------------------------------------------------------------------

    def _resolve_model_path(
        self,
        provider: int,
        model_name: str,
        commit_hash: Optional[str],
        context: grpc.ServicerContext,
    ) -> Path:
        if provider == model_pb2.HUGGING_FACE:
            try:
                return _resolve_hf_model_path(
                    self._cache_root, model_name, commit_hash
                )
            except FileNotFoundError as exc:
                context.abort(grpc.StatusCode.NOT_FOUND, str(exc))
        elif provider in (model_pb2.NGC, model_pb2.GCS):
            context.abort(
                grpc.StatusCode.UNIMPLEMENTED,
                "Peer model serving is HuggingFace-only in v1",
            )
        else:
            context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"unknown provider value: {provider}",
            )
        # unreachable; abort raises
        raise RuntimeError("provider dispatch did not abort or return")

    # ------------------------------------------------------------------
    # ModelService RPCs
    # ------------------------------------------------------------------

    def EnsureModelDownloaded(
        self, request, context
    ) -> Iterator[model_pb2.ModelStatusUpdate]:
        """Return DOWNLOADED if the model is in the local cache, ERROR otherwise.

        A peer never initiates a download. Use the central MX server's
        ModelService for actual download orchestration.
        """
        try:
            self._resolve_model_path(
                request.provider, request.model_name, None, context
            )
        except grpc.RpcError:
            # _resolve_model_path called context.abort, which raises.
            return
        yield model_pb2.ModelStatusUpdate(
            model_name=request.model_name,
            status=model_pb2.DOWNLOADED,
            message="peer cache hit",
            provider=request.provider,
        )

    def ListModelFiles(self, request, context) -> model_pb2.ModelFileList:
        model_path = self._resolve_model_path(
            request.provider, request.model_name, None, context
        )
        files = _walk_files(model_path)
        total = sum(size for _, size in files)
        return model_pb2.ModelFileList(
            model_name=request.model_name,
            files=[
                model_pb2.ModelFileInfo(relative_path=rel, size=size)
                for rel, size in files
            ],
            total_size=total,
        )

    def StreamModelFiles(
        self, request, context
    ) -> Iterator[model_pb2.FileChunk]:
        model_path = self._resolve_model_path(
            request.provider, request.model_name, None, context
        )

        chunk_size = (
            request.chunk_size
            if request.chunk_size > 0
            else DEFAULT_TRANSFER_CHUNK_SIZE
        )

        # Surface the snapshot revision in the first chunk so the client
        # can lay files out under the same HF snapshot directory.
        commit_hash: Optional[str] = None
        if request.provider == model_pb2.HUGGING_FACE:
            commit_hash = model_path.name

        files = _walk_files(model_path)
        if not files:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"No files found under {model_path}",
            )
            return

        is_first_chunk = True
        total_files = len(files)
        for file_idx, (rel_path, total_size) in enumerate(files):
            is_last_file = file_idx == total_files - 1
            if not _is_safe_relative(model_path, rel_path):
                context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"Refusing to serve unsafe path: {rel_path!r}",
                )
                return
            full_path = model_path / rel_path
            if total_size == 0:
                emit_commit = commit_hash if is_first_chunk else None
                is_first_chunk = False
                yield model_pb2.FileChunk(
                    relative_path=rel_path,
                    data=b"",
                    offset=0,
                    total_size=0,
                    is_last_chunk=True,
                    is_last_file=is_last_file,
                    commit_hash=emit_commit,
                )
                continue

            try:
                f = open(full_path, "rb")
            except OSError as exc:
                context.abort(
                    grpc.StatusCode.INTERNAL,
                    f"Failed to open {rel_path!r}: {exc}",
                )
                return

            try:
                offset = 0
                while True:
                    buf = f.read(chunk_size)
                    if not buf:
                        break
                    chunk_len = len(buf)
                    is_last_chunk = (offset + chunk_len) >= total_size
                    emit_commit = commit_hash if is_first_chunk else None
                    is_first_chunk = False
                    yield model_pb2.FileChunk(
                        relative_path=rel_path,
                        data=buf,
                        offset=offset,
                        total_size=total_size,
                        is_last_chunk=is_last_chunk,
                        is_last_file=is_last_file and is_last_chunk,
                        commit_hash=emit_commit,
                    )
                    offset += chunk_len
            finally:
                f.close()
