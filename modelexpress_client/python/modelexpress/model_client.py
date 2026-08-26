# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client for the ModelExpress Server model cache.

Wraps the ``ModelService`` RPCs and installs what they return into the local
Hugging Face cache. Two entry points, matching the two moments a worker needs
files from the server:

- :meth:`ModelCacheClient.install_metadata_snapshot` runs before the engine
  starts. It fetches everything except weights, which is enough for config and
  tokenizer resolution, and leaves P2P as the first choice for the weights.
- :meth:`ModelCacheClient.install_weight_files` runs after a P2P miss and adds
  the weights to the snapshot the engine already resolved.

The server streams files relative to its own snapshot directory; layout and
atomicity live in :mod:`modelexpress.model_snapshot`.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Mapping, Sequence

import grpc

from . import auth
from . import model_pb2
from . import model_pb2_grpc
from .client import _get_server_url
from .model_snapshot import (
    ModelSnapshotCache,
    ModelSnapshotError,
    SnapshotSink,
    safe_commit_hash,
    split_by_weight,
)

logger = logging.getLogger("modelexpress.model_client")

# Case-insensitive: huggingface_hub only treats the lowercase form as a commit
# hash, but an uppercase one still names a commit and must not be allowed to
# resolve to a different one.
_COMMIT_HASH_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")

DEFAULT_MAX_MESSAGE_SIZE = 100 * 1024 * 1024
# The server default is 32 KiB, which costs one round trip per 32 KiB of a
# multi-GB weight file. Stay an order of magnitude below gRPC's usual 4 MiB
# message ceiling so the server never has to raise its encoding limit.
DEFAULT_CHUNK_SIZE = 1024 * 1024
MAX_CHUNK_SIZE = (1 << 32) - 1


class ModelCacheError(RuntimeError):
    """Raised when the server's download or file stream cannot be trusted."""


class ModelCacheClient:
    """Synchronous client for the ModelExpress ``ModelService`` RPCs."""

    def __init__(
        self,
        server_url: str | None = None,
        cache_directory: str | os.PathLike[str] | None = None,
        chunk_size: int | None = None,
        max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
    ):
        self.server_url = _get_server_url(server_url)
        self.cache_directory = cache_directory
        self.chunk_size = DEFAULT_CHUNK_SIZE if chunk_size is None else chunk_size
        if not 0 < self.chunk_size <= MAX_CHUNK_SIZE:
            raise ValueError(f"chunk_size must be between 1 and {MAX_CHUNK_SIZE} bytes")
        if max_message_size <= 0:
            raise ValueError("max_message_size must be positive")

        self._max_message_size = max_message_size
        self._channel: grpc.Channel | None = None
        self._stub: model_pb2_grpc.ModelServiceStub | None = None

    def __enter__(self) -> "ModelCacheClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def stub(self) -> model_pb2_grpc.ModelServiceStub:
        """Return (and lazily create) the model-service stub."""
        if self._stub is None:
            options = [
                ("grpc.max_send_message_length", self._max_message_size),
                ("grpc.max_receive_message_length", self._max_message_size),
                # No RPC here carries a deadline, because a cold-cache download
                # is legitimately slow. Keepalive is what separates "slow" from
                # "dead": without it a silently dropped connection blocks the
                # engine's startup path forever, so the pod neither becomes
                # ready nor crash-loops.
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 10_000),
                ("grpc.keepalive_permit_without_calls", 1),
                ("grpc.http2.max_pings_without_data", 0),
            ]
            self._channel = auth.with_auth(
                grpc.insecure_channel(self.server_url, options=options)
            )
            self._stub = model_pb2_grpc.ModelServiceStub(self._channel)
            logger.debug("ModelCacheClient connected to %s", self.server_url)
        return self._stub

    def close(self) -> None:
        """Close the underlying gRPC channel."""
        if self._channel is not None:
            self._channel.close()
        self._channel = None
        self._stub = None

    # -- RPCs -----------------------------------------------------------------

    def ensure_downloaded(
        self,
        model_name: str,
        provider: int = model_pb2.HUGGING_FACE,
        ignore_weights: bool = False,
        revision: str | None = None,
    ) -> str | None:
        """Block until the server reports the model as downloaded.

        ``ignore_weights`` asks for a metadata-only download. Servers that
        key their registry entries on the weight mode keep that claim separate
        from a later full-weight request, so the weight phase asks again and
        gets its own download. Servers older than that key on the model name
        alone: the metadata-only claim registers the model as complete there,
        no later weight fetch happens, and the weights become unreachable
        through the cache -- see the server requirement in DEPLOYMENT.md.

        ``revision`` pins the download to one commit. The server resolves a
        pinned revision before claiming its download lease, so the claim is
        scoped to that exact commit and the reply names it whether the
        download was fresh or already present. Servers from before the
        revision field ignore the pin.

        Returns the commit the server resolved the request to, or ``None``
        when it named none. A server that already holds an unpinned model
        answers without naming a revision, so ``None`` is ordinary and means
        only that the caller learned nothing -- not that the local cache is
        current.
        """
        request = model_pb2.ModelDownloadRequest(
            model_name=model_name,
            provider=provider,
            ignore_weights=ignore_weights,
        )
        if revision is not None:
            request.revision = revision
        for update in self.stub.EnsureModelDownloaded(request):
            if update.message:
                logger.info("Model %s: %s", model_name, update.message)
            if update.status == model_pb2.DOWNLOADED:
                return _reported_revision(model_name, update)
            if update.status == model_pb2.ERROR:
                raise ModelCacheError(
                    f"ModelExpress failed to download {model_name}: "
                    f"{update.message or 'unknown server error'}"
                )
            if update.status != model_pb2.DOWNLOADING:
                raise ModelCacheError(
                    f"ModelExpress reported unknown status {update.status} for {model_name}"
                )
        raise ModelCacheError(
            f"ModelExpress status stream ended before {model_name} was downloaded"
        )

    def list_files(
        self,
        model_name: str,
        provider: int = model_pb2.HUGGING_FACE,
        ignore_weights: bool = False,
        revision: str | None = None,
    ) -> dict[str, int]:
        """Return the server's file manifest as {relative_path: size}.

        ``revision`` pins the manifest to one snapshot, so a server whose
        default revision moves between this call and the stream cannot answer
        the two from different commits.
        """
        request = model_pb2.ModelFilesRequest(
            model_name=model_name,
            provider=provider,
            ignore_weights=ignore_weights,
        )
        if revision is not None:
            request.revision = revision
        response = self.stub.ListModelFiles(request)
        return _manifest_to_dict(response)

    # -- Installation ---------------------------------------------------------

    def install_metadata_snapshot(
        self,
        model_name: str,
        provider: int = model_pb2.HUGGING_FACE,
        requested_revision: str | None = None,
    ) -> Path:
        """Install every non-weight file as a resolvable Hugging Face snapshot.

        Asks the server for a metadata-only download, so a cold server does
        not fetch the weights before ``RdmaStrategy`` has had its chance at
        them. Returns the snapshot directory.

        ``requested_revision`` is the revision the engine asked for -- a
        branch, a tag, or a commit hash. Asking the server for it is what
        makes the engine's own resolution succeed afterwards, since the engine
        looks the snapshot up by the revision it requested rather than by
        whichever one the server holds by default. A pinned request reuses a
        local snapshot whenever that commit is already complete on disk; an
        unpinned one needs the server to name the revision it is holding,
        which it does not do for a model it already has.
        """
        revision = self.ensure_downloaded(
            model_name, provider, ignore_weights=True, revision=requested_revision
        )
        if requested_revision is not None:
            if revision is None:
                raise ModelCacheError(
                    f"ModelExpress did not confirm revision {requested_revision!r} for "
                    f"{model_name}; the server may predate pinned-revision support, and "
                    "installing its default revision would not be what was asked for"
                )
            # A branch or tag legitimately resolves to some other string. A
            # commit hash names one revision and cannot: resolving it to
            # another would install that one and leave a ref pointing the
            # engine's request at it.
            if (
                _COMMIT_HASH_PATTERN.match(requested_revision)
                and requested_revision.lower() != revision.lower()
            ):
                raise ModelCacheError(
                    f"ModelExpress resolved {requested_revision!r} to commit {revision} "
                    f"for {model_name}; a commit hash cannot resolve to another commit"
                )
        # Follow-up calls carry the commit the server resolved, not the string
        # the engine wrote: a branch or tag that moves between these calls would
        # otherwise answer them from two different commits.
        manifest = self.list_files(
            model_name, provider, ignore_weights=True, revision=revision
        )
        # Still split: an older server ignores ignore_weights and answers with
        # the whole repository.
        metadata_paths, _ = split_by_weight(manifest.keys())
        if not metadata_paths:
            raise ModelCacheError(
                f"ModelExpress returned no non-weight files for {model_name}"
            )
        expected = {path: manifest[path] for path in metadata_paths}

        cache = ModelSnapshotCache(model_name, self.cache_directory)
        with cache.lock():
            if requested_revision is not None:
                existing = cache.resolve_pinned_snapshot(expected, revision)
            else:
                existing = cache.resolve_snapshot(expected, revision)
            if existing is not None:
                # The ref belongs to the request, not to the snapshot: an
                # earlier install of this same commit under a different
                # revision left no ref this one can be found by. Reuse skips
                # publish(), so record it here or the engine's lookup fails
                # against a directory that is sitting right there.
                cache.write_revision_ref(revision, requested_revision)
                logger.info("Reusing local snapshot for %s at %s", model_name, existing)
                return existing

            staging = cache.staging()
            try:
                commit_hash = self._stream_into(
                    model_name,
                    provider,
                    metadata_paths,
                    expected,
                    staging,
                    expected_commit=revision,
                    revision=revision,
                )
                snapshot_path = staging.publish(
                    commit_hash, expected, requested_revision=requested_revision
                )
            except BaseException:
                staging.discard()
                raise

        logger.info(
            "Installed %d metadata files for %s at %s",
            len(expected),
            model_name,
            snapshot_path,
        )
        return snapshot_path

    def install_weight_files(
        self,
        model_name: str,
        snapshot_path: Path,
        provider: int = model_pb2.HUGGING_FACE,
    ) -> None:
        """Add the model's weight files to an already published snapshot.

        The download is pinned to the commit the snapshot is named after, so
        a server whose default revision has moved past this snapshot fetches
        the snapshot's own weights instead of answering from the newer
        default. Refuses to write weights the server reports under a
        different commit than the snapshot is named after: the engine
        addresses pinned revisions by directory name, so mixing commits would
        hand it weights from one revision under the name of another.
        """
        try:
            revision = self.ensure_downloaded(
                model_name, provider, revision=snapshot_path.name
            )
        except grpc.RpcError as exc:
            # Resolving a pin needs the Hub, and the server's cache fallback
            # during a Hub outage exists only on the unpinned path -- so a
            # pinned request would turn an outage the unpinned path survives
            # into a failure. Degrade to the unpinned request instead; the
            # revision check below and the stream's first-chunk commit
            # validation still refuse weights from any other commit.
            logger.warning(
                "Pinned download of %s at commit %s failed (%s); retrying unpinned",
                model_name,
                snapshot_path.name,
                exc,
            )
            revision = self.ensure_downloaded(model_name, provider)
        if revision is not None and revision != snapshot_path.name:
            raise ModelCacheError(
                f"Server resolved {model_name} to commit {revision} but the local "
                f"snapshot is {snapshot_path.name}; refusing to mix revisions"
            )
        manifest = self.list_files(model_name, provider, revision=revision)
        _, weight_paths = split_by_weight(manifest.keys())
        if not weight_paths:
            raise ModelCacheError(
                f"ModelExpress returned no weight files for {model_name}"
            )
        expected = {path: manifest[path] for path in weight_paths}

        cache = ModelSnapshotCache(model_name, self.cache_directory)
        with cache.lock():
            if cache.has_files(snapshot_path, expected):
                logger.info("Weights for %s already present at %s", model_name, snapshot_path)
                return

            patch = cache.patch(snapshot_path)
            try:
                self._stream_into(
                    model_name,
                    provider,
                    weight_paths,
                    expected,
                    patch,
                    expected_commit=snapshot_path.name,
                    revision=revision,
                )
                patch.commit()
            except BaseException:
                patch.rollback()
                raise
            finally:
                patch.close()

        logger.info(
            "Installed %d weight files for %s at %s",
            len(expected),
            model_name,
            snapshot_path,
        )

    # -- Stream handling ------------------------------------------------------

    def _stream_into(
        self,
        model_name: str,
        provider: int,
        paths: Sequence[str],
        expected: Mapping[str, int],
        sink: SnapshotSink,
        expected_commit: str | None = None,
        revision: str | None = None,
    ) -> str:
        """Stream ``paths`` into ``sink``, validating the protocol as it goes.

        Returns the commit hash the server reported for the snapshot. When
        ``expected_commit`` is given, the stream is rejected on the very first
        chunk if the commit differs -- the alternative is transferring the
        whole model before noticing, which for a sharded checkpoint means tens
        of gigabytes thrown away. ``revision`` pins the server to one snapshot
        so the manifest and the stream cannot come from different commits.
        """
        request = model_pb2.ModelFilesRequest(
            model_name=model_name,
            provider=provider,
            chunk_size=self.chunk_size,
            file_selector=model_pb2.ModelFileSelector(paths=list(paths)),
        )
        if revision is not None:
            request.revision = revision

        commit_hash: str | None = None
        received: dict[str, int] = {}
        current_path: str | None = None
        current_size = 0
        current_total = 0
        saw_final_marker = False

        for chunk in self.stub.StreamModelFiles(request):
            if saw_final_marker:
                raise ModelCacheError("Server sent data after the final stream marker")

            if commit_hash is None:
                if not chunk.HasField("commit_hash"):
                    raise ModelCacheError("First file chunk did not carry a commit hash")
                commit_hash = _require_commit_hash(chunk.commit_hash)
                if expected_commit is not None and commit_hash != expected_commit:
                    raise ModelCacheError(
                        f"Server streamed commit {commit_hash} but the local snapshot "
                        f"is {expected_commit}; refusing to mix revisions"
                    )
            elif chunk.HasField("commit_hash") and chunk.commit_hash != commit_hash:
                raise ModelCacheError("Server changed the commit hash mid-stream")

            relative_path = chunk.relative_path
            if relative_path not in expected:
                raise ModelCacheError(f"Server streamed an unrequested file: {relative_path!r}")

            if current_path != relative_path:
                if current_path is not None:
                    raise ModelCacheError(
                        f"Server started {relative_path!r} before {current_path!r} finished"
                    )
                if relative_path in received:
                    raise ModelCacheError(f"Server streamed {relative_path!r} twice")
                if chunk.offset != 0:
                    raise ModelCacheError(
                        f"First chunk of {relative_path!r} has offset {chunk.offset}"
                    )
                if chunk.total_size != expected[relative_path]:
                    raise ModelCacheError(
                        f"Size mismatch for {relative_path!r}: manifest "
                        f"{expected[relative_path]}, stream {chunk.total_size}"
                    )
                sink.begin_file(relative_path)
                current_path = relative_path
                current_size = 0
                current_total = chunk.total_size
            elif chunk.total_size != current_total:
                raise ModelCacheError(f"Server changed the size of {relative_path!r} mid-file")

            if chunk.offset != current_size:
                raise ModelCacheError(
                    f"Unexpected offset {chunk.offset} for {relative_path!r}, "
                    f"expected {current_size}"
                )
            if current_size + len(chunk.data) > current_total:
                raise ModelCacheError(f"Data for {relative_path!r} exceeds its advertised size")
            if chunk.is_last_file and not chunk.is_last_chunk:
                raise ModelCacheError("Final-file marker set before the file's final chunk")

            sink.write(chunk.data)
            current_size += len(chunk.data)

            if chunk.is_last_chunk:
                if current_size != current_total:
                    raise ModelCacheError(
                        f"Incomplete file {relative_path!r}: received {current_size}, "
                        f"expected {current_total}"
                    )
                sink.end_file()
                received[relative_path] = current_size
                current_path = None
                saw_final_marker = chunk.is_last_file
            elif current_size == current_total:
                raise ModelCacheError(f"File {relative_path!r} completed without a final chunk")

        if commit_hash is None:
            raise ModelCacheError("Server streamed no model files")
        if not saw_final_marker:
            raise ModelCacheError("Stream ended before the final file marker")
        if received != dict(expected):
            missing = sorted(set(expected) - set(received))
            raise ModelCacheError(f"Stream did not match the manifest; missing files: {missing}")
        return commit_hash


def _manifest_to_dict(manifest: model_pb2.ModelFileList) -> dict[str, int]:
    """Validate a file manifest and return it keyed by relative path."""
    files: dict[str, int] = {}
    for file_info in manifest.files:
        relative_path = file_info.relative_path
        if relative_path in files:
            raise ModelCacheError(f"Server returned duplicate file path: {relative_path!r}")
        files[relative_path] = file_info.size

    if not files:
        raise ModelCacheError("Server returned an empty model file manifest")
    total = sum(files.values())
    if total != manifest.total_size:
        raise ModelCacheError(
            f"Manifest total mismatch: files add up to {total} bytes, "
            f"manifest advertises {manifest.total_size} bytes"
        )
    return files


def _reported_revision(model_name: str, update: model_pb2.ModelStatusUpdate) -> str | None:
    """Return the commit an update names, or None when it names none.

    An unusable value degrades to None rather than raising. This runs on the
    engine's startup path, and the only thing the value buys is the reuse
    shortcut: dropping it costs one metadata stream, while raising would cost
    the worker its start. Nothing downstream trusts it unchecked either --
    the stream reports the commit again and validates it there.
    """
    if not update.HasField("resolved_revision"):
        return None
    try:
        return safe_commit_hash(update.resolved_revision)
    except ModelSnapshotError:
        logger.warning(
            "ModelExpress reported an unusable revision %r for %s; ignoring it",
            update.resolved_revision,
            model_name,
        )
        return None


def _require_commit_hash(commit_hash: str) -> str:
    try:
        return safe_commit_hash(commit_hash)
    except ModelSnapshotError as exc:
        raise ModelCacheError(str(exc)) from exc
