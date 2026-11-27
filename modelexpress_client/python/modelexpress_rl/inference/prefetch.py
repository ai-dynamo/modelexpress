# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare registered object-storage versions outside the serving pause."""

from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import posixpath
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TextIO

import grpc
from modelexpress_rl import envs as rl_envs
from modelexpress_rl import refit_pb2
from modelexpress_rl.control import WeightVersion, WeightVersionState, _weight_version
from modelexpress_rl.object_storage import ObjectStorageType
from modelexpress_rl.train import WeightPayloadFormat
from .checkpoint_store import CheckpointRecord, CheckpointState
from .receiver import (
    _LocalCheckpoint,
    _ObjectStorageVersion,
    PreparationSuperseded,
    PreparedCheckpoint,
    speculative_version_id,
)
from .version_chain import resolve_replay_chain

if TYPE_CHECKING:
    from .client import ModelExpressGeneratorClient

logger = logging.getLogger(__name__)
POLL_SECONDS = 2.0
LOCK_FILE = ".prefetch.lock"
_REPLAY_FORMATS = frozenset(
    {WeightPayloadFormat.XOR_DELTA, WeightPayloadFormat.FULL_HF_CHECKPOINT}
)


@dataclass(frozen=True)
class PrefetchSettings:
    model_name: str
    max_replay_chain_length: int
    rpc_timeout_seconds: float
    storage_type: str = "S3"
    streamed: bool = False
    s3_watch: bool = False


@contextlib.contextmanager
def background_workers():
    previous = os.environ.get("MX_REFIT_DELTA_WORKERS")
    os.environ["MX_REFIT_DELTA_WORKERS"] = str(
        min(
            rl_envs.MX_REFIT_PREFETCH_WORKERS,
            rl_envs.MX_REFIT_DELTA_WORKERS,
            os.cpu_count() or 8,
        )
    )
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MX_REFIT_DELTA_WORKERS", None)
        else:
            os.environ["MX_REFIT_DELTA_WORKERS"] = previous


class RefitPrefetcher:
    def __init__(
        self,
        settings: PrefetchSettings,
        *,
        service: Any,
        checkpoint: _LocalCheckpoint,
        prepare_registered: Callable[[_ObjectStorageVersion, str], PreparedCheckpoint]
        | None = None,
    ) -> None:
        self._settings = settings
        self._service = service
        self._checkpoint = checkpoint
        self._store = checkpoint.store
        self._prepare_registered = prepare_registered
        self._owner: TextIO | None = None
        self._next_poll = 0.0
        self._registry_listing = True
        self._failed: set[tuple[str, str]] = set()
        self._s3 = checkpoint.reader
        self._writer_versions: dict[str, str] = {}
        self._resolution_retry: dict[tuple[str, str], tuple[float, float]] = {}
        self._list_retry_at = 0.0
        self._list_delay = POLL_SECONDS

    @classmethod
    def for_client(cls, client: ModelExpressGeneratorClient) -> RefitPrefetcher | None:
        from .methods.canonical_delta import CanonicalDeltaUpdateMethod

        canonical = next(
            (
                method
                for method in client._require_runtime().methods
                if isinstance(method, CanonicalDeltaUpdateMethod)
            ),
            None,
        )
        if canonical is None:
            return None

        def prepare_registered(
            version: _ObjectStorageVersion, head: str
        ) -> PreparedCheckpoint:
            from ..version import WeightVersionRef

            client._serving_version_id = canonical._checkpoint.store.active_version()
            with canonical.background_preparation(head):
                handle = client._stage_weight(WeightVersionRef(version.version_id))
                try:
                    if handle._update is None:
                        raise PreparationSuperseded("target is already serving")
                    return handle._update.prepared.checkpoint
                finally:
                    handle.release()

        return cls(
            PrefetchSettings(
                model_name=client.model_name,
                max_replay_chain_length=client._max_replay_chain_length,
                rpc_timeout_seconds=client._rpc_timeout_seconds,
                storage_type=canonical._config.storage_type.value,
                streamed=canonical._stream_full_checkpoints,
                s3_watch=rl_envs.MX_REFIT_PREFETCH_S3_WATCH,
            ),
            service=client._service,
            checkpoint=canonical._checkpoint,
            prepare_registered=prepare_registered,
        )

    def close(self) -> None:
        if self._owner is not None:
            self._owner.close()
            self._owner = None

    def poll(self) -> None:
        if time.monotonic() < self._next_poll:
            return
        self._next_poll = time.monotonic() + POLL_SECONDS
        if self._owner is None:
            owner = (self._store.cache / LOCK_FILE).open("a+")
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                owner.close()
                return
            self._owner = owner
        try:
            with background_workers():
                self.poll_once()
        except Exception:
            logger.exception("Background version polling failed")

    def poll_once(self) -> str | None:
        """Prepare one reachable version, retaining the currently served head."""
        state = self._store.state()
        if state is not None and state.status is CheckpointState.UPDATING:
            self._checkpoint.recover_incomplete_preparation()
            state = self._store.state()
        if state is None:
            return None
        spec = self._store.speculation()
        head = (
            spec["base_version"]
            if spec is not None and spec.get("version") == state.version
            else state.version
        )
        if self._store.active_version() != head:
            return None
        version = self._next_registered(head)
        speculative = False
        if (
            version is None
            and self._settings.s3_watch
            and self._settings.storage_type == "S3"
        ):
            version = self._next_speculative(state)
            speculative = version is not None
        if version is None:
            return None
        self._write_status(phase="preparing", version=version.version_id)
        try:
            if not speculative and self._prepare_registered is not None:
                prepared = self._prepare_registered(version, head)
            else:
                prepared = self._checkpoint.prepare_chain(
                    (version,),
                    expected_head=head,
                    speculative_uri=version.uri if speculative else None,
                )
        except PreparationSuperseded:
            return None
        except Exception as error:
            self._failed.add((version.version_id, head))
            self._write_status(
                phase="failed", version=version.version_id, error=str(error)
            )
            logger.warning(
                "Background preparation failed; leaving the update to the engine",
                exc_info=True,
            )
            self._checkpoint.recover_incomplete_preparation()
            return None
        self._write_status(
            phase="prepared", version=prepared.target_version, metrics=prepared.metrics
        )
        return prepared.target_version

    def _next_registered(self, head: str) -> _ObjectStorageVersion | None:
        """Return the registered version after ``head`` toward the newest one.

        Only the next version is prepared, so the cache head never passes a
        version the engine may still be asked to install.
        """
        target = self._newest_registered() if self._registry_listing else None
        if target is None or target.version_id == head:
            return None
        key = (target.version_id, head)
        now = time.monotonic()
        retry_at, delay = self._resolution_retry.get(key, (0.0, POLL_SECONDS))
        if now < retry_at:
            return None
        try:
            chain = resolve_replay_chain(
                target_version_id=target.version_id,
                fetch_ready_version=self._fetch_ready_version,
                max_chain_length=self._settings.max_replay_chain_length,
                stop_before_version_id=head,
            )
        except RuntimeError as error:
            self._resolution_retry = {key: (now + delay, min(60.0, delay * 2))}
            logger.info("prefetch cannot resolve %s: %s", target.version_id, error)
            return None
        self._resolution_retry.clear()
        if any(
            version.object_storage.storage_type
            is not ObjectStorageType(self._settings.storage_type)
            for version in chain
        ):
            return None
        version = chain[0]
        if (version.version_id, head) in self._failed:
            return None
        if (
            self._settings.streamed
            and version.payload_format is WeightPayloadFormat.FULL_HF_CHECKPOINT
        ):
            # Preparing it here would hold the cache the stream's writer waits on.
            return None
        return _ObjectStorageVersion(
            version_id=version.version_id,
            base_version_id=version.base_version_id,
            payload_format=version.payload_format,
            uri=version.object_storage.uri,
        )

    def _newest_registered(self) -> WeightVersion | None:
        if time.monotonic() < self._list_retry_at:
            return None
        try:
            response = self._service.ListWeightVersions(
                refit_pb2.ListWeightVersionsRequest(
                    model_name=self._settings.model_name
                ),
                timeout=self._settings.rpc_timeout_seconds,
            )
        except grpc.RpcError as error:
            if error.code() is grpc.StatusCode.UNIMPLEMENTED:
                self._registry_listing = False
                logger.info(
                    "MX server cannot list versions; background preparation is unavailable"
                )
            else:
                self._list_retry_at = time.monotonic() + self._list_delay
                self._list_delay = min(60.0, self._list_delay * 2)
                logger.warning("prefetch could not list versions: %s", error)
            return None
        self._list_retry_at = 0.0
        self._list_delay = POLL_SECONDS
        candidates = []
        for message in response.versions:
            try:
                version = _weight_version(message)
            except RuntimeError:
                continue
            if (
                version.state is WeightVersionState.READY
                and version.payload_format in _REPLAY_FORMATS
                and version.object_storage is not None
                and version.object_storage.storage_type
                is ObjectStorageType(self._settings.storage_type)
            ):
                candidates.append(version)
        return max(
            candidates, key=lambda version: version.created_at_unix_ms, default=None
        )

    def _fetch_ready_version(self, version_id: str) -> WeightVersion:
        try:
            response = self._service.GetWeightVersion(
                refit_pb2.GetWeightVersionRequest(uid=version_id),
                timeout=self._settings.rpc_timeout_seconds,
            )
        except grpc.RpcError as error:
            raise RuntimeError(f"failed to resolve revision {version_id!r}") from error
        if not response.HasField("version"):
            raise RuntimeError(f"MX returned no revision {version_id!r}")
        version = _weight_version(response.version)
        if (
            version.state is not WeightVersionState.READY
            or version.model_name != self._settings.model_name
        ):
            raise RuntimeError(f"revision {version_id!r} is not a READY model version")
        return version

    def _next_speculative(self, head: CheckpointRecord) -> _ObjectStorageVersion | None:
        """Return the unregistered delta whose snapshot names the head as base.

        None when its preparation from the head already failed.
        """
        speculation = self._store.speculation()
        if speculation is not None and speculation.get("version") == head.version:
            return None
        head_uri = (head.source or {}).get("uri") or self._registered_uri(head.version)
        if head_uri is None:
            return None
        # Snapshots live at <prefix>/<identity>/<index>, and identities sort in
        # publish order. A delta names its base by the writer's version string.
        snapshot = posixpath.dirname(head_uri)
        identity = posixpath.basename(snapshot)
        prefix = posixpath.dirname(snapshot)
        index_name = posixpath.basename(head_uri)
        if not identity or not prefix.startswith("s3://"):
            return None
        head_version = self._writer_version(head_uri, default=identity)
        for child in self._s3.list_children(prefix, start_after=identity):
            if child <= identity or child.startswith(("_", ".")):
                continue
            uri = f"{prefix}/{child}/{index_name}"
            if not self._is_complete_child_delta(uri, base_version=head_version):
                continue
            version_id = speculative_version_id(uri)
            if (version_id, head.version) in self._failed:
                return None
            return _ObjectStorageVersion(
                version_id=version_id,
                base_version_id=head.version,
                payload_format=WeightPayloadFormat.XOR_DELTA,
                uri=uri,
            )
        return None

    def _registered_uri(self, version_id: str) -> str | None:
        try:
            storage = self._fetch_ready_version(version_id).object_storage
        except RuntimeError:
            return None
        return None if storage is None else storage.uri

    def _writer_version(self, uri: str, *, default: str) -> str:
        """Return the version the snapshot's writer recorded, or ``default``."""
        if uri not in self._writer_versions:
            try:
                index = json.loads(self._s3.get(uri))
                version = index["metadata"]["version"]
            except Exception:
                version = None
            self._writer_versions[uri] = (
                version if isinstance(version, str) else default
            )
        return self._writer_versions[uri]

    def _is_complete_child_delta(self, uri: str, *, base_version: str) -> bool:
        try:
            index = json.loads(self._s3.get(uri))
        except Exception:
            return False
        metadata = index.get("metadata") if isinstance(index, dict) else None
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if (
            not isinstance(metadata, dict)
            or not isinstance(weight_map, dict)
            or metadata.get("delta_encoding") != "xor"
            or metadata.get("base_version") != base_version
        ):
            return False
        # Writers upload the index last, but every shard must exist before reading.
        snapshot = posixpath.dirname(uri)
        return all(
            self._s3.exists(f"{snapshot}/{shard}") for shard in set(weight_map.values())
        )

    def _write_status(self, **fields: Any) -> None:
        status = dict(
            fields, pid=os.getpid(), updated_at_unix_ms=int(time.time() * 1000)
        )
        path = self._store.cache / "prefetch.json"
        temporary = path.with_name(f"{path.name}.tmp")
        try:
            temporary.write_text(json.dumps(status, sort_keys=True))
            temporary.replace(path)
        except OSError:
            logger.debug("prefetch status is unavailable", exc_info=True)
