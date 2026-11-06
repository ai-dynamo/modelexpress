# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare registered object-storage versions outside the serving pause."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import grpc
from modelexpress import auth
from modelexpress_rl import refit_pb2, refit_pb2_grpc
from modelexpress_rl.control import WeightVersion, WeightVersionState, _weight_version
from modelexpress_rl.object_storage import ObjectStorageType
from modelexpress_rl.s3 import S3Client
from modelexpress_rl.azure import AzureBlobReader
from modelexpress_rl.train import WeightPayloadFormat
from .checkpoint_store import CheckpointState, LocalCheckpointStore
from .receiver import (
    ObjectStorageGeneratorConfig,
    _LocalCheckpoint,
    _ObjectStorageVersion,
    PreparationSuperseded,
)
from .version_chain import resolve_replay_chain

logger = logging.getLogger(__name__)
POLL_SECONDS = 2.0
LOCK_FILE = ".prefetch.lock"
_REPLAY_FORMATS = frozenset(
    {WeightPayloadFormat.XOR_DELTA, WeightPayloadFormat.FULL_HF_CHECKPOINT}
)


@dataclass(frozen=True)
class PrefetchSettings:
    model_name: str
    cache_model_name: str
    server_url: str
    refit_checkpoint_dir: str
    initial_base_version_id: str
    seed_checkpoint_path: str | None
    refit_checkpoint_max_size_gb: int | None
    endpoint_url: str | None
    region_name: str | None
    max_replay_chain_length: int
    rpc_timeout_seconds: float
    parent_pid: int
    storage_type: str = "S3"
    streamed: bool = False

    def object_storage(self) -> ObjectStorageGeneratorConfig:
        return ObjectStorageGeneratorConfig(
            storage_type=ObjectStorageType(self.storage_type),
            initial_base_version_id=self.initial_base_version_id,
            seed_checkpoint_path=self.seed_checkpoint_path,
            refit_checkpoint_dir=self.refit_checkpoint_dir,
            refit_checkpoint_max_size_gb=self.refit_checkpoint_max_size_gb,
            endpoint_url=self.endpoint_url,
            region_name=self.region_name,
        )

    def store(self) -> LocalCheckpointStore:
        return LocalCheckpointStore(
            root=self.refit_checkpoint_dir, model_name=self.cache_model_name
        )


def launch_prefetcher(settings: PrefetchSettings) -> subprocess.Popen:
    environment = dict(
        os.environ, MX_REFIT_DELTA_WORKERS=str(min(8, os.cpu_count() or 1))
    )
    return subprocess.Popen(
        [sys.executable, "-m", __name__, "--settings", json.dumps(asdict(settings))],
        env=environment,
    )


class RefitPrefetcher:
    def __init__(
        self, settings: PrefetchSettings, *, service: Any, checkpoint: _LocalCheckpoint
    ) -> None:
        self._settings = settings
        self._service = service
        self._checkpoint = checkpoint
        self._store = checkpoint.store
        self._channel = None
        self._registry_listing = True
        self._failed: set[tuple[str, str]] = set()

    @classmethod
    def connect(cls, settings: PrefetchSettings) -> RefitPrefetcher:
        channel = auth.with_auth(grpc.insecure_channel(settings.server_url))
        reader = (
            S3Client(
                endpoint_url=settings.endpoint_url, region_name=settings.region_name
            )
            if settings.storage_type == "S3"
            else AzureBlobReader()
        )
        try:
            checkpoint = _LocalCheckpoint(
                model_name=settings.cache_model_name,
                config=settings.object_storage(),
                reader=reader,
            )
            result = cls(
                settings,
                service=refit_pb2_grpc.RefitServiceStub(channel),
                checkpoint=checkpoint,
            )
        except Exception:
            channel.close()
            reader.close()
            raise
        result._channel = channel
        return result

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
        self._checkpoint.reader.close()

    def poll_once(self) -> str | None:
        state = self._store.state()
        if (
            state is None
            or state.status is not CheckpointState.READY
            or self._store.active_version() != state.version
        ):
            return None
        version = self._next_registered(state.version)
        if version is None:
            return None
        self._write_status(phase="preparing", version=version.version_id)
        try:
            prepared = self._checkpoint.prepare_chain(
                (version,), expected_head=state.version
            )
        except PreparationSuperseded:
            return None
        except Exception as error:
            self._failed.add((version.version_id, state.version))
            self._write_status(
                phase="failed", version=version.version_id, error=str(error)
            )
            logger.warning(
                "Background preparation failed; leaving the update to the engine",
                exc_info=True,
            )
            try:
                self._checkpoint.recover_incomplete_preparation()
            except Exception:
                logger.exception("Background preparation recovery failed")
            return None
        self._write_status(
            phase="prepared", version=prepared.target_version, metrics=prepared.metrics
        )
        return prepared.target_version

    def run(self) -> None:
        while True:
            try:
                os.kill(self._settings.parent_pid, 0)
            except ProcessLookupError:
                return
            try:
                self.poll_once()
            except Exception:
                logger.exception("Background version polling failed")
            time.sleep(POLL_SECONDS)

    def _next_registered(self, head: str) -> _ObjectStorageVersion | None:
        """Return the registered version after ``head`` toward the newest one.

        Only the next version is prepared, so the cache head never passes a
        version the engine may still be asked to install.
        """
        target = self._newest_registered() if self._registry_listing else None
        if target is None or target.version_id == head:
            return None
        try:
            chain = resolve_replay_chain(
                target_version_id=target.version_id,
                fetch_ready_version=self._fetch_ready_version,
                max_chain_length=self._settings.max_replay_chain_length,
                stop_before_version_id=head,
            )
        except RuntimeError as error:
            logger.info("prefetch cannot resolve %s: %s", target.version_id, error)
            return None
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
                logger.warning("prefetch could not list versions: %s", error)
            return None
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--settings", required=True, help="JSON PrefetchSettings")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = PrefetchSettings(**json.loads(args.settings))
    cache_dir = settings.store().cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / LOCK_FILE).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.info("another prefetcher already serves %s", cache_dir)
            return 0
        with contextlib.suppress(OSError):
            os.nice(10)
        prefetcher = RefitPrefetcher.connect(settings)
        try:
            prefetcher.run()
        finally:
            prefetcher.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
