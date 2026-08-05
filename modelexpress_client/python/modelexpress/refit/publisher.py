# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal launch attestation and exact-base canonical S3 publisher."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .api import (
    PublicationMode,
    PublisherConfig,
    PublisherStatus,
    PublishResult,
    S3Config,
    VersionId,
)
from .catalog import GrpcRevisionCatalog, RevisionCatalog
from .manifest import RevisionManifest, RevisionRecord, RevisionState
from .s3 import S3Uploader
from .source.canonical import CanonicalDeltaEncoder, RetainedBaseStore
from .source.megatron_bridge import (
    MegatronBridgeHfBucketConfig,
    for_each_megatron_hf_bucket,
)


class PublisherError(RuntimeError):
    """A V0 publication cannot complete without violating its contract."""


class PublisherStateError(PublisherError):
    """Publisher lifecycle does not permit the requested operation."""


@dataclass(frozen=True)
class _RemoteFailure:
    detail: str


class Publisher:
    """Publish launch metadata and later exact-base canonical deltas over S3."""

    def __init__(
        self,
        *,
        model: object,
        launch_checkpoint: str | Path,
        scratch_directory: str | Path,
        megatron_config: MegatronBridgeHfBucketConfig,
        catalog: RevisionCatalog | None = None,
        s3_client: Any = None,
        poll_interval_seconds: float = 0.25,
        request_timeout_seconds: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(megatron_config, MegatronBridgeHfBucketConfig):
            raise TypeError("megatron_config must be MegatronBridgeHfBucketConfig")
        for field, value in (
            ("poll_interval_seconds", poll_interval_seconds),
            ("request_timeout_seconds", request_timeout_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{field} must be finite and positive")
        if not callable(sleep):
            raise TypeError("sleep must be callable")
        self._model = model
        self._launch_checkpoint = Path(launch_checkpoint).resolve()
        self._base_store = RetainedBaseStore(scratch_directory)
        self._megatron_config = megatron_config
        self._catalog = catalog
        self._s3_client = s3_client
        self._uploader: S3Uploader | None = None
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._request_timeout_seconds = float(request_timeout_seconds)
        self._sleep = sleep
        self._config: PublisherConfig | None = None
        self._current_version: str | None = None
        self._current_state: RevisionState | None = None
        self._pending_manifest: RevisionManifest | None = None
        self._pending_candidate: object | None = None
        self._initialized = False
        self._closed = False
        self._active = False
        self._lock = threading.Lock()

    def initialize(self, config: PublisherConfig) -> None:
        """Attest launch version 0 in place and initialize rank-zero clients."""
        local_error: Exception | None = None
        claimed_active = False
        rank = 0
        with self._lock:
            try:
                if self._closed:
                    raise PublisherStateError("publisher is deregistered")
                if self._initialized:
                    raise PublisherStateError("publisher is already initialized")
                if self._active:
                    raise PublisherStateError("a publisher operation is already active")
                self._validate_config(config)
                rank = self._rank()
                self._active = True
                claimed_active = True
            except Exception as exc:
                local_error = exc
        try:
            self._collective_agree(
                "publisher configuration",
                local_error,
                self._config_identity(config),
            )
            launch = None
            local_error = None
            try:
                launch = self._base_store.seed_launch(
                    self._launch_checkpoint,
                    maximum_tensor_bytes=self._megatron_config.bucket_bytes,
                )
                self._megatron_config = self._megatron_config.with_schema(launch.schema)
            except Exception as exc:
                local_error = exc
            self._collective_agree(
                "launch attestation",
                local_error,
                self._launch_identity(launch),
            )

            initialized: bool | _RemoteFailure | None = None
            if rank == 0:
                try:
                    if self._catalog is None:
                        self._catalog = GrpcRevisionCatalog(config.catalog_endpoint)
                    self._uploader = S3Uploader(
                        config.s3,
                        client=self._s3_client,
                        request_timeout_seconds=self._request_timeout_seconds,
                    )
                    initialized = True
                except Exception as exc:
                    initialized = _RemoteFailure(str(exc) or type(exc).__name__)
            initialized = self._broadcast(initialized)
            if isinstance(initialized, _RemoteFailure):
                raise PublisherError(initialized.detail)
            if initialized is not True:
                raise PublisherError("rank-zero publisher initialization failed")
            with self._lock:
                self._config = config
                self._current_version = "0"
                self._initialized = True
                self._active = False
                claimed_active = False
        except Exception:
            self._close_resources()
            raise
        finally:
            if claimed_active:
                with self._lock:
                    self._active = False

    def publish_version(
        self,
        version: VersionId,
        *,
        base_version: VersionId | None = None,
    ) -> PublishResult:
        """Publish on rank zero; every rank participates in Megatron capture."""
        local_error: Exception | None = None
        claimed_active = False
        request_identity = None
        with self._lock:
            try:
                if not self._initialized:
                    raise PublisherStateError("publisher is not initialized")
                if self._closed:
                    raise PublisherStateError("publisher is deregistered")
                if self._active:
                    raise PublisherStateError("a publisher operation is already active")
                if not isinstance(version, str) or not version:
                    raise PublisherError("target version must be non-empty")
                if base_version is not None and (
                    not isinstance(base_version, str) or not base_version
                ):
                    raise PublisherError("base_version must be non-empty when provided")
                pending = self._pending_manifest
                request_identity = (
                    version,
                    base_version,
                    self._current_version,
                    None if pending is None else pending.target_version,
                    None if pending is None else pending.base_version,
                )
                self._active = True
                claimed_active = True
            except Exception as exc:
                local_error = exc
        try:
            self._collective_agree(
                "publication request",
                local_error,
                request_identity,
            )
            pending = self._pending_manifest
            if pending is not None:
                same_request = (
                    version == pending.target_version
                    and base_version == pending.base_version
                )
                chains_from_pending = base_version == pending.target_version
                observed = self._complete_pending_publication(
                    pending,
                    wait=(
                        self._require_config().publication_mode is PublicationMode.BLOCK
                    ),
                )
                if observed.state is RevisionState.READY:
                    if same_request:
                        return observed
                    if chains_from_pending:
                        raise PublisherError("pending target is not committed")
                    raise PublisherError(
                        "a publication is pending external commit for "
                        f"{pending.target_version!r}"
                    )
                if same_request:
                    return observed
            if version == "0":
                if base_version is not None:
                    raise PublisherError("launch version 0 has no base_version")
                result, manifest = self._publish_launch()
            else:
                result, manifest = self._publish_target(version, base_version)
            if result.state is RevisionState.READY:
                with self._lock:
                    self._pending_manifest = manifest
                    if version == "0":
                        self._current_state = result.state
                if self._require_config().publication_mode is PublicationMode.BLOCK:
                    return self._complete_pending_publication(manifest, wait=True)
                return result
            with self._lock:
                self._current_version = version
                self._current_state = result.state
            return result
        finally:
            if claimed_active:
                with self._lock:
                    self._active = False

    def status(self) -> PublisherStatus:
        with self._lock:
            config = self._config
            return PublisherStatus(
                model_id=config.model_id if config is not None else "",
                current_version=self._current_version,
                state=self._current_state,
                publication_mode=(
                    config.publication_mode
                    if config is not None
                    else PublicationMode.BLOCK
                ),
            )

    def deregister(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._active:
                raise PublisherStateError(
                    "cannot deregister during an active publication"
                )
            self._closed = True
            self._initialized = False
            candidate = self._pending_candidate
            self._pending_candidate = None
            self._pending_manifest = None
        abort = getattr(candidate, "abort", None)
        if callable(abort):
            abort()
        self._close_resources()

    def _publish_launch(self) -> tuple[PublishResult, RevisionManifest]:
        config = self._require_config()
        launch = self._base_store.current
        manifest = RevisionManifest(
            model_id=config.model_id,
            target_version="0",
            target_digest=launch.target_digest,
            format_digest=launch.format_digest,
        )
        outcome: RevisionRecord | _RemoteFailure | None = None
        if self._rank() == 0:
            try:
                outcome = self._publish_manifest(manifest)
            except Exception as exc:
                outcome = _RemoteFailure(str(exc) or type(exc).__name__)
        record = self._broadcast(outcome)
        if isinstance(record, _RemoteFailure):
            raise PublisherError(record.detail)
        if not isinstance(record, RevisionRecord):
            raise PublisherError("rank-zero launch publication returned no record")
        self._validate_record(record, manifest)
        return PublishResult(config.model_id, "0", record.state), manifest

    def _publish_target(
        self, version: str, base_version: str | None
    ) -> tuple[PublishResult, RevisionManifest]:
        config = self._require_config()
        exact = self._current_version
        if base_version is None:
            raise PublisherError("canonical publication requires base_version")
        if base_version != exact:
            raise PublisherError(
                f"requested base {base_version!r} does not match exact retained base "
                f"{exact!r}"
            )
        if version == base_version:
            raise PublisherError("target version must differ from exact retained base")

        preflight: RevisionRecord | _RemoteFailure | None = None
        if self._rank() == 0:
            try:
                preflight = self._attest_catalog_base(base_version)
            except Exception as exc:
                preflight = _RemoteFailure(str(exc) or type(exc).__name__)
        base_record = self._broadcast(preflight)
        if isinstance(base_record, _RemoteFailure):
            raise PublisherError(base_record.detail)
        if not isinstance(base_record, RevisionRecord):
            raise PublisherError("rank-zero exact-base preflight returned no record")

        encoder = None
        encoder_ready: bool | _RemoteFailure | None = None
        if self._rank() == 0:
            try:
                encoder = CanonicalDeltaEncoder(
                    model_id=config.model_id,
                    target_version=version,
                    base_store=self._base_store,
                    uploader=self._require_uploader(),
                    bucket_bytes=self._megatron_config.bucket_bytes,
                )
                encoder_ready = True
            except Exception as exc:
                encoder_ready = _RemoteFailure(str(exc) or type(exc).__name__)
        encoder_ready = self._broadcast(encoder_ready)
        if isinstance(encoder_ready, _RemoteFailure):
            raise PublisherError(encoder_ready.detail)
        if encoder_ready is not True:
            raise PublisherError("rank-zero canonical encoder initialization failed")
        try:
            for_each_megatron_hf_bucket(
                self._model,
                self._megatron_config,
                (
                    encoder.consume_bucket
                    if encoder is not None
                    else self._unexpected_nonzero_bucket
                ),
            )
        except Exception as exc:
            if encoder is not None:
                encoder.abort()
            raise PublisherError(str(exc) or type(exc).__name__) from exc

        outcome: tuple[RevisionRecord, object, bool] | _RemoteFailure | None = None
        pending_candidate = None
        if self._rank() == 0:
            assert encoder is not None
            try:
                publication = encoder.finish()
                manifest = RevisionManifest(
                    model_id=config.model_id,
                    target_version=version,
                    base_version=base_version,
                    base_digest=base_record.manifest.target_digest,
                    target_digest=publication.snapshot.target_digest,
                    format_digest=publication.snapshot.format_digest,
                    payload=publication.payload.object,
                )
                record = self._publish_manifest(manifest)
                deferred = record.state is RevisionState.READY
                if deferred:
                    pending_candidate = publication.candidate
                else:
                    self._base_store.promote(publication.candidate)
                outcome = (record, manifest, deferred)
            except Exception as exc:
                encoder.abort()
                outcome = _RemoteFailure(str(exc) or type(exc).__name__)
        published = self._broadcast(outcome)
        if isinstance(published, _RemoteFailure):
            raise PublisherError(published.detail)
        if not isinstance(published, tuple) or len(published) != 3:
            raise PublisherError("rank-zero target publication returned no record")
        record, manifest, deferred = published
        if not isinstance(record, RevisionRecord) or not isinstance(
            manifest, RevisionManifest
        ):
            raise PublisherError("rank-zero target publication returned invalid data")
        if not isinstance(deferred, bool):
            raise PublisherError("rank-zero target publication returned invalid data")
        self._validate_record(record, manifest)
        if deferred:
            self._pending_candidate = pending_candidate
        return PublishResult(config.model_id, version, record.state), manifest

    def _complete_pending_publication(
        self,
        manifest: RevisionManifest,
        *,
        wait: bool,
    ) -> PublishResult:
        result = (
            self._wait_for_commit(manifest)
            if wait
            else self._observe_pending_publication(manifest)
        )
        if result.state is RevisionState.READY:
            return result
        outcome: bool | _RemoteFailure | None = None
        if self._rank() == 0:
            try:
                candidate = self._pending_candidate
                if candidate is not None:
                    self._base_store.promote(candidate)
                outcome = True
            except Exception as exc:
                outcome = _RemoteFailure(str(exc) or type(exc).__name__)
        promoted = self._broadcast(outcome)
        if isinstance(promoted, _RemoteFailure):
            raise PublisherError(promoted.detail)
        if promoted is not True:
            raise PublisherError("rank-zero exact-base promotion did not complete")
        with self._lock:
            self._current_version = manifest.target_version
            self._current_state = result.state
            self._pending_candidate = None
            self._pending_manifest = None
        return result

    def _attest_catalog_base(self, version: str) -> RevisionRecord:
        config = self._require_config()
        base = self._base_store.current
        if base.version != version:
            raise PublisherError(
                f"rank-zero storage does not retain exact base {version!r}"
            )
        record = self._require_catalog().get_revision(config.model_id, version)
        manifest = record.manifest
        if (
            manifest.model_id != config.model_id
            or manifest.target_version != version
            or manifest.target_digest != base.target_digest
            or manifest.format_digest != base.format_digest
        ):
            raise PublisherError("catalog exact base differs from local attestation")
        if record.state is not RevisionState.COMMITTED:
            raise PublisherError("catalog exact base is not committed")
        return record

    def _publish_manifest(self, manifest: RevisionManifest) -> RevisionRecord:
        record = self._require_catalog().publish_revision(manifest)
        self._validate_record(record, manifest)
        return record

    @staticmethod
    def _validate_record(record: RevisionRecord, manifest: RevisionManifest) -> None:
        if record.manifest != manifest:
            raise PublisherError("catalog returned a different immutable manifest")
        if record.state not in {RevisionState.READY, RevisionState.COMMITTED}:
            raise PublisherError(f"catalog returned illegal state {record.state!r}")

    def _observe_pending_publication(self, manifest: RevisionManifest) -> PublishResult:
        config = self._require_config()
        outcome: RevisionRecord | _RemoteFailure | None = None
        if self._rank() == 0:
            try:
                outcome = self._require_catalog().get_revision(
                    config.model_id,
                    manifest.target_version,
                )
                self._validate_record(outcome, manifest)
            except Exception as exc:
                outcome = _RemoteFailure(str(exc) or type(exc).__name__)
        record = self._broadcast(outcome)
        if isinstance(record, _RemoteFailure):
            raise PublisherError(record.detail)
        if not isinstance(record, RevisionRecord):
            raise PublisherError("rank-zero exact-get returned no record")
        self._validate_record(record, manifest)
        return PublishResult(config.model_id, manifest.target_version, record.state)

    def _wait_for_commit(self, manifest: RevisionManifest) -> PublishResult:
        config = self._require_config()
        version = manifest.target_version
        outcome: RevisionRecord | _RemoteFailure | None = None
        if self._rank() == 0:
            try:
                while True:
                    record = self._require_catalog().get_revision(
                        config.model_id,
                        version,
                    )
                    self._validate_record(record, manifest)
                    if record.state is RevisionState.COMMITTED:
                        outcome = record
                        break
                    if record.state is not RevisionState.READY:
                        raise PublisherError(
                            f"catalog returned illegal state {record.state!r}"
                        )
                    self._sleep(self._poll_interval_seconds)
            except Exception as exc:
                outcome = _RemoteFailure(str(exc) or type(exc).__name__)
        record = self._broadcast(outcome)
        if isinstance(record, _RemoteFailure):
            raise PublisherError(record.detail)
        if not isinstance(record, RevisionRecord):
            raise PublisherError("rank-zero BLOCK poll returned no record")
        self._validate_record(record, manifest)
        return PublishResult(config.model_id, version, record.state)

    def _broadcast(self, value: object | None) -> object:
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return value
        values = [value]
        torch.distributed.broadcast_object_list(
            values,
            src=0,
            group=self._megatron_config.metadata_group,
        )
        return values[0]

    def _collective_agree(
        self,
        label: str,
        local_error: Exception | None,
        identity: object,
    ) -> None:
        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        if not distributed:
            if local_error is not None:
                raise local_error
            return
        detail = (
            None
            if local_error is None
            else str(local_error) or type(local_error).__name__
        )
        contribution = (detail, identity)
        gathered = [None] * torch.distributed.get_world_size(
            self._megatron_config.metadata_group
        )
        try:
            torch.distributed.all_gather_object(
                gathered,
                contribution,
                group=self._megatron_config.metadata_group,
            )
        except Exception as exc:
            raise PublisherError(f"{label} agreement failed: {exc}") from exc
        failures = [
            f"rank {rank}: {item[0]}"
            for rank, item in enumerate(gathered)
            if isinstance(item, tuple) and len(item) == 2 and item[0]
        ]
        if failures:
            raise PublisherError("; ".join(failures))
        if any(item != contribution for item in gathered):
            raise PublisherError(f"{label} differs across trainer ranks")

    def _rank(self) -> int:
        callback = self._megatron_config.rank
        if callback is not None:
            rank = callback()
        elif torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank(self._megatron_config.metadata_group)
        else:
            rank = 0
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
            raise PublisherError("publisher rank is invalid")
        return rank

    @staticmethod
    def _unexpected_nonzero_bucket(_bucket: object) -> None:
        raise PublisherError("Megatron source emitted a bucket on a nonzero rank")

    @staticmethod
    def _validate_config(config: PublisherConfig) -> None:
        if not isinstance(config, PublisherConfig):
            raise TypeError("config must be PublisherConfig")
        if not isinstance(config.model_id, str) or not config.model_id:
            raise PublisherError("model_id must be non-empty")
        if not isinstance(config.catalog_endpoint, str) or not config.catalog_endpoint:
            raise PublisherError("catalog_endpoint must be non-empty")
        if not isinstance(config.s3, S3Config):
            raise PublisherError("s3 must be S3Config")
        if not isinstance(config.s3.bucket, str) or not config.s3.bucket.strip():
            raise PublisherError("S3 bucket must be non-empty")
        if not isinstance(config.publication_mode, PublicationMode):
            raise PublisherError("publication_mode must be PublicationMode")

    def _config_identity(self, config: object) -> object:
        if not isinstance(config, PublisherConfig):
            return None
        s3 = config.s3
        s3_identity = (
            None
            if not isinstance(s3, S3Config)
            else (s3.bucket, s3.prefix, s3.endpoint_url, s3.region_name)
        )
        return (
            config.model_id,
            config.catalog_endpoint,
            s3_identity,
            (
                config.publication_mode.value
                if isinstance(config.publication_mode, PublicationMode)
                else None
            ),
            self._megatron_config.bucket_bytes,
            self._megatron_config.hf_model_path,
            self._megatron_config.vocab_size,
        )

    @staticmethod
    def _launch_identity(launch: object) -> object:
        if launch is None:
            return None
        return (
            launch.version,
            launch.format_digest,
            launch.target_digest,
            tuple(
                (
                    tensor.name,
                    tensor.shape,
                    str(tensor.dtype),
                    tensor.byte_size,
                    tensor.content_digest,
                )
                for tensor in launch.tensors
            ),
        )

    def _require_config(self) -> PublisherConfig:
        if self._config is None:
            raise PublisherStateError("publisher is not initialized")
        return self._config

    def _require_catalog(self) -> RevisionCatalog:
        if self._catalog is None:
            raise PublisherStateError("publisher catalog is not initialized")
        return self._catalog

    def _require_uploader(self) -> S3Uploader:
        if self._uploader is None:
            raise PublisherStateError("publisher S3 uploader is not initialized")
        return self._uploader

    def _close_resources(self) -> None:
        if self._uploader is not None:
            self._uploader.close()
            self._uploader = None
        close = getattr(self._catalog, "close", None)
        if callable(close):
            close()
