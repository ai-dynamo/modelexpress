# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete MX-owned Publisher for verified exact-base CANONICAL revisions."""

from __future__ import annotations

import math
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import grpc
import torch

from .api import (
    PublishResult,
    PublisherConfig,
    PublisherStatus,
    TransportKind,
    VersionId,
)
from .catalog import GrpcRevisionCatalog, RevisionCatalog
from .codec import CodecError, compress_payload, encode_delta
from .manifest import (
    ChangeState,
    DeltaLocation,
    DeltaTransferMethod,
    PublicationMode,
    RankDelta,
    RevisionLifecycleState,
    RevisionManifest,
    RevisionRank,
)
from .source.base import CanonicalBucketConsumer
from .source.canonical import (
    CanonicalCapture,
    CanonicalDeltaEncoder,
    CanonicalFormatIdentity,
    CanonicalSnapshot,
    FilesystemCanonicalBaseStore,
)
from .transport import (
    CanonicalTransport,
    CanonicalTransportIdentity,
    StoredObject,
    canonical_object_key,
)
from .transport.base import validate_checksum, validate_relative_key
from .transport.filesystem import FilesystemCanonicalTransport
from .transport.s3 import S3CanonicalTransport


class PublisherError(RuntimeError):
    """A configured CANONICAL publication cannot be completed safely."""


class PublisherStateError(PublisherError):
    """Publisher lifecycle does not permit the requested operation."""


class PublicationCancelled(PublisherError):
    """A BLOCK wait was cancelled after immutable publication reached READY."""


_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class _RemoteFailure:
    error_type: str
    detail: str


@dataclass(frozen=True)
class _InitializationReady:
    config: PublisherConfig
    initial_base_version: str
    format_identity: CanonicalFormatIdentity
    source_format_digest: str
    transport_identity: CanonicalTransportIdentity
    maximum_bucket_bytes: int
    maximum_encoded_ratio: float


@dataclass(frozen=True)
class _PreflightReady:
    intent: _PublicationIntent


@dataclass(frozen=True)
class _PublicationRequest:
    model_id: str
    target_version: str
    base_version: str
    format_identity: CanonicalFormatIdentity
    source_format_digest: str
    transfer_method: DeltaTransferMethod
    delta_method: str
    compression_algorithm: str
    publication_mode: PublicationMode
    catalog_endpoint: str
    transport_identity: CanonicalTransportIdentity
    maximum_bucket_bytes: int
    maximum_encoded_ratio: float


@dataclass(frozen=True)
class _PublicationIntent:
    request: _PublicationRequest
    format_digest: str
    base_digest: str
    producer_id: str

    @property
    def model_id(self) -> str:
        return self.request.model_id

    @property
    def target_version(self) -> str:
        return self.request.target_version

    @property
    def base_version(self) -> str:
        return self.request.base_version

    @property
    def format_identity(self) -> CanonicalFormatIdentity:
        return self.request.format_identity

    @property
    def transfer_method(self) -> DeltaTransferMethod:
        return self.request.transfer_method

    @property
    def delta_method(self) -> str:
        return self.request.delta_method

    @property
    def compression_algorithm(self) -> str:
        return self.request.compression_algorithm

    @property
    def publication_mode(self) -> PublicationMode:
        return self.request.publication_mode

    @property
    def catalog_endpoint(self) -> str:
        return self.request.catalog_endpoint

    @property
    def transport_identity(self) -> CanonicalTransportIdentity:
        return self.request.transport_identity

    @property
    def maximum_bucket_bytes(self) -> int:
        return self.request.maximum_bucket_bytes

    @property
    def maximum_encoded_ratio(self) -> float:
        return self.request.maximum_encoded_ratio


@dataclass(frozen=True)
class _CaptureComplete:
    target_version: str


@dataclass(frozen=True)
class _ReadyPublication:
    manifest: RevisionManifest
    result: PublishResult


@dataclass
class _RankZeroPrepared:
    config: PublisherConfig
    catalog: RevisionCatalog
    transport: CanonicalTransport
    base: CanonicalSnapshot
    encoder: CanonicalDeltaEncoder
    children: list[StoredObject]


class _TorchCoordinator:
    @property
    def rank(self) -> int:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

    def broadcast(self, value: object | None) -> object:
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            if self.rank != 0:
                raise PublisherError(
                    "nonzero publisher rank requires distributed coordination"
                )
            return value
        values = [value]
        try:
            torch.distributed.broadcast_object_list(values, src=0)
        except Exception as exc:
            raise PublisherError(
                f"rank-0 publication outcome broadcast failed: {exc}"
            ) from exc
        return values[0]

    def agree(self, value: object) -> object:
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return value
        gathered: list[object | None] = [None] * torch.distributed.get_world_size()
        try:
            torch.distributed.all_gather_object(gathered, value)
        except Exception as exc:
            raise PublisherError(f"publisher rank agreement failed: {exc}") from exc
        failures = [
            f"rank {rank}: {item.error_type}: {item.detail}"
            for rank, item in enumerate(gathered)
            if isinstance(item, _RemoteFailure)
        ]
        if failures:
            raise PublisherError(
                "publisher rank agreement failed: " + "; ".join(failures)
            )
        if any(item != gathered[0] for item in gathered[1:]):
            if isinstance(value, _InitializationReady):
                subject = "publisher initialization"
            elif isinstance(value, _PublicationRequest):
                subject = "publication request"
            else:
                subject = "capture outcome"
            raise PublisherError(f"{subject} differs across trainer ranks")
        return gathered[0]


class Publisher:
    """Publish complete HF-canonical exact-base deltas and own their resources."""

    def __init__(
        self,
        *,
        capture: Callable[[VersionId, CanonicalBucketConsumer], None],
        base_store: FilesystemCanonicalBaseStore,
        initial_base_version: VersionId,
        producer_id: str,
        format_identity: CanonicalFormatIdentity,
        catalog: RevisionCatalog | None = None,
        transport: CanonicalTransport | None = None,
        coordinator: Any = None,
        source_close: Callable[[], None] | None = None,
        allow_filesystem_transport: bool = False,
        poll_interval_seconds: float = 0.25,
        rpc_timeout_seconds: float = 5.0,
        maximum_encoded_ratio: float = 1.0,
        maximum_bucket_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        if not callable(capture):
            raise TypeError("capture must be callable")
        if not isinstance(initial_base_version, str) or not initial_base_version:
            raise ValueError("initial_base_version must be non-empty")
        if not isinstance(producer_id, str) or not producer_id:
            raise ValueError("producer_id must be non-empty")
        if not isinstance(format_identity, CanonicalFormatIdentity):
            raise TypeError("format_identity must be CanonicalFormatIdentity")
        if not isinstance(capture, CanonicalCapture):
            raise TypeError("capture must be a CanonicalCapture bound to its HF schema")
        capture_identity = capture.format_identity
        if capture_identity != format_identity:
            raise ValueError("capture format identity differs from Publisher identity")
        capture_format_digest = capture.format_digest
        if not math.isfinite(poll_interval_seconds) or poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be finite and positive")
        if not math.isfinite(rpc_timeout_seconds) or rpc_timeout_seconds <= 0:
            raise ValueError("rpc_timeout_seconds must be finite and positive")
        if not math.isfinite(maximum_encoded_ratio) or maximum_encoded_ratio <= 0:
            raise ValueError("maximum_encoded_ratio must be finite and positive")
        if (
            not isinstance(maximum_bucket_bytes, int)
            or isinstance(maximum_bucket_bytes, bool)
            or maximum_bucket_bytes <= 0
        ):
            raise ValueError("maximum_bucket_bytes must be a positive integer")
        self._capture = capture
        self._base_store = base_store
        self._initial_base_version = initial_base_version
        self._producer_id = producer_id
        self._format_identity = format_identity
        self._capture_format_digest = capture_format_digest
        self._catalog = catalog
        self._transport = transport
        self._coordinator = coordinator or _TorchCoordinator()
        self._source_close = source_close
        self._allow_filesystem_transport = allow_filesystem_transport
        self._poll_interval_seconds = poll_interval_seconds
        self._rpc_timeout_seconds = rpc_timeout_seconds
        self._maximum_encoded_ratio = maximum_encoded_ratio
        self._maximum_bucket_bytes = maximum_bucket_bytes

        self._condition = threading.Condition(threading.RLock())
        self._cancel = threading.Event()
        self._initialized = False
        self._deregistering = False
        self._deregistered = False
        self._resources_closed = False
        self._deregister_error: str | None = None
        self._active_operations = 0
        self._operation_threads: set[int] = set()
        self._observer_thread: threading.Thread | None = None
        self._observer_target: _ReadyPublication | None = None
        self._config: PublisherConfig | None = None
        self._current_version: VersionId | None = None
        self._current_state: RevisionLifecycleState | None = None
        self._eligible_base_version: VersionId | None = None

    def initialize(self, config: PublisherConfig) -> None:
        """Validate CANONICAL-only policy and acquire catalog/transport resources."""
        with self._condition:
            if self._deregistered or self._deregistering:
                raise PublisherStateError("publisher is deregistered")
            if self._initialized:
                raise PublisherStateError("publisher is already initialized")

            local_error: Exception | None = None
            try:
                self._validate_config(config)
                if self._rank() == 0:
                    base = self._base_store.open_snapshot(self._initial_base_version)
                    if not base.tensors:
                        raise PublisherError(
                            "initial exact base has no canonical tensor coverage"
                        )
                    if base.format_identity != self._format_identity:
                        raise PublisherError(
                            "initial exact base format identity differs from Publisher source"
                        )
                    if base.format_digest != self._capture_format_digest:
                        raise PublisherError(
                            "initial exact base format digest differs from Publisher capture"
                        )

                catalog = self._catalog
                if catalog is None:
                    catalog = GrpcRevisionCatalog(config.catalog_endpoint)
                    self._catalog = catalog
                transport = self._transport
                if transport is None:
                    transport = self._build_transport(config)
                    self._transport = transport
                self._validate_catalog_capabilities(catalog)
                self._validate_transport_identity(config, transport)

                local_contribution: object = _InitializationReady(
                    config=config,
                    initial_base_version=self._initial_base_version,
                    format_identity=self._format_identity,
                    source_format_digest=self._capture_format_digest,
                    transport_identity=transport.identity,
                    maximum_bucket_bytes=self._maximum_bucket_bytes,
                    maximum_encoded_ratio=self._maximum_encoded_ratio,
                )
            except Exception as exc:
                local_error = exc
                local_contribution = _RemoteFailure(type(exc).__name__, str(exc))

            try:
                agreed = self._agree(local_contribution)
                if isinstance(agreed, _RemoteFailure):
                    if local_error is not None:
                        raise local_error
                    self._raise_remote_failure(agreed)
                if local_error is not None:
                    raise local_error
                if not isinstance(agreed, _InitializationReady):
                    raise PublisherError(
                        "publisher initialization agreement was invalid"
                    )
                if agreed != local_contribution:
                    raise PublisherError(
                        "publisher initialization differs across trainer ranks"
                    )
            except Exception as exc:
                rollback_error = self._rollback_initialization()
                if rollback_error is not None:
                    raise PublisherError(f"{exc}; {rollback_error}") from exc
                raise

            self._config = config
            self._current_version = self._initial_base_version
            self._current_state = None
            self._eligible_base_version = self._initial_base_version
            self._initialized = True

    def _rollback_initialization(self) -> str | None:
        errors: list[Exception] = []
        resources = (
            self._source_close,
            getattr(self._transport, "close", None),
            getattr(self._catalog, "close", None),
            getattr(self._base_store, "close", None),
        )
        for close in resources:
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:
                errors.append(exc)
        error_detail = None
        if errors:
            error_detail = "publisher initialization rollback failed: " + "; ".join(
                str(error) for error in errors
            )
        self._initialized = False
        self._deregistered = True
        self._resources_closed = True
        self._deregister_error = error_detail
        self._condition.notify_all()
        return error_detail

    def publish_version(
        self,
        version: VersionId,
        layers: Sequence[str] | None = None,
        *,
        base_version: VersionId | None = None,
    ) -> PublishResult:
        """Capture on every rank; publish immutable CANONICAL bytes on rank 0."""
        with self._condition:
            if not self._initialized:
                if self._deregistered or self._deregistering:
                    raise PublisherStateError("publisher is deregistered")
                raise PublisherStateError("publisher is not initialized")
            if self._deregistered or self._deregistering:
                raise PublisherStateError("publisher is deregistered")
            if self._active_operations:
                raise PublisherStateError("a publisher operation is already active")
            request_error: Exception | None = None
            try:
                if layers is not None:
                    raise PublisherError(
                        "Phase 2.1 CANONICAL publication is complete-model only"
                    )
                if not isinstance(version, str) or not version:
                    raise PublisherError("target version must be non-empty")
                selected_base = self._eligible_base_version
                if selected_base is None:
                    raise PublisherStateError("publisher has no exact base version")
                if base_version is None:
                    raise PublisherError(
                        "CANONICAL publication requires an explicit base_version"
                    )
                if not isinstance(base_version, str) or not base_version:
                    raise PublisherError("base_version must be a non-empty string")
                if base_version != selected_base:
                    raise PublisherError(
                        f"requested base {base_version!r} is not the policy-eligible "
                        f"exact base {selected_base!r}"
                    )
                if selected_base == version:
                    raise PublisherError(
                        "target version must differ from exact base version"
                    )
                local_request: object = self._publication_request(version, base_version)
            except Exception as exc:
                request_error = exc
                local_request = _RemoteFailure(type(exc).__name__, str(exc))
            rank = self._rank()
            self._active_operations += 1
            self._operation_threads.add(threading.get_ident())

        prepared: _RankZeroPrepared | None = None
        try:
            agreed_request = self._agree(local_request)
            if isinstance(agreed_request, _RemoteFailure):
                if request_error is not None:
                    raise request_error
                self._raise_remote_failure(agreed_request)
            if not isinstance(agreed_request, _PublicationRequest):
                raise PublisherError("publisher request agreement was invalid")

            preflight: object | None = None
            preflight_error: Exception | None = None
            if rank == 0:
                try:
                    local_base = self._base_store.open_snapshot(
                        agreed_request.base_version
                    )
                    intent = self._attest_publication_request(
                        agreed_request, local_base
                    )
                    prepared = self._prepare_rank_zero(version, local_base)
                    preflight = _PreflightReady(intent)
                except Exception as exc:
                    preflight_error = exc
                    preflight = _RemoteFailure(type(exc).__name__, str(exc))
            agreed = self._coordinator.broadcast(preflight if rank == 0 else None)
            if isinstance(agreed, _RemoteFailure):
                if preflight_error is not None:
                    raise preflight_error
                self._raise_remote_failure(agreed)
            if not isinstance(agreed, _PreflightReady):
                raise PublisherError(
                    "rank-0 publication preflight broadcast was invalid"
                )
            if agreed.intent.request != agreed_request:
                raise PublisherError(
                    "rank-0 publication preflight changed the agreed request"
                )
            agreed_intent = agreed.intent

            capture_error: Exception | None = None
            try:
                consumer = (
                    prepared.encoder.consume_bucket
                    if rank == 0 and prepared is not None
                    else self._unexpected_nonzero_bucket
                )
                self._capture(version, consumer)
            except Exception as exc:
                capture_error = exc

            capture_contribution: object = (
                _CaptureComplete(version)
                if capture_error is None
                else _RemoteFailure(type(capture_error).__name__, str(capture_error))
            )
            agreed_capture = self._agree(capture_contribution)
            if isinstance(agreed_capture, _RemoteFailure):
                if capture_error is not None:
                    raise capture_error
                self._raise_remote_failure(agreed_capture)
            if agreed_capture != _CaptureComplete(version):
                raise PublisherError("publisher capture agreement was invalid")

            local_outcome: object | None = None
            if rank == 0:
                assert prepared is not None
                try:
                    local_outcome = self._publish_ready_rank_zero(prepared, version)
                except Exception as exc:
                    local_outcome = _RemoteFailure(type(exc).__name__, str(exc))

            outcome = self._coordinator.broadcast(local_outcome if rank == 0 else None)
            if isinstance(outcome, _RemoteFailure):
                self._raise_remote_failure(outcome)
            if not isinstance(outcome, _ReadyPublication):
                raise PublisherError(
                    "rank-0 publication broadcast returned an invalid outcome"
                )
            self._validate_ready_publication(outcome, agreed_intent)
            self._record_result(outcome.result)
            config = self._require_config()
            if config.publication_mode is PublicationMode.ASYNC:
                if rank == 0 and outcome.result.state is RevisionLifecycleState.READY:
                    self._retarget_async_observer(outcome)
                return outcome.result
            if outcome.result.state is RevisionLifecycleState.COMMITTED:
                return outcome.result
            return self._wait_for_commit(
                outcome.manifest,
                outcome.result.created,
            )
        finally:
            if prepared is not None:
                try:
                    prepared.encoder.abort()
                except Exception:
                    pass
            with self._condition:
                self._active_operations -= 1
                self._operation_threads.discard(threading.get_ident())
                self._condition.notify_all()

    def status(self) -> PublisherStatus:
        with self._condition:
            model_id = self._config.model_id if self._config is not None else ""
            mode = (
                self._config.publication_mode
                if self._config is not None
                else PublicationMode.BLOCK
            )
            return PublisherStatus(
                model_id=model_id,
                current_version=self._current_version,
                state=self._current_state,
                publication_mode=mode,
            )

    def deregister(self) -> None:
        """Cancel BLOCK waits, drain publication, then close owned resources in order."""
        with self._condition:
            if threading.get_ident() in self._operation_threads:
                raise PublisherStateError(
                    "deregister cannot run reentrantly from a publisher operation"
                )
            if self._resources_closed:
                if self._deregister_error is not None:
                    raise PublisherError(self._deregister_error)
                return
            if self._deregistering:
                while not self._resources_closed:
                    self._condition.wait()
                if self._deregister_error is not None:
                    raise PublisherError(self._deregister_error)
                return
            self._deregistering = True
            self._cancel.set()
            self._observer_target = None
            self._condition.notify_all()
            while self._active_operations or self._observer_thread is not None:
                self._condition.wait()
            self._initialized = False

        errors: list[Exception] = []
        resources = (
            self._source_close,
            getattr(self._transport, "close", None),
            getattr(self._catalog, "close", None),
            getattr(self._base_store, "close", None),
        )
        for close in resources:
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:
                errors.append(exc)
        error_detail = None
        if errors:
            error_detail = (
                "publisher deregistration resource close failed: "
                + "; ".join(str(error) for error in errors)
            )
        with self._condition:
            self._deregistered = True
            self._resources_closed = True
            self._deregister_error = error_detail
            self._condition.notify_all()
        if error_detail is not None:
            raise PublisherError(error_detail)

    def _validate_config(self, config: PublisherConfig) -> None:
        if not isinstance(config, PublisherConfig):
            raise TypeError("config must be PublisherConfig")
        if not isinstance(config.model_id, str) or not config.model_id:
            raise PublisherError("model_id must be non-empty")
        if not isinstance(config.catalog_endpoint, str) or not config.catalog_endpoint:
            raise PublisherError("catalog_endpoint must be non-empty")
        if not isinstance(config.publication_mode, PublicationMode):
            raise PublisherError(
                "publication_mode must be PublicationMode.BLOCK or PublicationMode.ASYNC"
            )
        if config.delta_transfer_method is not DeltaTransferMethod.CANONICAL:
            raise PublisherError("Phase 2.1 Publisher supports CANONICAL transfer only")
        if config.delta_method is None:
            raise PublisherError("CANONICAL publication requires delta_method")
        if config.compression_algorithm is None:
            raise PublisherError("CANONICAL publication requires compression_algorithm")
        try:
            encode_delta(config.delta_method, b"", b"")
            compress_payload(config.compression_algorithm, b"")
        except CodecError as exc:
            raise PublisherError(str(exc)) from exc
        if config.recovery_store is not None:
            raise PublisherError("RecoveryStore execution is outside Phase 2.1")
        if config.full_anchor_interval is not None:
            raise PublisherError(
                "full anchors are not a Phase 2.1 normal-delivery payload"
            )
        if config.transport.kind is TransportKind.FILESYSTEM:
            if not self._allow_filesystem_transport:
                raise PublisherError(
                    "filesystem CANONICAL transport is local-test-only"
                )
        elif config.transport.kind is not TransportKind.S3:
            raise PublisherError("Phase 2.1 CANONICAL transport must be S3")
        self._configured_transport_identity(config)

    def _build_transport(self, config: PublisherConfig) -> CanonicalTransport:
        identity = self._configured_transport_identity(config)
        if identity.kind == "s3":
            parsed = urlsplit(identity.namespace)
            return S3CanonicalTransport(
                parsed.netloc,
                parsed.path.removeprefix("/"),
                request_timeout_seconds=self._rpc_timeout_seconds,
            )
        return FilesystemCanonicalTransport(identity.namespace)

    def _configured_transport_identity(
        self, config: PublisherConfig
    ) -> CanonicalTransportIdentity:
        parsed = urlsplit(config.transport.root_uri)
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise PublisherError(
                "canonical transport root_uri contains unsupported components"
            )
        if config.transport.kind is TransportKind.S3:
            if parsed.scheme != "s3" or not parsed.netloc:
                raise PublisherError("S3 canonical transport needs s3://bucket/prefix")
            prefix = unquote(parsed.path).removeprefix("/")
            if prefix:
                try:
                    prefix = str(validate_relative_key(prefix))
                except ValueError as exc:
                    raise PublisherError(
                        "S3 canonical transport prefix is invalid"
                    ) from exc
            namespace = f"s3://{parsed.netloc}"
            if prefix:
                namespace = f"{namespace}/{prefix}"
            return CanonicalTransportIdentity("s3", namespace)
        if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
            raise PublisherError(
                "filesystem canonical transport needs file:///absolute/path"
            )
        path = unquote(parsed.path)
        if not path.startswith("/"):
            raise PublisherError("filesystem canonical transport path must be absolute")
        return CanonicalTransportIdentity("filesystem", str(Path(path).resolve()))

    def _validate_transport_identity(
        self, config: PublisherConfig, transport: CanonicalTransport
    ) -> None:
        expected = self._configured_transport_identity(config)
        identity = getattr(transport, "identity", None)
        if identity != expected:
            raise PublisherError(
                f"injected transport {identity!r} does not match configured {expected!r}"
            )

    @staticmethod
    def _validate_catalog_capabilities(catalog: RevisionCatalog) -> None:
        if not callable(getattr(catalog, "_get_revision_with_timeout", None)):
            raise PublisherError("catalog must provide bounded revision polling")
        if not callable(getattr(catalog, "_publish_revision_with_timeout", None)):
            raise PublisherError("catalog must provide bounded revision publication")

    def _rank(self) -> int:
        rank = getattr(self._coordinator, "rank", None)
        rank = rank() if callable(rank) else rank
        if not isinstance(rank, int) or rank < 0:
            raise PublisherError(f"invalid publisher coordinator rank {rank!r}")
        return rank

    @staticmethod
    def _unexpected_nonzero_bucket(_bucket: object) -> None:
        raise PublisherError(
            "canonical source invoked the bucket consumer on a nonzero rank"
        )

    def _publication_request(
        self, version: str, base_version: str
    ) -> _PublicationRequest:
        config = self._require_config()
        transport = self._require_transport()
        return _PublicationRequest(
            model_id=config.model_id,
            target_version=version,
            base_version=base_version,
            format_identity=self._format_identity,
            source_format_digest=self._capture_format_digest,
            transfer_method=DeltaTransferMethod.CANONICAL,
            delta_method=config.delta_method,
            compression_algorithm=config.compression_algorithm,
            publication_mode=config.publication_mode,
            catalog_endpoint=config.catalog_endpoint,
            transport_identity=transport.identity,
            maximum_bucket_bytes=self._maximum_bucket_bytes,
            maximum_encoded_ratio=self._maximum_encoded_ratio,
        )

    def _attest_publication_request(
        self, request: _PublicationRequest, base: CanonicalSnapshot
    ) -> _PublicationIntent:
        if base.version != request.base_version:
            raise PublisherError(
                "rank-0 exact base version differs from the agreed request"
            )
        if base.format_identity != request.format_identity:
            raise PublisherError(
                "rank-0 exact base format identity differs from the agreed request"
            )
        if base.format_digest != request.source_format_digest:
            raise PublisherError(
                "rank-0 exact base format digest differs from the agreed request"
            )
        try:
            catalog_base = self._bounded_get(request.model_id, request.base_version)
        except Exception as exc:
            raise PublisherError(
                f"bounded catalog base attestation failed: {exc}"
            ) from exc
        manifest = catalog_base.manifest
        if manifest.model_id != request.model_id or manifest.version != base.version:
            raise PublisherError(
                "catalog exact base model/version differs from local attestation"
            )
        if manifest.format_digest != base.format_digest:
            raise PublisherError(
                "catalog exact base format digest differs from local attestation"
            )
        if manifest.target_digest != base.target_digest:
            raise PublisherError(
                "catalog exact base digest differs from local attestation"
            )
        allowed_states = {RevisionLifecycleState.COMMITTED}
        if request.publication_mode is PublicationMode.ASYNC:
            allowed_states.add(RevisionLifecycleState.READY)
        if catalog_base.state not in allowed_states:
            raise PublisherError(
                "catalog exact base is not eligible for the publication policy"
            )
        return _PublicationIntent(
            request=request,
            format_digest=base.format_digest,
            base_digest=base.target_digest,
            producer_id=self._producer_id,
        )

    def _agree(self, value: object) -> object:
        agree = getattr(self._coordinator, "agree", None)
        if not callable(agree):
            raise PublisherError(
                "publisher coordinator does not support rank agreement"
            )
        return agree(value)

    def _prepare_rank_zero(
        self, version: str, base: CanonicalSnapshot
    ) -> _RankZeroPrepared:
        config = self._require_config()
        catalog = self._require_catalog()
        transport = self._require_transport()
        children: list[StoredObject] = []

        def publish_bucket(bucket) -> Any:
            key = canonical_object_key(
                config.model_id,
                base.version,
                version,
                f"bucket-{bucket.ordinal:08d}.mxcd",
            )
            stored = transport.publish(key, bucket.data, bucket.checksum)
            if stored.checksum != bucket.checksum or stored.size != len(bucket.data):
                raise PublisherError(
                    "transport returned mismatched canonical bucket metadata"
                )
            self._validate_stored_object(config, stored, key)
            transport.verify(stored)
            children.append(stored)
            return stored.location

        encoder = CanonicalDeltaEncoder(
            model_id=config.model_id,
            target_version=version,
            base_store=self._base_store,
            base=base,
            delta_method=config.delta_method,
            compression_algorithm=config.compression_algorithm,
            publish_bucket=publish_bucket,
            maximum_encoded_ratio=self._maximum_encoded_ratio,
            maximum_bucket_bytes=self._maximum_bucket_bytes,
        )
        return _RankZeroPrepared(config, catalog, transport, base, encoder, children)

    def _publish_ready_rank_zero(
        self, prepared: _RankZeroPrepared, version: str
    ) -> _ReadyPublication:
        config = prepared.config
        catalog = prepared.catalog
        transport = prepared.transport
        base = prepared.base
        children = prepared.children
        publication = prepared.encoder.finish()

        root: StoredObject | None = None
        if publication.changed:
            for child in children:
                transport.verify(child)
            if publication.root_bytes is None or publication.root_checksum is None:
                raise PublisherError(
                    "dirty canonical publication has no root index bytes"
                )
            root_key = canonical_object_key(
                config.model_id,
                base.version,
                version,
                "root.json",
            )
            root = transport.publish(
                root_key,
                publication.root_bytes,
                publication.root_checksum,
            )
            if root.checksum != publication.root_checksum or root.size != len(
                publication.root_bytes
            ):
                raise PublisherError(
                    "transport returned mismatched canonical root metadata"
                )
            self._validate_stored_object(config, root, root_key)
            transport.verify(root)

        manifest = RevisionManifest(
            model_id=config.model_id,
            version=version,
            base_version=base.version,
            transfer_method=DeltaTransferMethod.CANONICAL,
            delta_method=config.delta_method,
            compression_algorithm=config.compression_algorithm,
            format_digest=publication.root_index.format_digest,
            base_digest=publication.root_index.base_digest,
            target_digest=publication.root_index.target_digest,
            ranks=(
                RevisionRank(
                    trainer_rank=0,
                    producer_id=self._producer_id,
                    source_layout_digest=publication.root_index.format_digest,
                    delta=(
                        RankDelta(
                            change_state=ChangeState.DIRTY,
                            checksum=root.checksum,
                            location=root.location,
                        )
                        if root is not None
                        else RankDelta(change_state=ChangeState.CLEAN)
                    ),
                ),
            ),
        )
        publish_bounded = getattr(catalog, "_publish_revision_with_timeout")
        try:
            published = publish_bounded(
                manifest,
                publisher_id=self._producer_id,
                publication_mode=config.publication_mode,
                timeout=self._rpc_timeout_seconds,
            )
        except grpc.RpcError as exc:
            raise PublisherError(f"bounded catalog publication failed: {exc}") from exc
        except Exception as exc:
            raise PublisherError(f"bounded catalog publication failed: {exc}") from exc
        record = published.revision
        if record.manifest != manifest:
            raise PublisherError(
                "catalog returned a different immutable revision manifest"
            )
        if record.state not in {
            RevisionLifecycleState.READY,
            RevisionLifecycleState.COMMITTED,
        }:
            raise PublisherError(
                f"catalog returned illegal revision state {record.state!r}"
            )
        result = PublishResult(
            model_id=config.model_id,
            version=version,
            state=record.state,
            created=published.created,
        )
        return _ReadyPublication(manifest, result)

    def _wait_for_commit(
        self, manifest: RevisionManifest, created: bool
    ) -> PublishResult:
        while True:
            if self._cancel.wait(self._poll_interval_seconds):
                raise PublicationCancelled(
                    f"BLOCK wait for {manifest.model_id}@{manifest.version} was cancelled; "
                    "the immutable revision remains READY"
                )
            try:
                record = self._bounded_get(manifest.model_id, manifest.version)
            except grpc.RpcError as exc:
                if exc.code() is grpc.StatusCode.DEADLINE_EXCEEDED:
                    continue
                raise PublisherError(f"catalog BLOCK wait failed: {exc}") from exc
            except Exception as exc:
                raise PublisherError(f"catalog BLOCK wait failed: {exc}") from exc
            if record.manifest != manifest:
                raise PublisherError(
                    "catalog changed the immutable manifest during BLOCK wait"
                )
            if record.state is RevisionLifecycleState.COMMITTED:
                result = PublishResult(
                    manifest.model_id,
                    manifest.version,
                    RevisionLifecycleState.COMMITTED,
                    created,
                )
                self._record_result(result)
                return result
            if record.state is not RevisionLifecycleState.READY:
                raise PublisherError(
                    f"catalog returned illegal revision state {record.state!r}"
                )

    def _record_result(self, result: PublishResult) -> None:
        with self._condition:
            if not (
                self._current_version == result.version
                and self._current_state is RevisionLifecycleState.COMMITTED
                and result.state is RevisionLifecycleState.READY
            ):
                self._current_version = result.version
                self._current_state = result.state
            config = self._config
            if config is not None and (
                config.publication_mode is PublicationMode.ASYNC
                or result.state is RevisionLifecycleState.COMMITTED
            ):
                self._eligible_base_version = result.version

    def _validate_ready_publication(
        self, publication: _ReadyPublication, intent: _PublicationIntent
    ) -> None:
        manifest = publication.manifest
        result = publication.result
        expected = (
            intent.model_id,
            intent.target_version,
            intent.base_version,
            intent.transfer_method,
            intent.delta_method,
            intent.compression_algorithm,
            intent.format_digest,
            intent.base_digest,
        )
        actual = (
            manifest.model_id,
            manifest.version,
            manifest.base_version,
            manifest.transfer_method,
            manifest.delta_method,
            manifest.compression_algorithm,
            manifest.format_digest,
            manifest.base_digest,
        )
        if actual != expected:
            raise PublisherError(
                "rank-0 publication identity differs from the agreed intent"
            )
        if not _SHA256.fullmatch(manifest.target_digest):
            raise PublisherError(
                "rank-0 publication target digest is not canonical SHA-256"
            )
        if (
            result.model_id != manifest.model_id
            or result.version != manifest.version
            or result.state
            not in {
                RevisionLifecycleState.READY,
                RevisionLifecycleState.COMMITTED,
            }
            or not isinstance(result.created, bool)
        ):
            raise PublisherError(
                "rank-0 publication result differs from its immutable manifest"
            )
        if len(manifest.ranks) != 1:
            raise PublisherError(
                "CANONICAL publication must contain exactly one rank entry"
            )
        rank = manifest.ranks[0]
        if (
            rank.trainer_rank != 0
            or rank.producer_id != intent.producer_id
            or rank.source_layout_digest != manifest.format_digest
            or rank.shards
            or rank.delta is None
            or rank.delta.delta_descriptor is not None
        ):
            raise PublisherError("CANONICAL publication rank-0 entry is invalid")
        if rank.delta.change_state is ChangeState.CLEAN:
            if rank.delta.checksum is not None or rank.delta.location is not None:
                raise PublisherError(
                    "clean CANONICAL publication carries a transfer reference"
                )
        elif rank.delta.change_state is ChangeState.DIRTY:
            if rank.delta.checksum is None or rank.delta.location is None:
                raise PublisherError(
                    "dirty CANONICAL publication has no root-index reference"
                )
            try:
                validate_checksum(rank.delta.checksum)
            except Exception as exc:
                raise PublisherError(
                    "dirty CANONICAL publication root checksum is invalid"
                ) from exc
            self._validate_stored_object(
                self._require_config(),
                StoredObject(rank.delta.location, rank.delta.checksum, 0),
                canonical_object_key(
                    intent.model_id,
                    intent.base_version,
                    intent.target_version,
                    "root.json",
                ),
            )
        else:
            raise PublisherError("CANONICAL publication has an illegal change state")

    def _retarget_async_observer(self, publication: _ReadyPublication) -> None:
        thread: threading.Thread | None = None
        with self._condition:
            if self._deregistering or self._deregistered:
                return
            self._observer_target = publication
            if self._observer_thread is None:
                thread = threading.Thread(
                    target=self._observe_async_publications,
                    name="mx-canonical-observer-worker",
                    daemon=True,
                )
                self._observer_thread = thread
            self._condition.notify_all()
        if thread is None:
            return
        try:
            thread.start()
        except Exception:
            with self._condition:
                if self._observer_thread is thread:
                    self._observer_thread = None
                    self._observer_target = None
                self._condition.notify_all()

    def _observe_async_publications(self) -> None:
        def observe() -> None:
            while True:
                with self._condition:
                    while self._observer_target is None and not self._cancel.is_set():
                        self._condition.wait()
                    if self._cancel.is_set():
                        return
                    publication = self._observer_target
                assert publication is not None
                if self._cancel.wait(self._poll_interval_seconds):
                    return
                with self._condition:
                    if publication != self._observer_target:
                        continue
                manifest = publication.manifest
                try:
                    record = self._bounded_get(manifest.model_id, manifest.version)
                except grpc.RpcError as exc:
                    if exc.code() is grpc.StatusCode.DEADLINE_EXCEEDED:
                        continue
                    record = None
                except Exception:
                    record = None
                with self._condition:
                    if publication != self._observer_target:
                        continue
                    if record is None or record.manifest != manifest:
                        self._observer_target = None
                        continue
                    if record.state is RevisionLifecycleState.COMMITTED:
                        if self._current_version == manifest.version:
                            self._current_state = RevisionLifecycleState.COMMITTED
                        self._observer_target = None
                    elif record.state is not RevisionLifecycleState.READY:
                        self._observer_target = None

        try:
            observe()
        finally:
            with self._condition:
                if self._observer_thread is threading.current_thread():
                    self._observer_thread = None
                self._observer_target = None
                self._condition.notify_all()

    def _bounded_get(self, model_id: str, version: str):
        catalog = self._require_catalog()
        bounded_get = getattr(catalog, "_get_revision_with_timeout", None)
        if not callable(bounded_get):
            raise PublisherError("catalog does not provide bounded revision polling")
        return bounded_get(model_id, version, timeout=self._rpc_timeout_seconds)

    def _validate_stored_object(
        self, config: PublisherConfig, stored: StoredObject, expected_key: str
    ) -> None:
        if (
            not isinstance(stored, StoredObject)
            or not isinstance(stored.size, int)
            or isinstance(stored.size, bool)
            or stored.size < 0
        ):
            raise PublisherError("transport returned invalid canonical object metadata")
        if not isinstance(stored.location, DeltaLocation):
            raise PublisherError("transport returned invalid canonical object metadata")
        try:
            validate_checksum(stored.checksum)
        except Exception as exc:
            raise PublisherError(
                "transport returned invalid canonical object checksum"
            ) from exc
        try:
            relative_key = validate_relative_key(expected_key)
        except ValueError as exc:
            raise PublisherError("requested canonical object key is invalid") from exc
        identity = self._configured_transport_identity(config)
        if identity.kind == "filesystem":
            location = stored.location.filesystem
            if location is None or stored.location.s3 is not None:
                raise PublisherError(
                    "transport returned a non-filesystem canonical location"
                )
            path = Path(location.path)
            if not path.is_absolute():
                raise PublisherError(
                    "transport returned a relative filesystem location"
                )
            expected_path = Path(identity.namespace).joinpath(*relative_key.parts)
            if path != expected_path:
                raise PublisherError(
                    "transport did not return the exact canonical object key"
                )
            return

        location = stored.location.s3
        if location is None or stored.location.filesystem is not None:
            raise PublisherError("transport returned a non-S3 canonical location")
        parsed = urlsplit(identity.namespace)
        prefix = parsed.path.removeprefix("/")
        expected_key = (
            f"{prefix}/{relative_key.as_posix()}" if prefix else relative_key.as_posix()
        )
        if location.bucket != parsed.netloc or location.key != expected_key:
            raise PublisherError(
                "transport did not return the exact canonical object key"
            )

    @staticmethod
    def _raise_remote_failure(failure: _RemoteFailure) -> None:
        if failure.error_type == PublicationCancelled.__name__:
            raise PublicationCancelled(failure.detail)
        raise PublisherError(
            f"rank-0 publication failed with {failure.error_type}: {failure.detail}"
        )

    def _require_config(self) -> PublisherConfig:
        if self._config is None:
            raise PublisherStateError("publisher is not initialized")
        return self._config

    def _require_catalog(self) -> RevisionCatalog:
        if self._catalog is None:
            raise PublisherStateError("publisher catalog is not initialized")
        return self._catalog

    def _require_transport(self) -> CanonicalTransport:
        if self._transport is None:
            raise PublisherStateError("publisher transport is not initialized")
        return self._transport
