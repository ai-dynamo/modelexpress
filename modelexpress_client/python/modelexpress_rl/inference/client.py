# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-local generator lifecycle for ModelExpress RL refit."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

import grpc
from modelexpress import auth, envs
from modelexpress.client import _get_server_url

from modelexpress_rl import envs as rl_envs
from modelexpress_rl.version import WeightVersionRef

from .. import refit_pb2, refit_pb2_grpc
from ..control import WeightVersion, WeightVersionState, _weight_version
from ..object_storage import ObjectStorageType
from ..train import WeightPayloadFormat
from .adapter import GeneratorEngineContext
from .plan import WeightSource
from .receiver import ObjectStorageGeneratorConfig
from .runtime import GeneratorRuntime
from .session import SessionUpdate

logger = logging.getLogger("modelexpress_rl.inference.client")


class _EngineState(str, Enum):
    READY = "READY"
    UNCERTAIN = "UNCERTAIN"


def _required(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} is required")
    return value


@dataclass(frozen=True)
class ModelExpressGeneratorConfig:
    """Immutable configuration for one rank-local generator client."""

    # Live rank-local objects required by the selected inference engine adapter.
    engine_context: GeneratorEngineContext
    # Logical model identity; defaults to MODEL_NAME.
    model_name: str | None = None
    # Fresh process-lifetime identity; generated when omitted.
    worker_id: str | None = None
    # Address of the central ModelExpress server; uses the standard MX default.
    server_url: str | None = None
    # Worker registration lifetime; defaults to three heartbeat intervals.
    registration_ttl_seconds: int | None = None
    # Weight-version lease lifetime; defaults to the registration lifetime.
    lease_ttl_seconds: int | None = None
    # Maximum source-discovery and transfer attempts for one staged update.
    max_transfer_attempts: int = 3
    # Maximum number of payload revisions applied by one target replay.
    max_replay_chain_length: int = 64
    # Deadline applied independently to each control-plane or manifest RPC.
    rpc_timeout_seconds: float = 30.0
    # Canonical object-storage checkpoint settings.
    object_storage: ObjectStorageGeneratorConfig | None = None
    # Ordered fallback for full-tensor sources. Canonical object storage is
    # currently isolated because peer installation does not advance its local
    # checkpoint base.
    source_order: tuple[WeightSource, ...] | None = None

    def __post_init__(self) -> None:
        """Validate explicit settings before client initialization."""
        if self.registration_ttl_seconds is not None:
            rl_envs.require_positive_int(
                self.registration_ttl_seconds, "registration_ttl_seconds"
            )
        if self.lease_ttl_seconds is not None:
            rl_envs.require_positive_int(self.lease_ttl_seconds, "lease_ttl_seconds")
        rl_envs.require_positive_int(
            self.max_transfer_attempts, "max_transfer_attempts"
        )
        rl_envs.require_positive_int(
            self.max_replay_chain_length, "max_replay_chain_length"
        )
        rl_envs.require_positive_float(self.rpc_timeout_seconds, "rpc_timeout_seconds")
        if self.source_order is not None:
            if not isinstance(self.source_order, tuple) or not self.source_order:
                raise ValueError("source_order must be a non-empty tuple")
            if any(
                not isinstance(source, WeightSource) for source in self.source_order
            ):
                raise TypeError("source_order entries must be WeightSource values")
            if len(set(self.source_order)) != len(self.source_order):
                raise ValueError("source_order must not contain duplicates")
            if (
                WeightSource.OBJECT_STORAGE in self.source_order
                and self.object_storage is None
            ):
                raise ValueError(
                    "OBJECT_STORAGE source requires object_storage settings"
                )
            if (
                self.object_storage is not None
                and WeightSource.OBJECT_STORAGE not in self.source_order
            ):
                raise ValueError(
                    "object_storage settings require OBJECT_STORAGE in source_order"
                )
            if self.object_storage is not None and self.source_order != (
                WeightSource.OBJECT_STORAGE,
            ):
                raise ValueError(
                    "object_storage currently requires source_order="
                    "(WeightSource.OBJECT_STORAGE,)"
                )


class StagedWeightHandle:
    """Local verified staging buffers for one exact WeightVersion."""

    def __init__(
        self,
        *,
        client: ModelExpressGeneratorClient,
        version_id: str,
        update: SessionUpdate | None,
    ) -> None:
        self._client = client
        self.version_id = version_id
        self._update = update
        self._no_op_released = False

    def release(self) -> None:
        """Release local staging buffers; repeated calls are idempotent."""
        self._client._release_staged(self)

    @property
    def metrics(self) -> dict[str, float]:
        """Return preparation metrics exposed by the selected adapter."""
        if self._update is None:
            return {}
        return self._update.prepared.metrics

    @property
    def applied(self) -> bool:
        """Return whether engine installation completed."""
        if self._update is None:
            return True
        return self._update.applied


class _VersionLease:
    """Keep one version protected through installation or staged release."""

    def __init__(
        self,
        *,
        client: ModelExpressGeneratorClient,
        version_id: str,
        lease_id: str,
        stop: threading.Event,
        renewal: threading.Thread,
    ) -> None:
        self._client = client
        self._version_id = version_id
        self._lease_id = lease_id
        self._stop = stop
        self._renewal = renewal
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._stop.set()
        self._renewal.join()
        self._client._delete_version_lease(
            version_id=self._version_id,
            lease_id=self._lease_id,
        )
        self._closed = True


class ModelExpressGeneratorClient:
    """Synchronous rank-local generator client for exact-version refit."""

    def __init__(self) -> None:
        self._channel: grpc.Channel | None = None
        self._stub: refit_pb2_grpc.RefitServiceStub | None = None
        self._registration_stop = threading.Event()
        self._registration_thread: threading.Thread | None = None
        self._operation_lock = threading.RLock()
        self._active_handle: StagedWeightHandle | None = None
        self._serving_version_id: str | None = None
        self._engine_state = _EngineState.READY
        self._runtime: GeneratorRuntime | None = None
        self._closed = False

    @classmethod
    def initialize(
        cls,
        config: ModelExpressGeneratorConfig,
    ) -> ModelExpressGeneratorClient:
        """Initialize one generator rank with immutable operating settings.

        ``config.engine_context`` contains the engine's live rank-local objects.
        Callers do not construct ModelExpress adapter or receiver implementations.
        """
        if not isinstance(config, ModelExpressGeneratorConfig):
            raise TypeError("config must be a ModelExpressGeneratorConfig")
        model_name = _required(config.model_name or envs.MODEL_NAME or "", "model_name")
        worker_id = _required(config.worker_id or uuid.uuid4().hex[:8], "worker_id")
        server_url = _get_server_url(config.server_url)
        registration_ttl_seconds = config.registration_ttl_seconds
        if registration_ttl_seconds is None:
            registration_ttl_seconds = envs.MX_HEARTBEAT_INTERVAL_SECS * 3
        lease_ttl_seconds = config.lease_ttl_seconds
        if lease_ttl_seconds is None:
            lease_ttl_seconds = registration_ttl_seconds
        registration_ttl_seconds = rl_envs.require_positive_int(
            registration_ttl_seconds, "registration_ttl_seconds"
        )
        if (
            config.object_storage is not None
            and config.object_storage.storage_type is not ObjectStorageType.S3
        ):
            raise ValueError("only S3 object storage is currently supported")

        client = cls()
        client.model_name = model_name
        client.worker_id = worker_id
        client.server_url = server_url
        client._registration_ttl_seconds = registration_ttl_seconds
        client._lease_ttl_seconds = lease_ttl_seconds
        client._rpc_timeout_seconds = config.rpc_timeout_seconds
        client._max_replay_chain_length = config.max_replay_chain_length
        try:
            runtime = GeneratorRuntime.initialize(
                engine_context=config.engine_context,
                worker_id=worker_id,
                server_url=server_url,
                object_storage=config.object_storage,
                source_order=config.source_order,
                max_transfer_attempts=config.max_transfer_attempts,
                rpc_timeout_seconds=config.rpc_timeout_seconds,
                service=lambda: client._service,
                start_lease=client._start_version_lease,
            )
            client._runtime = runtime
            client._serving_version_id = runtime.initial_version_id
            if client._serving_version_id is not None:
                client._validate_initial_base(client._serving_version_id)
            client._register_worker()
            client._registration_thread = threading.Thread(
                target=client._renew_worker_registration,
                name=f"modelexpress-refit-renew-{worker_id}",
                daemon=True,
            )
            try:
                client._registration_thread.start()
            except Exception:
                client._registration_thread = None
                raise
        except Exception:
            client.close()
            raise
        return client

    def stage_weight(self, *, version: WeightVersionRef) -> StagedWeightHandle:
        """Synchronously transfer and verify one full-weight version."""
        if not isinstance(version, WeightVersionRef):
            raise TypeError("version must be a WeightVersionRef")
        with self._operation_lock:
            if self._active_handle is not None:
                if self._active_handle.version_id == version.version_id:
                    return self._active_handle
                raise RuntimeError("another generator update is still active")
            assert self._runtime is not None
            if (
                self._runtime.initial_version_id is not None
                and version.version_id == self._serving_version_id
                and self._engine_state is _EngineState.READY
            ):
                self._fetch_ready_version(
                    version.version_id,
                    target_version_id=version.version_id,
                )
                self._active_handle = StagedWeightHandle(
                    client=self,
                    version_id=version.version_id,
                    update=None,
                )
                return self._active_handle
            if self._runtime.initial_version_id is not None:
                chain = self._resolve_replay_chain(version.version_id)
                update = (
                    self._runtime.session.stage(chain[0])
                    if len(chain) == 1
                    else self._runtime.session.stage_chain(chain)
                )
            else:
                ready = self._get_ready_version(version.version_id)
                update = self._runtime.session.stage(ready)
            self._active_handle = StagedWeightHandle(
                client=self,
                version_id=version.version_id,
                update=update,
            )
            return self._active_handle

    def apply_weight(self, staged: StagedWeightHandle) -> Any:
        """Install a verified local staged version at the caller's safe point."""
        if not isinstance(staged, StagedWeightHandle) or staged._client is not self:
            raise ValueError("staged handle does not belong to this client")
        with self._operation_lock:
            if staged._update is None:
                if staged._no_op_released:
                    raise RuntimeError("staged weight has already been released")
                return None
            if staged._update.released:
                raise RuntimeError("staged weight has already been released")
            assert self._runtime is not None
            try:
                result = self._runtime.session.apply(staged._update)
            except BaseException:
                if staged._update.installation_started and not staged._update.applied:
                    self._engine_state = _EngineState.UNCERTAIN
                raise
            self._serving_version_id = staged.version_id
            self._engine_state = _EngineState.READY
            return result

    def close(self) -> None:
        """Stop renewal and release control-plane and adapter resources."""
        if self._closed:
            return
        with self._operation_lock:
            if self._active_handle is not None:
                self._release_staged(self._active_handle)
        if self._registration_thread is not None:
            self._registration_stop.set()
            self._registration_thread.join()
            self._registration_thread = None
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        self._closed = True

    def __enter__(self) -> ModelExpressGeneratorClient:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()

    @property
    def _service(self) -> refit_pb2_grpc.RefitServiceStub:
        if self._channel is None:
            self._channel = auth.with_auth(grpc.insecure_channel(self.server_url))
            self._stub = refit_pb2_grpc.RefitServiceStub(self._channel)
        assert self._stub is not None
        return self._stub

    def _register_worker(self) -> None:
        self._service.RegisterWorker(
            refit_pb2.RegisterWorkerRequest(
                worker=refit_pb2.WorkerRegistration(
                    worker_id=self.worker_id,
                    role=refit_pb2.WORKER_ROLE_GENERATOR,
                    model_name=self.model_name,
                ),
                ttl_seconds=self._registration_ttl_seconds,
            ),
            timeout=self._rpc_timeout_seconds,
        )

    def _renew_worker_registration(self) -> None:
        interval_seconds = max(self._registration_ttl_seconds / 3, 0.1)
        while not self._registration_stop.wait(interval_seconds):
            try:
                self._register_worker()
            except grpc.RpcError as error:
                logger.warning("worker registration renewal failed: %s", error)
                continue
            except Exception:
                logger.exception("unexpected worker registration renewal failure")
                continue

    def _get_ready_version(self, version_id: str) -> WeightVersion:
        version = self._fetch_ready_version(
            version_id,
            target_version_id=version_id,
        )
        assert self._runtime is not None
        self._runtime.session.validate(version)
        return version

    def _fetch_ready_version(
        self,
        version_id: str,
        *,
        target_version_id: str,
    ) -> WeightVersion:
        try:
            response = self._service.GetWeightVersion(
                refit_pb2.GetWeightVersionRequest(uid=version_id),
                timeout=self._rpc_timeout_seconds,
            )
        except grpc.RpcError as error:
            raise RuntimeError(
                f"target {target_version_id!r}: failed to resolve revision "
                f"{version_id!r}: {error.details()}"
            ) from error
        if not response.HasField("version"):
            raise RuntimeError(
                f"target {target_version_id!r}: MX GetWeightVersion response is "
                f"missing revision {version_id!r}"
            )
        version = _weight_version(response.version)
        if version.version_id != version_id:
            raise RuntimeError(
                f"target {target_version_id!r}: requested revision {version_id!r} "
                f"but MX returned {version.version_id!r}"
            )
        if version.state is not WeightVersionState.READY:
            raise RuntimeError(
                f"target {target_version_id!r}: revision {version_id!r} is not READY"
            )
        if version.model_name != self.model_name:
            raise RuntimeError(
                f"target {target_version_id!r}: revision {version_id!r} model_name "
                "does not match the generator"
            )
        return version

    def _resolve_replay_chain(
        self,
        target_version_id: str,
    ) -> tuple[WeightVersion, ...]:
        """Resolve a canonical chain completely before payload preparation."""
        serving_version_id = self._serving_version_id
        if serving_version_id is None:
            raise RuntimeError("canonical replay requires a known serving version")

        reverse_chain: list[WeightVersion] = []
        seen: set[str] = set()
        revision_id = target_version_id
        layout_signature: str | None = None
        while True:
            if reverse_chain and revision_id == serving_version_id:
                break
            if revision_id in seen:
                raise RuntimeError(
                    f"target {target_version_id!r}: cycle detected at revision "
                    f"{revision_id!r}"
                )
            if len(reverse_chain) >= self._max_replay_chain_length:
                raise RuntimeError(
                    f"target {target_version_id!r}: replay exceeds maximum chain "
                    f"length {self._max_replay_chain_length} before revision "
                    f"{revision_id!r}"
                )
            seen.add(revision_id)
            try:
                version = self._fetch_ready_version(
                    revision_id,
                    target_version_id=target_version_id,
                )
            except RuntimeError as error:
                if revision_id != target_version_id:
                    raise RuntimeError(
                        f"target {target_version_id!r}: chain does not match serving "
                        f"version {serving_version_id!r}; {error}"
                    ) from error
                raise
            if version.layout_signature:
                if layout_signature is None:
                    layout_signature = version.layout_signature
                elif version.layout_signature != layout_signature:
                    raise RuntimeError(
                        f"target {target_version_id!r}: layout format mismatch at "
                        f"revision {revision_id!r}"
                    )
            if version.object_storage is None:
                raise RuntimeError(
                    f"target {target_version_id!r}: no legal source for revision "
                    f"{revision_id!r}; object-storage source is missing"
                )
            reverse_chain.append(version)
            if version.version_id == serving_version_id:
                break
            if version.payload_format is WeightPayloadFormat.FULL_HF_CHECKPOINT:
                if version.base_version_id is not None:
                    raise RuntimeError(
                        f"target {target_version_id!r}: FULL_HF_CHECKPOINT revision "
                        f"{revision_id!r} must not have base_version_id"
                    )
                break
            if version.payload_format is not WeightPayloadFormat.XOR_DELTA:
                raise RuntimeError(
                    f"target {target_version_id!r}: revision {revision_id!r} has "
                    f"unsupported replay format {version.payload_format.value}"
                )
            if version.base_version_id is None:
                raise RuntimeError(
                    f"target {target_version_id!r}: XOR_DELTA revision "
                    f"{revision_id!r} is missing base_version_id"
                )
            revision_id = version.base_version_id

        chain = tuple(reversed(reverse_chain))
        assert self._runtime is not None
        for version in chain:
            self._runtime.session.validate(version)
        return chain

    def _validate_initial_base(self, version_id: str) -> None:
        response = self._service.GetWeightVersion(
            refit_pb2.GetWeightVersionRequest(uid=version_id),
            timeout=self._rpc_timeout_seconds,
        )
        if not response.HasField("version"):
            raise RuntimeError("MX GetWeightVersion response is missing version")
        version = _weight_version(response.version)
        if version.state is not WeightVersionState.READY:
            raise RuntimeError(f"initial base {version_id!r} is not READY")
        if version.model_name != self.model_name:
            raise RuntimeError("initial base model_name does not match the generator")

    def _register_lease(self, version_id: str):
        response = self._service.RegisterVersionLease(
            refit_pb2.RegisterVersionLeaseRequest(
                version_id=version_id,
                worker_id=self.worker_id,
                ttl_seconds=self._lease_ttl_seconds,
            ),
            timeout=self._rpc_timeout_seconds,
        )
        if not response.HasField("lease"):
            raise RuntimeError("MX RegisterVersionLease response is missing lease")
        return response.lease

    def _start_version_lease(self, version_id: str) -> _VersionLease:
        lease = self._register_lease(version_id)
        stop = threading.Event()

        def renew() -> None:
            interval_seconds = max(self._lease_ttl_seconds / 3, 0.1)
            while not stop.wait(interval_seconds):
                try:
                    self._register_lease(version_id)
                except grpc.RpcError as error:
                    logger.warning(
                        "version %s lease renewal failed: %s",
                        version_id,
                        error,
                    )
                except Exception:
                    logger.exception(
                        "unexpected version %s lease renewal failure",
                        version_id,
                    )

        renewal = threading.Thread(
            target=renew,
            name=f"modelexpress-refit-lease-{self.worker_id}",
            daemon=True,
        )
        try:
            renewal.start()
        except Exception:
            self._delete_version_lease(
                version_id=version_id,
                lease_id=lease.lease_id,
            )
            raise
        return _VersionLease(
            client=self,
            version_id=version_id,
            lease_id=lease.lease_id,
            stop=stop,
            renewal=renewal,
        )

    def _delete_version_lease(self, *, version_id: str, lease_id: str) -> None:
        self._service.DeleteVersionLease(
            refit_pb2.DeleteVersionLeaseRequest(
                version_id=version_id,
                lease_id=lease_id,
                worker_id=self.worker_id,
            ),
            timeout=self._rpc_timeout_seconds,
        )

    def _release_staged(self, staged: StagedWeightHandle) -> None:
        if staged._client is not self:
            raise ValueError("staged handle does not belong to this client")
        with self._operation_lock:
            if staged._update is None:
                if staged._no_op_released:
                    return
                staged._no_op_released = True
                if self._active_handle is staged:
                    self._active_handle = None
                return
            if staged._update.released:
                return
            assert self._runtime is not None
            self._runtime.session.release(staged._update)
            if self._active_handle is staged:
                self._active_handle = None


__all__ = [
    "ModelExpressGeneratorClient",
    "ModelExpressGeneratorConfig",
    "StagedWeightHandle",
    "WeightSource",
]
