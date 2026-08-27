# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-local generator lifecycle for ModelExpress RL refit."""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Any

import grpc
from modelexpress import auth, envs
from modelexpress.client import MxClient, _get_server_url

from modelexpress_rl import envs as rl_envs
from modelexpress_rl.train import WeightPayloadFormat
from modelexpress_rl.version import WeightVersionRef

from .. import refit_pb2, refit_pb2_grpc
from ..control import WeightVersion, WeightVersionState, _weight_version
from ..object_storage import ObjectStorageType
from .adapter import (
    GeneratorEngineAdapter,
    GeneratorEngineContext,
    GeneratorTransferInputs,
)
from .engines import _create_generator_adapter
from .receiver import ObjectStorageGeneratorConfig
from .refit_strategy import RefitStrategy
from .refit_strategy.peer import _PeerRefitStrategy
from .refit_strategy.trainer import _TrainerRefitStrategy

logger = logging.getLogger("modelexpress_rl.inference.client")


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
    # Deadline applied independently to each control-plane or manifest RPC.
    rpc_timeout_seconds: float = 30.0
    # Canonical object-storage checkpoint settings. Omit for the NIXL path.
    object_storage: ObjectStorageGeneratorConfig | None = None

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
        rl_envs.require_positive_float(self.rpc_timeout_seconds, "rpc_timeout_seconds")


class StagedWeightHandle:
    """Local verified staging buffers for one exact WeightVersion."""

    def __init__(
        self,
        *,
        client: ModelExpressGeneratorClient,
        version_id: str,
        staged: Any,
        lease: _VersionLease,
    ) -> None:
        self._client = client
        self.version_id = version_id
        self._staged = staged
        self._lease = lease
        self._applied = False
        self._apply_result: Any = None
        self._released = False

    def release(self) -> None:
        """Release local staging buffers; repeated calls are idempotent."""
        self._client._release_staged(self)

    @property
    def metrics(self) -> dict[str, float]:
        """Return preparation metrics exposed by the selected adapter."""
        return dict(getattr(self._staged, "metrics", {}))

    @property
    def applied(self) -> bool:
        """Return whether engine installation completed."""
        return self._applied


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
        self._adapter: GeneratorEngineAdapter | None = None
        self._p2p_client: MxClient | None = None
        self._refit_strategies: tuple[RefitStrategy, ...] = ()
        self._uses_s3 = False
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

        adapter = _create_generator_adapter(
            engine_context=config.engine_context,
            worker_id=worker_id,
            object_storage=config.object_storage,
        )
        client = cls()
        client.model_name = model_name
        client.worker_id = worker_id
        client.server_url = server_url
        client._registration_ttl_seconds = registration_ttl_seconds
        client._lease_ttl_seconds = lease_ttl_seconds
        client._max_transfer_attempts = config.max_transfer_attempts
        client._rpc_timeout_seconds = config.rpc_timeout_seconds
        client._adapter = adapter
        client._uses_s3 = config.object_storage is not None
        if config.object_storage is not None:
            client._serving_version_id = config.object_storage.initial_base_version_id
        else:
            client._p2p_client = MxClient(server_url=server_url)
            client._refit_strategies = (
                _PeerRefitStrategy(
                    adapter=adapter,
                    p2p_client=client._p2p_client,
                    worker_id=worker_id,
                    max_transfer_attempts=config.max_transfer_attempts,
                    rpc_timeout_seconds=config.rpc_timeout_seconds,
                ),
                _TrainerRefitStrategy(
                    adapter=adapter,
                    service=lambda: client._service,
                    max_transfer_attempts=config.max_transfer_attempts,
                    rpc_timeout_seconds=config.rpc_timeout_seconds,
                ),
            )
        try:
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
            ready = self._get_ready_version(version.version_id)
            staged, lease = self._stage_with_lease(ready)
            self._active_handle = StagedWeightHandle(
                client=self,
                version_id=version.version_id,
                staged=staged,
                lease=lease,
            )
            return self._active_handle

    def apply_weight(self, staged: StagedWeightHandle) -> Any:
        """Install a verified local staged version at the caller's safe point."""
        if not isinstance(staged, StagedWeightHandle) or staged._client is not self:
            raise ValueError("staged handle does not belong to this client")
        with self._operation_lock:
            if staged._released:
                raise RuntimeError("staged weight has already been released")
            if staged._applied:
                return staged._apply_result
            primary_error: BaseException | None = None
            try:
                staged._apply_result = self._adapter.apply_weight(staged._staged)
                staged._applied = True
                self._serving_version_id = staged.version_id
                if self._p2p_client is not None:
                    for attempt in range(2):
                        try:
                            self._adapter.publish_weight_version(
                                version_id=staged.version_id,
                                staged=staged._staged,
                                p2p_client=self._p2p_client,
                                worker_id=self.worker_id,
                            )
                            break
                        except Exception:
                            if attempt == 0:
                                logger.warning(
                                    "failed to publish applied version %s as a P2P "
                                    "source; retrying once",
                                    staged.version_id,
                                    exc_info=True,
                                )
                            else:
                                logger.exception(
                                    "failed to publish applied version %s as a P2P "
                                    "source after retry",
                                    staged.version_id,
                                )
                return staged._apply_result
            except BaseException as error:
                primary_error = error
                raise
            finally:
                try:
                    staged._lease.close()
                except grpc.RpcError:
                    if primary_error is None:
                        raise
                    logger.warning(
                        "failed to release version %s lease while handling %s",
                        staged.version_id,
                        type(primary_error).__name__,
                        exc_info=True,
                    )

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
        if self._adapter is not None:
            self._adapter.close()
            self._adapter = None
        if self._p2p_client is not None:
            self._p2p_client.close()
            self._p2p_client = None
        self._refit_strategies = ()
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
        response = self._service.GetWeightVersion(
            refit_pb2.GetWeightVersionRequest(uid=version_id),
            timeout=self._rpc_timeout_seconds,
        )
        if not response.HasField("version"):
            raise RuntimeError("MX GetWeightVersion response is missing version")
        version = _weight_version(response.version)
        if version.state is not WeightVersionState.READY:
            raise RuntimeError(f"weight version {version_id!r} is not READY")
        if version.model_name != self.model_name:
            raise RuntimeError("weight version model_name does not match the generator")
        if self._uses_s3:
            if version.payload_format is not WeightPayloadFormat.XOR_DELTA:
                raise RuntimeError("S3 generator requires XOR_DELTA weight versions")
            if version.base_version_id is None:
                raise RuntimeError("XOR_DELTA version is missing base_version_id")
            if (
                version.version_id != self._serving_version_id
                and version.base_version_id != self._serving_version_id
            ):
                raise RuntimeError(
                    f"weight version base {version.base_version_id!r} does not match "
                    f"serving version {self._serving_version_id!r}"
                )
        return version

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

    def _stage_with_lease(
        self, version: WeightVersion
    ) -> tuple[object, _VersionLease]:
        lease = self._start_version_lease(version.version_id)
        try:
            if self._uses_s3:
                if version.object_storage is None:
                    raise RuntimeError(
                        "The version is missing its object storage source"
                    )
                if version.object_storage.storage_type is not ObjectStorageType.S3:
                    raise RuntimeError("S3 generator requires S3 object storage")
                inputs = GeneratorTransferInputs(
                    version_id=version.version_id,
                    base_version_id=version.base_version_id,
                    layout_signature=version.layout_signature,
                    payload_format=version.payload_format,
                    sources=(),
                    object_storage=version.object_storage,
                )
                last_error: grpc.RpcError | RuntimeError | None = None
                for _ in range(self._max_transfer_attempts):
                    try:
                        return self._adapter.stage_weight(inputs), lease
                    except (grpc.RpcError, RuntimeError) as error:
                        last_error = error
                assert last_error is not None
                raise last_error

            for strategy in self._refit_strategies:
                staged = strategy.stage(version)
                if staged is not None:
                    return staged, lease
            raise RuntimeError(
                f"no usable refit source for weight version {version.version_id!r}"
            )
        except BaseException as primary_error:
            try:
                lease.close()
            except grpc.RpcError:
                logger.warning(
                    "failed to release version %s lease while handling %s",
                    version.version_id,
                    type(primary_error).__name__,
                    exc_info=True,
                )
            raise

    def _release_staged(self, staged: StagedWeightHandle) -> None:
        if staged._client is not self:
            raise ValueError("staged handle does not belong to this client")
        with self._operation_lock:
            if staged._released:
                return
            primary_error: BaseException | None = None
            try:
                self._adapter.release_staged_weight(staged._staged)
            except BaseException as error:
                primary_error = error
                raise
            finally:
                lease_error: grpc.RpcError | None = None
                try:
                    staged._lease.close()
                except grpc.RpcError as error:
                    lease_error = error
                staged._released = True
                if self._active_handle is staged:
                    self._active_handle = None
                if lease_error is not None:
                    if primary_error is None:
                        raise lease_error
                    logger.warning(
                        "failed to release version %s lease while handling %s",
                        staged.version_id,
                        type(primary_error).__name__,
                        exc_info=lease_error,
                    )


__all__ = [
    "ModelExpressGeneratorClient",
    "ModelExpressGeneratorConfig",
    "StagedWeightHandle",
]
