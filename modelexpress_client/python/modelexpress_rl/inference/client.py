# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rank-local generator lifecycle for ModelExpress RL refit."""

from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import grpc

from modelexpress import auth, envs
from modelexpress.client import _get_server_url
from modelexpress_rl import envs as rl_envs
from modelexpress_rl.train import WeightPayloadFormat
from modelexpress_rl.version import WeightVersionRef

from .. import refit_pb2, refit_pb2_grpc
from .adapter import (
    GeneratorEngineContext,
    GeneratorEngineAdapter,
    GeneratorInstallationMode,
    GeneratorSource,
    GeneratorTransferInputs,
)
from .engines import _create_generator_adapter

logger = logging.getLogger("modelexpress_rl.inference.client")


def _required(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} is required")
    return value


def _installation_mode(
    value: GeneratorInstallationMode | None,
) -> GeneratorInstallationMode:
    try:
        return value or GeneratorInstallationMode(
            rl_envs.MX_GENERATOR_INSTALLATION_MODE
        )
    except ValueError as error:
        raise ValueError(
            "invalid MX_GENERATOR_INSTALLATION_MODE="
            f"{rl_envs.MX_GENERATOR_INSTALLATION_MODE!r}"
        ) from error


def _payload_format(value: WeightPayloadFormat | None) -> WeightPayloadFormat:
    try:
        return value or WeightPayloadFormat(rl_envs.MX_WEIGHT_PAYLOAD_FORMAT)
    except ValueError as error:
        raise ValueError(
            f"invalid MX_WEIGHT_PAYLOAD_FORMAT={rl_envs.MX_WEIGHT_PAYLOAD_FORMAT!r}"
        ) from error


@dataclass(frozen=True)
class ModelExpressGeneratorConfig:
    """Immutable configuration for one rank-local generator client."""

    engine_context: GeneratorEngineContext
    model_name: str | None = None
    installation_mode: GeneratorInstallationMode | None = None
    payload_format: WeightPayloadFormat | None = None
    worker_endpoint: str | None = None
    worker_id: str | None = None
    server_url: str | None = None
    registration_ttl_seconds: int | None = None
    lease_ttl_seconds: int | None = None
    max_transfer_attempts: int = 3
    rpc_timeout_seconds: float = 30.0


class StagedWeightHandle:
    """Local verified staging buffers for one exact WeightVersion."""

    def __init__(
        self,
        *,
        client: ModelExpressGeneratorClient,
        version_id: str,
        staged: Any,
    ) -> None:
        self._client = client
        self.version_id = version_id
        self._staged = staged
        self._applied = False
        self._apply_result: Any = None
        self._released = False

    def wait(self) -> None:
        """Return after staging is complete; the synchronous client is already done."""

    def release(self) -> None:
        """Release local staging buffers; repeated calls are idempotent."""
        self._client._release_staged(self)


class ModelExpressGeneratorClient:
    """Synchronous rank-local generator client for exact-version refit."""

    def __init__(self) -> None:
        self._channel: grpc.Channel | None = None
        self._stub: refit_pb2_grpc.RefitServiceStub | None = None
        self._registration_stop = threading.Event()
        self._registration_thread: threading.Thread | None = None
        self._operation_lock = threading.RLock()
        self._active_handle: StagedWeightHandle | None = None
        self._cached_plan: Any = None
        self._cached_fingerprint: tuple | None = None
        self._serving_version_id: str | None = None
        self._adapter: GeneratorEngineAdapter | None = None
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
        installation_mode = _installation_mode(config.installation_mode)
        payload_format = _payload_format(config.payload_format)
        worker_endpoint = _required(
            config.worker_endpoint
            or (
                f"{envs.MX_WORKER_HOST}:{envs.MX_WORKER_GRPC_PORT}"
                if envs.MX_WORKER_HOST
                else ""
            ),
            "worker_endpoint",
        )
        worker_id = _required(config.worker_id or uuid.uuid4().hex[:8], "worker_id")
        server_url = _get_server_url(config.server_url)
        if installation_mode is GeneratorInstallationMode.UNSPECIFIED:
            raise ValueError("installation_mode must be specified")
        if payload_format is WeightPayloadFormat.UNSPECIFIED:
            raise ValueError("payload_format must be specified")
        registration_ttl_seconds = config.registration_ttl_seconds
        if registration_ttl_seconds is None:
            registration_ttl_seconds = envs.MX_HEARTBEAT_INTERVAL_SECS * 3
        lease_ttl_seconds = config.lease_ttl_seconds
        if lease_ttl_seconds is None:
            lease_ttl_seconds = registration_ttl_seconds
        if registration_ttl_seconds <= 0:
            raise ValueError("registration_ttl_seconds must be positive")
        if lease_ttl_seconds <= 0:
            raise ValueError("lease_ttl_seconds must be positive")
        if config.max_transfer_attempts <= 0:
            raise ValueError("max_transfer_attempts must be positive")
        if config.rpc_timeout_seconds <= 0:
            raise ValueError("rpc_timeout_seconds must be positive")

        adapter = _create_generator_adapter(
            engine=rl_envs.MX_GENERATOR_ENGINE,
            engine_context=config.engine_context,
            worker_id=worker_id,
        )
        try:
            if installation_mode not in adapter.supported_installation_modes:
                raise ValueError(
                    "adapter does not support installation mode "
                    f"{installation_mode.value}"
                )
            if payload_format not in adapter.supported_payload_formats:
                raise ValueError(
                    f"adapter does not support payload format {payload_format.value}"
                )
        except Exception:
            adapter.close()
            raise

        client = cls()
        client.model_name = model_name
        client.installation_mode = installation_mode
        client.payload_format = payload_format
        client.worker_endpoint = worker_endpoint
        client.worker_id = worker_id
        client.server_url = server_url
        client._registration_ttl_seconds = registration_ttl_seconds
        client._lease_ttl_seconds = lease_ttl_seconds
        client._max_transfer_attempts = config.max_transfer_attempts
        client._rpc_timeout_seconds = config.rpc_timeout_seconds
        client._adapter = adapter
        try:
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
        """Synchronously transfer and verify a STAGED full-weight version."""
        if self.installation_mode is not GeneratorInstallationMode.STAGED:
            raise RuntimeError("stage_weight is available only in STAGED mode")
        if not isinstance(version, WeightVersionRef):
            raise TypeError("version must be a WeightVersionRef")
        with self._operation_lock:
            if self._active_handle is not None:
                if self._active_handle.version_id == version.version_id:
                    return self._active_handle
                raise RuntimeError("another generator update is still active")
            ready = self._get_ready_version(version.version_id)
            staged = self._stage_with_lease(ready)
            self._active_handle = StagedWeightHandle(
                client=self,
                version_id=version.version_id,
                staged=staged,
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
            staged._apply_result = self._adapter.apply_weight(staged._staged)
            staged._applied = True
            self._serving_version_id = staged.version_id
            return staged._apply_result

    def update_weight(self, *, version: WeightVersionRef) -> Any:
        """Update live destinations in DIRECT mode."""
        del version
        raise NotImplementedError("DIRECT installation is not implemented")

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
                    endpoint=self.worker_endpoint,
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

    def _get_ready_version(self, version_id: str) -> refit_pb2.WeightVersion:
        version = self._service.GetWeightVersion(
            refit_pb2.GetWeightVersionRequest(uid=version_id),
            timeout=self._rpc_timeout_seconds,
        )
        if version.state != refit_pb2.WEIGHT_VERSION_STATE_READY:
            raise RuntimeError(f"weight version {version_id!r} is not READY")
        if version.model_name != self.model_name:
            raise RuntimeError("weight version model_name does not match the generator")
        if version.payload_format != self._proto_payload_format:
            raise RuntimeError(
                "weight version payload_format does not match the generator"
            )
        return version

    @property
    def _proto_payload_format(self) -> int:
        return {
            WeightPayloadFormat.FULL_TENSOR: refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_TENSOR,
            WeightPayloadFormat.XOR_DELTA: refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA,
        }[self.payload_format]

    def _fetch_manifest(self, shard) -> bytes:
        with grpc.insecure_channel(shard.manifest_endpoint) as channel:
            response = refit_pb2_grpc.RefitWorkerServiceStub(
                channel
            ).GetWeightVersionShardManifest(
                refit_pb2.GetWeightVersionShardManifestRequest(
                    version_id=shard.version_id,
                    source_slot_id=shard.source_slot_id,
                ),
                timeout=self._rpc_timeout_seconds,
            )
        digest = hashlib.sha256(response.manifest).hexdigest()
        if (
            response.manifest_digest != shard.manifest_digest
            or digest != shard.manifest_digest
        ):
            raise RuntimeError(
                f"manifest digest mismatch for source slot {shard.source_slot_id!r}"
            )
        return response.manifest

    def _discover_sources(
        self,
        version: refit_pb2.WeightVersion,
    ) -> GeneratorTransferInputs:
        response = self._service.ListWeightVersionShards(
            refit_pb2.ListWeightVersionShardsRequest(version_id=version.uid),
            timeout=self._rpc_timeout_seconds,
        )
        candidates = defaultdict(list)
        for shard in response.shards:
            candidates[shard.source_slot_id].append(shard)

        selected = []
        for source_slot_id in version.expected_source_slots:
            failures = []
            for shard in sorted(
                candidates[source_slot_id], key=lambda item: item.worker_id
            ):
                try:
                    manifest = self._fetch_manifest(shard)
                except (grpc.RpcError, RuntimeError) as error:
                    failures.append(str(error))
                    continue
                selected.append(
                    GeneratorSource(
                        source_slot_id=source_slot_id,
                        worker_id=shard.worker_id,
                        manifest_endpoint=shard.manifest_endpoint,
                        manifest_digest=shard.manifest_digest,
                        transport=shard.transport,
                        manifest=manifest,
                    )
                )
                break
            else:
                detail = f": {failures[-1]}" if failures else ""
                raise RuntimeError(
                    f"no usable source for required slot {source_slot_id!r}{detail}"
                )

        return GeneratorTransferInputs(
            version_id=version.uid,
            layout_signature=version.layout_signature,
            payload_format=self.payload_format,
            sources=tuple(selected),
        )

    def _transfer_plan(self, inputs: GeneratorTransferInputs) -> Any:
        reusable = (
            self._cached_plan is not None
            and self._cached_fingerprint == inputs.physical_fingerprint
            and self._adapter.validate_transfer_plan(self._cached_plan, inputs)
        )
        if not reusable:
            self._cached_plan = self._adapter.create_transfer_plan(inputs)
            self._cached_fingerprint = inputs.physical_fingerprint
        return self._cached_plan

    def _register_lease(self, version_id: str):
        return self._service.RegisterVersionLease(
            refit_pb2.RegisterVersionLeaseRequest(
                version_id=version_id,
                worker_id=self.worker_id,
                ttl_seconds=self._lease_ttl_seconds,
            ),
            timeout=self._rpc_timeout_seconds,
        )

    def _stage_with_lease(self, version: refit_pb2.WeightVersion) -> Any:
        lease = self._register_lease(version.uid)
        stop = threading.Event()

        def renew() -> None:
            interval_seconds = max(self._lease_ttl_seconds / 3, 0.1)
            while not stop.wait(interval_seconds):
                try:
                    self._register_lease(version.uid)
                except grpc.RpcError as error:
                    logger.warning(
                        "version %s lease renewal failed: %s",
                        version.uid,
                        error,
                    )
                    continue
                except Exception:
                    logger.exception(
                        "unexpected version %s lease renewal failure",
                        version.uid,
                    )
                    continue

        renewal = threading.Thread(
            target=renew,
            name=f"modelexpress-refit-lease-{self.worker_id}",
            daemon=True,
        )
        renewal.start()
        primary_error: BaseException | None = None
        try:
            last_error: grpc.RpcError | RuntimeError | None = None
            for _attempt in range(self._max_transfer_attempts):
                try:
                    inputs = self._discover_sources(version)
                    return self._adapter.stage_weight(self._transfer_plan(inputs))
                except (grpc.RpcError, RuntimeError) as error:
                    last_error = error
                    # A failed transfer may have invalidated transport state even
                    # when the source metadata fingerprint is unchanged.
                    self._cached_plan = None
                    self._cached_fingerprint = None
            assert last_error is not None
            raise last_error
        except BaseException as error:
            primary_error = error
            raise
        finally:
            stop.set()
            renewal.join()
            try:
                self._service.DeleteVersionLease(
                    refit_pb2.DeleteVersionLeaseRequest(
                        version_id=version.uid,
                        lease_id=lease.lease_id,
                        worker_id=self.worker_id,
                    ),
                    timeout=self._rpc_timeout_seconds,
                )
            except grpc.RpcError:
                if primary_error is None:
                    raise
                logger.warning(
                    "failed to release lease %s while handling %s",
                    lease.lease_id,
                    type(primary_error).__name__,
                    exc_info=True,
                )

    def _release_staged(self, staged: StagedWeightHandle) -> None:
        if staged._client is not self:
            raise ValueError("staged handle does not belong to this client")
        with self._operation_lock:
            if staged._released:
                return
            self._adapter.release_staged_weight(staged._staged)
            staged._released = True
            if self._active_handle is staged:
                self._active_handle = None


__all__ = [
    "ModelExpressGeneratorClient",
    "ModelExpressGeneratorConfig",
    "StagedWeightHandle",
]
