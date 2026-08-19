# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-facing trainer lifecycle for ModelExpress RL refit."""

from __future__ import annotations

import os
import threading
import uuid
from typing import Any

import grpc
from modelexpress import auth, envs
from modelexpress.client import _get_server_url

from .. import envs as rl_envs
from .. import refit_pb2, refit_pb2_grpc
from ..version import WeightVersionRef
from .adapter import (
    CompletionFence,
    NixlMetadataProvider,
    StagedWeightVersionShardData,
    TrainerEngineAdapter,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionShardManifestPublisher,
)
from .resources import _TrainerResources


def _required(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} is required")
    return value


def _trainer_adapter(
    *,
    manager: NixlMetadataProvider,
    nixl_metadata_endpoint: str,
) -> TrainerEngineAdapter:
    engine = rl_envs.MX_TRAINER_ENGINE
    if engine != "MEGATRON":
        raise ValueError(f"unsupported MX_TRAINER_ENGINE={engine!r}")

    from .engines.megatron import MegatronTrainerAdapter

    return MegatronTrainerAdapter(
        manager=manager,
        nixl_metadata_endpoint=nixl_metadata_endpoint,
    )


def _nixl_metadata_endpoint(manager: NixlMetadataProvider) -> str:
    host = _required(envs.MX_WORKER_HOST, "MX_WORKER_HOST")
    if manager.listen_port is None:
        raise ValueError("NIXL manager must have a metadata listen port")
    return f"{host}:{manager.listen_port}"


def _staging_mode(value: TrainerStagingMode | None) -> TrainerStagingMode:
    try:
        return value or TrainerStagingMode(rl_envs.MX_TRAINER_STAGING_MODE)
    except ValueError as error:
        raise ValueError(
            f"invalid MX_TRAINER_STAGING_MODE={rl_envs.MX_TRAINER_STAGING_MODE!r}"
        ) from error


def _payload_format(value: WeightPayloadFormat | None) -> WeightPayloadFormat:
    try:
        return value or WeightPayloadFormat(rl_envs.MX_WEIGHT_PAYLOAD_FORMAT)
    except ValueError as error:
        raise ValueError(
            f"invalid MX_WEIGHT_PAYLOAD_FORMAT={rl_envs.MX_WEIGHT_PAYLOAD_FORMAT!r}"
        ) from error


class StagedWeightVersionShard:
    """One immutable rank-local shard staged for a global weight version."""

    def __init__(
        self,
        *,
        client: ModelExpressTrainerClient,
        version: WeightVersionRef,
        staged: StagedWeightVersionShardData,
    ) -> None:
        self._client = client
        self._version = version
        self._staged = staged
        self._publish_lock = threading.Lock()
        self._published = False

    @property
    def source_reuse_ready(self) -> CompletionFence:
        """Fence after which the trainer may reuse or mutate its input tensors."""
        return self._staged.source_reuse_ready

    def publish(self) -> None:
        """Publish this staged shard; repeated calls are idempotent."""
        with self._publish_lock:
            if self._published:
                return
            self._client._publish_staged_shard(
                version=self._version,
                staged=self._staged,
            )
            self._published = True


class ModelExpressTrainerClient:
    """Rank-local capture, staging, and publication client for trainer actors."""

    def __init__(self) -> None:
        self._channel: grpc.Channel | None = None
        self._stub: refit_pb2_grpc.RefitServiceStub | None = None
        self._published_shards: dict[str, list[StagedWeightVersionShardData]] = {}
        self._registration_stop = threading.Event()
        self._registration_thread: threading.Thread | None = None
        self._adapter: TrainerEngineAdapter | None = None
        self._resources: _TrainerResources | None = None

    @property
    def source_slot_id(self) -> str:
        """Return the logical source contribution owned by this client."""
        return self._get_adapter().source_slot_id

    @classmethod
    def initialize(
        cls,
        *,
        manager: NixlMetadataProvider | None = None,
        manifest_publisher: WeightVersionShardManifestPublisher | None = None,
        device_id: int | None = None,
        agent_name: str | None = None,
        model_name: str | None = None,
        staging_mode: TrainerStagingMode | None = None,
        payload_format: WeightPayloadFormat | None = None,
        worker_endpoint: str | None = None,
        worker_id: str | None = None,
        server_url: str | None = None,
        registration_ttl_seconds: int | None = None,
        rpc_timeout_seconds: float = 30.0,
    ) -> ModelExpressTrainerClient:
        """Initialize a trainer worker and connect it to the MX control plane.

        ``worker_endpoint`` is this trainer's peer-reachable manifest service.
        ``server_url`` is the central ModelExpress ``RefitService`` address.
        """
        model_name = _required(model_name or envs.MODEL_NAME or "", "model_name")
        staging_mode = _staging_mode(staging_mode)
        payload_format = _payload_format(payload_format)
        worker_id = _required(worker_id or uuid.uuid4().hex[:8], "worker_id")
        if staging_mode is TrainerStagingMode.UNSPECIFIED:
            raise ValueError("staging_mode must be specified")
        if payload_format is WeightPayloadFormat.UNSPECIFIED:
            raise ValueError("payload_format must be specified")
        if registration_ttl_seconds is None:
            registration_ttl_seconds = envs.MX_HEARTBEAT_INTERVAL_SECS * 3
        if registration_ttl_seconds <= 0:
            raise ValueError("registration_ttl_seconds must be positive")
        if rpc_timeout_seconds <= 0:
            raise ValueError("rpc_timeout_seconds must be positive")
        if (manager is None) != (manifest_publisher is None):
            raise ValueError("manager and manifest_publisher must be provided together")

        resources = None
        adapter = None
        if manager is None:
            if device_id is None:
                local_rank = os.environ.get("LOCAL_RANK")
                if local_rank is None:
                    raise ValueError("device_id or LOCAL_RANK is required")
                device_id = int(local_rank)
            resources = _TrainerResources.initialize(
                device_id=device_id,
                agent_name=agent_name,
            )
            manager = resources.manager
            manifest_publisher = resources.manifest_service
            worker_endpoint = resources.worker_endpoint
        else:
            worker_endpoint = _required(
                worker_endpoint
                or (
                    f"{envs.MX_WORKER_HOST}:{envs.MX_WORKER_GRPC_PORT}"
                    if envs.MX_WORKER_HOST
                    else ""
                ),
                "worker_endpoint",
            )
            adapter = _trainer_adapter(
                manager=manager,
                nixl_metadata_endpoint=_nixl_metadata_endpoint(manager),
            )
            cls._validate_adapter(adapter, staging_mode, payload_format)

        assert manager is not None
        assert manifest_publisher is not None

        client = cls()
        client.model_name = model_name
        client.staging_mode = staging_mode
        client.payload_format = payload_format
        client.worker_id = worker_id
        client.worker_endpoint = worker_endpoint
        client.server_url = _get_server_url(server_url)
        client._adapter = adapter
        client._manager = manager
        client._nixl_metadata_endpoint = _nixl_metadata_endpoint(manager)
        client._manifest_publisher = manifest_publisher
        client._resources = resources
        client._registration_ttl_seconds = registration_ttl_seconds
        client._rpc_timeout_seconds = rpc_timeout_seconds
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

    @staticmethod
    def _validate_adapter(
        adapter: TrainerEngineAdapter,
        staging_mode: TrainerStagingMode,
        payload_format: WeightPayloadFormat,
    ) -> None:
        if staging_mode not in adapter.supported_staging_modes:
            raise ValueError(
                f"adapter does not support staging mode {staging_mode.value}"
            )
        if payload_format not in adapter.supported_payload_formats:
            raise ValueError(
                f"adapter does not support payload format {payload_format.value}"
            )

    def _get_adapter(self) -> TrainerEngineAdapter:
        if self._adapter is None:
            adapter = _trainer_adapter(
                manager=self._manager,
                nixl_metadata_endpoint=self._nixl_metadata_endpoint,
            )
            self._validate_adapter(adapter, self.staging_mode, self.payload_format)
            self._adapter = adapter
        return self._adapter

    def register_tensors(self, tensors: dict[str, Any]) -> None:
        """Register stable trainer tensors with the selected transport."""
        self._manager.register_tensors(tensors)

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
                    role=refit_pb2.WORKER_ROLE_TRAINER,
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
            except grpc.RpcError:
                # A later renewal retries after transient control-plane failure.
                continue

    def stage_shard(
        self,
        *,
        version: WeightVersionRef,
        tensors: Any,
    ) -> StagedWeightVersionShard:
        """Capture one immutable rank-local shard for ``version``."""
        if not isinstance(version, WeightVersionRef):
            raise TypeError("version must be a WeightVersionRef")
        staged = self._get_adapter().stage_shard(
            tensors=tensors,
            staging_mode=self.staging_mode,
            payload_format=self.payload_format,
        )
        return StagedWeightVersionShard(client=self, version=version, staged=staged)

    def _publish_staged_shard(
        self,
        *,
        version: WeightVersionRef,
        staged: StagedWeightVersionShardData,
    ) -> None:
        source_slot_id = self._get_adapter().source_slot_id
        staged.publish_ready.wait()
        manifest_endpoint = self._manifest_publisher.publish_manifest(
            version_id=version.version_id,
            source_slot_id=source_slot_id,
            manifest=staged.manifest,
        )
        shard = refit_pb2.WeightVersionShard(
            version_id=version.version_id,
            source_slot_id=source_slot_id,
            worker_id=self.worker_id,
            tensor_count=staged.manifest.tensor_count,
            total_bytes=staged.manifest.total_bytes,
            manifest_digest=staged.manifest.digest,
            manifest_endpoint=_required(manifest_endpoint, "manifest_endpoint"),
            transport=staged.manifest.transport,
        )
        self._service.CreateWeightVersionShard(
            refit_pb2.CreateWeightVersionShardRequest(shard=shard),
            timeout=self._rpc_timeout_seconds,
        )
        # Keep the adapter-owned buffers alive while the published version can
        # still be selected as a source. Eviction/release is a later lifecycle
        # operation, not the staged handle's Python object lifetime.
        self._published_shards.setdefault(version.version_id, []).append(staged)

    def release_version(self, *, version: WeightVersionRef) -> None:
        """Withdraw this worker's shard after the version is retired.

        The framework must call this only after the control plane has moved the
        version to ``RELEASING``. Once the shard is deleted, ModelExpress no
        longer advertises this worker's buffers as a transfer source and an
        in-place trainer may resume mutating them.
        """
        if not isinstance(version, WeightVersionRef):
            raise TypeError("version must be a WeightVersionRef")
        staged = self._published_shards.get(version.version_id)
        if staged is None:
            return
        self._service.DeleteWeightVersionShard(
            refit_pb2.DeleteWeightVersionShardRequest(
                version_id=version.version_id,
                source_slot_id=self._get_adapter().source_slot_id,
                worker_id=self.worker_id,
            ),
            timeout=self._rpc_timeout_seconds,
        )
        del self._published_shards[version.version_id]

    def close(self) -> None:
        """Close the underlying gRPC channel."""
        if self._registration_thread is not None:
            self._registration_stop.set()
            self._registration_thread.join()
            self._registration_thread = None
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None
        self._published_shards.clear()
        if self._resources is not None:
            self._resources.close()
            self._resources = None

    def __enter__(self) -> ModelExpressTrainerClient:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


__all__ = [
    "ModelExpressTrainerClient",
    "StagedWeightVersionShard",
]
