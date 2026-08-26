# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-facing trainer lifecycle for ModelExpress RL refit."""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass
from typing import Any

import grpc
from modelexpress import auth, envs
from modelexpress.client import _get_server_url

from .. import envs as rl_envs
from .. import refit_pb2, refit_pb2_grpc
from ..version import WeightVersionRef
from .adapter import (
    NixlMetadataProvider,
    StagedWeightVersionShardData,
    TrainerEngineAdapter,
    TrainerStagingMode,
    WeightPayloadFormat,
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
    if engine == "MEGATRON":
        from .engines.megatron import MegatronTrainerAdapter

        adapter_type = MegatronTrainerAdapter
    elif engine == "FSDP":
        from .engines.fsdp import FSDPTrainerAdapter

        adapter_type = FSDPTrainerAdapter
    else:
        raise ValueError(f"unsupported MX_TRAINER_ENGINE={engine!r}")
    return adapter_type(manager=manager, nixl_metadata_endpoint=nixl_metadata_endpoint)


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


@dataclass(frozen=True)
class ModelExpressTrainerConfig:
    """Immutable configuration for one rank-local trainer client."""

    # CUDA device used by the rank-local NIXL manager; defaults to LOCAL_RANK.
    device_id: int | None = None
    # NIXL process identity; generated from the rank and a fresh suffix when omitted.
    agent_name: str | None = None
    # Logical model identity; defaults to MODEL_NAME.
    model_name: str | None = None
    # How trainer tensors are staged; defaults to MX_TRAINER_STAGING_MODE.
    staging_mode: TrainerStagingMode | None = None
    # Weight representation published by this client; defaults to MX_WEIGHT_PAYLOAD_FORMAT.
    payload_format: WeightPayloadFormat | None = None
    # Fresh process-lifetime identity; generated when omitted.
    worker_id: str | None = None
    # Address of the central ModelExpress server; uses the standard MX default.
    server_url: str | None = None
    # Worker registration lifetime; defaults to three heartbeat intervals.
    registration_ttl_seconds: int | None = None
    # Deadline applied independently to each control-plane RPC.
    rpc_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Validate explicit settings before client initialization."""
        if self.payload_format is WeightPayloadFormat.UNSPECIFIED:
            raise ValueError("payload_format must be specified")
        if self.registration_ttl_seconds is not None:
            rl_envs.require_positive_int(
                self.registration_ttl_seconds, "registration_ttl_seconds"
            )
        rl_envs.require_positive_float(self.rpc_timeout_seconds, "rpc_timeout_seconds")


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
        self._bound_tensors: Any | None = None
        self._closed = False

    @property
    def source_slot_id(self) -> str:
        """Return the logical source contribution owned by this client."""
        return self._get_adapter().source_slot_id

    def bind_tensors(self, tensors: Any) -> str:
        """Bind the stable engine tensors used by subsequent publications."""
        if self._closed:
            raise RuntimeError("trainer client is closed")
        if tensors is None:
            raise ValueError("tensors must not be None")
        if self._bound_tensors is not None:
            raise RuntimeError("trainer tensors are already bound")
        source_slot_id = self._get_adapter().bind_tensors(tensors)
        self._bound_tensors = tensors
        return source_slot_id

    @classmethod
    def initialize(
        cls,
        config: ModelExpressTrainerConfig,
    ) -> ModelExpressTrainerClient:
        """Initialize a trainer worker and connect it to the MX control plane.

        ModelExpress owns the rank-local transport, manifest service, and engine
        adapter. ``config`` contains only framework-provided settings.
        """
        if not isinstance(config, ModelExpressTrainerConfig):
            raise TypeError("config must be a ModelExpressTrainerConfig")
        model_name = _required(config.model_name or envs.MODEL_NAME or "", "model_name")
        staging_mode = _staging_mode(config.staging_mode)
        payload_format = _payload_format(config.payload_format)
        worker_id = _required(config.worker_id or uuid.uuid4().hex[:8], "worker_id")
        if staging_mode is TrainerStagingMode.UNSPECIFIED:
            raise ValueError("staging_mode must be specified")
        if payload_format is WeightPayloadFormat.UNSPECIFIED:
            raise ValueError("payload_format must be specified")
        registration_ttl_seconds = config.registration_ttl_seconds
        if registration_ttl_seconds is None:
            registration_ttl_seconds = envs.MX_HEARTBEAT_INTERVAL_SECS * 3
        registration_ttl_seconds = rl_envs.require_positive_int(
            registration_ttl_seconds, "registration_ttl_seconds"
        )
        device_id = config.device_id
        if device_id is None:
            local_rank = os.environ.get("LOCAL_RANK")
            if local_rank is None:
                raise ValueError("config.device_id or LOCAL_RANK is required")
            device_id = int(local_rank)
        resources = _TrainerResources.initialize(
            device_id=device_id,
            agent_name=config.agent_name,
        )

        client = cls()
        client.model_name = model_name
        client.staging_mode = staging_mode
        client.payload_format = payload_format
        client.worker_id = worker_id
        client.worker_endpoint = resources.worker_endpoint
        client.server_url = _get_server_url(config.server_url)
        client._adapter = None
        client._manager = resources.manager
        client._nixl_metadata_endpoint = _nixl_metadata_endpoint(resources.manager)
        client._manifest_publisher = resources.manifest_service
        client._resources = resources
        client._registration_ttl_seconds = registration_ttl_seconds
        client._rpc_timeout_seconds = config.rpc_timeout_seconds
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
        if self._closed:
            raise RuntimeError("trainer client is closed")
        if self._adapter is None:
            try:
                adapter = _trainer_adapter(
                    manager=self._manager,
                    nixl_metadata_endpoint=self._nixl_metadata_endpoint,
                )
                self._validate_adapter(adapter, self.staging_mode, self.payload_format)
                self._adapter = adapter
            except Exception:
                self.close()
                raise
        return self._adapter

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
        if self._closed:
            raise RuntimeError("trainer client is closed")
        if not isinstance(version, WeightVersionRef):
            raise TypeError("version must be a WeightVersionRef")
        staged = self._get_adapter().stage_shard(
            tensors=tensors,
            staging_mode=self.staging_mode,
            payload_format=self.payload_format,
        )
        return StagedWeightVersionShard(client=self, version=version, staged=staged)

    def publish_version(self, *, version: WeightVersionRef) -> None:
        """Stage and publish the bound tensors for one weight version."""
        if self._bound_tensors is None:
            raise RuntimeError("bind_tensors() must be called before publish_version()")
        self.stage_shard(version=version, tensors=self._bound_tensors).publish()

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
        if self._closed:
            raise RuntimeError("trainer client is closed")
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
        if self._closed:
            return
        if self._registration_thread is not None:
            self._registration_stop.set()
            self._registration_thread.join()
            self._registration_thread = None
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None
        self._published_shards.clear()
        self._bound_tensors = None
        if self._resources is not None:
            self._resources.close()
            self._resources = None
        self._closed = True

    def __enter__(self) -> ModelExpressTrainerClient:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


__all__ = [
    "ModelExpressTrainerClient",
    "ModelExpressTrainerConfig",
    "StagedWeightVersionShard",
]
