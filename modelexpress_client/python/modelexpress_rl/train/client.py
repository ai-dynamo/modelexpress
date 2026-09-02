# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-facing trainer lifecycle for ModelExpress RL refit."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import grpc
import torch
from modelexpress import auth, envs
from modelexpress.client import _get_server_url

from .. import envs as rl_envs
from .. import refit_pb2, refit_pb2_grpc
from ..object_storage import ObjectStorageType
from ..version import WeightVersionRef
from .adapter import (
    TrainerStagingMode,
    WeightPayloadFormat,
)
from .context import TrainerEngineContext
from .runtime import PublicationArtifact, TrainerRuntime


def _required(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} is required")
    return value


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
class ObjectStorageConfig:
    """Object-storage and seed-base settings for canonical publication."""

    storage_type: ObjectStorageType
    uri_prefix: str
    initial_base_version_id: str
    seed_checkpoint_path: str | Path
    endpoint_url: str | None = None
    region_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.storage_type, ObjectStorageType):
            raise TypeError("storage_type must be an ObjectStorageType")
        _required(self.uri_prefix, "object_storage.uri_prefix")
        _required(
            self.initial_base_version_id,
            "object_storage.initial_base_version_id",
        )
        if not str(self.seed_checkpoint_path).strip():
            raise ValueError("object_storage.seed_checkpoint_path is required")

@dataclass(frozen=True)
class ModelExpressTrainerConfig:
    """Immutable configuration for one rank-local trainer client."""

    device_id: int | None = None
    agent_name: str | None = None
    model_name: str | None = None
    staging_mode: TrainerStagingMode | None = None
    payload_format: WeightPayloadFormat | None = None
    worker_id: str | None = None
    server_url: str | None = None
    registration_ttl_seconds: int | None = None
    rpc_timeout_seconds: float = 30.0
    process_group: Any | None = None
    object_storage: ObjectStorageConfig | None = None
    engine_context: TrainerEngineContext | None = None

    def __post_init__(self) -> None:
        if self.payload_format is WeightPayloadFormat.UNSPECIFIED:
            raise ValueError("payload_format must be specified")
        if self.registration_ttl_seconds is not None:
            rl_envs.require_positive_int(
                self.registration_ttl_seconds, "registration_ttl_seconds"
            )
        rl_envs.require_positive_float(self.rpc_timeout_seconds, "rpc_timeout_seconds")


class StagedWeightVersionShard:
    """One immutable rank-local artifact staged for a global weight version."""

    def __init__(
        self,
        *,
        client: ModelExpressTrainerClient,
        version: WeightVersionRef,
        staged: PublicationArtifact,
    ) -> None:
        self._client = client
        self._version = version
        self._staged = staged
        self._publish_lock = threading.Lock()
        self._published = False

    def publish(self) -> None:
        with self._publish_lock:
            if self._published:
                return
            self._client._publish_staged_shard(
                version=self._version, staged=self._staged
            )
            self._published = True


class ModelExpressTrainerClient:
    """Rank-local capture, staging, and publication client for trainer actors."""

    def __init__(self) -> None:
        self._channel: grpc.Channel | None = None
        self._stub: refit_pb2_grpc.RefitServiceStub | None = None
        self._registration_stop = threading.Event()
        self._registration_thread: threading.Thread | None = None
        self._runtime: TrainerRuntime | None = None
        self._closed = False

    @classmethod
    def initialize(
        cls, config: ModelExpressTrainerConfig
    ) -> ModelExpressTrainerClient:
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
        ttl = config.registration_ttl_seconds
        if ttl is None:
            ttl = envs.MX_HEARTBEAT_INTERVAL_SECS * 3
        ttl = rl_envs.require_positive_int(ttl, "registration_ttl_seconds")

        use_storage = config.object_storage is not None
        if use_storage:
            assert config.object_storage is not None
            if config.object_storage.storage_type is not ObjectStorageType.S3:
                raise ValueError("only S3 object storage is currently supported")
            if (
                staging_mode is not TrainerStagingMode.WRITE_TO_STORAGE
                or payload_format is not WeightPayloadFormat.XOR_DELTA
            ):
                raise ValueError(
                    "object storage publication requires WRITE_TO_STORAGE and XOR_DELTA"
                )
        elif staging_mode is TrainerStagingMode.WRITE_TO_STORAGE:
            raise ValueError("WRITE_TO_STORAGE requires config.object_storage")

        client = cls()
        client.model_name = model_name
        client.staging_mode = staging_mode
        client.payload_format = payload_format
        client.worker_id = worker_id
        client.server_url = _get_server_url(config.server_url)
        client._registration_ttl_seconds = ttl
        client._rpc_timeout_seconds = config.rpc_timeout_seconds
        try:
            client._runtime = TrainerRuntime.initialize(
                engine_context=config.engine_context,
                device_id=config.device_id,
                agent_name=config.agent_name,
                model_name=model_name,
                staging_mode=staging_mode,
                payload_format=payload_format,
                worker_id=worker_id,
                object_storage=config.object_storage,
                process_group=config.process_group,
                service=lambda: client._service,
                rpc_timeout_seconds=config.rpc_timeout_seconds,
            )
            client.worker_endpoint = client._runtime.worker_endpoint
            if client._runtime.requires_registration:
                client._register_worker()
                thread = threading.Thread(
                    target=client._renew_worker_registration,
                    name=f"modelexpress-refit-renew-{worker_id}",
                    daemon=True,
                )
                thread.start()
                client._registration_thread = thread
        except Exception:
            client.close()
            raise
        return client

    def _active_runtime(self) -> TrainerRuntime:
        if self._closed:
            raise RuntimeError("trainer client is closed")
        assert self._runtime is not None
        return self._runtime

    @property
    def source_slot_id(self) -> str:
        return self._active_runtime().source_slot_id

    def prepare_delta_base(
        self, *, hf_tensor_iter: Iterable[list[tuple[str, torch.Tensor]]]
    ) -> None:
        self._active_runtime().prepare_delta_base(hf_tensor_iter=hf_tensor_iter)

    def bind_tensors(self, tensors: Any) -> str:
        return self._active_runtime().bind_tensors(tensors)

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
        interval = max(self._registration_ttl_seconds / 3, 0.1)
        while not self._registration_stop.wait(interval):
            try:
                self._register_worker()
            except grpc.RpcError:
                continue

    def stage_shard(
        self,
        *,
        version: WeightVersionRef,
        tensors: Any = None,
        hf_tensor_iter: Iterable[list[tuple[str, torch.Tensor]]] | None = None,
    ) -> StagedWeightVersionShard:
        if self._closed:
            raise RuntimeError("trainer client is closed")
        if not isinstance(version, WeightVersionRef):
            raise TypeError("version must be a WeightVersionRef")
        staged = self._active_runtime().stage(
            version=version,
            tensors=tensors,
            hf_tensor_iter=hf_tensor_iter,
        )
        return StagedWeightVersionShard(client=self, version=version, staged=staged)

    def publish_version(self, *, version: WeightVersionRef) -> None:
        self._active_runtime().publish_bound(version=version)

    def _publish_staged_shard(
        self, *, version: WeightVersionRef, staged: PublicationArtifact
    ) -> None:
        self._active_runtime().publish(version=version, staged=staged)

    def release_version(self, *, version: WeightVersionRef) -> None:
        if self._closed:
            raise RuntimeError("trainer client is closed")
        if not isinstance(version, WeightVersionRef):
            raise TypeError("version must be a WeightVersionRef")
        self._active_runtime().release(version=version)

    def pop_metrics(self) -> dict[str, int | float]:
        return self._active_runtime().pop_metrics()

    def close(self) -> None:
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
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        self._closed = True

    def __enter__(self) -> ModelExpressTrainerClient:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


__all__ = [
    "ModelExpressTrainerClient",
    "ModelExpressTrainerConfig",
    "ObjectStorageConfig",
    "StagedWeightVersionShard",
]
