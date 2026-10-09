# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-orchestrator client for the ModelExpress refit control plane."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import grpc
from modelexpress import auth
from modelexpress.client import _get_server_url

from . import refit_pb2, refit_pb2_grpc
from .object_storage import ObjectStorageSource, ObjectStorageType
from .train import WeightPayloadFormat
from .version import TrainerTensorsMetadata, WeightVersionRef


class WeightVersionState(str, Enum):
    """Public lifecycle state of an immutable weight version."""

    STAGING = "STAGING"
    READY = "READY"
    RELEASING = "RELEASING"


@dataclass(frozen=True)
class TrainerMesh:
    """Expected logical shards and physical trainers for one refit group."""

    mesh_id: str
    model_name: str
    generation: int
    workers: dict[str, TrainerTensorsMetadata]


@dataclass(frozen=True)
class WeightVersion:
    """Framework-facing representation of one MX weight version."""

    version_id: str
    model_name: str
    payload_format: WeightPayloadFormat
    layout_signature: str
    state: WeightVersionState
    created_at_unix_ms: int
    trainer_mesh_generation: int
    base_version_id: str | None = None
    object_storage: ObjectStorageSource | None = None
    trainer_mesh_id: str | None = None
    version_number: int | None = None

    def __post_init__(self) -> None:
        generation = self.trainer_mesh_generation
        if type(generation) is not int or not 0 <= generation < 2**64:
            raise ValueError("trainer_mesh_generation must be a uint64 integer")
        if self.trainer_mesh_id is not None and generation == 0:
            raise ValueError(
                "mesh-backed WeightVersion requires a positive trainer_mesh_generation; recreate the version"
            )
        if self.trainer_mesh_id is None and generation != 0:
            raise ValueError("trainer_mesh_generation requires trainer_mesh_id")

    @property
    def ref(self) -> WeightVersionRef:
        """Return the opaque reference passed to trainer and generator actors."""
        return WeightVersionRef(self.version_id)


def _required(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} is required")
    return value


def _mesh(mesh: refit_pb2.TrainerMesh) -> TrainerMesh:
    return TrainerMesh(
        mesh_id=mesh.mesh_id,
        model_name=mesh.model_name,
        generation=mesh.generation,
        workers={
            worker_id: TrainerTensorsMetadata(
                logical_shard_id=metadata.logical_shard_id,
                metadata_endpoint=metadata.metadata_endpoint,
            )
            for worker_id, metadata in mesh.workers.items()
        },
    )


def _response_mesh(response, rpc_name: str) -> TrainerMesh:
    if not response.HasField("mesh"):
        raise RuntimeError(f"MX {rpc_name} response is missing mesh")
    return _mesh(response.mesh)


def _mesh_workers(
    workers: dict[str, TrainerTensorsMetadata],
) -> dict[str, refit_pb2.TrainerTensorsMetadata]:
    if not isinstance(workers, dict):
        raise TypeError("workers must be a dict")
    if not workers:
        raise ValueError("workers must not be empty")
    encoded = {}
    for worker_id, metadata in workers.items():
        if not isinstance(metadata, TrainerTensorsMetadata):
            raise TypeError("workers values must be TrainerTensorsMetadata")
        encoded[_required(worker_id, "worker_id")] = refit_pb2.TrainerTensorsMetadata(
            logical_shard_id=_required(metadata.logical_shard_id, "logical_shard_id"),
            metadata_endpoint=_required(metadata.metadata_endpoint, "metadata_endpoint"),
        )
    return encoded


def _weight_version(version: refit_pb2.WeightVersion) -> WeightVersion:
    payload_formats = {
        refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_TENSOR: WeightPayloadFormat.FULL_TENSOR,
        refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA: WeightPayloadFormat.XOR_DELTA,
        refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_HF_CHECKPOINT: (
            WeightPayloadFormat.FULL_HF_CHECKPOINT
        ),
    }
    states = {
        refit_pb2.WEIGHT_VERSION_STATE_STAGING: WeightVersionState.STAGING,
        refit_pb2.WEIGHT_VERSION_STATE_READY: WeightVersionState.READY,
        refit_pb2.WEIGHT_VERSION_STATE_RELEASING: WeightVersionState.RELEASING,
    }
    storage_types = {
        refit_pb2.OBJECT_STORAGE_TYPE_S3: ObjectStorageType.S3,
        refit_pb2.OBJECT_STORAGE_TYPE_AZURE: ObjectStorageType.AZURE,
        refit_pb2.OBJECT_STORAGE_TYPE_GCS: ObjectStorageType.GCS,
    }
    try:
        payload_format = payload_formats[version.payload_format]
        state = states[version.state]
    except KeyError as error:
        raise RuntimeError("MX returned an unspecified WeightVersion enum") from error
    object_storage = None
    if version.HasField("object_storage"):
        try:
            storage_type = storage_types[version.object_storage.storage_type]
        except KeyError as error:
            raise RuntimeError(
                "MX returned an unspecified object storage type"
            ) from error
        object_storage = ObjectStorageSource(
            storage_type=storage_type,
            uri=version.object_storage.uri,
        )
    return WeightVersion(
        trainer_mesh_generation=version.trainer_mesh_generation,
        version_id=version.uid,
        model_name=version.model_name,
        payload_format=payload_format,
        base_version_id=(
            version.base_version_id if version.HasField("base_version_id") else None
        ),
        object_storage=object_storage,
        layout_signature=version.layout_signature,
        state=state,
        created_at_unix_ms=version.created_at_unix_ms,
        trainer_mesh_id=(
            version.trainer_mesh_id if version.HasField("trainer_mesh_id") else None
        ),
        version_number=version.version_number if version.HasField("version_number") else None,
    )


def _response_version(response, rpc_name: str) -> WeightVersion:
    if not response.HasField("version"):
        raise RuntimeError(f"MX {rpc_name} response is missing version")
    return _weight_version(response.version)


class ModelExpressControlClient:
    """Synchronous public client used by an RL framework orchestrator."""

    def __init__(self) -> None:
        self._channel: grpc.Channel | None = None
        self._stub: refit_pb2_grpc.RefitServiceStub | None = None

    @classmethod
    def connect(
        cls,
        *,
        server_url: str | None = None,
        rpc_timeout_seconds: float = 30.0,
    ) -> ModelExpressControlClient:
        """Connect to the MX control plane without exposing protobuf details."""
        if rpc_timeout_seconds <= 0:
            raise ValueError("rpc_timeout_seconds must be positive")
        client = cls()
        client.server_url = _get_server_url(server_url)
        client._rpc_timeout_seconds = rpc_timeout_seconds
        return client

    @property
    def _service(self) -> refit_pb2_grpc.RefitServiceStub:
        if self._channel is None:
            self._channel = auth.with_auth(grpc.insecure_channel(self.server_url))
            self._stub = refit_pb2_grpc.RefitServiceStub(self._channel)
        assert self._stub is not None
        return self._stub

    def create_trainer_mesh(
        self,
        *,
        model_name: str,
        idempotency_key: str,
        workers: dict[str, TrainerTensorsMetadata],
    ) -> TrainerMesh:
        """Declare complete expected trainer membership for a refit group."""
        response = self._service.CreateTrainerMesh(
            refit_pb2.CreateTrainerMeshRequest(
                model_name=_required(model_name, "model_name"),
                idempotency_key=_required(idempotency_key, "idempotency_key"),
                workers=_mesh_workers(workers),
            ),
            timeout=self._rpc_timeout_seconds,
        )
        return _response_mesh(response, "CreateTrainerMesh")

    def get_trainer_mesh(self, mesh_id: str) -> TrainerMesh:
        response = self._service.GetTrainerMesh(
            refit_pb2.GetTrainerMeshRequest(mesh_id=_required(mesh_id, "mesh_id")),
            timeout=self._rpc_timeout_seconds,
        )
        return _response_mesh(response, "GetTrainerMesh")

    def update_trainer_mesh(
        self,
        *,
        mesh_id: str,
        expected_generation: int,
        workers: dict[str, TrainerTensorsMetadata],
    ) -> TrainerMesh:
        """Replace physical members if the mesh generation still matches."""
        response = self._service.UpdateTrainerMesh(
            refit_pb2.UpdateTrainerMeshRequest(
                mesh_id=_required(mesh_id, "mesh_id"),
                expected_generation=expected_generation,
                workers=_mesh_workers(workers),
            ),
            timeout=self._rpc_timeout_seconds,
        )
        return _response_mesh(response, "UpdateTrainerMesh")

    def delete_trainer_mesh(self, mesh_id: str) -> None:
        self._service.DeleteTrainerMesh(
            refit_pb2.DeleteTrainerMeshRequest(mesh_id=_required(mesh_id, "mesh_id")),
            timeout=self._rpc_timeout_seconds,
        )

    def create_weight_version(
        self,
        *,
        model_name: str,
        idempotency_key: str,
        payload_format: WeightPayloadFormat,
        trainer_mesh_id: str | None = None,
        version_number: int | None = None,
        uid: str | None = None,
        base_version_id: str | None = None,
        object_storage: ObjectStorageSource | None = None,
        state: WeightVersionState = WeightVersionState.STAGING,
    ) -> WeightVersion:
        """Create one global version with its initial lifecycle state."""
        _required(model_name, "model_name")
        _required(idempotency_key, "idempotency_key")
        if payload_format is WeightPayloadFormat.UNSPECIFIED:
            raise ValueError("payload_format must be specified")
        if state not in {WeightVersionState.STAGING, WeightVersionState.READY}:
            raise ValueError("new weight version state must be STAGING or READY")
        if object_storage is not None and not isinstance(
            object_storage, ObjectStorageSource
        ):
            raise TypeError("object_storage must be an ObjectStorageSource")
        if trainer_mesh_id is not None:
            _required(trainer_mesh_id, "trainer_mesh_id")
            if object_storage is not None:
                raise ValueError("trainer_mesh_id is not valid for object storage")
        if object_storage is None and trainer_mesh_id is None:
            raise ValueError("trainer_mesh_id is required for worker publication")
        request = refit_pb2.CreateWeightVersionRequest(
            model_name=model_name,
            idempotency_key=idempotency_key,
            payload_format={
                WeightPayloadFormat.FULL_TENSOR: refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_TENSOR,
                WeightPayloadFormat.XOR_DELTA: refit_pb2.WEIGHT_PAYLOAD_FORMAT_XOR_DELTA,
                WeightPayloadFormat.FULL_HF_CHECKPOINT: (
                    refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_HF_CHECKPOINT
                ),
            }[payload_format],
            state={
                WeightVersionState.STAGING: refit_pb2.WEIGHT_VERSION_STATE_STAGING,
                WeightVersionState.READY: refit_pb2.WEIGHT_VERSION_STATE_READY,
            }[state],
        )
        if uid is not None:
            request.uid = _required(uid, "uid")
        if trainer_mesh_id is not None:
            request.trainer_mesh_id = trainer_mesh_id
        if version_number is not None:
            request.version_number = version_number
        if base_version_id is not None:
            request.base_version_id = _required(base_version_id, "base_version_id")
        if object_storage is not None:
            request.object_storage.uri = object_storage.uri
            request.object_storage.storage_type = {
                ObjectStorageType.S3: refit_pb2.OBJECT_STORAGE_TYPE_S3,
                ObjectStorageType.AZURE: refit_pb2.OBJECT_STORAGE_TYPE_AZURE,
                ObjectStorageType.GCS: refit_pb2.OBJECT_STORAGE_TYPE_GCS,
            }[object_storage.storage_type]
        response = self._service.CreateWeightVersion(
            request,
            timeout=self._rpc_timeout_seconds,
        )
        return _response_version(response, "CreateWeightVersion")

    def get_weight_version(self, version_id: str) -> WeightVersion:
        """Read the current lifecycle state of one weight version."""
        response = self._service.GetWeightVersion(
            refit_pb2.GetWeightVersionRequest(uid=_required(version_id, "version_id")),
            timeout=self._rpc_timeout_seconds,
        )
        return _response_version(response, "GetWeightVersion")

    def update_weight_version_state(
        self,
        version_id: str,
        state: WeightVersionState,
    ) -> WeightVersion:
        """Update one version's lifecycle state."""
        if not isinstance(state, WeightVersionState):
            raise TypeError("state must be a WeightVersionState")
        response = self._service.UpdateWeightVersionState(
            refit_pb2.UpdateWeightVersionStateRequest(
                uid=_required(version_id, "version_id"),
                state={
                    WeightVersionState.STAGING: refit_pb2.WEIGHT_VERSION_STATE_STAGING,
                    WeightVersionState.READY: refit_pb2.WEIGHT_VERSION_STATE_READY,
                    WeightVersionState.RELEASING: refit_pb2.WEIGHT_VERSION_STATE_RELEASING,
                }[state],
            ),
            timeout=self._rpc_timeout_seconds,
        )
        return _response_version(response, "UpdateWeightVersionState")

    def list_weight_versions(
        self, *, model_name: str, trainer_mesh_id: str | None = None,
    ) -> list[WeightVersion]:
        request = refit_pb2.ListWeightVersionsRequest(model_name=_required(model_name, "model_name"))
        if trainer_mesh_id is not None:
            request.trainer_mesh_id = _required(trainer_mesh_id, "trainer_mesh_id")
        response = self._service.ListWeightVersions(request, timeout=self._rpc_timeout_seconds)
        return [_weight_version(version) for version in response.versions]

    def delete_weight_version(self, version_id: str) -> WeightVersion:
        """Move a STAGING or READY version to RELEASING."""
        response = self._service.DeleteWeightVersion(
            refit_pb2.DeleteWeightVersionRequest(
                uid=_required(version_id, "version_id")
            ),
            timeout=self._rpc_timeout_seconds,
        )
        return _response_version(response, "DeleteWeightVersion")

    def close(self) -> None:
        """Close the underlying gRPC channel."""
        if self._channel is not None:
            self._channel.close()
            self._channel = None
            self._stub = None

    def __enter__(self) -> ModelExpressControlClient:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


__all__ = [
    "ModelExpressControlClient",
    "ObjectStorageSource",
    "ObjectStorageType",
    "TrainerMesh",
    "TrainerTensorsMetadata",
    "WeightVersion",
    "WeightVersionState",
]
