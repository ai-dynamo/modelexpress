# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent import futures

import grpc
import pytest
from modelexpress_rl import control as control_module
from modelexpress_rl import (
    ModelExpressControlClient,
    ObjectStorageSource,
    ObjectStorageType,
    WeightPayloadFormat,
    WeightVersionState,
    TrainerTensorsMetadata,
    refit_pb2,
    refit_pb2_grpc,
)


class _RefitService(refit_pb2_grpc.RefitServiceServicer):
    def __init__(self) -> None:
        self.version = None
        self.mesh = refit_pb2.TrainerMesh(
            mesh_id="mesh-a", model_name="test/model", generation=1
        )

    def CreateTrainerMesh(self, request, _context):
        self.mesh = refit_pb2.TrainerMesh(
            mesh_id="mesh-a",
            model_name=request.model_name,
            generation=1,
            workers=request.workers,
        )
        return refit_pb2.CreateTrainerMeshResponse(mesh=self.mesh)

    def GetTrainerMesh(self, request, context):
        if self.mesh is None or request.mesh_id != self.mesh.mesh_id:
            context.abort(grpc.StatusCode.NOT_FOUND, "mesh not found")
        return refit_pb2.GetTrainerMeshResponse(mesh=self.mesh)

    def UpdateTrainerMesh(self, request, context):
        if self.mesh is None or request.mesh_id != self.mesh.mesh_id:
            context.abort(grpc.StatusCode.NOT_FOUND, "mesh not found")
        if request.expected_generation != self.mesh.generation:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "stale generation")
        self.mesh.generation += 1
        self.mesh.workers.clear()
        for worker_id, metadata in request.workers.items():
            self.mesh.workers[worker_id].CopyFrom(metadata)
        return refit_pb2.UpdateTrainerMeshResponse(mesh=self.mesh)

    def DeleteTrainerMesh(self, request, context):
        if self.mesh is None or request.mesh_id != self.mesh.mesh_id:
            context.abort(grpc.StatusCode.NOT_FOUND, "mesh not found")
        self.mesh = None
        return refit_pb2.DeleteTrainerMeshResponse()

    def CreateWeightVersion(
        self, request, _context
    ) -> refit_pb2.CreateWeightVersionResponse:
        if (
            self.version is not None
            and self.version.idempotency_key == request.idempotency_key
        ):
            return refit_pb2.CreateWeightVersionResponse(version=self.version)
        self.version = refit_pb2.WeightVersion(
            uid=request.uid if request.HasField("uid") else "version-a",
            model_name=request.model_name,
            idempotency_key=request.idempotency_key,
            payload_format=request.payload_format,
            state=request.state,
            created_at_unix_ms=1234,
        )
        if request.HasField("base_version_id"):
            self.version.base_version_id = request.base_version_id
        if request.HasField("object_storage"):
            self.version.object_storage.CopyFrom(request.object_storage)
        if request.HasField("trainer_mesh_id"):
            self.version.trainer_mesh_id = request.trainer_mesh_id
            self.version.trainer_mesh_generation = self.mesh.generation
        if request.HasField("version_number"):
            self.version.version_number = request.version_number
        return refit_pb2.CreateWeightVersionResponse(version=self.version)

    def ListWeightVersions(self, request, _context):
        versions = []
        if self.version is not None and self.version.model_name == request.model_name:
            if not request.HasField("trainer_mesh_id") or self.version.trainer_mesh_id == request.trainer_mesh_id:
                versions.append(self.version)
        return refit_pb2.ListWeightVersionsResponse(versions=versions)

    def GetWeightVersion(self, request, context):
        if self.version is None or request.uid != self.version.uid:
            context.abort(grpc.StatusCode.NOT_FOUND, "version not found")
        return refit_pb2.GetWeightVersionResponse(version=self.version)

    def DeleteWeightVersion(self, request, context):
        if self.version is None or request.uid != self.version.uid:
            context.abort(grpc.StatusCode.NOT_FOUND, "version not found")
        self.version.state = refit_pb2.WEIGHT_VERSION_STATE_RELEASING
        return refit_pb2.DeleteWeightVersionResponse(version=self.version)

    def UpdateWeightVersionState(self, request, context):
        if self.version is None or request.uid != self.version.uid:
            context.abort(grpc.StatusCode.NOT_FOUND, "version not found")
        self.version.state = request.state
        return refit_pb2.UpdateWeightVersionStateResponse(version=self.version)


def test_weight_version_requires_generation_at_construction() -> None:
    with pytest.raises(TypeError, match="trainer_mesh_generation"):
        control_module.WeightVersion(
            version_id="version-a",
            model_name="test/model",
            payload_format=WeightPayloadFormat.FULL_TENSOR,
            layout_signature="",
            state=WeightVersionState.STAGING,
            created_at_unix_ms=1234,
            trainer_mesh_id="mesh-a",
        )


def test_control_client_rejects_missing_version_response():
    with pytest.raises(RuntimeError, match="GetWeightVersion.*missing version"):
        control_module._response_version(
            refit_pb2.GetWeightVersionResponse(),
            "GetWeightVersion",
        )


def test_control_client_manages_trainer_mesh():
    service = _RefitService()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    refit_pb2_grpc.add_RefitServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    control = ModelExpressControlClient.connect(server_url=f"127.0.0.1:{port}")
    try:
        created = control.create_trainer_mesh(
            model_name="test/model",
            idempotency_key="run-a",
            workers={"worker-a": TrainerTensorsMetadata("shard-a", "worker-a:5000")},
        )
        fetched = control.get_trainer_mesh(created.mesh_id)
        updated = control.update_trainer_mesh(
            mesh_id=created.mesh_id,
            expected_generation=created.generation,
            workers={"worker-b": TrainerTensorsMetadata("shard-a", "worker-b:5000")},
        )
        with pytest.raises(grpc.RpcError) as stale:
            control.update_trainer_mesh(
                mesh_id=created.mesh_id,
                expected_generation=created.generation,
                workers={"worker-c": TrainerTensorsMetadata("shard-a", "worker-c:5000")},
            )
        control.delete_trainer_mesh(created.mesh_id)
    finally:
        control.close()
        server.stop(grace=None).wait()

    assert fetched == created
    assert created.workers == {"worker-a": TrainerTensorsMetadata("shard-a", "worker-a:5000")}
    assert updated.generation == created.generation + 1
    assert updated.workers == {"worker-b": TrainerTensorsMetadata("shard-a", "worker-b:5000")}
    assert stale.value.code() is grpc.StatusCode.FAILED_PRECONDITION


def test_control_client_links_version_to_trainer_mesh():
    service = _RefitService()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    refit_pb2_grpc.add_RefitServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    control = ModelExpressControlClient.connect(server_url=f"127.0.0.1:{port}")
    try:
        version = control.create_weight_version(
            model_name="test/model",
            idempotency_key="step-a",
            payload_format=WeightPayloadFormat.FULL_TENSOR,
            trainer_mesh_id="mesh-a",
            version_number=7,
        )
        fetched = control.get_weight_version(version.version_id)
        assert control.list_weight_versions(model_name="test/model", trainer_mesh_id="mesh-a") == [version]
        assert control.list_weight_versions(model_name="other/model") == []
        assert control.list_weight_versions(model_name="test/model", trainer_mesh_id="other-mesh") == []
        with pytest.raises(ValueError, match="trainer_mesh_id is required"):
            control.create_weight_version(
                model_name="test/model",
                idempotency_key="step-b",
                payload_format=WeightPayloadFormat.FULL_TENSOR,
            )
    finally:
        control.close()
        server.stop(grace=None).wait()

    assert version.trainer_mesh_id == "mesh-a"
    assert version.version_number == 7
    assert fetched == version


def test_control_client_owns_global_weight_version_lifecycle() -> None:
    service = _RefitService()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    refit_pb2_grpc.add_RefitServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    try:
        control = ModelExpressControlClient.connect(server_url=f"127.0.0.1:{port}")
        created = control.create_weight_version(
            model_name="test/model",
            idempotency_key="training-step-7",
            payload_format=WeightPayloadFormat.FULL_TENSOR,
            trainer_mesh_id="mesh-a",
        )
        fetched = control.get_weight_version(created.version_id)
        service.mesh.generation = 2
        retried = control.create_weight_version(
            model_name="test/model",
            idempotency_key="training-step-7",
            payload_format=WeightPayloadFormat.FULL_TENSOR,
            trainer_mesh_id="mesh-a",
        )
        assert retried.trainer_mesh_generation == created.trainer_mesh_generation == 1
        fresh = control.create_weight_version(
            model_name="test/model",
            idempotency_key="training-step-8",
            payload_format=WeightPayloadFormat.FULL_TENSOR,
            trainer_mesh_id="mesh-a",
        )
        assert fresh.trainer_mesh_generation == 2
        ready = control.update_weight_version_state(
            created.version_id,
            WeightVersionState.READY,
        )
        deleted = control.delete_weight_version(created.version_id)
    finally:
        if "control" in locals():
            control.close()
        server.stop(grace=None).wait()

    assert created.ref.version_id == "version-a"
    assert created.payload_format is WeightPayloadFormat.FULL_TENSOR
    assert created.trainer_mesh_id == "mesh-a"
    assert created.trainer_mesh_generation == 1
    assert created.state is WeightVersionState.STAGING
    assert fetched == created
    assert ready.state is WeightVersionState.READY
    assert deleted.state is WeightVersionState.RELEASING


@pytest.mark.parametrize(
    ("storage_type", "uri"),
    [
        (ObjectStorageType.S3, "s3://weights/run/v7/index.json"),
        (ObjectStorageType.AZURE, "az://weights/run/v7/index.json"),
        (ObjectStorageType.GCS, "gs://weights/run/v7/index.json"),
    ],
)
def test_control_client_round_trips_object_storage_source(storage_type, uri):
    service = _RefitService()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    refit_pb2_grpc.add_RefitServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    try:
        control = ModelExpressControlClient.connect(server_url=f"127.0.0.1:{port}")
        created = control.create_weight_version(
            model_name="test/model",
            idempotency_key="training-step-7",
            payload_format=WeightPayloadFormat.XOR_DELTA,
            uid="caller-version",
            base_version_id="base-a",
            object_storage=ObjectStorageSource(
                storage_type=storage_type,
                uri=uri,
            ),
            state=WeightVersionState.READY,
        )
    finally:
        if "control" in locals():
            control.close()
        server.stop(grace=None).wait()

    assert created.object_storage == ObjectStorageSource(
        storage_type=storage_type,
        uri=uri,
    )
    assert created.version_id == "caller-version"
    assert created.state is WeightVersionState.READY


def test_control_client_round_trips_full_hf_checkpoint():
    service = _RefitService()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    refit_pb2_grpc.add_RefitServiceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    try:
        control = ModelExpressControlClient.connect(server_url=f"127.0.0.1:{port}")
        created = control.create_weight_version(
            model_name="test/model",
            idempotency_key="training-step-25",
            payload_format=WeightPayloadFormat.FULL_HF_CHECKPOINT,
            object_storage=ObjectStorageSource(
                storage_type=ObjectStorageType.S3,
                uri="s3://weights/run/v25/model.safetensors.index.json",
            ),
        )
    finally:
        if "control" in locals():
            control.close()
        server.stop(grace=None).wait()

    assert refit_pb2.WEIGHT_PAYLOAD_FORMAT_FULL_HF_CHECKPOINT == 3
    assert created.payload_format is WeightPayloadFormat.FULL_HF_CHECKPOINT
    assert created.base_version_id is None


def test_expected_source_slots_is_removed_from_public_contract():
    import inspect

    assert "expected_source_slots" not in refit_pb2.WeightVersion.DESCRIPTOR.fields_by_name
    assert "expected_source_slots" not in refit_pb2.CreateWeightVersionRequest.DESCRIPTOR.fields_by_name
    assert "expected_source_slots" not in inspect.signature(
        ModelExpressControlClient.create_weight_version
    ).parameters


def test_control_client_validates_framework_inputs_before_rpc():
    control = ModelExpressControlClient.connect(server_url="127.0.0.1:1")
    try:
        with pytest.raises(ValueError, match="state"):
            control.create_weight_version(
                model_name="test/model",
                idempotency_key="attempt-a",
                payload_format=WeightPayloadFormat.FULL_TENSOR,
                state=WeightVersionState.RELEASING,
            )
        with pytest.raises(ValueError, match="payload_format"):
            control.create_weight_version(
                model_name="test/model",
                idempotency_key="attempt-a",
                payload_format=WeightPayloadFormat.UNSPECIFIED,
            )
        with pytest.raises(ValueError, match="uid"):
            control.create_weight_version(
                model_name="test/model",
                idempotency_key="attempt-a",
                payload_format=WeightPayloadFormat.FULL_TENSOR,
                trainer_mesh_id="mesh-a",
                uid=" ",
            )
    finally:
        control.close()
