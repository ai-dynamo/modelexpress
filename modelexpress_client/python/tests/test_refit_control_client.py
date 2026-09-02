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
    refit_pb2,
    refit_pb2_grpc,
)


class _RefitService(refit_pb2_grpc.RefitServiceServicer):
    def __init__(self) -> None:
        self.version = None

    def CreateWeightVersion(self, request, _context):
        self.version = refit_pb2.WeightVersion(
            uid=request.uid if request.HasField("uid") else "version-a",
            model_name=request.model_name,
            idempotency_key=request.idempotency_key,
            payload_format=request.payload_format,
            expected_source_slots=request.expected_source_slots,
            state=request.state,
            created_at_unix_ms=1234,
        )
        if request.HasField("base_version_id"):
            self.version.base_version_id = request.base_version_id
        if request.HasField("object_storage"):
            self.version.object_storage.CopyFrom(request.object_storage)
        return refit_pb2.CreateWeightVersionResponse(version=self.version)

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


def test_control_client_rejects_missing_version_response():
    with pytest.raises(RuntimeError, match="GetWeightVersion.*missing version"):
        control_module._response_version(
            refit_pb2.GetWeightVersionResponse(),
            "GetWeightVersion",
        )


def test_control_client_owns_global_weight_version_lifecycle():
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
            expected_source_slots=[
                "publisher:global-rank:0",
                "publisher:global-rank:1",
            ],
        )
        fetched = control.get_weight_version(created.version_id)
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
    assert created.expected_source_slots == (
        "publisher:global-rank:0",
        "publisher:global-rank:1",
    )
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
    assert created.expected_source_slots == ()
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
                expected_source_slots=["rank:0"],
            )
        with pytest.raises(ValueError, match="uid"):
            control.create_weight_version(
                model_name="test/model",
                idempotency_key="attempt-a",
                payload_format=WeightPayloadFormat.FULL_TENSOR,
                uid=" ",
            )
    finally:
        control.close()
