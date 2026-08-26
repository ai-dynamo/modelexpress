# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent import futures

import grpc
import pytest
from modelexpress_rl import (
    ModelExpressControlClient,
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
            uid="version-a",
            model_name=request.model_name,
            version_number=request.version_number,
            idempotency_key=request.idempotency_key,
            payload_format=request.payload_format,
            expected_source_slots=request.expected_source_slots,
            state=refit_pb2.WEIGHT_VERSION_STATE_STAGING,
            created_at_unix_ms=1234,
        )
        return self.version

    def GetWeightVersion(self, request, context):
        if self.version is None or request.uid != self.version.uid:
            context.abort(grpc.StatusCode.NOT_FOUND, "version not found")
        return self.version

    def DeleteWeightVersion(self, request, context):
        if self.version is None or request.uid != self.version.uid:
            context.abort(grpc.StatusCode.NOT_FOUND, "version not found")
        self.version.state = refit_pb2.WEIGHT_VERSION_STATE_RELEASING
        return self.version


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
            version_number=7,
            idempotency_key="training-step-7",
            payload_format=WeightPayloadFormat.FULL_TENSOR,
            expected_source_slots=[
                "publisher:global-rank:0",
                "publisher:global-rank:1",
            ],
        )
        fetched = control.get_weight_version(created.version_id)
        deleted = control.delete_weight_version(created.version_id)
    finally:
        if "control" in locals():
            control.close()
        server.stop(grace=None).wait()

    assert created.ref.version_id == "version-a"
    assert created.version_number == 7
    assert created.payload_format is WeightPayloadFormat.FULL_TENSOR
    assert created.expected_source_slots == (
        "publisher:global-rank:0",
        "publisher:global-rank:1",
    )
    assert created.state is WeightVersionState.STAGING
    assert fetched == created
    assert deleted.state is WeightVersionState.RELEASING


def test_control_client_validates_framework_inputs_before_rpc():
    control = ModelExpressControlClient.connect(server_url="127.0.0.1:1")
    try:
        with pytest.raises(ValueError, match="expected_source_slots"):
            control.create_weight_version(
                model_name="test/model",
                idempotency_key="attempt-a",
                payload_format=WeightPayloadFormat.FULL_TENSOR,
                expected_source_slots=[],
            )
        with pytest.raises(ValueError, match="payload_format"):
            control.create_weight_version(
                model_name="test/model",
                idempotency_key="attempt-a",
                payload_format=WeightPayloadFormat.UNSPECIFIED,
                expected_source_slots=["rank:0"],
            )
    finally:
        control.close()
