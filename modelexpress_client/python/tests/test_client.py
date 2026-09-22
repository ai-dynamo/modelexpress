# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from modelexpress import p2p_pb2
from modelexpress.client import MxClient


def test_publish_metadata_adds_pod_identity_without_changing_public_api(monkeypatch):
    monkeypatch.setenv("POD_NAME", "source-0")
    monkeypatch.setenv("POD_UID", "pod-uid")
    monkeypatch.setenv("POD_NAMESPACE", "default")
    stub = MagicMock()
    stub.PublishMetadata.return_value = p2p_pb2.PublishMetadataResponse(
        success=True,
        mx_source_id="source-id",
    )
    client = MxClient("localhost:8001")
    client._channel = MagicMock()
    client._stub = stub

    source_id = client.publish_metadata(
        p2p_pb2.SourceIdentity(model_name="model"),
        p2p_pb2.WorkerMetadata(worker_rank=0),
        "worker-id",
    )

    assert source_id == "source-id"
    request = stub.PublishMetadata.call_args.args[0]
    assert request.pod_name == "source-0"
    assert request.pod_uid == "pod-uid"
    assert request.pod_namespace == "default"
