# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

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


def test_list_sources_passes_strict_to_server():
    stub = MagicMock()
    client = MxClient("localhost:8001")
    client._channel = MagicMock()
    client._stub = stub
    identity = p2p_pb2.SourceIdentity(model_name="model")

    client.list_sources(identity=identity, strict=True)

    request = stub.ListSources.call_args.args[0]
    assert request.strict is True
    assert request.identity == identity


@pytest.mark.parametrize(
    "action",
    [
        p2p_pb2.DONOR_LEASE_ACTION_ACQUIRE,
        p2p_pb2.DONOR_LEASE_ACTION_RENEW,
        p2p_pb2.DONOR_LEASE_ACTION_RELEASE,
    ],
)
@pytest.mark.parametrize("success", [True, False])
def test_donor_lease_returns_action_success(action, success):
    stub = MagicMock()
    stub.DonorLease.return_value = p2p_pb2.DonorLeaseResponse(success=success)
    client = MxClient("localhost:8001")
    client._channel = MagicMock()
    client._stub = stub
    identity = p2p_pb2.SourceIdentity(model_name="model")

    assert client.donor_lease(identity, "owner", action, 60) is success
    request = stub.DonorLease.call_args.args[0]
    assert request.identity == identity
    assert request.owner_token == "owner"
    assert request.action == action
    assert request.ttl_seconds == 60
