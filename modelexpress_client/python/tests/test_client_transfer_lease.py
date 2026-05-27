# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from modelexpress import p2p_pb2
from modelexpress.client import MxClient


class _FakeStub:
    def __init__(self):
        self.requests = []

    def BeginTransferLease(self, request, timeout=None):
        self.requests.append(("begin", request, timeout))
        return p2p_pb2.TransferLeaseResponse(
            success=True,
            lease=p2p_pb2.TransferLease(
                lease_id="lease-1",
                status=p2p_pb2.TRANSFER_LEASE_STATUS_ACTIVE,
            ),
        )

    def RenewTransferLease(self, request, timeout=None):
        self.requests.append(("renew", request, timeout))
        return p2p_pb2.TransferLeaseResponse(
            success=True,
            lease=p2p_pb2.TransferLease(
                lease_id=request.lease_id,
                status=p2p_pb2.TRANSFER_LEASE_STATUS_ACTIVE,
            ),
        )

    def CompleteTransferLease(self, request, timeout=None):
        self.requests.append(("complete", request, timeout))
        return p2p_pb2.TransferLeaseResponse(
            success=True,
            lease=p2p_pb2.TransferLease(
                lease_id=request.lease_id,
                status=request.status,
                error_message=request.error_message,
            ),
        )

    def GetTransferLease(self, request, timeout=None):
        self.requests.append(("get", request, timeout))
        return p2p_pb2.GetTransferLeaseResponse(
            found=True,
            lease=p2p_pb2.TransferLease(lease_id=request.lease_id),
        )

    def ListTransferLeases(self, request, timeout=None):
        self.requests.append(("list", request, timeout))
        return p2p_pb2.ListTransferLeasesResponse(
            leases=[
                p2p_pb2.TransferLease(
                    lease_id="lease-1",
                    mx_source_id=request.mx_source_id,
                    target_worker_id=request.target_worker_id,
                    status=request.status_filter,
                )
            ]
        )


class _FakeClient(MxClient):
    def __init__(self, stub):
        super().__init__(server_url="unused:0")
        self._fake_stub = stub

    @property
    def stub(self):
        return self._fake_stub


def test_begin_transfer_lease_sends_request_fields():
    stub = _FakeStub()
    client = _FakeClient(stub)

    lease = client.begin_transfer_lease(
        mx_source_id="source",
        source_worker_id="source-worker",
        target_worker_id="target-worker",
        target_worker_rank=3,
        model_version=11,
        ttl_millis=5000,
        metadata={"role": "trainer"},
        lease_id="lease-1",
    )

    assert lease.lease_id == "lease-1"
    method, request, timeout = stub.requests[-1]
    assert method == "begin"
    assert timeout == 30
    assert request.mx_source_id == "source"
    assert request.source_worker_id == "source-worker"
    assert request.target_worker_id == "target-worker"
    assert request.target_worker_rank == 3
    assert request.model_version == 11
    assert request.ttl_millis == 5000
    assert request.metadata["role"] == "trainer"


def test_renew_complete_get_and_list_transfer_lease():
    stub = _FakeStub()
    client = _FakeClient(stub)

    renewed = client.renew_transfer_lease("lease-1", ttl_millis=7000)
    completed = client.complete_transfer_lease(
        "lease-1",
        status=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
    )
    fetched = client.get_transfer_lease("lease-1")
    listed = client.list_transfer_leases(
        mx_source_id="source",
        target_worker_id="target-worker",
        status_filter=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
        model_version_filter=17,
        source_worker_id="source-worker",
    )

    assert renewed.status == p2p_pb2.TRANSFER_LEASE_STATUS_ACTIVE
    assert completed.status == p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED
    assert fetched.found
    assert len(listed.leases) == 1
    assert listed.leases[0].mx_source_id == "source"
    assert listed.leases[0].target_worker_id == "target-worker"
    assert stub.requests[-1][1].model_version_filter == 17
    assert stub.requests[-1][1].source_worker_id == "source-worker"
    assert [method for method, _request, _timeout in stub.requests] == [
        "renew",
        "complete",
        "get",
        "list",
    ]


def test_begin_transfer_lease_failure_raises():
    class _FailingStub(_FakeStub):
        def BeginTransferLease(self, request, timeout=None):
            return p2p_pb2.TransferLeaseResponse(success=False, message="duplicate")

    client = _FakeClient(_FailingStub())

    with pytest.raises(RuntimeError, match="duplicate"):
        client.begin_transfer_lease(
            mx_source_id="source",
            source_worker_id="source-worker",
            target_worker_id="target-worker",
            target_worker_rank=0,
            model_version=1,
        )
