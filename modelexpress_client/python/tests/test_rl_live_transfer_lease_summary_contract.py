# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import uuid

import grpc
import pytest

from modelexpress import p2p_pb2
from modelexpress.client import MxClient
from modelexpress.rl_metadata import RlSourceRole
from modelexpress.rl_transfer_report import RlTransferAttempt, RlTransferReport
from modelexpress.rl_transfer_lease import (
    RlTransferLeaseCoordinator,
    summarize_report_leases,
)

_LIVE_SERVER_ENV = "MX_LIVE_SERVER_URL"

pytestmark = pytest.mark.skipif(
    not os.environ.get(_LIVE_SERVER_ENV),
    reason=f"{_LIVE_SERVER_ENV} is not set",
)


def _begin_transfer_lease_or_skip(client: MxClient, **kwargs):
    try:
        return client.begin_transfer_lease(**kwargs)
    except grpc.RpcError as exc:
        if exc.code() == grpc.StatusCode.UNIMPLEMENTED:
            pytest.skip("server does not expose transfer lease RPCs")
        raise


def test_live_server_report_lease_summary_scopes_source_worker():
    client = MxClient(server_url=os.environ[_LIVE_SERVER_ENV])
    lease_id = f"lease-summary-{uuid.uuid4().hex}"
    other_lease_id = f"lease-summary-other-{uuid.uuid4().hex}"
    mx_source_id = f"live-summary-source-{uuid.uuid4().hex[:8]}"
    target_worker_id = f"target-summary-{uuid.uuid4().hex[:8]}"

    try:
        _begin_transfer_lease_or_skip(
            client,
            lease_id=lease_id,
            mx_source_id=mx_source_id,
            source_worker_id="source-worker",
            target_worker_id=target_worker_id,
            target_worker_rank=1,
            model_version=31,
            ttl_millis=5_000,
            metadata={"contract": "summary"},
        )
        client.complete_transfer_lease(
            lease_id,
            status=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
        )
        _begin_transfer_lease_or_skip(
            client,
            lease_id=other_lease_id,
            mx_source_id=mx_source_id,
            source_worker_id="source-worker-other",
            target_worker_id=target_worker_id,
            target_worker_rank=1,
            model_version=31,
            ttl_millis=5_000,
            metadata={"contract": "summary-other-source"},
        )
        client.complete_transfer_lease(
            other_lease_id,
            status=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
        )

        report = RlTransferReport(
            requested_model_version=None,
            resolved_model_version=31,
            receiver_rank=1,
            attempts=(
                RlTransferAttempt(
                    mx_source_id=mx_source_id,
                    worker_id="source-worker",
                    worker_rank=0,
                    role=RlSourceRole.TRAINER,
                    model_version=31,
                    success=True,
                    lease_id=lease_id,
                ),
            ),
        )
        assert report.single_lease_source_worker_id == "source-worker"

        coordinator = RlTransferLeaseCoordinator(
            mx_client=client,
            target_worker_id=target_worker_id,
            ttl_seconds=5,
        )
        unscoped_inventory = coordinator.list_target_leases(
            mx_source_id=mx_source_id,
            statuses=(p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,),
            model_version=report.resolved_model_version,
        )
        assert [lease.lease_id for lease in unscoped_inventory.leases] == [
            lease_id,
            other_lease_id,
        ]

        scoped_inventory = coordinator.list_target_leases(
            mx_source_id=mx_source_id,
            statuses=(p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,),
            model_version=report.resolved_model_version,
            source_worker_id=report.single_lease_source_worker_id or "",
        )
        summary = summarize_report_leases(report, scoped_inventory)

        assert scoped_inventory.discovery_supported
        assert [lease.lease_id for lease in scoped_inventory.leases] == [lease_id]
        assert summary.report_lease_ids == (lease_id,)
        assert [lease.lease_id for lease in summary.matching_leases] == [lease_id]
        assert summary.missing_lease_ids == ()
        assert summary.non_completed_matching_leases == ()
    finally:
        client.close()
