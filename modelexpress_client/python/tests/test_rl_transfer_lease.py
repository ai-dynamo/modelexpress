# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import grpc
import pytest

from modelexpress import p2p_pb2
from modelexpress.rl_metadata import RlSourceCandidate, RlSourceMetadata, RlSourceRole
from modelexpress.rl_transfer_lease import (
    RlTransferLeaseCoordinator,
    RlTransferLeaseInventory,
    summarize_report_leases,
)
from modelexpress.rl_transfer_report import RlTransferAttempt, RlTransferReport


class _LeaseClient:
    def __init__(self):
        self.begins = []
        self.renews = []
        self.completes = []
        self.lists = []
        self.leases_by_status = {}

    def begin_transfer_lease(self, **kwargs):
        self.begins.append(kwargs)
        return p2p_pb2.TransferLease(lease_id="lease-1")

    def renew_transfer_lease(self, lease_id, *, ttl_millis=0):
        self.renews.append((lease_id, ttl_millis))
        return p2p_pb2.TransferLease(lease_id=lease_id)

    def complete_transfer_lease(self, lease_id, *, status, error_message=""):
        self.completes.append((lease_id, status, error_message))
        return p2p_pb2.TransferLease(lease_id=lease_id, status=status)

    def list_transfer_leases(
        self,
        *,
        mx_source_id="",
        target_worker_id="",
        status_filter=None,
        model_version_filter=None,
        source_worker_id="",
    ):
        self.lists.append(
            (
                mx_source_id,
                target_worker_id,
                status_filter,
                model_version_filter,
                source_worker_id,
            )
        )
        return p2p_pb2.ListTransferLeasesResponse(
            leases=self.leases_by_status.get(status_filter, ())
        )


def _candidate():
    return RlSourceCandidate(
        mx_source_id="source",
        worker_id="source-worker",
        model_name="test-model",
        worker_rank=0,
        metadata=RlSourceMetadata(
            model_version=7,
            role=RlSourceRole.TRAINER,
            world_size=1,
        ),
    )


def _lease(
    lease_id,
    *,
    version,
    status,
    updated_at,
    created_at=1,
):
    return p2p_pb2.TransferLease(
        lease_id=lease_id,
        mx_source_id="source",
        source_worker_id=f"source-worker-{lease_id}",
        target_worker_id="target-worker",
        model_version=version,
        status=status,
        created_at=created_at,
        updated_at=updated_at,
    )


def test_transfer_lease_completes_successful_context():
    client = _LeaseClient()
    coordinator = RlTransferLeaseCoordinator(
        mx_client=client,
        target_worker_id="target-worker",
        ttl_seconds=5,
    )

    with coordinator.lease_candidate(_candidate(), receiver_rank=3):
        pass

    assert client.begins == [
        {
            "mx_source_id": "source",
            "source_worker_id": "source-worker",
            "target_worker_id": "target-worker",
            "target_worker_rank": 3,
            "model_version": 7,
            "ttl_millis": 5000,
            "metadata": {"mx_rl_role": "trainer"},
        }
    ]
    assert client.completes == [
        ("lease-1", p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED, "")
    ]


def test_transfer_lease_marks_failed_context():
    client = _LeaseClient()
    coordinator = RlTransferLeaseCoordinator(
        mx_client=client,
        target_worker_id="target-worker",
        ttl_seconds=5,
    )

    with pytest.raises(RuntimeError, match="transfer failed"):
        with coordinator.lease_candidate(_candidate(), receiver_rank=0):
            raise RuntimeError("transfer failed")

    assert client.completes == [
        ("lease-1", p2p_pb2.TRANSFER_LEASE_STATUS_FAILED, "transfer failed")
    ]


def test_transfer_lease_noops_when_client_does_not_support_leases():
    coordinator = RlTransferLeaseCoordinator(
        mx_client=object(),
        target_worker_id="target-worker",
        ttl_seconds=5,
    )

    with coordinator.lease_candidate(_candidate(), receiver_rank=0):
        pass


def test_transfer_lease_inventory_summarizes_latest_target_attempts():
    inventory = RlTransferLeaseInventory(
        target_worker_id="target-worker",
        leases=(
            _lease(
                "lease-v6-completed",
                version=6,
                status=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
                updated_at=10,
            ),
            _lease(
                "lease-v7-failed",
                version=7,
                status=p2p_pb2.TRANSFER_LEASE_STATUS_FAILED,
                updated_at=12,
            ),
            _lease(
                "lease-v7-active",
                version=7,
                status=p2p_pb2.TRANSFER_LEASE_STATUS_ACTIVE,
                updated_at=11,
            ),
        ),
    )

    assert inventory.discovery_supported
    assert [lease.lease_id for lease in inventory.completed_leases] == [
        "lease-v6-completed"
    ]
    assert [lease.lease_id for lease in inventory.non_completed_leases] == [
        "lease-v7-failed",
        "lease-v7-active",
    ]
    assert inventory.latest_model_version() == 7
    assert [lease.lease_id for lease in inventory.latest_attempts()] == [
        "lease-v7-active",
        "lease-v7-failed",
    ]
    assert [lease.lease_id for lease in inventory.latest_non_completed_attempts] == [
        "lease-v7-active",
        "lease-v7-failed",
    ]


def test_transfer_lease_summary_correlates_report_attempts_with_inventory():
    report = RlTransferReport(
        requested_model_version=None,
        resolved_model_version=7,
        receiver_rank=0,
        attempts=(
            RlTransferAttempt(
                mx_source_id="source",
                worker_id="worker-a",
                worker_rank=0,
                role=RlSourceRole.TRAINER,
                model_version=7,
                success=False,
                lease_id="lease-v7-failed",
            ),
            RlTransferAttempt(
                mx_source_id="source",
                worker_id="worker-b",
                worker_rank=1,
                role=RlSourceRole.TRAINER,
                model_version=7,
                success=True,
                lease_id="lease-v7-completed",
            ),
            RlTransferAttempt(
                mx_source_id="source",
                worker_id="worker-b",
                worker_rank=1,
                role=RlSourceRole.TRAINER,
                model_version=7,
                success=True,
                lease_id="lease-v7-completed",
            ),
            RlTransferAttempt(
                mx_source_id="source",
                worker_id="worker-c",
                worker_rank=2,
                role=RlSourceRole.TRAINER,
                model_version=7,
                success=False,
                lease_id="lease-v7-missing",
            ),
        ),
    )
    inventory = RlTransferLeaseInventory(
        target_worker_id="target-worker",
        leases=(
            _lease(
                "lease-v7-completed",
                version=7,
                status=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
                updated_at=20,
            ),
            _lease(
                "lease-v7-failed",
                version=7,
                status=p2p_pb2.TRANSFER_LEASE_STATUS_FAILED,
                updated_at=10,
            ),
        ),
    )

    summary = summarize_report_leases(report, inventory)

    assert summary.report is report
    assert summary.inventory is inventory
    assert summary.report_lease_ids == (
        "lease-v7-failed",
        "lease-v7-completed",
        "lease-v7-missing",
    )
    assert [lease.lease_id for lease in summary.matching_leases] == [
        "lease-v7-failed",
        "lease-v7-completed",
    ]
    assert summary.missing_lease_ids == ("lease-v7-missing",)
    assert summary.has_missing_leases
    assert [
        lease.lease_id for lease in summary.non_completed_matching_leases
    ] == ["lease-v7-failed"]


def test_transfer_lease_summary_handles_absent_report():
    inventory = RlTransferLeaseInventory(
        target_worker_id="target-worker",
        discovery_supported=False,
    )

    summary = summarize_report_leases(None, inventory)

    assert summary.report_lease_ids == ()
    assert summary.matching_leases == ()
    assert summary.missing_lease_ids == ()
    assert not summary.has_missing_leases


def test_transfer_lease_coordinator_lists_target_attempts_by_status():
    client = _LeaseClient()
    client.leases_by_status = {
        p2p_pb2.TRANSFER_LEASE_STATUS_ACTIVE: [
            _lease(
                "lease-active",
                version=8,
                status=p2p_pb2.TRANSFER_LEASE_STATUS_ACTIVE,
                updated_at=11,
            )
        ],
        p2p_pb2.TRANSFER_LEASE_STATUS_EXPIRED: [
            _lease(
                "lease-expired",
                version=7,
                status=p2p_pb2.TRANSFER_LEASE_STATUS_EXPIRED,
                updated_at=12,
            )
        ],
    }
    coordinator = RlTransferLeaseCoordinator(
        mx_client=client,
        target_worker_id="target-worker",
        ttl_seconds=5,
    )

    inventory = coordinator.list_target_leases(
        mx_source_id="source",
        statuses=(
            p2p_pb2.TRANSFER_LEASE_STATUS_ACTIVE,
            p2p_pb2.TRANSFER_LEASE_STATUS_EXPIRED,
        ),
        model_version=8,
        source_worker_id="source-worker",
    )

    assert inventory.discovery_supported
    assert [lease.lease_id for lease in inventory.leases] == [
        "lease-active",
        "lease-expired",
    ]
    assert client.lists == [
        (
            "source",
            "target-worker",
            p2p_pb2.TRANSFER_LEASE_STATUS_ACTIVE,
            8,
            "source-worker",
        ),
        (
            "source",
            "target-worker",
            p2p_pb2.TRANSFER_LEASE_STATUS_EXPIRED,
            8,
            "source-worker",
        ),
    ]


def test_transfer_lease_coordinator_lists_all_target_attempts_once():
    client = _LeaseClient()
    client.leases_by_status = {
        None: [
            _lease(
                "lease-completed",
                version=9,
                status=p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,
                updated_at=20,
            )
        ]
    }
    coordinator = RlTransferLeaseCoordinator(
        mx_client=client,
        target_worker_id="target-worker",
        ttl_seconds=5,
    )

    inventory = coordinator.list_target_leases(mx_source_id="source")

    assert [lease.lease_id for lease in inventory.leases] == ["lease-completed"]
    assert client.lists == [("source", "target-worker", None, None, "")]


def test_transfer_lease_inventory_reports_unsupported_old_server():
    coordinator = RlTransferLeaseCoordinator(
        mx_client=object(),
        target_worker_id="target-worker",
        ttl_seconds=5,
    )

    inventory = coordinator.list_target_leases()

    assert not inventory.discovery_supported
    assert inventory.leases == ()


def test_transfer_lease_inventory_noops_for_unimplemented_list_rpc():
    class _Unimplemented(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNIMPLEMENTED

    class _OldServerClient:
        def list_transfer_leases(self, **kwargs):
            raise _Unimplemented()

    coordinator = RlTransferLeaseCoordinator(
        mx_client=_OldServerClient(),
        target_worker_id="target-worker",
        ttl_seconds=5,
    )

    inventory = coordinator.list_target_leases()

    assert not inventory.discovery_supported
    assert inventory.latest_model_version() is None


def test_transfer_lease_noops_for_old_server_unimplemented_rpc():
    class _Unimplemented(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNIMPLEMENTED

    class _OldServerClient:
        def begin_transfer_lease(self, **kwargs):
            raise _Unimplemented()

    coordinator = RlTransferLeaseCoordinator(
        mx_client=_OldServerClient(),
        target_worker_id="target-worker",
        ttl_seconds=5,
    )

    with coordinator.lease_candidate(_candidate(), receiver_rank=0):
        pass
