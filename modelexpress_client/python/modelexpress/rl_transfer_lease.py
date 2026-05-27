# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional durable transfer leases for RL weight pulls."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import logging
import threading
from types import TracebackType

import grpc

from modelexpress import p2p_pb2
from modelexpress.rl_metadata import RlSourceCandidate
from modelexpress.rl_transfer_report import RlTransferReport

logger = logging.getLogger("modelexpress.rl_transfer_lease")

_DEFAULT_TTL_SECONDS = 30.0
_MIN_RENEW_INTERVAL_SECONDS = 1.0
_NON_COMPLETED_LEASE_STATUSES = frozenset(
    {
        p2p_pb2.TRANSFER_LEASE_STATUS_ACTIVE,
        p2p_pb2.TRANSFER_LEASE_STATUS_FAILED,
        p2p_pb2.TRANSFER_LEASE_STATUS_EXPIRED,
    }
)


@dataclass(frozen=True)
class RlTransferLeaseInventory:
    """Discovered transfer leases for one target worker."""

    target_worker_id: str
    leases: tuple["p2p_pb2.TransferLease", ...] = ()
    discovery_supported: bool = True

    @property
    def active_leases(self) -> tuple["p2p_pb2.TransferLease", ...]:
        return self.with_statuses((p2p_pb2.TRANSFER_LEASE_STATUS_ACTIVE,))

    @property
    def completed_leases(self) -> tuple["p2p_pb2.TransferLease", ...]:
        return self.with_statuses((p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED,))

    @property
    def failed_leases(self) -> tuple["p2p_pb2.TransferLease", ...]:
        return self.with_statuses((p2p_pb2.TRANSFER_LEASE_STATUS_FAILED,))

    @property
    def expired_leases(self) -> tuple["p2p_pb2.TransferLease", ...]:
        return self.with_statuses((p2p_pb2.TRANSFER_LEASE_STATUS_EXPIRED,))

    @property
    def non_completed_leases(self) -> tuple["p2p_pb2.TransferLease", ...]:
        return self.with_statuses(_NON_COMPLETED_LEASE_STATUSES)

    @property
    def latest_non_completed_attempts(self) -> tuple["p2p_pb2.TransferLease", ...]:
        return self.latest_attempts(statuses=_NON_COMPLETED_LEASE_STATUSES)

    def with_statuses(
        self,
        statuses: Iterable[int],
    ) -> tuple["p2p_pb2.TransferLease", ...]:
        status_set = {int(status) for status in statuses}
        return tuple(lease for lease in self.leases if lease.status in status_set)

    def latest_model_version(self, *, statuses: Iterable[int] | None = None) -> int | None:
        leases = self._filtered(statuses)
        if not leases:
            return None
        return max(int(lease.model_version) for lease in leases)

    def latest_attempts(
        self,
        *,
        statuses: Iterable[int] | None = None,
    ) -> tuple["p2p_pb2.TransferLease", ...]:
        latest = self.latest_model_version(statuses=statuses)
        if latest is None:
            return ()
        leases = [
            lease
            for lease in self._filtered(statuses)
            if int(lease.model_version) == latest
        ]
        return tuple(
            sorted(
                leases,
                key=lambda lease: (
                    int(lease.updated_at),
                    int(lease.created_at),
                    lease.lease_id,
                ),
            )
        )

    def _filtered(
        self,
        statuses: Iterable[int] | None,
    ) -> tuple["p2p_pb2.TransferLease", ...]:
        if statuses is None:
            return self.leases
        return self.with_statuses(statuses)


@dataclass(frozen=True)
class RlTransferLeaseReportSummary:
    """Join a local receive report with durable server-side lease records."""

    report: RlTransferReport | None
    inventory: RlTransferLeaseInventory
    report_lease_ids: tuple[str, ...]
    matching_leases: tuple["p2p_pb2.TransferLease", ...]
    missing_lease_ids: tuple[str, ...]

    @property
    def has_missing_leases(self) -> bool:
        return bool(self.missing_lease_ids)

    @property
    def non_completed_matching_leases(self) -> tuple["p2p_pb2.TransferLease", ...]:
        return tuple(
            lease
            for lease in self.matching_leases
            if lease.status != p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED
        )


def summarize_report_leases(
    report: RlTransferReport | None,
    inventory: RlTransferLeaseInventory,
) -> RlTransferLeaseReportSummary:
    """Correlate receive-report lease IDs with server-discovered lease state."""
    if report is None:
        report_lease_ids = ()
    else:
        report_lease_ids = tuple(
            dict.fromkeys(
                attempt.lease_id
                for attempt in report.attempts
                if attempt.lease_id
            )
        )
    leases_by_id = {lease.lease_id: lease for lease in inventory.leases}
    matching_leases = tuple(
        leases_by_id[lease_id]
        for lease_id in report_lease_ids
        if lease_id in leases_by_id
    )
    missing_lease_ids = tuple(
        lease_id for lease_id in report_lease_ids if lease_id not in leases_by_id
    )
    return RlTransferLeaseReportSummary(
        report=report,
        inventory=inventory,
        report_lease_ids=report_lease_ids,
        matching_leases=matching_leases,
        missing_lease_ids=missing_lease_ids,
    )


def transfer_lease_key(candidate: RlSourceCandidate) -> tuple[str, str, int]:
    return (candidate.mx_source_id, candidate.worker_id, candidate.worker_rank)


class LeasedTransferError(RuntimeError):
    def __init__(
        self,
        exc: Exception,
        lease_ids_by_candidate: dict[tuple[str, str, int], str],
    ) -> None:
        super().__init__(str(exc))
        self.lease_ids_by_candidate = dict(lease_ids_by_candidate)

    def lease_id_for(self, candidate: RlSourceCandidate) -> str:
        return self.lease_ids_by_candidate.get(transfer_lease_key(candidate), "")


class RlTransferLeaseCoordinator:
    """Begin/renew/complete MX transfer leases when the client supports them."""

    def __init__(
        self,
        *,
        mx_client: object,
        target_worker_id: str,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._mx_client = mx_client
        self._target_worker_id = target_worker_id
        self._ttl_seconds = max(_MIN_RENEW_INTERVAL_SECONDS, ttl_seconds)

    def lease_candidate(
        self,
        candidate: RlSourceCandidate,
        *,
        receiver_rank: int,
    ) -> "RlTransferLease":
        begin = getattr(self._mx_client, "begin_transfer_lease", None)
        if begin is None:
            return RlTransferLease.disabled()

        try:
            lease = begin(
                mx_source_id=candidate.mx_source_id,
                source_worker_id=candidate.worker_id,
                target_worker_id=self._target_worker_id,
                target_worker_rank=receiver_rank,
                model_version=candidate.metadata.model_version,
                ttl_millis=self._ttl_millis,
                metadata={
                    "mx_rl_role": candidate.metadata.role.value,
                },
            )
        except (NotImplementedError, grpc.RpcError) as exc:
            if isinstance(exc, NotImplementedError) or _is_unimplemented_rpc(exc):
                return RlTransferLease.disabled()
            raise

        return RlTransferLease(
            mx_client=self._mx_client,
            lease_id=lease.lease_id,
            ttl_millis=self._ttl_millis,
        )

    def list_target_leases(
        self,
        *,
        mx_source_id: str = "",
        statuses: Iterable[int] | None = None,
        model_version: int | None = None,
        source_worker_id: str = "",
    ) -> RlTransferLeaseInventory:
        """List persisted transfer attempts for this target worker.

        Older servers that do not expose lease discovery return an unsupported,
        empty inventory so framework adapters can preserve compatibility.
        """
        list_leases = getattr(self._mx_client, "list_transfer_leases", None)
        if list_leases is None:
            return RlTransferLeaseInventory(
                target_worker_id=self._target_worker_id,
                discovery_supported=False,
            )

        try:
            leases = _list_leases_by_status(
                list_leases,
                mx_source_id=mx_source_id,
                target_worker_id=self._target_worker_id,
                statuses=statuses,
                model_version=model_version,
                source_worker_id=source_worker_id,
            )
        except (NotImplementedError, grpc.RpcError) as exc:
            if isinstance(exc, NotImplementedError) or _is_unimplemented_rpc(exc):
                return RlTransferLeaseInventory(
                    target_worker_id=self._target_worker_id,
                    discovery_supported=False,
                )
            raise

        return RlTransferLeaseInventory(
            target_worker_id=self._target_worker_id,
            leases=leases,
            discovery_supported=True,
        )

    @property
    def _ttl_millis(self) -> int:
        return int(self._ttl_seconds * 1000)


class RlTransferLease:
    """Context manager that renews a transfer lease until completion."""

    def __init__(
        self,
        *,
        mx_client: object | None = None,
        lease_id: str = "",
        ttl_millis: int = 0,
    ) -> None:
        self._mx_client = mx_client
        self._lease_id = lease_id
        self._ttl_millis = ttl_millis
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @classmethod
    def disabled(cls) -> "RlTransferLease":
        return cls()

    @property
    def lease_id(self) -> str:
        return self._lease_id

    def __enter__(self) -> "RlTransferLease":
        if self._mx_client is None or not self._lease_id:
            return self
        interval = max(_MIN_RENEW_INTERVAL_SECONDS, self._ttl_millis / 2000)
        self._thread = threading.Thread(
            target=self._renew_until_stopped,
            args=(interval,),
            name=f"mx-rl-transfer-lease-{self._lease_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._mx_client is None or not self._lease_id:
            return False

        status = (
            p2p_pb2.TRANSFER_LEASE_STATUS_FAILED
            if exc is not None
            else p2p_pb2.TRANSFER_LEASE_STATUS_COMPLETED
        )
        error_message = "" if exc is None else str(exc)
        try:
            self._mx_client.complete_transfer_lease(
                self._lease_id,
                status=status,
                error_message=error_message,
            )
        except Exception:
            logger.warning(
                "Failed to complete MX transfer lease: lease_id=%s",
                self._lease_id,
                exc_info=True,
            )
        return False

    def _renew_until_stopped(self, interval: float) -> None:
        while not self._stop.wait(interval):
            try:
                self._mx_client.renew_transfer_lease(
                    self._lease_id,
                    ttl_millis=self._ttl_millis,
                )
            except Exception:
                logger.warning(
                    "Failed to renew MX transfer lease: lease_id=%s",
                    self._lease_id,
                    exc_info=True,
                )
                return


def _is_unimplemented_rpc(exc: grpc.RpcError) -> bool:
    try:
        return exc.code() == grpc.StatusCode.UNIMPLEMENTED
    except Exception:
        return False


def _list_leases_by_status(
    list_leases,
    *,
    mx_source_id: str,
    target_worker_id: str,
    statuses: Iterable[int] | None,
    model_version: int | None,
    source_worker_id: str,
) -> tuple["p2p_pb2.TransferLease", ...]:
    if statuses is None:
        response = list_leases(
            mx_source_id=mx_source_id,
            target_worker_id=target_worker_id,
            model_version_filter=model_version,
            source_worker_id=source_worker_id,
        )
        return tuple(response.leases)

    leases_by_id: dict[str, "p2p_pb2.TransferLease"] = {}
    for status in statuses:
        response = list_leases(
            mx_source_id=mx_source_id,
            target_worker_id=target_worker_id,
            status_filter=int(status),
            model_version_filter=model_version,
            source_worker_id=source_worker_id,
        )
        for lease in response.leases:
            leases_by_id.setdefault(lease.lease_id, lease)
    return tuple(leases_by_id.values())
