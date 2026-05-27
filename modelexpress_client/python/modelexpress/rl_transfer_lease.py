# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional durable transfer leases for RL weight pulls."""

from __future__ import annotations

import logging
import threading
from types import TracebackType

import grpc

from modelexpress import p2p_pb2
from modelexpress.rl_metadata import RlSourceCandidate

logger = logging.getLogger("modelexpress.rl_transfer_lease")

_DEFAULT_TTL_SECONDS = 30.0
_MIN_RENEW_INTERVAL_SECONDS = 1.0


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
