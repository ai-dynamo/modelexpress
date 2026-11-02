# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate one object-store donor for concurrent vLLM replicas."""

import atexit
import logging
import threading
import time
import uuid

import grpc

from . import envs, p2p_pb2
from .client import MxClient
from .metadata.payload import accelerators_compatible

logger = logging.getLogger(__name__)
LEASE_TTL = 60
RENEW_INTERVAL = 20
POLL_INTERVAL = 2


def _ready(ctx, ranks, timeout=30):
    instances = ctx.mx_client.list_sources(
        identity=ctx.identity,
        status_filter=p2p_pb2.SOURCE_STATUS_READY,
        strict=True,
        timeout=timeout,
    ).instances
    found = {
        instance.worker_rank
        for instance in instances
        if accelerators_compatible(
            ctx.accelerator_backend.name,
            instance.accelerator,
            mx_source_type=ctx.identity.mx_source_type,
            quantization=ctx.identity.quantization,
            dtype=ctx.identity.dtype,
        )
    }
    return set(range(ranks)).issubset(found)


class DonorLease:
    def __init__(self, ctx, token, ranks, acquired_at):
        self.ctx, self.token, self.ranks = ctx, token, ranks
        self.deadline = acquired_at + LEASE_TTL
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        atexit.register(self.stop)

    def _run(self):
        interval = RENEW_INTERVAL
        while not self.stopped.wait(
            min(interval, max(0, self.deadline - time.monotonic()))
        ):
            started = time.monotonic()
            if started >= self.deadline:
                logger.error("MX donor lease renewal deadline passed; ownership is unknown")
                return
            try:
                if not self.ctx.mx_client.donor_lease(
                    self.ctx.identity,
                    self.token,
                    p2p_pb2.DONOR_LEASE_ACTION_RENEW,
                    LEASE_TTL,
                ):
                    logger.error("MX donor lease was lost")
                    return
            except Exception:
                logger.exception("MX donor lease renewal failed; retrying until deadline")
                interval = POLL_INTERVAL
                continue
            self.deadline = started + LEASE_TTL
            interval = RENEW_INTERVAL
            try:
                if _ready(self.ctx, self.ranks):
                    break
            except Exception:
                logger.exception("MX donor readiness check failed; continuing lease renewal")
            interval = max(0, started + RENEW_INTERVAL - time.monotonic())
        try:
            self.ctx.mx_client.donor_lease(
                self.ctx.identity,
                self.token,
                p2p_pb2.DONOR_LEASE_ACTION_RELEASE,
                LEASE_TTL,
            )
        except Exception:
            logger.exception("MX donor lease release failed; it will expire")

    def stop(self):
        self.stopped.set()
        if threading.current_thread() is not self.thread:
            self.thread.join(timeout=45)
        atexit.unregister(self.stop)


def coordinate(ctx, expected):
    """Elect once on rank zero and broadcast to the replica's workers."""
    if ctx.adapter is None:
        raise RuntimeError("Cold start requires an engine adapter")
    errors = []
    if not isinstance(ctx.mx_client, MxClient) or not ctx.p2p_enabled:
        errors.append("a centralized MX server and P2P are required")
    if ctx.is_reload or ctx.p2p_role != "main":
        errors.append("reload and draft election are not supported")
    states = ctx.adapter.all_gather_state((errors, ctx.global_rank, ctx.worker_rank))
    ranks = len(states)
    all_errors = [state[0] for state in states]
    if (
        ranks != expected
        or sorted(state[1] for state in states) != list(range(ranks))
        or any(state[1] != state[2] for state in states)
    ):
        all_errors.append("worker group does not match the configured parallel topology")
    if any(all_errors):
        raise RuntimeError(f"Unsupported MX cold start configuration: {all_errors}")
    decision = None
    if ctx.global_rank == 0:
        deadline = time.monotonic() + envs.MX_P2P_COLD_START_WAIT_TIMEOUT_SECS
        token = None
        acquired = False

        def remaining():
            value = deadline - time.monotonic()
            if value <= 0:
                raise TimeoutError("MX cold start wait deadline expired")
            return value

        def call_with_deadline(function, *args, rpc_timeout):
            wait = remaining()
            try:
                return function(*args, timeout=min(rpc_timeout, wait))
            except grpc.RpcError as exc:
                if wait <= rpc_timeout and exc.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                    raise TimeoutError("MX cold start wait deadline expired") from exc
                raise

        def release():
            nonlocal acquired
            try:
                ctx.mx_client.donor_lease(
                    ctx.identity,
                    token,
                    p2p_pb2.DONOR_LEASE_ACTION_RELEASE,
                    LEASE_TTL,
                    timeout=min(10, remaining()),
                )
            except Exception:
                logger.warning("MX donor lease release failed or timed out; it will expire")
            acquired = False

        try:
            while True:
                if call_with_deadline(_ready, ctx, ranks, rpc_timeout=30):
                    decision = ("follower", "")
                    break
                token = uuid.uuid4().hex
                acquired_at = time.monotonic()
                acquired = call_with_deadline(
                    ctx.mx_client.donor_lease,
                    ctx.identity,
                    token,
                    p2p_pb2.DONOR_LEASE_ACTION_ACQUIRE,
                    LEASE_TTL,
                    rpc_timeout=10,
                )
                if acquired:
                    ready = call_with_deadline(_ready, ctx, ranks, rpc_timeout=30)
                    remaining()
                    if ready:
                        release()
                        decision = ("follower", "")
                    else:
                        ctx.donor_lease = DonorLease(ctx, token, ranks, acquired_at)
                        acquired = False
                        decision = ("donor", "")
                    break
                time.sleep(min(POLL_INTERVAL, remaining()))
        except Exception as exc:
            if acquired:
                release()
            if isinstance(exc, TimeoutError) or time.monotonic() >= deadline:
                logger.warning(
                    "MX cold start wait timed out; skipping RDMA for this replica"
                )
            else:
                logger.warning(
                    "MX cold start election failed (%s); skipping RDMA for this replica",
                    exc,
                )
            decision = ("fallback", "")
    role, _ = ctx.adapter.broadcast_state(decision)
    ctx.skip_rdma = role == "fallback"
    logger.info("MX cold start: %s", role)
