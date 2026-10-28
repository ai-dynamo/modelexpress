# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate one object-store donor for concurrent vLLM replicas."""

import atexit
import logging
import threading
import time
import uuid

from . import p2p_pb2
from .client import MxClient
from .metadata.payload import accelerators_compatible

logger = logging.getLogger(__name__)
LEASE_TTL = 60
RENEW_INTERVAL = 20
POLL_INTERVAL = 2


def _ready(ctx, ranks):
    instances = ctx.mx_client.list_sources(
        identity=ctx.identity, status_filter=p2p_pb2.SOURCE_STATUS_READY, strict=True
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
    def __init__(self, ctx, token, ranks):
        self.ctx, self.token, self.ranks = ctx, token, ranks
        self.stopped = threading.Event()
        self.lost = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        atexit.register(self.stop)

    def _run(self):
        try:
            while not self.stopped.wait(RENEW_INTERVAL):
                if not self.ctx.mx_client.donor_lease(
                    self.ctx.identity,
                    self.token,
                    p2p_pb2.DONOR_LEASE_ACTION_RENEW,
                    LEASE_TTL,
                ):
                    self.lost.set()
                    logger.error("MX donor lease was lost")
                    return
                if _ready(self.ctx, self.ranks):
                    break
        except Exception:
            self.lost.set()
            logger.exception("MX donor lease renewal failed")
            return
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


def coordinate(ctx, config):
    """Elect once on rank zero and broadcast to the replica's workers."""
    parallel = config.parallel_config
    errors = []
    if not isinstance(ctx.mx_client, MxClient) or not ctx.p2p_enabled:
        errors.append("a centralized MX server and P2P are required")
    if (
        ctx.is_reload
        or ctx.p2p_role != "main"
        or getattr(config, "speculative_config", None)
    ):
        errors.append("reload and speculative decoding are not supported")
    if (
        getattr(parallel, "pipeline_parallel_size", 1) != 1
        or getattr(parallel, "data_parallel_size", 1) != 1
        or getattr(parallel, "enable_expert_parallel", False)
    ):
        errors.append("only tensor parallelism is supported")
    if ctx.adapter is None:
        raise RuntimeError("Cold start requires an engine adapter")
    all_errors = ctx.adapter.all_gather_state(errors)
    if any(all_errors):
        raise RuntimeError(f"Unsupported MX cold start configuration: {all_errors}")
    ranks = parallel.tensor_parallel_size
    decision = None
    if ctx.global_rank == 0:
        try:
            while True:
                if _ready(ctx, ranks):
                    decision = ("follower", "")
                    break
                token = uuid.uuid4().hex
                if ctx.mx_client.donor_lease(
                    ctx.identity, token, p2p_pb2.DONOR_LEASE_ACTION_ACQUIRE, LEASE_TTL
                ):
                    try:
                        ready = _ready(ctx, ranks)
                    except BaseException:
                        ctx.mx_client.donor_lease(
                            ctx.identity,
                            token,
                            p2p_pb2.DONOR_LEASE_ACTION_RELEASE,
                            LEASE_TTL,
                        )
                        raise
                    if ready:
                        ctx.mx_client.donor_lease(
                            ctx.identity,
                            token,
                            p2p_pb2.DONOR_LEASE_ACTION_RELEASE,
                            LEASE_TTL,
                        )
                        decision = ("follower", "")
                    else:
                        ctx.donor_lease = DonorLease(ctx, token, ranks)
                        decision = ("donor", "")
                    break
                time.sleep(POLL_INTERVAL)
        except Exception as exc:
            decision = ("error", str(exc))
    role, error = ctx.adapter.broadcast_state(decision)
    if role == "error":
        raise RuntimeError(f"MX cold start election failed: {error}")
    logger.info("MX cold start: %s", role)
