# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lease-based coordination of simultaneous vLLM cold starts."""

from __future__ import annotations

import atexit
import logging
import threading
import time
import uuid

from . import envs, p2p_pb2
from .client import MxClient
from .metadata.payload import accelerators_compatible

logger = logging.getLogger(__name__)
LEASE_TTL = 60
RENEW_INTERVAL = 20
POLL_INTERVAL = 2
PREFIX = "mx-cold-"


def ready_groups(instances, ranks: int, ctx=None) -> list[str]:
    groups: dict[str, set[int]] = {}
    for instance in instances:
        worker = instance.worker_id
        if not worker.startswith(PREFIX):
            continue
        group, sep, suffix = worker.rpartition("-")
        if not sep or not suffix.isdecimal() or int(suffix) != instance.worker_rank:
            continue
        try:
            uuid.UUID(group[len(PREFIX) :])
        except ValueError:
            continue
        if ctx is not None and not accelerators_compatible(
            ctx.accelerator_backend.name,
            instance.accelerator,
            mx_source_type=ctx.identity.mx_source_type,
            quantization=ctx.identity.quantization,
            dtype=ctx.identity.dtype,
        ):
            continue
        groups.setdefault(group, set()).add(instance.worker_rank)
    return sorted(
        group for group, found in groups.items() if found == set(range(ranks))
    )


def _groups(ctx, ranks):
    response = ctx.mx_client.list_sources(
        identity=ctx.identity, status_filter=p2p_pb2.SOURCE_STATUS_READY, strict=True
    )
    return ready_groups(response.instances, ranks, ctx)


class DonorLease:
    def __init__(self, ctx, token: str, group: str, ranks: int):
        self.ctx, self.token, self.group, self.ranks = ctx, token, group, ranks
        self.stop_event = threading.Event()
        self.started = time.monotonic()
        self.publication_started = None
        self.lost = threading.Event()
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="mx-donor-lease"
        )
        self.thread.start()
        atexit.register(self.stop)

    def _run(self):
        published = False
        while not self.stop_event.wait(RENEW_INTERVAL):
            try:
                if not published:
                    published = self.group in _groups(self.ctx, self.ranks)
                    now = time.monotonic()
                    startup_timeout = envs.MX_COLD_START_TIMEOUT_SECS
                    publish_timeout = envs.MX_PUBLISH_TIMEOUT_SECS
                    if (
                        startup_timeout > 0 and now - self.started > startup_timeout
                    ) or (
                        publish_timeout > 0
                        and self.publication_started is not None
                        and now - self.publication_started > publish_timeout
                    ):
                        logger.error(
                            "Donor startup or publication timed out; releasing lease"
                        )
                        self.lost.set()
                        break
                if not self.ctx.mx_client.donor_lease(
                    self.ctx.identity,
                    self.token,
                    p2p_pb2.DONOR_LEASE_ACTION_RENEW,
                    LEASE_TTL,
                ):
                    logger.error("Donor lease lost; publication is fenced")
                    self.lost.set()
                    return
            except Exception:
                logger.exception("Donor lease renewal failed; publication will expire")
                self.lost.set()
                return
        try:
            self.ctx.mx_client.donor_lease(
                self.ctx.identity,
                self.token,
                p2p_pb2.DONOR_LEASE_ACTION_RELEASE,
                LEASE_TTL,
            )
        except Exception:
            logger.exception("Failed to release donor lease")

    def stop(self):
        self.stop_event.set()
        if threading.current_thread() is not self.thread:
            self.thread.join(timeout=45)
        atexit.unregister(self.stop)


def coordinate(ctx, config):
    """Elect one replica and share the decision through the existing engine group."""
    parallel = config.parallel_config
    errors = []
    if not isinstance(ctx.mx_client, MxClient):
        errors.append("centralized MX server is required")
    if not ctx.p2p_enabled:
        errors.append("P2P must be enabled")
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
    ranks = parallel.tensor_parallel_size
    if ctx.adapter is None:
        raise RuntimeError("Cold start requires an engine adapter")
    all_errors = ctx.adapter.all_gather_state(errors)
    if any(all_errors):
        raise RuntimeError(f"Unsupported MX cold start configuration: {all_errors}")
    decision = None
    if ctx.global_rank == 0:
        try:
            while True:
                groups = _groups(ctx, ranks)
                if groups:
                    decision = ("follower", groups[0], "")
                    break
                token = uuid.uuid4().hex
                if ctx.mx_client.donor_lease(
                    ctx.identity, token, p2p_pb2.DONOR_LEASE_ACTION_ACQUIRE, LEASE_TTL
                ):
                    try:
                        groups = _groups(ctx, ranks)
                    except BaseException:
                        ctx.mx_client.donor_lease(
                            ctx.identity,
                            token,
                            p2p_pb2.DONOR_LEASE_ACTION_RELEASE,
                            LEASE_TTL,
                        )
                        raise
                    if groups:
                        ctx.mx_client.donor_lease(
                            ctx.identity,
                            token,
                            p2p_pb2.DONOR_LEASE_ACTION_RELEASE,
                            LEASE_TTL,
                        )
                        decision = ("follower", groups[0], "")
                    else:
                        decision = ("donor", PREFIX + uuid.uuid4().hex, token)
                    break
                time.sleep(POLL_INTERVAL)
        except Exception as exc:
            decision = ("error", str(exc), "")
    role, group, token = ctx.adapter.broadcast_state(decision)
    if role == "error":
        raise RuntimeError(f"MX cold start election failed: {group}")
    ctx.cold_start_role, ctx.cold_start_group, ctx.donor_lease_token = (
        role,
        group,
        token,
    )
    if role == "donor":
        ctx.worker_id = f"{group}-{ctx.worker_rank}"
        if ctx.global_rank == 0:
            ctx.donor_lease = DonorLease(ctx, token, group, ranks)
    logger.info("MX cold start elected %s for group %s", role, group)
